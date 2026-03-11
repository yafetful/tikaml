"""Dixon-Coles model for football match prediction.

Each team has attack (α) and defense (β) parameters.
Higher α = stronger attack, higher β = weaker defense (concede more).

    λ_home = exp(α_home + β_away + γ)
    μ_away = exp(α_away + β_home)

Low-score correction ρ adjusts P(0-0), P(0-1), P(1-0), P(1-1).
Time decay gives recent matches more weight.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


class DixonColesModel:

    def __init__(self, half_life_days=180):
        self.half_life_days = half_life_days
        self.teams = []
        self.team_idx = {}
        self.attack = None
        self.defense = None
        self.home_adv = None
        self.rho = None
        self._fitted = False

    @staticmethod
    def _tau(x, y, lambda_h, mu_a, rho):
        """Low-score correction factor."""
        if x == 0 and y == 0:
            return 1 - lambda_h * mu_a * rho
        elif x == 0 and y == 1:
            return 1 + lambda_h * rho
        elif x == 1 and y == 0:
            return 1 + mu_a * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _compute_weights(self, dates, current_date):
        """Time decay weights: half-life exponential decay."""
        days_diff = (current_date - dates).dt.days.values.astype(float)
        return np.exp(-0.693 * days_diff / self.half_life_days)

    def _neg_log_likelihood(self, params, home_goals, away_goals,
                            home_idx, away_idx, weights):
        """Vectorized negative log-likelihood."""
        n_teams = len(self.teams)
        attack = params[:n_teams]
        defense = params[n_teams:2 * n_teams]
        home_adv = params[2 * n_teams]
        rho = params[2 * n_teams + 1]

        lambda_h = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
        mu_a = np.exp(attack[away_idx] + defense[home_idx])

        # Clip lambdas to prevent overflow
        lambda_h = np.clip(lambda_h, 0.01, 10.0)
        mu_a = np.clip(mu_a, 0.01, 10.0)

        hg = home_goals.astype(int)
        ag = away_goals.astype(int)

        # Vectorized tau
        tau = np.ones(len(hg))
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1 - lambda_h[m00] * mu_a[m00] * rho
        tau[m01] = 1 + lambda_h[m01] * rho
        tau[m10] = 1 + mu_a[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-10, None)

        log_ll = weights * (
            np.log(tau)
            + poisson.logpmf(hg, lambda_h)
            + poisson.logpmf(ag, mu_a)
        )
        return -np.sum(log_ll)

    def fit(self, df, current_date=None):
        """Fit Dixon-Coles model on match results.

        Args:
            df: DataFrame with columns [date, home_team, away_team,
                home_goals, away_goals]
            current_date: reference date for time decay (default: max date)

        Returns:
            self
        """
        if current_date is None:
            current_date = df["date"].max()

        all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams = all_teams
        self.team_idx = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)

        home_idx = np.array([self.team_idx[t] for t in df["home_team"]])
        away_idx = np.array([self.team_idx[t] for t in df["away_team"]])
        home_goals = df["home_goals"].values.astype(float)
        away_goals = df["away_goals"].values.astype(float)
        weights = self._compute_weights(df["date"], current_date)

        # Initial parameters
        x0 = np.zeros(2 * n_teams + 2)
        x0[2 * n_teams] = 0.25       # home advantage
        x0[2 * n_teams + 1] = -0.05  # rho

        # Bounds
        bounds = [(-2, 2)] * n_teams        # attack
        bounds += [(-2, 2)] * n_teams       # defense
        bounds.append((0.01, 1.0))          # home_adv > 0
        bounds.append((-0.5, 0.5))          # rho

        # Constraint: sum(attack) = 0
        constraints = [{
            "type": "eq",
            "fun": lambda p, nt=n_teams: np.sum(p[:nt])
        }]

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_goals, away_goals, home_idx, away_idx, weights),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        self.attack = result.x[:n_teams]
        self.defense = result.x[n_teams:2 * n_teams]
        self.home_adv = result.x[2 * n_teams]
        self.rho = result.x[2 * n_teams + 1]
        self._fitted = True

        return self

    def _get_lambdas(self, home_team, away_team):
        """Get expected goals (lambda) for both teams."""
        h = self.team_idx.get(home_team)
        a = self.team_idx.get(away_team)

        # Fallback for unknown teams: use median parameters
        if h is None:
            att_h = np.median(self.attack)
            def_h = np.median(self.defense)
        else:
            att_h = self.attack[h]
            def_h = self.defense[h]

        if a is None:
            att_a = np.median(self.attack)
            def_a = np.median(self.defense)
        else:
            att_a = self.attack[a]
            def_a = self.defense[a]

        lambda_h = np.exp(att_h + def_a + self.home_adv)
        mu_a = np.exp(att_a + def_h)
        return lambda_h, mu_a

    def predict_score_matrix(self, home_team, away_team, max_goals=7):
        """Predict score probability matrix P(home=i, away=j).

        Returns:
            (max_goals, max_goals) numpy array
        """
        lambda_h, mu_a = self._get_lambdas(home_team, away_team)

        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                tau = self._tau(i, j, lambda_h, mu_a, self.rho)
                matrix[i, j] = tau * poisson.pmf(i, lambda_h) * poisson.pmf(j, mu_a)

        matrix /= matrix.sum()
        return matrix

    def predict_1x2(self, home_team, away_team, max_goals=7):
        """Predict [P(home_win), P(draw), P(away_win)]."""
        matrix = self.predict_score_matrix(home_team, away_team, max_goals)
        # matrix[i,j] = P(home=i, away=j)
        # Home wins: i > j → below diagonal
        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()
        probs = np.array([p_home, p_draw, p_away])
        return probs / probs.sum()

    def team_ratings(self):
        """Return team ratings as a DataFrame."""
        import pandas as pd
        return pd.DataFrame({
            "team": self.teams,
            "attack": self.attack,
            "defense": self.defense,
            "strength": self.attack - self.defense,
        }).sort_values("strength", ascending=False).reset_index(drop=True)
