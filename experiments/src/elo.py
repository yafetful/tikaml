"""Dynamic Elo rating system with Poisson goal prediction.

Elo updates use goal-difference adjusted K-factor.
Elo difference maps to expected goals via league averages.
Season regression: Elo regresses 1/3 toward mean at season start.
"""

import numpy as np
from scipy.stats import poisson


class EloModel:

    def __init__(self, k=20, home_advantage=100, initial_elo=1500,
                 season_regression=1 / 3):
        self.k = k
        self.home_advantage = home_advantage
        self.initial_elo = initial_elo
        self.season_regression = season_regression
        self.elo = {}  # team_id -> elo rating
        self._league_avg_home = {}  # league -> avg home goals
        self._league_avg_away = {}  # league -> avg away goals

    def _get_elo(self, team):
        return self.elo.get(team, self.initial_elo)

    def _expected_score(self, elo_a, elo_b):
        """Expected score for team A (0 to 1)."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

    def _k_adjusted(self, goal_diff):
        """K-factor adjusted by goal difference."""
        return self.k * (1 + 0.5 * np.log1p(abs(goal_diff)))

    def update(self, home_team, away_team, home_goals, away_goals):
        """Update Elo ratings after a match."""
        elo_h = self._get_elo(home_team) + self.home_advantage
        elo_a = self._get_elo(away_team)

        expected_h = self._expected_score(elo_h, elo_a)

        # Actual result
        if home_goals > away_goals:
            actual_h = 1.0
        elif home_goals == away_goals:
            actual_h = 0.5
        else:
            actual_h = 0.0

        goal_diff = home_goals - away_goals
        k_adj = self._k_adjusted(goal_diff)

        self.elo[home_team] = self._get_elo(home_team) + k_adj * (actual_h - expected_h)
        self.elo[away_team] = self._get_elo(away_team) + k_adj * (expected_h - actual_h)

    def season_reset(self):
        """Regress all Elo ratings toward the mean at season start."""
        r = self.season_regression
        for team in self.elo:
            self.elo[team] = self.elo[team] * (1 - r) + self.initial_elo * r

    def set_league_averages(self, league, avg_home_goals, avg_away_goals):
        """Set league-level average goals for Elo→lambda mapping."""
        self._league_avg_home[league] = avg_home_goals
        self._league_avg_away[league] = avg_away_goals

    def predict_lambdas(self, home_team, away_team, league=None):
        """Convert Elo difference to expected goals (lambda)."""
        elo_h = self._get_elo(home_team) + self.home_advantage
        elo_a = self._get_elo(away_team)
        elo_diff = elo_h - elo_a

        avg_h = self._league_avg_home.get(league, 1.5)
        avg_a = self._league_avg_away.get(league, 1.15)

        lambda_home = avg_h * np.exp(0.001 * elo_diff)
        lambda_away = avg_a * np.exp(-0.001 * elo_diff)

        return np.clip(lambda_home, 0.2, 4.5), np.clip(lambda_away, 0.2, 4.5)

    def predict_score_matrix(self, home_team, away_team, league=None, max_goals=7):
        """Predict 7x7 score probability matrix."""
        lh, la = self.predict_lambdas(home_team, away_team, league)
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
        matrix /= matrix.sum()
        return matrix

    def predict_1x2(self, home_team, away_team, league=None, max_goals=7):
        """Predict [P(home), P(draw), P(away)]."""
        matrix = self.predict_score_matrix(home_team, away_team, league, max_goals)
        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()
        probs = np.array([p_home, p_draw, p_away])
        return probs / probs.sum()

    def fit(self, df):
        """Process all historical matches to build Elo ratings.

        Args:
            df: DataFrame sorted by date with columns:
                date, league, season, home_team, away_team,
                home_goals, away_goals
        """
        df = df.sort_values("date")
        prev_season = {}  # league -> current season

        # Compute league averages
        for (league, season), group in df.groupby(["league", "season"]):
            avg_h = group["home_goals"].mean()
            avg_a = group["away_goals"].mean()
            self.set_league_averages(league, avg_h, avg_a)

        for _, row in df.iterrows():
            league = row["league"]
            season = row["season"]

            # Season reset when season changes
            if league in prev_season and prev_season[league] != season:
                self.season_reset()
            prev_season[league] = season

            self.update(
                row["home_team"], row["away_team"],
                row["home_goals"], row["away_goals"]
            )

        return self

    def team_ratings(self, top_n=20):
        """Return sorted team ratings."""
        import pandas as pd
        ratings = pd.DataFrame([
            {"team": t, "elo": e} for t, e in self.elo.items()
        ]).sort_values("elo", ascending=False).reset_index(drop=True)
        return ratings.head(top_n)
