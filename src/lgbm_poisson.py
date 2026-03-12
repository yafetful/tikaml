"""LightGBM Poisson regression model for football count prediction.

Trains two independent models:
- Home model: predicts λ_home (Poisson rate for home team)
- Away model: predicts λ_away (Poisson rate for away team)

Primary use is goal prediction with Dixon-Coles correction, but also
supports corners and yellow cards via custom target columns and feature sets.
"""

import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from scipy.stats import poisson


# Features used for training (must exist in feature table)
FEATURE_COLS = [
    # Rolling xG features (core)
    "xg_rolling_home", "xg_rolling_away",
    "xg_rolling_conceded_home", "xg_rolling_conceded_away",
    "xgot_rolling_home", "xgot_rolling_away",
    "xgot_rolling_conceded_home", "xgot_rolling_conceded_away",
    # Rolling tactical features
    "shots_rolling_home", "shots_rolling_away",
    "shots_on_target_rolling_home", "shots_on_target_rolling_away",
    "ppda_rolling_home", "ppda_rolling_away",
    "prog_passes_rolling_home", "prog_passes_rolling_away",
    "prog_carries_rolling_home", "prog_carries_rolling_away",
    "touches_in_box_rolling_home", "touches_in_box_rolling_away",
    "final_third_entries_rolling_home", "final_third_entries_rolling_away",
    "poss_won_att3rd_rolling_home", "poss_won_att3rd_rolling_away",
    "possession_rolling_home", "possession_rolling_away",
    # Form
    "goals_rolling_home", "goals_rolling_away",
    "goals_rolling_conceded_home", "goals_rolling_conceded_away",
    # Batch 1: Venue-specific rolling (home-only / away-only blended)
    "xg_venue_rolling_home", "xg_venue_rolling_away",
    "xgot_venue_rolling_home", "xgot_venue_rolling_away",
    "goals_venue_rolling_home", "goals_venue_rolling_away",
    "shots_venue_rolling_home", "shots_venue_rolling_away",
    # Batch 1: Derived features
    "xg_overperformance_home", "xg_overperformance_away",
    "shots_on_target_pct_home", "shots_on_target_pct_away",
    "clean_sheet_pct_home", "clean_sheet_pct_away",
    # H2H features
    "h2h_win_pct_home", "h2h_goal_diff_home", "h2h_matches",
    # Relative strength (team vs league avg)
    "xg_rolling_rel_home", "xg_rolling_rel_away",
    "xgot_rolling_rel_home", "xgot_rolling_rel_away",
    "shots_rolling_rel_home", "shots_rolling_rel_away",
    # Momentum
    "momentum_home", "momentum_away",
    # Draw-prone features
    "draw_rate_home", "draw_rate_away",
    "league_draw_rate",
    "xg_diff_abs", "defensive_strength",
    # Context
    "days_rest_home", "days_rest_away", "days_rest_diff",
    "is_midweek", "season_stage",
    "league_avg_goals",
    "points_diff", "position_diff",
    # Match importance
    "is_relegation_battle", "is_title_race",
    "points_to_safety", "points_to_leader",
    "match_importance",
    # Lineup rotation features
    "lineup_changes_home", "lineup_changes_away",
    "lineup_stability_home", "lineup_stability_away",
    "formation_change_home", "formation_change_away",
    "squad_rotation_rate_home", "squad_rotation_rate_away",
    # Market odds implied probabilities
    "odds_prob_home", "odds_prob_draw", "odds_prob_away",
]

# Corner prediction features: base features + corner-specific rolling
CORNER_FEATURE_COLS = FEATURE_COLS + [
    "corners_rolling_home", "corners_rolling_away",
    "corners_rolling_conceded_home", "corners_rolling_conceded_away",
    "referee_corners_rolling",
]

# Yellow card prediction features: base features + yellow/foul/referee rolling
YELLOW_FEATURE_COLS = FEATURE_COLS + [
    "yellows_rolling_home", "yellows_rolling_away",
    "yellows_rolling_conceded_home", "yellows_rolling_conceded_away",
    "referee_yellows_rolling",
    "fouls_rolling_home", "fouls_rolling_away",
]

LGB_PARAMS = {
    "objective": "poisson",
    "metric": "poisson",
    "n_estimators": 1025,
    "learning_rate": 0.073,
    "max_depth": 7,
    "num_leaves": 17,
    "min_child_samples": 33,
    "subsample": 0.63,
    "colsample_bytree": 0.58,
    "reg_alpha": 0.04,
    "reg_lambda": 0.033,
    "min_split_gain": 0.096,
    "verbose": -1,
}


class LGBMPoissonModel:

    def __init__(self, rho=-0.108, temperature=0.90, params=None,
                 target_home="home_goals", target_away="away_goals",
                 feature_list=None, lambda_clip=(0.1, 5.0)):
        self.rho = rho
        self.temperature = temperature
        self.params = params or LGB_PARAMS.copy()
        self.model_home = None
        self.model_away = None
        self.feature_cols = None
        self.target_home = target_home
        self.target_away = target_away
        self._feature_list = feature_list or FEATURE_COLS
        self.lambda_clip = lambda_clip

    def _prepare_features(self, df):
        """Extract feature matrix, filling NaN with column medians."""
        available = [c for c in self._feature_list if c in df.columns]
        self.feature_cols = available
        X = df[available].copy()
        # Fill NaN with training median
        if not hasattr(self, "_medians") or self._medians is None:
            self._medians = X.median()
        X = X.fillna(self._medians)
        return X

    def fit(self, df, val_df=None):
        """Train home and away Poisson models.

        Args:
            df: DataFrame with feature columns and target columns.
            val_df: Optional validation DataFrame for early stopping.
                    If None, uses last 20% of training data (by time).
        """
        # Drop rows where target or key features are missing
        mask = df[self.target_home].notna() & df[self.target_away].notna()
        df_train = df[mask].copy()

        X = self._prepare_features(df_train)
        y_home = df_train[self.target_home].values
        y_away = df_train[self.target_away].values

        # Early stopping setup
        callbacks = [lgb.early_stopping(50, verbose=False)]

        if val_df is not None:
            X_val = self._prepare_features(val_df)
            y_val_home = val_df[self.target_home].values
            y_val_away = val_df[self.target_away].values
        else:
            # Use last 20% as validation (temporal split)
            split = int(len(X) * 0.8)
            X_val = X.iloc[split:]
            y_val_home = y_home[split:]
            y_val_away = y_away[split:]
            X = X.iloc[:split]
            y_home = y_home[:split]
            y_away = y_away[:split]

        # Train home goals model
        self.model_home = lgb.LGBMRegressor(**self.params)
        self.model_home.fit(X, y_home,
                            eval_set=[(X_val, y_val_home)],
                            callbacks=callbacks)

        # Train away goals model
        self.model_away = lgb.LGBMRegressor(**self.params)
        self.model_away.fit(X, y_away,
                            eval_set=[(X_val, y_val_away)],
                            callbacks=callbacks)

        return self

    def predict_lambdas(self, df):
        """Predict λ_home and λ_away for each match.

        Returns:
            (lambda_home, lambda_away) arrays
        """
        X = self._prepare_features(df)
        # Support both LGBMRegressor (training) and Booster (loaded)
        if isinstance(self.model_home, lgb.Booster):
            lambda_home = self.model_home.predict(X.values)
            lambda_away = self.model_away.predict(X.values)
        else:
            lambda_home = self.model_home.predict(X)
            lambda_away = self.model_away.predict(X)
        # Clip to reasonable range
        lambda_home = np.clip(lambda_home, *self.lambda_clip)
        lambda_away = np.clip(lambda_away, *self.lambda_clip)
        return lambda_home, lambda_away

    @staticmethod
    def _tau(i, j, lh, la, rho):
        """Dixon-Coles low-score correction."""
        if i == 0 and j == 0:
            return 1 - lh * la * rho
        elif i == 0 and j == 1:
            return 1 + lh * rho
        elif i == 1 and j == 0:
            return 1 + la * rho
        elif i == 1 and j == 1:
            return 1 - rho
        return 1.0

    def predict_score_matrix(self, lh, la, max_goals=7):
        """Build 7×7 score probability matrix from λ parameters."""
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                tau = self._tau(i, j, lh, la, self.rho)
                matrix[i, j] = tau * poisson.pmf(i, lh) * poisson.pmf(j, la)
        matrix /= matrix.sum()
        return matrix

    @staticmethod
    def _apply_temperature(probs, temperature):
        """Temperature scaling: T<1 sharpens (more confident), T>1 softens."""
        if temperature == 1.0:
            return probs
        log_p = np.log(np.clip(probs, 1e-8, 1.0))
        scaled = log_p / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_p = np.exp(scaled)
        return exp_p / exp_p.sum(axis=1, keepdims=True)

    def predict_1x2(self, df, max_goals=7):
        """Predict [P(home), P(draw), P(away)] for each match.

        Applies temperature scaling to correct conservative bias at extremes.

        Returns:
            (N, 3) array of probabilities
        """
        lh_arr, la_arr = self.predict_lambdas(df)
        probs = []
        for lh, la in zip(lh_arr, la_arr):
            matrix = self.predict_score_matrix(lh, la, max_goals)
            p_home = np.tril(matrix, -1).sum()
            p_draw = np.trace(matrix)
            p_away = np.triu(matrix, 1).sum()
            p = np.array([p_home, p_draw, p_away])
            probs.append(p / p.sum())
        probs = np.array(probs)
        if self.temperature != 1.0:
            probs = self._apply_temperature(probs, self.temperature)
        return probs

    def save(self, directory="models"):
        """Save trained model to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        self.model_home.booster_.save_model(str(path / "lgbm_home.txt"))
        self.model_away.booster_.save_model(str(path / "lgbm_away.txt"))

        meta = {
            "rho": self.rho,
            "temperature": self.temperature,
            "feature_cols": self.feature_cols,
            "medians": self._medians.to_dict(),
            "params": self.params,
            "target_home": self.target_home,
            "target_away": self.target_away,
            "lambda_clip": list(self.lambda_clip),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  模型已保存到 {path}/")

    @classmethod
    def load(cls, directory="models"):
        """Load trained model from directory."""
        path = Path(directory)
        with open(path / "meta.json") as f:
            meta = json.load(f)

        model = cls(
            rho=meta["rho"],
            temperature=meta["temperature"],
            params=meta["params"],
            target_home=meta.get("target_home", "home_goals"),
            target_away=meta.get("target_away", "away_goals"),
            lambda_clip=tuple(meta.get("lambda_clip", [0.1, 5.0])),
        )
        model.feature_cols = meta["feature_cols"]
        model._feature_list = meta["feature_cols"]

        import pandas as pd
        model._medians = pd.Series(meta["medians"])

        model.model_home = lgb.Booster(model_file=str(path / "lgbm_home.txt"))
        model.model_away = lgb.Booster(model_file=str(path / "lgbm_away.txt"))

        print(f"  模型已加载: {len(model.feature_cols)} 特征, ρ={model.rho}, T={model.temperature}")
        return model

    def feature_importance(self, top_n=20):
        """Return top feature importances (average of home and away models)."""
        import pandas as pd
        imp_home = self.model_home.feature_importances_
        imp_away = self.model_away.feature_importances_
        avg_imp = (imp_home + imp_away) / 2
        fi = pd.DataFrame({
            "feature": self.feature_cols,
            "importance_home": imp_home,
            "importance_away": imp_away,
            "importance_avg": avg_imp,
        }).sort_values("importance_avg", ascending=False)
        return fi.head(top_n)
