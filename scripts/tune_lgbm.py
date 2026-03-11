"""Optuna hyperparameter tuning for LightGBM Poisson model.

Uses forward-chain validation (train on seasons < test, validate on test season).
Optimizes RPS on the validation set.

Usage:
    source .venv/bin/activate
    python scripts/tune_lgbm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation import evaluate_predictions, match_outcome
from src.lgbm_poisson import FEATURE_COLS

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_PATH = Path("data/opta/processed/features.csv")
# Use 2022-2023 as primary tuning target, validate on 2023-2024
TUNE_SEASON = "2022-2023"
VALIDATE_SEASON = "2023-2024"


def _tau(i, j, lh, la, rho):
    if i == 0 and j == 0:
        return 1 - lh * la * rho
    elif i == 0 and j == 1:
        return 1 + lh * rho
    elif i == 1 and j == 0:
        return 1 + la * rho
    elif i == 1 and j == 1:
        return 1 - rho
    return 1.0


def predict_1x2_from_lambdas(lh_arr, la_arr, rho=-0.05, max_goals=7):
    probs = []
    for lh, la in zip(lh_arr, la_arr):
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                tau = _tau(i, j, lh, la, rho)
                matrix[i, j] = tau * poisson.pmf(i, lh) * poisson.pmf(j, la)
        matrix /= matrix.sum()
        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()
        p = np.array([p_home, p_draw, p_away])
        probs.append(p / p.sum())
    return np.array(probs)


def prepare_features(df, feature_cols, medians=None):
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    if medians is None:
        medians = X.median()
    X = X.fillna(medians)
    return X, available, medians


def objective(trial, df, feature_cols):
    """Optuna objective: train LightGBM with trial params, return avg RPS."""
    params = {
        "objective": "poisson",
        "metric": "poisson",
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
        "verbose": -1,
    }
    rho = trial.suggest_float("rho", -0.15, 0.0)

    # Time-weighted training (optional)
    use_time_weight = trial.suggest_categorical("use_time_weight", [True, False])
    weight_half_life_years = trial.suggest_float("weight_half_life_years", 1.0, 5.0) \
        if use_time_weight else None

    rps_scores = []

    for test_season in [TUNE_SEASON, VALIDATE_SEASON]:
        all_train = df[
            (df["season"] < test_season) &
            (df["season"] >= "2016-2017")
        ].copy()

        test = df[df["season"] == test_season].copy()
        if len(test) == 0 or len(all_train) < 500:
            continue

        # Split train/val for early stopping
        seasons = sorted(all_train["season"].unique())
        if len(seasons) < 2:
            continue
        prev_season = seasons[-1]
        val_data = all_train[all_train["season"] == prev_season]
        train_data = all_train[all_train["season"] < prev_season]

        X_train, used_cols, medians = prepare_features(train_data, feature_cols)
        X_val = train_data_to_X(val_data, used_cols, medians)
        X_test = train_data_to_X(test, used_cols, medians)

        y_train_home = train_data["home_goals"].values
        y_train_away = train_data["away_goals"].values
        y_val_home = val_data["home_goals"].values
        y_val_away = val_data["away_goals"].values

        callbacks = [lgb.early_stopping(50, verbose=False)]

        # Sample weights (time decay)
        sample_weight = None
        if use_time_weight and weight_half_life_years is not None:
            max_date = train_data["date"].max()
            days_ago = (max_date - train_data["date"]).dt.days.values
            sample_weight = np.power(0.5, days_ago / (weight_half_life_years * 365))

        # Train home model
        model_home = lgb.LGBMRegressor(**params)
        model_home.fit(X_train, y_train_home,
                       eval_set=[(X_val, y_val_home)],
                       callbacks=callbacks,
                       sample_weight=sample_weight)

        # Train away model
        model_away = lgb.LGBMRegressor(**params)
        model_away.fit(X_train, y_train_away,
                       eval_set=[(X_val, y_val_away)],
                       callbacks=callbacks,
                       sample_weight=sample_weight)

        # Predict
        lh = np.clip(model_home.predict(X_test), 0.1, 5.0)
        la = np.clip(model_away.predict(X_test), 0.1, 5.0)
        probs = predict_1x2_from_lambdas(lh, la, rho=rho)

        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in test.iterrows()
        ])

        metrics = evaluate_predictions(probs, outcomes)
        rps_scores.append(metrics["rps"])

    return np.mean(rps_scores) if rps_scores else 1.0


def train_data_to_X(df, used_cols, medians):
    X = df[used_cols].copy()
    return X.fillna(medians)


def main():
    print("=" * 70)
    print("TikaML: LightGBM Optuna 超参数调优")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  数据: {len(df)} 场比赛")

    feature_cols = FEATURE_COLS

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    print(f"\n开始调优 (目标: 最小化 {TUNE_SEASON} + {VALIDATE_SEASON} 平均 RPS)...")
    print(f"  特征数: {len([c for c in feature_cols if c in df.columns])}")

    study.optimize(
        lambda trial: objective(trial, df, feature_cols),
        n_trials=80,
        show_progress_bar=True,
    )

    print(f"\n{'=' * 70}")
    print(f"最佳 RPS: {study.best_value:.4f}")
    print(f"{'=' * 70}")
    print("\n最佳超参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best params
    import json
    out_path = Path("data/opta/processed/best_lgbm_params.json")
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n参数已保存: {out_path}")


if __name__ == "__main__":
    main()
