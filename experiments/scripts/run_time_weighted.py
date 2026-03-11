"""Test time-weighted training: weight recent matches more heavily.

Usage:
    source .venv/bin/activate
    python scripts/run_time_weighted.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import FEATURE_COLS, LGB_PARAMS
from src.evaluation import evaluate_predictions, match_outcome
from scipy.stats import poisson

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}


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


def predict_1x2(lh_arr, la_arr, rho=-0.108, max_goals=7):
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


def prepare_features(df, medians=None):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    if medians is None:
        medians = X.median()
    return X.fillna(medians), available, medians


def train_with_time_weights(df, test_season, half_life_years):
    all_train = df[
        (df["season"] < test_season) &
        (df["season"] >= "2016-2017")
    ].copy()
    if len(all_train) < 500:
        return None

    seasons = sorted(all_train["season"].unique())
    if len(seasons) < 2:
        return None
    prev_season = seasons[-1]
    val_data = all_train[all_train["season"] == prev_season]
    train_data = all_train[all_train["season"] < prev_season]

    X_train, cols, medians = prepare_features(train_data)
    X_val = train_data_to_X(val_data, cols, medians)

    y_home = train_data["home_goals"].values
    y_away = train_data["away_goals"].values
    y_val_home = val_data["home_goals"].values
    y_val_away = val_data["away_goals"].values

    # Time weights
    max_date = train_data["date"].max()
    days_ago = (max_date - train_data["date"]).dt.days.values
    sample_weight = np.power(0.5, days_ago / (half_life_years * 365))

    params = LGB_PARAMS.copy()
    callbacks = [lgb.early_stopping(50, verbose=False)]

    model_home = lgb.LGBMRegressor(**params)
    model_home.fit(X_train, y_home, eval_set=[(X_val, y_val_home)],
                   callbacks=callbacks, sample_weight=sample_weight)

    model_away = lgb.LGBMRegressor(**params)
    model_away.fit(X_train, y_away, eval_set=[(X_val, y_val_away)],
                   callbacks=callbacks, sample_weight=sample_weight)

    return model_home, model_away, cols, medians


def train_data_to_X(df, cols, medians):
    return df[cols].copy().fillna(medians)


def run():
    print("=" * 70)
    print("TikaML: 时间加权训练实验")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)

    half_lives = [1.0, 1.5, 2.0, 3.0, 5.0, None]  # None = no weighting
    results_by_hl = {hl: [] for hl in half_lives}

    for test_season in TEST_SEASONS:
        print(f"\n测试赛季: {test_season}")
        test = df[df["season"] == test_season].copy()
        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in test.iterrows()
        ])

        for hl in half_lives:
            if hl is not None:
                result = train_with_time_weights(df, test_season, hl)
            else:
                # No weighting (baseline)
                from src.lgbm_poisson import LGBMPoissonModel
                lgbm = LGBMPoissonModel(rho=-0.108)
                all_train = df[
                    (df["season"] < test_season) &
                    (df["season"] >= "2016-2017")
                ].copy()
                seasons = sorted(all_train["season"].unique())
                prev = seasons[-1]
                lgbm.fit(all_train[all_train["season"] < prev],
                         val_df=all_train[all_train["season"] == prev])
                probs = lgbm.predict_1x2(test)
                m = evaluate_predictions(probs, outcomes)
                results_by_hl[hl].append(m["rps"])
                continue

            if result is None:
                results_by_hl[hl].append(None)
                continue

            model_home, model_away, cols, medians = result
            X_test = train_data_to_X(test, cols, medians)
            lh = np.clip(model_home.predict(X_test), 0.1, 5.0)
            la = np.clip(model_away.predict(X_test), 0.1, 5.0)
            probs = predict_1x2(lh, la)
            m = evaluate_predictions(probs, outcomes)
            results_by_hl[hl].append(m["rps"])

    print(f"\n{'=' * 70}")
    print("汇总: 不同 half-life 的平均 RPS")
    print(f"{'=' * 70}")
    print(f"\n  {'Half-life':>12} {'2022-23':>10} {'2023-24':>10} {'2024-25':>10} {'平均':>10}")
    for hl in half_lives:
        rps_list = results_by_hl[hl]
        label = f"{hl}年" if hl else "无加权"
        vals = [f"{r:.4f}" if r else "  N/A " for r in rps_list]
        valid = [r for r in rps_list if r is not None]
        avg = np.mean(valid) if valid else 0
        print(f"  {label:>12} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {avg:>10.4f}")


if __name__ == "__main__":
    run()
