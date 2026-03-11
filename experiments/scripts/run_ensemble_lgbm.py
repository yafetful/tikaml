"""Experiment: LightGBM ensemble (multiple models with different seeds/subsets).

Tests whether averaging multiple LightGBM models reduces variance and improves RPS.

Usage:
    source .venv/bin/activate
    python scripts/run_ensemble_lgbm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS, LGB_PARAMS
from src.evaluation import evaluate_predictions, match_outcome

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}
N_MODELS = 5  # Number of models to average


def train_ensemble(df, test_season, n_models=5):
    """Train multiple LightGBM models with different seeds."""
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

    models = []
    for seed in range(n_models):
        params = LGB_PARAMS.copy()
        params["random_state"] = seed * 42
        params["subsample_seed"] = seed * 42
        # Slight variation in subsample and colsample for diversity
        params["subsample"] = max(0.5, params["subsample"] - 0.05 * seed)
        params["colsample_bytree"] = max(0.4, params["colsample_bytree"] - 0.03 * seed)

        lgbm = LGBMPoissonModel(rho=-0.108, params=params)
        lgbm.fit(train_data, val_df=val_data)
        models.append(lgbm)

    return models


def predict_ensemble(models, test_df):
    """Average predictions from multiple models."""
    all_probs = []
    all_lh = []
    all_la = []

    for model in models:
        probs = model.predict_1x2(test_df)
        lh, la = model.predict_lambdas(test_df)
        all_probs.append(probs)
        all_lh.append(lh)
        all_la.append(la)

    avg_probs = np.mean(all_probs, axis=0)
    # Renormalize
    row_sums = avg_probs.sum(axis=1, keepdims=True)
    avg_probs = avg_probs / row_sums

    return avg_probs


def run():
    print("=" * 70)
    print("TikaML: LightGBM Ensemble (Bagging) 验证")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  数据: {len(df)} 场比赛, {len(df.columns)} 列")
    print(f"  集成模型数: {N_MODELS}")

    all_results = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        # Train ensemble
        models = train_ensemble(df, test_season, N_MODELS)
        if models is None:
            print("  训练失败")
            continue

        # Also train single model for comparison
        single = train_ensemble(df, test_season, 1)

        for league in LEAGUES:
            test = df[(df["season"] == test_season) & (df["league"] == league)]
            if len(test) == 0:
                continue

            # Ensemble predictions
            ens_probs = predict_ensemble(models, test)

            # Single model predictions
            single_probs = predict_ensemble(single, test)

            outcomes = np.array([
                match_outcome(r["home_goals"], r["away_goals"])
                for _, r in test.iterrows()
            ])

            ens_m = evaluate_predictions(ens_probs, outcomes)
            sin_m = evaluate_predictions(single_probs, outcomes)

            name = LEAGUE_NAMES.get(league, league)
            print(f"\n  {name} ({league}) — {len(test)} 场")
            print(f"    {'模型':<18} {'RPS':>8} {'Brier':>8} {'准确率':>8}")
            print(f"    {'Ensemble({N_MODELS})':<18} {ens_m['rps']:>8.4f} "
                  f"{ens_m['brier']:>8.4f} {ens_m['accuracy']:>7.1%}")
            print(f"    {'Single':<18} {sin_m['rps']:>8.4f} "
                  f"{sin_m['brier']:>8.4f} {sin_m['accuracy']:>7.1%}")

            all_results.append({
                "season": test_season, "league": league,
                "rps_ens": ens_m["rps"], "rps_single": sin_m["rps"],
                "acc_ens": ens_m["accuracy"],
            })

    if all_results:
        rdf = pd.DataFrame(all_results)
        print(f"\n{'=' * 70}")
        print("汇总")
        print(f"{'=' * 70}")
        print(f"\n  {'联赛':<6} {'Ensemble':>10} {'Single':>10} {'差值':>8}")
        for league in LEAGUES:
            lg = rdf[rdf["league"] == league]
            if len(lg) == 0:
                continue
            name = LEAGUE_NAMES.get(league, league)
            ens = lg["rps_ens"].mean()
            sin = lg["rps_single"].mean()
            d = ens - sin
            print(f"  {name:<6} {ens:>10.4f} {sin:>10.4f} {d:>+8.4f}")

        ens_total = rdf["rps_ens"].mean()
        sin_total = rdf["rps_single"].mean()
        d_total = ens_total - sin_total
        print(f"\n  {'总计':<6} {ens_total:>10.4f} {sin_total:>10.4f} {d_total:>+8.4f}")


if __name__ == "__main__":
    run()
