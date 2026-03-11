"""Calibrated LightGBM evaluation.

Tests LightGBM + isotonic calibration trained on prior season.

Usage:
    source .venv/bin/activate
    python scripts/run_calibrated.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel
from src.calibration import ProbabilityCalibrator
from src.evaluation import evaluate_predictions, match_outcome

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}


def train_lgbm_model(features_df, test_season):
    all_train = features_df[
        (features_df["season"] < test_season) &
        (features_df["season"] >= "2016-2017")
    ].copy()
    if len(all_train) < 500:
        return None
    seasons = sorted(all_train["season"].unique())
    if len(seasons) < 2:
        return None
    prev_season = seasons[-1]
    val_data = all_train[all_train["season"] == prev_season]
    train_data = all_train[all_train["season"] < prev_season]
    lgbm = LGBMPoissonModel(rho=-0.108)
    lgbm.fit(train_data, val_df=val_data)
    return lgbm


def run():
    print("=" * 70)
    print("TikaML: LightGBM + 校准验证")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  数据: {len(df)} 场, {len(df.columns)} 列")

    all_results = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        df_test = df[df["season"] == test_season].copy()
        lgbm = train_lgbm_model(df, test_season)
        if lgbm is None:
            continue

        # Raw predictions on test set
        raw_probs = lgbm.predict_1x2(df_test)
        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in df_test.iterrows()
        ])

        # --- Build calibration set from 2 prior seasons ---
        prior_seasons = sorted(
            s for s in df["season"].unique()
            if "2018-2019" <= s < test_season
        )
        cal_probs_list = []
        cal_outcomes_list = []
        cal_leagues_list = []

        for cs in prior_seasons[-2:]:
            cs_test = df[df["season"] == cs].copy()
            cs_lgbm = train_lgbm_model(df, cs)
            if cs_lgbm is None:
                continue
            cs_probs = cs_lgbm.predict_1x2(cs_test)
            cs_outcomes = np.array([
                match_outcome(r["home_goals"], r["away_goals"])
                for _, r in cs_test.iterrows()
            ])
            cal_probs_list.append(cs_probs)
            cal_outcomes_list.append(cs_outcomes)
            cal_leagues_list.append(cs_test["league"].values)

        # Global calibration
        if cal_probs_list:
            cal_probs_all = np.vstack(cal_probs_list)
            cal_outcomes_all = np.concatenate(cal_outcomes_list)

            calibrator = ProbabilityCalibrator()
            calibrator.fit(cal_probs_all, cal_outcomes_all)
            cal_probs = calibrator.predict(raw_probs)
        else:
            cal_probs = raw_probs

        # --- Evaluate ---
        for league in LEAGUES:
            mask = df_test["league"].values == league
            if mask.sum() == 0:
                continue

            raw_m = evaluate_predictions(raw_probs[mask], outcomes[mask])
            cal_m = evaluate_predictions(cal_probs[mask], outcomes[mask])

            name = LEAGUE_NAMES.get(league, league)
            print(f"\n  {name} ({league}) — {mask.sum()} 场")
            print(f"    {'模型':<18} {'RPS':>8} {'Brier':>8} {'准确率':>8}")
            print(f"    {'LGBM+Calibrated':<18} {cal_m['rps']:>8.4f} "
                  f"{cal_m['brier']:>8.4f} {cal_m['accuracy']:>7.1%}")
            print(f"    {'LGBM(raw)':<18} {raw_m['rps']:>8.4f} "
                  f"{raw_m['brier']:>8.4f} {raw_m['accuracy']:>7.1%}")

            all_results.append({
                "season": test_season, "league": league,
                "rps_cal": cal_m["rps"], "rps_raw": raw_m["rps"],
                "acc_cal": cal_m["accuracy"], "acc_raw": raw_m["accuracy"],
            })

    if all_results:
        rdf = pd.DataFrame(all_results)
        print(f"\n{'=' * 70}")
        print("汇总")
        print(f"{'=' * 70}")

        print(f"\n  {'联赛':<6} {'Calibrated':>12} {'Raw':>12} {'差值':>10}")
        for league in LEAGUES:
            lg = rdf[rdf["league"] == league]
            if len(lg) == 0:
                continue
            name = LEAGUE_NAMES.get(league, league)
            cal_avg = lg["rps_cal"].mean()
            raw_avg = lg["rps_raw"].mean()
            d = cal_avg - raw_avg
            print(f"  {name:<6} {cal_avg:>12.4f} {raw_avg:>12.4f} {d:>+10.4f}")

        cal_total = rdf["rps_cal"].mean()
        raw_total = rdf["rps_raw"].mean()
        d_total = cal_total - raw_total
        print(f"\n  {'总计':<6} {cal_total:>12.4f} {raw_total:>12.4f} {d_total:>+10.4f}")
        print(f"\n  校准后准确率: {rdf['acc_cal'].mean():.1%} "
              f"(原始: {rdf['acc_raw'].mean():.1%})")


if __name__ == "__main__":
    run()
