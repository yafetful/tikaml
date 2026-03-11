"""Phase 2: LightGBM Poisson model evaluation with forward-chain validation.

Compares LightGBM against Dixon-Coles baseline.

Usage:
    source .venv/bin/activate
    python scripts/run_lgbm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel
from src.dixon_coles import DixonColesModel
from src.evaluation import evaluate_predictions, match_outcome

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}


def run_validation():
    print("=" * 70)
    print("TikaML Phase 2: LightGBM Poisson 模型验证")
    print("=" * 70)

    # Load feature table
    print("\n加载特征表...")
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df[df["season"] != "2025-2026"].copy()
    print(f"  {len(df)} 场比赛, {len(df.columns)} 列")

    # Check feature availability
    from src.lgbm_poisson import FEATURE_COLS
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    print(f"  可用特征: {len(available)}/{len(FEATURE_COLS)}")
    if missing:
        print(f"  缺失特征: {missing}")

    all_results = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        # Train ONE LightGBM model on ALL leagues (cross-league training)
        all_train = df[df["season"] < test_season].copy()
        all_train_lgbm = all_train[all_train["season"] >= "2016-2017"].copy()

        # Use last season before test as validation for early stopping
        prev_season = sorted(all_train_lgbm["season"].unique())[-1]
        val_data = all_train_lgbm[all_train_lgbm["season"] == prev_season]
        train_data = all_train_lgbm[all_train_lgbm["season"] < prev_season]

        lgbm = LGBMPoissonModel(rho=-0.108)
        lgbm.fit(train_data, val_df=val_data)

        for league in LEAGUES:
            league_df = df[df["league"] == league].copy()
            train = league_df[league_df["season"] < test_season]
            test = league_df[league_df["season"] == test_season]

            if len(test) == 0:
                continue

            # LightGBM predictions (model already trained)
            lgbm_probs = lgbm.predict_1x2(test)
            lgbm_probs = lgbm.predict_1x2(test)

            # Dixon-Coles baseline (for comparison)
            dc = DixonColesModel(half_life_days=180)
            test_start = test["date"].min()
            dc.fit(train[["date", "home_team", "away_team",
                          "home_goals", "away_goals"]],
                   current_date=test_start)

            dc_probs = []
            for _, row in test.iterrows():
                try:
                    p = dc.predict_1x2(row["home_team"], row["away_team"])
                    if np.any(np.isnan(p)):
                        p = np.array([0.40, 0.30, 0.30])
                except (KeyError, ValueError):
                    p = np.array([0.40, 0.30, 0.30])
                dc_probs.append(p)
            dc_probs = np.array(dc_probs)

            # Outcomes
            outcomes = np.array([
                match_outcome(r["home_goals"], r["away_goals"])
                for _, r in test.iterrows()
            ])

            # Evaluate
            lgbm_metrics = evaluate_predictions(lgbm_probs, outcomes)
            dc_metrics = evaluate_predictions(dc_probs, outcomes)

            name = LEAGUE_NAMES.get(league, league)
            print(f"\n  {name} ({league}) — {len(test)} 场")
            print(f"    {'模型':<18} {'RPS':>8} {'Brier':>8} {'LogLoss':>8} {'准确率':>8}")
            print(f"    {'LightGBM':<18} {lgbm_metrics['rps']:>8.4f} "
                  f"{lgbm_metrics['brier']:>8.4f} {lgbm_metrics['log_loss']:>8.4f} "
                  f"{lgbm_metrics['accuracy']:>7.1%}")
            print(f"    {'Dixon-Coles':<18} {dc_metrics['rps']:>8.4f} "
                  f"{dc_metrics['brier']:>8.4f} {dc_metrics['log_loss']:>8.4f} "
                  f"{dc_metrics['accuracy']:>7.1%}")

            delta_rps = lgbm_metrics["rps"] - dc_metrics["rps"]
            print(f"    → LightGBM vs DC: RPS {delta_rps:+.4f} "
                  f"({'改善' if delta_rps < 0 else '退步'})")

            all_results.append({
                "season": test_season,
                "league": league,
                "n_matches": lgbm_metrics["n_matches"],
                "rps_lgbm": lgbm_metrics["rps"],
                "rps_dc": dc_metrics["rps"],
                "acc_lgbm": lgbm_metrics["accuracy"],
                "acc_dc": dc_metrics["accuracy"],
                "brier_lgbm": lgbm_metrics["brier"],
            })

        # Print feature importance for the last model
        print(f"\n  特征重要性 (最后一个模型):")
        fi = lgbm.feature_importance(top_n=15)
        for _, r in fi.iterrows():
            print(f"    {r['feature']:<40} {r['importance_avg']:>6.0f}")

    # Summary
    results_df = pd.DataFrame(all_results)
    print(f"\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")

    print(f"\n各联赛 LightGBM vs Dixon-Coles 平均 RPS:")
    print(f"  {'联赛':<6} {'LightGBM':>10} {'Dixon-Coles':>12} {'差值':>8} {'准确率':>8}")
    for league in LEAGUES:
        lg = results_df[results_df["league"] == league]
        if len(lg) == 0:
            continue
        name = LEAGUE_NAMES.get(league, league)
        avg_lgbm = lg["rps_lgbm"].mean()
        avg_dc = lg["rps_dc"].mean()
        avg_acc = lg["acc_lgbm"].mean()
        delta = avg_lgbm - avg_dc
        print(f"  {name:<6} {avg_lgbm:>10.4f} {avg_dc:>12.4f} {delta:>+8.4f} {avg_acc:>7.1%}")

    overall_lgbm = results_df["rps_lgbm"].mean()
    overall_dc = results_df["rps_dc"].mean()
    overall_acc = results_df["acc_lgbm"].mean()
    delta = overall_lgbm - overall_dc
    print(f"\n  {'总计':<6} {overall_lgbm:>10.4f} {overall_dc:>12.4f} "
          f"{delta:>+8.4f} {overall_acc:>7.1%}")

    print(f"\n参考基准:")
    print(f"  Dixon-Coles: ~0.205")
    print(f"  本方案目标:  ~0.190-0.198")
    print(f"  博彩公司:    ~0.185")


if __name__ == "__main__":
    run_validation()
