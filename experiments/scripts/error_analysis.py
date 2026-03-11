"""Error analysis: find where model predictions are weakest.

Analyzes RPS by:
1. Probability bins (confident vs uncertain predictions)
2. Match outcome type (home/draw/away)
3. Goal difference categories
4. League
5. Season stage

Usage:
    source .venv/bin/activate
    python scripts/error_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel
from src.evaluation import ranked_probability_score, match_outcome

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]


def run_analysis():
    print("=" * 70)
    print("TikaML: 误差分析")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Collect all predictions across test seasons
    all_preds = []
    all_outcomes = []
    all_meta = []

    for test_season in TEST_SEASONS:
        all_train = df[
            (df["season"] < test_season) &
            (df["season"] >= "2016-2017")
        ].copy()
        test = df[df["season"] == test_season].copy()

        if len(test) == 0 or len(all_train) < 500:
            continue

        seasons = sorted(all_train["season"].unique())
        prev_season = seasons[-1]
        val_data = all_train[all_train["season"] == prev_season]
        train_data = all_train[all_train["season"] < prev_season]

        lgbm = LGBMPoissonModel(rho=-0.108)
        lgbm.fit(train_data, val_df=val_data)

        probs = lgbm.predict_1x2(test)
        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in test.iterrows()
        ])

        for i, (_, row) in enumerate(test.iterrows()):
            rps = ranked_probability_score(probs[i], outcomes[i])
            all_preds.append(probs[i])
            all_outcomes.append(outcomes[i])
            all_meta.append({
                "rps": rps,
                "outcome": outcomes[i],
                "outcome_label": ["home", "draw", "away"][outcomes[i]],
                "league": row["league"],
                "season": row["season"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
                "goal_diff": abs(row["home_goals"] - row["away_goals"]),
                "total_goals": row["home_goals"] + row["away_goals"],
                "max_prob": probs[i].max(),
                "pred_outcome": np.argmax(probs[i]),
                "pred_correct": np.argmax(probs[i]) == outcomes[i],
                "home_prob": probs[i][0],
                "draw_prob": probs[i][1],
                "away_prob": probs[i][2],
                "season_stage": row.get("season_stage", 0.5),
                "points_diff": row.get("points_diff", 0),
            })

    meta_df = pd.DataFrame(all_meta)
    print(f"\n总测试样本: {len(meta_df)}")
    print(f"总体 RPS: {meta_df['rps'].mean():.4f}")
    print(f"总体准确率: {meta_df['pred_correct'].mean():.1%}")

    # 1. RPS by actual outcome
    print(f"\n{'=' * 50}")
    print("1. 按实际结果分析")
    print(f"{'=' * 50}")
    for outcome_label in ["home", "draw", "away"]:
        subset = meta_df[meta_df["outcome_label"] == outcome_label]
        print(f"  {outcome_label:>5}: RPS={subset['rps'].mean():.4f}, "
              f"n={len(subset)}, "
              f"预测正确={subset['pred_correct'].mean():.1%}")

    # 2. RPS by confidence level
    print(f"\n{'=' * 50}")
    print("2. 按预测置信度分析")
    print(f"{'=' * 50}")
    bins = [(0, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 1.0)]
    for lo, hi in bins:
        subset = meta_df[(meta_df["max_prob"] >= lo) & (meta_df["max_prob"] < hi)]
        if len(subset) > 0:
            print(f"  P(max) [{lo:.2f}, {hi:.2f}): RPS={subset['rps'].mean():.4f}, "
                  f"n={len(subset)}, "
                  f"准确率={subset['pred_correct'].mean():.1%}")

    # 3. RPS by goal difference
    print(f"\n{'=' * 50}")
    print("3. 按进球差分析")
    print(f"{'=' * 50}")
    for gd_label, gd_filter in [
        ("平局 (GD=0)", meta_df["goal_diff"] == 0),
        ("小胜 (GD=1)", meta_df["goal_diff"] == 1),
        ("中胜 (GD=2)", meta_df["goal_diff"] == 2),
        ("大胜 (GD≥3)", meta_df["goal_diff"] >= 3),
    ]:
        subset = meta_df[gd_filter]
        if len(subset) > 0:
            print(f"  {gd_label:<14}: RPS={subset['rps'].mean():.4f}, "
                  f"n={len(subset)} ({len(subset)/len(meta_df):.1%})")

    # 4. RPS by league
    print(f"\n{'=' * 50}")
    print("4. 按联赛分析")
    print(f"{'=' * 50}")
    league_names = {"EPL": "英超", "LL": "西甲", "SEA": "意甲",
                    "BUN": "德甲", "LI1": "法甲"}
    for league in ["EPL", "LL", "SEA", "BUN", "LI1"]:
        subset = meta_df[meta_df["league"] == league]
        name = league_names.get(league, league)
        # Breakdown by outcome
        home_rps = subset[subset["outcome"] == 0]["rps"].mean()
        draw_rps = subset[subset["outcome"] == 1]["rps"].mean()
        away_rps = subset[subset["outcome"] == 2]["rps"].mean()
        print(f"  {name} ({league}): RPS={subset['rps'].mean():.4f}, "
              f"准确率={subset['pred_correct'].mean():.1%}")
        print(f"    Home={home_rps:.4f}, Draw={draw_rps:.4f}, Away={away_rps:.4f}")

    # 5. RPS by season stage
    print(f"\n{'=' * 50}")
    print("5. 按赛季阶段分析")
    print(f"{'=' * 50}")
    for label, lo, hi in [
        ("开局 (0-0.3)", 0, 0.3),
        ("中期 (0.3-0.6)", 0.3, 0.6),
        ("后期 (0.6-0.8)", 0.6, 0.8),
        ("收官 (0.8-1.0)", 0.8, 1.0),
    ]:
        subset = meta_df[
            (meta_df["season_stage"] >= lo) & (meta_df["season_stage"] < hi)
        ]
        if len(subset) > 0:
            print(f"  {label:<16}: RPS={subset['rps'].mean():.4f}, "
                  f"n={len(subset)}, 准确率={subset['pred_correct'].mean():.1%}")

    # 6. Calibration analysis
    print(f"\n{'=' * 50}")
    print("6. 概率校准分析 (predicted vs actual frequency)")
    print(f"{'=' * 50}")
    for outcome_idx, label in [(0, "Home"), (1, "Draw"), (2, "Away")]:
        prob_col = ["home_prob", "draw_prob", "away_prob"][outcome_idx]
        print(f"\n  {label}:")
        prob_bins = [(0, 0.15), (0.15, 0.25), (0.25, 0.35),
                     (0.35, 0.45), (0.45, 0.55), (0.55, 0.70), (0.70, 1.0)]
        for lo, hi in prob_bins:
            mask = (meta_df[prob_col] >= lo) & (meta_df[prob_col] < hi)
            subset = meta_df[mask]
            if len(subset) >= 10:
                actual_freq = (subset["outcome"] == outcome_idx).mean()
                pred_avg = subset[prob_col].mean()
                diff = actual_freq - pred_avg
                print(f"    P=[{lo:.2f},{hi:.2f}): pred={pred_avg:.3f}, "
                      f"actual={actual_freq:.3f}, "
                      f"diff={diff:+.3f}, n={len(subset)}")

    # 7. Upset analysis (low-prob outcomes)
    print(f"\n{'=' * 50}")
    print("7. 冷门分析 (模型预测概率 < 25% 的实际结果)")
    print(f"{'=' * 50}")
    upset_count = 0
    for i, row in meta_df.iterrows():
        outcome = row["outcome"]
        pred_prob = [row["home_prob"], row["draw_prob"], row["away_prob"]][outcome]
        if pred_prob < 0.25:
            upset_count += 1
    print(f"  冷门场次 (实际结果的预测概率<25%): {upset_count}/{len(meta_df)} "
          f"({upset_count/len(meta_df):.1%})")

    # Average RPS contribution from upsets vs non-upsets
    upset_rps = []
    normal_rps = []
    for i, row in meta_df.iterrows():
        outcome = int(row["outcome"])
        pred_prob = [row["home_prob"], row["draw_prob"], row["away_prob"]][outcome]
        if pred_prob < 0.25:
            upset_rps.append(row["rps"])
        else:
            normal_rps.append(row["rps"])
    print(f"  冷门 RPS: {np.mean(upset_rps):.4f}")
    print(f"  非冷门 RPS: {np.mean(normal_rps):.4f}")
    print(f"  冷门对总 RPS 贡献: "
          f"{len(upset_rps) / len(meta_df) * np.mean(upset_rps):.4f} "
          f"({len(upset_rps) / len(meta_df) * np.mean(upset_rps) / meta_df['rps'].mean():.1%})")


if __name__ == "__main__":
    run_analysis()
