"""Phase 1 baseline: Dixon-Coles model evaluation with forward-chain validation.

Usage:
    source .venv/bin/activate
    python scripts/run_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.parser import load_matches
from src.dixon_coles import DixonColesModel
from src.evaluation import evaluate_predictions, match_outcome

# Seasons to test (forward-chain: train on all prior seasons)
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}


def baseline_always_home(n):
    """Baseline: always predict home win with typical probabilities."""
    return np.tile([0.45, 0.27, 0.28], (n, 1))


def baseline_uniform(n):
    """Baseline: uniform probabilities."""
    return np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))


def run_validation():
    print("=" * 70)
    print("TikaML Phase 1: Dixon-Coles 基线验证")
    print("=" * 70)

    # Load all matches
    print("\n加载比赛数据...")
    matches = load_matches()
    print(f"  总比赛数: {len(matches)}")
    print(f"  联赛: {sorted(matches['league'].unique())}")
    print(f"  赛季: {sorted(matches['season'].unique())}")

    # Filter out ongoing season (2025-2026)
    matches = matches[matches["season"] != "2025-2026"].copy()
    print(f"  去除进行中赛季后: {len(matches)}")

    all_results = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        for league in LEAGUES:
            league_matches = matches[matches["league"] == league]
            train = league_matches[league_matches["season"] < test_season]
            test = league_matches[league_matches["season"] == test_season]

            if len(test) == 0:
                continue

            # Fit Dixon-Coles on training data
            test_start_date = test["date"].min()
            model = DixonColesModel(half_life_days=180)
            model.fit(train, current_date=test_start_date)

            # Predict test matches
            probs_list = []
            outcomes = []
            skipped = 0
            for _, row in test.iterrows():
                outcome = match_outcome(row["home_goals"], row["away_goals"])
                try:
                    probs = model.predict_1x2(row["home_team"], row["away_team"])
                    if np.any(np.isnan(probs)):
                        probs = np.array([0.40, 0.30, 0.30])
                except (KeyError, ValueError):
                    # Unknown team (newly promoted)
                    probs = np.array([0.40, 0.30, 0.30])
                    skipped += 1
                probs_list.append(probs)
                outcomes.append(outcome)

            probs_arr = np.array(probs_list)
            outcomes_arr = np.array(outcomes)

            # Evaluate
            dc_metrics = evaluate_predictions(probs_arr, outcomes_arr)
            home_metrics = evaluate_predictions(
                baseline_always_home(len(outcomes_arr)), outcomes_arr)
            uniform_metrics = evaluate_predictions(
                baseline_uniform(len(outcomes_arr)), outcomes_arr)

            name = LEAGUE_NAMES.get(league, league)
            print(f"\n  {name} ({league}) — {len(test)} 场, 新队fallback: {skipped}")
            print(f"    {'模型':<18} {'RPS':>8} {'Brier':>8} {'LogLoss':>8} {'准确率':>8}")
            print(f"    {'Dixon-Coles':<18} {dc_metrics['rps']:>8.4f} "
                  f"{dc_metrics['brier']:>8.4f} {dc_metrics['log_loss']:>8.4f} "
                  f"{dc_metrics['accuracy']:>7.1%}")
            print(f"    {'固定主胜':<18} {home_metrics['rps']:>8.4f} "
                  f"{home_metrics['brier']:>8.4f} {home_metrics['log_loss']:>8.4f} "
                  f"{home_metrics['accuracy']:>7.1%}")
            print(f"    {'均匀分布':<18} {uniform_metrics['rps']:>8.4f} "
                  f"{uniform_metrics['brier']:>8.4f} {uniform_metrics['log_loss']:>8.4f} "
                  f"{uniform_metrics['accuracy']:>7.1%}")

            all_results.append({
                "season": test_season,
                "league": league,
                "n_matches": dc_metrics["n_matches"],
                "rps_dc": dc_metrics["rps"],
                "rps_home": home_metrics["rps"],
                "rps_uniform": uniform_metrics["rps"],
                "acc_dc": dc_metrics["accuracy"],
                "brier_dc": dc_metrics["brier"],
                "log_loss_dc": dc_metrics["log_loss"],
            })

            # Print top teams
            ratings = model.team_ratings()
            print(f"    Top-5 攻击力: {', '.join(ratings.head(5)['team'].tolist())}")

    # Summary table
    results_df = pd.DataFrame(all_results)
    print(f"\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")

    # Per-league average
    print(f"\n各联赛 Dixon-Coles 平均 RPS:")
    print(f"  {'联赛':<6} {'RPS':>8} {'vs 主胜':>10} {'vs 均匀':>10} {'准确率':>8}")
    for league in LEAGUES:
        lg = results_df[results_df["league"] == league]
        if len(lg) == 0:
            continue
        name = LEAGUE_NAMES.get(league, league)
        avg_rps = lg["rps_dc"].mean()
        avg_home = lg["rps_home"].mean()
        avg_uni = lg["rps_uniform"].mean()
        avg_acc = lg["acc_dc"].mean()
        print(f"  {name:<6} {avg_rps:>8.4f} {avg_rps - avg_home:>+10.4f} "
              f"{avg_rps - avg_uni:>+10.4f} {avg_acc:>7.1%}")

    # Overall
    overall_rps = results_df["rps_dc"].mean()
    overall_home = results_df["rps_home"].mean()
    overall_uni = results_df["rps_uniform"].mean()
    overall_acc = results_df["acc_dc"].mean()
    print(f"\n  {'总计':<6} {overall_rps:>8.4f} {overall_rps - overall_home:>+10.4f} "
          f"{overall_rps - overall_uni:>+10.4f} {overall_acc:>7.1%}")

    print(f"\n参考基准 (文档):")
    print(f"  随机猜测 RPS: ~0.286")
    print(f"  简单Elo RPS:  ~0.215")
    print(f"  Dixon-Coles:  ~0.205 (学术参考)")
    print(f"  Stacking目标: ~0.190")
    print(f"  博彩公司:     ~0.185")


if __name__ == "__main__":
    run_validation()
