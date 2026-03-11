"""Phase 3: Stacking ensemble evaluation.

Combines Dixon-Coles + LightGBM + Elo via meta-learner + isotonic calibration.

Usage:
    source .venv/bin/activate
    python scripts/run_stacking.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dixon_coles import DixonColesModel
from src.lgbm_poisson import LGBMPoissonModel
from src.elo import EloModel
from src.stacking import StackedPredictor
from src.evaluation import evaluate_predictions, match_outcome

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}


def predict_dc_for_df(df_test, df_train_all):
    """Get Dixon-Coles predictions aligned with df_test rows."""
    probs = []
    lambdas = []

    for league in df_test["league"].unique():
        league_test = df_test[df_test["league"] == league]
        league_train = df_train_all[df_train_all["league"] == league]

        dc = DixonColesModel(half_life_days=180)
        dc.fit(league_train[["date", "home_team", "away_team",
                              "home_goals", "away_goals"]],
               current_date=league_test["date"].min())

        for idx, row in league_test.iterrows():
            try:
                p = dc.predict_1x2(row["home_team"], row["away_team"])
                lh, la = dc._get_lambdas(row["home_team"], row["away_team"])
                if np.any(np.isnan(p)):
                    p, lh, la = np.array([0.40, 0.30, 0.30]), 1.4, 1.1
            except (KeyError, ValueError):
                p, lh, la = np.array([0.40, 0.30, 0.30]), 1.4, 1.1
            probs.append((idx, p))
            lambdas.append((idx, [lh, la]))

    # Sort by original index to align with df_test
    probs.sort(key=lambda x: x[0])
    lambdas.sort(key=lambda x: x[0])
    return np.array([p for _, p in probs]), np.array([l for _, l in lambdas])


def predict_elo_for_df(df_test, df_train_all):
    """Get Elo predictions aligned with df_test rows."""
    elo = EloModel(k=20, home_advantage=100)
    elo.fit(df_train_all[["date", "league", "season",
                           "home_team", "away_team",
                           "home_goals", "away_goals"]])

    probs = []
    lambdas = []
    # Process test matches in date order, updating Elo as we go
    for idx, row in df_test.sort_values("date").iterrows():
        p = elo.predict_1x2(row["home_team"], row["away_team"], row["league"])
        lh, la = elo.predict_lambdas(row["home_team"], row["away_team"], row["league"])
        probs.append((idx, p))
        lambdas.append((idx, [lh, la]))
        elo.update(row["home_team"], row["away_team"],
                   row["home_goals"], row["away_goals"])

    # Sort by original index
    probs.sort(key=lambda x: x[0])
    lambdas.sort(key=lambda x: x[0])
    return np.array([p for _, p in probs]), np.array([l for _, l in lambdas])


def predict_lgbm_for_df(df_test, lgbm_model):
    """Get LightGBM predictions aligned with df_test rows."""
    lgbm_probs = lgbm_model.predict_1x2(df_test)
    lh, la = lgbm_model.predict_lambdas(df_test)
    lgbm_lambdas = np.column_stack([lh, la])
    return lgbm_probs, lgbm_lambdas


def train_lgbm_model(features_df, test_season):
    """Train cross-league LightGBM with early stopping."""
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


def collect_predictions(df_season, df_train_all, features_df, lgbm_model):
    """Collect aligned predictions from all 3 base models for a season."""
    dc_probs, dc_lambdas = predict_dc_for_df(df_season, df_train_all)
    elo_probs, elo_lambdas = predict_elo_for_df(df_season, df_train_all)
    lgbm_probs, lgbm_lambdas = predict_lgbm_for_df(df_season, lgbm_model)

    outcomes = np.array([
        match_outcome(r["home_goals"], r["away_goals"])
        for _, r in df_season.iterrows()
    ])

    return dc_probs, dc_lambdas, lgbm_probs, lgbm_lambdas, \
        elo_probs, elo_lambdas, outcomes


def run_stacking():
    print("=" * 70)
    print("TikaML Phase 3: Stacking Ensemble 验证")
    print("=" * 70)

    # Load features table (has everything: matches + stats + features)
    print("\n加载数据...")
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} 场比赛, {len(df.columns)} 列")

    all_results = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        df_test = df[df["season"] == test_season].copy()
        df_train = df[df["season"] < test_season].copy()

        # Train LightGBM for this test season
        lgbm = train_lgbm_model(df, test_season)
        if lgbm is None:
            print("  LightGBM 训练失败，跳过")
            continue

        # --- Build stacking training data from prior seasons ---
        stack_seasons = sorted(
            s for s in df["season"].unique()
            if "2018-2019" <= s < test_season
        )
        print(f"  Stacking 训练赛季: {stack_seasons}")

        meta_features_all = []
        outcomes_all = []

        for ss in stack_seasons:
            ss_test = df[df["season"] == ss].copy()
            ss_train = df[df["season"] < ss].copy()

            # Train LightGBM for this stacking season
            ss_lgbm = train_lgbm_model(df, ss)
            if ss_lgbm is None:
                continue

            dc_p, dc_l, lg_p, lg_l, el_p, el_l, outs = collect_predictions(
                ss_test, ss_train, df, ss_lgbm)

            meta = StackedPredictor.build_meta_features(
                dc_p, dc_l, lg_p, lg_l, el_p, el_l)
            meta_features_all.append(meta)
            outcomes_all.append(outs)

        if not meta_features_all:
            print("  Stacking 训练数据不足，跳过")
            continue

        meta_X = np.vstack(meta_features_all)
        meta_y = np.concatenate(outcomes_all)
        print(f"  Stacking 训练样本: {len(meta_y)}")

        # --- Evaluate on test season ---
        dc_probs, dc_lambdas, lgbm_probs, lgbm_lambdas, \
            elo_probs, elo_lambdas, outcomes = collect_predictions(
                df_test, df_train, df, lgbm)

        meta_test = StackedPredictor.build_meta_features(
            dc_probs, dc_lambdas, lgbm_probs, lgbm_lambdas,
            elo_probs, elo_lambdas)

        # Train both meta-learners for comparison
        stacker_lgbm = StackedPredictor(meta_learner="lgbm")
        stacker_lgbm.fit(meta_X, meta_y, calibration_split=0.3)
        stacked_probs_lgbm = stacker_lgbm.predict(meta_test)

        stacker_lr = StackedPredictor(meta_learner="lr")
        stacker_lr.fit(meta_X, meta_y, calibration_split=0.3)
        stacked_probs_lr = stacker_lr.predict(meta_test)

        # Use the better one
        stacked_probs = stacked_probs_lgbm

        # Per-league results
        for league in LEAGUES:
            mask = df_test["league"].values == league
            if mask.sum() == 0:
                continue

            s_lgbm_m = evaluate_predictions(stacked_probs_lgbm[mask], outcomes[mask])
            s_lr_m = evaluate_predictions(stacked_probs_lr[mask], outcomes[mask])
            l_m = evaluate_predictions(lgbm_probs[mask], outcomes[mask])
            d_m = evaluate_predictions(dc_probs[mask], outcomes[mask])
            e_m = evaluate_predictions(elo_probs[mask], outcomes[mask])

            name = LEAGUE_NAMES.get(league, league)
            print(f"\n  {name} ({league}) — {mask.sum()} 场")
            print(f"    {'模型':<18} {'RPS':>8} {'Brier':>8} {'准确率':>8}")
            print(f"    {'Stack(LGBM)':<18} {s_lgbm_m['rps']:>8.4f} "
                  f"{s_lgbm_m['brier']:>8.4f} {s_lgbm_m['accuracy']:>7.1%}")
            print(f"    {'Stack(LR)':<18} {s_lr_m['rps']:>8.4f} "
                  f"{s_lr_m['brier']:>8.4f} {s_lr_m['accuracy']:>7.1%}")
            print(f"    {'LightGBM':<18} {l_m['rps']:>8.4f} "
                  f"{l_m['brier']:>8.4f} {l_m['accuracy']:>7.1%}")
            print(f"    {'Dixon-Coles':<18} {d_m['rps']:>8.4f} "
                  f"{d_m['brier']:>8.4f} {d_m['accuracy']:>7.1%}")
            print(f"    {'Elo':<18} {e_m['rps']:>8.4f} "
                  f"{e_m['brier']:>8.4f} {e_m['accuracy']:>7.1%}")

            all_results.append({
                "season": test_season, "league": league,
                "n": mask.sum(),
                "rps_stack": s_lgbm_m["rps"], "rps_stack_lr": s_lr_m["rps"],
                "rps_lgbm": l_m["rps"],
                "rps_dc": d_m["rps"], "rps_elo": e_m["rps"],
                "acc_stack": s_lgbm_m["accuracy"], "acc_lgbm": l_m["accuracy"],
            })

    # Summary
    if all_results:
        rdf = pd.DataFrame(all_results)
        print(f"\n{'=' * 70}")
        print("汇总")
        print(f"{'=' * 70}")

        print(f"\n  {'联赛':<6} {'Stacking':>10} {'LightGBM':>10} "
              f"{'Dixon-Coles':>12} {'Elo':>8} {'准确率(S)':>10}")
        for league in LEAGUES:
            lg = rdf[rdf["league"] == league]
            if len(lg) == 0:
                continue
            name = LEAGUE_NAMES.get(league, league)
            print(f"  {name:<6} {lg['rps_stack'].mean():>10.4f} "
                  f"{lg['rps_lgbm'].mean():>10.4f} "
                  f"{lg['rps_dc'].mean():>12.4f} "
                  f"{lg['rps_elo'].mean():>8.4f} "
                  f"{lg['acc_stack'].mean():>9.1%}")

        print(f"\n  {'总计':<6} {rdf['rps_stack'].mean():>10.4f} "
              f"{rdf['rps_lgbm'].mean():>10.4f} "
              f"{rdf['rps_dc'].mean():>12.4f} "
              f"{rdf['rps_elo'].mean():>8.4f} "
              f"{rdf['acc_stack'].mean():>9.1%}")

        # Improvement summary
        print(f"\n  Stacking vs 最佳单模型 (LightGBM):")
        for league in LEAGUES:
            lg = rdf[rdf["league"] == league]
            if len(lg) == 0:
                continue
            name = LEAGUE_NAMES.get(league, league)
            d = lg["rps_stack"].mean() - lg["rps_lgbm"].mean()
            print(f"    {name}: {d:+.4f}")
        d_total = rdf["rps_stack"].mean() - rdf["rps_lgbm"].mean()
        print(f"    总计: {d_total:+.4f}")

        print(f"\n参考基准:")
        print(f"  Stacking 目标: ~0.190")
        print(f"  博彩公司:     ~0.185")


if __name__ == "__main__":
    run_stacking()
