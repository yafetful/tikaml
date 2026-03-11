"""Experiment: Optimized weighted ensemble with per-league calibration.

Instead of complex stacking, simply optimize weights for averaging
DC + LightGBM + Elo probabilities, with per-league isotonic calibration.

Usage:
    source .venv/bin/activate
    python scripts/run_weighted_ensemble.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel
from src.dixon_coles import DixonColesModel
from src.elo import EloModel
from src.evaluation import evaluate_predictions, match_outcome, ranked_probability_score

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}


def predict_dc_for_df(df_test, df_train_all):
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
    probs.sort(key=lambda x: x[0])
    lambdas.sort(key=lambda x: x[0])
    return np.array([p for _, p in probs]), np.array([l for _, l in lambdas])


def predict_elo_for_df(df_test, df_train_all):
    elo = EloModel(k=20, home_advantage=100)
    elo.fit(df_train_all[["date", "league", "season",
                           "home_team", "away_team",
                           "home_goals", "away_goals"]])
    probs = []
    for idx, row in df_test.sort_values("date").iterrows():
        p = elo.predict_1x2(row["home_team"], row["away_team"], row["league"])
        probs.append((idx, p))
        elo.update(row["home_team"], row["away_team"],
                   row["home_goals"], row["away_goals"])
    probs.sort(key=lambda x: x[0])
    return np.array([p for _, p in probs])


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


def optimize_weights(dc_probs, lgbm_probs, elo_probs, outcomes):
    """Find optimal blending weights to minimize RPS."""
    def objective(w):
        w = np.abs(w)  # ensure non-negative
        w = w / w.sum()  # normalize
        blended = w[0] * dc_probs + w[1] * lgbm_probs + w[2] * elo_probs
        # Normalize each row
        row_sums = blended.sum(axis=1, keepdims=True)
        blended = blended / row_sums
        return np.mean([
            ranked_probability_score(blended[i], outcomes[i])
            for i in range(len(outcomes))
        ])

    # Try multiple starting points
    best_result = None
    best_rps = float("inf")
    for init_w in [
        [0.33, 0.34, 0.33],
        [0.2, 0.6, 0.2],
        [0.1, 0.8, 0.1],
        [0.3, 0.5, 0.2],
        [0.15, 0.7, 0.15],
    ]:
        result = minimize(
            objective, init_w, method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-5}
        )
        if result.fun < best_rps:
            best_rps = result.fun
            best_result = result

    w = np.abs(best_result.x)
    w = w / w.sum()
    return w


def calibrate_per_league(test_probs, test_leagues,
                         cal_probs, cal_outcomes, cal_leagues):
    """Train isotonic calibration on cal set, apply to test set per league."""
    calibrated = test_probs.copy()

    for league in np.unique(test_leagues):
        cal_mask = cal_leagues == league
        test_mask = test_leagues == league
        if cal_mask.sum() < 30 or test_mask.sum() == 0:
            continue

        for i in range(3):
            y_binary = (cal_outcomes[cal_mask] == i).astype(int)
            cal = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
            cal.fit(cal_probs[cal_mask, i], y_binary)
            calibrated[test_mask, i] = cal.predict(test_probs[test_mask, i])

    # Renormalize
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    calibrated /= row_sums

    return calibrated


def run():
    print("=" * 70)
    print("TikaML: 加权集成 + 联赛校准")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  数据: {len(df)} 场比赛")

    all_results = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        df_test = df[df["season"] == test_season].copy()
        df_train = df[df["season"] < test_season].copy()

        # Train LightGBM
        lgbm = train_lgbm_model(df, test_season)
        if lgbm is None:
            continue

        # Get base model predictions
        dc_probs, _ = predict_dc_for_df(df_test, df_train)
        elo_probs = predict_elo_for_df(df_test, df_train)
        lgbm_probs = lgbm.predict_1x2(df_test)

        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in df_test.iterrows()
        ])

        # --- Method 1: Optimize global weights on prior season ---
        # Use the season before test as weight optimization set
        prior_seasons = sorted(
            s for s in df["season"].unique()
            if "2018-2019" <= s < test_season
        )
        if len(prior_seasons) >= 2:
            opt_season = prior_seasons[-1]
            opt_test = df[df["season"] == opt_season].copy()
            opt_train = df[df["season"] < opt_season].copy()

            opt_lgbm = train_lgbm_model(df, opt_season)
            if opt_lgbm is not None:
                opt_dc, _ = predict_dc_for_df(opt_test, opt_train)
                opt_elo = predict_elo_for_df(opt_test, opt_train)
                opt_lgbm_probs = opt_lgbm.predict_1x2(opt_test)
                opt_outcomes = np.array([
                    match_outcome(r["home_goals"], r["away_goals"])
                    for _, r in opt_test.iterrows()
                ])

                weights = optimize_weights(
                    opt_dc, opt_lgbm_probs, opt_elo, opt_outcomes)
            else:
                weights = np.array([0.15, 0.70, 0.15])
        else:
            weights = np.array([0.15, 0.70, 0.15])

        print(f"  最优权重: DC={weights[0]:.3f}, LGBM={weights[1]:.3f}, Elo={weights[2]:.3f}")

        # Apply weights
        weighted_probs = (weights[0] * dc_probs +
                          weights[1] * lgbm_probs +
                          weights[2] * elo_probs)
        row_sums = weighted_probs.sum(axis=1, keepdims=True)
        weighted_probs /= row_sums

        # --- Method 2: Weighted + per-league calibration ---
        # Build calibration set from prior seasons
        cal_probs_list = []
        cal_outcomes_list = []
        cal_leagues_list = []

        for cs in prior_seasons[-2:]:  # Last 2 prior seasons
            cs_test = df[df["season"] == cs].copy()
            cs_train = df[df["season"] < cs].copy()
            cs_lgbm = train_lgbm_model(df, cs)
            if cs_lgbm is None:
                continue
            cs_dc, _ = predict_dc_for_df(cs_test, cs_train)
            cs_elo = predict_elo_for_df(cs_test, cs_train)
            cs_lgbm_probs = cs_lgbm.predict_1x2(cs_test)
            cs_blended = (weights[0] * cs_dc +
                          weights[1] * cs_lgbm_probs +
                          weights[2] * cs_elo)
            cs_blended /= cs_blended.sum(axis=1, keepdims=True)
            cs_outs = np.array([
                match_outcome(r["home_goals"], r["away_goals"])
                for _, r in cs_test.iterrows()
            ])
            cal_probs_list.append(cs_blended)
            cal_outcomes_list.append(cs_outs)
            cal_leagues_list.append(cs_test["league"].values)

        if cal_probs_list:
            cal_probs_arr = np.vstack(cal_probs_list)
            cal_outcomes_arr = np.concatenate(cal_outcomes_list)
            cal_leagues_arr = np.concatenate(cal_leagues_list)

            # Calibrate per league
            calibrated_probs = calibrate_per_league(
                weighted_probs, df_test["league"].values,
                cal_probs_arr, cal_outcomes_arr, cal_leagues_arr,
            )
        else:
            calibrated_probs = weighted_probs

        # --- Evaluate all methods ---
        for league in LEAGUES:
            mask = df_test["league"].values == league
            if mask.sum() == 0:
                continue

            w_m = evaluate_predictions(weighted_probs[mask], outcomes[mask])
            c_m = evaluate_predictions(calibrated_probs[mask], outcomes[mask])
            l_m = evaluate_predictions(lgbm_probs[mask], outcomes[mask])
            d_m = evaluate_predictions(dc_probs[mask], outcomes[mask])

            name = LEAGUE_NAMES.get(league, league)
            print(f"\n  {name} ({league}) — {mask.sum()} 场")
            print(f"    {'模型':<18} {'RPS':>8} {'Brier':>8} {'准确率':>8}")
            print(f"    {'Weighted+Cal':<18} {c_m['rps']:>8.4f} "
                  f"{c_m['brier']:>8.4f} {c_m['accuracy']:>7.1%}")
            print(f"    {'Weighted':<18} {w_m['rps']:>8.4f} "
                  f"{w_m['brier']:>8.4f} {w_m['accuracy']:>7.1%}")
            print(f"    {'LightGBM':<18} {l_m['rps']:>8.4f} "
                  f"{l_m['brier']:>8.4f} {l_m['accuracy']:>7.1%}")
            print(f"    {'Dixon-Coles':<18} {d_m['rps']:>8.4f} "
                  f"{d_m['brier']:>8.4f} {d_m['accuracy']:>7.1%}")

            all_results.append({
                "season": test_season, "league": league,
                "rps_weighted": w_m["rps"],
                "rps_cal": c_m["rps"],
                "rps_lgbm": l_m["rps"], "rps_dc": d_m["rps"],
                "acc_weighted": w_m["accuracy"],
                "acc_cal": c_m["accuracy"],
            })

    if all_results:
        rdf = pd.DataFrame(all_results)
        print(f"\n{'=' * 70}")
        print("汇总")
        print(f"{'=' * 70}")

        print(f"\n  {'联赛':<6} {'W+Cal':>10} {'Weighted':>10} {'LightGBM':>10} {'DC':>10}")
        for league in LEAGUES:
            lg = rdf[rdf["league"] == league]
            if len(lg) == 0:
                continue
            name = LEAGUE_NAMES.get(league, league)
            print(f"  {name:<6} {lg['rps_cal'].mean():>10.4f} "
                  f"{lg['rps_weighted'].mean():>10.4f} "
                  f"{lg['rps_lgbm'].mean():>10.4f} "
                  f"{lg['rps_dc'].mean():>10.4f}")

        print(f"\n  {'总计':<6} {rdf['rps_cal'].mean():>10.4f} "
              f"{rdf['rps_weighted'].mean():>10.4f} "
              f"{rdf['rps_lgbm'].mean():>10.4f} "
              f"{rdf['rps_dc'].mean():>10.4f}")

        d = rdf['rps_weighted'].mean() - rdf['rps_lgbm'].mean()
        d2 = rdf['rps_cal'].mean() - rdf['rps_lgbm'].mean()
        print(f"\n  Weighted vs LightGBM: {d:+.4f}")
        print(f"  W+Cal vs LightGBM:   {d2:+.4f}")


if __name__ == "__main__":
    run()
