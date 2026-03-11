"""Score matrix calibration: learn systematic corrections to the 7x7 matrix.

The Poisson model has known biases:
- Underestimates 0-0 and 1-1 draws
- Overestimates some high-score outcomes
- Fixed ρ doesn't adapt to match context

This experiment:
1. Learns per-cell multiplicative corrections to the score matrix
2. Learns context-dependent corrections (e.g., derby → boost draws)
3. Tests "draw boost" — a simple multiplier on diagonal cells

Usage:
    source .venv/bin/activate
    python scripts/run_matrix_calibration.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS
from src.evaluation import (evaluate_predictions, match_outcome,
                             ranked_probability_score)

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
MAX_GOALS = 7


def _tau(i, j, lh, la, rho):
    if i == 0 and j == 0:
        return max(0, 1 - lh * la * rho)
    elif i == 0 and j == 1:
        return max(0, 1 + lh * rho)
    elif i == 1 and j == 0:
        return max(0, 1 + la * rho)
    elif i == 1 and j == 1:
        return max(0, 1 - rho)
    return 1.0


def build_matrices_batch(lh_arr, la_arr, rho=-0.108):
    """Vectorized: build score matrices for all matches at once."""
    n = len(lh_arr)
    goals = np.arange(MAX_GOALS)

    # Poisson PMFs: (n, MAX_GOALS)
    pmf_h = poisson.pmf(goals[None, :], lh_arr[:, None])
    pmf_a = poisson.pmf(goals[None, :], la_arr[:, None])

    # Outer product: (n, MAX_GOALS, MAX_GOALS)
    matrices = pmf_h[:, :, None] * pmf_a[:, None, :]

    # DC tau correction
    matrices[:, 0, 0] *= np.maximum(0, 1 - lh_arr * la_arr * rho)
    matrices[:, 0, 1] *= np.maximum(0, 1 + lh_arr * rho)
    matrices[:, 1, 0] *= np.maximum(0, 1 + la_arr * rho)
    matrices[:, 1, 1] *= np.maximum(0, 1 - rho)

    # Normalize
    totals = matrices.sum(axis=(1, 2), keepdims=True)
    matrices = np.where(totals > 0, matrices / totals, matrices)
    return matrices


def matrices_to_1x2_batch(matrices):
    """Vectorized: convert score matrices to 1x2 probabilities."""
    n = len(matrices)
    probs = np.zeros((n, 3))
    for b in range(n):
        probs[b, 0] = np.tril(matrices[b], -1).sum()
        probs[b, 1] = np.trace(matrices[b])
        probs[b, 2] = np.triu(matrices[b], 1).sum()
    totals = probs.sum(axis=1, keepdims=True)
    return np.where(totals > 0, probs / totals, probs)


def rps_batch(probs, outcomes):
    """Compute mean RPS for a batch."""
    n = len(outcomes)
    rps_sum = 0.0
    for i in range(n):
        cum_pred = np.cumsum(probs[i])
        cum_act = np.cumsum([1.0 if j == outcomes[i] else 0.0 for j in range(3)])
        rps_sum += np.sum((cum_pred[:2] - cum_act[:2]) ** 2) / 2
    return rps_sum / n


def train_lgbm(df, test_season):
    all_train = df[
        (df["season"] < test_season) &
        (df["season"] >= "2016-2017")
    ].copy()
    if len(all_train) < 500:
        return None
    seasons = sorted(all_train["season"].unique())
    prev = seasons[-1]
    val = all_train[all_train["season"] == prev]
    train = all_train[all_train["season"] < prev]
    lgbm = LGBMPoissonModel(rho=-0.108)
    lgbm.fit(train, val_df=val)
    return lgbm


# --- Experiment 1: Draw Boost ---
def predict_with_draw_boost(lh_arr, la_arr, rho, draw_boost):
    """Apply multiplicative boost to draw scores (diagonal)."""
    matrices = build_matrices_batch(lh_arr, la_arr, rho)
    for k in range(MAX_GOALS):
        matrices[:, k, k] *= draw_boost
    totals = matrices.sum(axis=(1, 2), keepdims=True)
    matrices = np.where(totals > 0, matrices / totals, matrices)
    return matrices_to_1x2_batch(matrices)


def optimize_draw_boost(lh_arr, la_arr, outcomes, rho=-0.108):
    def obj(params):
        probs = predict_with_draw_boost(lh_arr, la_arr, rho, params[0])
        return rps_batch(probs, outcomes)
    result = minimize(obj, [1.05], method="Nelder-Mead",
                      options={"maxiter": 200})
    return result.x[0], result.fun


# --- Experiment 2: Score-specific corrections ---
KEY_CELLS = [
    (0, 0), (1, 1), (2, 2),  # draws
    (1, 0), (0, 1),           # 1-0, 0-1
    (2, 1), (1, 2),           # 2-1, 1-2
    (2, 0), (0, 2),           # 2-0, 0-2
    (3, 1), (1, 3),           # 3-1, 1-3
]


def predict_with_corrections(lh_arr, la_arr, corrections, rho=-0.108):
    matrices = build_matrices_batch(lh_arr, la_arr, rho)
    for (i, j), mult in corrections.items():
        matrices[:, i, j] *= mult
    totals = matrices.sum(axis=(1, 2), keepdims=True)
    matrices = np.where(totals > 0, matrices / totals, matrices)
    return matrices_to_1x2_batch(matrices)


def optimize_score_corrections(lh_arr, la_arr, outcomes, rho=-0.108):
    def obj(params):
        corrections = {cell: np.clip(params[i], 0.5, 2.0)
                       for i, cell in enumerate(KEY_CELLS)}
        probs = predict_with_corrections(lh_arr, la_arr, corrections, rho)
        return rps_batch(probs, outcomes)
    x0 = np.ones(len(KEY_CELLS))
    result = minimize(obj, x0, method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 1e-4})
    corrections = {cell: np.clip(result.x[i], 0.5, 2.0)
                   for i, cell in enumerate(KEY_CELLS)}
    return corrections, result.fun


# --- Experiment 3: Context-dependent ρ ---
def predict_with_adaptive_rho(lh_arr, la_arr, features_df, params):
    base_rho, xg_diff_coeff, draw_rate_coeff = params
    xg_diff = features_df.get("xg_diff_abs", pd.Series(0, index=features_df.index)).fillna(0).values
    draw_rate = features_df.get("draw_rate_home", pd.Series(0.25, index=features_df.index)).fillna(0.25).values
    rho_arr = base_rho + xg_diff_coeff * xg_diff + draw_rate_coeff * (draw_rate - 0.25)
    rho_arr = np.clip(rho_arr, -0.3, 0.1)

    # Build matrices with per-match rho
    goals = np.arange(MAX_GOALS)
    pmf_h = poisson.pmf(goals[None, :], lh_arr[:, None])
    pmf_a = poisson.pmf(goals[None, :], la_arr[:, None])
    matrices = pmf_h[:, :, None] * pmf_a[:, None, :]
    matrices[:, 0, 0] *= np.maximum(0, 1 - lh_arr * la_arr * rho_arr)
    matrices[:, 0, 1] *= np.maximum(0, 1 + lh_arr * rho_arr)
    matrices[:, 1, 0] *= np.maximum(0, 1 + la_arr * rho_arr)
    matrices[:, 1, 1] *= np.maximum(0, 1 - rho_arr)
    totals = matrices.sum(axis=(1, 2), keepdims=True)
    matrices = np.where(totals > 0, matrices / totals, matrices)
    return matrices_to_1x2_batch(matrices)


def optimize_adaptive_rho(lh_arr, la_arr, features_df, outcomes):
    def obj(params):
        probs = predict_with_adaptive_rho(lh_arr, la_arr, features_df, params)
        return rps_batch(probs, outcomes)
    best, best_rps = None, float("inf")
    for x0 in [[-0.108, 0, 0], [-0.10, -0.02, 0.1], [-0.12, 0.01, -0.1]]:
        result = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 300})
        if result.fun < best_rps:
            best_rps = result.fun
            best = result.x
    return best, best_rps


# --- Experiment 4: 1x2 Adjustment ---
def predict_with_1x2_adjustment(base_probs, adj_params):
    adjusted = base_probs.copy()
    adjusted[:, 0] *= adj_params[0]
    adjusted[:, 1] *= adj_params[1]
    adjusted[:, 2] *= adj_params[2]
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def optimize_1x2_adjustment(base_probs, outcomes):
    def obj(params):
        params = np.abs(params)
        adjusted = predict_with_1x2_adjustment(base_probs, params)
        return rps_batch(adjusted, outcomes)
    result = minimize(obj, [1.0, 1.0, 1.0], method="Nelder-Mead",
                      options={"maxiter": 300})
    return np.abs(result.x), result.fun


def run():
    print("=" * 70)
    print("TikaML: Score Matrix 校准实验")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  数据: {len(df)} 场")

    results = {
        "baseline": [],
        "draw_boost": [],
        "score_corrections": [],
        "adaptive_rho": [],
        "1x2_adjustment": [],
    }

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        test = df[df["season"] == test_season].copy()
        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in test.iterrows()
        ])

        lgbm = train_lgbm(df, test_season)
        if lgbm is None:
            continue

        lh, la = lgbm.predict_lambdas(test)
        base_probs = lgbm.predict_1x2(test)
        base_m = evaluate_predictions(base_probs, outcomes)
        results["baseline"].append(base_m["rps"])
        print(f"  基线: RPS={base_m['rps']:.4f}  Acc={base_m['accuracy']:.1%}")

        # Optimization season
        prior_seasons = sorted(
            s for s in df["season"].unique()
            if "2018-2019" <= s < test_season
        )
        if not prior_seasons:
            continue
        opt_s = prior_seasons[-1]
        opt_test = df[df["season"] == opt_s].copy()
        opt_lgbm = train_lgbm(df, opt_s)
        if opt_lgbm is None:
            continue
        opt_lh, opt_la = opt_lgbm.predict_lambdas(opt_test)
        opt_probs = opt_lgbm.predict_1x2(opt_test)
        opt_outs = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in opt_test.iterrows()
        ])

        # Exp 1: Draw Boost
        best_boost, _ = optimize_draw_boost(opt_lh, opt_la, opt_outs)
        print(f"  Draw Boost: {best_boost:.3f}")
        db_probs = predict_with_draw_boost(lh, la, -0.108, best_boost)
        db_m = evaluate_predictions(db_probs, outcomes)
        results["draw_boost"].append(db_m["rps"])
        print(f"    RPS={db_m['rps']:.4f}  Acc={db_m['accuracy']:.1%}  "
              f"({db_m['rps'] - base_m['rps']:+.4f})")

        # Exp 2: Score Corrections
        corrections, _ = optimize_score_corrections(opt_lh, opt_la, opt_outs)
        print(f"  Score Corrections: ", end="")
        for cell, mult in sorted(corrections.items()):
            if abs(mult - 1.0) > 0.01:
                print(f"{cell[0]}-{cell[1]}:{mult:.2f} ", end="")
        print()
        sc_probs = predict_with_corrections(lh, la, corrections)
        sc_m = evaluate_predictions(sc_probs, outcomes)
        results["score_corrections"].append(sc_m["rps"])
        print(f"    RPS={sc_m['rps']:.4f}  Acc={sc_m['accuracy']:.1%}  "
              f"({sc_m['rps'] - base_m['rps']:+.4f})")

        # Exp 3: Adaptive ρ
        ar_params, _ = optimize_adaptive_rho(
            opt_lh, opt_la, opt_test, opt_outs)
        print(f"  Adaptive ρ: base={ar_params[0]:.3f}, "
              f"xg_diff={ar_params[1]:.3f}, draw_rate={ar_params[2]:.3f}")
        ar_probs = predict_with_adaptive_rho(lh, la, test, ar_params)
        ar_m = evaluate_predictions(ar_probs, outcomes)
        results["adaptive_rho"].append(ar_m["rps"])
        print(f"    RPS={ar_m['rps']:.4f}  Acc={ar_m['accuracy']:.1%}  "
              f"({ar_m['rps'] - base_m['rps']:+.4f})")

        # Exp 4: 1x2 Adjustment
        adj_params, _ = optimize_1x2_adjustment(opt_probs, opt_outs)
        print(f"  1x2 Adjustment: H={adj_params[0]:.3f}, "
              f"D={adj_params[1]:.3f}, A={adj_params[2]:.3f}")
        adj_probs = predict_with_1x2_adjustment(base_probs, adj_params)
        adj_m = evaluate_predictions(adj_probs, outcomes)
        results["1x2_adjustment"].append(adj_m["rps"])
        print(f"    RPS={adj_m['rps']:.4f}  Acc={adj_m['accuracy']:.1%}  "
              f"({adj_m['rps'] - base_m['rps']:+.4f})")

        # Draw prediction counts
        actual_draws = (outcomes == 1).sum()
        db_draws = (np.argmax(db_probs, axis=1) == 1).sum()
        adj_draws = (np.argmax(adj_probs, axis=1) == 1).sum()
        print(f"\n  平局: 实际={actual_draws}, DrawBoost预测={db_draws}, "
              f"1x2Adj预测={adj_draws}")

    # Summary
    print(f"\n{'=' * 70}")
    print("汇总: 三赛季平均 RPS")
    print(f"{'=' * 70}")

    print(f"\n  {'方法':<22} {'2022-23':>10} {'2023-24':>10} {'2024-25':>10} {'平均':>10}")
    for method, name in [
        ("baseline", "LGBM Baseline"),
        ("draw_boost", "Draw Boost"),
        ("score_corrections", "Score Corrections"),
        ("adaptive_rho", "Adaptive ρ"),
        ("1x2_adjustment", "1x2 Adjustment"),
    ]:
        vals = results[method]
        if len(vals) < 3:
            continue
        avg = np.mean(vals)
        print(f"  {name:<22} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {avg:>10.4f}")

    # Best vs baseline
    base_avg = np.mean(results["baseline"])
    for method, name in [
        ("draw_boost", "Draw Boost"),
        ("score_corrections", "Score Corrections"),
        ("adaptive_rho", "Adaptive ρ"),
        ("1x2_adjustment", "1x2 Adjustment"),
    ]:
        if len(results[method]) >= 3:
            avg = np.mean(results[method])
            print(f"  {name} vs Baseline: {avg - base_avg:+.4f}")


if __name__ == "__main__":
    run()
