"""Bivariate Poisson experiment: test whether λ₃ improves draw prediction.

Standard Poisson: home ~ Poisson(λ_h), away ~ Poisson(λ_a) — independent
Bivariate Poisson: home = X₁+X₃, away = X₂+X₃ — correlated via λ₃

λ₃ creates positive goal correlation → boosts draw probabilities (1-1, 2-2 etc.)

Tests:
  1. Fixed λ₃ values (0.05, 0.10, 0.15, 0.20, 0.25)
  2. Optimized λ₃ on prior season
  3. Per-match λ₃ via third LightGBM model

Usage:
    source .venv/bin/activate
    python scripts/run_bivariate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS, LGB_PARAMS
from src.bivariate_poisson import bivariate_score_matrix, matrix_to_1x2
from src.evaluation import (evaluate_predictions, match_outcome,
                             ranked_probability_score)

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}


def train_lgbm(df, test_season):
    """Train standard LightGBM Poisson model."""
    all_train = df[
        (df["season"] < test_season) &
        (df["season"] >= "2016-2017")
    ].copy()
    if len(all_train) < 500:
        return None
    seasons = sorted(all_train["season"].unique())
    if len(seasons) < 2:
        return None
    prev = seasons[-1]
    val = all_train[all_train["season"] == prev]
    train = all_train[all_train["season"] < prev]
    lgbm = LGBMPoissonModel(rho=-0.108)
    lgbm.fit(train, val_df=val)
    return lgbm


def predict_bivariate_1x2(lh_arr, la_arr, lambda3, rho=-0.108):
    """Predict 1x2 probabilities using bivariate Poisson."""
    probs = []
    for lh, la in zip(lh_arr, la_arr):
        matrix = bivariate_score_matrix(lh, la, lambda3, max_goals=7, rho=rho)
        p = matrix_to_1x2(matrix)
        probs.append(p)
    return np.array(probs)


def predict_bivariate_1x2_perMatch(lh_arr, la_arr, l3_arr, rho=-0.108):
    """Predict 1x2 with per-match λ₃."""
    probs = []
    for lh, la, l3 in zip(lh_arr, la_arr, l3_arr):
        matrix = bivariate_score_matrix(lh, la, l3, max_goals=7, rho=rho)
        p = matrix_to_1x2(matrix)
        probs.append(p)
    return np.array(probs)


def optimize_lambda3(lh_arr, la_arr, outcomes, rho=-0.108):
    """Find optimal fixed λ₃ on validation data."""
    def objective(l3):
        probs = predict_bivariate_1x2(lh_arr, la_arr, l3, rho=rho)
        return np.mean([
            ranked_probability_score(probs[i], outcomes[i])
            for i in range(len(outcomes))
        ])

    result = minimize_scalar(objective, bounds=(0.0, 0.4), method="bounded",
                             options={"maxiter": 50})
    return result.x, result.fun


def train_lambda3_model(df, test_season):
    """Train a LightGBM model to predict per-match λ₃.

    Target: min(home_goals, away_goals) as proxy for common scoring component.
    """
    all_train = df[
        (df["season"] < test_season) &
        (df["season"] >= "2016-2017")
    ].copy()
    if len(all_train) < 500:
        return None, None, None

    seasons = sorted(all_train["season"].unique())
    if len(seasons) < 2:
        return None, None, None
    prev = seasons[-1]
    val = all_train[all_train["season"] == prev]
    train = all_train[all_train["season"] < prev]

    # Target: min(home, away) as proxy for shared scoring
    y_train = np.minimum(
        train["home_goals"].values, train["away_goals"].values
    ).astype(float)
    y_val = np.minimum(
        val["home_goals"].values, val["away_goals"].values
    ).astype(float)

    available = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[available].copy()
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = val[available].copy().fillna(medians)

    params = LGB_PARAMS.copy()
    params["objective"] = "poisson"
    params["n_estimators"] = 500
    params["learning_rate"] = 0.05

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    return model, available, medians


def run():
    print("=" * 70)
    print("TikaML: 双变量泊松分布实验")
    print("  标准泊松 vs 双变量泊松 (λ₃ 共同进球因子)")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  数据: {len(df)} 场")

    fixed_l3_values = [0.05, 0.10, 0.15, 0.20, 0.25]

    # Collect results
    results = {
        "standard": [],
        "optimized_l3": [],
        "perMatch_l3": [],
    }
    for l3 in fixed_l3_values:
        results[f"fixed_{l3:.2f}"] = []

    opt_l3_values = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        df_test = df[df["season"] == test_season].copy()
        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in df_test.iterrows()
        ])

        # Train LightGBM
        lgbm = train_lgbm(df, test_season)
        if lgbm is None:
            print("  训练数据不足，跳过")
            continue

        lh, la = lgbm.predict_lambdas(df_test)

        # 1. Standard Poisson (baseline)
        std_probs = lgbm.predict_1x2(df_test)
        std_m = evaluate_predictions(std_probs, outcomes)
        results["standard"].append(std_m["rps"])

        # 2. Fixed λ₃ values
        for l3 in fixed_l3_values:
            bp_probs = predict_bivariate_1x2(lh, la, l3, rho=-0.108)
            bp_m = evaluate_predictions(bp_probs, outcomes)
            results[f"fixed_{l3:.2f}"].append(bp_m["rps"])

        # 3. Optimized λ₃ on prior season
        prior_seasons = sorted(
            s for s in df["season"].unique()
            if "2018-2019" <= s < test_season
        )
        if len(prior_seasons) >= 1:
            opt_s = prior_seasons[-1]
            opt_test = df[df["season"] == opt_s].copy()
            opt_lgbm = train_lgbm(df, opt_s)
            if opt_lgbm is not None:
                opt_lh, opt_la = opt_lgbm.predict_lambdas(opt_test)
                opt_outs = np.array([
                    match_outcome(r["home_goals"], r["away_goals"])
                    for _, r in opt_test.iterrows()
                ])
                best_l3, best_rps = optimize_lambda3(
                    opt_lh, opt_la, opt_outs, rho=-0.108)
                print(f"  优化 λ₃ = {best_l3:.4f} (验证集 RPS={best_rps:.4f})")
                opt_l3_values.append(best_l3)
            else:
                best_l3 = 0.10
                opt_l3_values.append(best_l3)
        else:
            best_l3 = 0.10
            opt_l3_values.append(best_l3)

        opt_probs = predict_bivariate_1x2(lh, la, best_l3, rho=-0.108)
        opt_m = evaluate_predictions(opt_probs, outcomes)
        results["optimized_l3"].append(opt_m["rps"])

        # 4. Per-match λ₃ model
        l3_model, l3_cols, l3_medians = train_lambda3_model(df, test_season)
        if l3_model is not None:
            X_test_l3 = df_test[l3_cols].copy().fillna(l3_medians)
            l3_pred = np.clip(l3_model.predict(X_test_l3), 0.01, 0.5)
            pm_probs = predict_bivariate_1x2_perMatch(lh, la, l3_pred, rho=-0.108)
            pm_m = evaluate_predictions(pm_probs, outcomes)
            results["perMatch_l3"].append(pm_m["rps"])
            print(f"  Per-match λ₃: mean={l3_pred.mean():.3f}, "
                  f"std={l3_pred.std():.3f}")
        else:
            results["perMatch_l3"].append(None)

        # Print season results
        print(f"\n  {'模型':<22} {'RPS':>8} {'差值':>8}")
        print(f"  {'─' * 40}")
        print(f"  {'Standard Poisson':<22} {std_m['rps']:>8.4f} {'(基线)':>8}")
        for l3 in fixed_l3_values:
            key = f"fixed_{l3:.2f}"
            rps = results[key][-1]
            d = rps - std_m["rps"]
            label = f"Bivariate λ₃={l3}"
            print(f"  {label:<22} {rps:>8.4f} {d:>+8.4f}")
        d_opt = opt_m["rps"] - std_m["rps"]
        print(f"  {'Bivariate (optimized)':<22} {opt_m['rps']:>8.4f} {d_opt:>+8.4f}")
        if results["perMatch_l3"][-1] is not None:
            d_pm = pm_m["rps"] - std_m["rps"]
            print(f"  {'Bivariate (per-match)':<22} {pm_m['rps']:>8.4f} {d_pm:>+8.4f}")

        # Draw analysis
        n_draws = (outcomes == 1).sum()
        n_total = len(outcomes)
        print(f"\n  平局分析: 实际平局 {n_draws}/{n_total} ({n_draws/n_total:.1%})")

        std_draw_p = std_probs[:, 1].mean()
        opt_draw_p = opt_probs[:, 1].mean()
        print(f"  标准模型平均 P(draw) = {std_draw_p:.3f}")
        print(f"  双变量模型平均 P(draw) = {opt_draw_p:.3f}")

        # Draw accuracy
        std_draw_pred = (np.argmax(std_probs, axis=1) == 1).sum()
        opt_draw_pred = (np.argmax(opt_probs, axis=1) == 1).sum()
        print(f"  标准模型预测平局数 = {std_draw_pred}")
        print(f"  双变量模型预测平局数 = {opt_draw_pred}")

    # Summary
    print(f"\n{'=' * 70}")
    print("汇总: 三赛季平均 RPS")
    print(f"{'=' * 70}")

    print(f"\n  {'模型':<24} {'2022-23':>10} {'2023-24':>10} {'2024-25':>10} {'平均':>10}")
    print(f"  {'─' * 64}")

    def _fmt_row(label, vals):
        parts = [f"{v:.4f}" if v is not None else "  N/A " for v in vals]
        valid = [v for v in vals if v is not None]
        avg = np.mean(valid) if valid else 0
        print(f"  {label:<24} {parts[0]:>10} {parts[1]:>10} {parts[2]:>10} {avg:>10.4f}")

    _fmt_row("Standard Poisson", results["standard"])
    for l3 in fixed_l3_values:
        _fmt_row(f"Bivariate λ₃={l3}", results[f"fixed_{l3:.2f}"])
    _fmt_row("Bivariate (optimized)", results["optimized_l3"])
    _fmt_row("Bivariate (per-match)", results["perMatch_l3"])

    # Best vs baseline
    if results["standard"] and results["optimized_l3"]:
        std_avg = np.mean(results["standard"])
        opt_avg = np.mean([v for v in results["optimized_l3"] if v is not None])
        pm_vals = [v for v in results["perMatch_l3"] if v is not None]
        pm_avg = np.mean(pm_vals) if pm_vals else None

        print(f"\n  标准泊松 vs 优化双变量: {opt_avg - std_avg:+.4f}")
        if pm_avg:
            print(f"  标准泊松 vs Per-match: {pm_avg - std_avg:+.4f}")

    if opt_l3_values:
        print(f"\n  各赛季优化 λ₃: {[f'{v:.4f}' for v in opt_l3_values]}")


if __name__ == "__main__":
    run()
