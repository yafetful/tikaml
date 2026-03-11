"""Draw Specialist: dedicated draw prediction + ensemble.

Strategy:
1. Train a binary classifier: draw vs not-draw (LightGBM + draw-relevant features)
2. Use its P(draw) to adjust the Poisson model's 1x2 probabilities
3. Test different blending strategies

Also tests: confidence-based draw injection — when teams are evenly matched,
increase draw probability more aggressively.

Usage:
    source .venv/bin/activate
    python scripts/run_draw_specialist.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS
from src.evaluation import (evaluate_predictions, match_outcome,
                             ranked_probability_score)

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]

# Extra draw-relevant features on top of FEATURE_COLS
DRAW_EXTRA_FEATURES = [
    "draw_rate_home", "draw_rate_away", "league_draw_rate",
    "xg_diff_abs", "defensive_strength",
    "momentum_home", "momentum_away",
    "h2h_win_pct_home", "h2h_goal_diff_home",
    "points_diff", "position_diff",
    "xg_rolling_home", "xg_rolling_away",
    "xg_rolling_conceded_home", "xg_rolling_conceded_away",
    "days_rest_diff", "is_midweek", "season_stage",
]


class DrawClassifier:
    """Binary classifier: is this match a draw?"""

    def __init__(self):
        self.model = None
        self.feature_cols = None
        self._medians = None

    def _prepare(self, df):
        available = [c for c in FEATURE_COLS + DRAW_EXTRA_FEATURES
                     if c in df.columns]
        # Deduplicate
        available = list(dict.fromkeys(available))
        self.feature_cols = available
        X = df[available].copy()
        if self._medians is None:
            self._medians = X.median()
        return X.fillna(self._medians)

    def fit(self, train_df, val_df=None):
        X_train = self._prepare(train_df)
        y_train = (train_df["home_goals"] == train_df["away_goals"]).astype(int).values

        callbacks = [lgb.early_stopping(50, verbose=False)]

        if val_df is not None:
            X_val = self._prepare(val_df)
            y_val = (val_df["home_goals"] == val_df["away_goals"]).astype(int).values
        else:
            split = int(len(X_train) * 0.85)
            X_val, y_val = X_train.iloc[split:], y_train[split:]
            X_train, y_train = X_train.iloc[:split], y_train[:split]

        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=3.0,  # Upweight draws (minority class)
            verbose=-1,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       callbacks=callbacks)
        return self

    def predict_proba(self, df):
        """Return P(draw) for each match."""
        X = self._prepare(df)
        return self.model.predict_proba(X)[:, 1]  # P(class=1) = P(draw)


def blend_with_draw_prob(poisson_probs, draw_probs, alpha):
    """Blend Poisson 1x2 with draw specialist.

    Strategy: redistribute probability mass toward/from draw.
    new_P(draw) = alpha * draw_specialist + (1-alpha) * poisson_P(draw)
    Redistribute remaining mass proportionally to H/A.
    """
    n = len(poisson_probs)
    blended = np.zeros((n, 3))

    for i in range(n):
        p = poisson_probs[i].copy()
        d_new = alpha * draw_probs[i] + (1 - alpha) * p[1]

        # Redistribute
        remaining = 1.0 - d_new
        ha_total = p[0] + p[2]
        if ha_total > 0:
            blended[i, 0] = p[0] / ha_total * remaining
            blended[i, 1] = d_new
            blended[i, 2] = p[2] / ha_total * remaining
        else:
            blended[i] = [remaining / 2, d_new, remaining / 2]

    return blended


def confidence_draw_injection(poisson_probs, draw_probs, threshold, boost):
    """Inject draw probability when teams are evenly matched.

    When max(P(H), P(A)) < threshold (uncertain), boost P(draw) by `boost`.
    """
    n = len(poisson_probs)
    result = poisson_probs.copy()

    for i in range(n):
        max_ha = max(result[i, 0], result[i, 2])
        if max_ha < threshold:
            # Even match — boost draw
            draw_boost = boost * draw_probs[i]
            result[i, 1] += draw_boost
            # Reduce H/A proportionally
            reduction = draw_boost / 2
            result[i, 0] = max(0.05, result[i, 0] - reduction)
            result[i, 2] = max(0.05, result[i, 2] - reduction)
            result[i] /= result[i].sum()

    return result


def optimize_blend(poisson_probs, draw_probs, outcomes):
    """Find optimal blend alpha."""
    def obj(params):
        alpha = np.clip(params[0], 0, 1)
        blended = blend_with_draw_prob(poisson_probs, draw_probs, alpha)
        return np.mean([
            ranked_probability_score(blended[i], outcomes[i])
            for i in range(len(outcomes))
        ])

    best_alpha = 0.0
    best_rps = float("inf")
    for a0 in np.arange(0, 0.6, 0.05):
        result = minimize(obj, [a0], method="Nelder-Mead",
                          options={"maxiter": 100})
        if result.fun < best_rps:
            best_rps = result.fun
            best_alpha = np.clip(result.x[0], 0, 1)
    return best_alpha, best_rps


def optimize_injection(poisson_probs, draw_probs, outcomes):
    """Find optimal threshold and boost for confidence-based injection."""
    def obj(params):
        threshold = np.clip(params[0], 0.3, 0.55)
        boost = np.clip(params[1], 0, 0.5)
        result = confidence_draw_injection(
            poisson_probs, draw_probs, threshold, boost)
        return np.mean([
            ranked_probability_score(result[i], outcomes[i])
            for i in range(len(outcomes))
        ])

    best = None
    best_rps = float("inf")
    for t0, b0 in [(0.42, 0.1), (0.45, 0.15), (0.40, 0.05), (0.48, 0.2)]:
        result = minimize(obj, [t0, b0], method="Nelder-Mead",
                          options={"maxiter": 200})
        if result.fun < best_rps:
            best_rps = result.fun
            best = result.x
    return np.clip(best[0], 0.3, 0.55), np.clip(best[1], 0, 0.5), best_rps


def train_models(df, test_season):
    all_train = df[
        (df["season"] < test_season) &
        (df["season"] >= "2016-2017")
    ].copy()
    if len(all_train) < 500:
        return None, None
    seasons = sorted(all_train["season"].unique())
    prev = seasons[-1]
    val = all_train[all_train["season"] == prev]
    train = all_train[all_train["season"] < prev]

    lgbm = LGBMPoissonModel(rho=-0.108)
    lgbm.fit(train, val_df=val)

    draw_clf = DrawClassifier()
    draw_clf.fit(train, val_df=val)

    return lgbm, draw_clf


def run():
    print("=" * 70)
    print("TikaML: 平局专家模型 + 融合策略")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  数据: {len(df)} 场")

    results = {
        "baseline": [],
        "draw_blend": [],
        "draw_inject": [],
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

        lgbm, draw_clf = train_models(df, test_season)
        if lgbm is None:
            continue

        base_probs = lgbm.predict_1x2(test)
        draw_probs = draw_clf.predict_proba(test)

        base_m = evaluate_predictions(base_probs, outcomes)
        results["baseline"].append(base_m["rps"])

        # Draw classifier analysis
        actual_draws = (outcomes == 1).sum()
        draw_pred_binary = (draw_probs > 0.5).sum()
        avg_draw_prob = draw_probs.mean()
        avg_draw_prob_actual = draw_probs[outcomes == 1].mean()
        avg_draw_prob_notdraw = draw_probs[outcomes != 1].mean()

        print(f"  基线: RPS={base_m['rps']:.4f}  Acc={base_m['accuracy']:.1%}")
        print(f"  平局分类器: P(draw) 均值={avg_draw_prob:.3f}")
        print(f"    实际平局时: P(draw)={avg_draw_prob_actual:.3f}")
        print(f"    非平局时: P(draw)={avg_draw_prob_notdraw:.3f}")
        print(f"    二分类预测平局数={draw_pred_binary} (实际={actual_draws})")

        # Optimize on prior season
        prior_seasons = sorted(
            s for s in df["season"].unique()
            if "2018-2019" <= s < test_season
        )
        if not prior_seasons:
            continue
        opt_s = prior_seasons[-1]
        opt_test = df[df["season"] == opt_s].copy()
        opt_lgbm, opt_draw = train_models(df, opt_s)
        if opt_lgbm is None:
            continue
        opt_probs = opt_lgbm.predict_1x2(opt_test)
        opt_dprobs = opt_draw.predict_proba(opt_test)
        opt_outs = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in opt_test.iterrows()
        ])

        # Strategy 1: Draw Blend
        alpha, _ = optimize_blend(opt_probs, opt_dprobs, opt_outs)
        blend_probs = blend_with_draw_prob(base_probs, draw_probs, alpha)
        blend_m = evaluate_predictions(blend_probs, outcomes)
        results["draw_blend"].append(blend_m["rps"])
        blend_draws = (np.argmax(blend_probs, axis=1) == 1).sum()
        print(f"\n  Draw Blend (α={alpha:.3f}):")
        print(f"    RPS={blend_m['rps']:.4f}  Acc={blend_m['accuracy']:.1%}  "
              f"({blend_m['rps'] - base_m['rps']:+.4f})")
        print(f"    预测平局数={blend_draws}")

        # Strategy 2: Confidence-based Injection
        threshold, boost, _ = optimize_injection(
            opt_probs, opt_dprobs, opt_outs)
        inject_probs = confidence_draw_injection(
            base_probs, draw_probs, threshold, boost)
        inject_m = evaluate_predictions(inject_probs, outcomes)
        results["draw_inject"].append(inject_m["rps"])
        inject_draws = (np.argmax(inject_probs, axis=1) == 1).sum()
        print(f"\n  Confidence Injection (threshold={threshold:.2f}, boost={boost:.3f}):")
        print(f"    RPS={inject_m['rps']:.4f}  Acc={inject_m['accuracy']:.1%}  "
              f"({inject_m['rps'] - base_m['rps']:+.4f})")
        print(f"    预测平局数={inject_draws}")

        # Draw feature importance
        if hasattr(draw_clf.model, 'feature_importances_'):
            fi = pd.DataFrame({
                'feature': draw_clf.feature_cols,
                'importance': draw_clf.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\n  平局分类器 Top-5 特征:")
            for _, row in fi.head(5).iterrows():
                print(f"    {row['feature']:<35} {row['importance']:>5}")

    # Summary
    print(f"\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")

    print(f"\n  {'方法':<24} {'2022-23':>10} {'2023-24':>10} {'2024-25':>10} {'平均':>10}")
    for key, name in [
        ("baseline", "LGBM Baseline"),
        ("draw_blend", "Draw Blend"),
        ("draw_inject", "Confidence Inject"),
    ]:
        vals = results[key]
        if len(vals) < 3:
            continue
        avg = np.mean(vals)
        d = avg - np.mean(results["baseline"])
        print(f"  {name:<24} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} "
              f"{avg:>10.4f} ({d:+.4f})")


if __name__ == "__main__":
    run()
