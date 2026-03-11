"""Hybrid model: Poisson LightGBM + Multiclass LightGBM ensemble.

The Poisson model is great at predicting scores when there's a clear favorite.
The multiclass model may better capture draw patterns since it optimizes
directly for outcome classification.

Usage:
    source .venv/bin/activate
    python scripts/run_hybrid.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS
from src.evaluation import evaluate_predictions, match_outcome

FEATURES_PATH = Path("data/opta/processed/features.csv")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}


class MulticlassPredictor:
    """Direct 1x2 multiclass classification via LightGBM."""

    def __init__(self):
        self.model = None
        self.feature_cols = None
        self._medians = None

    def _prepare(self, df):
        available = [c for c in FEATURE_COLS if c in df.columns]
        self.feature_cols = available
        X = df[available].copy()
        if self._medians is None:
            self._medians = X.median()
        X = X.fillna(self._medians)
        return X

    def fit(self, train_df, val_df=None):
        X_train = self._prepare(train_df)
        y_train = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in train_df.iterrows()
        ])

        callbacks = [lgb.early_stopping(50, verbose=False)]

        if val_df is not None:
            X_val = self._prepare(val_df)
            y_val = np.array([
                match_outcome(r["home_goals"], r["away_goals"])
                for _, r in val_df.iterrows()
            ])
        else:
            split = int(len(X_train) * 0.85)
            X_val = X_train.iloc[split:]
            y_val = y_train[split:]
            X_train = X_train.iloc[:split]
            y_train = y_train[:split]

        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=21,
            min_child_samples=40,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbose=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, df):
        X = self._prepare(df)
        return self.model.predict_proba(X)


def train_models(df, test_season):
    all_train = df[
        (df["season"] < test_season) &
        (df["season"] >= "2016-2017")
    ].copy()
    if len(all_train) < 500:
        return None, None

    seasons = sorted(all_train["season"].unique())
    if len(seasons) < 2:
        return None, None
    prev_season = seasons[-1]
    val_data = all_train[all_train["season"] == prev_season]
    train_data = all_train[all_train["season"] < prev_season]

    # Train Poisson model
    poisson_model = LGBMPoissonModel(rho=-0.108)
    poisson_model.fit(train_data, val_df=val_data)

    # Train Multiclass model
    multi_model = MulticlassPredictor()
    multi_model.fit(train_data, val_df=val_data)

    return poisson_model, multi_model


def optimize_blend_weight(poisson_probs, multi_probs, outcomes):
    """Find optimal alpha: blended = alpha * poisson + (1-alpha) * multi."""
    from src.evaluation import ranked_probability_score

    best_alpha = 0.5
    best_rps = float("inf")

    for alpha in np.arange(0.0, 1.01, 0.05):
        blended = alpha * poisson_probs + (1 - alpha) * multi_probs
        row_sums = blended.sum(axis=1, keepdims=True)
        blended = blended / row_sums
        rps = np.mean([
            ranked_probability_score(blended[i], outcomes[i])
            for i in range(len(outcomes))
        ])
        if rps < best_rps:
            best_rps = rps
            best_alpha = alpha

    return best_alpha


def run():
    print("=" * 70)
    print("TikaML: Poisson + Multiclass 混合模型")
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

        poisson_model, multi_model = train_models(df, test_season)
        if poisson_model is None:
            continue

        poisson_probs = poisson_model.predict_1x2(df_test)
        multi_probs = multi_model.predict_proba(df_test)

        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in df_test.iterrows()
        ])

        # Optimize blend weight on prior season
        prior_seasons = sorted(
            s for s in df["season"].unique()
            if "2018-2019" <= s < test_season
        )
        if len(prior_seasons) >= 1:
            opt_season = prior_seasons[-1]
            opt_test = df[df["season"] == opt_season].copy()
            opt_p, opt_m = train_models(df, opt_season)
            if opt_p is not None:
                opt_pp = opt_p.predict_1x2(opt_test)
                opt_mp = opt_m.predict_proba(opt_test)
                opt_outs = np.array([
                    match_outcome(r["home_goals"], r["away_goals"])
                    for _, r in opt_test.iterrows()
                ])
                alpha = optimize_blend_weight(opt_pp, opt_mp, opt_outs)
            else:
                alpha = 0.5
        else:
            alpha = 0.5

        print(f"  最优混合权重: α={alpha:.2f} (Poisson={alpha:.0%}, Multi={1-alpha:.0%})")

        # Blend
        blended = alpha * poisson_probs + (1 - alpha) * multi_probs
        blended /= blended.sum(axis=1, keepdims=True)

        for league in LEAGUES:
            mask = df_test["league"].values == league
            if mask.sum() == 0:
                continue

            p_m = evaluate_predictions(poisson_probs[mask], outcomes[mask])
            m_m = evaluate_predictions(multi_probs[mask], outcomes[mask])
            b_m = evaluate_predictions(blended[mask], outcomes[mask])

            name = LEAGUE_NAMES.get(league, league)
            print(f"\n  {name} ({league}) — {mask.sum()} 场")
            print(f"    {'模型':<18} {'RPS':>8} {'Brier':>8} {'准确率':>8}")
            print(f"    {'Hybrid':<18} {b_m['rps']:>8.4f} "
                  f"{b_m['brier']:>8.4f} {b_m['accuracy']:>7.1%}")
            print(f"    {'Poisson':<18} {p_m['rps']:>8.4f} "
                  f"{p_m['brier']:>8.4f} {p_m['accuracy']:>7.1%}")
            print(f"    {'Multiclass':<18} {m_m['rps']:>8.4f} "
                  f"{m_m['brier']:>8.4f} {m_m['accuracy']:>7.1%}")

            # Check draw prediction
            n_draws_actual = (outcomes[mask] == 1).sum()
            n_draws_poisson = (np.argmax(poisson_probs[mask], axis=1) == 1).sum()
            n_draws_multi = (np.argmax(multi_probs[mask], axis=1) == 1).sum()
            n_draws_hybrid = (np.argmax(blended[mask], axis=1) == 1).sum()
            print(f"    实际平局={n_draws_actual}, "
                  f"Poisson预测={n_draws_poisson}, "
                  f"Multi预测={n_draws_multi}, "
                  f"Hybrid预测={n_draws_hybrid}")

            all_results.append({
                "season": test_season, "league": league,
                "rps_hybrid": b_m["rps"],
                "rps_poisson": p_m["rps"],
                "rps_multi": m_m["rps"],
                "acc_hybrid": b_m["accuracy"],
            })

    if all_results:
        rdf = pd.DataFrame(all_results)
        print(f"\n{'=' * 70}")
        print("汇总")
        print(f"{'=' * 70}")

        print(f"\n  {'联赛':<6} {'Hybrid':>10} {'Poisson':>10} {'Multi':>10}")
        for league in LEAGUES:
            lg = rdf[rdf["league"] == league]
            if len(lg) == 0:
                continue
            name = LEAGUE_NAMES.get(league, league)
            print(f"  {name:<6} {lg['rps_hybrid'].mean():>10.4f} "
                  f"{lg['rps_poisson'].mean():>10.4f} "
                  f"{lg['rps_multi'].mean():>10.4f}")

        print(f"\n  {'总计':<6} {rdf['rps_hybrid'].mean():>10.4f} "
              f"{rdf['rps_poisson'].mean():>10.4f} "
              f"{rdf['rps_multi'].mean():>10.4f}")

        d = rdf['rps_hybrid'].mean() - rdf['rps_poisson'].mean()
        print(f"\n  Hybrid vs Poisson: {d:+.4f}")


if __name__ == "__main__":
    run()
