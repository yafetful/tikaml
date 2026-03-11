"""Adaptive blending: route to different models based on confidence.

- High confidence (max_prob > threshold): trust Poisson more
- Low confidence: blend Poisson + Multiclass + DC equally

Also tests: full 4-model weighted ensemble (DC + Poisson + Multiclass + Elo).

Usage:
    source .venv/bin/activate
    python scripts/run_adaptive_blend.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS
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


class MulticlassPredictor:
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
        return X.fillna(self._medians)

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
            X_val, y_val = X_train.iloc[split:], y_train[split:]
            X_train, y_train = X_train.iloc[:split], y_train[:split]

        self.model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3,
            n_estimators=800, learning_rate=0.05, max_depth=6,
            num_leaves=21, min_child_samples=40,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0, verbose=-1,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       callbacks=callbacks)
        return self

    def predict_proba(self, df):
        X = self._prepare(df)
        return self.model.predict_proba(X)


def predict_dc(df_test, df_train):
    probs = []
    for league in df_test["league"].unique():
        lt = df_test[df_test["league"] == league]
        ltr = df_train[df_train["league"] == league]
        dc = DixonColesModel(half_life_days=180)
        dc.fit(ltr[["date", "home_team", "away_team", "home_goals", "away_goals"]],
               current_date=lt["date"].min())
        for idx, row in lt.iterrows():
            try:
                p = dc.predict_1x2(row["home_team"], row["away_team"])
                if np.any(np.isnan(p)):
                    p = np.array([0.40, 0.30, 0.30])
            except (KeyError, ValueError):
                p = np.array([0.40, 0.30, 0.30])
            probs.append((idx, p))
    probs.sort(key=lambda x: x[0])
    return np.array([p for _, p in probs])


def predict_elo(df_test, df_train):
    elo = EloModel(k=20, home_advantage=100)
    elo.fit(df_train[["date", "league", "season",
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
    val = all_train[all_train["season"] == prev_season]
    train = all_train[all_train["season"] < prev_season]

    poisson = LGBMPoissonModel(rho=-0.108)
    poisson.fit(train, val_df=val)
    multi = MulticlassPredictor()
    multi.fit(train, val_df=val)
    return poisson, multi


def optimize_4model_weights(dc_p, poi_p, mul_p, elo_p, outcomes):
    """Optimize weights for 4-model ensemble."""
    def objective(w):
        w = np.abs(w)
        w = w / w.sum()
        blended = w[0]*dc_p + w[1]*poi_p + w[2]*mul_p + w[3]*elo_p
        blended /= blended.sum(axis=1, keepdims=True)
        return np.mean([
            ranked_probability_score(blended[i], outcomes[i])
            for i in range(len(outcomes))
        ])

    best_result = None
    best_rps = float("inf")
    for init_w in [
        [0.15, 0.35, 0.35, 0.15],
        [0.10, 0.50, 0.30, 0.10],
        [0.20, 0.40, 0.20, 0.20],
        [0.10, 0.45, 0.45, 0.00],
        [0.25, 0.25, 0.25, 0.25],
    ]:
        result = minimize(objective, init_w, method="Nelder-Mead",
                          options={"maxiter": 500, "xatol": 1e-5})
        if result.fun < best_rps:
            best_rps = result.fun
            best_result = result
    w = np.abs(best_result.x)
    return w / w.sum()


def run():
    print("=" * 70)
    print("TikaML: 4模型加权集成 (DC + Poisson + Multi + Elo)")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  数据: {len(df)} 场")

    all_results = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        df_test = df[df["season"] == test_season].copy()
        df_train = df[df["season"] < test_season].copy()

        poisson, multi = train_models(df, test_season)
        if poisson is None:
            continue

        poi_probs = poisson.predict_1x2(df_test)
        mul_probs = multi.predict_proba(df_test)
        dc_probs = predict_dc(df_test, df_train)
        elo_probs = predict_elo(df_test, df_train)

        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in df_test.iterrows()
        ])

        # Optimize weights on prior season
        prior_seasons = sorted(
            s for s in df["season"].unique()
            if "2018-2019" <= s < test_season
        )
        if len(prior_seasons) >= 1:
            opt_s = prior_seasons[-1]
            opt_test = df[df["season"] == opt_s].copy()
            opt_train = df[df["season"] < opt_s].copy()
            opt_poi, opt_mul = train_models(df, opt_s)
            if opt_poi is not None:
                opt_pp = opt_poi.predict_1x2(opt_test)
                opt_mp = opt_mul.predict_proba(opt_test)
                opt_dc = predict_dc(opt_test, opt_train)
                opt_elo = predict_elo(opt_test, opt_train)
                opt_outs = np.array([
                    match_outcome(r["home_goals"], r["away_goals"])
                    for _, r in opt_test.iterrows()
                ])
                weights = optimize_4model_weights(
                    opt_dc, opt_pp, opt_mp, opt_elo, opt_outs)
            else:
                weights = np.array([0.15, 0.35, 0.35, 0.15])
        else:
            weights = np.array([0.15, 0.35, 0.35, 0.15])

        print(f"  权重: DC={weights[0]:.3f}, Poisson={weights[1]:.3f}, "
              f"Multi={weights[2]:.3f}, Elo={weights[3]:.3f}")

        # 4-model blend
        blended = (weights[0]*dc_probs + weights[1]*poi_probs +
                   weights[2]*mul_probs + weights[3]*elo_probs)
        blended /= blended.sum(axis=1, keepdims=True)

        for league in LEAGUES:
            mask = df_test["league"].values == league
            if mask.sum() == 0:
                continue

            b_m = evaluate_predictions(blended[mask], outcomes[mask])
            p_m = evaluate_predictions(poi_probs[mask], outcomes[mask])

            name = LEAGUE_NAMES.get(league, league)
            print(f"\n  {name} ({league}) — {mask.sum()} 场")
            print(f"    {'模型':<18} {'RPS':>8} {'Brier':>8} {'准确率':>8}")
            print(f"    {'4-Model Blend':<18} {b_m['rps']:>8.4f} "
                  f"{b_m['brier']:>8.4f} {b_m['accuracy']:>7.1%}")
            print(f"    {'Poisson Only':<18} {p_m['rps']:>8.4f} "
                  f"{p_m['brier']:>8.4f} {p_m['accuracy']:>7.1%}")

            all_results.append({
                "season": test_season, "league": league,
                "rps_blend": b_m["rps"], "rps_poisson": p_m["rps"],
                "acc_blend": b_m["accuracy"],
            })

    if all_results:
        rdf = pd.DataFrame(all_results)
        print(f"\n{'=' * 70}")
        print("汇总")
        print(f"{'=' * 70}")

        print(f"\n  {'联赛':<6} {'4-Model':>10} {'Poisson':>10} {'差值':>8}")
        for league in LEAGUES:
            lg = rdf[rdf["league"] == league]
            if len(lg) == 0:
                continue
            name = LEAGUE_NAMES.get(league, league)
            b = lg["rps_blend"].mean()
            p = lg["rps_poisson"].mean()
            print(f"  {name:<6} {b:>10.4f} {p:>10.4f} {b-p:>+8.4f}")

        b_t = rdf["rps_blend"].mean()
        p_t = rdf["rps_poisson"].mean()
        print(f"\n  {'总计':<6} {b_t:>10.4f} {p_t:>10.4f} {b_t-p_t:>+8.4f}")
        print(f"  准确率: {rdf['acc_blend'].mean():.1%}")


if __name__ == "__main__":
    run()
