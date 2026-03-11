"""Test market odds as features for prediction improvement.

Bookmaker odds capture information our model lacks:
- Injuries, suspensions, lineup decisions
- Market consensus (wisdom of crowds)
- Tactical matchup analysis by experts

This experiment:
1. Converts Pinnacle opening odds to implied probabilities
2. Adds them as features to the LightGBM model
3. Tests hybrid approach: model + odds consensus

Key question: how much of the RPS gap (0.199→0.185) is explained by market info?

Usage:
    source .venv/bin/activate
    python scripts/run_odds_features.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS, LGB_PARAMS
from src.evaluation import evaluate_predictions, match_outcome
import lightgbm as lgb

FEATURES_PATH = Path("data/opta/processed/features.csv")
ODDS_DIR = Path("data/odds")
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]

# Map Football-Data.co.uk league codes to our codes
LEAGUE_MAP = {
    "E0": "EPL",
    "SP1": "LL",
    "I1": "SEA",
    "D1": "BUN",
    "F1": "LI1",
}

# Team name mapping: Football-Data.co.uk → Opta
TEAM_NAME_MAP = {
    # EPL
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion",
    "Newcastle": "Newcastle United",
    "Sheffield United": "Sheffield United",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Bournemouth": "AFC Bournemouth",
    "Ipswich": "Ipswich Town",
    "Luton": "Luton Town",
    "Norwich": "Norwich City",
    "Watford": "Watford",
    "West Brom": "West Bromwich Albion",
    "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town",
    "Burnley": "Burnley",
    "Sunderland": "Sunderland",
    # La Liga
    "Ath Madrid": "Atlético de Madrid",
    "Ath Bilbao": "Athletic Club",
    "Betis": "Real Betis",
    "Sociedad": "Real Sociedad",
    "Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol",
    "Sp Gijon": "Sporting de Gijón",
    "La Coruna": "Deportivo de La Coruña",
    "Malaga": "Málaga",
    "Las Palmas": "Las Palmas",
    "Leganes": "Leganés",
    "Valladolid": "Real Valladolid",
    "Alaves": "Alavés",
    "Almeria": "Almería",
    "Cadiz": "Cádiz",
    "Celta": "Celta de Vigo",
    "Oviedo": "Real Oviedo",
    # Serie A
    "Inter": "Internazionale",
    "AC Milan": "AC Milan",
    "Napoli": "Napoli",
    "Lazio": "Lazio",
    "Roma": "Roma",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Torino": "Torino",
    "Verona": "Hellas Verona",
    "Sassuolo": "Sassuolo",
    "Spezia": "Spezia",
    "Salernitana": "Salernitana",
    "Lecce": "Lecce",
    "Spal": "Ars et Labor",
    # Bundesliga
    "Bayern Munich": "Bayern München",
    "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Borussia Mönchengladbach",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Mainz": "1. FSV Mainz 05",
    "Hertha": "Hertha Berlin",
    "Hoffenheim": "TSG Hoffenheim",
    "Wolfsburg": "VfL Wolfsburg",
    "Stuttgart": "VfB Stuttgart",
    "Augsburg": "Augsburg",
    "Freiburg": "SC Freiburg",
    "Union Berlin": "1. FC Union Berlin",
    "Cologne": "1. FC Köln",
    "FC Koln": "Köln",
    "Schalke 04": "FC Schalke 04",
    "Fortuna Dusseldorf": "Fortuna Düsseldorf",
    "Paderborn": "SC Paderborn 07",
    "Greuther Furth": "SpVgg Greuther Fürth",
    "Bielefeld": "Arminia Bielefeld",
    "Heidenheim": "1. FC Heidenheim",
    "Darmstadt": "Darmstadt 98",
    "St Pauli": "FC St. Pauli",
    "Holstein Kiel": "Holstein Kiel",
    "Hamburg": "Hamburger SV",
    "Hannover": "Hannover 96",
    "Nurnberg": "Nürnberg",
    # Ligue 1
    "Paris SG": "Paris Saint-Germain",
    "Marseille": "Marseille",
    "Lyon": "Lyon",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Rennes": "Rennes",
    "Nice": "Nice",
    "Lens": "RC Lens",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "Montpellier": "Montpellier",
    "Toulouse": "Toulouse",
    "Reims": "Stade de Reims",
    "Brest": "Stade Brestois 29",
    "Clermont": "Clermont Foot",
    "Lorient": "Lorient",
    "Metz": "Metz",
    "Ajaccio": "Ajaccio",
    "Auxerre": "Auxerre",
    "Angers": "Angers SCO",
    "Le Havre": "Le Havre AC",
    "St Etienne": "Saint-Étienne",
    "Amiens": "Amiens SC",
    "Nimes": "Nîmes",
}


def season_code_to_name(code):
    """Convert '2324' to '2023-2024'."""
    start = int("20" + code[:2])
    end = int("20" + code[2:])
    return f"{start}-{end}"


def load_all_odds():
    """Load and merge all odds data."""
    all_dfs = []
    for f in sorted(os.listdir(ODDS_DIR)):
        if not f.endswith(".csv"):
            continue
        parts = f.replace(".csv", "").split("_")
        if len(parts) != 2:
            continue
        fd_league = parts[0]
        season_code = parts[1]
        our_league = LEAGUE_MAP.get(fd_league)
        if not our_league:
            continue

        season_name = season_code_to_name(season_code)

        try:
            df = pd.read_csv(
                ODDS_DIR / f, encoding="latin1", on_bad_lines="skip")
        except Exception:
            continue

        # Essential columns
        needed = ["Date", "HomeTeam", "AwayTeam"]
        odds_cols = []
        # Pinnacle opening odds (best for implied probability)
        for col in ["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
                     "B365H", "B365D", "B365A", "BbAvH", "BbAvD", "BbAvA"]:
            if col in df.columns:
                odds_cols.append(col)

        if not all(c in df.columns for c in needed):
            continue

        df = df[needed + odds_cols].copy()
        df["league"] = our_league
        df["season"] = season_name

        # Parse date
        for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
            try:
                df["date"] = pd.to_datetime(df["Date"], format=fmt)
                break
            except Exception:
                continue
        else:
            try:
                df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
            except Exception:
                continue

        # Map team names
        df["home_team"] = df["HomeTeam"].map(
            lambda x: TEAM_NAME_MAP.get(x, x))
        df["away_team"] = df["AwayTeam"].map(
            lambda x: TEAM_NAME_MAP.get(x, x))

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    odds = pd.concat(all_dfs, ignore_index=True)
    odds = odds.sort_values("date").reset_index(drop=True)
    return odds


def odds_to_implied_probs(odds_h, odds_d, odds_a):
    """Convert decimal odds to implied probabilities (with overround removal)."""
    raw_h = 1.0 / odds_h
    raw_d = 1.0 / odds_d
    raw_a = 1.0 / odds_a
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def merge_odds_with_features(features_df, odds_df):
    """Merge odds data with feature table by date + teams."""
    # Create matching key
    features_df = features_df.copy()
    features_df["match_key"] = (
        features_df["date"].dt.strftime("%Y-%m-%d") + "_" +
        features_df["home_team"] + "_" + features_df["away_team"]
    )

    odds_df = odds_df.copy()
    odds_df["match_key"] = (
        odds_df["date"].dt.strftime("%Y-%m-%d") + "_" +
        odds_df["home_team"] + "_" + odds_df["away_team"]
    )

    # Direct merge
    merged = features_df.merge(
        odds_df[["match_key"] + [c for c in odds_df.columns
                                  if c.startswith(("PS", "B365", "BbAv"))]],
        on="match_key", how="left"
    )

    # For unmatched, try fuzzy date matching (±1 day)
    unmatched = merged[merged["PSH"].isna() if "PSH" in merged.columns
                       else merged["B365H"].isna()].index
    if len(unmatched) > 0:
        # Try matching by league+season+teams (ignoring exact date)
        features_df["fuzzy_key"] = (
            features_df["league"] + "_" + features_df["season"] + "_" +
            features_df["home_team"] + "_" + features_df["away_team"]
        )
        odds_df["fuzzy_key"] = (
            odds_df["league"] + "_" + odds_df["season"] + "_" +
            odds_df["home_team"] + "_" + odds_df["away_team"]
        )

        fuzzy_odds = odds_df.drop_duplicates("fuzzy_key", keep="last")
        fuzzy_map = fuzzy_odds.set_index("fuzzy_key")[
            [c for c in odds_df.columns if c.startswith(("PS", "B365", "BbAv"))]
        ].to_dict("index")

        for idx in unmatched:
            fk = features_df.loc[idx, "fuzzy_key"]
            if fk in fuzzy_map:
                for col, val in fuzzy_map[fk].items():
                    if col in merged.columns:
                        merged.loc[idx, col] = val

    return merged


def run():
    print("=" * 70)
    print("TikaML: 博彩赔率特征实验")
    print("  将 Pinnacle/Bet365 开盘赔率转换为特征")
    print("=" * 70)

    # Load data
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  特征数据: {len(df)} 场")

    odds = load_all_odds()
    print(f"  赔率数据: {len(odds)} 场")

    # Merge
    merged = merge_odds_with_features(df, odds)

    # Compute implied probabilities
    # Prefer Pinnacle, fallback to Bet365, then market average
    for prefix, h, d, a in [
        ("pin", "PSH", "PSD", "PSA"),
        ("b365", "B365H", "B365D", "B365A"),
        ("avg", "BbAvH", "BbAvD", "BbAvA"),
    ]:
        if h in merged.columns and d in merged.columns:
            valid = merged[h].notna() & merged[d].notna() & merged[a].notna()
            valid &= (merged[h] > 1) & (merged[d] > 1) & (merged[a] > 1)
            ip_h, ip_d, ip_a = odds_to_implied_probs(
                merged.loc[valid, h].values,
                merged.loc[valid, d].values,
                merged.loc[valid, a].values,
            )
            merged.loc[valid, f"odds_prob_h_{prefix}"] = ip_h
            merged.loc[valid, f"odds_prob_d_{prefix}"] = ip_d
            merged.loc[valid, f"odds_prob_a_{prefix}"] = ip_a

    # Check coverage
    for prefix in ["pin", "b365", "avg"]:
        col = f"odds_prob_h_{prefix}"
        if col in merged.columns:
            coverage = merged[col].notna().mean()
            print(f"  {prefix} 赔率覆盖率: {coverage:.1%}")

    # Use best available odds
    odds_col_h = "odds_prob_h_pin"
    odds_col_d = "odds_prob_d_pin"
    odds_col_a = "odds_prob_a_pin"
    for prefix in ["pin", "b365", "avg"]:
        h_col = f"odds_prob_h_{prefix}"
        if h_col in merged.columns and merged[h_col].notna().any():
            odds_col_h = h_col
            odds_col_d = f"odds_prob_d_{prefix}"
            odds_col_a = f"odds_prob_a_{prefix}"
            break

    # Fill missing odds with fallback
    for prefix in ["pin", "b365", "avg"]:
        h_col = f"odds_prob_h_{prefix}"
        d_col = f"odds_prob_d_{prefix}"
        a_col = f"odds_prob_a_{prefix}"
        if h_col in merged.columns:
            mask = merged[odds_col_h].isna() & merged[h_col].notna()
            merged.loc[mask, odds_col_h] = merged.loc[mask, h_col]
            merged.loc[mask, odds_col_d] = merged.loc[mask, d_col]
            merged.loc[mask, odds_col_a] = merged.loc[mask, a_col]

    final_coverage = merged[odds_col_h].notna().mean()
    print(f"  最终赔率覆盖率: {final_coverage:.1%}")

    # Define feature sets
    base_features = FEATURE_COLS.copy()
    odds_features = [odds_col_h, odds_col_d, odds_col_a]
    enhanced_features = base_features + odds_features

    all_results = []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        all_train = merged[
            (merged["season"] < test_season) &
            (merged["season"] >= "2018-2019")  # Only seasons with odds
        ].copy()

        seasons = sorted(all_train["season"].unique())
        if len(seasons) < 2:
            continue
        prev = seasons[-1]
        val = all_train[all_train["season"] == prev]
        train = all_train[all_train["season"] < prev]
        test = merged[merged["season"] == test_season].copy()

        outcomes = np.array([
            match_outcome(r["home_goals"], r["away_goals"])
            for _, r in test.iterrows()
        ])

        # Check odds coverage in test set
        test_odds_coverage = test[odds_col_h].notna().mean()
        print(f"  测试集赔率覆盖率: {test_odds_coverage:.1%}")

        # --- Model 1: LGBM Baseline (no odds) ---
        lgbm_base = LGBMPoissonModel(rho=-0.108)
        lgbm_base.fit(train, val_df=val)
        base_probs = lgbm_base.predict_1x2(test)
        base_m = evaluate_predictions(base_probs, outcomes)
        print(f"  LGBM (无赔率): RPS={base_m['rps']:.4f}  Acc={base_m['accuracy']:.1%}")

        # --- Model 2: LGBM with odds features ---
        available_enhanced = [c for c in enhanced_features if c in train.columns]
        has_odds = all(c in train.columns for c in odds_features)

        if has_odds and train[odds_col_h].notna().sum() > 500:
            # Train with odds features
            X_train = train[available_enhanced].copy()
            medians = X_train.median()
            X_train = X_train.fillna(medians)
            X_val = val[available_enhanced].copy().fillna(medians)
            X_test = test[available_enhanced].copy().fillna(medians)

            params = LGB_PARAMS.copy()
            callbacks = [lgb.early_stopping(50, verbose=False)]

            model_h = lgb.LGBMRegressor(**params)
            model_h.fit(X_train, train["home_goals"].values,
                        eval_set=[(X_val, val["home_goals"].values)],
                        callbacks=callbacks)

            model_a = lgb.LGBMRegressor(**params)
            model_a.fit(X_train, train["away_goals"].values,
                        eval_set=[(X_val, val["away_goals"].values)],
                        callbacks=callbacks)

            lh = np.clip(model_h.predict(X_test), 0.1, 5.0)
            la = np.clip(model_a.predict(X_test), 0.1, 5.0)

            from scipy.stats import poisson
            odds_probs = []
            for h, a in zip(lh, la):
                matrix = np.zeros((7, 7))
                for i in range(7):
                    for j in range(7):
                        tau = 1.0
                        rho = -0.108
                        if i == 0 and j == 0: tau = max(0, 1 - h*a*rho)
                        elif i == 0 and j == 1: tau = max(0, 1 + h*rho)
                        elif i == 1 and j == 0: tau = max(0, 1 + a*rho)
                        elif i == 1 and j == 1: tau = max(0, 1 - rho)
                        matrix[i, j] = tau * poisson.pmf(i, h) * poisson.pmf(j, a)
                matrix /= matrix.sum()
                p_h = np.tril(matrix, -1).sum()
                p_d = np.trace(matrix)
                p_a = np.triu(matrix, 1).sum()
                p = np.array([p_h, p_d, p_a])
                odds_probs.append(p / p.sum())
            odds_probs = np.array(odds_probs)

            odds_m = evaluate_predictions(odds_probs, outcomes)
            print(f"  LGBM+赔率: RPS={odds_m['rps']:.4f}  Acc={odds_m['accuracy']:.1%}  "
                  f"({odds_m['rps'] - base_m['rps']:+.4f})")

            # Feature importance for odds features
            imp_h = dict(zip(available_enhanced, model_h.feature_importances_))
            imp_a = dict(zip(available_enhanced, model_a.feature_importances_))
            for col in odds_features:
                avg_imp = (imp_h.get(col, 0) + imp_a.get(col, 0)) / 2
                print(f"    {col}: importance={avg_imp:.0f}")
        else:
            odds_m = {"rps": None, "accuracy": None}
            odds_probs = None
            print(f"  LGBM+赔率: 赔率数据不足，跳过")

        # --- Model 3: Pure odds baseline ---
        if test[odds_col_h].notna().sum() > 100:
            pure_odds = np.zeros((len(test), 3))
            valid_mask = test[odds_col_h].notna()
            pure_odds[valid_mask, 0] = test.loc[valid_mask, odds_col_h].values
            pure_odds[valid_mask, 1] = test.loc[valid_mask, odds_col_d].values
            pure_odds[valid_mask, 2] = test.loc[valid_mask, odds_col_a].values
            # Fill missing with model predictions
            pure_odds[~valid_mask] = base_probs[~valid_mask]
            # Normalize
            pure_odds /= pure_odds.sum(axis=1, keepdims=True)

            pure_m = evaluate_predictions(pure_odds, outcomes)
            print(f"  纯赔率:     RPS={pure_m['rps']:.4f}  Acc={pure_m['accuracy']:.1%}")

            # Blend: model + odds
            for alpha in [0.3, 0.5, 0.7]:
                blended = alpha * base_probs + (1 - alpha) * pure_odds
                blended /= blended.sum(axis=1, keepdims=True)
                blend_m = evaluate_predictions(blended, outcomes)
                print(f"  Model+Odds(α={alpha:.1f}): RPS={blend_m['rps']:.4f}  "
                      f"Acc={blend_m['accuracy']:.1%}")

            # Draw analysis
            odds_draws = (np.argmax(pure_odds, axis=1) == 1).sum()
            actual_draws = (outcomes == 1).sum()
            print(f"\n  平局: 实际={actual_draws}, 赔率预测={odds_draws}, 模型预测=0")
        else:
            pure_m = {"rps": None}
            print(f"  纯赔率: 数据不足")

        all_results.append({
            "season": test_season,
            "rps_base": base_m["rps"],
            "rps_odds_feat": odds_m["rps"],
            "rps_pure_odds": pure_m.get("rps"),
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")

    rdf = pd.DataFrame(all_results)
    print(f"\n  {'方法':<18} ", end="")
    for s in TEST_SEASONS:
        print(f"  {s[:7]:>10}", end="")
    print(f"  {'平均':>10}")

    for col, name in [
        ("rps_base", "LGBM 无赔率"),
        ("rps_odds_feat", "LGBM+赔率特征"),
        ("rps_pure_odds", "纯赔率"),
    ]:
        vals = rdf[col].values
        valid = [v for v in vals if v is not None and not np.isnan(v)]
        print(f"  {name:<18}", end="")
        for v in vals:
            if v is not None and not np.isnan(v):
                print(f"  {v:>10.4f}", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        if valid:
            print(f"  {np.mean(valid):>10.4f}")
        else:
            print()


if __name__ == "__main__":
    run()
