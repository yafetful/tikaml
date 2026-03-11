"""Inference engine for football match prediction.

Extracts team states from historical data, builds feature vectors for
upcoming matches, trains/loads model, and generates 7x7 score matrix predictions.

Usage:
    from src.inference import MatchPredictor
    predictor = MatchPredictor()
    predictor.train()  # Train on latest data
    result = predictor.predict("Arsenal", "Chelsea", "EPL", "2025-2026", week=28)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS

FEATURES_PATH = Path("data/opta/processed/features.csv")
MODEL_DIR = Path("models")

# Map of rolling feature columns to the raw stat they came from
# Used to extract team state from either _home or _away perspective
ROLLING_HOME_COLS = [c for c in FEATURE_COLS if c.endswith("_home")]
ROLLING_AWAY_COLS = [c for c in FEATURE_COLS if c.endswith("_away")]

# Features that are match-level (not team-specific)
MATCH_LEVEL_FEATURES = [
    "days_rest_diff", "is_midweek", "season_stage",
    "league_avg_goals", "points_diff", "position_diff",
    "is_relegation_battle", "is_title_race",
    "points_to_safety", "points_to_leader",
    "match_importance",
    "h2h_win_pct_home", "h2h_goal_diff_home", "h2h_matches",
    "xg_diff_abs", "defensive_strength", "league_draw_rate",
]

LEAGUE_TOTAL_ROUNDS = {
    "EPL": 38, "LL": 38, "SEA": 38, "BUN": 34, "LI1": 38,
}


class MatchPredictor:
    """End-to-end match prediction engine."""

    def __init__(self, features_path=None):
        self.features_path = features_path or FEATURES_PATH
        self.df = None
        self.model = None
        self._loaded = False

    def load_data(self):
        """Load historical feature table."""
        if self._loaded:
            return
        self.df = pd.read_csv(
            self.features_path, parse_dates=["date"], low_memory=False
        )
        self.df = self.df.sort_values("date").reset_index(drop=True)
        self._loaded = True

    def train(self, min_season="2016-2017", save=False):
        """Train model on all available data.

        Args:
            min_season: Earliest season to include in training.
            save: If True, save trained model to disk.
        """
        self.load_data()

        all_data = self.df[self.df["season"] >= min_season].copy()

        # Full training: use all data, last 20% by time for early stopping
        n = len(all_data)
        split = int(n * 0.8)
        train = all_data.iloc[:split]
        val = all_data.iloc[split:]

        self.model = LGBMPoissonModel(rho=-0.108)
        self.model.fit(train, val_df=val)

        date_min = all_data["date"].min().strftime("%Y-%m-%d")
        date_max = all_data["date"].max().strftime("%Y-%m-%d")
        print(f"  模型训练完成: {len(train)} 训练 / {len(val)} 验证")
        print(f"  数据范围: {date_min} ~ {date_max} ({n} 场)")

        if save:
            self.model.save(str(MODEL_DIR))

        return self

    def load_model(self):
        """Load a previously saved model from disk."""
        self.load_data()
        self.model = LGBMPoissonModel.load(str(MODEL_DIR))
        return self

    def _get_team_state(self, team_name, before_date=None):
        """Extract a team's current rolling feature state from history.

        Returns dict of {feature_name: value} using the _home/_away convention
        from the team's most recent match.
        """
        self.load_data()

        # Find team's matches
        mask_home = self.df["home_team"] == team_name
        mask_away = self.df["away_team"] == team_name
        team_matches = self.df[mask_home | mask_away].copy()

        if before_date is not None:
            team_matches = team_matches[team_matches["date"] < before_date]

        if len(team_matches) == 0:
            return {}, None, None

        last_match = team_matches.iloc[-1]
        was_home = last_match["home_team"] == team_name

        state = {}
        # Extract rolling features from the team's perspective
        if was_home:
            for col in ROLLING_HOME_COLS:
                if col in last_match.index:
                    state[col] = last_match[col]
        else:
            # Team was away → map _away features to _home for prediction
            for col in ROLLING_AWAY_COLS:
                home_col = col.replace("_away", "_home")
                if col in last_match.index and home_col in FEATURE_COLS:
                    state[home_col] = last_match[col]

        # Also extract venue-specific features differently
        # (venue rolling needs special handling since it depends on home/away)

        return state, last_match["date"], was_home

    def _get_team_last_date(self, team_name, before_date=None):
        """Get team's most recent match date."""
        mask = (self.df["home_team"] == team_name) | \
               (self.df["away_team"] == team_name)
        matches = self.df[mask]
        if before_date is not None:
            matches = matches[matches["date"] < before_date]
        if len(matches) == 0:
            return None
        return matches.iloc[-1]["date"]

    def _compute_h2h(self, home_team, away_team):
        """Compute H2H features from history."""
        self.load_data()
        mask = (
            ((self.df["home_team"] == home_team) &
             (self.df["away_team"] == away_team)) |
            ((self.df["home_team"] == away_team) &
             (self.df["away_team"] == home_team))
        )
        h2h = self.df[mask].copy()

        if len(h2h) < 2:
            return {
                "h2h_win_pct_home": np.nan,
                "h2h_goal_diff_home": np.nan,
                "h2h_matches": 0,
            }

        # Last 10 H2H meetings
        h2h = h2h.tail(10)
        wins = 0
        total_gd = 0
        for _, row in h2h.iterrows():
            if row["home_team"] == home_team:
                gd = row["home_goals"] - row["away_goals"]
            else:
                gd = row["away_goals"] - row["home_goals"]
            total_gd += gd
            if gd > 0:
                wins += 1
            elif gd == 0:
                wins += 0.5

        n = len(h2h)
        return {
            "h2h_win_pct_home": wins / n,
            "h2h_goal_diff_home": total_gd / n,
            "h2h_matches": n,
        }

    def _compute_league_context(self, league, season, match_date, week=None):
        """Compute league-level context features."""
        self.load_data()
        ls = self.df[
            (self.df["league"] == league) &
            (self.df["season"] == season) &
            (self.df["date"] < match_date)
        ]

        league_avg_goals = ls["total_goals"].mean() if len(ls) > 0 else 2.7
        league_draw_rate = (
            (ls["home_goals"] == ls["away_goals"]).mean()
            if len(ls) > 0 else 0.25
        )

        max_week = LEAGUE_TOTAL_ROUNDS.get(league, 38)
        if week is not None:
            season_stage = week / max_week
        else:
            season_stage = 0.5

        return {
            "league_avg_goals": league_avg_goals,
            "league_draw_rate": league_draw_rate,
            "season_stage": min(season_stage, 1.0),
        }

    def _compute_table_features(self, home_team, away_team, league, season,
                                match_date):
        """Compute points_diff, position_diff, and match importance from league table."""
        RELEGATION_SPOTS = {
            "EPL": 3, "LL": 3, "SEA": 3, "BUN": 3, "LI1": 3,
        }
        TOTAL_TEAMS = {
            "EPL": 20, "LL": 20, "SEA": 20, "BUN": 18, "LI1": 18,
        }

        self.load_data()
        ls = self.df[
            (self.df["league"] == league) &
            (self.df["season"] == season) &
            (self.df["date"] < match_date)
        ].copy()

        if len(ls) == 0:
            return {
                "points_diff": 0, "position_diff": 0,
                "is_relegation_battle": 0, "is_title_race": 0,
                "points_to_safety": np.nan, "points_to_leader": np.nan,
                "match_importance": 0.0,
            }

        # Build points table
        points = {}
        for _, row in ls.iterrows():
            h, a = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]
            if hg > ag:
                points[h] = points.get(h, 0) + 3
                points[a] = points.get(a, 0)
            elif hg == ag:
                points[h] = points.get(h, 0) + 1
                points[a] = points.get(a, 0) + 1
            else:
                points[h] = points.get(h, 0)
                points[a] = points.get(a, 0) + 3

        h_pts = points.get(home_team, 0)
        a_pts = points.get(away_team, 0)
        points_diff = h_pts - a_pts

        # Position diff
        sorted_teams = sorted(points.items(), key=lambda x: -x[1])
        positions = {t: i + 1 for i, (t, _) in enumerate(sorted_teams)}
        h_pos = positions.get(home_team, len(positions) // 2)
        a_pos = positions.get(away_team, len(positions) // 2)
        position_diff = a_pos - h_pos  # Positive = home ranked higher

        # Match importance features
        rel_spots = RELEGATION_SPOTS.get(league, 3)
        n_teams = TOTAL_TEAMS.get(league, 20)
        n = len(sorted_teams)

        if n >= 4:
            sorted_pts = sorted(points.values(), reverse=True)
            safety_idx = min(n - rel_spots, n - 1)
            safety_pts = sorted_pts[safety_idx] if safety_idx >= 0 else 0
            leader_pts = sorted_pts[0]

            # is_relegation_battle
            h_near_rel = (h_pos > n - rel_spots) or (h_pts - safety_pts <= 3)
            a_near_rel = (a_pos > n - rel_spots) or (a_pts - safety_pts <= 3)
            is_rel = int(h_near_rel or a_near_rel)

            # is_title_race
            h_in_title = (h_pos <= 3) and (leader_pts - h_pts <= 6)
            a_in_title = (a_pos <= 3) and (leader_pts - a_pts <= 6)
            is_title = int(h_in_title or a_in_title)

            # points_to_safety (lower-ranked team)
            lower_pts = min(h_pts, a_pts)
            pts_to_safety = lower_pts - safety_pts

            # points_to_leader (higher-ranked team)
            higher_pts = max(h_pts, a_pts)
            pts_to_leader = leader_pts - higher_pts
        else:
            is_rel = 0
            is_title = 0
            pts_to_safety = np.nan
            pts_to_leader = np.nan

        return {
            "points_diff": points_diff,
            "position_diff": position_diff,
            "is_relegation_battle": is_rel,
            "is_title_race": is_title,
            "points_to_safety": pts_to_safety,
            "points_to_leader": pts_to_leader,
        }

    def _compute_draw_features(self, home_team, away_team, match_date):
        """Compute draw-prone features."""
        self.load_data()

        draw_rates = {}
        for team_name, suffix in [(home_team, "home"), (away_team, "away")]:
            mask = (
                ((self.df["home_team"] == team_name) |
                 (self.df["away_team"] == team_name)) &
                (self.df["date"] < match_date)
            )
            matches = self.df[mask].tail(20)
            if len(matches) >= 5:
                draws = (matches["home_goals"] == matches["away_goals"]).mean()
                draw_rates[f"draw_rate_{suffix}"] = draws
            else:
                draw_rates[f"draw_rate_{suffix}"] = np.nan

        return draw_rates

    def _compute_momentum(self, team_name, match_date):
        """Compute momentum feature for a team."""
        self.load_data()
        mask = (
            ((self.df["home_team"] == team_name) |
             (self.df["away_team"] == team_name)) &
            (self.df["date"] < match_date)
        )
        matches = self.df[mask].copy()

        if len(matches) < 5:
            return np.nan

        results = []
        for _, row in matches.iterrows():
            if row["home_team"] == team_name:
                gd = row["home_goals"] - row["away_goals"]
            else:
                gd = row["away_goals"] - row["home_goals"]
            if gd > 0:
                results.append(3)
            elif gd == 0:
                results.append(1)
            else:
                results.append(0)

        short = np.mean(results[-5:])
        long_val = np.mean(results[-15:]) if len(results) >= 15 else np.mean(results)
        return short - long_val

    def _compute_lineup_features(self, team_name, match_date, suffix):
        """Estimate lineup rotation features for an upcoming match.

        Since we don't have the actual lineup for the upcoming match,
        we use the team's recent historical averages as estimates.

        Args:
            team_name: Team name string.
            match_date: Date of the upcoming match.
            suffix: "home" or "away".

        Returns:
            dict with lineup feature columns.
        """
        self.load_data()

        mask = (
            ((self.df["home_team"] == team_name) |
             (self.df["away_team"] == team_name)) &
            (self.df["date"] < match_date)
        )
        matches = self.df[mask].tail(5)

        result = {}

        # lineup_changes: average of recent values
        lc_col_h = "lineup_changes_home"
        lc_col_a = "lineup_changes_away"
        lc_vals = []
        for _, row in matches.iterrows():
            if row["home_team"] == team_name:
                v = row.get(lc_col_h)
            else:
                v = row.get(lc_col_a)
            if pd.notna(v):
                lc_vals.append(v)
        result[f"lineup_changes_{suffix}"] = (
            np.mean(lc_vals) if lc_vals else np.nan
        )

        # lineup_stability: latest value
        ls_col_h = "lineup_stability_home"
        ls_col_a = "lineup_stability_away"
        ls_val = np.nan
        for _, row in matches.iloc[::-1].iterrows():
            if row["home_team"] == team_name:
                v = row.get(ls_col_h)
            else:
                v = row.get(ls_col_a)
            if pd.notna(v):
                ls_val = v
                break
        result[f"lineup_stability_{suffix}"] = ls_val

        # formation_change: average of recent values
        fc_col_h = "formation_change_home"
        fc_col_a = "formation_change_away"
        fc_vals = []
        for _, row in matches.iterrows():
            if row["home_team"] == team_name:
                v = row.get(fc_col_h)
            else:
                v = row.get(fc_col_a)
            if pd.notna(v):
                fc_vals.append(v)
        result[f"formation_change_{suffix}"] = (
            np.mean(fc_vals) if fc_vals else np.nan
        )

        # squad_rotation_rate: latest value
        sr_col_h = "squad_rotation_rate_home"
        sr_col_a = "squad_rotation_rate_away"
        sr_val = np.nan
        for _, row in matches.iloc[::-1].iterrows():
            if row["home_team"] == team_name:
                v = row.get(sr_col_h)
            else:
                v = row.get(sr_col_a)
            if pd.notna(v):
                sr_val = v
                break
        result[f"squad_rotation_rate_{suffix}"] = sr_val

        return result

    def build_feature_row(self, home_team, away_team, league, season,
                          match_date, week=None, odds=None):
        """Build complete feature vector for an upcoming match.

        Args:
            home_team: Home team name (must match names in features.csv)
            away_team: Away team name
            league: League code (EPL, LL, SEA, BUN, LI1)
            season: Season string (e.g. "2025-2026")
            match_date: Match date (datetime or string)
            week: Match week number (optional)
            odds: Optional dict with decimal odds, e.g.
                  {"home": 2.1, "draw": 3.4, "away": 3.5} or
                  {"prob_home": 0.45, "prob_draw": 0.28, "prob_away": 0.27}

        Returns:
            dict with all FEATURE_COLS values
        """
        self.load_data()

        if isinstance(match_date, str):
            match_date = pd.Timestamp(match_date)

        features = {}

        # 1. Home team rolling features (from _home perspective)
        home_state, home_last_date, _ = self._get_team_state(
            home_team, before_date=match_date)
        for col in ROLLING_HOME_COLS:
            features[col] = home_state.get(col, np.nan)

        # 2. Away team rolling features (from _away perspective)
        away_state_raw, away_last_date, _ = self._get_team_state(
            away_team, before_date=match_date)
        # away_state_raw has features in _home convention, map to _away
        for col in ROLLING_AWAY_COLS:
            home_equiv = col.replace("_away", "_home")
            features[col] = away_state_raw.get(home_equiv, np.nan)

        # 3. Days rest
        if home_last_date is not None:
            features["days_rest_home"] = (match_date - home_last_date).days
        else:
            features["days_rest_home"] = np.nan
        if away_last_date is not None:
            features["days_rest_away"] = (match_date - away_last_date).days
        else:
            features["days_rest_away"] = np.nan
        features["days_rest_diff"] = (
            (features["days_rest_home"] or 0) - (features["days_rest_away"] or 0)
        )

        # 4. Match context
        features["is_midweek"] = int(match_date.dayofweek in (1, 2, 3))

        # 5. League context
        league_ctx = self._compute_league_context(
            league, season, match_date, week)
        features.update(league_ctx)

        # 6. Table features (includes importance sub-features)
        table_feats = self._compute_table_features(
            home_team, away_team, league, season, match_date)
        features.update(table_feats)

        # 6b. Composite match_importance (needs season_stage from step 5)
        season_stage = features.get("season_stage", 0.5)
        zone_score = 0.0
        if features.get("is_relegation_battle", 0):
            zone_score += 0.4
        if features.get("is_title_race", 0):
            zone_score += 0.4
        pts_gap = abs(features.get("points_diff", 0))
        proximity_score = max(0, 1.0 - pts_gap / 15.0) * 0.3
        stage_weight = 1.0 + max(0, season_stage - 0.7) * 2.0 if not np.isnan(season_stage) else 1.0
        features["match_importance"] = round(
            (zone_score + proximity_score) * stage_weight, 4)

        # 7. H2H
        h2h = self._compute_h2h(home_team, away_team)
        features.update(h2h)

        # 8. Relative strength (team rolling / league avg)
        for base in ["xg_rolling", "xgot_rolling", "shots_rolling"]:
            h_val = features.get(f"{base}_home")
            a_val = features.get(f"{base}_away")
            if h_val is not None and a_val is not None:
                avg = (h_val + a_val) / 2 if (h_val + a_val) > 0 else 1
                features[f"{base}_rel_home"] = h_val / avg if avg else np.nan
                features[f"{base}_rel_away"] = a_val / avg if avg else np.nan

        # 9. Momentum
        features["momentum_home"] = self._compute_momentum(
            home_team, match_date)
        features["momentum_away"] = self._compute_momentum(
            away_team, match_date)

        # 10. Draw features
        draw_feats = self._compute_draw_features(
            home_team, away_team, match_date)
        features.update(draw_feats)

        # xg_diff_abs
        xg_h = features.get("xg_rolling_home")
        xg_a = features.get("xg_rolling_away")
        if xg_h is not None and xg_a is not None:
            features["xg_diff_abs"] = abs(xg_h - xg_a)

        # defensive_strength
        xgc_h = features.get("xg_rolling_conceded_home")
        xgc_a = features.get("xg_rolling_conceded_away")
        if xgc_h is not None and xgc_a is not None:
            features["defensive_strength"] = (xgc_h + xgc_a) / 2

        # 11. Lineup rotation features (estimated from recent history)
        lineup_home = self._compute_lineup_features(
            home_team, match_date, "home")
        features.update(lineup_home)
        lineup_away = self._compute_lineup_features(
            away_team, match_date, "away")
        features.update(lineup_away)

        # 12. Odds implied probabilities
        if odds is not None:
            if "prob_home" in odds:
                # Already in probability form
                features["odds_prob_home"] = odds["prob_home"]
                features["odds_prob_draw"] = odds["prob_draw"]
                features["odds_prob_away"] = odds["prob_away"]
            elif "home" in odds:
                # Decimal odds — convert with overround removal
                raw_h = 1.0 / odds["home"]
                raw_d = 1.0 / odds["draw"]
                raw_a = 1.0 / odds["away"]
                total = raw_h + raw_d + raw_a
                features["odds_prob_home"] = raw_h / total
                features["odds_prob_draw"] = raw_d / total
                features["odds_prob_away"] = raw_a / total
        else:
            features["odds_prob_home"] = np.nan
            features["odds_prob_draw"] = np.nan
            features["odds_prob_away"] = np.nan

        return features

    def predict(self, home_team, away_team, league, season, match_date,
                week=None, max_goals=7, odds=None):
        """Predict match outcome.

        Args:
            odds: Optional dict with decimal odds {"home": 2.1, "draw": 3.4, "away": 3.5}
                  or implied probabilities {"prob_home": 0.45, "prob_draw": 0.28, "prob_away": 0.27}

        Returns:
            dict with:
                - score_matrix: 7x7 numpy array of score probabilities
                - probs_1x2: [P(home), P(draw), P(away)]
                - lambda_home: predicted home λ
                - lambda_away: predicted away λ
                - top_scores: list of (home, away, prob) top 5 most likely scores
                - features: dict of computed features
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(match_date, str):
            match_date = pd.Timestamp(match_date)

        # Build feature row
        features = self.build_feature_row(
            home_team, away_team, league, season, match_date, week, odds=odds)

        # Create DataFrame for prediction
        feat_df = pd.DataFrame([features])

        # Predict λ
        lh, la = self.model.predict_lambdas(feat_df)
        lh, la = lh[0], la[0]

        # Build 7x7 score matrix
        matrix = self.model.predict_score_matrix(lh, la, max_goals)

        # 1x2 probabilities
        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()
        probs_1x2 = np.array([p_home, p_draw, p_away])
        probs_1x2 /= probs_1x2.sum()

        # Top most likely scores (overall)
        flat = matrix.flatten()
        top_idx = flat.argsort()[::-1][:5]
        top_scores = []
        for idx in top_idx:
            i, j = divmod(idx, max_goals)
            top_scores.append((int(i), int(j), float(flat[idx])))

        # 3 score groups: most likely home win, draw, away win
        score_groups = _build_score_groups(matrix, max_goals)

        # Recommended score: top score from predicted outcome group
        outcome_idx = int(np.argmax(probs_1x2))  # 0=home, 1=draw, 2=away
        group_map = {0: "home_win", 1: "draw", 2: "away_win"}
        pred_group = next(
            g for g in score_groups if g["type"] == group_map[outcome_idx])
        rec = pred_group["top_scores"][0]
        recommended_score = {
            "home_goals": rec[0],
            "away_goals": rec[1],
            "prob": rec[2],
            "label": f"{rec[0]}-{rec[1]}",
        }

        return {
            "score_matrix": matrix,
            "probs_1x2": probs_1x2,
            "predicted_outcome": ["home_win", "draw", "away_win"][outcome_idx],
            "recommended_score": recommended_score,
            "lambda_home": float(lh),
            "lambda_away": float(la),
            "top_scores": top_scores,
            "score_groups": score_groups,
            "features": features,
        }

    def predict_batch(self, matches, max_goals=7):
        """Predict multiple matches.

        Args:
            matches: list of dicts with keys:
                home_team, away_team, league, season, match_date, week (optional)

        Returns:
            list of prediction dicts
        """
        results = []
        for m in matches:
            result = self.predict(
                m["home_team"], m["away_team"],
                m["league"], m["season"],
                m["match_date"],
                week=m.get("week"),
                max_goals=max_goals,
                odds=m.get("odds"),
            )
            result["match_info"] = m
            results.append(result)
        return results


def _build_score_groups(matrix, max_goals=7):
    """Build 3 score prediction groups: home win, draw, away win.

    For each outcome type, returns the top score and its probability,
    plus the aggregate probability for that outcome type.
    """
    groups = []

    # Home win scores (i > j)
    home_scores = []
    for i in range(max_goals):
        for j in range(i):
            home_scores.append((i, j, matrix[i, j]))
    home_scores.sort(key=lambda x: -x[2])
    top_home = home_scores[:3] if home_scores else []
    home_total = sum(s[2] for s in home_scores)
    groups.append({
        "type": "home_win",
        "label": "主胜",
        "total_prob": home_total,
        "top_scores": [(s[0], s[1], s[2]) for s in top_home],
    })

    # Draw scores (i == j)
    draw_scores = [(i, i, matrix[i, i]) for i in range(max_goals)]
    draw_scores.sort(key=lambda x: -x[2])
    top_draw = draw_scores[:3]
    draw_total = sum(s[2] for s in draw_scores)
    groups.append({
        "type": "draw",
        "label": "平局",
        "total_prob": draw_total,
        "top_scores": [(s[0], s[1], s[2]) for s in top_draw],
    })

    # Away win scores (i < j)
    away_scores = []
    for i in range(max_goals):
        for j in range(i + 1, max_goals):
            away_scores.append((i, j, matrix[i, j]))
    away_scores.sort(key=lambda x: -x[2])
    top_away = away_scores[:3] if away_scores else []
    away_total = sum(s[2] for s in away_scores)
    groups.append({
        "type": "away_win",
        "label": "客胜",
        "total_prob": away_total,
        "top_scores": [(s[0], s[1], s[2]) for s in top_away],
    })

    return groups


def format_prediction(result, show_matrix=True):
    """Format a prediction result for display."""
    info = result.get("match_info", {})
    home = info.get("home_team", "Home")
    away = info.get("away_team", "Away")
    p = result["probs_1x2"]
    lh = result["lambda_home"]
    la = result["lambda_away"]

    lines = []
    lines.append(f"  {home} vs {away}")
    lines.append(f"  λ_home={lh:.2f}  λ_away={la:.2f}")
    lines.append(f"  主胜 {p[0]:.1%}  |  平局 {p[1]:.1%}  |  客胜 {p[2]:.1%}")

    # Most likely outcome + recommended score
    outcomes = ["主胜", "平局", "客胜"]
    pred_idx = np.argmax(p)
    rec = result.get("recommended_score", {})
    rec_label = rec.get("label", "")
    rec_prob = rec.get("prob", 0)
    lines.append(
        f"  预测结果: {outcomes[pred_idx]}  |  推荐比分: {rec_label} ({rec_prob:.1%})"
    )

    # 3 score groups
    groups = result.get("score_groups", [])
    if groups:
        lines.append(f"\n  三组比分预测:")
        for g in groups:
            label = g["label"]
            total = g["total_prob"]
            top = g["top_scores"]
            scores_str = ", ".join(f"{s[0]}-{s[1]}({s[2]:.1%})" for s in top)
            lines.append(f"    {label} [{total:.1%}]: {scores_str}")
    else:
        lines.append(f"  最可能比分:")
        for h, a, prob in result["top_scores"]:
            lines.append(f"    {h}-{a}: {prob:.1%}")

    if show_matrix:
        matrix = result["score_matrix"]
        lines.append(f"\n  7×7 比分概率矩阵 (%):")
        lines.append(f"  {'':>4}" + "".join(f"{j:>6}" for j in range(7)))
        for i in range(7):
            row_str = f"  {i:>3}|"
            for j in range(7):
                val = matrix[i, j] * 100
                if val >= 1:
                    row_str += f"{val:>5.1f}%"
                elif val >= 0.1:
                    row_str += f"{val:>5.2f}%"
                else:
                    row_str += f"    - "
            lines.append(row_str)

    return "\n".join(lines)
