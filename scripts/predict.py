"""Predict upcoming football matches.

Fetches upcoming fixtures from Opta API (or uses manual input),
trains the model on latest data, and outputs 7x7 score matrix predictions.

Usage:
    source .venv/bin/activate

    # Predict specific matches
    python scripts/predict.py --matches "Arsenal vs Chelsea" "Liverpool vs Man City"

    # Predict next matchday for a league
    python scripts/predict.py --league EPL --fetch

    # Predict from a JSON file
    python scripts/predict.py --file fixtures.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference import MatchPredictor, format_prediction

FEATURES_PATH = Path("data/opta/processed/features.csv")

# Team name aliases (common short names → full Opta names)
TEAM_ALIASES = {
    # EPL
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "villa": "Aston Villa",
    "bournemouth": "AFC Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton & Hove Albion",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "ipswich": "Ipswich Town",
    "leicester": "Leicester City",
    "liverpool": "Liverpool",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester united": "Manchester United",
    "newcastle": "Newcastle United",
    "nottingham forest": "Nottingham Forest",
    "forest": "Nottingham Forest",
    "southampton": "Southampton",
    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "west ham": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    # La Liga
    "barcelona": "Barcelona",
    "fc barcelona": "Barcelona",
    "real madrid": "Real Madrid",
    "atletico madrid": "Atlético de Madrid",
    "atletico": "Atlético de Madrid",
    "sevilla": "Sevilla",
    "real sociedad": "Real Sociedad",
    "villarreal": "Villarreal",
    "athletic bilbao": "Athletic Club",
    "betis": "Real Betis",
    "real betis": "Real Betis",
    # Serie A
    "inter": "Inter",
    "inter milan": "Inter",
    "ac milan": "AC Milan",
    "milan": "AC Milan",
    "juventus": "Juventus",
    "napoli": "Napoli",
    "roma": "Roma",
    "lazio": "Lazio",
    "atalanta": "Atalanta",
    "fiorentina": "Fiorentina",
    # Bundesliga
    "bayern": "Bayern München",
    "bayern munich": "Bayern München",
    "bayern munchen": "Bayern München",
    "dortmund": "Borussia Dortmund",
    "leverkusen": "Bayer Leverkusen",
    "leipzig": "RB Leipzig",
    "rb leipzig": "RB Leipzig",
    "frankfurt": "Eintracht Frankfurt",
    # Ligue 1
    "psg": "Paris Saint-Germain",
    "marseille": "Marseille",
    "lyon": "Lyon",
    "monaco": "Monaco",
    "lille": "Lille",
}


def resolve_team_name(name, known_teams):
    """Resolve a team name to its canonical Opta name."""
    # Direct match
    if name in known_teams:
        return name

    # Alias match
    lower = name.lower().strip()
    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]

    # Fuzzy match (substring)
    matches = [t for t in known_teams if lower in t.lower()]
    if len(matches) == 1:
        return matches[0]

    # Try reverse substring
    matches = [t for t in known_teams if t.lower() in lower]
    if len(matches) == 1:
        return matches[0]

    return name  # Return as-is, will fail gracefully later


def detect_league(home_team, away_team, df):
    """Auto-detect league from team names."""
    for team in [home_team, away_team]:
        mask = (df["home_team"] == team) | (df["away_team"] == team)
        matches = df[mask]
        if len(matches) > 0:
            return matches.iloc[-1]["league"]
    return None


def parse_match_string(match_str, known_teams, df):
    """Parse 'Team A vs Team B' into match info dict."""
    # Split on 'vs', 'v', or '-'
    for sep in [" vs ", " v ", " - "]:
        if sep in match_str:
            parts = match_str.split(sep, 1)
            home = resolve_team_name(parts[0].strip(), known_teams)
            away = resolve_team_name(parts[1].strip(), known_teams)

            league = detect_league(home, away, df)
            # Detect current season
            latest = df[df["league"] == league] if league else df
            season = latest["season"].iloc[-1] if len(latest) > 0 else "2025-2026"

            return {
                "home_team": home,
                "away_team": away,
                "league": league or "EPL",
                "season": season,
                "match_date": datetime.now().strftime("%Y-%m-%d"),
            }

    raise ValueError(f"无法解析比赛: '{match_str}'. 请使用格式: 'Team A vs Team B'")


def fetch_upcoming_fixtures(league, token=None):
    """Fetch upcoming fixtures from Opta API."""
    from scripts.opta_etl import extract_token, api_get, COMPETITIONS

    if token is None:
        print("  提取 Opta API token...")
        token = extract_token()

    comp_id = COMPETITIONS.get(league)
    if not comp_id:
        print(f"  未知联赛: {league}")
        return []

    # Get current season
    data = api_get("tournamentcalendar", token, params={"comp": comp_id})
    if not data or "competition" not in data:
        print(f"  无法获取赛季信息")
        return []

    # Find active season
    calendars = data["competition"][0].get("tournamentCalendar", [])
    active = [c for c in calendars if c.get("active") == "yes"]
    if not active:
        active = sorted(calendars, key=lambda x: x.get("startDate", ""))[-1:]

    if not active:
        return []

    season_id = active[0]["id"]
    season_name = active[0]["name"]
    print(f"  当前赛季: {season_name}")

    # Fetch match list (paginated to avoid 413 errors)
    all_matches = []
    for page in range(1, 5):
        time.sleep(0.5)
        match_data = api_get("match", token, params={
            "comp": comp_id,
            "tmcl": season_id,
            "_pgSz": "100",
            "_pgNm": str(page),
        })
        if not match_data or "match" not in match_data:
            break
        all_matches.extend(match_data["match"])
        if len(match_data["match"]) < 100:
            break

    match_data = {"match": all_matches} if all_matches else None

    if not match_data or "match" not in match_data:
        print("  无法获取赛程")
        return []

    fixtures = []
    today = datetime.now().strftime("%Y-%m-%d")
    safe_season = season_name.replace("/", "-")

    for m in match_data["match"]:
        match_info = m.get("matchInfo", {})
        live_data = m.get("liveData", {})
        match_details = live_data.get("matchDetails", {})

        # Only unplayed matches
        status = match_details.get("matchStatus", "")
        if status in ("Played", "FullTime"):
            continue

        match_date = match_info.get("date", "").replace("Z", "")
        if match_date < today:
            continue

        contestants = match_info.get("contestant", [])
        home_team = next(
            (c for c in contestants if c.get("position") == "home"), {})
        away_team = next(
            (c for c in contestants if c.get("position") == "away"), {})

        fixtures.append({
            "home_team": home_team.get("name", ""),
            "away_team": away_team.get("name", ""),
            "league": league,
            "season": safe_season,
            "match_date": match_date[:10],  # YYYY-MM-DD only
            "week": match_info.get("week", ""),
        })

    # Sort by date
    fixtures.sort(key=lambda x: x["match_date"])
    return fixtures


def run():
    parser = argparse.ArgumentParser(description="TikaML 比赛预测")
    parser.add_argument("--matches", nargs="+",
                        help="比赛列表 (格式: 'Team A vs Team B')")
    parser.add_argument("--league", choices=["EPL", "LL", "SEA", "BUN", "LI1"],
                        help="联赛代码 (配合 --fetch 使用)")
    parser.add_argument("--fetch", action="store_true",
                        help="从 Opta API 获取未来赛程")
    parser.add_argument("--file", type=str,
                        help="从 JSON 文件读取比赛列表")
    parser.add_argument("--date", type=str, default=None,
                        help="比赛日期 (默认: 今天)")
    parser.add_argument("--odds", nargs=3, type=float, metavar=("H", "D", "A"),
                        help="十进制赔率 (如: 2.10 3.40 3.50)")
    parser.add_argument("--no-matrix", action="store_true",
                        help="不显示 7×7 矩阵")
    parser.add_argument("--limit", type=int, default=10,
                        help="最多预测几场 (默认: 10)")

    args = parser.parse_args()

    print("=" * 70)
    print("TikaML: 足球比赛预测系统")
    print("=" * 70)

    # Initialize predictor
    predictor = MatchPredictor()

    # Train model
    print("\n训练模型...")
    predictor.train()

    # Get known teams
    known_teams = set(predictor.df["home_team"].unique()) | \
                  set(predictor.df["away_team"].unique())

    # Build fixture list
    fixtures = []

    if args.fetch and args.league:
        print(f"\n获取 {args.league} 未来赛程...")
        fixtures = fetch_upcoming_fixtures(args.league)
        if fixtures:
            print(f"  找到 {len(fixtures)} 场未来比赛")
        else:
            print("  未找到未来比赛")
            return

    elif args.file:
        with open(args.file) as f:
            fixtures = json.load(f)
        print(f"\n从文件加载 {len(fixtures)} 场比赛")

    elif args.matches:
        for match_str in args.matches:
            try:
                fixture = parse_match_string(match_str, known_teams,
                                             predictor.df)
                if args.date:
                    fixture["match_date"] = args.date
                fixtures.append(fixture)
            except ValueError as e:
                print(f"  错误: {e}")

    else:
        # Demo mode: predict some interesting matches
        print("\n演示模式: 预测示例比赛")
        match_date = args.date or datetime.now().strftime("%Y-%m-%d")
        fixtures = [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "league": "EPL",
                "season": "2025-2026",
                "match_date": match_date,
            },
            {
                "home_team": "Liverpool",
                "away_team": "Manchester City",
                "league": "EPL",
                "season": "2025-2026",
                "match_date": match_date,
            },
            {
                "home_team": "Barcelona",
                "away_team": "Real Madrid",
                "league": "LL",
                "season": "2025-2026",
                "match_date": match_date,
            },
            {
                "home_team": "Bayern München",
                "away_team": "Borussia Dortmund",
                "league": "BUN",
                "season": "2025-2026",
                "match_date": match_date,
            },
            {
                "home_team": "Inter",
                "away_team": "AC Milan",
                "league": "SEA",
                "season": "2025-2026",
                "match_date": match_date,
            },
        ]

    # Limit
    fixtures = fixtures[:args.limit]

    # Predict
    print(f"\n{'─' * 70}")
    print(f"预测 {len(fixtures)} 场比赛")
    print(f"{'─' * 70}")

    for i, fixture in enumerate(fixtures):
        home = fixture["home_team"]
        away = fixture["away_team"]
        league = fixture["league"]
        season = fixture.get("season", "2025-2026")
        match_date = fixture.get("match_date",
                                 datetime.now().strftime("%Y-%m-%d"))
        week = fixture.get("week")

        print(f"\n{'━' * 60}")
        print(f"  [{i+1}/{len(fixtures)}] {home} vs {away}")
        print(f"  {league} | {match_date}" +
              (f" | 第{week}轮" if week else ""))
        print(f"{'━' * 60}")

        try:
            # Build odds dict if provided
            odds = fixture.get("odds")
            if odds is None and args.odds and len(fixtures) == 1:
                odds = {"home": args.odds[0], "draw": args.odds[1],
                        "away": args.odds[2]}

            result = predictor.predict(
                home, away, league, season, match_date,
                week=int(week) if week else None,
                odds=odds)
            result["match_info"] = fixture
            print(format_prediction(result, show_matrix=not args.no_matrix))
        except Exception as e:
            print(f"  预测失败: {e}")

    # Summary
    print(f"\n{'=' * 70}")
    print("预测完成")
    print("=" * 70)
    print("注意: 预测基于历史统计数据，不包含伤病/阵容/转会等实时信息。")
    print("      实际比赛结果受多种因素影响，请谨慎参考。")


if __name__ == "__main__":
    run()
