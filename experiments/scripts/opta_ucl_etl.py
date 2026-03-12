#!/usr/bin/env python3
"""ETL script for UEFA Champions League data via The Analyst WP proxy.

The performfeeds API token doesn't grant UCL access, so we use
The Analyst's WordPress REST API as a proxy (X-SDAPI-Token auth).

Usage:
    python experiments/scripts/opta_ucl_etl.py [--start-season 2016]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# ─── Config ────────────────────────────────────────────────────────────────

UCL_COMP_ID = "4oogyu6o156iphvdvphwpck10"
SDAPI_TOKEN = "LRkJ2MjwlC8RxUfVkne4"
BASE_URL = "https://theanalyst.com/wp-json/sdapi/v1"

RAW_DIR = Path("data/opta/raw")
LEAGUE_CODE = "UCL"

REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://theanalyst.com/competition/uefa-champions-league",
    "X-SDAPI-Token": SDAPI_TOKEN,
    "Accept": "application/json",
}

# ─── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ucl_etl")

# ─── API Client ────────────────────────────────────────────────────────────


def api_get(path, params=None):
    """Make a GET request to The Analyst WP proxy."""
    url = f"{BASE_URL}/{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{qs}"

    for attempt in range(MAX_RETRIES):
        try:
            req = Request(url, headers=HEADERS)
            resp = urlopen(req, timeout=30)
            return json.loads(resp.read())
        except HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                log.warning(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            body = ""
            try:
                body = e.read().decode()[:200]
            except Exception:
                pass
            log.error(f"  HTTP {e.code}: {body}")
            if e.code in (400, 401, 403, 404):
                return None
            time.sleep(1)
        except Exception as e:
            log.error(f"  Network error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                return None
    return None


# ─── Step 1: Fetch season list ─────────────────────────────────────────────


def fetch_seasons():
    """Get all UCL seasons from tournament calendar."""
    log.info("Phase 1: Fetching UCL tournament calendar")

    data = api_get("raw/soccerdata/tournamentcalendar", {"comp": UCL_COMP_ID})
    if not data:
        log.error("Failed to fetch tournament calendar!")
        sys.exit(1)

    comps = data.get("competition", [])
    if not isinstance(comps, list):
        comps = [comps]

    seasons = []
    for comp in comps:
        entries = comp.get("tournamentCalendar", [])
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            name = entry.get("name", "")
            sid = entry.get("id", "")
            start = entry.get("startDate", "")[:10]
            end = entry.get("endDate", "")[:10]
            # Convert "2024/2025" → "2024-2025" for directory naming
            safe_name = name.replace("/", "-")
            seasons.append({
                "id": sid,
                "name": name,
                "safe_name": safe_name,
                "start": start,
                "end": end,
            })

    seasons.sort(key=lambda s: s["start"])
    log.info(f"  Found {len(seasons)} UCL seasons")

    # Save calendar
    cal_dir = RAW_DIR / "tournamentcalendar"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_file = cal_dir / "ucl_seasons.json"
    with open(cal_file, "w") as f:
        json.dump(seasons, f, indent=2, ensure_ascii=False)

    return seasons


# ─── Step 2: Fetch match list per season ───────────────────────────────────


def fetch_match_list(season):
    """Get all match IDs for a season from tournament schedule."""
    tmcl = season["id"]
    safe_name = season["safe_name"]

    out_dir = RAW_DIR / "match" / LEAGUE_CODE / safe_name
    out_file = out_dir / "matches.json"

    if out_file.exists():
        with open(out_file) as f:
            cached = json.load(f)
        if cached:
            log.info(f"    [CACHED] {len(cached)} matches")
            return cached

    time.sleep(REQUEST_DELAY)
    data = api_get("soccerdata/tournamentschedule", {"tmcl": tmcl})
    if not data:
        log.warning(f"    No schedule data for {season['name']}")
        return []

    match_ids = []
    for md in data.get("matchDate", []):
        matches = md.get("match", [])
        if not isinstance(matches, list):
            matches = [matches]
        for m in matches:
            match_ids.append({
                "match_id": m.get("id", ""),
                "date": m.get("date", md.get("date", "")),
                "home_name": m.get("homeContestantName", ""),
                "away_name": m.get("awayContestantName", ""),
            })

    log.info(f"    Schedule: {len(match_ids)} matches")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(match_ids, f, indent=2, ensure_ascii=False)

    return match_ids


# ─── Step 3: Fetch matchstats per match ────────────────────────────────────


def fetch_matchstats(match_id, safe_season):
    """Fetch detailed matchstats for a single match."""
    out_dir = RAW_DIR / "matchstats" / LEAGUE_CODE / safe_season
    out_file = out_dir / f"{match_id}.json"

    if out_file.exists():
        return "cached"

    time.sleep(REQUEST_DELAY)
    data = api_get("raw/soccerdata/matchstats", {"fx": match_id})
    if not data:
        return "failed"

    # Verify it has actual match data (not just empty shell)
    live_data = data.get("liveData", {})
    match_details = live_data.get("matchDetails", {})

    # Check if match has been played
    if not match_details:
        return "unplayed"

    scores = match_details.get("scores", {})
    if not scores.get("ft") and not scores.get("total"):
        # Could be unplayed or have different score format
        match_status = match_details.get("matchStatus", "")
        if match_status not in ("Played", "FullTime"):
            return "unplayed"

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(data, f, ensure_ascii=False)

    return "ok"


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Fetch UCL matchstats from Opta via The Analyst")
    parser.add_argument("--start-season", type=int, default=2016,
                        help="Earliest season start year (default: 2016)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only fetch calendar and match lists, skip matchstats")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("UCL Data ETL Pipeline")
    log.info("=" * 60)

    # Step 1: Get season list
    seasons = fetch_seasons()

    # Filter by start season
    filtered = [s for s in seasons if int(s["name"][:4]) >= args.start_season]
    log.info(f"\n  Fetching {len(filtered)} seasons (from {args.start_season})")
    for s in filtered:
        log.info(f"    {s['name']:15s}  {s['start']} ~ {s['end']}  id={s['id'][:12]}...")

    # Step 2 & 3: For each season, get match list then matchstats
    total_fetched = 0
    total_cached = 0
    total_failed = 0
    total_unplayed = 0

    for season in filtered:
        log.info(f"\n{'='*60}")
        log.info(f"  [{season['name']}] Fetching matches...")

        match_list = fetch_match_list(season)
        if not match_list:
            continue

        if args.dry_run:
            log.info(f"    [DRY RUN] Would fetch {len(match_list)} matchstats")
            continue

        log.info(f"    Fetching matchstats for {len(match_list)} matches...")
        season_ok = 0
        season_cached = 0
        season_failed = 0
        season_unplayed = 0

        for i, m in enumerate(match_list):
            mid = m["match_id"]
            result = fetch_matchstats(mid, season["safe_name"])

            if result == "ok":
                season_ok += 1
            elif result == "cached":
                season_cached += 1
            elif result == "unplayed":
                season_unplayed += 1
            else:
                season_failed += 1

            # Progress every 20 matches
            done = i + 1
            if done % 20 == 0 or done == len(match_list):
                log.info(
                    f"    Progress: {done}/{len(match_list)} "
                    f"(new={season_ok}, cached={season_cached}, "
                    f"unplayed={season_unplayed}, failed={season_failed})"
                )

        total_fetched += season_ok
        total_cached += season_cached
        total_failed += season_failed
        total_unplayed += season_unplayed

    # Summary
    log.info(f"\n{'='*60}")
    log.info("ETL Complete!")
    log.info(f"  新下载: {total_fetched}")
    log.info(f"  已缓存: {total_cached}")
    log.info(f"  未进行: {total_unplayed}")
    log.info(f"  失败:   {total_failed}")
    log.info(f"  总计:   {total_fetched + total_cached + total_unplayed + total_failed}")

    # Count final files
    stats_dir = RAW_DIR / "matchstats" / LEAGUE_CODE
    if stats_dir.exists():
        total_files = sum(1 for _ in stats_dir.rglob("*.json"))
        log.info(f"\n  磁盘上 UCL matchstats 文件: {total_files}")


if __name__ == "__main__":
    main()
