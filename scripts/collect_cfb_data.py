"""Collect historical NCAA_FB game data for model training (verification session).

Source: the College Football Data API (api.collegefootballdata.com), which
REQUIRES a free API key (Bearer token). Pass it with --api-key or via the
CFBD_API_KEY environment variable; the script fails with a clear message when
neither is set. NEVER run in CI: the key is a personal credential and the
service is rate-limited -- run manually during the verification session.

Requires the train extra: uv sync --extra train

Usage:
    uv run python scripts/collect_cfb_data.py --seasons 2023 2024 \
        --api-key "$CFBD_API_KEY" --out data/cfb_2023_2024.parquet

Output is one row per team per completed regular-season game (date, season,
week, teams, points, result), consistent with scripts/collect_nfl_data.py
conventions. SP+ ratings (the college analog of the NFL's EPA block) come
from the CFBD /ratings/sp endpoint and are joined in training prep. Building
the full training parquet (features + simulation probabilities per market)
additionally requires replaying games through the simulation-engine and
feature builder; see the roadmap's verification session notes.

FOOTBALL pools NFL + NCAA_FB (ADR-026): pair this with
scripts/collect_nfl_data.py and set league_is_nfl=0 for these rows.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import httpx

GAMES_URL = "https://api.collegefootballdata.com/games"


def _team_rows(game: dict[str, Any]) -> list[dict[str, Any]]:
    if not game.get("completed", False):
        return []
    home_points, away_points = game.get("home_points"), game.get("away_points")
    if home_points is None or away_points is None:
        return []
    shared = {
        "date": str(game.get("start_date", ""))[:10],
        "league": "NCAA_FB",
        "season": int(game.get("season", 0)),
        "game_type": str(game.get("season_type", "")),
        "week": game.get("week"),
        "game_id": game.get("id"),
    }
    rows = []
    for team, opponent, pf, pa, is_home in (
        (game.get("home_team"), game.get("away_team"), int(home_points), int(away_points), True),
        (game.get("away_team"), game.get("home_team"), int(away_points), int(home_points), False),
    ):
        rows.append(
            {
                **shared,
                "team": str(team or ""),
                "opponent": str(opponent or ""),
                "is_home": is_home,
                "points_for": pf,
                "points_against": pa,
                # college overtime always produces a winner: only W/L
                "result": "W" if pf > pa else "L",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, required=True, help="Seasons to fetch, e.g. 2023 2024")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key", default=None, help="CFBD API key (or set CFBD_API_KEY)")
    parser.add_argument("--season-type", default="regular", help="regular or postseason")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between CFBD requests")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise SystemExit(
            "a CFBD API key is required: pass --api-key or set CFBD_API_KEY "
            "(free key from https://collegefootballdata.com/key). This script never runs in CI."
        )

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install the train extra first: uv sync --extra train") from exc

    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0, headers=headers) as client:
        for season in args.seasons:
            print(f"fetching NCAA_FB {season} ({args.season_type})...")
            response = client.get(GAMES_URL, params={"year": season, "seasonType": args.season_type})
            response.raise_for_status()
            for game in response.json():
                rows.extend(_team_rows(game))
            time.sleep(args.sleep)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)
    print(f"wrote {len(frame)} team-game rows to {args.out}")


if __name__ == "__main__":
    main()
