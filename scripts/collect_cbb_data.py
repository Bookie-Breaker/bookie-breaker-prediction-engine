"""Collect historical NCAA_BB game data for model training (verification session).

Source: the College Basketball Data API (api.collegebasketballdata.com),
which REQUIRES a free API key (Bearer token). Pass it with --api-key or via
the CBBD_API_KEY environment variable; the script fails with a clear message
when neither is set. NEVER run in CI: the key is a personal credential and
the service is rate-limited -- run manually during the verification session.

Requires the train extra: uv sync --extra train

Usage:
    uv run python scripts/collect_cbb_data.py --seasons 2024 2025 \
        --api-key "$CBBD_API_KEY" --out data/cbb_2024_2025.parquet

Seasons use the ending calendar year (2025 = the 2024-25 season). Output is
one row per team per completed regular-season game (date, season, teams,
points, result), consistent with scripts/collect_nba_data.py conventions:
running ratings, pace, and recent form are reconstructable in training prep.
The opponent-adjusted efficiency margin (the NCAA_BB model's dominant signal)
comes from the CBBD /ratings endpoint and is joined in training prep.
Building the full training parquet (features + simulation probabilities per
market) additionally requires replaying games through the simulation-engine
and feature builder; see the roadmap's verification session notes.

NCAA_BB trains its own single-league model (ADR-026; not pooled with the NBA),
and ships dormant until the model_versions sport_enum gains the value and the
league enables at season start -- see core/leagues.py.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import httpx

GAMES_URL = "https://api.collegebasketballdata.com/games"


def _first(record: dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-None value among camelCase/snake_case keys."""
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def _team_rows(game: dict[str, Any]) -> list[dict[str, Any]]:
    status = str(_first(game, "status", "gameStatus") or "")
    home_points = _first(game, "homePoints", "home_points")
    away_points = _first(game, "awayPoints", "away_points")
    if home_points is None or away_points is None:
        return []  # unplayed; a completed game always carries both scores
    if status and status.lower() not in ("final", "f", "completed"):
        return []
    shared = {
        "date": str(_first(game, "startDate", "start_date", "startTime") or "")[:10],
        "league": "NCAA_BB",
        "season": int(_first(game, "season") or 0),
        "game_type": str(_first(game, "seasonType", "season_type") or ""),
        "game_id": _first(game, "id", "gameId"),
    }
    home_team = _first(game, "homeTeam", "home_team")
    away_team = _first(game, "awayTeam", "away_team")
    rows = []
    for team, opponent, pf, pa, is_home in (
        (home_team, away_team, int(home_points), int(away_points), True),
        (away_team, home_team, int(away_points), int(home_points), False),
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
    parser.add_argument("--seasons", nargs="+", type=int, required=True, help="Seasons (ending year), e.g. 2024 2025")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key", default=None, help="CBBD API key (or set CBBD_API_KEY)")
    parser.add_argument("--season-type", default="regular", help="regular or postseason")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between CBBD requests")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("CBBD_API_KEY")
    if not api_key:
        raise SystemExit(
            "a CBBD API key is required: pass --api-key or set CBBD_API_KEY "
            "(free key from https://collegebasketballdata.com/key). This script never runs in CI."
        )

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install the train extra first: uv sync --extra train") from exc

    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0, headers=headers) as client:
        for season in args.seasons:
            print(f"fetching NCAA_BB {season} ({args.season_type})...")
            response = client.get(GAMES_URL, params={"season": season, "seasonType": args.season_type})
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
