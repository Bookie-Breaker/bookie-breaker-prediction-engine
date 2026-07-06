"""Collect historical MLB game data for model training (verification session).

NEVER run in CI: MLB StatsAPI (statsapi.mlb.com) is free and documented with
no key or hard rate limit, but it is MLB's production infrastructure -- be a
good citizen and run manually with the default sleeps, not from automation.

Requires the train extra: uv sync --extra train

Usage:
    uv run python scripts/collect_mlb_data.py --seasons 2024 2025 \
        --out data/mlb_2024_2025.parquet

Fetches the regular-season schedule per season with linescore and probable
pitcher hydrates, one month at a time. Output is one row per team per
completed game (date, season, teams, runs, result, probable pitchers as
announced), consistent with scripts/collect_soccer_data.py conventions:
running team stats and starter FIP/ERA are reconstructable in training prep,
and building the full training parquet (features + simulation probabilities
per market) additionally requires replaying games through the
simulation-engine and feature builder; see the roadmap's verification
session notes.

NCAA_BSB has no collector this wave: the league is dormant in BookieBreaker
(no schedule ingestion downstream) and college baseball has no comparably
documented free stats feed. The pooled BASEBALL model trains on MLB rows
with league_is_mlb=1 until that changes (ADR-026).
"""

import argparse
import time
from pathlib import Path
from typing import Any

import httpx

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_SPORT_ID = 1
SEASON_MONTHS = range(3, 12)  # March through November covers every MLB season


def _month_ranges(season: int) -> list[tuple[str, str]]:
    ranges = []
    for month in SEASON_MONTHS:
        last_day = 30 if month in (4, 6, 9, 11) else (28 if month == 2 else 31)
        ranges.append((f"{season}-{month:02d}-01", f"{season}-{month:02d}-{last_day}"))
    return ranges


def _pitcher(side: dict[str, Any]) -> tuple[str, str]:
    probable = side.get("probablePitcher") or {}
    return str(probable.get("fullName", "")), str(probable.get("id", "") or "")


def _team_rows(game: dict[str, Any]) -> list[dict[str, Any]]:
    if game.get("status", {}).get("codedGameState") != "F":
        return []
    teams = game.get("teams") or {}
    home, away = teams.get("home"), teams.get("away")
    if not home or not away or home.get("score") is None or away.get("score") is None:
        return []

    home_pitcher, home_pitcher_id = _pitcher(home)
    away_pitcher, away_pitcher_id = _pitcher(away)
    shared = {
        "date": str(game.get("officialDate") or game.get("gameDate", ""))[:10],
        "league": "MLB",
        "season": int(game.get("season", 0)),
        "game_type": str(game.get("gameType", "")),
        "game_pk": game.get("gamePk"),
        "day_night": str(game.get("dayNight", "")),
    }
    rows = []
    for side, opponent, pitchers, is_home in (
        (home, away, (home_pitcher, home_pitcher_id, away_pitcher, away_pitcher_id), True),
        (away, home, (away_pitcher, away_pitcher_id, home_pitcher, home_pitcher_id), False),
    ):
        runs_for, runs_against = int(side["score"]), int(opponent["score"])
        pitcher, pitcher_id, opp_pitcher, opp_pitcher_id = pitchers
        rows.append(
            {
                **shared,
                "team": str(side.get("team", {}).get("name", "")),
                "opponent": str(opponent.get("team", {}).get("name", "")),
                "is_home": is_home,
                "runs_for": runs_for,
                "runs_against": runs_against,
                "result": "W" if runs_for > runs_against else "L",
                "probable_pitcher": pitcher,
                "probable_pitcher_id": pitcher_id,
                "opponent_probable_pitcher": opp_pitcher,
                "opponent_probable_pitcher_id": opp_pitcher_id,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, required=True, help="Seasons to fetch, e.g. 2024 2025")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--game-types", nargs="+", default=["R"], help="MLB game types (R regular, P postseason)")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between StatsAPI requests")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install the train extra first: uv sync --extra train") from exc

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0) as client:
        for season in args.seasons:
            for start, end in _month_ranges(season):
                print(f"fetching MLB {start}..{end}...")
                response = client.get(
                    SCHEDULE_URL,
                    params={
                        "sportId": MLB_SPORT_ID,
                        "startDate": start,
                        "endDate": end,
                        "gameTypes": ",".join(args.game_types),
                        "hydrate": "probablePitcher,linescore",
                    },
                )
                response.raise_for_status()
                for day in response.json().get("dates", []):
                    for game in day.get("games", []):
                        rows.extend(_team_rows(game))
                time.sleep(args.sleep)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)
    print(f"wrote {len(frame)} team-game rows to {args.out}")


if __name__ == "__main__":
    main()
