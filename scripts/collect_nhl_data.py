"""Collect historical NHL game data for model training (verification session).

Source: the NHL's public web API (api-web.nhle.com), keyless and documented
by community reverse-engineering. NEVER run in CI: it is the league's
production infrastructure -- run manually with the default sleeps. The
per-team club-schedule-season endpoint returns a full season at once, so one
request per team per season covers everything; games are de-duplicated by id.

Requires the train extra: uv sync --extra train

Usage:
    uv run python scripts/collect_nhl_data.py --seasons 20232024 20242025 \
        --out data/nhl_2324_2425.parquet

Seasons use the NHL's 8-digit form (start year + end year, e.g. 20242025).
Output is one row per team per completed regular-season game (date, season,
teams, goals, result), consistent with scripts/collect_mlb_data.py
conventions: running goals-for/against, shots, save pct, and special-teams
rates are reconstructable in training prep (the boxscore endpoint carries
shots and special teams; this schedule pass captures scores and identity).
Building the full training parquet (features + simulation probabilities per
market) additionally requires replaying games through the simulation-engine
and feature builder; see the roadmap's verification session notes.

NCAA_HKY has no collector this wave: the league is gated on Odds API line
coverage (ADR-026), so the HOCKEY model is single-league NHL until it lands.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import httpx

SCHEDULE_URL = "https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
# The 32 current NHL club abbreviations (api-web.nhle.com uses these codes).
NHL_TEAMS = [
    "ANA",
    "BOS",
    "BUF",
    "CAR",
    "CBJ",
    "CGY",
    "CHI",
    "COL",
    "DAL",
    "DET",
    "EDM",
    "FLA",
    "LAK",
    "MIN",
    "MTL",
    "NJD",
    "NSH",
    "NYI",
    "NYR",
    "OTT",
    "PHI",
    "PIT",
    "SEA",
    "SJS",
    "STL",
    "TBL",
    "TOR",
    "UTA",
    "VAN",
    "VGK",
    "WPG",
    "WSH",
]
_COMPLETED_STATES = {"OFF", "FINAL"}


def _name(team: dict[str, Any]) -> str:
    place = (team.get("placeName") or {}).get("default", "")
    common = (team.get("commonName") or {}).get("default", "")
    return f"{place} {common}".strip() or str(team.get("abbrev", ""))


def _team_rows(game: dict[str, Any], regular_only: bool) -> list[dict[str, Any]]:
    if game.get("gameState") not in _COMPLETED_STATES:
        return []
    if regular_only and game.get("gameType") != 2:  # 2 = regular season, 3 = playoffs
        return []
    home, away = game.get("homeTeam") or {}, game.get("awayTeam") or {}
    if home.get("score") is None or away.get("score") is None:
        return []
    shared = {
        "date": str(game.get("gameDate", ""))[:10],
        "league": "NHL",
        "season": int(game.get("season", 0)),
        "game_type": int(game.get("gameType", 0)),
        "game_id": game.get("id"),
    }
    rows = []
    for team, opponent, is_home in ((home, away, True), (away, home, False)):
        goals_for, goals_against = int(team["score"]), int(opponent["score"])
        rows.append(
            {
                **shared,
                "team": _name(team),
                "team_abbrev": str(team.get("abbrev", "")),
                "opponent": _name(opponent),
                "is_home": is_home,
                "goals_for": goals_for,
                "goals_against": goals_against,
                # OT/SO always breaks a tie, so results are only W/L
                "result": "W" if goals_for > goals_against else "L",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons", nargs="+", type=int, required=True, help="8-digit NHL seasons, e.g. 20232024 20242025"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teams", nargs="+", default=NHL_TEAMS, help="Club abbreviations to fetch")
    parser.add_argument("--postseason", action="store_true", help="Include playoff games (default regular only)")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between NHL API requests")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install the train extra first: uv sync --extra train") from exc

    seen_game_ids: set[Any] = set()
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0) as client:
        for season in args.seasons:
            for team in args.teams:
                print(f"fetching NHL {team} {season}...")
                response = client.get(SCHEDULE_URL.format(team=team, season=season))
                response.raise_for_status()
                for game in response.json().get("games", []):
                    if game.get("id") in seen_game_ids:
                        continue  # each game appears on both clubs' schedules
                    seen_game_ids.add(game.get("id"))
                    rows.extend(_team_rows(game, regular_only=not args.postseason))
                time.sleep(args.sleep)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)
    print(f"wrote {len(frame)} team-game rows to {args.out}")


if __name__ == "__main__":
    main()
