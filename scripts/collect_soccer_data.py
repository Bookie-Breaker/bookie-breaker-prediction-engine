"""Collect historical soccer match data for model training (verification session).

NEVER run in CI: ESPN's site API is undocumented and keyless; hammer it from
CI and it will start returning garbage or blocking. Run manually with
generous sleeps.

Requires the train extra: uv sync --extra train

Usage:
    uv run python scripts/collect_soccer_data.py \
        --start 2022-11-20 --end 2022-12-18 --out data/soccer_wc2022.parquet
    uv run python scripts/collect_soccer_data.py \
        --competitions eng.1 --start 2024-08-01 --end 2025-05-31 \
        --out data/soccer_epl.parquet

Pools competitions (ADR-026: one SOCCER model, competition as a feature).
The intended first corpus is honest and small: the 2026 World Cup group
stage (live), the 2022 World Cup, and recent EPL seasons. Output is one row
per team per completed match (date, competition, teams, goals, result,
round text), from which running form and attack/defense strengths are
reconstructable in training prep. Building the full training parquet
(features + simulation probabilities per market) additionally requires
replaying matches through the simulation-engine and feature builder; see
the roadmap's verification session notes.
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"
DEFAULT_COMPETITIONS = ["fifa.world", "eng.1"]
CHUNK_DAYS = 28  # ESPN accepts dates=YYYYMMDD-YYYYMMDD ranges


def _date_chunks(start: date, end: date) -> list[str]:
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end)
        chunks.append(f"{cursor:%Y%m%d}-{chunk_end:%Y%m%d}")
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _round_text(competition: dict[str, Any]) -> str:
    notes = competition.get("notes") or []
    if notes and isinstance(notes[0], dict):
        return str(notes[0].get("headline", ""))
    return ""


def _team_rows(event: dict[str, Any], code: str) -> list[dict[str, Any]]:
    competition = (event.get("competitions") or [{}])[0]
    if not competition.get("status", event.get("status", {})).get("type", {}).get("completed", False):
        return []
    competitors = competition.get("competitors") or []
    if len(competitors) != 2:
        return []
    by_side = {c.get("homeAway"): c for c in competitors}
    home, away = by_side.get("home"), by_side.get("away")
    if home is None or away is None:
        return []

    def score(competitor: dict[str, Any]) -> int:
        try:
            return int(float(competitor.get("score", 0)))
        except (TypeError, ValueError):
            return 0

    home_goals, away_goals = score(home), score(away)
    shared = {
        "date": str(event.get("date", ""))[:10],
        "competition": code,
        "season": event.get("season", {}).get("year"),
        "season_type": event.get("season", {}).get("type"),
        "round_text": _round_text(competition),
        "event_id": event.get("id"),
    }
    rows = []
    for competitor, opponent, is_home in ((home, away, True), (away, home, False)):
        goals_for = home_goals if is_home else away_goals
        goals_against = away_goals if is_home else home_goals
        rows.append(
            {
                **shared,
                "team": competitor.get("team", {}).get("displayName", ""),
                "opponent": opponent.get("team", {}).get("displayName", ""),
                "is_home": is_home,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "result": "W" if goals_for > goals_against else ("D" if goals_for == goals_against else "L"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competitions", nargs="+", default=DEFAULT_COMPETITIONS, help="ESPN league codes")
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds between ESPN requests")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install the train extra first: uv sync --extra train") from exc

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0) as client:
        for code in args.competitions:
            for chunk in _date_chunks(args.start, args.end):
                print(f"fetching {code} {chunk}...")
                response = client.get(SCOREBOARD_URL.format(code=code), params={"dates": chunk, "limit": 400})
                response.raise_for_status()
                for event in response.json().get("events", []):
                    rows.extend(_team_rows(event, code))
                time.sleep(args.sleep)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)
    print(f"wrote {len(frame)} team-match rows to {args.out}")


if __name__ == "__main__":
    main()
