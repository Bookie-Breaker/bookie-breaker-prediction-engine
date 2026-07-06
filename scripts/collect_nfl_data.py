"""Collect historical NFL game data for model training (verification session).

Source: the nflverse `games.csv` static release (keyless, versioned, CDN-
served). It is a plain file download, not a live API, so it is gentle to
fetch -- but keep it out of CI regardless: training-data collection belongs
to the manual verification session, not automation.

Requires the train extra: uv sync --extra train

Usage:
    uv run python scripts/collect_nfl_data.py --seasons 2023 2024 \
        --out data/nfl_2023_2024.parquet

Output is one row per team per completed regular-season game (date, season,
week, teams, points, result), consistent with scripts/collect_mlb_data.py
conventions: running team stats (points per game, points allowed, turnover
margin) are reconstructable in training prep. EPA and SP+ are NOT in this
file -- EPA comes from nflverse's play-by-play release (NFL only) and SP+
from CFBD (college only); both are joined in training prep. Building the
full training parquet (features + simulation probabilities per market)
additionally requires replaying games through the simulation-engine and
feature builder; see the roadmap's verification session notes.

FOOTBALL pools NFL + NCAA_FB (ADR-026): pair this with
scripts/collect_cfb_data.py and set league_is_nfl in training prep.
"""

import argparse
from pathlib import Path
from typing import Any

# nflverse publishes the canonical schedule/results file here (keyless, static).
GAMES_CSV_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"


def _team_rows(game: dict[str, Any]) -> list[dict[str, Any]]:
    home_score, away_score = game.get("home_score"), game.get("away_score")
    if home_score is None or away_score is None:
        return []  # not yet played
    shared = {
        "date": str(game.get("gameday", ""))[:10],
        "league": "NFL",
        "season": int(game.get("season", 0)),
        "game_type": str(game.get("game_type", "")),
        "week": game.get("week"),
        "game_id": game.get("game_id"),
    }
    rows = []
    for team, opponent, pf, pa, is_home in (
        (game.get("home_team"), game.get("away_team"), int(home_score), int(away_score), True),
        (game.get("away_team"), game.get("home_team"), int(away_score), int(home_score), False),
    ):
        rows.append(
            {
                **shared,
                "team": str(team or ""),
                "opponent": str(opponent or ""),
                "is_home": is_home,
                "points_for": pf,
                "points_against": pa,
                "result": "W" if pf > pa else ("T" if pf == pa else "L"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, required=True, help="Seasons to keep, e.g. 2023 2024")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--game-types", nargs="+", default=["REG"], help="nflverse game types (REG regular, POST postseason)"
    )
    parser.add_argument("--url", default=GAMES_CSV_URL, help="Override the nflverse games.csv location")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install the train extra first: uv sync --extra train") from exc

    print(f"downloading nflverse games.csv from {args.url}...")
    frame = pd.read_csv(args.url)
    wanted_seasons = set(args.seasons)
    wanted_types = set(args.game_types)

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        if int(record.get("season", 0)) not in wanted_seasons:
            continue
        if str(record.get("game_type", "")) not in wanted_types:
            continue
        rows.append(record)

    team_rows: list[dict[str, Any]] = []
    for game in rows:
        team_rows.extend(_team_rows(game))

    out_frame = pd.DataFrame(team_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_parquet(args.out)
    print(f"wrote {len(out_frame)} team-game rows to {args.out}")


if __name__ == "__main__":
    main()
