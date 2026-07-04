"""Collect historical NBA game data for model training (verification session).

NEVER run in CI: stats.nba.com rate-limits aggressively and may block
datacenter IPs. Run from a residential connection with generous sleeps.

Requires the train extra: uv sync --extra train

Usage:
    uv run python scripts/collect_nba_data.py --seasons 2022-23 2023-24 2024-25 \
        --out data/nba_games.parquet

Output: one row per completed game with final scores and per-team season
context. Building the full training parquet (features + simulation
probabilities per market) additionally requires replaying games through the
simulation-engine and feature builder; see the roadmap's verification
session notes.
"""

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", required=True, help='Seasons like "2023-24"')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds between nba_api requests")
    args = parser.parse_args()

    try:
        import pandas as pd
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install the train extra first: uv sync --extra train") from exc

    frames = []
    for season in args.seasons:
        print(f"fetching {season} game logs...")
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        )
        frame = finder.get_data_frames()[0]
        frame["SEASON_LABEL"] = season
        frames.append(frame)
        time.sleep(args.sleep)

    combined = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out)
    print(f"wrote {len(combined)} team-game rows to {args.out}")


if __name__ == "__main__":
    main()
