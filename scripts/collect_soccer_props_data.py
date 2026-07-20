"""Collect historical soccer player-prop data for prop model training (SKELETON).

NEVER run in CI: like the other collect scripts, the intended sources are
undocumented/keyless site APIs (ESPN match summaries for shots and scorer
data) that must be hit manually with generous sleeps.

STATUS: documented placeholder (Phase 7 Wave 3). Real prop data collection
is deferred to a verification session: building the training parquet needs
graded player-stat lines (which The Odds API only exposes on a paid
historical plan) plus per-player simulation over-probabilities replayed
through the simulation-engine's player capture. Until both exist, the
SOCCER prop model ships as the synthetic bootstrap
(scripts/train.py --sport SOCCER --market player_prop --synthetic).

Planned output (one row per player per stat per completed match, mirroring
the game collect scripts' shape):
    date, competition, player_external_id, player_name, team,
    stat_type (player_goal_scorer_anytime | player_shots |
    player_shots_on_target), line, actual, outcome, season

Usage (once implemented):
    uv run python scripts/collect_soccer_props_data.py \
        --competitions eng.1 --start 2025-08-01 --end 2026-05-31 \
        --out data/soccer_props_epl.parquet
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competitions", nargs="*", default=["fifa.world", "eng.1"])
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", type=Path, help="Output parquet path")
    parser.parse_args()
    raise SystemExit(
        "collect_soccer_props_data.py is a Phase 7 Wave 3 skeleton: real prop data "
        "collection is deferred to a verification session (see the module docstring). "
        "Train the synthetic bootstrap instead: "
        "uv run python scripts/train.py --sport SOCCER --market player_prop --synthetic"
    )


if __name__ == "__main__":
    main()
