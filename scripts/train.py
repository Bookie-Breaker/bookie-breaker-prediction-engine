"""Train a sport's adjustment model and save (optionally register) the artifact.

Usage:
    uv run python scripts/train.py --synthetic --out ./models
    uv run python scripts/train.py --data data/nba_training.parquet --out /models \
        --register-db "$DATABASE_URL"

The synthetic path is the Phase 2 bootstrap (see core/training/synthetic.py);
real data comes from scripts/collect_nba_data.py once graded outcomes exist.
Registering requires the predictions schema to be migrated (alembic).

Sports other than BASKETBALL become trainable in their league waves, when
their feature registry and synthetic generator are registered (ADR-026);
until then they fail with a clear registry error.

--market player_prop trains the sport's unified player-prop model instead
(Phase 7 Wave 3): prop feature registry, prop synthetic generator, "props"
artifact family, and a single PLAYER_PROP model_versions row when
registering.
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prediction_engine.core.training.synthetic import (  # noqa: E402
    get_prop_synthetic_generator,
    get_synthetic_generator,
)
from prediction_engine.core.training.train import save_artifact, train_model  # noqa: E402

# Model keys: the five sports plus NCAA_BB, which trains its own
# single-league model (see core/leagues.py).
SPORTS = ["BASKETBALL", "FOOTBALL", "BASEBALL", "SOCCER", "HOCKEY", "NCAA_BB"]

# CLI market choice -> train_model market value.
MARKETS = {"game": "GAME", "player_prop": "PLAYER_PROP"}


def load_parquet_dataset(path: Path, sport: str, market: str = "GAME"):  # type: ignore[no-untyped-def]
    """Load a collected real-data training set (requires the train extra)."""
    import numpy as np
    import pandas as pd

    from prediction_engine.core.features.registry import get_features, get_prop_features
    from prediction_engine.core.training.dataset import TrainingSet

    feature_names = get_prop_features(sport) if market == "PLAYER_PROP" else get_features(sport)
    frame = pd.read_parquet(path)
    required = {"sim_probability", "outcome", "season"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"training parquet is missing columns: {sorted(missing)}")
    features = [
        {name: (None if pd.isna(row.get(name)) else float(row[name])) for name in feature_names}
        for _, row in frame.iterrows()
    ]
    return TrainingSet(
        features=features,
        sim_probs=frame["sim_probability"].to_numpy(dtype=np.float64),
        outcomes=frame["outcome"].to_numpy(dtype=np.float64),
        seasons=frame["season"].to_numpy(dtype=np.int64),
    )


async def register(database_url: str, artifact_dir: Path, sport: str, market: str = "GAME") -> None:
    from prediction_engine.core.model.artifact import ArtifactBundle
    from prediction_engine.db.engine import create_engine
    from prediction_engine.db.repository import ModelVersionRepository

    bundle = ArtifactBundle.load(artifact_dir)
    engine = create_engine(database_url)
    repo = ModelVersionRepository(engine)
    trained_at = datetime.fromisoformat(str(bundle.metadata["trained_at"]))
    records = await repo.register(
        sport=sport,
        market_types=["PLAYER_PROP"] if market == "PLAYER_PROP" else ["SPREAD", "TOTAL", "MONEYLINE"],
        version=bundle.version_tag,
        trained_at=trained_at,
        training_range=(trained_at - timedelta(days=365 * 4), trained_at),
        training_samples=int(bundle.metadata.get("training_samples", 0)),
        evaluation_metrics=dict(bundle.metadata.get("metrics", {})),
        feature_names=bundle.feature_names,
        artifact_path=str(artifact_dir),
        notes=f"trained via scripts/train.py ({bundle.metadata.get('data_label')})",
    )
    await engine.dispose()
    print(f"registered {len(records)} model_versions rows for {bundle.version_tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", default="BASKETBALL", choices=SPORTS)
    parser.add_argument(
        "--market",
        default="game",
        choices=sorted(MARKETS),
        help="Model family: 'game' (unified SPREAD/TOTAL/MONEYLINE) or 'player_prop' (unified prop model)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Train the synthetic bootstrap model")
    source.add_argument("--data", type=Path, help="Parquet file from a scripts/collect_*_data.py run")
    parser.add_argument("--out", type=Path, default=Path("./models"), help="MODEL_DIR to write the artifact into")
    parser.add_argument("--rounds", type=int, default=200, help="Boosting rounds")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Train the GBT + XGBoost-random-forest ensemble (Phase 7 Wave 4) instead of the single GBT",
    )
    parser.add_argument("--register-db", metavar="DATABASE_URL", help="Register + activate in this database")
    args = parser.parse_args()

    market = MARKETS[args.market]
    is_prop = market == "PLAYER_PROP"
    if args.synthetic:
        generator = get_prop_synthetic_generator(args.sport) if is_prop else get_synthetic_generator(args.sport)
        dataset = generator()
    else:
        dataset = load_parquet_dataset(args.data, args.sport, market)
    label = "synthetic" if args.synthetic else f"real:{args.data.name}"

    started = datetime.now(tz=UTC)
    result = train_model(
        dataset, n_rounds=args.rounds, data_label=label, sport=args.sport, market=market, ensemble=args.ensemble
    )
    artifact_dir = save_artifact(result, args.out)
    elapsed = (datetime.now(tz=UTC) - started).total_seconds()

    print(f"trained {result.bundle.version_tag} on {result.training_samples} rows in {elapsed:.1f}s")
    print(f"metrics: {result.metrics}")
    print(f"artifact: {artifact_dir}")

    if args.register_db:
        asyncio.run(register(args.register_db, artifact_dir, args.sport, market))


if __name__ == "__main__":
    main()
