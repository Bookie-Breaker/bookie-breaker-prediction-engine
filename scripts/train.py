"""Train the NBA adjustment model and save (optionally register) the artifact.

Usage:
    uv run python scripts/train.py --synthetic --out ./models
    uv run python scripts/train.py --data data/nba_training.parquet --out /models \
        --register-db "$DATABASE_URL"

The synthetic path is the Phase 2 bootstrap (see core/training/synthetic.py);
real data comes from scripts/collect_nba_data.py once graded outcomes exist.
Registering requires the predictions schema to be migrated (alembic).
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prediction_engine.core.training.synthetic import generate_synthetic_dataset  # noqa: E402
from prediction_engine.core.training.train import save_artifact, train_model  # noqa: E402


def load_parquet_dataset(path: Path):  # type: ignore[no-untyped-def]
    """Load a collected real-data training set (requires the train extra)."""
    import numpy as np
    import pandas as pd

    from prediction_engine.core.features.registry import NBA_FEATURES
    from prediction_engine.core.training.dataset import TrainingSet

    frame = pd.read_parquet(path)
    required = {"sim_probability", "outcome", "season"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"training parquet is missing columns: {sorted(missing)}")
    features = [
        {name: (None if pd.isna(row.get(name)) else float(row[name])) for name in NBA_FEATURES}
        for _, row in frame.iterrows()
    ]
    return TrainingSet(
        features=features,
        sim_probs=frame["sim_probability"].to_numpy(dtype=np.float64),
        outcomes=frame["outcome"].to_numpy(dtype=np.float64),
        seasons=frame["season"].to_numpy(dtype=np.int64),
    )


async def register(database_url: str, artifact_dir: Path) -> None:
    from prediction_engine.core.model.artifact import ArtifactBundle
    from prediction_engine.db.engine import create_engine
    from prediction_engine.db.repository import ModelVersionRepository

    bundle = ArtifactBundle.load(artifact_dir)
    engine = create_engine(database_url)
    repo = ModelVersionRepository(engine)
    trained_at = datetime.fromisoformat(str(bundle.metadata["trained_at"]))
    records = await repo.register(
        sport="BASKETBALL",
        market_types=["SPREAD", "TOTAL", "MONEYLINE"],
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
    parser.add_argument("--sport", default="BASKETBALL", choices=["BASKETBALL"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Train the synthetic bootstrap model")
    source.add_argument("--data", type=Path, help="Parquet file from scripts/collect_nba_data.py")
    parser.add_argument("--out", type=Path, default=Path("./models"), help="MODEL_DIR to write the artifact into")
    parser.add_argument("--rounds", type=int, default=200, help="Boosting rounds")
    parser.add_argument("--register-db", metavar="DATABASE_URL", help="Register + activate in this database")
    args = parser.parse_args()

    dataset = generate_synthetic_dataset() if args.synthetic else load_parquet_dataset(args.data)
    label = "synthetic" if args.synthetic else f"real:{args.data.name}"

    started = datetime.now(tz=UTC)
    result = train_model(dataset, n_rounds=args.rounds, data_label=label)
    artifact_dir = save_artifact(result, args.out)
    elapsed = (datetime.now(tz=UTC) - started).total_seconds()

    print(f"trained {result.bundle.version_tag} on {result.training_samples} rows in {elapsed:.1f}s")
    print(f"metrics: {result.metrics}")
    print(f"artifact: {artifact_dir}")

    if args.register_db:
        asyncio.run(register(args.register_db, artifact_dir))


if __name__ == "__main__":
    main()
