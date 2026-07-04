"""Model registry: DB rows <-> disk artifacts, with in-memory caching.

Registry rows: the unified NBA model covers SPREAD/TOTAL/MONEYLINE with
market type as a feature, registered as three model_versions rows (one per
market type, satisfying the partial unique active index) sharing one
artifact_path.

ensure_bootstrap() makes a fresh deployment self-sufficient: if no active
model rows exist it registers the newest on-disk artifact, and if no
artifact exists either it trains the synthetic bootstrap in-process
(seconds; see core/training/synthetic.py).
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from prediction_engine.core.model.artifact import ArtifactBundle, find_latest_artifact
from prediction_engine.db.repository import ModelVersionRecord, ModelVersionRepository

logger = logging.getLogger(__name__)

MARKET_TYPES = ["SPREAD", "TOTAL", "MONEYLINE"]
SPORT = "BASKETBALL"


@dataclass
class LoadedModel:
    record: ModelVersionRecord
    bundle: ArtifactBundle


class ModelRegistry:
    def __init__(self, repo: ModelVersionRepository, model_dir: Path) -> None:
        self._repo = repo
        self._model_dir = model_dir
        self._active: dict[str, LoadedModel] = {}
        self._bootstrap_lock = asyncio.Lock()

    async def try_bootstrap(self) -> bool:
        """Non-fatal bootstrap for startup: the database may not be migrated
        yet when the container first starts (task up runs before
        db:migrate), so failure just leaves the registry empty and
        get_active retries lazily on first use."""
        try:
            await self.ensure_bootstrap()
        except Exception:  # noqa: BLE001 - degraded start beats a crash loop
            logger.warning("model bootstrap failed at startup; will retry on first request", exc_info=True)
            return False
        return True

    async def get_active(self, market_type: str) -> LoadedModel | None:
        if not self._active:
            async with self._bootstrap_lock:
                if not self._active:
                    await self.ensure_bootstrap()
        return self._active.get(market_type)

    async def ensure_bootstrap(self) -> None:
        active = await self._repo.active_models()
        if not active:
            artifact_dir = find_latest_artifact(self._model_dir)
            if artifact_dir is None:
                logger.warning("no model artifact found under %s; training synthetic bootstrap", self._model_dir)
                artifact_dir = self._train_bootstrap()
            bundle = ArtifactBundle.load(artifact_dir)
            trained_at = datetime.fromisoformat(str(bundle.metadata["trained_at"]))
            active = await self._repo.register(
                sport=SPORT,
                market_types=MARKET_TYPES,
                version=bundle.version_tag,
                trained_at=trained_at,
                training_range=(trained_at - timedelta(days=365 * 4), trained_at),
                training_samples=int(bundle.metadata.get("training_samples", 0)),
                evaluation_metrics=dict(bundle.metadata.get("metrics", {})),
                feature_names=bundle.feature_names,
                artifact_path=str(artifact_dir),
                notes=f"bootstrap ({bundle.metadata.get('data_label', 'unknown')} data)",
            )
            logger.info("registered bootstrap model %s for %s markets", bundle.version_tag, len(active))
        await self._load_active(active)

    def _train_bootstrap(self) -> Path:
        from prediction_engine.core.training.synthetic import generate_synthetic_dataset
        from prediction_engine.core.training.train import save_artifact, train_model

        result = train_model(generate_synthetic_dataset(), n_rounds=100, data_label="synthetic")
        return save_artifact(result, self._model_dir)

    async def _load_active(self, records: list[ModelVersionRecord]) -> None:
        bundles: dict[str, ArtifactBundle] = {}
        loaded: dict[str, LoadedModel] = {}
        for record in records:
            if record.artifact_path not in bundles:
                bundles[record.artifact_path] = ArtifactBundle.load(Path(record.artifact_path))
            loaded[record.model_type] = LoadedModel(record=record, bundle=bundles[record.artifact_path])
        self._active = loaded

    def active_for(self, market_type: str) -> LoadedModel | None:
        return self._active.get(market_type)

    def active_map(self) -> dict[str, str]:
        return {f"{model.record.sport}_{market}": str(model.record.id) for market, model in self._active.items()}
