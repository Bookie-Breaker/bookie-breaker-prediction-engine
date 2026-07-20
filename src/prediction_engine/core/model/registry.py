"""Model registry: DB rows <-> disk artifacts, with in-memory caching.

Registry rows: each sport's unified model covers SPREAD/TOTAL/MONEYLINE
with market type as a feature, registered as three model_versions rows (one
per market type, satisfying the partial unique active index) sharing one
artifact_path. The in-memory active map is keyed by (sport, market_type).

ensure_bootstrap(sport) makes a fresh deployment self-sufficient: if no
active model rows exist for the sport it registers the newest on-disk
artifact under models/{sport_lowercase}/unified/, and if no artifact exists
either it trains the synthetic bootstrap in-process (seconds; see
core/training/synthetic.py). Basketball artifacts have lived at
models/basketball/unified/ since Phase 2 — the per-sport path scheme is
identical for them, so no legacy-path check is needed.

Player props (Phase 7 Wave 3): each sport's unified prop model covers all
of its prop stats (stat type is a feature) and registers as a single
model_versions row with model_type PLAYER_PROP, keyed (sport,
"PLAYER_PROP") in the active map. Prop artifacts live under
models/{sport}/props/ and bootstrap lazily on the first prop request
(ensure_prop_bootstrap), not at startup.

Champion/challenger (Phase 7 Wave 4): the in-memory map is keyed
(sport, market_type, role). Champions are the default serve path -- every
pre-Wave-4 accessor keeps its signature and behavior. Challengers never
bootstrap synthetically: get_challenger returns None when no active
challenger row exists (absence simply means no experiment is running).
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

PROP_MODEL_TYPE = "PLAYER_PROP"


@dataclass
class LoadedModel:
    record: ModelVersionRecord
    bundle: ArtifactBundle


class ModelRegistry:
    def __init__(self, repo: ModelVersionRepository, model_dir: Path, sports: list[str] | None = None) -> None:
        self._repo = repo
        self._model_dir = model_dir
        self._sports = list(sports) if sports is not None else ["BASKETBALL"]
        self._active: dict[tuple[str, str, str], LoadedModel] = {}
        self._bootstrap_lock = asyncio.Lock()

    async def try_bootstrap(self) -> bool:
        """Non-fatal bootstrap for startup: the database may not be migrated
        yet when the container first starts (task up runs before
        db:migrate), so failure just leaves the registry empty and
        get_active retries lazily on first use."""
        ok = True
        for sport in self._sports:
            try:
                await self.ensure_bootstrap(sport)
            except Exception:  # noqa: BLE001 - degraded start beats a crash loop
                logger.warning(
                    "model bootstrap for %s failed at startup; will retry on first request", sport, exc_info=True
                )
                ok = False
        return ok

    async def get_active(self, sport: str, market_type: str) -> LoadedModel | None:
        """The champion model for (sport, market_type), bootstrapping if needed."""
        if market_type == PROP_MODEL_TYPE:
            if (sport, PROP_MODEL_TYPE, "champion") not in self._active:
                async with self._bootstrap_lock:
                    if (sport, PROP_MODEL_TYPE, "champion") not in self._active:
                        await self.ensure_prop_bootstrap(sport)
            return self._active.get((sport, PROP_MODEL_TYPE, "champion"))
        if not self._has_sport(sport):
            async with self._bootstrap_lock:
                if not self._has_sport(sport):
                    await self.ensure_bootstrap(sport)
        return self._active.get((sport, market_type, "champion"))

    async def get_challenger(self, sport: str, market_type: str) -> LoadedModel | None:
        """The active challenger for (sport, market_type), or None.

        Challengers lazy-load from the database on every call (a retrain
        job may register one at any time) but their artifact bundle is
        cached and reused while the active row's id is unchanged. There is
        no synthetic bootstrap for challengers: no active row means no
        experiment is running.
        """
        record = await self._repo.get_active(sport, market_type, role="challenger")
        key = (sport, market_type, "challenger")
        if record is None:
            self._active.pop(key, None)
            return None
        cached = self._active.get(key)
        if cached is not None and cached.record.id == record.id:
            return cached
        loaded = LoadedModel(record=record, bundle=ArtifactBundle.load(Path(record.artifact_path)))
        self._active[key] = loaded
        return loaded

    async def reload_sport(self, sport: str) -> None:
        """Drop every cached model for a sport and reload its active rows.

        Called after promotion so the in-memory map reflects the flipped
        roles immediately.
        """
        for key in [key for key in self._active if key[0] == sport]:
            del self._active[key]
        records = await self._repo.list_models(sport=sport, is_active=True)
        await self._load_active(records)

    def _has_sport(self, sport: str) -> bool:
        """Whether the sport's game-market champions are loaded (props are keyed separately)."""
        return any(key[0] == sport and key[1] in MARKET_TYPES and key[2] == "champion" for key in self._active)

    async def ensure_bootstrap(self, sport: str) -> None:
        # Prop rows share the sport but bootstrap separately, so the game
        # bootstrap only counts game-market model_versions rows.
        all_active = await self._repo.list_models(sport=sport, is_active=True)
        active = [record for record in all_active if record.model_type in MARKET_TYPES]
        if not active:
            artifact_dir = find_latest_artifact(self._model_dir, sport.lower())
            if artifact_dir is None:
                logger.warning(
                    "no %s model artifact found under %s; training synthetic bootstrap", sport, self._model_dir
                )
                artifact_dir = self._train_bootstrap(sport)
            bundle = ArtifactBundle.load(artifact_dir)
            trained_at = datetime.fromisoformat(str(bundle.metadata["trained_at"]))
            active = await self._repo.register(
                sport=sport,
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

    async def ensure_prop_bootstrap(self, sport: str) -> None:
        """Bootstrap the sport's unified player-prop model (Phase 7 Wave 3).

        Mirrors ensure_bootstrap: register the newest on-disk props
        artifact, training the synthetic prop bootstrap in-process when no
        artifact exists either. Raises ValueError (via the prop registries)
        for sports without a prop wave yet.
        """
        active = await self._repo.list_models(sport=sport, market_type=PROP_MODEL_TYPE, is_active=True)
        if not active:
            artifact_dir = find_latest_artifact(self._model_dir, sport.lower(), family="props")
            if artifact_dir is None:
                logger.warning(
                    "no %s prop model artifact found under %s; training synthetic bootstrap", sport, self._model_dir
                )
                artifact_dir = self._train_bootstrap(sport, market=PROP_MODEL_TYPE)
            bundle = ArtifactBundle.load(artifact_dir)
            trained_at = datetime.fromisoformat(str(bundle.metadata["trained_at"]))
            active = await self._repo.register(
                sport=sport,
                market_types=[PROP_MODEL_TYPE],
                version=bundle.version_tag,
                trained_at=trained_at,
                training_range=(trained_at - timedelta(days=365 * 4), trained_at),
                training_samples=int(bundle.metadata.get("training_samples", 0)),
                evaluation_metrics=dict(bundle.metadata.get("metrics", {})),
                feature_names=bundle.feature_names,
                artifact_path=str(artifact_dir),
                notes=f"prop bootstrap ({bundle.metadata.get('data_label', 'unknown')} data)",
            )
            logger.info("registered bootstrap prop model %s for %s", bundle.version_tag, sport)
        await self._load_active(active)

    def _train_bootstrap(self, sport: str, market: str = "GAME") -> Path:
        from prediction_engine.core.training.synthetic import get_prop_synthetic_generator, get_synthetic_generator
        from prediction_engine.core.training.train import save_artifact, train_model

        generator = get_prop_synthetic_generator(sport) if market == PROP_MODEL_TYPE else get_synthetic_generator(sport)
        result = train_model(generator(), n_rounds=100, data_label="synthetic", sport=sport, market=market)
        return save_artifact(result, self._model_dir)

    async def _load_active(self, records: list[ModelVersionRecord]) -> None:
        bundles: dict[str, ArtifactBundle] = {}
        for record in records:
            if record.artifact_path not in bundles:
                bundles[record.artifact_path] = ArtifactBundle.load(Path(record.artifact_path))
            self._active[(record.sport, record.model_type, record.role)] = LoadedModel(
                record=record, bundle=bundles[record.artifact_path]
            )

    def active_for(self, sport: str, market_type: str) -> LoadedModel | None:
        return self._active.get((sport, market_type, "champion"))

    def active_map(self) -> dict[str, str]:
        """Loaded models by "{sport}_{market}" key; non-champion roles get a role suffix.

        Champion keys keep their pre-Wave-4 shape so health consumers are
        unaffected by the role dimension.
        """
        result: dict[str, str] = {}
        for (sport, market, role), model in self._active.items():
            key = f"{sport}_{market}" if role == "champion" else f"{sport}_{market}_{role}"
            result[key] = str(model.record.id)
        return result
