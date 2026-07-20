"""Background retraining jobs: assemble real outcomes, train, register a challenger.

POST /models/retrain enqueues run() on the running event loop (FastAPI
BackgroundTasks); the CPU-bound train_model call is pushed to a worker
thread so the API stays responsive. Job state lives in Redis under
pred:retrain:{job_id} with a 24h TTL:

    {status: queued|assembling|training|registered|failed|insufficient_data,
     sport, ensemble, detail, rows, model_version_id, updated_at}

The freshly trained model registers as an ACTIVE CHALLENGER (never the
champion): it starts shadow-scoring immediately and only reaches the serve
path through the promotion endpoint (or an operator-raised AB split).
"""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis

from prediction_engine.clients.statistics import StatisticsClient
from prediction_engine.core.model.registry import MARKET_TYPES
from prediction_engine.core.training.assemble import assemble_training_rows
from prediction_engine.core.training.train import save_artifact, train_model
from prediction_engine.db.repository import ModelVersionRepository, PredictionRepository

logger = logging.getLogger(__name__)

RETRAIN_STATUS_PREFIX = "pred:retrain:"


class RetrainService:
    def __init__(
        self,
        model_repo: ModelVersionRepository,
        prediction_repo: PredictionRepository,
        statistics: StatisticsClient,
        redis_client: "aioredis.Redis",
        model_dir: Path,
        status_ttl_seconds: int = 86_400,
    ) -> None:
        self._model_repo = model_repo
        self._prediction_repo = prediction_repo
        self._statistics = statistics
        self._redis = redis_client
        self._model_dir = model_dir
        self._status_ttl = status_ttl_seconds

    async def _set_status(self, job_id: str, status: str, **extra: Any) -> None:
        payload = {
            "job_id": job_id,
            "status": status,
            "updated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
            **extra,
        }
        await self._redis.set(f"{RETRAIN_STATUS_PREFIX}{job_id}", json.dumps(payload), ex=self._status_ttl)

    async def status(self, job_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(f"{RETRAIN_STATUS_PREFIX}{job_id}")
        if raw is None:
            return None
        result: dict[str, Any] = json.loads(raw)
        return result

    async def start(self, sport: str, ensemble: bool, min_rows: int) -> str:
        """Create the job id and its queued status; the caller schedules run()."""
        job_id = str(uuid.uuid4())
        await self._set_status(job_id, "queued", sport=sport, ensemble=ensemble, min_rows=min_rows)
        return job_id

    async def run(self, job_id: str, sport: str, ensemble: bool, min_rows: int) -> None:
        """Assemble -> train (walk-forward) -> save artifact -> register challenger."""
        try:
            await self._set_status(job_id, "assembling", sport=sport, ensemble=ensemble)
            assembled = await assemble_training_rows(sport, MARKET_TYPES, self._prediction_repo, self._statistics)
            if assembled.rows < min_rows:
                await self._set_status(
                    job_id,
                    "insufficient_data",
                    sport=sport,
                    ensemble=ensemble,
                    rows=assembled.rows,
                    detail=(
                        f"assembled {assembled.rows} graded rows "
                        f"({assembled.games_pending} games not yet final, {assembled.rows_pushed} pushes dropped); "
                        f"need {min_rows}"
                    ),
                )
                return
            await self._set_status(job_id, "training", sport=sport, ensemble=ensemble, rows=assembled.rows)
            result = await asyncio.to_thread(
                train_model,
                assembled.dataset,
                data_label="real:assembled",
                sport=sport,
                ensemble=ensemble,
            )
            artifact_dir = save_artifact(result, self._model_dir)
            trained_at = datetime.fromisoformat(str(result.bundle.metadata["trained_at"]))
            records = await self._model_repo.register(
                sport=sport,
                market_types=MARKET_TYPES,
                version=result.bundle.version_tag,
                trained_at=trained_at,
                training_range=(trained_at - timedelta(days=365 * 4), trained_at),
                training_samples=result.training_samples,
                evaluation_metrics=result.metrics,
                feature_names=result.bundle.feature_names,
                artifact_path=str(artifact_dir),
                notes=f"retrain job {job_id} ({'ensemble' if ensemble else 'gbt'}, {assembled.rows} rows)",
                role="challenger",
                activate=True,
            )
            await self._set_status(
                job_id,
                "registered",
                sport=sport,
                ensemble=ensemble,
                rows=assembled.rows,
                model_version_id=str(records[0].id),
                detail=f"registered challenger {result.bundle.version_tag} for {len(records)} markets",
            )
            logger.info("retrain job %s registered challenger %s for %s", job_id, result.bundle.version_tag, sport)
        except Exception as exc:  # noqa: BLE001 - job failures land in the status key, not the server log alone
            logger.exception("retrain job %s failed", job_id)
            await self._set_status(job_id, "failed", sport=sport, ensemble=ensemble, detail=str(exc))
