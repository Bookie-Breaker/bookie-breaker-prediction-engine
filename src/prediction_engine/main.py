"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI

from prediction_engine import __version__
from prediction_engine.api.envelope import RequestIDMiddleware
from prediction_engine.api.errors import register_error_handlers
from prediction_engine.api.routes import edges, health, models, predictions
from prediction_engine.clients.lines import LinesClient
from prediction_engine.clients.reconcile import GameReconciler
from prediction_engine.clients.simulation import SimulationClient
from prediction_engine.clients.statistics import StatisticsClient
from prediction_engine.config import Settings, get_settings
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.predictor import Predictor
from prediction_engine.db.engine import create_engine
from prediction_engine.db.repository import ModelVersionRepository, PredictionRepository
from prediction_engine.services.health import HealthService
from prediction_engine.telemetry import configure_telemetry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings.database_url)
        redis_client: aioredis.Redis = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))

        statistics = StatisticsClient(settings.statistics_service_url, http_client)
        lines = LinesClient(settings.lines_service_url, http_client)
        simulation = SimulationClient(settings.simulation_service_url, http_client)
        reconciler = GameReconciler(lines, redis_client, ttl_seconds=settings.game_map_ttl_seconds)
        features = FeatureBuilder(statistics, lines, reconciler)

        model_repo = ModelVersionRepository(engine)
        prediction_repo = PredictionRepository(engine)
        registry = ModelRegistry(model_repo, settings.model_dir)
        await registry.try_bootstrap()

        app.state.model_repo = model_repo
        app.state.prediction_repo = prediction_repo
        app.state.registry = registry
        app.state.predictor = Predictor(
            statistics,
            lines,
            simulation,
            reconciler,
            features,
            registry,
            prediction_repo,
            redis_client,
            idempotency_ttl=settings.idempotency_ttl_seconds,
        )
        app.state.health_service = HealthService(statistics, lines, prediction_repo, registry, redis_client)
        try:
            yield
        finally:
            await http_client.aclose()
            await redis_client.aclose()
            await engine.dispose()

    app = FastAPI(
        title="BookieBreaker Prediction Engine",
        version=__version__,
        description="ML-calibrated probabilities and confidence intervals over simulation output.",
        contact={
            "name": "BookieBreaker",
            "url": "https://github.com/Bookie-Breaker",
            "email": "jsamuelsen11@gmail.com",
        },
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        servers=[{"url": "http://localhost:8004", "description": "Local development"}],
        openapi_tags=[
            {"name": "edges", "description": "Predictions combined with current market lines."},
            {"name": "health", "description": "Service health and active models."},
            {"name": "models", "description": "Model version registry and retraining."},
            {"name": "predictions", "description": "Generate and fetch calibrated predictions."},
        ],
        lifespan=lifespan,
    )
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)
    app.include_router(predictions.router, prefix="/api/v1/predict")
    app.include_router(edges.router, prefix="/api/v1/predict")
    app.include_router(models.router, prefix="/api/v1/predict")
    app.include_router(health.router, prefix="/api/v1/predict")
    configure_telemetry(app, settings)
    return app


app = create_app()
