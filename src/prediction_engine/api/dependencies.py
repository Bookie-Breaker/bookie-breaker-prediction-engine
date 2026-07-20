"""FastAPI dependency accessors backed by app.state."""

from fastapi import Request

from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.predictor import Predictor
from prediction_engine.db.repository import ModelVersionRepository, PredictionRepository
from prediction_engine.services.experiments import ExperimentService
from prediction_engine.services.health import HealthService
from prediction_engine.services.retrain import RetrainService


def get_predictor(request: Request) -> Predictor:
    predictor: Predictor = request.app.state.predictor
    return predictor


def get_prediction_repo(request: Request) -> PredictionRepository:
    repo: PredictionRepository = request.app.state.prediction_repo
    return repo


def get_model_repo(request: Request) -> ModelVersionRepository:
    repo: ModelVersionRepository = request.app.state.model_repo
    return repo


def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry = request.app.state.registry
    return registry


def get_health_service(request: Request) -> HealthService:
    service: HealthService = request.app.state.health_service
    return service


def get_retrain_service(request: Request) -> RetrainService:
    service: RetrainService = request.app.state.retrain_service
    return service


def get_experiment_service(request: Request) -> ExperimentService:
    service: ExperimentService = request.app.state.experiment_service
    return service
