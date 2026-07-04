"""Runtime configuration via environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    port: int = 8004
    log_level: str = "info"
    database_url: str = (
        "postgres://predictions_svc:localdev@localhost:5432/bookiebreaker?search_path=predictions,public"
    )
    redis_url: str = "redis://localhost:6379"
    statistics_service_url: str = "http://localhost:8002"
    lines_service_url: str = "http://localhost:8001"
    simulation_service_url: str = "http://localhost:8003"
    model_dir: Path = Path("./models")

    idempotency_ttl_seconds: int = 86_400  # 24h per api-contracts README
    game_map_ttl_seconds: int = 86_400  # statistics<->lines game id mapping cache

    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "prediction-engine"


@lru_cache
def get_settings() -> Settings:
    return Settings()
