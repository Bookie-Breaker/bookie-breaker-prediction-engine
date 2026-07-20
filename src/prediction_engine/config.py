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
    prediction_sports: str = "BASKETBALL"  # comma-separated sports to bootstrap at startup

    idempotency_ttl_seconds: int = 86_400  # 24h per api-contracts README
    game_map_ttl_seconds: int = 86_400  # statistics<->lines game id mapping cache

    # Champion/challenger serving (Phase 7 Wave 4)
    shadow_scoring_enabled: bool = True  # zero-risk: shadow rows never reach read paths
    # Deterministic % of games served by the active challenger (champion
    # becomes the shadow row). 0 = shadow-only. Raising this begins real
    # bankroll exposure to the challenger -- operator decision.
    ab_split_pct: int = 0
    promotion_min_samples: int = 300  # graded shadow pairs required to promote
    retrain_status_ttl_seconds: int = 86_400  # pred:retrain:{job_id} status keys

    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "prediction-engine"

    @property
    def prediction_sports_list(self) -> list[str]:
        return [sport.strip().upper() for sport in self.prediction_sports.split(",") if sport.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
