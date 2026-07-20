"""Request/response models mirroring api-contracts/prediction-engine-api.md."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Game markets requestable via market_types. PLAYER_PROP rows are requested
# through the props list instead and appear in responses with
# market_type == "PLAYER_PROP" (response market_type fields are plain str).
MarketType = Literal["SPREAD", "TOTAL", "MONEYLINE"]

DEFAULT_MARKETS: list[MarketType] = ["SPREAD", "TOTAL", "MONEYLINE"]

Side = Literal["HOME", "AWAY", "DRAW", "OVER", "UNDER", "YES", "NO"]

PropSide = Literal["OVER", "UNDER", "YES", "NO"]

_SIDE_FIELD = Field(
    default=None,
    description=(
        "Selection side (HOME, AWAY, DRAW, OVER, UNDER, YES, NO). Null for predictions created before Phase 6."
    ),
)


class PropRequest(BaseModel):
    """One requested player prop (Phase 7 Wave 3).

    line is required for count/yardage stats and must exactly match a
    half-step line in the simulation's over grid; it stays None for yes/no
    stats. side None emits both OVER and UNDER rows (or the single YES row
    for yes/no stats).
    """

    player_external_id: str = Field(description="The statistics-service player UUID (as a string).")
    player_name: str | None = None
    stat_type: str = Field(description="Canonical prop stat key (Odds API market key), e.g. player_points.")
    line: float | None = Field(default=None, description="Prop line (half-step); null for yes/no stats.")
    side: PropSide | None = Field(default=None, description="Requested side; null emits both sides (or YES).")


class PredictionRequest(BaseModel):
    game_id: str
    simulation_run_id: str
    market_types: list[MarketType] = Field(default_factory=lambda: list(DEFAULT_MARKETS))
    props: list[PropRequest] = Field(
        default_factory=list,
        description="Player props to predict (requires a simulation run captured with player stats).",
    )

    @model_validator(mode="after")
    def _require_markets_or_props(self) -> "PredictionRequest":
        if not self.market_types and not self.props:
            raise ValueError("market_types and props cannot both be empty")
        return self


class PredictionItem(BaseModel):
    id: str
    market_type: str
    side: Side | None = _SIDE_FIELD
    selection: str
    predicted_probability: float
    simulation_probability: float | None
    adjustment_magnitude: float
    confidence_lower: float | None
    confidence_upper: float | None
    model_version_id: str
    feature_importance: dict[str, float]
    created_at: str
    # Player-prop rows only (market_type == "PLAYER_PROP"); null otherwise.
    player_external_id: str | None = None
    stat_type: str | None = None
    prop_line: float | None = None


class PredictionGroupData(BaseModel):
    game_id: str
    simulation_run_id: str
    predictions: list[PredictionItem]
    features_used: dict[str, float | int | bool | None]
    feature_source_versions: dict[str, str | None]


class FeatureVectorData(BaseModel):
    id: str
    features: dict[str, float | int | bool | None]
    feature_source_versions: dict[str, str | None]


class PredictionDetailData(BaseModel):
    id: str
    game_id: str
    model_version_id: str
    market_type: str
    side: Side | None = _SIDE_FIELD
    selection: str
    predicted_probability: float
    simulation_probability: float | None
    adjustment_magnitude: float
    confidence_lower: float | None
    confidence_upper: float | None
    feature_importance: dict[str, float]
    feature_vector: FeatureVectorData
    created_at: str
    # Player-prop rows only (market_type == "PLAYER_PROP"); null otherwise.
    player_external_id: str | None = None
    stat_type: str | None = None
    prop_line: float | None = None


class LatestPredictionsData(BaseModel):
    game_id: str
    predictions: list[PredictionItem]


class EdgeItem(BaseModel):
    prediction_id: str
    market_type: str
    selection: str
    predicted_probability: float
    implied_probability: float
    edge_percentage: float
    best_odds_american: int
    sportsbook_key: str


class EdgesData(BaseModel):
    game_id: str
    edges: list[EdgeItem]


class ModelVersionData(BaseModel):
    id: str
    sport: str
    market_type: str
    version_tag: str
    algorithm: str
    training_date: str
    training_samples: int
    evaluation_metrics: dict[str, float]
    is_active: bool
    notes: str | None = None


class ModelVersionDetailData(ModelVersionData):
    feature_names: list[str]


class RetrainConfig(BaseModel):
    min_samples: int = 1000
    test_split: float = Field(default=0.2, gt=0.0, lt=1.0)
    date_from: str | None = None
    date_to: str | None = None


class RetrainRequest(BaseModel):
    sport: str
    market_type: str
    training_config: RetrainConfig = RetrainConfig()


class RetrainData(BaseModel):
    retrain_id: str
    sport: str
    market_type: str
    status: str
    started_at: str
    estimated_duration_minutes: int


class HealthData(BaseModel):
    status: str
    service: str = "prediction-engine"
    version: str
    uptime_seconds: int
    dependencies: dict[str, str]
    active_models: dict[str, str]
