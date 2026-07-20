"""NCAA_BB prediction flow: the single-league (NCAA_BB, market) registry
end-to-end for a college game with a real trained NCAA_BB bootstrap model
(stubbed clients). Two-way moneyline: one HOME row. The model key is the
league name, not the BASKETBALL sport (see core/leagues.py)."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from prediction_engine.api.schemas import PredictionRequest
from prediction_engine.clients.lines import BestLine, LineMovement, LineSnapshot
from prediction_engine.clients.simulation import SimulationResult, SimulationRun
from prediction_engine.clients.statistics import (
    AdvancedStats,
    DefensiveStats,
    Game,
    OffensiveStats,
    StatBlocks,
    TeamRef,
    TeamStats,
)
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.predictor import Predictor
from prediction_engine.core.training.synthetic import generate_ncaa_bb_synthetic_dataset
from prediction_engine.core.training.train import save_artifact, train_model
from prediction_engine.db.repository import ModelVersionRecord, PredictionRecord

GAME_ID = "game-cbb-duke-ku"
RUN_ID = "run-cbb-duke-ku"


def make_game() -> Game:
    return Game(
        id=GAME_ID,
        league="NCAA_BB",
        status="SCHEDULED",
        home_team=TeamRef(id="t-duke", name="Duke Blue Devils"),
        away_team=TeamRef(id="t-ku", name="Kansas Jayhawks"),
        scheduled_start="2026-12-05T00:00:00Z",
        season=2026,
        season_type="REGULAR",
    )


def make_run() -> SimulationRun:
    return SimulationRun(
        simulation_run_id=RUN_ID,
        game_id=GAME_ID,
        status="completed",
        converged=True,
        result=SimulationResult(
            home_win_probability=0.67,
            away_win_probability=0.33,
            mean_margin=6.5,
            mean_total=145.5,
            spread_cover_probabilities={"-6.5": 0.50, "-5.5": 0.54},
            total_over_probabilities={"144.5": 0.52, "145.5": 0.50},
        ),
    )


def team_stats(team_id: str, aem: float) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=12,
        stats=StatBlocks(
            offensive=OffensiveStats(points_per_game=78.0, three_point_pct=0.35, offensive_rating=116.0, pace=68.0),
            defensive=DefensiveStats(points_allowed_per_game=68.0, defensive_rating=95.0),
            advanced=AdvancedStats(net_rating=21.0, adjusted_efficiency_margin=aem),
        ),
    )


class FakeStatistics:
    async def get_game(self, game_id: str) -> Game:
        return make_game()

    async def get_team_stats(self, team_id: str, rolling_window: int | None = None) -> TeamStats:
        return team_stats(team_id, aem=26.0 if team_id == "t-duke" else 16.0)

    async def list_recent_games(self, team_id: str, date_to: str, limit: int = 10) -> list[Game]:
        return [
            Game(
                id=f"prev-{team_id}",
                league="NCAA_BB",
                status="FINAL",
                home_team=TeamRef(id=team_id),
                away_team=TeamRef(id="x"),
                scheduled_start="2026-12-03T00:00:00Z",
            )
        ]

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-12-05T00:00:00Z"


class FakeLines:
    async def best_lines(self, game_external_id: str, market_type: str | None = None) -> list[BestLine]:
        return [
            BestLine(
                market_type="MONEYLINE",
                selection="Duke Blue Devils",
                side="HOME",
                best_odds_american=-240,
                implied_probability=0.68,
                sportsbook_key="draftkings",
            ),
            BestLine(
                market_type="MONEYLINE",
                selection="Kansas Jayhawks",
                side="AWAY",
                best_odds_american=+200,
                implied_probability=0.33,
                sportsbook_key="fanduel",
            ),
            BestLine(
                market_type="SPREAD",
                selection="Duke Blue Devils -6.5",
                side="HOME",
                line_value=-6.5,
                best_odds_american=-110,
                implied_probability=0.52,
                sportsbook_key="draftkings",
            ),
            BestLine(
                market_type="TOTAL",
                selection="Over 144.5",
                side="OVER",
                line_value=144.5,
                best_odds_american=-110,
                implied_probability=0.52,
                sportsbook_key="fanduel",
            ),
        ]

    async def movement(self, game_external_id: str, market_type: str = "SPREAD") -> list[LineMovement]:
        return [LineMovement(total_movement=0.5)]

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        return [LineSnapshot(id="s1", game_id="ext-cbb", sportsbook_key="dk", side="HOME", line_value=-6.5)]


class FakeReconciler:
    async def resolve(self, game: Game) -> str | None:
        return "ext-cbb"


class FakeSimulation:
    async def get_run(self, run_id: str) -> SimulationRun:
        return make_run()


class FakeModelVersionRepo:
    def __init__(self) -> None:
        self.records: list[ModelVersionRecord] = []

    async def list_models(
        self, sport: str | None = None, market_type: str | None = None, is_active: bool | None = None
    ) -> list[ModelVersionRecord]:
        return [
            record
            for record in self.records
            if (sport is None or record.sport == sport) and (is_active is None or record.is_active == is_active)
        ]

    async def register(self, sport: str, market_types: list[str], **kwargs: Any) -> list[ModelVersionRecord]:
        created = [
            ModelVersionRecord(
                id=uuid.uuid4(),
                sport=sport,
                model_type=market,
                version=kwargs["version"],
                algorithm="xgboost",
                trained_at=kwargs.get("trained_at", datetime.now(tz=UTC)),
                training_samples=kwargs.get("training_samples", 0),
                evaluation_metrics=kwargs.get("evaluation_metrics", {}),
                feature_names=kwargs.get("feature_names", []),
                is_active=True,
                artifact_path=kwargs["artifact_path"],
                notes=kwargs.get("notes"),
            )
            for market in market_types
        ]
        self.records.extend(created)
        return created


class FakeRepo:
    def __init__(self) -> None:
        self.persisted_features: dict[str, Any] | None = None

    async def insert_predictions(
        self,
        rows: list[dict[str, Any]],
        features: dict[str, Any],
        feature_sources: dict[str, Any],
        row_features: list[dict[str, Any]] | None = None,
    ) -> list[PredictionRecord]:
        self.persisted_features = features
        return [
            PredictionRecord(
                id=uuid.uuid4(),
                game_external_id=row["game_external_id"],
                model_version_id=row["model_version_id"],
                league=row["league"],
                market_type=row["market_type"],
                side=row["side"],
                selection=row["selection"],
                predicted_probability=row["predicted_probability"],
                simulation_probability=row["simulation_probability"],
                implied_probability=row["implied_probability"],
                edge=row["edge"],
                confidence_lower=row["confidence_lower"],
                confidence_upper=row["confidence_upper"],
                feature_importance=row["feature_importance"],
                created_at=datetime.now(tz=UTC),
            )
            for row in rows
        ]


class FakeRedis:
    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        return None

    async def publish(self, channel: str, payload: str) -> None:
        return None


@pytest.fixture(scope="module")
def ncaa_bb_registry(tmp_path_factory: pytest.TempPathFactory) -> ModelRegistry:
    model_dir = tmp_path_factory.mktemp("ncaa-bb-models")
    result = train_model(
        generate_ncaa_bb_synthetic_dataset(n_rows=600), n_rounds=8, data_label="synthetic", sport="NCAA_BB"
    )
    save_artifact(result, model_dir)
    return ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]


def make_predictor(ncaa_bb_registry: ModelRegistry, repo: FakeRepo) -> Predictor:
    statistics = FakeStatistics()
    lines = FakeLines()
    reconciler = FakeReconciler()
    builder = FeatureBuilder(statistics, lines, reconciler)  # type: ignore[arg-type]
    return Predictor(
        statistics=statistics,  # type: ignore[arg-type]
        lines=lines,  # type: ignore[arg-type]
        simulation=FakeSimulation(),  # type: ignore[arg-type]
        reconciler=reconciler,  # type: ignore[arg-type]
        features=builder,
        registry=ncaa_bb_registry,
        repo=repo,  # type: ignore[arg-type]
        redis_client=FakeRedis(),  # type: ignore[arg-type]
    )


class TestNcaaBbTwoWayFlow:
    async def test_all_three_markets_produce_rows_with_sides(self, ncaa_bb_registry) -> None:
        predictor = make_predictor(ncaa_bb_registry, FakeRepo())
        request = PredictionRequest(
            game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["SPREAD", "TOTAL", "MONEYLINE"]
        )
        response = await predictor.create_predictions(request)

        sides = [(p.market_type, p.side) for p in response.predictions]
        assert sides == [("SPREAD", "HOME"), ("TOTAL", "OVER"), ("MONEYLINE", "HOME")]
        assert all(0.0 < p.predicted_probability < 1.0 for p in response.predictions)

    async def test_moneyline_is_single_home_row(self, ncaa_bb_registry) -> None:
        predictor = make_predictor(ncaa_bb_registry, FakeRepo())
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        assert len(response.predictions) == 1
        prediction = response.predictions[0]
        assert prediction.side == "HOME"
        assert prediction.selection == "Duke Blue Devils ML"
        assert prediction.simulation_probability == 0.67
        assert not any(p.side in ("DRAW", "AWAY") for p in response.predictions)

    async def test_features_come_from_the_ncaa_bb_registry(self, ncaa_bb_registry) -> None:
        repo = FakeRepo()
        predictor = make_predictor(ncaa_bb_registry, repo)
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        features = response.features_used
        assert features["home_adjusted_efficiency_margin"] == 26.0
        assert features["away_adjusted_efficiency_margin"] == 16.0
        assert features["home_offensive_rating"] == 116.0
        assert "home_injury_impact" not in features  # no college injury source
        assert "league_is_mlb" not in features
        assert repo.persisted_features == features

    async def test_model_is_keyed_by_league_not_basketball_sport(self, ncaa_bb_registry) -> None:
        # The active model registered for this flow is the NCAA_BB single-league
        # model, not the pooled BASKETBALL one (see core/leagues.py).
        repo = FakeRepo()
        predictor = make_predictor(ncaa_bb_registry, repo)
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)
        loaded = await ncaa_bb_registry.get_active("NCAA_BB", "MONEYLINE")
        assert loaded is not None
        assert str(loaded.record.id) == response.predictions[0].model_version_id
        # the flow bootstrapped NCAA_BB, never the pooled BASKETBALL model
        assert set(ncaa_bb_registry.active_map()) == {
            "NCAA_BB_SPREAD",
            "NCAA_BB_TOTAL",
            "NCAA_BB_MONEYLINE",
        }
