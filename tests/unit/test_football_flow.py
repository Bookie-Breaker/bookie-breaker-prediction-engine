"""Football prediction flow: the (sport, market) registry end-to-end for an
NFL game with a real trained FOOTBALL bootstrap model (stubbed clients). The
moneyline stays two-way (a tie is a PUSH): one HOME row, no DRAW/AWAY."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from prediction_engine.api.schemas import PredictionRequest
from prediction_engine.clients.lines import BestLine, LineMovement, LineSnapshot
from prediction_engine.clients.simulation import SimulationResult, SimulationRun
from prediction_engine.clients.statistics import FootballStats, Game, StatBlocks, TeamRef, TeamStats
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.predictor import Predictor
from prediction_engine.core.training.synthetic import generate_football_synthetic_dataset
from prediction_engine.core.training.train import save_artifact, train_model
from prediction_engine.db.repository import ModelVersionRecord, PredictionRecord

GAME_ID = "game-nfl-kc-buf"
RUN_ID = "run-nfl-kc-buf"


def make_game() -> Game:
    return Game(
        id=GAME_ID,
        league="NFL",
        status="SCHEDULED",
        home_team=TeamRef(id="t-kc", name="Kansas City Chiefs"),
        away_team=TeamRef(id="t-buf", name="Buffalo Bills"),
        scheduled_start="2026-11-15T18:00:00Z",
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
            home_win_probability=0.61,
            away_win_probability=0.39,
            mean_margin=3.5,
            mean_total=47.5,
            draw_probability=0.004,
            spread_cover_probabilities={"-3.5": 0.51, "-2.5": 0.55},
            total_over_probabilities={"47.5": 0.50, "48.5": 0.46},
        ),
    )


def team_stats(team_id: str, epa_off: float) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=10,
        stats=StatBlocks(
            football=FootballStats(
                points_per_game=27.0,
                points_allowed_per_game=20.0,
                drives_per_game=11.0,
                points_per_drive_off=2.2,
                points_per_drive_def=1.8,
                epa_per_play_off=epa_off,
                epa_per_play_def=-0.03,
                turnover_margin_per_game=0.5,
            )
        ),
    )


class FakeStatistics:
    async def get_game(self, game_id: str) -> Game:
        return make_game()

    async def get_team_stats(self, team_id: str, rolling_window: int | None = None) -> TeamStats:
        return team_stats(team_id, epa_off=0.14 if team_id == "t-kc" else 0.05)

    async def list_recent_games(self, team_id: str, date_to: str, limit: int = 10) -> list[Game]:
        return [
            Game(
                id=f"prev-{team_id}",
                league="NFL",
                status="FINAL",
                home_team=TeamRef(id=team_id),
                away_team=TeamRef(id="x"),
                scheduled_start="2026-11-09T18:00:00Z",
            )
        ]

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-11-15T00:00:00Z"


class FakeLines:
    async def best_lines(self, game_external_id: str, market_type: str | None = None) -> list[BestLine]:
        return [
            BestLine(
                market_type="MONEYLINE",
                selection="Kansas City Chiefs",
                side="HOME",
                best_odds_american=-150,
                implied_probability=0.60,
                sportsbook_key="draftkings",
            ),
            BestLine(
                market_type="MONEYLINE",
                selection="Buffalo Bills",
                side="AWAY",
                best_odds_american=+130,
                implied_probability=0.43,
                sportsbook_key="fanduel",
            ),
            BestLine(
                market_type="SPREAD",
                selection="Kansas City Chiefs -3.5",
                side="HOME",
                line_value=-3.5,
                best_odds_american=-110,
                implied_probability=0.52,
                sportsbook_key="draftkings",
            ),
            BestLine(
                market_type="TOTAL",
                selection="Over 47.5",
                side="OVER",
                line_value=47.5,
                best_odds_american=-110,
                implied_probability=0.52,
                sportsbook_key="fanduel",
            ),
        ]

    async def movement(self, game_external_id: str, market_type: str = "SPREAD") -> list[LineMovement]:
        return [LineMovement(total_movement=0.5)]

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        return [LineSnapshot(id="s1", game_id="ext-nfl", sportsbook_key="dk", side="HOME", line_value=-3.5)]


class FakeReconciler:
    async def resolve(self, game: Game) -> str | None:
        return "ext-nfl"


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
def football_registry(tmp_path_factory: pytest.TempPathFactory) -> ModelRegistry:
    model_dir = tmp_path_factory.mktemp("football-models")
    result = train_model(
        generate_football_synthetic_dataset(n_games=300), n_rounds=8, data_label="synthetic", sport="FOOTBALL"
    )
    save_artifact(result, model_dir)
    return ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]


def make_predictor(football_registry: ModelRegistry, repo: FakeRepo) -> Predictor:
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
        registry=football_registry,
        repo=repo,  # type: ignore[arg-type]
        redis_client=FakeRedis(),  # type: ignore[arg-type]
    )


class TestFootballTwoWayFlow:
    async def test_all_three_markets_produce_rows_with_sides(self, football_registry) -> None:
        predictor = make_predictor(football_registry, FakeRepo())
        request = PredictionRequest(
            game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["SPREAD", "TOTAL", "MONEYLINE"]
        )
        response = await predictor.create_predictions(request)

        sides = [(p.market_type, p.side) for p in response.predictions]
        assert sides == [("SPREAD", "HOME"), ("TOTAL", "OVER"), ("MONEYLINE", "HOME")]
        assert all(0.0 < p.predicted_probability < 1.0 for p in response.predictions)

    async def test_moneyline_is_single_home_row_tie_is_push(self, football_registry) -> None:
        predictor = make_predictor(football_registry, FakeRepo())
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        assert len(response.predictions) == 1
        prediction = response.predictions[0]
        assert prediction.side == "HOME"
        assert prediction.selection == "Kansas City Chiefs ML"
        assert prediction.simulation_probability == 0.61
        # a tie is a PUSH, not a third selection: no DRAW/AWAY rows
        assert not any(p.side in ("DRAW", "AWAY") for p in response.predictions)

    async def test_features_come_from_the_football_registry(self, football_registry) -> None:
        repo = FakeRepo()
        predictor = make_predictor(football_registry, repo)
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        features = response.features_used
        assert features["home_points_per_game"] == 27.0
        assert features["home_epa_per_play_off"] == pytest.approx(0.14)
        assert features["home_sp_plus_rating"] is None  # NFL has no SP+ source
        assert features["league_is_nfl"] == 1.0
        assert features["home_bye_week"] == 0.0
        assert "home_offensive_rating" not in features  # not the basketball path
        # market-independent bundle carries no row-level sim/draw features
        assert "sim_draw_probability" not in features
        assert repo.persisted_features == features
