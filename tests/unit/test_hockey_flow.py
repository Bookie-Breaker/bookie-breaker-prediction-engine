"""Hockey prediction flow: the (sport, market) registry end-to-end for an NHL
game with a real trained HOCKEY bootstrap model (stubbed clients). The
moneyline stays two-way (OT/SO resolves): one HOME row, no DRAW/AWAY."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from prediction_engine.api.schemas import PredictionRequest
from prediction_engine.clients.lines import BestLine, LineMovement, LineSnapshot
from prediction_engine.clients.simulation import SimulationResult, SimulationRun
from prediction_engine.clients.statistics import Game, HockeyStats, StatBlocks, TeamRef, TeamStats
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.predictor import Predictor
from prediction_engine.core.training.synthetic import generate_hockey_synthetic_dataset
from prediction_engine.core.training.train import save_artifact, train_model
from prediction_engine.db.repository import ModelVersionRecord, PredictionRecord

GAME_ID = "game-nhl-col-vgk"
RUN_ID = "run-nhl-col-vgk"


def make_game() -> Game:
    return Game(
        id=GAME_ID,
        league="NHL",
        status="SCHEDULED",
        home_team=TeamRef(id="t-col", name="Colorado Avalanche"),
        away_team=TeamRef(id="t-vgk", name="Vegas Golden Knights"),
        scheduled_start="2026-11-15T02:00:00Z",
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
            home_win_probability=0.56,
            away_win_probability=0.44,
            mean_margin=0.4,
            mean_total=6.1,
            spread_cover_probabilities={"-1.5": 0.38, "1.5": 0.70},
            total_over_probabilities={"5.5": 0.55, "6.5": 0.42},
        ),
    )


def team_stats(team_id: str, save_pct: float) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=18,
        stats=StatBlocks(
            hockey=HockeyStats(
                goals_for_per_game=3.3,
                goals_against_per_game=2.7,
                shots_for_per_game=31.0,
                shots_against_per_game=29.0,
                power_play_pct=0.24,
                penalty_kill_pct=0.82,
                team_save_pct=save_pct,
            )
        ),
    )


class FakeStatistics:
    async def get_game(self, game_id: str) -> Game:
        return make_game()

    async def get_team_stats(self, team_id: str, rolling_window: int | None = None) -> TeamStats:
        return team_stats(team_id, save_pct=0.918 if team_id == "t-col" else 0.900)

    async def list_recent_games(self, team_id: str, date_to: str, limit: int = 10) -> list[Game]:
        return [
            Game(
                id=f"prev-{team_id}",
                league="NHL",
                status="FINAL",
                home_team=TeamRef(id=team_id),
                away_team=TeamRef(id="x"),
                scheduled_start="2026-11-14T02:00:00Z",
            )
        ]

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-11-15T00:00:00Z"


class FakeLines:
    async def best_lines(self, game_external_id: str, market_type: str | None = None) -> list[BestLine]:
        return [
            BestLine(
                market_type="MONEYLINE",
                selection="Colorado Avalanche",
                side="HOME",
                best_odds_american=-130,
                implied_probability=0.57,
                sportsbook_key="draftkings",
            ),
            BestLine(
                market_type="MONEYLINE",
                selection="Vegas Golden Knights",
                side="AWAY",
                best_odds_american=+110,
                implied_probability=0.48,
                sportsbook_key="fanduel",
            ),
            BestLine(
                market_type="SPREAD",
                selection="Colorado Avalanche -1.5",
                side="HOME",
                line_value=-1.5,
                best_odds_american=+160,
                implied_probability=0.38,
                sportsbook_key="draftkings",
            ),
            BestLine(
                market_type="TOTAL",
                selection="Over 5.5",
                side="OVER",
                line_value=5.5,
                best_odds_american=-115,
                implied_probability=0.53,
                sportsbook_key="fanduel",
            ),
        ]

    async def movement(self, game_external_id: str, market_type: str = "SPREAD") -> list[LineMovement]:
        return [LineMovement(total_movement=0.5)]

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        return [LineSnapshot(id="s1", game_id="ext-nhl", sportsbook_key="dk", side="HOME", line_value=-1.5)]


class FakeReconciler:
    async def resolve(self, game: Game) -> str | None:
        return "ext-nhl"


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
def hockey_registry(tmp_path_factory: pytest.TempPathFactory) -> ModelRegistry:
    model_dir = tmp_path_factory.mktemp("hockey-models")
    result = train_model(
        generate_hockey_synthetic_dataset(n_games=300), n_rounds=8, data_label="synthetic", sport="HOCKEY"
    )
    save_artifact(result, model_dir)
    return ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]


def make_predictor(hockey_registry: ModelRegistry, repo: FakeRepo) -> Predictor:
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
        registry=hockey_registry,
        repo=repo,  # type: ignore[arg-type]
        redis_client=FakeRedis(),  # type: ignore[arg-type]
    )


class TestHockeyTwoWayFlow:
    async def test_all_three_markets_produce_rows_with_sides(self, hockey_registry) -> None:
        predictor = make_predictor(hockey_registry, FakeRepo())
        request = PredictionRequest(
            game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["SPREAD", "TOTAL", "MONEYLINE"]
        )
        response = await predictor.create_predictions(request)

        sides = [(p.market_type, p.side) for p in response.predictions]
        assert sides == [("SPREAD", "HOME"), ("TOTAL", "OVER"), ("MONEYLINE", "HOME")]
        assert all(0.0 < p.predicted_probability < 1.0 for p in response.predictions)

    async def test_moneyline_is_single_home_row(self, hockey_registry) -> None:
        predictor = make_predictor(hockey_registry, FakeRepo())
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        assert len(response.predictions) == 1
        prediction = response.predictions[0]
        assert prediction.side == "HOME"
        assert prediction.selection == "Colorado Avalanche ML"
        assert prediction.simulation_probability == 0.56
        assert not any(p.side in ("DRAW", "AWAY") for p in response.predictions)

    async def test_features_come_from_the_hockey_registry(self, hockey_registry) -> None:
        repo = FakeRepo()
        predictor = make_predictor(hockey_registry, repo)
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        features = response.features_used
        assert features["home_team_save_pct"] == 0.918
        assert features["away_team_save_pct"] == 0.900
        assert features["home_power_play_pct"] == 0.24
        assert features["home_back_to_back"] == 1.0  # played yesterday
        assert "home_offensive_rating" not in features
        assert "sim_draw_probability" not in features
        assert "league_is_nhl" not in features
        assert repo.persisted_features == features
