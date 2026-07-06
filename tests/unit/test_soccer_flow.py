"""Soccer prediction flow: Wave 0 three-way path with the Wave 1 feature
vector and a real trained SOCCER bootstrap model (stubbed service clients)."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from prediction_engine.api.schemas import PredictionRequest
from prediction_engine.clients.lines import BestLine, LineMovement, LineSnapshot
from prediction_engine.clients.simulation import SimulationResult, SimulationRun
from prediction_engine.clients.statistics import Game, SoccerStats, StatBlocks, TeamRef, TeamStats
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.predictor import Predictor
from prediction_engine.core.training.synthetic import generate_soccer_synthetic_dataset
from prediction_engine.core.training.train import save_artifact, train_model
from prediction_engine.db.repository import ModelVersionRecord, PredictionRecord

GAME_ID = "game-wc-final"
RUN_ID = "run-wc-final"


def make_game() -> Game:
    return Game(
        id=GAME_ID,
        league="FIFA_WC",
        status="SCHEDULED",
        home_team=TeamRef(id="t-arg", name="Argentina"),
        away_team=TeamRef(id="t-fra", name="France"),
        scheduled_start="2026-07-19T20:00:00Z",
        season=2026,
        season_type="POSTSEASON",
    )


def make_run() -> SimulationRun:
    return SimulationRun(
        simulation_run_id=RUN_ID,
        game_id=GAME_ID,
        status="completed",
        converged=True,
        result=SimulationResult(
            home_win_probability=0.44,
            away_win_probability=0.30,
            draw_probability=0.26,
            mean_margin=0.4,
            mean_total=2.7,
            spread_cover_probabilities={"-0.5": 0.44, "0.5": 0.70},
            total_over_probabilities={"2.5": 0.52, "3.5": 0.27},
        ),
    )


def team_stats(team_id: str, attack: float) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=6,
        stats=StatBlocks(
            soccer=SoccerStats(
                goals_for_per_match=2.0,
                goals_against_per_match=0.8,
                attack_strength=attack,
                defense_strength=0.75,
                form_points_last5=12,
            )
        ),
    )


class FakeStatistics:
    async def get_game(self, game_id: str) -> Game:
        return make_game()

    async def get_team_stats(self, team_id: str, rolling_window: int | None = None) -> TeamStats:
        return team_stats(team_id, attack=1.4 if team_id == "t-arg" else 1.3)

    async def list_recent_games(self, team_id: str, date_to: str, limit: int = 10) -> list[Game]:
        return [
            Game(
                id=f"prev-{team_id}",
                league="FIFA_WC",
                status="FINAL",
                home_team=TeamRef(id=team_id),
                away_team=TeamRef(id="x"),
                scheduled_start="2026-07-14T20:00:00Z",
            )
        ]

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-07-19T00:00:00Z"


class FakeLines:
    async def best_lines(self, game_external_id: str, market_type: str | None = None) -> list[BestLine]:
        def line(side: str, selection: str, implied: float) -> BestLine:
            return BestLine(
                market_type="MONEYLINE",
                selection=selection,
                side=side,
                best_odds_american=-110,
                implied_probability=implied,
                sportsbook_key="draftkings",
            )

        return [line("HOME", "Argentina", 0.42), line("DRAW", "Draw", 0.29), line("AWAY", "France", 0.32)]

    async def movement(self, game_external_id: str, market_type: str = "SPREAD") -> list[LineMovement]:
        return [LineMovement(total_movement=0.25)]

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        return [LineSnapshot(id="s1", game_id="ext-wc", sportsbook_key="dk", side="HOME", line_value=-0.5)]


class FakeReconciler:
    async def resolve(self, game: Game) -> str | None:
        return "ext-wc"


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
        self, rows: list[dict[str, Any]], features: dict[str, Any], feature_sources: dict[str, Any]
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
def soccer_registry(tmp_path_factory: pytest.TempPathFactory) -> ModelRegistry:
    model_dir = tmp_path_factory.mktemp("soccer-models")
    result = train_model(
        generate_soccer_synthetic_dataset(n_games=160), n_rounds=8, data_label="synthetic", sport="SOCCER"
    )
    save_artifact(result, model_dir)
    return ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]


def make_predictor(soccer_registry: ModelRegistry, repo: FakeRepo) -> Predictor:
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
        registry=soccer_registry,
        repo=repo,  # type: ignore[arg-type]
        redis_client=FakeRedis(),  # type: ignore[arg-type]
    )


class TestSoccerThreeWayFlow:
    async def test_fifa_wc_moneyline_emits_renormalized_three_way_rows(self, soccer_registry) -> None:
        repo = FakeRepo()
        predictor = make_predictor(soccer_registry, repo)
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        assert [p.side for p in response.predictions] == ["HOME", "DRAW", "AWAY"]
        assert [p.selection for p in response.predictions] == ["Argentina ML", "Draw", "France ML"]
        assert [p.simulation_probability for p in response.predictions] == [0.44, 0.26, 0.30]
        assert sum(p.predicted_probability for p in response.predictions) == pytest.approx(1.0, abs=1e-4)
        assert all(0.0 < p.predicted_probability < 1.0 for p in response.predictions)

    async def test_features_come_from_the_soccer_block(self, soccer_registry) -> None:
        repo = FakeRepo()
        predictor = make_predictor(soccer_registry, repo)
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        features = response.features_used
        assert features["home_attack_strength"] == 1.4
        assert features["away_attack_strength"] == 1.3
        assert features["home_form_points_last5"] == 12.0
        assert features["is_knockout"] == 1.0  # POSTSEASON knockout final
        assert features["competition_is_fifa_wc"] == 1.0
        assert features["home_rest_days"] == 4.0
        assert "home_offensive_rating" not in features
        assert repo.persisted_features == features

    async def test_all_three_markets_produce_rows(self, soccer_registry) -> None:
        predictor = make_predictor(soccer_registry, FakeRepo())
        request = PredictionRequest(
            game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["SPREAD", "TOTAL", "MONEYLINE"]
        )
        response = await predictor.create_predictions(request)
        sides = [(p.market_type, p.side) for p in response.predictions]
        assert sides == [
            ("SPREAD", "HOME"),
            ("TOTAL", "OVER"),
            ("MONEYLINE", "HOME"),
            ("MONEYLINE", "DRAW"),
            ("MONEYLINE", "AWAY"),
        ]
