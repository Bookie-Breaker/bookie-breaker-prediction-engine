"""Baseball prediction flow: the (sport, market) registry end-to-end for an
MLB game with probable pitchers and a real trained BASEBALL bootstrap model
(stubbed service clients). The moneyline stays two-way: one HOME row."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from prediction_engine.api.schemas import PredictionRequest
from prediction_engine.clients.lines import BestLine, LineMovement, LineSnapshot
from prediction_engine.clients.simulation import SimulationResult, SimulationRun
from prediction_engine.clients.statistics import BaseballStats, Game, ProbablePitcher, StatBlocks, TeamRef, TeamStats
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.predictor import Predictor
from prediction_engine.core.training.synthetic import generate_baseball_synthetic_dataset
from prediction_engine.core.training.train import save_artifact, train_model
from prediction_engine.db.repository import ModelVersionRecord, PredictionRecord

GAME_ID = "game-mlb-nyy-bos"
RUN_ID = "run-mlb-nyy-bos"


def make_game() -> Game:
    return Game(
        id=GAME_ID,
        league="MLB",
        status="SCHEDULED",
        home_team=TeamRef(id="t-nyy", name="New York Yankees"),
        away_team=TeamRef(id="t-bos", name="Boston Red Sox"),
        scheduled_start="2026-07-10T23:05:00Z",
        season=2026,
        season_type="REGULAR",
        home_probable_pitcher=ProbablePitcher(
            name="Ace Home", external_id="p-1", throws="R", era=2.90, fip=3.05, k_bb_pct=0.22
        ),
        away_probable_pitcher=ProbablePitcher(
            name="Lefty Away", external_id="p-2", throws="L", era=4.35, fip=4.20, k_bb_pct=0.12
        ),
    )


def make_run() -> SimulationRun:
    return SimulationRun(
        simulation_run_id=RUN_ID,
        game_id=GAME_ID,
        status="completed",
        converged=True,
        result=SimulationResult(
            home_win_probability=0.58,
            away_win_probability=0.42,
            mean_margin=0.7,
            mean_total=8.6,
            spread_cover_probabilities={"-1.5": 0.40, "1.5": 0.72},
            total_over_probabilities={"8.5": 0.49, "9.5": 0.36},
        ),
    )


def team_stats(team_id: str, woba: float) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=88,
        stats=StatBlocks(
            baseball=BaseballStats(
                runs_scored_per_game=4.9,
                runs_allowed_per_game=4.2,
                team_woba=woba,
                team_era=3.95,
                team_fip=4.00,
                bullpen_era=3.70,
            )
        ),
    )


class FakeStatistics:
    async def get_game(self, game_id: str) -> Game:
        return make_game()

    async def get_team_stats(self, team_id: str, rolling_window: int | None = None) -> TeamStats:
        return team_stats(team_id, woba=0.335 if team_id == "t-nyy" else 0.320)

    async def list_recent_games(self, team_id: str, date_to: str, limit: int = 10) -> list[Game]:
        return [
            Game(
                id=f"prev-{team_id}",
                league="MLB",
                status="FINAL",
                home_team=TeamRef(id=team_id),
                away_team=TeamRef(id="x"),
                scheduled_start="2026-07-09T23:05:00Z",
            )
        ]

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-07-10T00:00:00Z"


class FakeLines:
    async def best_lines(self, game_external_id: str, market_type: str | None = None) -> list[BestLine]:
        return [
            BestLine(
                market_type="MONEYLINE",
                selection="New York Yankees",
                side="HOME",
                best_odds_american=-135,
                implied_probability=0.57,
                sportsbook_key="draftkings",
            ),
            BestLine(
                market_type="MONEYLINE",
                selection="Boston Red Sox",
                side="AWAY",
                best_odds_american=+115,
                implied_probability=0.47,
                sportsbook_key="fanduel",
            ),
            BestLine(
                market_type="SPREAD",
                selection="New York Yankees -1.5",
                side="HOME",
                line_value=-1.5,
                best_odds_american=+140,
                implied_probability=0.42,
                sportsbook_key="draftkings",
            ),
            BestLine(
                market_type="TOTAL",
                selection="Over 8.5",
                side="OVER",
                line_value=8.5,
                best_odds_american=-110,
                implied_probability=0.52,
                sportsbook_key="fanduel",
            ),
        ]

    async def movement(self, game_external_id: str, market_type: str = "SPREAD") -> list[LineMovement]:
        return [LineMovement(total_movement=0.5)]

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        return [LineSnapshot(id="s1", game_id="ext-mlb", sportsbook_key="dk", side="HOME", line_value=-1.5)]


class FakeReconciler:
    async def resolve(self, game: Game) -> str | None:
        return "ext-mlb"


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
def baseball_registry(tmp_path_factory: pytest.TempPathFactory) -> ModelRegistry:
    model_dir = tmp_path_factory.mktemp("baseball-models")
    result = train_model(
        generate_baseball_synthetic_dataset(n_games=240), n_rounds=8, data_label="synthetic", sport="BASEBALL"
    )
    save_artifact(result, model_dir)
    return ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]


def make_predictor(baseball_registry: ModelRegistry, repo: FakeRepo) -> Predictor:
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
        registry=baseball_registry,
        repo=repo,  # type: ignore[arg-type]
        redis_client=FakeRedis(),  # type: ignore[arg-type]
    )


class TestBaseballTwoWayFlow:
    async def test_all_three_markets_produce_rows_with_sides(self, baseball_registry) -> None:
        predictor = make_predictor(baseball_registry, FakeRepo())
        request = PredictionRequest(
            game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["SPREAD", "TOTAL", "MONEYLINE"]
        )
        response = await predictor.create_predictions(request)

        sides = [(p.market_type, p.side) for p in response.predictions]
        assert sides == [("SPREAD", "HOME"), ("TOTAL", "OVER"), ("MONEYLINE", "HOME")]
        assert all(0.0 < p.predicted_probability < 1.0 for p in response.predictions)

    async def test_moneyline_is_a_single_home_row_not_three_way(self, baseball_registry) -> None:
        predictor = make_predictor(baseball_registry, FakeRepo())
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        assert len(response.predictions) == 1
        prediction = response.predictions[0]
        assert prediction.side == "HOME"
        assert prediction.selection == "New York Yankees ML"
        assert prediction.simulation_probability == 0.58
        # no DRAW/AWAY rows for baseball (two-way, not ADR-027 three-way)
        assert not any(p.side in ("DRAW", "AWAY") for p in response.predictions)

    async def test_spread_targets_the_market_run_line(self, baseball_registry) -> None:
        predictor = make_predictor(baseball_registry, FakeRepo())
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["SPREAD"])
        response = await predictor.create_predictions(request)

        prediction = response.predictions[0]
        assert prediction.selection == "New York Yankees -1.5"
        assert prediction.simulation_probability == 0.40

    async def test_features_come_from_the_baseball_registry(self, baseball_registry) -> None:
        repo = FakeRepo()
        predictor = make_predictor(baseball_registry, repo)
        request = PredictionRequest(game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        features = response.features_used
        assert features["home_team_woba"] == 0.335
        assert features["away_team_woba"] == 0.32
        assert features["home_starter_fip"] == 3.05
        assert features["away_starter_fip"] == 4.20
        assert features["starter_fip_diff"] == pytest.approx(3.05 - 4.20)
        assert features["home_starter_announced"] == 1.0
        assert features["away_starter_announced"] == 1.0
        assert features["league_is_mlb"] == 1.0
        assert features["home_rest_days"] == 0.0  # played yesterday
        assert "home_offensive_rating" not in features
        assert "sim_draw_probability" not in features
        assert repo.persisted_features == features
