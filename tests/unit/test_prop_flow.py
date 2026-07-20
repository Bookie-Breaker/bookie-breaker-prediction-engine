"""Player-prop prediction flow (Phase 7 Wave 3): the predictor prop branch
with a real trained SOCCER prop bootstrap and stubbed service clients --
OVER/UNDER complement pairs, the YES/NO path, contract 422s, persisted
player columns, and lazy registry prop bootstrap."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from prediction_engine.api.errors import NotFoundError, UnprocessableError
from prediction_engine.api.schemas import PredictionRequest, PropRequest
from prediction_engine.clients.simulation import (
    PlayerDistributions,
    PlayerSimulation,
    PlayerStatSimulation,
    SimulationResult,
    SimulationRun,
)
from prediction_engine.clients.statistics import Game, SoccerStats, StatBlocks, TeamRef, TeamStats
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.predictor import Predictor
from prediction_engine.core.training.synthetic import generate_soccer_prop_dataset, generate_soccer_synthetic_dataset
from prediction_engine.core.training.train import save_artifact, train_model
from prediction_engine.db.repository import ModelVersionRecord, PredictionRecord

GAME_ID = "game-epl-derby"
RUN_ID = "run-epl-derby"
PLAYER_ID = "11111111-2222-3333-4444-555555555555"


def make_game() -> Game:
    return Game(
        id=GAME_ID,
        league="EPL",
        status="SCHEDULED",
        home_team=TeamRef(id="t-lfc", name="Liverpool"),
        away_team=TeamRef(id="t-eve", name="Everton"),
        scheduled_start="2026-07-19T16:00:00Z",
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
            home_win_probability=0.55,
            away_win_probability=0.20,
            draw_probability=0.25,
            mean_margin=0.9,
            mean_total=2.9,
            spread_cover_probabilities={"-0.5": 0.55, "0.5": 0.80},
            total_over_probabilities={"2.5": 0.55, "3.5": 0.30},
        ),
    )


def make_player_distributions() -> PlayerDistributions:
    return PlayerDistributions(
        simulation_run_id=RUN_ID,
        game_id=GAME_ID,
        players={
            PLAYER_ID: PlayerSimulation(
                name="Mo Salah",
                team="HOME",
                stats={
                    "player_goal_scorer_anytime": PlayerStatSimulation(yes_probability=0.62),
                    "player_shots": PlayerStatSimulation(
                        distribution={"mean": 2.4, "std": 1.5},
                        over_probabilities={"1.5": 0.61, "2.5": 0.38, "3.5": 0.18},
                    ),
                },
            )
        },
    )


class FakeStatistics:
    async def get_game(self, game_id: str) -> Game:
        return make_game()

    async def get_team_stats(self, team_id: str, rolling_window: int | None = None) -> TeamStats:
        return TeamStats(
            team_id=team_id,
            season=2026,
            games_played=30,
            stats=StatBlocks(
                soccer=SoccerStats(
                    goals_for_per_match=1.9,
                    goals_against_per_match=1.0,
                    attack_strength=1.2,
                    defense_strength=0.9,
                    form_points_last5=10,
                )
            ),
        )

    async def list_recent_games(self, team_id: str, date_to: str, limit: int = 10) -> list[Game]:
        return []

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-07-19T00:00:00Z"


class FakeLines:
    async def best_lines(self, game_external_id: str, market_type: str | None = None) -> list[Any]:
        return []

    async def movement(self, game_external_id: str, market_type: str = "SPREAD") -> list[Any]:
        return []

    async def current_lines(self, **kwargs: Any) -> list[Any]:
        return []


class FakeReconciler:
    async def resolve(self, game: Game) -> str | None:
        return None


class FakeSimulation:
    def __init__(self, with_players: bool = True) -> None:
        self._with_players = with_players

    async def get_run(self, run_id: str) -> SimulationRun:
        return make_run()

    async def get_player_distributions(self, run_id: str) -> PlayerDistributions:
        if not self._with_players:
            raise NotFoundError(f"player distributions for simulation run {run_id} not found in simulation-engine")
        return make_player_distributions()


class FakeModelVersionRepo:
    def __init__(self) -> None:
        self.records: list[ModelVersionRecord] = []

    async def list_models(
        self, sport: str | None = None, market_type: str | None = None, is_active: bool | None = None
    ) -> list[ModelVersionRecord]:
        return [
            record
            for record in self.records
            if (sport is None or record.sport == sport)
            and (market_type is None or record.model_type == market_type)
            and (is_active is None or record.is_active == is_active)
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
        self.rows: list[dict[str, Any]] = []

    async def insert_predictions(
        self,
        rows: list[dict[str, Any]],
        features: dict[str, Any],
        feature_sources: dict[str, Any],
        row_features: list[dict[str, Any]] | None = None,
    ) -> list[PredictionRecord]:
        self.rows = rows
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
                player_external_id=row.get("player_external_id"),
                stat_type=row.get("stat_type"),
                prop_line=row.get("prop_line"),
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
def prop_registry(tmp_path_factory: pytest.TempPathFactory) -> ModelRegistry:
    """Registry with pre-trained SOCCER props + game artifacts on disk."""
    model_dir = tmp_path_factory.mktemp("prop-models")
    prop_result = train_model(
        generate_soccer_prop_dataset(n_rows=800),
        n_rounds=8,
        data_label="synthetic",
        sport="SOCCER",
        market="PLAYER_PROP",
    )
    save_artifact(prop_result, model_dir)
    game_result = train_model(
        generate_soccer_synthetic_dataset(n_games=160), n_rounds=8, data_label="synthetic", sport="SOCCER"
    )
    save_artifact(game_result, model_dir)
    return ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]


def make_predictor(registry: ModelRegistry, repo: FakeRepo, with_players: bool = True) -> Predictor:
    statistics = FakeStatistics()
    lines = FakeLines()
    reconciler = FakeReconciler()
    builder = FeatureBuilder(statistics, lines, reconciler)  # type: ignore[arg-type]
    return Predictor(
        statistics=statistics,  # type: ignore[arg-type]
        lines=lines,  # type: ignore[arg-type]
        simulation=FakeSimulation(with_players),  # type: ignore[arg-type]
        reconciler=reconciler,  # type: ignore[arg-type]
        features=builder,
        registry=registry,
        repo=repo,  # type: ignore[arg-type]
        redis_client=FakeRedis(),  # type: ignore[arg-type]
    )


def prop_request(**overrides: Any) -> PredictionRequest:
    prop_kwargs: dict[str, Any] = {
        "player_external_id": PLAYER_ID,
        "player_name": "Mo Salah",
        "stat_type": "player_shots",
        "line": 2.5,
        "side": None,
    }
    prop_kwargs.update(overrides)
    return PredictionRequest(
        game_id=GAME_ID, simulation_run_id=RUN_ID, market_types=[], props=[PropRequest(**prop_kwargs)]
    )


class TestPropOverUnder:
    async def test_emits_complement_pair_summing_to_one(self, prop_registry) -> None:
        repo = FakeRepo()
        predictor = make_predictor(prop_registry, repo)
        response = await predictor.create_predictions(prop_request())

        assert [p.side for p in response.predictions] == ["OVER", "UNDER"]
        assert all(p.market_type == "PLAYER_PROP" for p in response.predictions)
        over, under = response.predictions
        assert over.predicted_probability + under.predicted_probability == pytest.approx(1.0, abs=1e-4)
        assert over.simulation_probability == pytest.approx(0.38)
        assert under.simulation_probability == pytest.approx(0.62)
        assert 0.0 < over.predicted_probability < 1.0
        # complementary conformal intervals
        assert over.confidence_lower == pytest.approx(1.0 - under.confidence_upper, abs=1e-4)
        assert over.confidence_upper == pytest.approx(1.0 - under.confidence_lower, abs=1e-4)

    async def test_response_and_rows_carry_player_columns(self, prop_registry) -> None:
        repo = FakeRepo()
        predictor = make_predictor(prop_registry, repo)
        response = await predictor.create_predictions(prop_request(side="OVER"))

        assert len(response.predictions) == 1
        item = response.predictions[0]
        assert item.player_external_id == PLAYER_ID
        assert item.stat_type == "player_shots"
        assert item.prop_line == 2.5
        assert item.selection == "Mo Salah player_shots Over 2.5"
        persisted = repo.rows[0]
        assert persisted["market_type"] == "PLAYER_PROP"
        assert persisted["player_external_id"] == PLAYER_ID
        assert persisted["stat_type"] == "player_shots"
        assert persisted["prop_line"] == 2.5
        assert persisted["implied_probability"] is None  # no prop line feed yet
        assert persisted["edge"] is None

    async def test_props_combine_with_game_markets(self, prop_registry) -> None:
        repo = FakeRepo()
        predictor = make_predictor(prop_registry, repo)
        request = PredictionRequest(
            game_id=GAME_ID,
            simulation_run_id=RUN_ID,
            market_types=["TOTAL"],
            props=[PropRequest(player_external_id=PLAYER_ID, stat_type="player_shots", line=2.5, side="OVER")],
        )
        response = await predictor.create_predictions(request)
        assert [(p.market_type, p.side) for p in response.predictions] == [
            ("TOTAL", "OVER"),
            ("PLAYER_PROP", "OVER"),
        ]
        assert response.predictions[0].player_external_id is None


class TestPropYesNo:
    async def test_default_side_emits_single_yes_row(self, prop_registry) -> None:
        predictor = make_predictor(prop_registry, FakeRepo())
        response = await predictor.create_predictions(prop_request(stat_type="player_goal_scorer_anytime", line=None))
        assert [p.side for p in response.predictions] == ["YES"]
        item = response.predictions[0]
        assert item.simulation_probability == pytest.approx(0.62)
        assert item.prop_line is None
        assert item.selection == "Mo Salah player_goal_scorer_anytime Yes"

    async def test_no_side_is_the_renormalized_complement(self, prop_registry) -> None:
        predictor = make_predictor(prop_registry, FakeRepo())
        yes = (
            await predictor.create_predictions(
                prop_request(stat_type="player_goal_scorer_anytime", line=None, side="YES")
            )
        ).predictions[0]
        no = (
            await predictor.create_predictions(
                prop_request(stat_type="player_goal_scorer_anytime", line=None, side="NO")
            )
        ).predictions[0]
        assert yes.predicted_probability + no.predicted_probability == pytest.approx(1.0, abs=1e-4)
        assert no.simulation_probability == pytest.approx(0.38)

    async def test_over_under_side_on_yes_no_stat_is_422(self, prop_registry) -> None:
        predictor = make_predictor(prop_registry, FakeRepo())
        with pytest.raises(UnprocessableError, match="yes/no prop"):
            await predictor.create_predictions(
                prop_request(stat_type="player_goal_scorer_anytime", line=None, side="OVER")
            )


class TestPropContract422s:
    async def test_missing_player_is_422(self, prop_registry) -> None:
        predictor = make_predictor(prop_registry, FakeRepo())
        with pytest.raises(UnprocessableError, match="not in simulation run"):
            await predictor.create_predictions(prop_request(player_external_id="unknown-player"))

    async def test_unregistered_stat_is_422(self, prop_registry) -> None:
        predictor = make_predictor(prop_registry, FakeRepo())
        with pytest.raises(UnprocessableError, match="not a registered SOCCER prop stat"):
            await predictor.create_predictions(prop_request(stat_type="player_points"))

    async def test_stat_absent_from_sim_payload_is_422(self, prop_registry) -> None:
        # player_shots_on_target is a registered soccer stat, but the run
        # was captured without it
        predictor = make_predictor(prop_registry, FakeRepo())
        with pytest.raises(UnprocessableError, match="no player_shots_on_target distribution"):
            await predictor.create_predictions(prop_request(stat_type="player_shots_on_target", line=0.5))

    async def test_line_off_the_grid_is_422(self, prop_registry) -> None:
        predictor = make_predictor(prop_registry, FakeRepo())
        with pytest.raises(UnprocessableError, match="outside the simulated player_shots grid"):
            await predictor.create_predictions(prop_request(line=2.0))

    async def test_missing_line_is_422(self, prop_registry) -> None:
        predictor = make_predictor(prop_registry, FakeRepo())
        with pytest.raises(UnprocessableError, match="line is required"):
            await predictor.create_predictions(prop_request(line=None))

    async def test_yes_side_on_count_stat_is_422(self, prop_registry) -> None:
        predictor = make_predictor(prop_registry, FakeRepo())
        with pytest.raises(UnprocessableError, match="over/under prop"):
            await predictor.create_predictions(prop_request(side="YES"))

    async def test_run_without_player_capture_is_422(self, prop_registry) -> None:
        predictor = make_predictor(prop_registry, FakeRepo(), with_players=False)
        with pytest.raises(UnprocessableError, match="no player distributions"):
            await predictor.create_predictions(prop_request())


class TestRegistryPropBootstrap:
    async def test_bootstrap_trains_and_registers_props_artifact(self, tmp_path) -> None:
        repo = FakeModelVersionRepo()
        registry = ModelRegistry(repo, tmp_path, sports=["SOCCER"])  # type: ignore[arg-type]

        loaded = await registry.get_active("SOCCER", "PLAYER_PROP")
        assert loaded is not None
        assert loaded.record.model_type == "PLAYER_PROP"
        assert loaded.record.version.startswith("SOCCER_props_")
        artifacts = list((tmp_path / "soccer" / "props").iterdir())
        assert len(artifacts) == 1
        assert (artifacts[0] / "model.ubj").is_file()
        # exactly one PLAYER_PROP model_versions row was registered
        assert [r.model_type for r in repo.records] == ["PLAYER_PROP"]

    async def test_prop_and_game_bootstraps_are_independent(self, tmp_path) -> None:
        repo = FakeModelVersionRepo()
        registry = ModelRegistry(repo, tmp_path, sports=["SOCCER"])  # type: ignore[arg-type]

        prop_loaded = await registry.get_active("SOCCER", "PLAYER_PROP")
        # a registered prop model must not satisfy the game bootstrap
        game_loaded = await registry.get_active("SOCCER", "SPREAD")
        assert prop_loaded is not None and game_loaded is not None
        assert game_loaded.record.model_type == "SPREAD"
        assert game_loaded.record.version.startswith("SOCCER_unified_")
        assert {r.model_type for r in repo.records} == {"PLAYER_PROP", "SPREAD", "TOTAL", "MONEYLINE"}
        active_keys = registry.active_map()
        assert "SOCCER_PLAYER_PROP" in active_keys
        assert "SOCCER_SPREAD" in active_keys

    async def test_unregistered_sport_prop_bootstrap_raises(self, tmp_path) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), tmp_path, sports=["HOCKEY"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="prop"):
            await registry.get_active("HOCKEY", "PLAYER_PROP")
