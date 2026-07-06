"""Three-way moneyline emission tests (ADR-027) with stubbed dependencies."""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from prediction_engine.api.errors import UnprocessableError
from prediction_engine.api.schemas import PredictionRequest
from prediction_engine.clients.lines import BestLine
from prediction_engine.clients.simulation import SimulationResult, SimulationRun
from prediction_engine.clients.statistics import Game, TeamRef
from prediction_engine.core.features.builder import FeatureBundle
from prediction_engine.core.predictor import Predictor
from prediction_engine.db.repository import PredictionRecord

HOME_TEAM = "Argentina"
AWAY_TEAM = "France"


def make_game(league: str) -> Game:
    return Game(
        id="game-1",
        league=league,
        status="SCHEDULED",
        home_team=TeamRef(id="t-home", name=HOME_TEAM),
        away_team=TeamRef(id="t-away", name=AWAY_TEAM),
        scheduled_start="2026-07-10T00:00:00Z",
    )


def make_run(home_win: float, draw: float) -> SimulationRun:
    return SimulationRun(
        simulation_run_id="run-1",
        game_id="game-1",
        status="completed",
        converged=True,
        result=SimulationResult(
            home_win_probability=home_win,
            away_win_probability=max(0.0, 1.0 - home_win - draw),
            draw_probability=draw,
            mean_margin=1.2,
            mean_total=2.6,
            spread_cover_probabilities={"-0.5": 0.5, "-1.5": 0.35},
            total_over_probabilities={"2.5": 0.55, "3.5": 0.3},
        ),
    )


class FakeStatistics:
    def __init__(self, game: Game) -> None:
        self._game = game

    async def get_game(self, game_id: str) -> Game:
        return self._game


class FakeSimulation:
    def __init__(self, run: SimulationRun) -> None:
        self._run = run

    async def get_run(self, run_id: str) -> SimulationRun:
        return self._run


class FakeLines:
    def __init__(self, best: list[BestLine]) -> None:
        self._best = best

    async def best_lines(self, game_external_id: str, market_type: str | None = None) -> list[BestLine]:
        return self._best


class FakeFeatureBuilder:
    def __init__(self, lines_id: str | None) -> None:
        self._lines_id = lines_id

    async def build(self, game: Game) -> FeatureBundle:
        return FeatureBundle(
            features={"net_rating_diff": 1.0}, sources={"stats": None}, lines_game_external_id=self._lines_id
        )


class FakeModel:
    def predict_adjustment(self, features: dict[str, Any]) -> float:
        return 0.0

    def feature_importance(self, features: dict[str, Any]) -> dict[str, float]:
        return {"sim_probability": 1.0}


class FakeCalibrator:
    """Shifts every probability up so three-way triples need renormalization."""

    def __init__(self, shift: float) -> None:
        self._shift = shift

    def apply(self, probability: float) -> float:
        return min(max(probability + self._shift, 0.001), 0.999)


class FakeConformal:
    def interval(self, probability: float) -> tuple[float, float]:
        return max(probability - 0.05, 0.0), min(probability + 0.05, 1.0)


class FakeRegistry:
    def __init__(self, shift: float = 0.1) -> None:
        bundle = SimpleNamespace(model=FakeModel(), calibrator=FakeCalibrator(shift), conformal=FakeConformal())
        self._loaded = SimpleNamespace(record=SimpleNamespace(id=uuid.uuid4()), bundle=bundle)

    async def get_active(self, sport: str, market_type: str) -> Any:
        if sport in ("BASKETBALL", "SOCCER"):
            return self._loaded
        raise ValueError(f"no synthetic generator registered for {sport}; added in its league wave")


class FakeRepo:
    async def insert_predictions(
        self, rows: list[dict[str, Any]], features: dict[str, Any], feature_sources: dict[str, Any]
    ) -> list[PredictionRecord]:
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


def make_predictor(
    league: str, run: SimulationRun, best: list[BestLine] | None = None, shift: float = 0.1
) -> Predictor:
    lines_id = "ext-1" if best is not None else None
    return Predictor(
        statistics=FakeStatistics(make_game(league)),  # type: ignore[arg-type]
        lines=FakeLines(best or []),  # type: ignore[arg-type]
        simulation=FakeSimulation(run),  # type: ignore[arg-type]
        reconciler=SimpleNamespace(),  # type: ignore[arg-type]
        features=FakeFeatureBuilder(lines_id),  # type: ignore[arg-type]
        registry=FakeRegistry(shift),  # type: ignore[arg-type]
        repo=FakeRepo(),  # type: ignore[arg-type]
        redis_client=FakeRedis(),  # type: ignore[arg-type]
    )


def moneyline_best_lines() -> list[BestLine]:
    def line(side: str, selection: str, implied: float) -> BestLine:
        return BestLine(
            market_type="MONEYLINE",
            selection=selection,
            side=side,
            best_odds_american=-110,
            implied_probability=implied,
            sportsbook_key="draftkings",
        )

    return [
        line("HOME", HOME_TEAM, 0.48),
        line("DRAW", "Draw", 0.30),
        line("AWAY", AWAY_TEAM, 0.28),
        # UNDER totals are never prediction targets and must be ignored
        BestLine(market_type="TOTAL", selection="Under 2.5", side="UNDER", implied_probability=0.5),
    ]


class TestThreeWayMoneyline:
    async def test_soccer_moneyline_emits_home_draw_away(self) -> None:
        predictor = make_predictor("FIFA_WC", make_run(home_win=0.5, draw=0.25), best=moneyline_best_lines())
        request = PredictionRequest(game_id="game-1", simulation_run_id="run-1", market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        assert [p.side for p in response.predictions] == ["HOME", "DRAW", "AWAY"]
        assert [p.selection for p in response.predictions] == [f"{HOME_TEAM} ML", "Draw", f"{AWAY_TEAM} ML"]
        assert [p.simulation_probability for p in response.predictions] == [0.5, 0.25, 0.25]
        # FakeCalibrator shifts each of (0.5, 0.25, 0.25) by +0.1 -> the
        # triple sums to 1.3 and must be renormalized to a distribution
        assert sum(p.predicted_probability for p in response.predictions) == pytest.approx(1.0, abs=1e-4)
        assert response.predictions[0].predicted_probability == pytest.approx(0.6 / 1.3, abs=1e-4)
        assert response.predictions[1].predicted_probability == pytest.approx(0.35 / 1.3, abs=1e-4)

    async def test_each_side_gets_its_own_implied_probability(self) -> None:
        predictor = make_predictor("EPL", make_run(home_win=0.5, draw=0.25), best=moneyline_best_lines())
        best_by_side = await predictor._best_lines_by_market("ext-1")

        # MONEYLINE keeps every side; the UNDER total is never a target
        assert set(best_by_side) == {("MONEYLINE", "HOME"), ("MONEYLINE", "DRAW"), ("MONEYLINE", "AWAY")}
        rows = predictor._market_inputs(
            "MONEYLINE", make_run(home_win=0.5, draw=0.25), make_game("EPL"), best_by_side, three_way_moneyline=True
        )
        assert [r.implied_probability for r in rows] == [0.48, 0.30, 0.28]

    async def test_away_probability_floors_at_zero(self) -> None:
        predictor = make_predictor("FIFA_WC", make_run(home_win=0.9, draw=0.2))
        request = PredictionRequest(game_id="game-1", simulation_run_id="run-1", market_types=["MONEYLINE"])
        response = await predictor.create_predictions(request)

        away = response.predictions[2]
        assert away.side == "AWAY"
        assert away.simulation_probability == 0.0
        assert away.predicted_probability > 0  # clipped, then renormalized

    async def test_two_way_sports_are_unchanged_single_home_row(self) -> None:
        predictor = make_predictor("NBA", make_run(home_win=0.75, draw=0.0))
        request = PredictionRequest(game_id="game-1", simulation_run_id="run-1")
        response = await predictor.create_predictions(request)

        by_market = {p.market_type: p for p in response.predictions}
        assert len(response.predictions) == 3
        assert by_market["MONEYLINE"].side == "HOME"
        assert by_market["MONEYLINE"].selection == f"{HOME_TEAM} ML"
        # no renormalization: exactly the calibrator output
        assert by_market["MONEYLINE"].predicted_probability == pytest.approx(0.85)
        assert by_market["SPREAD"].side == "HOME"
        assert by_market["TOTAL"].side == "OVER"

    async def test_unknown_league_is_unprocessable(self) -> None:
        predictor = make_predictor("XFL", make_run(home_win=0.5, draw=0.0))
        request = PredictionRequest(game_id="game-1", simulation_run_id="run-1")
        with pytest.raises(UnprocessableError, match="unknown league 'XFL'"):
            await predictor.create_predictions(request)

    async def test_sport_without_model_is_unprocessable(self) -> None:
        predictor = make_predictor("NHL", make_run(home_win=0.5, draw=0.0))
        request = PredictionRequest(game_id="game-1", simulation_run_id="run-1")
        with pytest.raises(UnprocessableError, match="no synthetic generator registered for HOCKEY"):
            await predictor.create_predictions(request)
