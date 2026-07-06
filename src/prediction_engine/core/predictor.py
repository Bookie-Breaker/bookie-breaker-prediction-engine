"""Prediction orchestration: sim run + features -> calibrated predictions.

Market line selection: the current best line from lines-service determines
which spread/total line the prediction targets (selection strings use full
team names, matching lines-service). When lines are unavailable the line
nearest the simulation mean is used and implied probability is omitted.
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import redis.asyncio as aioredis

from prediction_engine.api.errors import DuplicateResourceError, NotFoundError, UnprocessableError
from prediction_engine.api.schemas import (
    EdgeItem,
    EdgesData,
    PredictionGroupData,
    PredictionItem,
    PredictionRequest,
    Side,
)
from prediction_engine.clients.lines import BestLine, LinesClient
from prediction_engine.clients.reconcile import GameReconciler
from prediction_engine.clients.simulation import SimulationClient, SimulationRun
from prediction_engine.clients.statistics import Game, StatisticsClient
from prediction_engine.core.edges import edge_percentage
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.features.registry import FeatureMap
from prediction_engine.core.leagues import THREE_WAY_MONEYLINE_SPORTS, sport_for_league
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.db.repository import PredictionRecord, PredictionRepository
from prediction_engine.events.publisher import publish_prediction_completed

logger = logging.getLogger(__name__)

IDEMPOTENCY_PREFIX = "pred:idempotency:"


def _interpolate(grid: dict[str, float], target: float) -> tuple[float, float]:
    """Return (nearest_line, interpolated probability) from a string-keyed grid."""
    points = sorted((float(k), v) for k, v in grid.items())
    lines = np.array([p[0] for p in points])
    probs = np.array([p[1] for p in points])
    nearest = float(lines[int(np.argmin(np.abs(lines - target)))])
    prob = float(np.interp(target, lines, probs))
    return nearest, prob


def _half_line(value: float) -> float:
    """Snap to the nearest half-point line (never a whole number)."""
    return round(value - 0.5) + 0.5


# Default side per market for prediction rows persisted before Phase 6
# (side is NULL on those rows; the single row per market was always the
# HOME/OVER perspective).
_DEFAULT_SIDE = {"SPREAD": "HOME", "MONEYLINE": "HOME", "TOTAL": "OVER"}


@dataclass(frozen=True)
class _RowInputs:
    """Per-prediction-row inputs derived from one market of one game."""

    side: str
    selection: str
    sim_probability: float
    implied_probability: float | None


class Predictor:
    def __init__(
        self,
        statistics: StatisticsClient,
        lines: LinesClient,
        simulation: SimulationClient,
        reconciler: GameReconciler,
        features: FeatureBuilder,
        registry: ModelRegistry,
        repo: PredictionRepository,
        redis_client: "aioredis.Redis",
        idempotency_ttl: int = 86_400,
    ) -> None:
        self._statistics = statistics
        self._lines = lines
        self._simulation = simulation
        self._reconciler = reconciler
        self._features = features
        self._registry = registry
        self._repo = repo
        self._redis = redis_client
        self._idempotency_ttl = idempotency_ttl

    async def _best_lines_by_market(self, lines_game_id: str | None) -> dict[tuple[str, str], BestLine]:
        """Best line per (market type, side), when lines are available.

        SPREAD keeps the HOME side and TOTAL keeps OVER (the single
        perspective each prediction row targets); MONEYLINE keeps every side
        (HOME/AWAY/DRAW) so three-way rows each carry their own
        market-implied probability (ADR-027).
        """
        if lines_game_id is None:
            return {}
        try:
            best = await self._lines.best_lines(lines_game_id)
        except Exception:  # noqa: BLE001 - lines are optional at predict time
            logger.warning("best lines unavailable for %s", lines_game_id, exc_info=True)
            return {}
        result: dict[tuple[str, str], BestLine] = {}
        for line in best:
            if (
                line.market_type == "MONEYLINE"
                and line.side in ("HOME", "AWAY", "DRAW")
                or line.market_type == "SPREAD"
                and line.side == "HOME"
                or line.market_type == "TOTAL"
                and line.side == "OVER"
            ):
                result[(line.market_type, line.side)] = line
        return result

    def _market_inputs(
        self,
        market: str,
        run: SimulationRun,
        game: Game,
        best_lines: dict[tuple[str, str], BestLine],
        three_way_moneyline: bool,
    ) -> list[_RowInputs]:
        """Return the prediction-row inputs a market emits (one per side).

        Two-way markets emit a single HOME (spread/moneyline) or OVER
        (total) row, exactly as before Phase 6. Three-way moneylines
        (ADR-027) emit HOME/DRAW/AWAY rows: draw from the simulation's
        draw_probability, away as the floored complement.
        """
        result = run.result

        def implied(side: str) -> float | None:
            best = best_lines.get((market, side))
            return best.implied_probability if best is not None else None

        if market == "MONEYLINE":
            home = _RowInputs("HOME", f"{game.home_team.name} ML", result.home_win_probability, implied("HOME"))
            if not three_way_moneyline:
                return [home]
            draw_prob = result.draw_probability
            away_prob = max(0.0, 1.0 - result.home_win_probability - draw_prob)
            return [
                home,
                _RowInputs("DRAW", "Draw", draw_prob, implied("DRAW")),
                _RowInputs("AWAY", f"{game.away_team.name} ML", away_prob, implied("AWAY")),
            ]

        if market == "SPREAD":
            best_line = best_lines.get(("SPREAD", "HOME"))
            target = best_line.line_value if best_line and best_line.line_value is not None else None
            if target is None:
                target = _half_line(-result.mean_margin)
            line, sim_prob = _interpolate(result.spread_cover_probabilities, target)
            return [_RowInputs("HOME", f"{game.home_team.name} {line:+g}", sim_prob, implied("HOME"))]

        best_line = best_lines.get(("TOTAL", "OVER"))
        target = best_line.line_value if best_line and best_line.line_value is not None else None
        if target is None:
            target = _half_line(result.mean_total)
        line, sim_prob = _interpolate(result.total_over_probabilities, target)
        return [_RowInputs("OVER", f"Over {line:g}", sim_prob, implied("OVER"))]

    async def create_predictions(
        self, request: PredictionRequest, idempotency_key: str | None = None
    ) -> PredictionGroupData:
        body_hash = hashlib.sha256(json.dumps(request.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()
        if idempotency_key is not None:
            stored = await self._redis.get(f"{IDEMPOTENCY_PREFIX}{idempotency_key}")
            if stored is not None:
                record = json.loads(stored)
                if record["body_hash"] != body_hash:
                    raise DuplicateResourceError("X-Idempotency-Key was already used with a different request body")
                return PredictionGroupData.model_validate(record["response"])

        game = await self._statistics.get_game(request.game_id)
        run = await self._simulation.get_run(request.simulation_run_id)
        if run.game_id != request.game_id:
            raise UnprocessableError(
                f"Simulation run {request.simulation_run_id} belongs to game {run.game_id}, not {request.game_id}"
            )

        try:
            sport = sport_for_league(game.league)
        except ValueError as exc:
            raise UnprocessableError(str(exc)) from exc
        three_way_moneyline = sport in THREE_WAY_MONEYLINE_SPORTS

        bundle = await self._features.build(game)
        bundle.sources["simulation_run_id"] = request.simulation_run_id
        best_lines = await self._best_lines_by_market(bundle.lines_game_external_id)

        rows: list[dict[str, Any]] = []
        for market in request.market_types:
            try:
                loaded = await self._registry.get_active(sport, market)
            except ValueError as exc:  # sport has no bootstrap path yet (later league wave)
                raise UnprocessableError(str(exc)) from exc
            if loaded is None:
                raise UnprocessableError(f"No active model for {sport} {market}")

            calibrated: list[float] = []
            importances: list[dict[str, float]] = []
            row_inputs = self._market_inputs(market, run, game, best_lines, three_way_moneyline)
            for inputs in row_inputs:
                features: FeatureMap = dict(bundle.features)
                features["sim_probability"] = inputs.sim_probability
                features["sim_margin_mean"] = run.result.mean_margin
                features["sim_total_mean"] = run.result.mean_total
                features["sim_converged"] = 1.0 if run.converged else 0.0
                features["market_is_spread"] = 1.0 if market == "SPREAD" else 0.0
                features["market_is_total"] = 1.0 if market == "TOTAL" else 0.0
                features["market_is_moneyline"] = 1.0 if market == "MONEYLINE" else 0.0

                adjustment = loaded.bundle.model.predict_adjustment(features)
                raw = float(np.clip(inputs.sim_probability + adjustment, 0.01, 0.99))
                calibrated.append(loaded.bundle.calibrator.apply(raw))
                importances.append(loaded.bundle.model.feature_importance(features))

            if len(calibrated) > 1:
                # Independently calibrated three-way probabilities are
                # renormalized to a proper distribution (ADR-027).
                total = sum(calibrated)
                calibrated = [value / total for value in calibrated]

            for inputs, value, importance in zip(row_inputs, calibrated, importances, strict=True):
                predicted = round(value, 5)
                lower, upper = loaded.bundle.conformal.interval(predicted)
                implied = inputs.implied_probability
                rows.append(
                    {
                        "game_external_id": request.game_id,
                        "model_version_id": loaded.record.id,
                        "league": game.league,
                        "market_type": market,
                        "side": inputs.side,
                        "selection": inputs.selection,
                        "predicted_probability": predicted,
                        "simulation_probability": round(inputs.sim_probability, 5),
                        "implied_probability": round(implied, 5) if implied is not None else None,
                        "edge": round(predicted - implied, 5) if implied is not None else None,
                        "confidence_lower": lower,
                        "confidence_upper": upper,
                        "feature_importance": importance,
                    }
                )

        records = await self._repo.insert_predictions(
            rows,
            features={k: v for k, v in bundle.features.items()},
            feature_sources=dict(bundle.sources),
        )

        edges_found = sum(1 for r in records if r.edge is not None and r.edge > 0)
        await publish_prediction_completed(
            self._redis,
            batch_id=str(uuid.uuid4()),
            game_ids=[request.game_id],
            league=game.league,
            market_types=list(request.market_types),
            predictions_count=len(records),
            edges_found=edges_found,
        )

        response = PredictionGroupData(
            game_id=request.game_id,
            simulation_run_id=request.simulation_run_id,
            predictions=[_to_item(record) for record in records],
            features_used=dict(bundle.features),
            feature_source_versions=dict(bundle.sources),
        )
        if idempotency_key is not None:
            await self._redis.set(
                f"{IDEMPOTENCY_PREFIX}{idempotency_key}",
                json.dumps({"body_hash": body_hash, "response": response.model_dump(mode="json")}),
                ex=self._idempotency_ttl,
            )
        return response

    async def edges_for_game(self, game_id: str, min_edge: float, market_type: str | None) -> EdgesData:
        markets = [market_type] if market_type else None
        records = await self._repo.latest_for_game(game_id, market_types=markets)
        if not records:
            raise NotFoundError(f"No predictions exist for game {game_id}")

        game = await self._statistics.get_game(game_id)
        lines_id = await self._reconciler.resolve(game)
        best_lines = await self._best_lines_by_market(lines_id)

        edges: list[EdgeItem] = []
        for record in records:
            side = record.side or _DEFAULT_SIDE.get(record.market_type, "HOME")
            best = best_lines.get((record.market_type, side))
            if best is None or best.implied_probability is None:
                continue
            edge_pct = edge_percentage(record.predicted_probability, best.implied_probability)
            if edge_pct < min_edge:
                continue
            edges.append(
                EdgeItem(
                    prediction_id=str(record.id),
                    market_type=record.market_type,
                    selection=record.selection,
                    predicted_probability=record.predicted_probability,
                    implied_probability=round(best.implied_probability, 4),
                    edge_percentage=edge_pct,
                    best_odds_american=best.best_odds_american,
                    sportsbook_key=best.sportsbook_key,
                )
            )
        edges.sort(key=lambda e: -e.edge_percentage)
        return EdgesData(game_id=game_id, edges=edges)


def _to_item(record: PredictionRecord) -> PredictionItem:
    return PredictionItem(
        id=str(record.id),
        market_type=record.market_type,
        # the check constraint guarantees the side vocabulary at the DB layer
        side=cast("Side | None", record.side),
        selection=record.selection,
        predicted_probability=record.predicted_probability,
        simulation_probability=record.simulation_probability,
        adjustment_magnitude=round(
            abs(record.predicted_probability - (record.simulation_probability or record.predicted_probability)), 5
        ),
        confidence_lower=record.confidence_lower,
        confidence_upper=record.confidence_upper,
        model_version_id=str(record.model_version_id),
        feature_importance=record.feature_importance,
        created_at=record.created_at.isoformat().replace("+00:00", "Z"),
    )
