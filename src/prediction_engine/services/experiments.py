"""Champion/challenger experiment orchestration (Phase 7 Wave 4).

Builds the graded comparison: both models' latest rows keyed by
(game, side, prop identity), intersected, joined to statistics-service
final scores, settled with core/settlement.py (pushes drop the pair), then
scored by core/experiments.py. Promotion delegates the atomic role flip to
ModelVersionRepository.promote and reloads the registry.
"""

import logging

import numpy as np

from prediction_engine.api.errors import NotFoundError, UnprocessableError
from prediction_engine.clients.statistics import StatisticsClient
from prediction_engine.core.experiments import ExperimentReport, build_report
from prediction_engine.core.model.registry import MARKET_TYPES, ModelRegistry
from prediction_engine.core.settlement import (
    DEFAULT_SIDE_BY_MARKET,
    is_three_way_moneyline_league,
    line_from_selection,
    scores_for_settlement,
    settle_outcome,
)
from prediction_engine.db.repository import (
    ModelVersionRecord,
    ModelVersionRepository,
    PredictionRecord,
    PredictionRepository,
)

logger = logging.getLogger(__name__)

_PairKey = tuple[str, str | None, str | None, str | None, float | None]


def _key(record: PredictionRecord) -> _PairKey:
    return (
        record.game_external_id,
        record.side,
        record.player_external_id,
        record.stat_type,
        record.prop_line,
    )


class ExperimentService:
    def __init__(
        self,
        model_repo: ModelVersionRepository,
        prediction_repo: PredictionRepository,
        statistics: StatisticsClient,
        registry: ModelRegistry,
        promotion_min_samples: int = 300,
    ) -> None:
        self._model_repo = model_repo
        self._prediction_repo = prediction_repo
        self._statistics = statistics
        self._registry = registry
        self._min_samples = promotion_min_samples

    async def _active_pair(self, sport: str, market_type: str) -> tuple[ModelVersionRecord, ModelVersionRecord]:
        champion = await self._model_repo.get_active(sport, market_type, role="champion")
        if champion is None:
            raise NotFoundError(f"No active champion for {sport} {market_type}")
        challenger = await self._model_repo.get_active(sport, market_type, role="challenger")
        if challenger is None:
            raise NotFoundError(f"No active challenger for {sport} {market_type}")
        return champion, challenger

    async def report(self, sport: str, market_type: str) -> ExperimentReport:
        """Score champion vs challenger over graded shadow pairs (FINAL games only)."""
        champion, challenger = await self._active_pair(sport, market_type)
        champion_rows = {_key(r): r for r in await self._prediction_repo.latest_rows_for_model(champion.id)}
        challenger_rows = {_key(r): r for r in await self._prediction_repo.latest_rows_for_model(challenger.id)}
        shared_keys = sorted(
            champion_rows.keys() & challenger_rows.keys(),
            key=lambda k: (k[0], k[1] or "", k[2] or "", k[3] or "", k[4] or 0.0),
        )

        scores: dict[str, tuple[int, int] | None] = {}
        leagues: dict[str, str] = {}
        champion_probs: list[float] = []
        challenger_probs: list[float] = []
        outcomes: list[float] = []
        for key in shared_keys:
            record = champion_rows[key]
            game_id = record.game_external_id
            if game_id not in scores:
                try:
                    game = await self._statistics.get_game(game_id)
                except Exception:  # noqa: BLE001 - an unfetchable game just stays ungraded
                    logger.warning("could not fetch game %s for experiment grading", game_id, exc_info=True)
                    scores[game_id] = None
                else:
                    scores[game_id] = scores_for_settlement(game)
                    leagues[game_id] = game.league
            settled = scores[game_id]
            if settled is None:
                continue
            side = record.side or DEFAULT_SIDE_BY_MARKET.get(record.market_type, "HOME")
            try:
                outcome = settle_outcome(
                    record.market_type,
                    side,
                    line_from_selection(record.market_type, record.selection),
                    settled[0],
                    settled[1],
                    three_way_moneyline=is_three_way_moneyline_league(leagues[game_id]),
                )
            except ValueError:
                logger.warning("unsettleable experiment row %s; skipping pair", record.id, exc_info=True)
                continue
            if outcome is None:  # push: no binary signal for either model
                continue
            champion_probs.append(record.predicted_probability)
            challenger_probs.append(challenger_rows[key].predicted_probability)
            outcomes.append(outcome)

        return build_report(
            sport=sport,
            market_type=market_type,
            champion_id=str(champion.id),
            challenger_id=str(challenger.id),
            champion_probs=np.asarray(champion_probs, dtype=np.float64),
            challenger_probs=np.asarray(challenger_probs, dtype=np.float64),
            outcomes=np.asarray(outcomes, dtype=np.float64),
            min_samples=self._min_samples,
        )

    async def promote(self, sport: str, market_type: str, force: bool) -> list[ModelVersionRecord]:
        """Validate the criteria (unless forced) and atomically flip roles.

        The unified model spans SPREAD/TOTAL/MONEYLINE (one shared
        artifact), so the flip covers all game markets together even though
        the criteria are evaluated on the requested market_type. Returns
        the new champion records.
        """
        report = await self.report(sport, market_type)
        if not report.promotion_ready and not force:
            raise UnprocessableError(
                f"challenger is not promotion-ready for {sport} {market_type}",
                details={"blockers": report.blockers},
            )
        records = await self._model_repo.promote(sport, MARKET_TYPES)
        await self._registry.reload_sport(sport)
        logger.info(
            "promoted challenger %s to champion for %s (%s markets, forced=%s)",
            records[0].version,
            sport,
            len(records),
            force,
        )
        return records
