"""Assemble real training data from stored predictions + game outcomes (Wave 4).

Joins the latest primary prediction row per (game, market, side) to its
stored feature vector (the exact vector the model saw at predict time) and
to the game's final score from statistics-service, settling each row into a
binary outcome with core/settlement.py. Push/void rows are dropped -- they
carry no binary signal. Games that are not FINAL (or lack scores) are
skipped and retried by a later run.

Player-prop rows are EXCLUDED in v1: props settle against per-player box
scores, and joining the canonical stat extraction (the emulator's prop
grading) into this pipeline is deferred until enough graded prop volume
exists to retrain on; POST /models/retrain 422s for market="player_prop".

Season is derived from the game date (scheduled_start year, preferring the
statistics-service season field when it is populated), which is what the
walk-forward split in train_model splits on.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from prediction_engine.clients.statistics import Game, StatisticsClient
from prediction_engine.core.features.registry import FeatureMap
from prediction_engine.core.leagues import LEAGUE_TO_SPORT, model_key_for_league
from prediction_engine.core.settlement import (
    DEFAULT_SIDE_BY_MARKET,
    is_three_way_moneyline_league,
    line_from_selection,
    scores_for_settlement,
    settle_outcome,
)
from prediction_engine.core.training.dataset import TrainingSet
from prediction_engine.db.repository import PredictionRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssembledTraining:
    dataset: TrainingSet
    rows: int
    games_graded: int
    games_pending: int
    rows_pushed: int
    rows_skipped: int


def leagues_for_model_key(model_key: str) -> list[str]:
    """Leagues whose predictions train this model key (pooled sports span
    several leagues; NCAA_BB maps to its own key, see core/leagues.py)."""
    return [league for league in LEAGUE_TO_SPORT if model_key_for_league(league) == model_key]


def _season_for_game(game: Game) -> int:
    if game.season > 0:
        return game.season
    if len(game.scheduled_start) >= 4 and game.scheduled_start[:4].isdigit():
        return int(game.scheduled_start[:4])
    raise ValueError(f"cannot derive a season for game {game.id}")


def _feature_map(features: dict[str, Any]) -> FeatureMap:
    return {str(name): (None if value is None else float(value)) for name, value in features.items()}


async def assemble_training_rows(
    sport: str,
    market_types: list[str],
    prediction_repo: PredictionRepository,
    statistics: StatisticsClient,
) -> AssembledTraining:
    """Build a TrainingSet for a sport's unified game-market model.

    ``sport`` is the model key (sport name, or NCAA_BB for its
    single-league model).
    """
    stored = await prediction_repo.training_rows(leagues_for_model_key(sport), market_types)

    games: dict[str, Game] = {}
    scores: dict[str, tuple[int, int] | None] = {}
    for game_id in {record.game_external_id for record, _ in stored}:
        try:
            game = games[game_id] = await statistics.get_game(game_id)
        except Exception:  # noqa: BLE001 - one unfetchable game must not sink the batch
            logger.warning("could not fetch game %s for outcome assembly; skipping", game_id, exc_info=True)
            scores[game_id] = None
            continue
        scores[game_id] = scores_for_settlement(game)

    features: list[FeatureMap] = []
    sim_probs: list[float] = []
    outcomes: list[float] = []
    seasons: list[int] = []
    rows_pushed = 0
    rows_skipped = 0
    for record, stored_features in stored:
        settled = scores.get(record.game_external_id)
        if settled is None or record.simulation_probability is None:
            rows_skipped += 1
            continue
        home_score, away_score = settled
        game = games[record.game_external_id]
        side = record.side or DEFAULT_SIDE_BY_MARKET.get(record.market_type, "HOME")
        try:
            outcome = settle_outcome(
                record.market_type,
                side,
                line_from_selection(record.market_type, record.selection),
                home_score,
                away_score,
                three_way_moneyline=is_three_way_moneyline_league(game.league),
            )
            season = _season_for_game(game)
        except ValueError:
            logger.warning("unsettleable prediction row %s; skipping", record.id, exc_info=True)
            rows_skipped += 1
            continue
        if outcome is None:
            rows_pushed += 1
            continue
        features.append(_feature_map(stored_features))
        sim_probs.append(record.simulation_probability)
        outcomes.append(outcome)
        seasons.append(season)

    dataset = TrainingSet(
        features=features,
        sim_probs=np.asarray(sim_probs, dtype=np.float64),
        outcomes=np.asarray(outcomes, dtype=np.float64),
        seasons=np.asarray(seasons, dtype=np.int64),
    )
    graded_games = {gid for gid, s in scores.items() if s is not None}
    return AssembledTraining(
        dataset=dataset,
        rows=len(features),
        games_graded=len(graded_games),
        games_pending=len(scores) - len(graded_games),
        rows_pushed=rows_pushed,
        rows_skipped=rows_skipped,
    )
