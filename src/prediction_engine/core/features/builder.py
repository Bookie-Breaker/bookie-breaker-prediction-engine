"""Assemble the market-independent feature map from upstream services.

Missing data becomes None (NaN to XGBoost, which handles missing natively).
Lines-service features are best-effort: when the game cannot be reconciled
to a lines-service external id, market features are None rather than
failing the prediction.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from prediction_engine.clients.lines import LinesClient
from prediction_engine.clients.reconcile import GameReconciler
from prediction_engine.clients.statistics import Game, StatisticsClient, TeamStats
from prediction_engine.core.features.injuries import MAX_PLAYERS_CONSIDERED, injury_impact
from prediction_engine.core.features.market import market_features
from prediction_engine.core.features.registry import FeatureMap
from prediction_engine.core.features.situational import rest_features

logger = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    features: FeatureMap
    sources: dict[str, str | None] = field(default_factory=dict)
    lines_game_external_id: str | None = None


def _game_date(game: Game) -> date:
    try:
        return datetime.fromisoformat(game.scheduled_start.replace("Z", "+00:00")).date()
    except ValueError:
        return date.today()


def _split_diff(stats: TeamStats) -> float | None:
    splits = stats.home_away_splits
    if splits is None:
        return None
    home_net = splits.home.points_per_game - splits.home.points_allowed_per_game
    away_net = splits.away.points_per_game - splits.away.points_allowed_per_game
    return home_net - away_net


class FeatureBuilder:
    def __init__(self, statistics: StatisticsClient, lines: LinesClient, reconciler: GameReconciler) -> None:
        self._statistics = statistics
        self._lines = lines
        self._reconciler = reconciler

    async def _team_features(self, team_id: str, prefix: str, game_date: date) -> FeatureMap:
        season, last5, last10, recent_games, injuries = await asyncio.gather(
            self._statistics.get_team_stats(team_id),
            self._statistics.get_team_stats(team_id, rolling_window=5),
            self._statistics.get_team_stats(team_id, rolling_window=10),
            self._statistics.list_recent_games(team_id, date_to=game_date.isoformat()),
            self._statistics.get_injuries(team_id),
        )

        players = {}
        for report in injuries[:MAX_PLAYERS_CONSIDERED]:
            try:
                players[report.player_id] = await self._statistics.get_player(report.player_id)
            except Exception:  # noqa: BLE001 - a missing player just weakens the proxy
                logger.debug("player %s lookup failed for injury impact", report.player_id)

        features: FeatureMap = {
            f"{prefix}_offensive_rating": season.stats.offensive.offensive_rating or None,
            f"{prefix}_defensive_rating": season.stats.defensive.defensive_rating or None,
            f"{prefix}_pace": season.stats.offensive.pace or None,
            f"{prefix}_net_rating": season.stats.advanced.net_rating or None,
            f"{prefix}_last5_ppg": last5.stats.offensive.points_per_game or None,
            f"{prefix}_last5_ppg_allowed": last5.stats.defensive.points_allowed_per_game or None,
            f"{prefix}_three_pct_last5": last5.stats.offensive.three_point_pct or None,
            f"{prefix}_net_rating_last10": last10.stats.advanced.net_rating or None,
            f"{prefix}_injury_impact": injury_impact(injuries, players),
        }
        if prefix == "home":
            features["home_away_split_diff"] = _split_diff(season)
        features.update(rest_features(game_date, recent_games, prefix))
        return features

    async def _market_block(self, game: Game) -> tuple[FeatureMap, str | None]:
        lines_id = await self._reconciler.resolve(game)
        empty: FeatureMap = {"line_movement": None, "n_books_reporting": None, "line_consensus_std": None}
        if lines_id is None:
            return empty, None
        try:
            movements = await self._lines.movement(lines_id, market_type="SPREAD")
            spread_lines = await self._lines.current_lines(league=game.league, game_id=lines_id, market_type="SPREAD")
        except Exception:  # noqa: BLE001 - market features are best-effort
            logger.warning("lines-service market features unavailable for %s", game.id, exc_info=True)
            return empty, lines_id
        return market_features(movements, spread_lines), lines_id

    async def build(self, game: Game) -> FeatureBundle:
        game_date = _game_date(game)
        home, away, (market_block, lines_id) = await asyncio.gather(
            self._team_features(game.home_team.id, "home", game_date),
            self._team_features(game.away_team.id, "away", game_date),
            self._market_block(game),
        )

        features: FeatureMap = {**home, **away, **market_block}

        def diff(a: str, b: str) -> float | None:
            left, right = features.get(a), features.get(b)
            return left - right if left is not None and right is not None else None

        features["pace_differential"] = diff("home_pace", "away_pace")
        features["net_rating_diff"] = diff("home_net_rating", "away_net_rating")
        features["rest_advantage"] = diff("home_rest_days", "away_rest_days")
        features["injury_impact_diff"] = diff("home_injury_impact", "away_injury_impact")

        stats_ts = await self._statistics.get_meta_timestamp("/api/v1/stats/health")
        sources: dict[str, str | None] = {
            "stats": stats_ts,
            "lines_game_external_id": lines_id,
        }
        return FeatureBundle(features=features, sources=sources, lines_game_external_id=lines_id)
