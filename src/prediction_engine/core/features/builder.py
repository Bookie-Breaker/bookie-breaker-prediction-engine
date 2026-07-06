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
from prediction_engine.clients.statistics import Game, ProbablePitcher, StatisticsClient, TeamStats
from prediction_engine.core.features.injuries import MAX_PLAYERS_CONSIDERED, injury_impact
from prediction_engine.core.features.market import market_features
from prediction_engine.core.features.registry import FeatureMap
from prediction_engine.core.features.situational import football_rest_features, rest_features
from prediction_engine.core.leagues import LEAGUE_TO_SPORT

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


def _starter_features(pitcher: ProbablePitcher | None, prefix: str) -> FeatureMap:
    """Probable-starter features, null-safe for unannounced starters.

    Unannounced (absent from the game payload) means stats are None (NaN to
    XGBoost) with the announced flag at 0.0 -- the flag itself is signal,
    because unannounced games have noisier outcomes. An announced starter
    with missing season stats (a debut call-up) keeps flag 1.0 with None
    stats.
    """
    if pitcher is None:
        return {
            f"{prefix}_starter_fip": None,
            f"{prefix}_starter_era": None,
            f"{prefix}_starter_kbb": None,
            f"{prefix}_starter_announced": 0.0,
        }
    return {
        f"{prefix}_starter_fip": pitcher.fip,
        f"{prefix}_starter_era": pitcher.era,
        f"{prefix}_starter_kbb": pitcher.k_bb_pct,
        f"{prefix}_starter_announced": 1.0,
    }


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

    async def _team_features(self, team_id: str, prefix: str, game_date: date, league: str) -> FeatureMap:
        season, last5, last10, recent_games, injuries = await asyncio.gather(
            self._statistics.get_team_stats(team_id),
            self._statistics.get_team_stats(team_id, rolling_window=5),
            self._statistics.get_team_stats(team_id, rolling_window=10),
            self._statistics.list_recent_games(team_id, date_to=game_date.isoformat()),
            self._statistics.get_injuries(team_id, league),
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

    async def _soccer_team_features(self, team_id: str, prefix: str, game_date: date) -> FeatureMap:
        """Team features from the SoccerStats block (ADR-026).

        No injuries for soccer: the statistics-service has no soccer injury
        feed, so the registry excludes injury features entirely rather than
        emitting always-null columns. Back-to-backs never occur in soccer
        scheduling, so only rest_days is kept from the rest block.
        """
        season, recent_games = await asyncio.gather(
            self._statistics.get_team_stats(team_id),
            self._statistics.list_recent_games(team_id, date_to=game_date.isoformat()),
        )
        soccer = season.stats.soccer
        features: FeatureMap = {
            f"{prefix}_attack_strength": (soccer.attack_strength or None) if soccer else None,
            f"{prefix}_defense_strength": (soccer.defense_strength or None) if soccer else None,
            f"{prefix}_goals_for_per_match": soccer.goals_for_per_match if soccer else None,
            f"{prefix}_goals_against_per_match": soccer.goals_against_per_match if soccer else None,
            f"{prefix}_form_points_last5": float(soccer.form_points_last5) if soccer else None,
            f"{prefix}_matches_played": float(season.games_played) if season.games_played else None,
        }
        features[f"{prefix}_rest_days"] = rest_features(game_date, recent_games, prefix)[f"{prefix}_rest_days"]
        return features

    async def _baseball_team_features(self, team_id: str, prefix: str, game_date: date) -> FeatureMap:
        """Team features from the BaseballStats block (ADR-026).

        No injury features for baseball in Wave 2 (a deliberate
        null-documentation, not an oversight): get_injuries covers MLB
        since the Wave 0 league generalization, but the NBA injury-impact
        proxy is minutes-based and does not transfer to baseball, and the
        probable-starter block already carries the dominant personnel
        signal. A validated count-based IL feature can join in a later
        wave. Back-to-backs are meaningless in a near-daily sport, so only
        rest_days is kept from the rest block.
        """
        season, recent_games = await asyncio.gather(
            self._statistics.get_team_stats(team_id),
            self._statistics.list_recent_games(team_id, date_to=game_date.isoformat()),
        )
        baseball = season.stats.baseball
        features: FeatureMap = {
            f"{prefix}_runs_scored_per_game": (baseball.runs_scored_per_game or None) if baseball else None,
            f"{prefix}_runs_allowed_per_game": (baseball.runs_allowed_per_game or None) if baseball else None,
            f"{prefix}_team_woba": (baseball.team_woba or None) if baseball else None,
            f"{prefix}_team_fip": (baseball.team_fip or None) if baseball else None,
            f"{prefix}_bullpen_era": (baseball.bullpen_era or None) if baseball else None,
        }
        features[f"{prefix}_rest_days"] = rest_features(game_date, recent_games, prefix)[f"{prefix}_rest_days"]
        return features

    async def _football_team_features(self, team_id: str, prefix: str, game_date: date, league: str) -> FeatureMap:
        """Team features from the FootballStats block (ADR-026).

        EPA and SP+ are league-gated (null-documented, not an oversight):
        EPA comes from nflverse and exists for the NFL only, SP+ comes from
        CFBD and exists for NCAA_FB only. The contract leaves the other
        league's fields absent, which the client model defaults to 0.0 --
        gating on the league turns those placeholder zeros into None (NaN
        to XGBoost) so the model never mistakes "no source" for "exactly
        league average". No injury features (no validated football
        injury-impact proxy); back-to-backs cannot happen in a weekly
        sport, so the rest block carries rest_days plus the bye flag.
        """
        season, recent_games = await asyncio.gather(
            self._statistics.get_team_stats(team_id),
            self._statistics.list_recent_games(team_id, date_to=game_date.isoformat()),
        )
        football = season.stats.football
        is_nfl = league == "NFL"
        features: FeatureMap = {
            f"{prefix}_points_per_game": (football.points_per_game or None) if football else None,
            f"{prefix}_points_allowed_per_game": (football.points_allowed_per_game or None) if football else None,
            f"{prefix}_points_per_drive_off": (football.points_per_drive_off or None) if football else None,
            f"{prefix}_points_per_drive_def": (football.points_per_drive_def or None) if football else None,
            # EPA and turnover margin are signed (0.0 is meaningful for the
            # NFL), so they pass through unfiltered when their league owns them
            f"{prefix}_epa_per_play_off": football.epa_per_play_off if football and is_nfl else None,
            f"{prefix}_epa_per_play_def": football.epa_per_play_def if football and is_nfl else None,
            f"{prefix}_sp_plus_rating": football.sp_plus_rating if football and not is_nfl else None,
            f"{prefix}_turnover_margin_per_game": football.turnover_margin_per_game if football else None,
        }
        features.update(football_rest_features(game_date, recent_games, prefix))
        return features

    async def _hockey_team_features(self, team_id: str, prefix: str, game_date: date) -> FeatureMap:
        """Team features from the HockeyStats block (ADR-026).

        No injury features (no validated hockey injury-impact proxy) and no
        starting-goaltender block (no confirmed-starter feed; team_save_pct
        carries the aggregate goaltending signal). The NBA situational
        module transfers directly: the NHL schedule is dense enough that
        back-to-backs are common and meaningful.
        """
        season, recent_games = await asyncio.gather(
            self._statistics.get_team_stats(team_id),
            self._statistics.list_recent_games(team_id, date_to=game_date.isoformat()),
        )
        hockey = season.stats.hockey
        features: FeatureMap = {
            f"{prefix}_goals_for_per_game": (hockey.goals_for_per_game or None) if hockey else None,
            f"{prefix}_goals_against_per_game": (hockey.goals_against_per_game or None) if hockey else None,
            f"{prefix}_shots_for_per_game": (hockey.shots_for_per_game or None) if hockey else None,
            f"{prefix}_shots_against_per_game": (hockey.shots_against_per_game or None) if hockey else None,
            f"{prefix}_power_play_pct": (hockey.power_play_pct or None) if hockey else None,
            f"{prefix}_penalty_kill_pct": (hockey.penalty_kill_pct or None) if hockey else None,
            f"{prefix}_team_save_pct": (hockey.team_save_pct or None) if hockey else None,
        }
        features.update(rest_features(game_date, recent_games, prefix))
        return features

    async def _college_basketball_team_features(self, team_id: str, prefix: str, game_date: date) -> FeatureMap:
        """Team features for NCAA_BB: the NBA block minus injuries plus CBBD.

        No injury features (null-documented, not an oversight): there is no
        reliable college injury source, so the NBA's status-weighted proxy
        has nothing honest to weight -- the columns would always be None.
        adjusted_efficiency_margin is the CBBD opponent-adjusted rating on
        the AdvancedStats block, absent (0.0 default -> None) elsewhere.
        """
        season, last5, last10, recent_games = await asyncio.gather(
            self._statistics.get_team_stats(team_id),
            self._statistics.get_team_stats(team_id, rolling_window=5),
            self._statistics.get_team_stats(team_id, rolling_window=10),
            self._statistics.list_recent_games(team_id, date_to=game_date.isoformat()),
        )
        features: FeatureMap = {
            f"{prefix}_offensive_rating": season.stats.offensive.offensive_rating or None,
            f"{prefix}_defensive_rating": season.stats.defensive.defensive_rating or None,
            f"{prefix}_pace": season.stats.offensive.pace or None,
            f"{prefix}_net_rating": season.stats.advanced.net_rating or None,
            f"{prefix}_adjusted_efficiency_margin": season.stats.advanced.adjusted_efficiency_margin or None,
            f"{prefix}_last5_ppg": last5.stats.offensive.points_per_game or None,
            f"{prefix}_last5_ppg_allowed": last5.stats.defensive.points_allowed_per_game or None,
            f"{prefix}_three_pct_last5": last5.stats.offensive.three_point_pct or None,
            f"{prefix}_net_rating_last10": last10.stats.advanced.net_rating or None,
        }
        if prefix == "home":
            features["home_away_split_diff"] = _split_diff(season)
        features.update(rest_features(game_date, recent_games, prefix))
        return features

    async def _build_football(self, game: Game, game_date: date) -> tuple[FeatureMap, str | None]:
        home, away, (market_block, lines_id) = await asyncio.gather(
            self._football_team_features(game.home_team.id, "home", game_date, game.league),
            self._football_team_features(game.away_team.id, "away", game_date, game.league),
            self._market_block(game),
        )
        features: FeatureMap = {**home, **away, **market_block}
        features["league_is_nfl"] = 1.0 if game.league == "NFL" else 0.0
        return features, lines_id

    async def _build_hockey(self, game: Game, game_date: date) -> tuple[FeatureMap, str | None]:
        home, away, (market_block, lines_id) = await asyncio.gather(
            self._hockey_team_features(game.home_team.id, "home", game_date),
            self._hockey_team_features(game.away_team.id, "away", game_date),
            self._market_block(game),
        )
        features: FeatureMap = {**home, **away, **market_block}
        return features, lines_id

    async def _build_college_basketball(self, game: Game, game_date: date) -> tuple[FeatureMap, str | None]:
        home, away, (market_block, lines_id) = await asyncio.gather(
            self._college_basketball_team_features(game.home_team.id, "home", game_date),
            self._college_basketball_team_features(game.away_team.id, "away", game_date),
            self._market_block(game),
        )

        features: FeatureMap = {**home, **away, **market_block}

        def diff(a: str, b: str) -> float | None:
            left, right = features.get(a), features.get(b)
            return left - right if left is not None and right is not None else None

        features["pace_differential"] = diff("home_pace", "away_pace")
        features["net_rating_diff"] = diff("home_net_rating", "away_net_rating")
        features["rest_advantage"] = diff("home_rest_days", "away_rest_days")
        return features, lines_id

    async def _build_baseball(self, game: Game, game_date: date) -> tuple[FeatureMap, str | None]:
        home, away, (market_block, lines_id) = await asyncio.gather(
            self._baseball_team_features(game.home_team.id, "home", game_date),
            self._baseball_team_features(game.away_team.id, "away", game_date),
            self._market_block(game),
        )
        features: FeatureMap = {**home, **away, **market_block}
        features.update(_starter_features(game.home_probable_pitcher, "home"))
        features.update(_starter_features(game.away_probable_pitcher, "away"))
        home_fip, away_fip = features["home_starter_fip"], features["away_starter_fip"]
        features["starter_fip_diff"] = home_fip - away_fip if home_fip is not None and away_fip is not None else None
        features["league_is_mlb"] = 1.0 if game.league == "MLB" else 0.0
        return features, lines_id

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

    async def _build_soccer(self, game: Game, game_date: date) -> tuple[FeatureMap, str | None]:
        home, away, (market_block, lines_id) = await asyncio.gather(
            self._soccer_team_features(game.home_team.id, "home", game_date),
            self._soccer_team_features(game.away_team.id, "away", game_date),
            self._market_block(game),
        )
        features: FeatureMap = {**home, **away, **market_block}
        features["is_knockout"] = 1.0 if game.season_type == "POSTSEASON" else 0.0
        features["competition_is_fifa_wc"] = 1.0 if game.league == "FIFA_WC" else 0.0
        return features, lines_id

    async def _build_basketball(self, game: Game, game_date: date) -> tuple[FeatureMap, str | None]:
        home, away, (market_block, lines_id) = await asyncio.gather(
            self._team_features(game.home_team.id, "home", game_date, game.league),
            self._team_features(game.away_team.id, "away", game_date, game.league),
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
        return features, lines_id

    async def build(self, game: Game) -> FeatureBundle:
        game_date = _game_date(game)
        sport = LEAGUE_TO_SPORT.get(game.league)
        if sport == "SOCCER":
            features, lines_id = await self._build_soccer(game, game_date)
        elif sport == "BASEBALL":
            features, lines_id = await self._build_baseball(game, game_date)
        elif sport == "FOOTBALL":
            features, lines_id = await self._build_football(game, game_date)
        elif sport == "HOCKEY":
            features, lines_id = await self._build_hockey(game, game_date)
        elif game.league == "NCAA_BB":
            # BASKETBALL sport, but its own single-league model (see leagues.py)
            features, lines_id = await self._build_college_basketball(game, game_date)
        else:
            features, lines_id = await self._build_basketball(game, game_date)

        stats_ts = await self._statistics.get_meta_timestamp("/api/v1/stats/health")
        sources: dict[str, str | None] = {
            "stats": stats_ts,
            "lines_game_external_id": lines_id,
        }
        return FeatureBundle(features=features, sources=sources, lines_game_external_id=lines_id)
