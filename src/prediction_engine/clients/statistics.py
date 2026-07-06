"""Typed async client for the statistics-service REST API (port 8002)."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from prediction_engine.clients.base import ServiceClient


class TeamRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str = ""
    abbreviation: str = ""


class Game(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    league: str
    status: str
    home_team: TeamRef
    away_team: TeamRef
    scheduled_start: str = ""
    season: int = 0


class OffensiveStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    points_per_game: float = 0.0
    field_goal_pct: float = 0.0
    three_point_pct: float = 0.0
    free_throw_pct: float = 0.0
    offensive_rating: float = 0.0
    pace: float = 0.0
    effective_fg_pct: float = 0.0


class DefensiveStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    points_allowed_per_game: float = 0.0
    opponent_fg_pct: float = 0.0
    opponent_three_point_pct: float = 0.0
    defensive_rating: float = 0.0


class AdvancedStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    net_rating: float = 0.0
    true_shooting_pct: float = 0.0
    turnover_pct: float = 0.0
    offensive_rebound_pct: float = 0.0


class StatBlocks(BaseModel):
    model_config = ConfigDict(extra="ignore")

    offensive: OffensiveStats = OffensiveStats()
    defensive: DefensiveStats = DefensiveStats()
    advanced: AdvancedStats = AdvancedStats()


class HomeAwaySplit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    wins: int = 0
    losses: int = 0
    points_per_game: float = 0.0
    points_allowed_per_game: float = 0.0


class HomeAwaySplits(BaseModel):
    model_config = ConfigDict(extra="ignore")

    home: HomeAwaySplit = HomeAwaySplit()
    away: HomeAwaySplit = HomeAwaySplit()


class TeamStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    team_id: str
    team_abbreviation: str = ""
    season: int = 0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    stats: StatBlocks = StatBlocks()
    home_away_splits: HomeAwaySplits | None = None


class InjuryReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    player_id: str
    player_name: str = ""
    team_id: str = ""
    status: str = ""


class PlayerSeasonStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    points_per_game: float = 0.0
    minutes_per_game: float = 0.0


class PlayerDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str = ""
    season_stats: PlayerSeasonStats | None = None


class StatisticsClient(ServiceClient):
    service_name = "statistics-service"

    async def get_game(self, game_id: str) -> Game:
        data = await self.get_data(f"/api/v1/stats/games/{game_id}", f"game {game_id}")
        return Game.model_validate(data)

    async def get_team_stats(self, team_id: str, rolling_window: int | None = None) -> TeamStats:
        params: dict[str, Any] = {"stat_type": "all"}
        if rolling_window is not None:
            params["rolling_window"] = rolling_window
        data = await self.get_data(f"/api/v1/stats/teams/{team_id}/stats", f"team stats {team_id}", params)
        return TeamStats.model_validate(data)

    async def list_recent_games(self, team_id: str, date_to: str, limit: int = 10) -> list[Game]:
        data = await self.get_data(
            "/api/v1/stats/games",
            f"recent games for team {team_id}",
            {"team": team_id, "status": "FINAL", "date_to": date_to, "limit": limit},
        )
        return [Game.model_validate(item) for item in data]

    async def get_injuries(self, team_id: str, league: str) -> list[InjuryReport]:
        data = await self.get_data(
            "/api/v1/stats/injuries", f"injuries for team {team_id}", {"league": league, "team_id": team_id}
        )
        return [InjuryReport.model_validate(item) for item in data]

    async def get_player(self, player_id: str) -> PlayerDetail:
        data = await self.get_data(f"/api/v1/stats/players/{player_id}", f"player {player_id}")
        return PlayerDetail.model_validate(data)

    async def health(self) -> bool:
        return await self.is_healthy("/api/v1/stats/health")
