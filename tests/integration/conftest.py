"""Integration fixtures: session-scoped Postgres + Redis containers.

The conftest replicates infra-ops init-db (schemas + public enums) before
running the Alembic migration, which is itself under test. Upstream
services are respx-mocked. Kept light for pre-push on modest hardware.
"""

import asyncio
import os
from collections.abc import Iterator
from typing import Any

import asyncpg
import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from prediction_engine.config import Settings
from prediction_engine.main import create_app

STATS_URL = "http://stats.test"
LINES_URL = "http://lines.test"
SIM_URL = "http://sim.test"

INIT_SQL = """
CREATE SCHEMA IF NOT EXISTS predictions;
CREATE TYPE league_enum AS ENUM ('NFL', 'NBA', 'MLB', 'NCAA_FB', 'NCAA_BB', 'NCAA_BSB');
CREATE TYPE market_type_enum AS ENUM
    ('SPREAD', 'TOTAL', 'MONEYLINE', 'PLAYER_PROP', 'TEAM_PROP', 'GAME_PROP', 'FUTURE', 'LIVE');
CREATE TYPE sport_enum AS ENUM ('FOOTBALL', 'BASKETBALL', 'BASEBALL');
"""


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        admin_url = f"postgresql://test:test@{host}:{port}/test"

        async def init_db() -> None:
            conn = await asyncpg.connect(admin_url)
            try:
                for statement in INIT_SQL.strip().split(";"):
                    if statement.strip():
                        await conn.execute(statement)
            finally:
                await conn.close()

        asyncio.run(init_db())
        yield f"postgres://test:test@{host}:{port}/test?search_path=predictions,public"


@pytest.fixture(scope="session")
def migrated_database_url(database_url: str) -> str:
    from alembic.config import Config

    from alembic import command

    os.environ["DATABASE_URL"] = database_url
    config = Config("alembic.ini")
    command.upgrade(config, "head")
    return database_url


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    with RedisContainer("redis:7-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"


@pytest.fixture(scope="session")
def client(
    migrated_database_url: str, redis_url: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[TestClient]:
    settings = Settings(
        database_url=migrated_database_url,
        redis_url=redis_url,
        statistics_service_url=STATS_URL,
        lines_service_url=LINES_URL,
        simulation_service_url=SIM_URL,
        model_dir=tmp_path_factory.mktemp("models"),
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def upstream() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def enveloped(data: Any) -> dict[str, Any]:
    return {"data": data, "meta": {"timestamp": "2026-07-04T12:00:00Z", "request_id": "req-test"}}


def game_payload(game_id: str) -> dict[str, Any]:
    return {
        "id": game_id,
        "league": "NBA",
        "status": "SCHEDULED",
        "home_team": {"id": "team-home", "name": "Los Angeles Lakers", "abbreviation": "LAL"},
        "away_team": {"id": "team-away", "name": "Boston Celtics", "abbreviation": "BOS"},
        "scheduled_start": "2026-07-05T00:00:00Z",
        "season": 2026,
    }


def team_stats_payload(team_id: str) -> dict[str, Any]:
    return {
        "team_id": team_id,
        "team_abbreviation": "LAL" if team_id == "team-home" else "BOS",
        "season": 2026,
        "games_played": 60,
        "wins": 35,
        "losses": 25,
        "stats": {
            "offensive": {
                "points_per_game": 114.0,
                "field_goal_pct": 0.48,
                "three_point_pct": 0.37,
                "free_throw_pct": 0.79,
                "offensive_rating": 115.0,
                "pace": 99.5,
                "effective_fg_pct": 0.55,
            },
            "defensive": {
                "points_allowed_per_game": 110.0,
                "opponent_fg_pct": 0.46,
                "opponent_three_point_pct": 0.35,
                "defensive_rating": 111.0,
            },
            "advanced": {"net_rating": 4.0, "turnover_pct": 12.5, "offensive_rebound_pct": 26.0},
        },
        "home_away_splits": {
            "home": {"wins": 20, "losses": 10, "points_per_game": 116.0, "points_allowed_per_game": 109.0},
            "away": {"wins": 15, "losses": 15, "points_per_game": 112.0, "points_allowed_per_game": 111.0},
        },
    }


def simulation_run_payload(run_id: str, game_id: str) -> dict[str, Any]:
    return {
        "simulation_run_id": run_id,
        "game_id": game_id,
        "status": "completed",
        "iterations_completed": 10000,
        "converged": True,
        "result": {
            "home_win_probability": 0.75,
            "away_win_probability": 0.25,
            "draw_probability": 0.0,
            "mean_home_score": 114.2,
            "mean_away_score": 108.1,
            "mean_total": 222.3,
            "mean_margin": 6.1,
            "spread_cover_probabilities": {"-4.5": 0.55, "-3.5": 0.58, "-2.5": 0.62},
            "total_over_probabilities": {"219.5": 0.58, "220.5": 0.55, "221.5": 0.52},
        },
    }


def mock_happy_path(router: respx.MockRouter, game_id: str, run_id: str) -> None:
    """Register all upstream mocks for a successful POST /predictions."""
    router.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
        return_value=Response(200, json=enveloped(game_payload(game_id)))
    )
    for team in ("team-home", "team-away"):
        router.get(f"{STATS_URL}/api/v1/stats/teams/{team}/stats").mock(
            return_value=Response(200, json=enveloped(team_stats_payload(team)))
        )
    router.get(f"{STATS_URL}/api/v1/stats/games").mock(
        return_value=Response(
            200,
            json=enveloped(
                [
                    {
                        "id": "prev-game",
                        "league": "NBA",
                        "status": "FINAL",
                        "home_team": {"id": "team-home"},
                        "away_team": {"id": "x"},
                        "scheduled_start": "2026-07-03T00:00:00Z",
                    }
                ]
            ),
        )
    )
    router.get(f"{STATS_URL}/api/v1/stats/injuries").mock(return_value=Response(200, json=enveloped([])))
    router.get(f"{STATS_URL}/api/v1/stats/health").mock(
        return_value=Response(200, json=enveloped({"status": "healthy"}))
    )

    router.get(f"{SIM_URL}/api/v1/sim/simulations/{run_id}").mock(
        return_value=Response(200, json=enveloped(simulation_run_payload(run_id, game_id)))
    )

    lines_ext_id = f"odds-{game_id}"
    router.get(f"{LINES_URL}/api/v1/lines/current").mock(
        return_value=Response(
            200,
            json=enveloped(
                [
                    {
                        "id": "snap-1",
                        "game_id": lines_ext_id,
                        "sportsbook_key": "draftkings",
                        "league": "NBA",
                        "market_type": "MONEYLINE",
                        "selection": "Los Angeles Lakers",
                        "side": "HOME",
                        "odds_american": -150,
                        "odds_decimal": 1.667,
                        "implied_probability": 0.600,
                        "timestamp": "2026-07-04T12:00:00Z",
                    },
                    {
                        "id": "snap-2",
                        "game_id": lines_ext_id,
                        "sportsbook_key": "draftkings",
                        "league": "NBA",
                        "market_type": "SPREAD",
                        "selection": "Los Angeles Lakers -3.5",
                        "side": "HOME",
                        "line_value": -3.5,
                        "odds_american": -110,
                        "odds_decimal": 1.909,
                        "implied_probability": 0.5238,
                        "timestamp": "2026-07-04T12:00:00Z",
                    },
                ]
            ),
        )
    )
    router.get(f"{LINES_URL}/api/v1/lines/game/{lines_ext_id}/movement").mock(
        return_value=Response(
            200,
            json=enveloped([{"game_id": lines_ext_id, "market_type": "SPREAD", "total_movement": 1.0}]),
        )
    )
    router.get(f"{LINES_URL}/api/v1/lines/game/{lines_ext_id}/best").mock(
        return_value=Response(
            200,
            json=enveloped(
                [
                    {
                        "market_type": "SPREAD",
                        "selection": "Los Angeles Lakers -3.5",
                        "side": "HOME",
                        "line_value": -3.5,
                        "best_odds_american": -110,
                        "best_odds_decimal": 1.909,
                        "implied_probability": 0.5238,
                        "sportsbook_key": "draftkings",
                    },
                    {
                        "market_type": "TOTAL",
                        "selection": "Over 220.5",
                        "side": "OVER",
                        "line_value": 220.5,
                        "best_odds_american": -108,
                        "best_odds_decimal": 1.926,
                        "implied_probability": 0.5192,
                        "sportsbook_key": "fanduel",
                    },
                    {
                        "market_type": "MONEYLINE",
                        "selection": "Los Angeles Lakers",
                        "side": "HOME",
                        "best_odds_american": -150,
                        "best_odds_decimal": 1.667,
                        "implied_probability": 0.600,
                        "sportsbook_key": "draftkings",
                    },
                ]
            ),
        )
    )
