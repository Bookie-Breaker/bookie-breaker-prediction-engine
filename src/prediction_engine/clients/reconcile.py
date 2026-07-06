"""Reconcile statistics-service game UUIDs with lines-service external ids.

The two services have disjoint id spaces: statistics-service uses
deterministic UUIDv5 ids derived from NBA game ids, while lines-service
stores the Odds API event id. No shared key exists in Phase 1/2, but
lines-service selections carry full team names identical to
statistics-service team names, so games are matched by (league, date, home
team name). Mappings are cached in Redis for 24h.

Soccer hardening (Phase 6 Wave 1): names are Unicode-folded with diacritics
stripped, and a per-league alias table maps the FIFA/ESPN naming style used
by statistics-service onto The Odds API's style (both sides of every alias
pair pass through _normalize, and lookups apply to both services' names).

A docs note recommends statistics-service eventually populate
external_ids.odds_api so this name-matching becomes unnecessary.
"""

import logging
import unicodedata
from datetime import datetime

import redis.asyncio as aioredis

from prediction_engine.clients.lines import LinesClient
from prediction_engine.clients.statistics import Game

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.casefold().split())


# League -> {variant name -> canonical name}, seeded from FIFA World Cup
# discrepancies between ESPN/FIFA naming (statistics-service side) and The
# Odds API (lines-service side). Values are the canonical form both sides
# collapse to; keys and values are normalized at import time so lookups can
# assume _normalize has already been applied.
_RAW_TEAM_ALIASES: dict[str, dict[str, str]] = {
    "FIFA_WC": {
        "côte d'ivoire": "ivory coast",
        "korea republic": "south korea",
        "ir iran": "iran",
        "usa": "united states",
        "united states of america": "united states",
        "china pr": "china",
        "dr congo": "congo dr",
    },
}

TEAM_ALIASES: dict[str, dict[str, str]] = {
    league: {_normalize(variant): _normalize(canonical) for variant, canonical in aliases.items()}
    for league, aliases in _RAW_TEAM_ALIASES.items()
}


def _canonical(name: str, league: str) -> str:
    normalized = _normalize(name)
    return TEAM_ALIASES.get(league, {}).get(normalized, normalized)


def _game_date(scheduled_start: str) -> str | None:
    try:
        return datetime.fromisoformat(scheduled_start.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


class GameReconciler:
    def __init__(self, lines: LinesClient, redis_client: "aioredis.Redis", ttl_seconds: int = 86_400) -> None:
        self._lines = lines
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def resolve(self, game: Game) -> str | None:
        """Return the lines-service external game id, or None if unmatched."""
        cache_key = f"pred:gamemap:{game.id}"
        cached = await self._redis.get(cache_key)
        if cached:
            return str(cached)

        date = _game_date(game.scheduled_start)
        try:
            snapshots = await self._lines.current_lines(
                league=game.league, market_type="MONEYLINE", date=date, limit=200
            )
        except Exception:  # noqa: BLE001 - reconciliation is best-effort
            logger.warning("lines-service unavailable during game reconciliation for %s", game.id, exc_info=True)
            return None

        home_name = _canonical(game.home_team.name, game.league)
        away_name = _canonical(game.away_team.name, game.league)
        seen_selections: list[str] = []
        for snapshot in snapshots:
            selection = _canonical(snapshot.selection, game.league)
            seen_selections.append(selection)
            if snapshot.side == "HOME" and selection.startswith(home_name):
                await self._redis.set(cache_key, snapshot.game_id, ex=self._ttl)
                return snapshot.game_id
            if snapshot.side == "AWAY" and selection.startswith(away_name):
                await self._redis.set(cache_key, snapshot.game_id, ex=self._ttl)
                return snapshot.game_id

        # INFO with both sides' unmatched names so live-window alias fixes
        # can be made from the log line alone.
        logger.info(
            "no lines-service game matched stats game %s: stats names [%s @ %s], "
            "unmatched lines selections %s (league=%s date=%s)",
            game.id,
            away_name,
            home_name,
            sorted(set(seen_selections)),
            game.league,
            date,
        )
        return None
