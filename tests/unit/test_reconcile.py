"""Game reconciliation matching tests with in-memory fakes."""

import logging
from typing import Any

import pytest

from prediction_engine.clients.lines import LineSnapshot
from prediction_engine.clients.reconcile import GameReconciler
from prediction_engine.clients.statistics import Game, TeamRef


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value


class FakeLines:
    def __init__(self, snapshots: list[LineSnapshot]) -> None:
        self._snapshots = snapshots
        self.calls: list[dict[str, Any]] = []

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        self.calls.append(kwargs)
        return self._snapshots


def make_game() -> Game:
    return Game(
        id="stats-uuid-1",
        league="NBA",
        status="SCHEDULED",
        home_team=TeamRef(id="h", name="Los Angeles Lakers"),
        away_team=TeamRef(id="a", name="Boston Celtics"),
        scheduled_start="2026-01-15T19:30:00Z",
    )


def snapshot(game_id: str, side: str, selection: str) -> LineSnapshot:
    return LineSnapshot(id="l1", game_id=game_id, side=side, selection=selection, market_type="MONEYLINE")


class TestGameReconciler:
    async def test_matches_home_team_name(self) -> None:
        lines = FakeLines([snapshot("odds-42", "HOME", "Los Angeles Lakers")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_game()) == "odds-42"

    async def test_matches_away_side_too(self) -> None:
        lines = FakeLines([snapshot("odds-42", "AWAY", "Boston Celtics")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_game()) == "odds-42"

    async def test_no_match_returns_none(self) -> None:
        lines = FakeLines([snapshot("odds-99", "HOME", "Denver Nuggets")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_game()) is None

    async def test_result_is_cached(self) -> None:
        redis = FakeRedis()
        lines = FakeLines([snapshot("odds-42", "HOME", "Los Angeles Lakers")])
        reconciler = GameReconciler(lines, redis)  # type: ignore[arg-type]
        await reconciler.resolve(make_game())
        await reconciler.resolve(make_game())
        assert len(lines.calls) == 1
        assert redis.store["pred:gamemap:stats-uuid-1"] == "odds-42"

    async def test_case_insensitive(self) -> None:
        lines = FakeLines([snapshot("odds-42", "HOME", "LOS ANGELES LAKERS -3.5")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_game()) == "odds-42"

    async def test_lines_failure_returns_none(self) -> None:
        class FailingLines:
            async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
                raise RuntimeError("boom")

        reconciler = GameReconciler(FailingLines(), FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_game()) is None


def make_soccer_game(home: str, away: str, league: str = "FIFA_WC") -> Game:
    return Game(
        id="stats-uuid-wc",
        league=league,
        status="SCHEDULED",
        home_team=TeamRef(id="h", name=home),
        away_team=TeamRef(id="a", name=away),
        scheduled_start="2026-07-10T20:00:00Z",
    )


class TestSoccerReconciliationHardening:
    async def test_diacritics_fold_to_ascii(self) -> None:
        # stats-side "Côte d'Ivoire" vs lines-side ASCII (alias table maps
        # both spellings onto "ivory coast")
        lines = FakeLines([snapshot("odds-7", "HOME", "Ivory Coast")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_soccer_game("Côte d'Ivoire", "Brazil")) == "odds-7"

    async def test_fifa_alias_table_applies_to_stats_side(self) -> None:
        lines = FakeLines([snapshot("odds-8", "HOME", "South Korea")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_soccer_game("Korea Republic", "Ghana")) == "odds-8"

    async def test_fifa_alias_table_applies_to_lines_side(self) -> None:
        # both sides collapse to the canonical "united states"
        lines = FakeLines([snapshot("odds-9", "AWAY", "USA")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_soccer_game("Mexico", "United States of America")) == "odds-9"

    async def test_iran_and_china_aliases(self) -> None:
        lines = FakeLines([snapshot("odds-10", "HOME", "Iran"), snapshot("odds-11", "AWAY", "China")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_soccer_game("IR Iran", "China PR")) == "odds-10"

    async def test_aliases_are_league_scoped(self) -> None:
        # "usa" only aliases inside FIFA_WC; an EPL game must not rewrite it
        lines = FakeLines([snapshot("odds-12", "HOME", "United States")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        assert await reconciler.resolve(make_soccer_game("USA", "Arsenal", league="EPL")) is None

    async def test_no_match_logs_both_name_lists_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        lines = FakeLines([snapshot("odds-13", "HOME", "Senegal"), snapshot("odds-13", "AWAY", "Ecuador")])
        reconciler = GameReconciler(lines, FakeRedis())  # type: ignore[arg-type]
        with caplog.at_level(logging.INFO, logger="prediction_engine.clients.reconcile"):
            assert await reconciler.resolve(make_soccer_game("Netherlands", "Qatar")) is None
        [record] = [r for r in caplog.records if "no lines-service game matched" in r.message]
        assert record.levelno == logging.INFO
        message = record.getMessage()
        # both the stats-side names and the unmatched lines-side selections
        assert "netherlands" in message and "qatar" in message
        assert "senegal" in message and "ecuador" in message
