"""Per-sport registry tests: features, synthetic generators, league mapping, models."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from prediction_engine.core.features.registry import NBA_FEATURES, SOCCER_FEATURES, get_features
from prediction_engine.core.leagues import LEAGUE_TO_SPORT, THREE_WAY_MONEYLINE_SPORTS, sport_for_league
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.training.synthetic import (
    generate_soccer_synthetic_dataset,
    generate_synthetic_dataset,
    get_synthetic_generator,
)
from prediction_engine.core.training.train import save_artifact, train_model
from prediction_engine.db.repository import ModelVersionRecord


class TestFeatureRegistry:
    def test_basketball_features_are_the_locked_nba_tuple(self) -> None:
        assert get_features("BASKETBALL") is NBA_FEATURES

    def test_soccer_features_registered(self) -> None:
        assert get_features("SOCCER") is SOCCER_FEATURES
        assert len(SOCCER_FEATURES) == len(set(SOCCER_FEATURES))
        # ADR-027 three-way essentials and the ADR-026 pooled-competition one-hot
        for required in ("sim_draw_probability", "selection_is_draw", "is_knockout", "competition_is_fifa_wc"):
            assert required in SOCCER_FEATURES
        # no injury features for soccer (no data source; see registry comment)
        assert not any("injury" in name for name in SOCCER_FEATURES)

    def test_unregistered_sport_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="no feature registry for HOCKEY; added in its league wave"):
            get_features("HOCKEY")


class TestSyntheticRegistry:
    def test_basketball_generator_registered(self) -> None:
        assert get_synthetic_generator("BASKETBALL") is generate_synthetic_dataset

    def test_soccer_generator_registered(self) -> None:
        assert get_synthetic_generator("SOCCER") is generate_soccer_synthetic_dataset

    def test_unregistered_sport_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="no synthetic generator registered for HOCKEY; added in its league wave"):
            get_synthetic_generator("HOCKEY")


class TestLeagueMapping:
    def test_all_ten_leagues_map_to_five_sports(self) -> None:
        assert len(LEAGUE_TO_SPORT) == 10
        assert set(LEAGUE_TO_SPORT.values()) == {"FOOTBALL", "BASKETBALL", "BASEBALL", "SOCCER", "HOCKEY"}
        assert sport_for_league("NBA") == "BASKETBALL"
        assert sport_for_league("FIFA_WC") == "SOCCER"
        assert sport_for_league("EPL") == "SOCCER"
        assert sport_for_league("NHL") == "HOCKEY"

    def test_unknown_league_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown league 'XFL'"):
            sport_for_league("XFL")

    def test_only_soccer_is_three_way(self) -> None:
        assert {"SOCCER"} == THREE_WAY_MONEYLINE_SPORTS


class FakeModelVersionRepo:
    """In-memory stand-in for ModelVersionRepository."""

    def __init__(self) -> None:
        self.records: list[ModelVersionRecord] = []

    async def list_models(
        self, sport: str | None = None, market_type: str | None = None, is_active: bool | None = None
    ) -> list[ModelVersionRecord]:
        return [
            record
            for record in self.records
            if (sport is None or record.sport == sport)
            and (market_type is None or record.model_type == market_type)
            and (is_active is None or record.is_active == is_active)
        ]

    async def register(self, sport: str, market_types: list[str], **kwargs: Any) -> list[ModelVersionRecord]:
        created = [
            ModelVersionRecord(
                id=uuid.uuid4(),
                sport=sport,
                model_type=market,
                version=kwargs["version"],
                algorithm="xgboost",
                trained_at=kwargs.get("trained_at", datetime.now(tz=UTC)),
                training_samples=kwargs.get("training_samples", 0),
                evaluation_metrics=kwargs.get("evaluation_metrics", {}),
                feature_names=kwargs.get("feature_names", []),
                is_active=True,
                artifact_path=kwargs["artifact_path"],
                notes=kwargs.get("notes"),
            )
            for market in market_types
        ]
        self.records.extend(created)
        return created


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory):
    """A MODEL_DIR pre-seeded with tiny basketball + soccer artifacts (fast bootstrap)."""
    directory = tmp_path_factory.mktemp("models")
    result = train_model(generate_synthetic_dataset(n_rows=800), n_rounds=8, data_label="synthetic")
    save_artifact(result, directory)
    soccer = train_model(
        generate_soccer_synthetic_dataset(n_games=160), n_rounds=8, data_label="synthetic", sport="SOCCER"
    )
    save_artifact(soccer, directory)
    return directory


class TestModelRegistry:
    async def test_lookup_is_keyed_by_sport_and_market(self, model_dir) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]
        await registry.ensure_bootstrap("BASKETBALL")

        for market in ("SPREAD", "TOTAL", "MONEYLINE"):
            loaded = await registry.get_active("BASKETBALL", market)
            assert loaded is not None
            assert loaded.record.sport == "BASKETBALL"
            assert loaded.record.model_type == market
        assert await registry.get_active("BASKETBALL", "PLAYER_PROP") is None
        assert set(registry.active_map()) == {"BASKETBALL_SPREAD", "BASKETBALL_TOTAL", "BASKETBALL_MONEYLINE"}

    async def test_soccer_bootstrap_registers_all_markets(self, model_dir) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]
        await registry.ensure_bootstrap("SOCCER")

        for market in ("SPREAD", "TOTAL", "MONEYLINE"):
            loaded = await registry.get_active("SOCCER", market)
            assert loaded is not None
            assert loaded.record.sport == "SOCCER"
            assert loaded.record.feature_names == list(SOCCER_FEATURES)

    async def test_bootstrap_unregistered_sport_raises(self, model_dir) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="no synthetic generator registered for HOCKEY"):
            await registry.ensure_bootstrap("HOCKEY")

    async def test_get_active_lazily_bootstraps_and_propagates_registry_errors(self, model_dir) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]
        # lazy bootstrap on first use (no explicit ensure_bootstrap call)
        assert await registry.get_active("BASKETBALL", "SPREAD") is not None
        with pytest.raises(ValueError, match="no synthetic generator registered for HOCKEY"):
            await registry.get_active("HOCKEY", "MONEYLINE")

    async def test_startup_bootstrap_reports_failure_without_crashing(self, model_dir) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), model_dir, sports=["BASKETBALL", "HOCKEY"])  # type: ignore[arg-type]
        assert await registry.try_bootstrap() is False
        assert await registry.get_active("BASKETBALL", "TOTAL") is not None

    async def test_startup_bootstrap_succeeds_when_soccer_configured(self, model_dir) -> None:
        # PREDICTION_SPORTS=BASKETBALL,SOCCER startup path (default stays BASKETBALL)
        registry = ModelRegistry(FakeModelVersionRepo(), model_dir, sports=["BASKETBALL", "SOCCER"])  # type: ignore[arg-type]
        assert await registry.try_bootstrap() is True
        assert await registry.get_active("SOCCER", "MONEYLINE") is not None
