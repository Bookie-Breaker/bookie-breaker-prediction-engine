"""Per-sport registry tests: features, synthetic generators, league mapping, models."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from prediction_engine.core.features.registry import (
    BASEBALL_FEATURES,
    FOOTBALL_FEATURES,
    HOCKEY_FEATURES,
    NBA_FEATURES,
    NCAA_BB_FEATURES,
    SOCCER_FEATURES,
    get_features,
)
from prediction_engine.core.leagues import (
    LEAGUE_TO_SPORT,
    THREE_WAY_MONEYLINE_SPORTS,
    model_key_for_league,
    sport_for_league,
)
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.core.training.synthetic import (
    generate_baseball_synthetic_dataset,
    generate_football_synthetic_dataset,
    generate_hockey_synthetic_dataset,
    generate_ncaa_bb_synthetic_dataset,
    generate_soccer_synthetic_dataset,
    generate_synthetic_dataset,
    get_synthetic_generator,
)
from prediction_engine.core.training.train import save_artifact, train_model
from prediction_engine.db.repository import ModelVersionRecord

# All five sports (and NCAA_BB) are registered as of Phase 6 Wave 3-5, so the
# "not yet registered" sentinel is a sport with no wave: CRICKET.
UNREGISTERED = "CRICKET"


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

    def test_baseball_features_registered(self) -> None:
        assert get_features("BASEBALL") is BASEBALL_FEATURES
        assert len(BASEBALL_FEATURES) == len(set(BASEBALL_FEATURES))
        # the probable-starter block, announced flags, and the ADR-026
        # pooled-league one-hot are the wave's mandatory additions
        for required in (
            "home_starter_fip",
            "away_starter_fip",
            "home_starter_era",
            "away_starter_era",
            "home_starter_kbb",
            "away_starter_kbb",
            "starter_fip_diff",
            "home_starter_announced",
            "away_starter_announced",
            "league_is_mlb",
        ):
            assert required in BASEBALL_FEATURES
        # baseball cannot tie: no draw features (two-way moneyline, ADR-027)
        assert not any("draw" in name for name in BASEBALL_FEATURES)
        # no injury features for baseball (null-documented; see registry comment)
        assert not any("injury" in name for name in BASEBALL_FEATURES)

    def test_football_features_registered(self) -> None:
        assert get_features("FOOTBALL") is FOOTBALL_FEATURES
        assert len(FOOTBALL_FEATURES) == len(set(FOOTBALL_FEATURES))
        # EPA + SP+ per side, the league one-hot, bye flags, and the tie-risk
        # sim signal are the wave's mandatory additions; but NO selection_is_draw
        # (moneylines stay two-way, a tie is a PUSH -- ADR-027).
        for required in (
            "home_epa_per_play_off",
            "away_epa_per_play_def",
            "home_sp_plus_rating",
            "away_sp_plus_rating",
            "league_is_nfl",
            "home_bye_week",
            "away_bye_week",
            "sim_draw_probability",
        ):
            assert required in FOOTBALL_FEATURES
        assert "selection_is_draw" not in FOOTBALL_FEATURES
        # weekly sport: no back-to-backs, and no injury proxy
        assert not any("back_to_back" in name for name in FOOTBALL_FEATURES)
        assert not any("injury" in name for name in FOOTBALL_FEATURES)

    def test_hockey_features_registered(self) -> None:
        assert get_features("HOCKEY") is HOCKEY_FEATURES
        assert len(HOCKEY_FEATURES) == len(set(HOCKEY_FEATURES))
        for required in (
            "home_goals_for_per_game",
            "away_goals_against_per_game",
            "home_power_play_pct",
            "away_penalty_kill_pct",
            "home_team_save_pct",
            "home_shots_for_per_game",
            "home_back_to_back",
            "away_back_to_back",
        ):
            assert required in HOCKEY_FEATURES
        # NHL finals resolve in OT/SO: two-way moneyline, no draw features
        assert not any("draw" in name for name in HOCKEY_FEATURES)
        # single-league NHL model: no league one-hot until NCAA_HKY lands
        assert "league_is_nhl" not in HOCKEY_FEATURES
        assert not any("injury" in name for name in HOCKEY_FEATURES)

    def test_ncaa_bb_features_registered(self) -> None:
        assert get_features("NCAA_BB") is NCAA_BB_FEATURES
        assert len(NCAA_BB_FEATURES) == len(set(NCAA_BB_FEATURES))
        # the CBBD adjusted-efficiency margin per side is the wave's addition
        assert "home_adjusted_efficiency_margin" in NCAA_BB_FEATURES
        assert "away_adjusted_efficiency_margin" in NCAA_BB_FEATURES
        # no injuries (no reliable college source) and no league one-hot
        # (single-league model, exactly like the NBA)
        assert not any("injury" in name for name in NCAA_BB_FEATURES)
        assert not any("league_is" in name for name in NCAA_BB_FEATURES)

    def test_unregistered_sport_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match=f"no feature registry for {UNREGISTERED}; added in its league wave"):
            get_features(UNREGISTERED)


class TestSyntheticRegistry:
    def test_basketball_generator_registered(self) -> None:
        assert get_synthetic_generator("BASKETBALL") is generate_synthetic_dataset

    def test_soccer_generator_registered(self) -> None:
        assert get_synthetic_generator("SOCCER") is generate_soccer_synthetic_dataset

    def test_baseball_generator_registered(self) -> None:
        assert get_synthetic_generator("BASEBALL") is generate_baseball_synthetic_dataset

    def test_football_generator_registered(self) -> None:
        assert get_synthetic_generator("FOOTBALL") is generate_football_synthetic_dataset

    def test_hockey_generator_registered(self) -> None:
        assert get_synthetic_generator("HOCKEY") is generate_hockey_synthetic_dataset

    def test_ncaa_bb_generator_registered(self) -> None:
        assert get_synthetic_generator("NCAA_BB") is generate_ncaa_bb_synthetic_dataset

    def test_unregistered_sport_fails_loudly(self) -> None:
        with pytest.raises(
            ValueError, match=f"no synthetic generator registered for {UNREGISTERED}; added in its league wave"
        ):
            get_synthetic_generator(UNREGISTERED)


class TestLeagueMapping:
    def test_all_ten_leagues_map_to_five_sports(self) -> None:
        assert len(LEAGUE_TO_SPORT) == 10
        assert set(LEAGUE_TO_SPORT.values()) == {"FOOTBALL", "BASKETBALL", "BASEBALL", "SOCCER", "HOCKEY"}
        assert sport_for_league("NBA") == "BASKETBALL"
        assert sport_for_league("FIFA_WC") == "SOCCER"
        assert sport_for_league("EPL") == "SOCCER"
        assert sport_for_league("NHL") == "HOCKEY"
        assert sport_for_league("NFL") == "FOOTBALL"
        assert sport_for_league("NCAA_FB") == "FOOTBALL"

    def test_model_key_pools_by_sport_except_ncaa_bb(self) -> None:
        # pooled sports key their model by sport (a league one-hot separates them)
        assert model_key_for_league("NFL") == "FOOTBALL"
        assert model_key_for_league("NCAA_FB") == "FOOTBALL"
        assert model_key_for_league("NHL") == "HOCKEY"
        assert model_key_for_league("NBA") == "BASKETBALL"
        # NCAA_BB is the exception: its own single-league model, keyed by league
        assert model_key_for_league("NCAA_BB") == "NCAA_BB"

    def test_unknown_league_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown league 'XFL'"):
            sport_for_league("XFL")

    def test_moneyline_shape_football_and_hockey_stay_two_way(self) -> None:
        # only soccer is three-way: football ties push, NHL resolves in OT/SO
        assert {"SOCCER"} == THREE_WAY_MONEYLINE_SPORTS
        assert "FOOTBALL" not in THREE_WAY_MONEYLINE_SPORTS
        assert "HOCKEY" not in THREE_WAY_MONEYLINE_SPORTS


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
        # game-market markets the sport never registers stay None
        assert await registry.get_active("BASKETBALL", "FUTURE") is None
        assert set(registry.active_map()) == {"BASKETBALL_SPREAD", "BASKETBALL_TOTAL", "BASKETBALL_MONEYLINE"}
        # PLAYER_PROP is keyed separately and bootstraps lazily (Wave 3)
        prop_loaded = await registry.get_active("BASKETBALL", "PLAYER_PROP")
        assert prop_loaded is not None
        assert prop_loaded.record.model_type == "PLAYER_PROP"
        assert "BASKETBALL_PLAYER_PROP" in registry.active_map()

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
        with pytest.raises(ValueError, match=f"no synthetic generator registered for {UNREGISTERED}"):
            await registry.ensure_bootstrap(UNREGISTERED)

    async def test_get_active_lazily_bootstraps_and_propagates_registry_errors(self, model_dir) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), model_dir)  # type: ignore[arg-type]
        # lazy bootstrap on first use (no explicit ensure_bootstrap call)
        assert await registry.get_active("BASKETBALL", "SPREAD") is not None
        with pytest.raises(ValueError, match=f"no synthetic generator registered for {UNREGISTERED}"):
            await registry.get_active(UNREGISTERED, "MONEYLINE")

    async def test_startup_bootstrap_reports_failure_without_crashing(self, model_dir) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), model_dir, sports=["BASKETBALL", UNREGISTERED])  # type: ignore[arg-type]
        assert await registry.try_bootstrap() is False
        assert await registry.get_active("BASKETBALL", "TOTAL") is not None

    async def test_startup_bootstrap_succeeds_when_soccer_configured(self, model_dir) -> None:
        # PREDICTION_SPORTS=BASKETBALL,SOCCER startup path (default stays BASKETBALL)
        registry = ModelRegistry(FakeModelVersionRepo(), model_dir, sports=["BASKETBALL", "SOCCER"])  # type: ignore[arg-type]
        assert await registry.try_bootstrap() is True
        assert await registry.get_active("SOCCER", "MONEYLINE") is not None


@pytest.fixture(scope="module")
def baseball_model_dir(tmp_path_factory: pytest.TempPathFactory):
    """A MODEL_DIR pre-seeded with a tiny baseball artifact (fast bootstrap)."""
    directory = tmp_path_factory.mktemp("baseball-models")
    result = train_model(
        generate_baseball_synthetic_dataset(n_games=240), n_rounds=8, data_label="synthetic", sport="BASEBALL"
    )
    save_artifact(result, directory)
    return directory


class TestBaseballModelRegistry:
    async def test_baseball_bootstrap_registers_all_markets(self, baseball_model_dir) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), baseball_model_dir)  # type: ignore[arg-type]
        await registry.ensure_bootstrap("BASEBALL")

        for market in ("SPREAD", "TOTAL", "MONEYLINE"):
            loaded = await registry.get_active("BASEBALL", market)
            assert loaded is not None
            assert loaded.record.sport == "BASEBALL"
            assert loaded.record.feature_names == list(BASEBALL_FEATURES)


@pytest.fixture(scope="module")
def waves345_model_dir(tmp_path_factory: pytest.TempPathFactory):
    """A MODEL_DIR pre-seeded with tiny football/hockey/NCAA_BB artifacts."""
    directory = tmp_path_factory.mktemp("waves345-models")
    for gen, sport, kw in (
        (generate_football_synthetic_dataset, "FOOTBALL", {"n_games": 300}),
        (generate_hockey_synthetic_dataset, "HOCKEY", {"n_games": 300}),
        (generate_ncaa_bb_synthetic_dataset, "NCAA_BB", {"n_rows": 600}),
    ):
        result = train_model(gen(**kw), n_rounds=8, data_label="synthetic", sport=sport)
        save_artifact(result, directory)
    return directory


class TestWaves345ModelRegistry:
    @pytest.mark.parametrize(
        ("sport", "features"),
        [
            ("FOOTBALL", FOOTBALL_FEATURES),
            ("HOCKEY", HOCKEY_FEATURES),
            ("NCAA_BB", NCAA_BB_FEATURES),
        ],
    )
    async def test_bootstrap_registers_all_markets(self, waves345_model_dir, sport, features) -> None:
        registry = ModelRegistry(FakeModelVersionRepo(), waves345_model_dir)  # type: ignore[arg-type]
        await registry.ensure_bootstrap(sport)

        for market in ("SPREAD", "TOTAL", "MONEYLINE"):
            loaded = await registry.get_active(sport, market)
            assert loaded is not None
            assert loaded.record.sport == sport
            assert loaded.record.feature_names == list(features)
