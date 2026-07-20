"""Player-prop registry invariants (Phase 7 Wave 3): order-locked tuples,
stat one-hots, and synthetic generator/registry agreement."""

import pytest

from prediction_engine.core.features.registry import (
    PROP_FEATURES_BY_SPORT,
    PROP_STATS_BY_SPORT,
    YES_NO_PROP_STATS,
    get_prop_features,
    get_prop_stats,
)
from prediction_engine.core.training.synthetic import (
    PROP_SYNTHETIC_GENERATORS,
    generate_basketball_prop_dataset,
    generate_soccer_prop_dataset,
    get_prop_synthetic_generator,
)

PROP_SPORTS = ("SOCCER", "BASKETBALL", "BASEBALL", "FOOTBALL")


class TestPropFeatureRegistry:
    def test_all_four_sports_registered(self) -> None:
        assert set(PROP_FEATURES_BY_SPORT) == set(PROP_SPORTS)
        assert set(PROP_STATS_BY_SPORT) == set(PROP_SPORTS)

    @pytest.mark.parametrize("sport", PROP_SPORTS)
    def test_tuple_structure_is_order_locked(self, sport: str) -> None:
        features = get_prop_features(sport)
        stats = get_prop_stats(sport)
        # baseline block, one one-hot per stat in stat order, side flags last
        expected = (
            "sim_prop_probability",
            "prop_line",
            "sim_stat_mean",
            "sim_stat_std",
            *(f"prop_is_{stat}" for stat in stats),
            "side_is_over",
            "side_is_yes",
        )
        assert features == expected

    @pytest.mark.parametrize("sport", PROP_SPORTS)
    def test_no_market_signal_block(self, sport: str) -> None:
        # documented exclusion: no per-player prop line feed from lines-service
        features = get_prop_features(sport)
        assert "line_movement" not in features
        assert "n_books_reporting" not in features
        assert "line_consensus_std" not in features

    def test_yes_no_stats_are_registered_stats(self) -> None:
        all_stats = {stat for stats in PROP_STATS_BY_SPORT.values() for stat in stats}
        assert all_stats >= YES_NO_PROP_STATS
        assert {"player_goal_scorer_anytime", "player_anytime_td"} == YES_NO_PROP_STATS

    @pytest.mark.parametrize("sport", ["HOCKEY", "NCAA_BB", "CRICKET"])
    def test_unregistered_sports_fail_loudly(self, sport: str) -> None:
        with pytest.raises(ValueError, match="prop"):
            get_prop_features(sport)
        with pytest.raises(ValueError, match="prop"):
            get_prop_stats(sport)


class TestPropSyntheticRegistry:
    def test_generators_match_feature_registries(self) -> None:
        assert set(PROP_SYNTHETIC_GENERATORS) == set(PROP_FEATURES_BY_SPORT)

    def test_unregistered_sport_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="prop"):
            get_prop_synthetic_generator("HOCKEY")

    @pytest.mark.parametrize("sport", PROP_SPORTS)
    def test_generated_rows_follow_the_tuple_order(self, sport: str) -> None:
        dataset = get_prop_synthetic_generator(sport)()
        expected = list(get_prop_features(sport))
        # dict.fromkeys(tuple) seeds every row, so insertion order == tuple order
        assert list(dataset.features[0]) == expected
        assert list(dataset.features[-1]) == expected

    def test_soccer_yes_no_rows_have_zero_line_and_yes_flag(self) -> None:
        dataset = generate_soccer_prop_dataset(n_rows=500)
        yes_rows = [f for f in dataset.features if f["prop_is_player_goal_scorer_anytime"] == 1.0]
        over_rows = [f for f in dataset.features if f["prop_is_player_shots"] == 1.0]
        assert yes_rows and over_rows
        assert all(f["side_is_yes"] == 1.0 and f["side_is_over"] == 0.0 for f in yes_rows)
        assert all(f["prop_line"] == 0.0 for f in yes_rows)
        assert all(f["side_is_over"] == 1.0 and f["side_is_yes"] == 0.0 for f in over_rows)
        # count lines are half-step and positive
        assert all(f["prop_line"] % 1.0 == 0.5 for f in over_rows)

    def test_basketball_dataset_is_deterministic(self) -> None:
        first = generate_basketball_prop_dataset(n_rows=300)
        again = generate_basketball_prop_dataset(n_rows=300)
        assert first.features == again.features
        assert (first.sim_probs == again.sim_probs).all()
        assert (first.outcomes == again.outcomes).all()

    def test_exactly_one_stat_one_hot_per_row(self) -> None:
        dataset = generate_basketball_prop_dataset(n_rows=300)
        for row in dataset.features[:100]:
            one_hots = [value for name, value in row.items() if name.startswith("prop_is_")]
            assert sum(v or 0.0 for v in one_hots) == 1.0
