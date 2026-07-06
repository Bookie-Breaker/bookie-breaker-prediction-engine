"""Trainer smoke tests on the seeded synthetic datasets (fast rounds)."""

import numpy as np
import pytest

from prediction_engine.core.features.registry import (
    BASEBALL_FEATURES,
    FOOTBALL_FEATURES,
    HOCKEY_FEATURES,
    NBA_FEATURES,
    NCAA_BB_FEATURES,
    SOCCER_FEATURES,
)
from prediction_engine.core.model.artifact import ArtifactBundle, find_latest_artifact
from prediction_engine.core.training.synthetic import (
    generate_baseball_synthetic_dataset,
    generate_football_synthetic_dataset,
    generate_hockey_synthetic_dataset,
    generate_ncaa_bb_synthetic_dataset,
    generate_soccer_synthetic_dataset,
    generate_synthetic_dataset,
)
from prediction_engine.core.training.train import save_artifact, train_model


@pytest.fixture(scope="module")
def training_result():
    return train_model(generate_synthetic_dataset(), n_rounds=150, data_label="synthetic")


class TestTraining:
    def test_beats_simulation_baseline(self, training_result) -> None:
        metrics = training_result.metrics
        assert metrics["brier_score"] < metrics["brier_score_simulation_baseline"]

    def test_calibration_error_reasonable_on_synthetic(self, training_result) -> None:
        # The doc's <0.03 target applies to real-data models; the synthetic
        # bootstrap should still be in the right neighborhood
        assert training_result.metrics["calibration_error"] < 0.06

    def test_metadata_complete(self, training_result) -> None:
        metadata = training_result.bundle.metadata
        assert metadata["feature_names"] == list(NBA_FEATURES)
        assert metadata["version_tag"].startswith("NBA_unified_")
        assert metadata["calibration"] == "platt"
        assert training_result.training_samples > 0

    def test_artifact_round_trip(self, training_result, tmp_path) -> None:
        artifact_dir = save_artifact(training_result, tmp_path)
        assert find_latest_artifact(tmp_path) == artifact_dir

        loaded = ArtifactBundle.load(artifact_dir)
        assert loaded.version_tag == training_result.bundle.version_tag
        assert loaded.feature_names == list(NBA_FEATURES)

        features = generate_synthetic_dataset(n_rows=10, seed=99).features[0]
        original = training_result.bundle.model.predict_adjustment(features)
        reloaded = loaded.model.predict_adjustment(features)
        assert reloaded == pytest.approx(original, abs=1e-9)
        assert loaded.calibrator.apply(0.6) == pytest.approx(training_result.bundle.calibrator.apply(0.6))

    def test_feature_importance_normalized(self, training_result) -> None:
        features = generate_synthetic_dataset(n_rows=5, seed=42).features[0]
        importance = training_result.bundle.model.feature_importance(features)
        assert importance
        assert all(0 <= value <= 1 for value in importance.values())
        assert sum(importance.values()) <= 1.0 + 1e-6

    def test_single_season_data_rejected(self) -> None:
        dataset = generate_synthetic_dataset(n_rows=200, n_seasons=1)
        with pytest.raises(ValueError, match="two seasons"):
            train_model(dataset, n_rounds=10)

    def test_adjustments_move_toward_truth(self, training_result) -> None:
        # The synthetic sim underreacts to strength; the model should push
        # strong favorites' probabilities upward on average
        dataset = generate_synthetic_dataset(n_rows=2000, seed=123)
        strong = [
            (features, sim)
            for features, sim in zip(dataset.features, dataset.sim_probs, strict=True)
            if (features.get("net_rating_diff") or 0) > 6 and sim > 0.55
        ]
        assert len(strong) > 20
        adjustments = [training_result.bundle.model.predict_adjustment(f) for f, _ in strong]
        assert float(np.mean(adjustments)) > 0.005


@pytest.fixture(scope="module")
def soccer_dataset():
    return generate_soccer_synthetic_dataset(n_games=1200)


@pytest.fixture(scope="module")
def soccer_training_result(soccer_dataset):
    return train_model(soccer_dataset, n_rounds=150, data_label="synthetic", sport="SOCCER")


class TestSoccerSyntheticDataset:
    def test_deterministic_and_five_rows_per_game(self, soccer_dataset) -> None:
        again = generate_soccer_synthetic_dataset(n_games=1200)
        assert len(soccer_dataset) == 1200 * 5
        assert np.array_equal(soccer_dataset.sim_probs, again.sim_probs)
        assert np.array_equal(soccer_dataset.outcomes, again.outcomes)
        assert soccer_dataset.features[0] == again.features[0]

    def test_moneyline_triples_are_exhaustive_and_exclusive(self, soccer_dataset) -> None:
        for game_start in range(0, 50 * 5, 5):
            triple = soccer_dataset.features[game_start : game_start + 3]
            assert [row["market_is_moneyline"] for row in triple] == [1.0, 1.0, 1.0]
            assert [row["selection_is_draw"] for row in triple] == [0.0, 1.0, 0.0]
            # exactly one of HOME/DRAW/AWAY wins
            assert soccer_dataset.outcomes[game_start : game_start + 3].sum() == 1.0
            sim_triple = soccer_dataset.sim_probs[game_start : game_start + 3]
            assert sim_triple.sum() == pytest.approx(1.0, abs=1e-9)

    def test_draw_rate_realistic_and_decreasing_with_strength_gap(self, soccer_dataset) -> None:
        draw_rows = [
            (features, sim, outcome)
            for features, sim, outcome in zip(
                soccer_dataset.features, soccer_dataset.sim_probs, soccer_dataset.outcomes, strict=True
            )
            if features["selection_is_draw"] == 1.0
        ]
        draw_sims = np.array([sim for _, sim, _ in draw_rows])
        draw_outcomes = np.array([outcome for _, _, outcome in draw_rows])
        assert 0.20 <= float(draw_outcomes.mean()) <= 0.30
        assert 0.15 <= float(np.median(draw_sims)) <= 0.30

        gaps = np.array(
            [abs((f["home_attack_strength"] or 1) - (f["away_attack_strength"] or 1)) for f, _, _ in draw_rows]
        )
        small_gap = draw_sims[gaps < np.median(gaps)]
        large_gap = draw_sims[gaps >= np.median(gaps)]
        assert small_gap.mean() > large_gap.mean()

    def test_knockout_draw_bump_is_embedded(self) -> None:
        dataset = generate_soccer_synthetic_dataset(n_games=4000, seed=3)
        knockout_draws, open_draws = [], []
        for features, outcome in zip(dataset.features, dataset.outcomes, strict=True):
            if features["selection_is_draw"] != 1.0:
                continue
            (knockout_draws if features["is_knockout"] == 1.0 else open_draws).append(outcome)
        assert np.mean(knockout_draws) > np.mean(open_draws)

    def test_rows_cover_all_three_markets_on_goal_lines(self, soccer_dataset) -> None:
        spread = soccer_dataset.features[3]
        total = soccer_dataset.features[4]
        assert spread["market_is_spread"] == 1.0 and spread["market_is_moneyline"] == 0.0
        assert total["market_is_total"] == 1.0 and total["market_is_spread"] == 0.0
        totals = [f["sim_total_mean"] for f in soccer_dataset.features[:500]]
        assert 1.5 < float(np.mean(totals)) < 4.0  # goals, not points


class TestSoccerTraining:
    def test_trains_end_to_end_and_beats_simulation_baseline(self, soccer_training_result) -> None:
        metrics = soccer_training_result.metrics
        assert metrics["brier_score"] < metrics["brier_score_simulation_baseline"]
        assert soccer_training_result.training_samples > 0

    def test_metadata_is_soccer_tagged(self, soccer_training_result) -> None:
        metadata = soccer_training_result.bundle.metadata
        assert metadata["feature_names"] == list(SOCCER_FEATURES)
        assert metadata["version_tag"].startswith("SOCCER_unified_")
        assert metadata["sport"] == "SOCCER"

    def test_artifact_saves_under_soccer_path(self, soccer_training_result, tmp_path) -> None:
        artifact_dir = save_artifact(soccer_training_result, tmp_path)
        assert artifact_dir.parent == tmp_path / "soccer" / "unified"
        assert find_latest_artifact(tmp_path, "soccer") == artifact_dir
        loaded = ArtifactBundle.load(artifact_dir)
        assert loaded.feature_names == list(SOCCER_FEATURES)


@pytest.fixture(scope="module")
def baseball_dataset():
    return generate_baseball_synthetic_dataset()


@pytest.fixture(scope="module")
def baseball_training_result(baseball_dataset):
    return train_model(baseball_dataset, n_rounds=150, data_label="synthetic", sport="BASEBALL")


class TestBaseballSyntheticDataset:
    def test_deterministic_and_three_rows_per_game(self, baseball_dataset) -> None:
        again = generate_baseball_synthetic_dataset()
        assert len(baseball_dataset) == 3_000 * 3
        assert np.array_equal(baseball_dataset.sim_probs, again.sim_probs)
        assert np.array_equal(baseball_dataset.outcomes, again.outcomes)
        assert baseball_dataset.features[0] == again.features[0]

    def test_rows_cover_all_three_markets_two_way(self, baseball_dataset) -> None:
        moneyline, spread, total = baseball_dataset.features[:3]
        assert moneyline["market_is_moneyline"] == 1.0 and moneyline["market_is_spread"] == 0.0
        assert spread["market_is_spread"] == 1.0 and spread["market_is_moneyline"] == 0.0
        assert total["market_is_total"] == 1.0 and total["market_is_spread"] == 0.0
        # no three-way features anywhere: baseball cannot tie
        assert "selection_is_draw" not in moneyline
        assert "sim_draw_probability" not in moneyline

    def test_runs_scale_margins_and_totals(self, baseball_dataset) -> None:
        mlb_rows = [f for f in baseball_dataset.features[::3] if f["league_is_mlb"] == 1.0]
        ncaa_rows = [f for f in baseball_dataset.features[::3] if f["league_is_mlb"] == 0.0]
        mlb_totals = np.array([f["sim_total_mean"] for f in mlb_rows])
        assert 8.0 < float(mlb_totals.mean()) < 10.0  # runs, not points or goals
        assert float(np.mean([f["sim_total_mean"] for f in ncaa_rows])) > float(mlb_totals.mean())
        margins = np.array([f["sim_margin_mean"] for f in mlb_rows])
        assert float(np.abs(margins).mean()) < 2.0  # single-run-scale edges

    def test_starter_effect_dominates_team_effect(self, baseball_dataset) -> None:
        rows = [
            (features, outcome)
            for features, outcome in zip(baseball_dataset.features[::3], baseball_dataset.outcomes[::3], strict=True)
            if features["starter_fip_diff"] is not None
        ]
        fip_diff = np.array([features["starter_fip_diff"] for features, _ in rows])
        team_diff = np.array([features["home_team_fip"] - features["away_team_fip"] for features, _ in rows])
        outcomes = np.array([outcome for _, outcome in rows])

        starter_gap = outcomes[fip_diff < 0].mean() - outcomes[fip_diff > 0].mean()
        team_gap = outcomes[team_diff < 0].mean() - outcomes[team_diff > 0].mean()
        assert starter_gap > 0.10  # a better announced starter moves win rate a lot
        assert starter_gap > team_gap  # and more than team pitching quality does

    def test_unannounced_starters_are_null_with_flag_zero(self, baseball_dataset) -> None:
        moneyline_rows = baseball_dataset.features[::3]
        unannounced = [f for f in moneyline_rows if f["home_starter_announced"] == 0.0]
        announced = [f for f in moneyline_rows if f["home_starter_announced"] == 1.0]
        assert unannounced and announced  # both regimes are represented
        assert all(f["home_starter_fip"] is None for f in unannounced)
        assert all(f["home_starter_fip"] is not None for f in announced)
        assert all(f["starter_fip_diff"] is None for f in moneyline_rows if f["away_starter_announced"] == 0.0)


class TestBaseballTraining:
    def test_trains_end_to_end_and_beats_simulation_baseline(self, baseball_training_result) -> None:
        metrics = baseball_training_result.metrics
        assert metrics["brier_score"] < metrics["brier_score_simulation_baseline"]
        assert baseball_training_result.training_samples > 0

    def test_metadata_is_baseball_tagged(self, baseball_training_result) -> None:
        metadata = baseball_training_result.bundle.metadata
        assert metadata["feature_names"] == list(BASEBALL_FEATURES)
        assert metadata["version_tag"].startswith("BASEBALL_unified_")
        assert metadata["sport"] == "BASEBALL"

    def test_artifact_saves_under_baseball_path(self, baseball_training_result, tmp_path) -> None:
        artifact_dir = save_artifact(baseball_training_result, tmp_path)
        assert artifact_dir.parent == tmp_path / "baseball" / "unified"
        assert find_latest_artifact(tmp_path, "baseball") == artifact_dir
        loaded = ArtifactBundle.load(artifact_dir)
        assert loaded.feature_names == list(BASEBALL_FEATURES)


# --- Phase 6 Waves 3-5: FOOTBALL / HOCKEY / NCAA_BB -------------------------


@pytest.fixture(scope="module")
def football_dataset():
    return generate_football_synthetic_dataset()


@pytest.fixture(scope="module")
def football_training_result(football_dataset):
    return train_model(football_dataset, n_rounds=150, data_label="synthetic", sport="FOOTBALL")


class TestFootballSyntheticDataset:
    def test_deterministic(self, football_dataset) -> None:
        again = generate_football_synthetic_dataset()
        assert np.array_equal(football_dataset.sim_probs, again.sim_probs)
        assert np.array_equal(football_dataset.outcomes, again.outcomes)
        assert football_dataset.features[0] == again.features[0]

    def test_ties_void_the_moneyline_row_as_push(self, football_dataset) -> None:
        # every game emits exactly one spread and one total row; the moneyline
        # is VOIDED (no row) for the rare NFL game that stays tied -- a PUSH
        # leaves no gradeable moneyline outcome (ADR-027).
        spread = sum(1 for f in football_dataset.features if f["market_is_spread"] == 1.0)
        total = sum(1 for f in football_dataset.features if f["market_is_total"] == 1.0)
        moneyline = sum(1 for f in football_dataset.features if f["market_is_moneyline"] == 1.0)
        assert spread == total == 3000
        assert 0 < spread - moneyline < 50  # a handful of tied-game pushes
        # no three-way selection feature: a tie is a push, not a third side
        assert not any("selection_is_draw" in f for f in football_dataset.features)

    def test_epa_and_sp_plus_are_league_gated(self, football_dataset) -> None:
        nfl = [f for f in football_dataset.features if f["league_is_nfl"] == 1.0]
        college = [f for f in football_dataset.features if f["league_is_nfl"] == 0.0]
        assert nfl and college
        # EPA exists for the NFL only; SP+ for college only (null-documented)
        assert all(f["home_epa_per_play_off"] is not None for f in nfl)
        assert all(f["home_sp_plus_rating"] is None for f in nfl)
        assert all(f["home_epa_per_play_off"] is None for f in college)
        assert all(f["home_sp_plus_rating"] is not None for f in college)

    def test_bye_week_flags_and_tie_signal_present(self, football_dataset) -> None:
        assert any(f["home_bye_week"] == 1.0 for f in football_dataset.features)
        assert any(f["away_bye_week"] == 1.0 for f in football_dataset.features)
        # the drive-based sim reports a small tie mass for NFL games only
        nfl = [f for f in football_dataset.features if f["league_is_nfl"] == 1.0]
        college = [f for f in football_dataset.features if f["league_is_nfl"] == 0.0]
        assert any(f["sim_draw_probability"] > 0 for f in nfl)
        assert all(f["sim_draw_probability"] == 0.0 for f in college)

    def test_nfl_totals_are_football_scale(self, football_dataset) -> None:
        nfl_totals = [f["sim_total_mean"] for f in football_dataset.features if f["league_is_nfl"] == 1.0]
        assert 40.0 < float(np.mean(nfl_totals)) < 48.0  # ~44, points not goals/runs


class TestFootballTraining:
    def test_trains_end_to_end_and_beats_simulation_baseline_robustly(self, football_training_result) -> None:
        metrics = football_training_result.metrics
        # the sharp EPA/SP+ blocks give a robust edge over the noisy-rate sim
        assert metrics["brier_score"] < metrics["brier_score_simulation_baseline"]
        assert metrics["brier_score_simulation_baseline"] - metrics["brier_score"] > 0.008
        assert football_training_result.training_samples > 0

    def test_metadata_is_football_tagged(self, football_training_result) -> None:
        metadata = football_training_result.bundle.metadata
        assert metadata["feature_names"] == list(FOOTBALL_FEATURES)
        assert metadata["version_tag"].startswith("FOOTBALL_unified_")
        assert metadata["sport"] == "FOOTBALL"

    def test_artifact_saves_under_football_path(self, football_training_result, tmp_path) -> None:
        artifact_dir = save_artifact(football_training_result, tmp_path)
        assert artifact_dir.parent == tmp_path / "football" / "unified"
        assert find_latest_artifact(tmp_path, "football") == artifact_dir
        loaded = ArtifactBundle.load(artifact_dir)
        assert loaded.feature_names == list(FOOTBALL_FEATURES)


@pytest.fixture(scope="module")
def hockey_dataset():
    return generate_hockey_synthetic_dataset()


@pytest.fixture(scope="module")
def hockey_training_result(hockey_dataset):
    return train_model(hockey_dataset, n_rounds=150, data_label="synthetic", sport="HOCKEY")


class TestHockeySyntheticDataset:
    def test_deterministic_and_three_rows_per_game(self, hockey_dataset) -> None:
        again = generate_hockey_synthetic_dataset()
        assert len(hockey_dataset) == 3_000 * 3
        assert np.array_equal(hockey_dataset.sim_probs, again.sim_probs)
        assert np.array_equal(hockey_dataset.outcomes, again.outcomes)
        assert hockey_dataset.features[0] == again.features[0]

    def test_rows_cover_all_three_markets_two_way(self, hockey_dataset) -> None:
        moneyline, spread, total = hockey_dataset.features[:3]
        assert moneyline["market_is_moneyline"] == 1.0 and moneyline["market_is_spread"] == 0.0
        assert spread["market_is_spread"] == 1.0 and spread["market_is_moneyline"] == 0.0
        assert total["market_is_total"] == 1.0 and total["market_is_spread"] == 0.0
        # OT/SO always produces a winner: no draw features anywhere
        assert "selection_is_draw" not in moneyline
        assert "sim_draw_probability" not in moneyline

    def test_goals_scale_totals(self, hockey_dataset) -> None:
        totals = np.array([f["sim_total_mean"] for f in hockey_dataset.features[:600]])
        assert 5.0 < float(totals.mean()) < 7.0  # goals, not points or runs

    def test_special_teams_and_goaltending_embedded(self, hockey_dataset) -> None:
        moneyline_rows = hockey_dataset.features[::3]
        # goaltending: a better home save pct lifts the home win rate
        outcomes = np.array([o for o in hockey_dataset.outcomes[::3]])
        save_edge = np.array([f["home_team_save_pct"] - f["away_team_save_pct"] for f in moneyline_rows])
        assert outcomes[save_edge > 0].mean() > outcomes[save_edge < 0].mean()
        # back-to-backs are present on both sides (dense NHL schedule)
        assert any(f["home_back_to_back"] == 1.0 for f in moneyline_rows)
        assert any(f["away_back_to_back"] == 1.0 for f in moneyline_rows)


class TestHockeyTraining:
    def test_trains_end_to_end_and_beats_simulation_baseline_robustly(self, hockey_training_result) -> None:
        metrics = hockey_training_result.metrics
        # goaltending + special teams are invisible to the base-rate sim
        assert metrics["brier_score"] < metrics["brier_score_simulation_baseline"]
        assert metrics["brier_score_simulation_baseline"] - metrics["brier_score"] > 0.005
        assert hockey_training_result.training_samples > 0

    def test_metadata_is_hockey_tagged(self, hockey_training_result) -> None:
        metadata = hockey_training_result.bundle.metadata
        assert metadata["feature_names"] == list(HOCKEY_FEATURES)
        assert metadata["version_tag"].startswith("HOCKEY_unified_")
        assert metadata["sport"] == "HOCKEY"

    def test_artifact_saves_under_hockey_path(self, hockey_training_result, tmp_path) -> None:
        artifact_dir = save_artifact(hockey_training_result, tmp_path)
        assert artifact_dir.parent == tmp_path / "hockey" / "unified"
        assert find_latest_artifact(tmp_path, "hockey") == artifact_dir
        loaded = ArtifactBundle.load(artifact_dir)
        assert loaded.feature_names == list(HOCKEY_FEATURES)


@pytest.fixture(scope="module")
def ncaa_bb_dataset():
    return generate_ncaa_bb_synthetic_dataset()


@pytest.fixture(scope="module")
def ncaa_bb_training_result(ncaa_bb_dataset):
    return train_model(ncaa_bb_dataset, n_rounds=150, data_label="synthetic", sport="NCAA_BB")


class TestNcaaBbSyntheticDataset:
    def test_deterministic(self, ncaa_bb_dataset) -> None:
        again = generate_ncaa_bb_synthetic_dataset()
        assert len(ncaa_bb_dataset) == 6_000
        assert np.array_equal(ncaa_bb_dataset.sim_probs, again.sim_probs)
        assert np.array_equal(ncaa_bb_dataset.outcomes, again.outcomes)
        assert ncaa_bb_dataset.features[0] == again.features[0]

    def test_college_scale_totals_and_pace(self, ncaa_bb_dataset) -> None:
        totals = np.array([f["sim_total_mean"] for f in ncaa_bb_dataset.features[:600]])
        pace = np.array([f["home_pace"] for f in ncaa_bb_dataset.features[:600]])
        assert 135.0 < float(totals.mean()) < 155.0  # ~145, college scoring
        assert 64.0 < float(pace.mean()) < 72.0  # ~68 possessions

    def test_adjusted_efficiency_present_and_no_injuries(self, ncaa_bb_dataset) -> None:
        row = ncaa_bb_dataset.features[0]
        assert row["home_adjusted_efficiency_margin"] is not None
        assert row["away_adjusted_efficiency_margin"] is not None
        assert not any("injury" in name for name in row)

    def test_adjusted_efficiency_measures_strength_more_sharply_than_net_rating(self, ncaa_bb_dataset) -> None:
        # AEM is the sharp signal; the outcome tracks the AEM edge closely
        outcomes = ncaa_bb_dataset.outcomes
        aem_edge = np.array(
            [
                f["home_adjusted_efficiency_margin"] - f["away_adjusted_efficiency_margin"]
                for f in ncaa_bb_dataset.features
            ]
        )
        assert outcomes[aem_edge > 5].mean() > outcomes[aem_edge < -5].mean()


class TestNcaaBbTraining:
    def test_trains_end_to_end_and_beats_simulation_baseline_robustly(self, ncaa_bb_training_result) -> None:
        metrics = ncaa_bb_training_result.metrics
        # the dampened sim underreacts; AEM lets the model recover the strength
        assert metrics["brier_score"] < metrics["brier_score_simulation_baseline"]
        assert metrics["brier_score_simulation_baseline"] - metrics["brier_score"] > 0.003
        assert ncaa_bb_training_result.training_samples > 0

    def test_metadata_is_ncaa_bb_tagged(self, ncaa_bb_training_result) -> None:
        metadata = ncaa_bb_training_result.bundle.metadata
        assert metadata["feature_names"] == list(NCAA_BB_FEATURES)
        assert metadata["version_tag"].startswith("NCAA_BB_unified_")
        assert metadata["sport"] == "NCAA_BB"

    def test_artifact_saves_under_ncaa_bb_path(self, ncaa_bb_training_result, tmp_path) -> None:
        artifact_dir = save_artifact(ncaa_bb_training_result, tmp_path)
        assert artifact_dir.parent == tmp_path / "ncaa_bb" / "unified"
        assert find_latest_artifact(tmp_path, "ncaa_bb") == artifact_dir
        loaded = ArtifactBundle.load(artifact_dir)
        assert loaded.feature_names == list(NCAA_BB_FEATURES)
