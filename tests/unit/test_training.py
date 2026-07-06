"""Trainer smoke tests on the seeded synthetic datasets (fast rounds)."""

import numpy as np
import pytest

from prediction_engine.core.features.registry import BASEBALL_FEATURES, NBA_FEATURES, SOCCER_FEATURES
from prediction_engine.core.model.artifact import ArtifactBundle, find_latest_artifact
from prediction_engine.core.training.synthetic import (
    generate_baseball_synthetic_dataset,
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
