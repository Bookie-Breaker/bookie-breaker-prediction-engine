"""Trainer smoke tests on the seeded synthetic dataset (fast rounds)."""

import numpy as np
import pytest

from prediction_engine.core.features.registry import NBA_FEATURES
from prediction_engine.core.model.artifact import ArtifactBundle, find_latest_artifact
from prediction_engine.core.training.synthetic import generate_synthetic_dataset
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
