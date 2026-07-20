"""Prop model training smoke tests (Phase 7 Wave 3): the synthetic prop
bootstrap beats the sim baseline, conformal coverage holds, and artifacts
land under the props family without touching unified paths."""

import numpy as np
import pytest
import xgboost as xgb

from prediction_engine.core.features.registry import (
    BASKETBALL_PROP_FEATURES,
    SOCCER_PROP_FEATURES,
)
from prediction_engine.core.model.artifact import ArtifactBundle, find_latest_artifact
from prediction_engine.core.training.synthetic import (
    generate_basketball_prop_dataset,
    generate_soccer_prop_dataset,
)
from prediction_engine.core.training.train import save_artifact, train_model


@pytest.fixture(scope="module")
def soccer_prop_result():
    dataset = generate_soccer_prop_dataset()
    return train_model(dataset, n_rounds=150, data_label="synthetic", sport="SOCCER", market="PLAYER_PROP")


@pytest.fixture(scope="module")
def basketball_prop_result():
    dataset = generate_basketball_prop_dataset()
    return train_model(dataset, n_rounds=150, data_label="synthetic", sport="BASKETBALL", market="PLAYER_PROP")


class TestSoccerPropTraining:
    def test_beats_simulation_baseline(self, soccer_prop_result) -> None:
        metrics = soccer_prop_result.metrics
        assert metrics["brier_score"] < metrics["brier_score_simulation_baseline"]
        assert soccer_prop_result.training_samples > 0

    def test_metadata_is_props_tagged(self, soccer_prop_result) -> None:
        metadata = soccer_prop_result.bundle.metadata
        assert metadata["feature_names"] == list(SOCCER_PROP_FEATURES)
        assert metadata["version_tag"].startswith("SOCCER_props_")
        assert metadata["family"] == "props"
        assert metadata["sport"] == "SOCCER"

    def test_artifact_saves_under_props_family(self, soccer_prop_result, tmp_path) -> None:
        artifact_dir = save_artifact(soccer_prop_result, tmp_path)
        assert artifact_dir.parent == tmp_path / "soccer" / "props"
        assert find_latest_artifact(tmp_path, "soccer", family="props") == artifact_dir
        # unified paths untouched: the game-model lookup still finds nothing
        assert find_latest_artifact(tmp_path, "soccer") is None
        loaded = ArtifactBundle.load(artifact_dir)
        assert loaded.feature_names == list(SOCCER_PROP_FEATURES)


class TestBasketballPropTraining:
    def test_beats_simulation_baseline(self, basketball_prop_result) -> None:
        metrics = basketball_prop_result.metrics
        assert metrics["brier_score"] < metrics["brier_score_simulation_baseline"]

    def test_metadata_is_props_tagged(self, basketball_prop_result) -> None:
        metadata = basketball_prop_result.bundle.metadata
        assert metadata["feature_names"] == list(BASKETBALL_PROP_FEATURES)
        # BASKETBALL keeps the historical NBA version-tag prefix
        assert metadata["version_tag"].startswith("NBA_props_")

    def test_conformal_coverage_near_ninety_percent(self, basketball_prop_result) -> None:
        # split conformal promises ~1-alpha coverage of the adjustment
        # residuals on exchangeable data; check on a fresh seeded sample
        bundle = basketball_prop_result.bundle
        fresh = generate_basketball_prop_dataset(n_rows=2_000, seed=97)
        feature_names = bundle.feature_names
        matrix = xgb.DMatrix(fresh.matrix(feature_names), feature_names=feature_names, missing=np.nan)
        adjustments = bundle.model._booster.predict(matrix)  # noqa: SLF001 - test-only booster access
        residuals = np.abs(fresh.targets - adjustments)
        coverage = float((residuals <= bundle.conformal.half_width).mean())
        assert bundle.conformal.alpha == 0.1
        assert 0.85 <= coverage <= 0.97

    def test_calibrated_probabilities_are_proper(self, basketball_prop_result) -> None:
        bundle = basketball_prop_result.bundle
        fresh = generate_basketball_prop_dataset(n_rows=200, seed=101)
        for features, sim_prob in zip(fresh.features[:50], fresh.sim_probs[:50], strict=False):
            adjustment = bundle.model.predict_adjustment(features)
            raw = float(np.clip(sim_prob + adjustment, 0.01, 0.99))
            calibrated = bundle.calibrator.apply(raw)
            assert 0.0 < calibrated < 1.0
