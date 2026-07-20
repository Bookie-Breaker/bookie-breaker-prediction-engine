"""Ensemble model, blend fitting, and artifact layout tests (Phase 7 Wave 4)."""

import json

import numpy as np
import pytest

from prediction_engine.core.model.artifact import ArtifactBundle
from prediction_engine.core.model.ensemble import (
    BLEND_FILENAME,
    RF_MODEL_FILENAME,
    EnsembleAdjustmentModel,
    fit_blend_weights,
)
from prediction_engine.core.model.xgb import AdjustmentModel
from prediction_engine.core.training.synthetic import generate_synthetic_dataset
from prediction_engine.core.training.train import save_artifact, train_model


@pytest.fixture(scope="module")
def single_result():
    return train_model(generate_synthetic_dataset(n_rows=1500), n_rounds=60, data_label="synthetic")


@pytest.fixture(scope="module")
def ensemble_result():
    return train_model(generate_synthetic_dataset(n_rows=1500), n_rounds=60, data_label="synthetic", ensemble=True)


class TestFitBlendWeights:
    def test_weights_lie_on_the_simplex(self) -> None:
        rng = np.random.default_rng(7)
        truth = rng.uniform(0.2, 0.8, size=500)
        outcomes = (rng.uniform(size=500) < truth).astype(np.float64)
        member_a = np.clip(truth + rng.normal(0, 0.10, size=500), 0.01, 0.99)
        member_b = np.clip(truth + rng.normal(0, 0.10, size=500), 0.01, 0.99)
        weights = fit_blend_weights([member_a, member_b], outcomes)
        assert len(weights) == 2
        assert all(w >= 0 for w in weights)
        assert sum(weights) == pytest.approx(1.0, abs=1e-9)

    def test_blend_brier_never_worse_than_best_member(self) -> None:
        # (1, 0) and (0, 1) are grid points, so the fitted blend's Brier is
        # bounded by the best single member's by construction; with
        # independent member noise the strict blend usually wins.
        rng = np.random.default_rng(11)
        truth = rng.uniform(0.2, 0.8, size=800)
        outcomes = (rng.uniform(size=800) < truth).astype(np.float64)
        members = [np.clip(truth + rng.normal(0, 0.12, size=800), 0.01, 0.99) for _ in range(2)]
        weights = fit_blend_weights(members, outcomes)

        def brier(probs: np.ndarray) -> float:
            return float(np.mean((probs - outcomes) ** 2))

        blended = np.asarray(weights) @ np.vstack(members)
        assert brier(blended) <= min(brier(m) for m in members) + 1e-12
        # both members carry signal here, so the blend should be genuinely mixed
        assert 0.0 < weights[0] < 1.0

    def test_identical_members_tie_toward_primary(self) -> None:
        rng = np.random.default_rng(3)
        outcomes = (rng.uniform(size=100) < 0.5).astype(np.float64)
        member = np.full(100, 0.5)
        weights = fit_blend_weights([member, member.copy()], outcomes)
        assert weights[0] == pytest.approx(1.0)

    def test_empty_members_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one member"):
            fit_blend_weights([], np.array([1.0]))


class TestEnsembleAdjustmentModel:
    def test_weight_validation(self, single_result) -> None:
        member = single_result.bundle.model
        assert isinstance(member, AdjustmentModel)
        with pytest.raises(ValueError, match="sum to 1"):
            EnsembleAdjustmentModel([member, member], [0.5, 0.6])
        with pytest.raises(ValueError, match="non-negative"):
            EnsembleAdjustmentModel([member, member], [1.5, -0.5])
        with pytest.raises(ValueError, match="2 members but 1 weights"):
            EnsembleAdjustmentModel([member, member], [1.0])

    def test_prediction_is_weighted_sum(self, single_result) -> None:
        member = single_result.bundle.model
        features = generate_synthetic_dataset(n_rows=5, seed=21).features[0]
        duo = EnsembleAdjustmentModel([member, member], [0.3, 0.7])
        # identical members: any simplex weighting reproduces the member
        assert duo.predict_adjustment(features) == pytest.approx(member.predict_adjustment(features), abs=1e-12)

    def test_feature_importance_normalized_and_ranked(self, ensemble_result) -> None:
        features = generate_synthetic_dataset(n_rows=5, seed=22).features[0]
        importance = ensemble_result.bundle.model.feature_importance(features)
        assert importance
        assert len(importance) <= 5
        assert all(0 <= value <= 1 for value in importance.values())
        assert sum(importance.values()) <= 1.0 + 1e-6


class TestEnsembleTraining:
    def test_trains_ensemble_with_simplex_metadata(self, ensemble_result) -> None:
        assert isinstance(ensemble_result.bundle.model, EnsembleAdjustmentModel)
        blend = ensemble_result.bundle.metadata["ensemble"]
        assert blend["members"] == ["gbt", "rf"]
        assert sum(blend["weights"]) == pytest.approx(1.0, abs=1e-9)
        assert all(w >= 0 for w in blend["weights"])

    def test_ensemble_beats_simulation_baseline(self, ensemble_result) -> None:
        metrics = ensemble_result.metrics
        assert metrics["brier_score"] < metrics["brier_score_simulation_baseline"]

    def test_rf_member_round_trips_ubj(self, ensemble_result, tmp_path) -> None:
        artifact_dir = save_artifact(ensemble_result, tmp_path)
        assert (artifact_dir / RF_MODEL_FILENAME).is_file()
        assert (artifact_dir / BLEND_FILENAME).is_file()

        loaded = ArtifactBundle.load(artifact_dir)
        assert isinstance(loaded.model, EnsembleAdjustmentModel)
        assert loaded.model.weights == ensemble_result.bundle.model.weights

        features = generate_synthetic_dataset(n_rows=10, seed=99).features[0]
        original = ensemble_result.bundle.model.predict_adjustment(features)
        assert loaded.model.predict_adjustment(features) == pytest.approx(original, abs=1e-9)
        # the RF member alone also round-trips exactly
        rf_original = ensemble_result.bundle.model.members[1].predict_adjustment(features)
        assert loaded.model.members[1].predict_adjustment(features) == pytest.approx(rf_original, abs=1e-9)

    def test_blend_json_layout(self, ensemble_result, tmp_path) -> None:
        artifact_dir = save_artifact(ensemble_result, tmp_path)
        blend = json.loads((artifact_dir / BLEND_FILENAME).read_text())
        assert blend["members"] == ["model.ubj", "model_rf.ubj"]
        assert sum(blend["weights"]) == pytest.approx(1.0, abs=1e-9)


class TestBackwardCompatibility:
    def test_single_model_artifact_has_no_blend_and_loads_as_before(self, single_result, tmp_path) -> None:
        artifact_dir = save_artifact(single_result, tmp_path)
        assert not (artifact_dir / BLEND_FILENAME).exists()
        assert not (artifact_dir / RF_MODEL_FILENAME).exists()

        loaded = ArtifactBundle.load(artifact_dir)
        assert isinstance(loaded.model, AdjustmentModel)
        features = generate_synthetic_dataset(n_rows=10, seed=99).features[0]
        assert loaded.model.predict_adjustment(features) == pytest.approx(
            single_result.bundle.model.predict_adjustment(features), abs=1e-9
        )
