"""Platt calibration tests."""

import numpy as np
import pytest

from prediction_engine.core.calibration import PlattCalibrator


class TestPlattCalibrator:
    def test_identity_passes_through(self) -> None:
        calibrator = PlattCalibrator.identity()
        for p in (0.1, 0.35, 0.5, 0.72, 0.9):
            assert calibrator.apply(p) == pytest.approx(p, abs=1e-9)

    def test_monotone_increasing(self) -> None:
        calibrator = PlattCalibrator(a=0.8, b=0.1)
        probs = [0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95]
        calibrated = [calibrator.apply(p) for p in probs]
        assert calibrated == sorted(calibrated)
        assert all(0.0 < c < 1.0 for c in calibrated)

    def test_fit_recovers_known_distortion(self) -> None:
        # Outcomes generated from sigmoid(2 * logit(p)): overconfident probs
        # should be pulled toward the extremes by the fitted calibrator
        rng = np.random.default_rng(3)
        raw = rng.uniform(0.15, 0.85, 8000)
        logits = np.log(raw / (1 - raw))
        true = 1 / (1 + np.exp(-2.0 * logits))
        outcomes = (rng.random(8000) < true).astype(np.float64)

        calibrator = PlattCalibrator.fit(raw, outcomes)
        assert calibrator.a == pytest.approx(2.0, abs=0.3)
        assert calibrator.apply(0.7) > 0.7  # sharpens overcautious probabilities
        assert calibrator.apply(0.3) < 0.3

    def test_extreme_inputs_do_not_blow_up(self) -> None:
        calibrator = PlattCalibrator(a=1.2, b=-0.1)
        assert 0.0 <= calibrator.apply(0.0) <= 1.0
        assert 0.0 <= calibrator.apply(1.0) <= 1.0
