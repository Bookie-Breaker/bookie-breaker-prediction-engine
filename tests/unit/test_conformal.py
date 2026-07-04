"""Split conformal interval tests."""

import numpy as np
import pytest

from prediction_engine.core.conformal import SplitConformal


class TestSplitConformal:
    def test_interval_symmetric_and_clipped(self) -> None:
        conformal = SplitConformal(half_width=0.08)
        lower, upper = conformal.interval(0.5)
        assert lower == pytest.approx(0.42)
        assert upper == pytest.approx(0.58)
        assert conformal.interval(0.03) == (0.0, pytest.approx(0.11))
        assert conformal.interval(0.97) == (pytest.approx(0.89), 1.0)

    def test_empirical_coverage_near_ninety_percent(self) -> None:
        rng = np.random.default_rng(5)
        calibration_residuals = rng.normal(0, 0.05, 2000)
        conformal = SplitConformal.fit(calibration_residuals)

        fresh = rng.normal(0, 0.05, 4000)
        covered = np.mean(np.abs(fresh) <= conformal.half_width)
        assert 0.85 <= covered <= 0.95

    def test_empty_residuals_fall_back(self) -> None:
        conformal = SplitConformal.fit(np.array([]))
        assert conformal.half_width == 0.1
