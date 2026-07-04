"""Evaluation metrics per algorithms/prediction-models.md."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def brier_score(probabilities: FloatArray, outcomes: FloatArray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def log_loss(probabilities: FloatArray, outcomes: FloatArray, eps: float = 1e-9) -> float:
    p = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1.0 - outcomes) * np.log(1.0 - p)))


def expected_calibration_error(probabilities: FloatArray, outcomes: FloatArray, n_bins: int = 10) -> float:
    """ECE with equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probabilities)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lo) & (probabilities < hi) if hi < 1.0 else (probabilities >= lo)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(float(outcomes[mask].mean()) - float(probabilities[mask].mean()))
    return float(ece)
