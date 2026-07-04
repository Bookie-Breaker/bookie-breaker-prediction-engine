"""Platt scaling calibration (ADR-014: Platt now, isotonic at 5000+ games).

Inference applies sigmoid(a * logit(p) + b) from stored parameters -- no
sklearn needed at runtime. Fitting (train-time only) uses sklearn logistic
regression on the logit of the uncalibrated probability.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class PlattCalibrator:
    a: float
    b: float

    def apply(self, probability: float) -> float:
        z = self.a * _logit(probability) + self.b
        return 1.0 / (1.0 + math.exp(-z))

    @classmethod
    def identity(cls) -> "PlattCalibrator":
        return cls(a=1.0, b=0.0)

    @classmethod
    def fit(cls, probabilities: npt.NDArray[np.float64], outcomes: npt.NDArray[np.float64]) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        clipped = np.clip(probabilities, _EPS, 1.0 - _EPS)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        model = LogisticRegression(C=1e6)  # effectively unregularized
        model.fit(logits, outcomes.astype(int))
        return cls(a=float(model.coef_[0][0]), b=float(model.intercept_[0]))
