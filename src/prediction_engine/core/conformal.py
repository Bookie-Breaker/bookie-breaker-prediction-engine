"""Split conformal confidence intervals (ADR-014).

The nonconformity score is the absolute residual of the model's predicted
adjustment on a held-out calibration split. The 90% quantile (with the
standard finite-sample correction) becomes a symmetric half-width around
the calibrated probability, clipped to [0, 1]. This quantifies model
uncertainty about the adjustment rather than outcome variance -- outcome
variance for a binary event is irreducible.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SplitConformal:
    half_width: float
    alpha: float = 0.1

    def interval(self, probability: float) -> tuple[float, float]:
        lower = max(probability - self.half_width, 0.0)
        upper = min(probability + self.half_width, 1.0)
        return round(lower, 5), round(upper, 5)

    @classmethod
    def fit(cls, residuals: npt.NDArray[np.float64], alpha: float = 0.1) -> "SplitConformal":
        scores = np.abs(residuals)
        n = len(scores)
        if n == 0:
            return cls(half_width=0.1, alpha=alpha)
        # finite-sample corrected quantile level: ceil((n+1)(1-alpha))/n
        level = min(math.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
        return cls(half_width=float(np.quantile(scores, level)), alpha=alpha)
