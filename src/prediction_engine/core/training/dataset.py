"""Training dataset container."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from prediction_engine.core.features.registry import FeatureMap
from prediction_engine.core.model.xgb import vectorize


@dataclass
class TrainingSet:
    """Rows are (features, simulation probability, binary outcome, season index)."""

    features: list[FeatureMap]
    sim_probs: npt.NDArray[np.float64]
    outcomes: npt.NDArray[np.float64]
    seasons: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.features)

    def matrix(self, feature_names: list[str]) -> npt.NDArray[np.float64]:
        return np.vstack([vectorize(row, feature_names) for row in self.features])

    @property
    def targets(self) -> npt.NDArray[np.float64]:
        """Adjustment targets: outcome - simulation probability."""
        result: npt.NDArray[np.float64] = self.outcomes - self.sim_probs
        return result

    def subset(self, mask: npt.NDArray[np.bool_]) -> "TrainingSet":
        indices = np.flatnonzero(mask)
        return TrainingSet(
            features=[self.features[i] for i in indices],
            sim_probs=self.sim_probs[mask],
            outcomes=self.outcomes[mask],
            seasons=self.seasons[mask],
        )
