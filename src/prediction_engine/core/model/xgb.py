"""Typed wrapper over the XGBoost booster.

Confines the partially-typed xgboost surface to one module. SHAP-style
feature attributions come from Booster.predict(pred_contribs=True) -- no
shap package dependency.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from prediction_engine.core.features.registry import FeatureMap


def vectorize(features: FeatureMap, feature_names: Sequence[str]) -> npt.NDArray[np.float64]:
    """Map a feature dict to the model's input vector; missing -> NaN."""
    row = np.full(len(feature_names), np.nan, dtype=np.float64)
    for i, name in enumerate(feature_names):
        value = features.get(name)
        if value is not None:
            row[i] = float(value)
    return row


class AdjustmentModel:
    """Predicts the probability adjustment (target = outcome - sim_prob)."""

    def __init__(self, booster: xgb.Booster, feature_names: Sequence[str]) -> None:
        self._booster = booster
        self._feature_names = list(feature_names)

    @classmethod
    def load(cls, path: Path, feature_names: Sequence[str]) -> "AdjustmentModel":
        booster = xgb.Booster()
        booster.load_model(str(path))
        return cls(booster, feature_names)

    def save(self, path: Path) -> None:
        self._booster.save_model(str(path))

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _dmatrix(self, features: FeatureMap) -> xgb.DMatrix:
        row = vectorize(features, self._feature_names).reshape(1, -1)
        return xgb.DMatrix(row, feature_names=self._feature_names, missing=np.nan)

    def predict_adjustment(self, features: FeatureMap) -> float:
        prediction = self._booster.predict(self._dmatrix(features))
        return float(prediction[0])

    def feature_importance(self, features: FeatureMap, top_k: int = 5) -> dict[str, float]:
        """Normalized absolute per-feature contributions for this prediction."""
        contribs = cast(
            npt.NDArray[np.float64],
            self._booster.predict(self._dmatrix(features), pred_contribs=True),
        )[0]
        # last column is the bias term
        magnitudes = np.abs(contribs[:-1])
        total = float(magnitudes.sum())
        if total == 0.0:
            return {}
        ranked = sorted(zip(self._feature_names, magnitudes, strict=True), key=lambda kv: -kv[1])[:top_k]
        return {name: round(float(value) / total, 4) for name, value in ranked if value > 0}
