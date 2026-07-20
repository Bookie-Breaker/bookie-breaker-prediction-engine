"""Ensemble adjustment model: weighted blend of XGBoost members (Wave 4).

Members are XGBoost boosters only, so the pickle-free artifact contract is
preserved: every member serializes to native .ubj. The v1 ensemble pairs
the existing gradient-boosted trees model with an XGBoost-emulated random
forest (num_parallel_tree with a single boosting round), whose decorrelated
per-node column sampling gives a genuinely different bias/variance profile
on the same features.

Blend weights live on the probability-adjustment scale: the ensemble's
predicted adjustment is the weighted sum of member adjustments, with
non-negative weights summing to 1 (fit on the calibration split by
minimizing Brier score over a simplex grid; see core/training/train.py).
"""

import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from prediction_engine.core.features.registry import FeatureMap
from prediction_engine.core.model.xgb import AdjustmentModel

BLEND_FILENAME = "blend.json"
RF_MODEL_FILENAME = "model_rf.ubj"
PRIMARY_MODEL_FILENAME = "model.ubj"

_WEIGHT_TOLERANCE = 1e-6

# XGBoost random-forest emulation: many parallel trees, one boosting round,
# full-strength learning rate, per-node column sampling for decorrelation.
RF_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "eta": 1.0,
    "subsample": 0.8,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 64,
    "min_child_weight": 10,
    "seed": 43,
}
RF_NUM_BOOST_ROUND = 1


def _validate_weights(members: Sequence[AdjustmentModel], weights: Sequence[float]) -> list[float]:
    if len(members) != len(weights):
        raise ValueError(f"got {len(members)} members but {len(weights)} weights")
    if not members:
        raise ValueError("an ensemble needs at least one member")
    if any(w < 0 for w in weights):
        raise ValueError(f"blend weights must be non-negative, got {list(weights)}")
    if abs(sum(weights) - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(f"blend weights must sum to 1, got {sum(weights)}")
    return [float(w) for w in weights]


class EnsembleAdjustmentModel:
    """Ordered members + simplex blend weights; same surface as AdjustmentModel."""

    def __init__(self, members: Sequence[AdjustmentModel], weights: Sequence[float]) -> None:
        self._members = list(members)
        self._weights = _validate_weights(members, weights)

    @property
    def members(self) -> list[AdjustmentModel]:
        return list(self._members)

    @property
    def weights(self) -> list[float]:
        return list(self._weights)

    @property
    def feature_names(self) -> list[str]:
        return self._members[0].feature_names

    def predict_adjustment(self, features: FeatureMap) -> float:
        return float(sum(w * m.predict_adjustment(features) for m, w in zip(self._members, self._weights, strict=True)))

    def feature_importance(self, features: FeatureMap, top_k: int = 5) -> dict[str, float]:
        """Blend-weighted average of member importances, re-ranked to top_k."""
        combined: dict[str, float] = {}
        for member, weight in zip(self._members, self._weights, strict=True):
            for name, value in member.feature_importance(features, top_k=top_k).items():
                combined[name] = combined.get(name, 0.0) + weight * value
        ranked = sorted(combined.items(), key=lambda kv: -kv[1])[:top_k]
        return {name: round(value, 4) for name, value in ranked if value > 0}

    def save_members(self, directory: Path) -> None:
        """Write model.ubj (primary GBT), model_rf.ubj, and blend.json.

        Loading a directory without blend.json falls back to the
        single-model layout, so every pre-Wave-4 artifact keeps working.
        """
        filenames = [PRIMARY_MODEL_FILENAME, RF_MODEL_FILENAME]
        if len(self._members) != len(filenames):
            raise ValueError(
                f"the v1 artifact layout stores exactly {len(filenames)} members, got {len(self._members)}"
            )
        for member, filename in zip(self._members, filenames, strict=True):
            member.save(directory / filename)
        (directory / BLEND_FILENAME).write_text(json.dumps({"members": filenames, "weights": self._weights}))

    @classmethod
    def load(cls, directory: Path, feature_names: Sequence[str]) -> "EnsembleAdjustmentModel":
        blend = json.loads((directory / BLEND_FILENAME).read_text())
        members = [AdjustmentModel.load(directory / str(filename), feature_names) for filename in blend["members"]]
        return cls(members, [float(w) for w in blend["weights"]])


def _simplex_grid(n_members: int, step: float) -> list[tuple[float, ...]]:
    """All non-negative weight vectors summing to 1 on a step grid."""
    ticks = int(round(1.0 / step))
    grid: list[tuple[float, ...]] = []
    # enumerate from full weight on the first (primary) member downward so
    # Brier ties resolve toward the primary
    for combo in itertools.product(range(ticks, -1, -1), repeat=n_members - 1):
        remainder = ticks - sum(combo)
        if remainder >= 0:
            grid.append(tuple(c * step for c in combo) + (remainder * step,))
    return grid


def fit_blend_weights(
    member_raw_probs: Sequence[npt.NDArray[np.float64]],
    outcomes: npt.NDArray[np.float64],
    step: float = 0.05,
) -> list[float]:
    """Simplex-grid search for the Brier-minimizing blend of member outputs.

    member_raw_probs holds each member's raw (pre-Platt) probabilities on
    the calibration split; the returned weights are non-negative and sum
    to 1. Ties resolve toward the earlier grid point, which favors weight
    on the primary (first) member.
    """
    if not member_raw_probs:
        raise ValueError("need at least one member to blend")
    stacked = np.vstack(member_raw_probs)
    best_weights: tuple[float, ...] | None = None
    best_brier = float("inf")
    for weights in _simplex_grid(len(member_raw_probs), step):
        blended = np.asarray(weights) @ stacked
        brier = float(np.mean((blended - outcomes) ** 2))
        if brier < best_brier - 1e-12:
            best_brier = brier
            best_weights = weights
    assert best_weights is not None  # the grid always contains at least one point
    return [round(w, 10) for w in best_weights]
