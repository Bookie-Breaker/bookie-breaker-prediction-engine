"""Model training pipeline per algorithms/prediction-models.md.

- Never random splits: seasons are the split unit. The last season is held
  out and divided between Platt calibration and conformal fitting; earlier
  seasons train the booster (expanding window).
- Sample weights decay exponentially with season age (half-life 2 seasons).
- Target is the adjustment: outcome - simulation probability.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from prediction_engine.core.calibration import PlattCalibrator
from prediction_engine.core.conformal import SplitConformal
from prediction_engine.core.features.registry import get_features
from prediction_engine.core.model.artifact import ArtifactBundle
from prediction_engine.core.model.xgb import AdjustmentModel
from prediction_engine.core.training.dataset import TrainingSet
from prediction_engine.core.training.evaluate import brier_score, expected_calibration_error, log_loss

SEASON_HALF_LIFE = 2.0

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "seed": 42,
}
DEFAULT_ROUNDS = 200

# Version-tag prefix per sport. BASKETBALL keeps the historical "NBA" prefix
# (locked into registered artifact tags since Phase 2); pooled-competition
# sports such as SOCCER tag with the sport name (ADR-026).
_VERSION_TAG_PREFIX: dict[str, str] = {"BASKETBALL": "NBA"}


@dataclass
class TrainingResult:
    bundle: ArtifactBundle
    metrics: dict[str, float]
    training_samples: int


def _season_weights(seasons: np.ndarray) -> np.ndarray:
    age = seasons.max() - seasons
    weights: np.ndarray = np.exp(-np.log(2.0) * age / SEASON_HALF_LIFE)
    return weights


def train_model(
    dataset: TrainingSet,
    n_rounds: int = DEFAULT_ROUNDS,
    params: dict[str, Any] | None = None,
    data_label: str = "synthetic",
    sport: str = "BASKETBALL",
) -> TrainingResult:
    feature_names = list(get_features(sport))
    params = {**DEFAULT_XGB_PARAMS, **(params or {})}

    last_season = int(dataset.seasons.max())
    train_mask = dataset.seasons < last_season
    holdout_mask = ~train_mask
    if not train_mask.any() or not holdout_mask.any():
        raise ValueError("training data must span at least two seasons for walk-forward validation")

    train_set = dataset.subset(train_mask)
    holdout = dataset.subset(holdout_mask)
    # Alternate holdout rows between calibration and conformal splits
    calib_mask = np.arange(len(holdout)) % 2 == 0
    calibration_set = holdout.subset(calib_mask)
    conformal_set = holdout.subset(~calib_mask)

    dtrain = xgb.DMatrix(
        train_set.matrix(feature_names),
        label=train_set.targets,
        weight=_season_weights(train_set.seasons),
        feature_names=feature_names,
        missing=np.nan,
    )
    dcalib = xgb.DMatrix(
        calibration_set.matrix(feature_names),
        label=calibration_set.targets,
        feature_names=feature_names,
        missing=np.nan,
    )
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dcalib, "calibration")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    model = AdjustmentModel(booster, feature_names)

    def raw_probs(subset: TrainingSet) -> np.ndarray:
        matrix = xgb.DMatrix(subset.matrix(feature_names), feature_names=feature_names, missing=np.nan)
        adjustments = booster.predict(matrix)
        return np.asarray(np.clip(subset.sim_probs + adjustments, 0.01, 0.99), dtype=np.float64)

    calibrator = PlattCalibrator.fit(raw_probs(calibration_set), calibration_set.outcomes)

    conformal_probs = raw_probs(conformal_set)
    conformal_matrix = xgb.DMatrix(conformal_set.matrix(feature_names), feature_names=feature_names, missing=np.nan)
    adjustment_residuals = conformal_set.targets - booster.predict(conformal_matrix)
    conformal = SplitConformal.fit(np.asarray(adjustment_residuals, dtype=np.float64))

    calibrated = np.array([calibrator.apply(float(p)) for p in conformal_probs])
    metrics = {
        "brier_score": round(brier_score(calibrated, conformal_set.outcomes), 5),
        "brier_score_simulation_baseline": round(brier_score(conformal_set.sim_probs, conformal_set.outcomes), 5),
        "log_loss": round(log_loss(calibrated, conformal_set.outcomes), 5),
        "calibration_error": round(expected_calibration_error(calibrated, conformal_set.outcomes), 5),
    }

    trained_at = datetime.now(tz=UTC)
    content_hash = hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()[:8]
    tag_prefix = _VERSION_TAG_PREFIX.get(sport, sport)
    version_tag = f"{tag_prefix}_unified_{trained_at:%Y%m%d}_{content_hash}"
    metadata = {
        "version_tag": version_tag,
        "sport": sport,
        "algorithm": "xgboost",
        "data_label": data_label,
        "feature_names": feature_names,
        "trained_at": trained_at.isoformat(),
        "training_samples": len(train_set),
        "seasons": [int(s) for s in sorted(set(dataset.seasons.tolist()))],
        "hyperparameters": {**params, "num_boost_round": n_rounds},
        "metrics": metrics,
        "calibration": "platt",
        "conformal_alpha": 0.1,
    }

    bundle = ArtifactBundle(model=model, calibrator=calibrator, conformal=conformal, metadata=metadata)
    return TrainingResult(bundle=bundle, metrics=metrics, training_samples=len(train_set))


def save_artifact(result: TrainingResult, model_dir: Path) -> Path:
    sport = str(result.bundle.metadata.get("sport", "BASKETBALL"))
    directory = model_dir / sport.lower() / "unified" / result.bundle.version_tag
    result.bundle.save(directory)
    return directory
