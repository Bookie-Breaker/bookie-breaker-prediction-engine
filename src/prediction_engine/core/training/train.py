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
from prediction_engine.core.features.registry import get_features, get_prop_features
from prediction_engine.core.model.artifact import ArtifactBundle
from prediction_engine.core.model.ensemble import (
    RF_NUM_BOOST_ROUND,
    RF_XGB_PARAMS,
    EnsembleAdjustmentModel,
    fit_blend_weights,
)
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
    market: str = "GAME",
    ensemble: bool = False,
) -> TrainingResult:
    """Train the adjustment model for a sport.

    market selects the model family: "GAME" (default, the unified
    SPREAD/TOTAL/MONEYLINE model) or "PLAYER_PROP" (the unified prop model,
    Phase 7 Wave 3), which uses the prop feature registry and saves under
    the "props" artifact family.

    ensemble=True (Phase 7 Wave 4) additionally trains an XGBoost
    random-forest member on the same walk-forward split, fits simplex blend
    weights on the calibration split by minimizing Brier, and then fits
    Platt + conformal on the BLENDED output, so calibration and intervals
    describe the model that actually serves.
    """
    is_prop = market == "PLAYER_PROP"
    feature_names = list(get_prop_features(sport)) if is_prop else list(get_features(sport))
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

    boosters: list[xgb.Booster] = [booster]
    weights: list[float] = [1.0]
    model: AdjustmentModel | EnsembleAdjustmentModel = AdjustmentModel(booster, feature_names)
    if ensemble:
        rf_booster = xgb.train(RF_XGB_PARAMS, dtrain, num_boost_round=RF_NUM_BOOST_ROUND)
        member_probs = [
            np.asarray(np.clip(calibration_set.sim_probs + b.predict(dcalib), 0.01, 0.99), dtype=np.float64)
            for b in (booster, rf_booster)
        ]
        weights = fit_blend_weights(member_probs, calibration_set.outcomes)
        boosters = [booster, rf_booster]
        model = EnsembleAdjustmentModel([AdjustmentModel(b, feature_names) for b in boosters], weights)

    def predict_adjustments(subset: TrainingSet) -> np.ndarray:
        matrix = xgb.DMatrix(subset.matrix(feature_names), feature_names=feature_names, missing=np.nan)
        stacked = np.vstack([b.predict(matrix) for b in boosters])
        return np.asarray(np.asarray(weights) @ stacked, dtype=np.float64)

    def raw_probs(subset: TrainingSet) -> np.ndarray:
        return np.asarray(np.clip(subset.sim_probs + predict_adjustments(subset), 0.01, 0.99), dtype=np.float64)

    calibrator = PlattCalibrator.fit(raw_probs(calibration_set), calibration_set.outcomes)

    conformal_probs = raw_probs(conformal_set)
    adjustment_residuals = conformal_set.targets - predict_adjustments(conformal_set)
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
    family = "props" if is_prop else "unified"
    version_tag = f"{tag_prefix}_{family}_{trained_at:%Y%m%d}_{content_hash}"
    metadata = {
        "version_tag": version_tag,
        "sport": sport,
        "family": family,
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
    if ensemble:
        metadata["ensemble"] = {
            "members": ["gbt", "rf"],
            "weights": weights,
            "rf_hyperparameters": {**RF_XGB_PARAMS, "num_boost_round": RF_NUM_BOOST_ROUND},
        }

    bundle = ArtifactBundle(model=model, calibrator=calibrator, conformal=conformal, metadata=metadata)
    return TrainingResult(bundle=bundle, metrics=metrics, training_samples=len(train_set))


def save_artifact(result: TrainingResult, model_dir: Path) -> Path:
    sport = str(result.bundle.metadata.get("sport", "BASKETBALL"))
    family = str(result.bundle.metadata.get("family", "unified"))
    directory = model_dir / sport.lower() / family / result.bundle.version_tag
    result.bundle.save(directory)
    return directory
