"""Model artifact bundle: model.ubj + calibration.json + conformal.json + metadata.json.

No pickle anywhere: the XGBoost native format plus JSON parameter files
mean inference needs neither sklearn nor version-pinned deserialization.
Layout: $MODEL_DIR/{sport_lowercase}/{family}/<version_tag>/ where family is
"unified" for the game-market models and "props" for the player-prop models
(Phase 7 Wave 3).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prediction_engine.core.calibration import PlattCalibrator
from prediction_engine.core.conformal import SplitConformal
from prediction_engine.core.model.xgb import AdjustmentModel

MODEL_FILENAME = "model.ubj"
CALIBRATION_FILENAME = "calibration.json"
CONFORMAL_FILENAME = "conformal.json"
METADATA_FILENAME = "metadata.json"


@dataclass
class ArtifactBundle:
    model: AdjustmentModel
    calibrator: PlattCalibrator
    conformal: SplitConformal
    metadata: dict[str, Any]

    @property
    def version_tag(self) -> str:
        return str(self.metadata.get("version_tag", "unknown"))

    @property
    def feature_names(self) -> list[str]:
        return [str(name) for name in self.metadata.get("feature_names", [])]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / MODEL_FILENAME)
        (directory / CALIBRATION_FILENAME).write_text(
            json.dumps({"method": "platt", "a": self.calibrator.a, "b": self.calibrator.b})
        )
        (directory / CONFORMAL_FILENAME).write_text(
            json.dumps({"alpha": self.conformal.alpha, "half_width": self.conformal.half_width})
        )
        (directory / METADATA_FILENAME).write_text(json.dumps(self.metadata, indent=2, default=str))

    @classmethod
    def load(cls, directory: Path) -> "ArtifactBundle":
        metadata = json.loads((directory / METADATA_FILENAME).read_text())
        feature_names = [str(name) for name in metadata["feature_names"]]
        calibration = json.loads((directory / CALIBRATION_FILENAME).read_text())
        conformal = json.loads((directory / CONFORMAL_FILENAME).read_text())
        return cls(
            model=AdjustmentModel.load(directory / MODEL_FILENAME, feature_names),
            calibrator=PlattCalibrator(a=float(calibration["a"]), b=float(calibration["b"])),
            conformal=SplitConformal(half_width=float(conformal["half_width"]), alpha=float(conformal["alpha"])),
            metadata=metadata,
        )


def find_latest_artifact(model_dir: Path, sport_dir: str = "basketball", family: str = "unified") -> Path | None:
    """Locate the most recently created artifact directory for a sport under MODEL_DIR.

    family selects the artifact subdir: "unified" (default, game markets)
    or "props" (player-prop models).
    """
    base = model_dir / sport_dir / family
    if not base.is_dir():
        return None
    candidates = [d for d in base.iterdir() if (d / METADATA_FILENAME).is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / METADATA_FILENAME).stat().st_mtime)
