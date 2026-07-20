"""Champion/challenger experiment evaluation (Phase 7 Wave 4).

Pure math over graded shadow pairs: for every (game, market, side) key both
models scored and whose game settled without a push, the champion's and
challenger's calibrated probabilities are compared against the realized
binary outcome. Promotion requires the challenger to beat the champion on
BOTH Brier score and log loss over at least ``min_samples`` graded pairs,
with expected calibration error no worse than the champion's plus
``ECE_TOLERANCE`` (a challenger may sharpen accuracy but must not
meaningfully degrade calibration, which the edge math depends on).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from prediction_engine.core.training.evaluate import brier_score, expected_calibration_error, log_loss

ECE_TOLERANCE = 0.005


@dataclass(frozen=True)
class ModelExperimentMetrics:
    model_version_id: str
    role: str
    brier_score: float | None
    log_loss: float | None
    calibration_error: float | None


@dataclass(frozen=True)
class ExperimentReport:
    sport: str
    market_type: str
    graded_pairs: int
    champion: ModelExperimentMetrics
    challenger: ModelExperimentMetrics
    promotion_ready: bool
    blockers: list[str]


def _metrics(
    model_version_id: str, role: str, probs: npt.NDArray[np.float64], outcomes: npt.NDArray[np.float64]
) -> ModelExperimentMetrics:
    if len(outcomes) == 0:
        return ModelExperimentMetrics(model_version_id, role, None, None, None)
    return ModelExperimentMetrics(
        model_version_id=model_version_id,
        role=role,
        brier_score=round(brier_score(probs, outcomes), 5),
        log_loss=round(log_loss(probs, outcomes), 5),
        calibration_error=round(expected_calibration_error(probs, outcomes), 5),
    )


def build_report(
    sport: str,
    market_type: str,
    champion_id: str,
    challenger_id: str,
    champion_probs: npt.NDArray[np.float64],
    challenger_probs: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    min_samples: int,
) -> ExperimentReport:
    """Score both models over the shared graded pairs and apply the promotion criteria."""
    if not (len(champion_probs) == len(challenger_probs) == len(outcomes)):
        raise ValueError("champion, challenger, and outcome arrays must be paired")
    champion = _metrics(champion_id, "champion", champion_probs, outcomes)
    challenger = _metrics(challenger_id, "challenger", challenger_probs, outcomes)
    graded_pairs = len(outcomes)

    blockers: list[str] = []
    if graded_pairs < min_samples:
        blockers.append(f"only {graded_pairs} graded pairs; promotion requires {min_samples}")
    if graded_pairs > 0:
        assert champion.brier_score is not None and challenger.brier_score is not None
        assert champion.log_loss is not None and challenger.log_loss is not None
        assert champion.calibration_error is not None and challenger.calibration_error is not None
        if challenger.brier_score >= champion.brier_score:
            blockers.append(f"challenger Brier {challenger.brier_score} does not beat champion {champion.brier_score}")
        if challenger.log_loss >= champion.log_loss:
            blockers.append(f"challenger log loss {challenger.log_loss} does not beat champion {champion.log_loss}")
        if challenger.calibration_error > champion.calibration_error + ECE_TOLERANCE:
            blockers.append(
                f"challenger ECE {challenger.calibration_error} exceeds champion "
                f"{champion.calibration_error} + {ECE_TOLERANCE}"
            )

    return ExperimentReport(
        sport=sport,
        market_type=market_type,
        graded_pairs=graded_pairs,
        champion=champion,
        challenger=challenger,
        promotion_ready=not blockers,
        blockers=blockers,
    )
