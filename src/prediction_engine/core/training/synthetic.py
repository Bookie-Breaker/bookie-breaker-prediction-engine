"""Seeded synthetic training data for the bootstrap model.

Real NBA training data collection (scripts/collect_nba_data.py) is deferred
to the verification session (offseason + datacenter IP blocking risk).
Until then, the service ships with a model trained on synthetic rows that
embed small known true effects, so calibration and conformal coverage are
meaningfully testable and the API produces sensible non-trivial
adjustments. Swapping in a real model is a single train.py run.

Embedded true effects the model can learn:
- The "simulation" systematically underreacts to team strength (its logit
  response is dampened to 0.7x the true one), so the model finds a genuine
  correction signal in net_rating_diff -- mirroring the real motivation for
  the ML layer (the simulation does not capture everything).
- rest advantage: +0.02 per normalized unit
- injury impact differential: -0.03 per normalized unit
- home back-to-back: -0.015
"""

import numpy as np

from prediction_engine.core.features.registry import NBA_FEATURES, FeatureMap
from prediction_engine.core.training.dataset import TrainingSet

SIM_DAMPENING = 0.7
REST_EFFECT = 0.02
INJURY_EFFECT = -0.03
BACK_TO_BACK_EFFECT = -0.015

_MARKETS = ("SPREAD", "TOTAL", "MONEYLINE")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return result


def generate_synthetic_dataset(n_rows: int = 6_000, seed: int = 7, n_seasons: int = 4) -> TrainingSet:
    rng = np.random.default_rng(seed)

    strength_diff = rng.normal(0.0, 0.7, n_rows)
    sim_probs = np.clip(_sigmoid(SIM_DAMPENING * strength_diff + rng.normal(0.0, 0.15, n_rows)), 0.05, 0.95)

    rest_advantage = rng.integers(-3, 4, n_rows).astype(np.float64)
    injury_diff = rng.normal(0.0, 3.0, n_rows)  # points-at-risk differential
    home_b2b = (rng.random(n_rows) < 0.15).astype(np.float64)
    market_idx = rng.integers(0, len(_MARKETS), n_rows)

    contextual_adjustment = (
        REST_EFFECT * (rest_advantage / 3.0)
        + INJURY_EFFECT * np.clip(injury_diff / 6.0, -1.5, 1.5) * -1.0  # more injured home team -> lower prob
        + BACK_TO_BACK_EFFECT * home_b2b
    )
    true_probs = np.clip(_sigmoid(strength_diff) + contextual_adjustment, 0.02, 0.98)
    outcomes = (rng.random(n_rows) < true_probs).astype(np.float64)
    seasons = (np.arange(n_rows) * n_seasons // n_rows).astype(np.int64)

    features: list[FeatureMap] = []
    for i in range(n_rows):
        pace_home = rng.normal(100, 3)
        pace_away = rng.normal(100, 3)
        net_home = strength_diff[i] * 4 + rng.normal(0, 1.5)
        net_away = -strength_diff[i] * 4 + rng.normal(0, 1.5)
        row: FeatureMap = dict.fromkeys(NBA_FEATURES, None)
        row.update(
            {
                "sim_probability": float(sim_probs[i]),
                "sim_margin_mean": float(strength_diff[i] * 5 + rng.normal(0, 1)),
                "sim_total_mean": float(rng.normal(224, 6)),
                "sim_converged": 1.0,
                "market_is_spread": 1.0 if market_idx[i] == 0 else 0.0,
                "market_is_total": 1.0 if market_idx[i] == 1 else 0.0,
                "market_is_moneyline": 1.0 if market_idx[i] == 2 else 0.0,
                "home_offensive_rating": float(112 + net_home / 2 + rng.normal(0, 1)),
                "home_defensive_rating": float(112 - net_home / 2 + rng.normal(0, 1)),
                "home_pace": float(pace_home),
                "home_net_rating": float(net_home),
                "away_offensive_rating": float(112 + net_away / 2 + rng.normal(0, 1)),
                "away_defensive_rating": float(112 - net_away / 2 + rng.normal(0, 1)),
                "away_pace": float(pace_away),
                "away_net_rating": float(net_away),
                "pace_differential": float(pace_home - pace_away),
                "net_rating_diff": float(net_home - net_away),
                "home_away_split_diff": float(rng.normal(2.0, 2.0)),
                "home_last5_ppg": float(112 + net_home / 2 + rng.normal(0, 3)),
                "home_last5_ppg_allowed": float(112 - net_home / 2 + rng.normal(0, 3)),
                "home_three_pct_last5": float(np.clip(rng.normal(0.36, 0.03), 0.25, 0.48)),
                "home_net_rating_last10": float(net_home + rng.normal(0, 2)),
                "away_last5_ppg": float(112 + net_away / 2 + rng.normal(0, 3)),
                "away_last5_ppg_allowed": float(112 - net_away / 2 + rng.normal(0, 3)),
                "away_three_pct_last5": float(np.clip(rng.normal(0.36, 0.03), 0.25, 0.48)),
                "away_net_rating_last10": float(net_away + rng.normal(0, 2)),
                "home_rest_days": float(max(rest_advantage[i], 0) + 1),
                "away_rest_days": float(max(-rest_advantage[i], 0) + 1),
                "rest_advantage": float(rest_advantage[i]),
                "home_back_to_back": float(home_b2b[i]),
                "away_back_to_back": 1.0 if rng.random() < 0.15 else 0.0,
                "home_injury_impact": float(-max(injury_diff[i], 0)),
                "away_injury_impact": float(-max(-injury_diff[i], 0)),
                "injury_impact_diff": float(-injury_diff[i]),
                "line_movement": float(rng.normal(0, 1.0)),
                "n_books_reporting": float(rng.integers(3, 9)),
                "line_consensus_std": float(abs(rng.normal(0.3, 0.2))),
            }
        )
        features.append(row)

    return TrainingSet(features=features, sim_probs=sim_probs, outcomes=outcomes, seasons=seasons)
