"""Seeded synthetic training data for per-sport bootstrap models.

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

from collections.abc import Callable

import numpy as np

from prediction_engine.core.features.registry import NBA_FEATURES, SOCCER_FEATURES, FeatureMap
from prediction_engine.core.training.dataset import TrainingSet

SIM_DAMPENING = 0.7
REST_EFFECT = 0.02
INJURY_EFFECT = -0.03
BACK_TO_BACK_EFFECT = -0.015

# Soccer generative model (independent Poisson goals, the classic
# Maher/Dixon-Coles structure): draws fall out of the goal distributions
# naturally in the 0.20-0.30 range and shrink as the strength gap grows.
SOCCER_BASE_GOALS = 1.15  # league-average goals per team per match
SOCCER_HOME_ADVANTAGE = 0.20  # log-scale boost to home expected goals
SOCCER_STRENGTH_STD = 0.45  # latent attack/conceding spread (log scale)
SOCCER_SIM_DAMPENING = 0.6  # the soccer "simulation" underreacts too
SOCCER_SIM_NOISE = 0.08  # logit-scale noise on simulated probabilities
SOCCER_REST_EFFECT = 0.06  # log-goal swing per normalized rest-advantage unit
KNOCKOUT_GOAL_FACTOR = 0.85  # knockouts are cagier -> fewer goals -> more draws

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


_MAX_GOALS = 12  # Poisson grid truncation; P(>12) is negligible at soccer rates


def _poisson_pmf(rates: np.ndarray) -> np.ndarray:
    """Per-game Poisson pmf over 0.._MAX_GOALS goals, shape (n, _MAX_GOALS+1)."""
    goals = np.arange(_MAX_GOALS + 1)
    log_factorial = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, _MAX_GOALS + 1)))))
    result: np.ndarray = np.exp(-rates[:, None] + goals[None, :] * np.log(rates)[:, None] - log_factorial[None, :])
    return result


def _dampen(true_probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The "simulation" underreacts: dampened logit response plus noise."""
    logits = np.log(true_probs / (1.0 - true_probs))
    noisy = SOCCER_SIM_DAMPENING * logits + rng.normal(0.0, SOCCER_SIM_NOISE, true_probs.shape)
    result: np.ndarray = np.clip(1.0 / (1.0 + np.exp(-noisy)), 0.02, 0.96)
    return result


def generate_soccer_synthetic_dataset(n_games: int = 1_200, seed: int = 11, n_seasons: int = 4) -> TrainingSet:
    """Per-selection binary rows for the pooled SOCCER bootstrap (ADR-026/027).

    Each game emits five rows: HOME/DRAW/AWAY moneyline sides plus a goal-line
    spread (HOME) and a totals (OVER) row. Truth is independent Poisson goals
    from latent attack/conceding strengths; the "simulation" sees the base
    rates only (dampened, per the NBA pattern), so rest advantage and the
    knockout draw-rate bump are genuine signals for the model to learn.
    """
    rng = np.random.default_rng(seed)

    attack_home = rng.normal(0.0, SOCCER_STRENGTH_STD, n_games)
    attack_away = rng.normal(0.0, SOCCER_STRENGTH_STD, n_games)
    concede_home = rng.normal(0.0, SOCCER_STRENGTH_STD, n_games)
    concede_away = rng.normal(0.0, SOCCER_STRENGTH_STD, n_games)

    is_fifa_wc = rng.random(n_games) < 0.30
    is_knockout = is_fifa_wc & (rng.random(n_games) < 0.40)
    home_rest = rng.integers(2, 8, n_games).astype(np.float64)
    away_rest = rng.integers(2, 8, n_games).astype(np.float64)
    rest_norm = np.clip(home_rest - away_rest, -4, 4) / 4.0

    # Base rates: what the simulation is allowed to know about
    lambda_home = SOCCER_BASE_GOALS * np.exp(attack_home + concede_away + SOCCER_HOME_ADVANTAGE)
    lambda_away = SOCCER_BASE_GOALS * np.exp(attack_away + concede_home)

    # True rates add the effects hidden from the simulation
    knockout_factor = np.where(is_knockout, KNOCKOUT_GOAL_FACTOR, 1.0)
    true_home = lambda_home * np.exp(SOCCER_REST_EFFECT * rest_norm / 2) * knockout_factor
    true_away = lambda_away * np.exp(-SOCCER_REST_EFFECT * rest_norm / 2) * knockout_factor

    # Simulation-visible outcome distribution from the base-rate Poisson grid
    joint = _poisson_pmf(lambda_home)[:, :, None] * _poisson_pmf(lambda_away)[:, None, :]
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    goals = np.arange(_MAX_GOALS + 1)
    margin_grid = goals[:, None] - goals[None, :]
    total_grid = goals[:, None] + goals[None, :]

    base_home = (joint * (margin_grid > 0)).sum(axis=(1, 2))
    base_draw = (joint * (margin_grid == 0)).sum(axis=(1, 2))
    base_away = 1.0 - base_home - base_draw

    spread_lines = rng.choice(np.array([-1.5, -0.5, 0.5, 1.5]), n_games)
    total_lines = rng.choice(np.array([2.5, 3.5]), n_games)
    base_cover = np.stack([(joint * (margin_grid > -line)).sum(axis=(1, 2)) for line in (-1.5, -0.5, 0.5, 1.5)])
    base_over = np.stack([(joint * (total_grid > line)).sum(axis=(1, 2)) for line in (2.5, 3.5)])
    spread_idx = np.searchsorted(np.array([-1.5, -0.5, 0.5, 1.5]), spread_lines)
    total_idx = np.searchsorted(np.array([2.5, 3.5]), total_lines)
    base_cover_prob = base_cover[spread_idx, np.arange(n_games)]
    base_over_prob = base_over[total_idx, np.arange(n_games)]

    sim_home = _dampen(base_home, rng)
    sim_draw = _dampen(base_draw, rng)
    sim_away = _dampen(base_away, rng)
    triple_mass = sim_home + sim_draw + sim_away
    sim_home, sim_draw, sim_away = sim_home / triple_mass, sim_draw / triple_mass, sim_away / triple_mass
    sim_cover = _dampen(base_cover_prob, rng)
    sim_over = _dampen(base_over_prob, rng)

    # Actual match outcomes sampled from the true (adjusted) rates
    home_goals = rng.poisson(true_home)
    away_goals = rng.poisson(true_away)
    margin = home_goals - away_goals
    total_goals = home_goals + away_goals

    game_seasons = (np.arange(n_games) * n_seasons // n_games).astype(np.int64)
    sim_margin_mean = lambda_home - lambda_away + rng.normal(0.0, 0.05, n_games)
    sim_total_mean = lambda_home + lambda_away + rng.normal(0.0, 0.1, n_games)

    features: list[FeatureMap] = []
    sim_probs: list[float] = []
    outcomes: list[float] = []
    seasons: list[int] = []

    for i in range(n_games):
        shared: FeatureMap = dict.fromkeys(SOCCER_FEATURES, None)
        shared.update(
            {
                "sim_margin_mean": float(sim_margin_mean[i]),
                "sim_total_mean": float(sim_total_mean[i]),
                "sim_draw_probability": float(sim_draw[i]),
                "sim_converged": 1.0,
                "home_attack_strength": float(np.exp(attack_home[i]) + rng.normal(0, 0.03)),
                "home_defense_strength": float(np.exp(concede_home[i]) + rng.normal(0, 0.03)),
                "away_attack_strength": float(np.exp(attack_away[i]) + rng.normal(0, 0.03)),
                "away_defense_strength": float(np.exp(concede_away[i]) + rng.normal(0, 0.03)),
                "home_goals_for_per_match": float(SOCCER_BASE_GOALS * np.exp(attack_home[i]) + rng.normal(0, 0.15)),
                "home_goals_against_per_match": float(
                    SOCCER_BASE_GOALS * np.exp(concede_home[i]) + rng.normal(0, 0.15)
                ),
                "away_goals_for_per_match": float(SOCCER_BASE_GOALS * np.exp(attack_away[i]) + rng.normal(0, 0.15)),
                "away_goals_against_per_match": float(
                    SOCCER_BASE_GOALS * np.exp(concede_away[i]) + rng.normal(0, 0.15)
                ),
                "home_form_points_last5": float(
                    np.clip(round(rng.normal(7 + 4 * (attack_home[i] - concede_home[i]), 2.5)), 0, 15)
                ),
                "away_form_points_last5": float(
                    np.clip(round(rng.normal(7 + 4 * (attack_away[i] - concede_away[i]), 2.5)), 0, 15)
                ),
                "home_matches_played": float(rng.integers(3, 8) if is_fifa_wc[i] else rng.integers(10, 39)),
                "away_matches_played": float(rng.integers(3, 8) if is_fifa_wc[i] else rng.integers(10, 39)),
                "home_rest_days": float(home_rest[i]),
                "away_rest_days": float(away_rest[i]),
                "is_knockout": 1.0 if is_knockout[i] else 0.0,
                "competition_is_fifa_wc": 1.0 if is_fifa_wc[i] else 0.0,
                "line_movement": float(rng.normal(0, 0.3)),
                "n_books_reporting": float(rng.integers(3, 9)),
                "line_consensus_std": float(abs(rng.normal(0.1, 0.08))),
            }
        )

        moneyline_rows = (
            (float(sim_home[i]), 1.0 if margin[i] > 0 else 0.0, 0.0),
            (float(sim_draw[i]), 1.0 if margin[i] == 0 else 0.0, 1.0),
            (float(sim_away[i]), 1.0 if margin[i] < 0 else 0.0, 0.0),
        )
        for sim_prob, outcome, selection_is_draw in moneyline_rows:
            row = dict(shared)
            row.update(
                {
                    "sim_probability": sim_prob,
                    "market_is_spread": 0.0,
                    "market_is_total": 0.0,
                    "market_is_moneyline": 1.0,
                    "selection_is_draw": selection_is_draw,
                }
            )
            features.append(row)
            sim_probs.append(sim_prob)
            outcomes.append(outcome)
            seasons.append(int(game_seasons[i]))

        two_way_rows = (
            ("market_is_spread", float(sim_cover[i]), 1.0 if margin[i] + spread_lines[i] > 0 else 0.0),
            ("market_is_total", float(sim_over[i]), 1.0 if total_goals[i] > total_lines[i] else 0.0),
        )
        for market_key, sim_prob, outcome in two_way_rows:
            row = dict(shared)
            row.update(
                {
                    "sim_probability": sim_prob,
                    "market_is_spread": 0.0,
                    "market_is_total": 0.0,
                    "market_is_moneyline": 0.0,
                    "selection_is_draw": 0.0,
                }
            )
            row[market_key] = 1.0
            features.append(row)
            sim_probs.append(sim_prob)
            outcomes.append(outcome)
            seasons.append(int(game_seasons[i]))

    return TrainingSet(
        features=features,
        sim_probs=np.array(sim_probs, dtype=np.float64),
        outcomes=np.array(outcomes, dtype=np.float64),
        seasons=np.array(seasons, dtype=np.int64),
    )


# Sport -> synthetic bootstrap generator. Each league wave registers its
# sport's generator here (ADR-026); until then bootstrap fails loudly.
SYNTHETIC_GENERATORS: dict[str, Callable[[], TrainingSet]] = {
    "BASKETBALL": generate_synthetic_dataset,
    "SOCCER": generate_soccer_synthetic_dataset,
}


def get_synthetic_generator(sport: str) -> Callable[[], TrainingSet]:
    """Return the synthetic dataset generator for a sport."""
    try:
        return SYNTHETIC_GENERATORS[sport]
    except KeyError:
        raise ValueError(f"no synthetic generator registered for {sport}; added in its league wave") from None
