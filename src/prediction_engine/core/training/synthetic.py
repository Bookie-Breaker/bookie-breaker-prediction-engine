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
from dataclasses import dataclass

import numpy as np

from prediction_engine.core.features.registry import (
    BASEBALL_FEATURES,
    BASEBALL_PROP_STATS,
    BASKETBALL_PROP_STATS,
    FOOTBALL_FEATURES,
    FOOTBALL_PROP_STATS,
    HOCKEY_FEATURES,
    NBA_FEATURES,
    NCAA_BB_FEATURES,
    SOCCER_FEATURES,
    SOCCER_PROP_STATS,
    FeatureMap,
    get_prop_features,
)
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

# Baseball generative model (independent Poisson runs; MLB + NCAA_BSB pooled
# per ADR-026): the dominant hidden effect is starter quality -- the
# "simulation" only sees team-level rates, so the announced starter's FIP
# diff is the model's biggest genuine correction signal, bigger than the
# team-strength correction. Unannounced starters still pitch (their latent
# quality shapes the outcome) but their stats are hidden from the features,
# so unannounced games are irreducibly noisier -- which makes the announced
# flags themselves informative.
BASEBALL_BASE_RUNS = 4.5  # MLB average runs per team per game
NCAA_BSB_BASE_RUNS = 5.9  # college baseball scores hotter
BASEBALL_HOME_ADVANTAGE = 0.12  # additive runs, split across the two sides
BASEBALL_TEAM_BAT_STD = 0.30  # latent offense spread (runs per game)
BASEBALL_TEAM_PITCH_STD = 0.25  # latent staff spread (runs prevented per game)
BASEBALL_STARTER_STD = 0.55  # latent starter spread (runs prevented) -- dominant
BASEBALL_ANNOUNCE_RATE = 0.85  # share of games with a probable starter posted
BASEBALL_SIM_DAMPENING = 0.65  # the baseball "simulation" underreacts too
BASEBALL_SIM_NOISE = 0.12  # logit-scale noise on simulated probabilities
BASEBALL_REST_EFFECT = 0.06  # runs swing per normalized rest-advantage unit
BASEBALL_FIP_BASE = 4.10  # league-average FIP anchor
BASEBALL_MLB_SHARE = 0.85  # MLB fraction of pooled rows (NCAA_BSB dormant)
_MAX_RUNS = 20  # Poisson grid truncation; P(>20) is negligible at run rates
_EXTRA_INNINGS_SLOPE = 0.35  # logit slope of P(home wins extras) on rate diff

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


def _poisson_pmf(rates: np.ndarray, max_count: int = _MAX_GOALS) -> np.ndarray:
    """Per-game Poisson pmf over 0..max_count, shape (n, max_count+1)."""
    counts = np.arange(max_count + 1)
    log_factorial = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, max_count + 1)))))
    result: np.ndarray = np.exp(-rates[:, None] + counts[None, :] * np.log(rates)[:, None] - log_factorial[None, :])
    return result


def _dampen(
    true_probs: np.ndarray,
    rng: np.random.Generator,
    dampening: float = SOCCER_SIM_DAMPENING,
    noise: float = SOCCER_SIM_NOISE,
) -> np.ndarray:
    """The "simulation" underreacts: dampened logit response plus noise."""
    logits = np.log(true_probs / (1.0 - true_probs))
    noisy = dampening * logits + rng.normal(0.0, noise, true_probs.shape)
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


def generate_baseball_synthetic_dataset(n_games: int = 3_000, seed: int = 13, n_seasons: int = 4) -> TrainingSet:
    """Per-selection binary rows for the pooled BASEBALL bootstrap (ADR-026).

    Each game emits three rows: a two-way HOME moneyline (baseball cannot
    tie -- zero-margin Poisson mass is resolved as strength-weighted
    extra-innings one-run wins), a run-line spread (the rate favorite laying
    -1.5, the dog getting +1.5), and a totals (OVER) row. Truth is
    independent Poisson runs from latent team batting/pitching plus starter
    quality; the "simulation" sees dampened team-only rates, so starter FIP
    (the dominant effect), rest advantage, and announcement status are
    genuine signals for the model to learn.
    """
    rng = np.random.default_rng(seed)

    is_mlb = rng.random(n_games) < BASEBALL_MLB_SHARE
    base_runs = np.where(is_mlb, BASEBALL_BASE_RUNS, NCAA_BSB_BASE_RUNS)

    bat_home = rng.normal(0.0, BASEBALL_TEAM_BAT_STD, n_games)
    bat_away = rng.normal(0.0, BASEBALL_TEAM_BAT_STD, n_games)
    pitch_home = rng.normal(0.0, BASEBALL_TEAM_PITCH_STD, n_games)
    pitch_away = rng.normal(0.0, BASEBALL_TEAM_PITCH_STD, n_games)
    starter_home = rng.normal(0.0, BASEBALL_STARTER_STD, n_games)
    starter_away = rng.normal(0.0, BASEBALL_STARTER_STD, n_games)
    announced_home = rng.random(n_games) < BASEBALL_ANNOUNCE_RATE
    announced_away = rng.random(n_games) < BASEBALL_ANNOUNCE_RATE

    rest_choices = np.array([0.0, 1.0, 2.0])
    home_rest = rng.choice(rest_choices, n_games, p=[0.55, 0.35, 0.10])
    away_rest = rng.choice(rest_choices, n_games, p=[0.55, 0.35, 0.10])
    rest_norm = (home_rest - away_rest) / 2.0

    # Base rates: what the simulation is allowed to know about (team level)
    lambda_home = np.clip(base_runs + BASEBALL_HOME_ADVANTAGE / 2 + bat_home - pitch_away, 0.5, None)
    lambda_away = np.clip(base_runs - BASEBALL_HOME_ADVANTAGE / 2 + bat_away - pitch_home, 0.5, None)

    # True rates add the effects hidden from the simulation
    true_home = np.clip(lambda_home - starter_away + BASEBALL_REST_EFFECT * rest_norm / 2, 0.3, None)
    true_away = np.clip(lambda_away - starter_home - BASEBALL_REST_EFFECT * rest_norm / 2, 0.3, None)

    # Simulation-visible outcome distribution from the base-rate Poisson grid
    joint = _poisson_pmf(lambda_home, _MAX_RUNS)[:, :, None] * _poisson_pmf(lambda_away, _MAX_RUNS)[:, None, :]
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    margin_pmf = np.stack(
        [np.trace(joint, offset=-m, axis1=1, axis2=2) for m in range(-_MAX_RUNS, _MAX_RUNS + 1)], axis=1
    )
    # Baseball never ties: move the zero-margin mass into one-run wins,
    # split by a strength-weighted extra-innings probability.
    p_extra_home = _sigmoid(_EXTRA_INNINGS_SLOPE * (lambda_home - lambda_away))
    tie_mass = margin_pmf[:, _MAX_RUNS].copy()
    margin_pmf[:, _MAX_RUNS] = 0.0
    margin_pmf[:, _MAX_RUNS + 1] += tie_mass * p_extra_home
    margin_pmf[:, _MAX_RUNS - 1] += tie_mass * (1.0 - p_extra_home)

    base_home = margin_pmf[:, _MAX_RUNS + 1 :].sum(axis=1)
    run_lines = np.where(lambda_home >= lambda_away, -1.5, 1.5)  # favorite lays the run line
    cover_minus = margin_pmf[:, _MAX_RUNS + 2 :].sum(axis=1)  # -1.5: win by 2+
    cover_plus = margin_pmf[:, _MAX_RUNS - 1 :].sum(axis=1)  # +1.5: lose by <=1 or win
    base_cover = np.where(run_lines < 0, cover_minus, cover_plus)

    total_pmf = np.stack(
        [np.trace(joint[:, ::-1, :], offset=t - _MAX_RUNS, axis1=1, axis2=2) for t in range(2 * _MAX_RUNS + 1)], axis=1
    )
    exp_total = lambda_home + lambda_away
    total_lines = np.floor(exp_total + rng.uniform(-1.0, 1.0, n_games)) + 0.5
    over_idx = np.ceil(total_lines).astype(np.int64)
    total_cdf = total_pmf.cumsum(axis=1)
    base_over = 1.0 - np.take_along_axis(total_cdf, (over_idx - 1)[:, None], axis=1)[:, 0]

    sim_home = _dampen(base_home, rng, BASEBALL_SIM_DAMPENING, BASEBALL_SIM_NOISE)
    sim_cover = _dampen(base_cover, rng, BASEBALL_SIM_DAMPENING, BASEBALL_SIM_NOISE)
    sim_over = _dampen(base_over, rng, BASEBALL_SIM_DAMPENING, BASEBALL_SIM_NOISE)

    # Actual game outcomes sampled from the true (starter-adjusted) rates
    home_runs = rng.poisson(true_home).astype(np.int64)
    away_runs = rng.poisson(true_away).astype(np.int64)
    ties = home_runs == away_runs
    extras_home_win = rng.random(n_games) < _sigmoid(_EXTRA_INNINGS_SLOPE * (true_home - true_away))
    home_runs = home_runs + (ties & extras_home_win)
    away_runs = away_runs + (ties & ~extras_home_win)
    margin = home_runs - away_runs
    total_runs = home_runs + away_runs

    game_seasons = (np.arange(n_games) * n_seasons // n_games).astype(np.int64)
    sim_margin_mean = lambda_home - lambda_away + rng.normal(0.0, 0.1, n_games)
    sim_total_mean = exp_total + rng.normal(0.0, 0.15, n_games)

    features: list[FeatureMap] = []
    sim_probs: list[float] = []
    outcomes: list[float] = []
    seasons: list[int] = []

    for i in range(n_games):
        shared: FeatureMap = dict.fromkeys(BASEBALL_FEATURES, None)
        shared.update(
            {
                "sim_margin_mean": float(sim_margin_mean[i]),
                "sim_total_mean": float(sim_total_mean[i]),
                "sim_converged": 1.0,
                "home_runs_scored_per_game": float(base_runs[i] + bat_home[i] + rng.normal(0, 0.12)),
                "home_runs_allowed_per_game": float(base_runs[i] - pitch_home[i] + rng.normal(0, 0.12)),
                "home_team_woba": float(0.315 + 0.020 * bat_home[i] + rng.normal(0, 0.003)),
                "home_team_fip": float(BASEBALL_FIP_BASE - 0.9 * pitch_home[i] + rng.normal(0, 0.05)),
                "home_bullpen_era": float(4.0 - 0.9 * pitch_home[i] + rng.normal(0, 0.30)),
                "away_runs_scored_per_game": float(base_runs[i] + bat_away[i] + rng.normal(0, 0.12)),
                "away_runs_allowed_per_game": float(base_runs[i] - pitch_away[i] + rng.normal(0, 0.12)),
                "away_team_woba": float(0.315 + 0.020 * bat_away[i] + rng.normal(0, 0.003)),
                "away_team_fip": float(BASEBALL_FIP_BASE - 0.9 * pitch_away[i] + rng.normal(0, 0.05)),
                "away_bullpen_era": float(4.0 - 0.9 * pitch_away[i] + rng.normal(0, 0.30)),
                "home_starter_announced": 1.0 if announced_home[i] else 0.0,
                "away_starter_announced": 1.0 if announced_away[i] else 0.0,
                "home_rest_days": float(home_rest[i]),
                "away_rest_days": float(away_rest[i]),
                "league_is_mlb": 1.0 if is_mlb[i] else 0.0,
                "line_movement": float(rng.normal(0, 0.4)),
                "n_books_reporting": float(rng.integers(3, 9)),
                "line_consensus_std": float(abs(rng.normal(0.15, 0.10))),
            }
        )
        home_fip = away_fip = None
        if announced_home[i]:
            home_fip = float(BASEBALL_FIP_BASE - 0.9 * starter_home[i] + rng.normal(0, 0.04))
            shared["home_starter_fip"] = home_fip
            shared["home_starter_era"] = float(home_fip + rng.normal(0, 0.30))
            shared["home_starter_kbb"] = float(np.clip(0.145 + 0.05 * starter_home[i] + rng.normal(0, 0.01), 0.0, 0.4))
        if announced_away[i]:
            away_fip = float(BASEBALL_FIP_BASE - 0.9 * starter_away[i] + rng.normal(0, 0.04))
            shared["away_starter_fip"] = away_fip
            shared["away_starter_era"] = float(away_fip + rng.normal(0, 0.30))
            shared["away_starter_kbb"] = float(np.clip(0.145 + 0.05 * starter_away[i] + rng.normal(0, 0.01), 0.0, 0.4))
        if home_fip is not None and away_fip is not None:
            shared["starter_fip_diff"] = home_fip - away_fip

        market_rows = (
            ("market_is_moneyline", float(sim_home[i]), 1.0 if margin[i] > 0 else 0.0),
            ("market_is_spread", float(sim_cover[i]), 1.0 if margin[i] + run_lines[i] > 0 else 0.0),
            ("market_is_total", float(sim_over[i]), 1.0 if total_runs[i] > total_lines[i] else 0.0),
        )
        for market_key, sim_prob, outcome in market_rows:
            row = dict(shared)
            row.update(
                {
                    "sim_probability": sim_prob,
                    "market_is_spread": 0.0,
                    "market_is_total": 0.0,
                    "market_is_moneyline": 0.0,
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


# Football generative model (Normal team scores; NFL + NCAA_FB pooled per
# ADR-026): the dominant hidden effect is measurement sharpness -- the
# "simulation" prices games from noisy scoring-rate estimates, while the
# EPA block (NFL) and SP+ rating (NCAA_FB) measure the same latent
# strengths much more precisely, so the EPA/SP+ diffs are the model's
# biggest genuine correction signals. Rest effects (bye boost, short-week
# penalty) are small and hidden from the simulation entirely.
# Tie handling (ADR-027): a tied NFL final is a moneyline PUSH, not a third
# outcome -- real graded training data would carry no decided outcome for
# those rows, so the generator mirrors that by generating two-way outcomes
# and VOIDING (excluding) the moneyline row of the rare still-tied games
# (~0.5% of NFL games: regulation ties that survive overtime). Spread and
# total rows always use half-point lines and cannot push. College overtime
# always produces a winner, so NCAA_FB rows never void.
FOOTBALL_NFL_BASE_POINTS = 22.0  # NFL average points per team (totals ~44)
NCAA_FB_BASE_POINTS = 29.0  # college scores hotter
FOOTBALL_HOME_ADVANTAGE = 2.0  # points, split across the two sides
FOOTBALL_OFF_STD = 4.0  # latent offense spread (points per game)
FOOTBALL_DEF_STD = 3.5  # latent defense spread (points conceded per game)
FOOTBALL_TEAM_SCORE_STD = 9.5  # single-game per-team score noise (margin std ~13.4)
FOOTBALL_RATE_NOISE = 5.0  # noise on the scoring-rate estimates the sim prices from
# The drive-based sim prices games from noisy team scoring-rate estimates,
# while EPA (NFL) and SP+ (NCAA_FB) measure the same latent strengths far
# more sharply -- so the sim is a genuinely weak baseline the model beats
# by reading the sharp blocks. That gap (not the tiny rest effects) is the
# football model's dominant edge, so RATE_NOISE is deliberately large.
FOOTBALL_EPA_NOISE = 0.006  # EPA measures the same latents far more sharply (NFL)
FOOTBALL_SP_NOISE = 0.6  # SP+ measures overall latent strength sharply (NCAA_FB)
FOOTBALL_PLAYS_PER_GAME = 62.0  # converts point latents to per-play EPA scale
FOOTBALL_DRIVES_PER_GAME = 11.0
FOOTBALL_BYE_EFFECT = 1.0  # points added to the side coming off a bye
FOOTBALL_SHORT_WEEK_EFFECT = -0.7  # points for a short (Thursday) week
FOOTBALL_SIM_DAMPENING = 0.55  # the football "simulation" underreacts too
FOOTBALL_SIM_NOISE = 0.12  # logit-scale noise on simulated probabilities
FOOTBALL_NFL_SHARE = 0.6  # NFL fraction of pooled rows
FOOTBALL_TIE_RATE_AFTER_OT = 0.15  # NFL: share of regulation ties surviving OT (~0.5% of games)
_FOOTBALL_OT_SLOPE = 0.15  # logit slope of P(home wins OT) on true mean diff
_PHI_SCALE = 1.702  # logistic approximation to the Normal CDF


def _phi(z: np.ndarray) -> np.ndarray:
    """Normal CDF via the standard logistic approximation (no scipy dep)."""
    return _sigmoid(_PHI_SCALE * z)


def generate_football_synthetic_dataset(n_games: int = 3_000, seed: int = 17, n_seasons: int = 4) -> TrainingSet:
    """Per-selection binary rows for the pooled FOOTBALL bootstrap (ADR-026).

    Each game emits up to three rows: a two-way HOME moneyline (voided when
    an NFL game ends tied -- a PUSH leaves no gradeable outcome, see the
    module constants), a half-point spread, and a half-point totals (OVER)
    row. Truth is Normal team scores from latent offense/defense strengths
    plus small hidden rest effects; the "simulation" prices dampened
    noisy-rate estimates, so the sharper EPA (NFL) / SP+ (NCAA_FB) blocks
    and the bye/short-week flags are genuine signals for the model.
    """
    rng = np.random.default_rng(seed)

    is_nfl = rng.random(n_games) < FOOTBALL_NFL_SHARE
    base_points = np.where(is_nfl, FOOTBALL_NFL_BASE_POINTS, NCAA_FB_BASE_POINTS)

    off_home = rng.normal(0.0, FOOTBALL_OFF_STD, n_games)
    off_away = rng.normal(0.0, FOOTBALL_OFF_STD, n_games)
    def_home = rng.normal(0.0, FOOTBALL_DEF_STD, n_games)  # points conceded above base
    def_away = rng.normal(0.0, FOOTBALL_DEF_STD, n_games)

    rest_choices = np.array([3.0, 6.0, 13.0])  # short week, normal week, off a bye
    home_rest = rng.choice(rest_choices, n_games, p=[0.12, 0.82, 0.06])
    away_rest = rng.choice(rest_choices, n_games, p=[0.12, 0.82, 0.06])
    home_bye, away_bye = (home_rest > 10).astype(np.float64), (away_rest > 10).astype(np.float64)
    home_short, away_short = (home_rest < 5).astype(np.float64), (away_rest < 5).astype(np.float64)

    # What the simulation is allowed to know: noisy scoring-rate estimates
    off_home_est = off_home + rng.normal(0.0, FOOTBALL_RATE_NOISE, n_games)
    off_away_est = off_away + rng.normal(0.0, FOOTBALL_RATE_NOISE, n_games)
    def_home_est = def_home + rng.normal(0.0, FOOTBALL_RATE_NOISE, n_games)
    def_away_est = def_away + rng.normal(0.0, FOOTBALL_RATE_NOISE, n_games)
    sim_mu_home = base_points + FOOTBALL_HOME_ADVANTAGE / 2 + off_home_est + def_away_est
    sim_mu_away = base_points - FOOTBALL_HOME_ADVANTAGE / 2 + off_away_est + def_home_est

    # True means add the exact latents and the rest effects hidden from the sim
    true_mu_home = (
        base_points
        + FOOTBALL_HOME_ADVANTAGE / 2
        + off_home
        + def_away
        + FOOTBALL_BYE_EFFECT * home_bye
        + FOOTBALL_SHORT_WEEK_EFFECT * home_short
    )
    true_mu_away = (
        base_points
        - FOOTBALL_HOME_ADVANTAGE / 2
        + off_away
        + def_home
        + FOOTBALL_BYE_EFFECT * away_bye
        + FOOTBALL_SHORT_WEEK_EFFECT * away_short
    )

    margin_std = float(np.sqrt(2.0) * FOOTBALL_TEAM_SCORE_STD)
    sim_margin_mean = sim_mu_home - sim_mu_away
    sim_total_mean = sim_mu_home + sim_mu_away
    base_home = 1.0 - _phi(-sim_margin_mean / margin_std)

    market_margin = sim_margin_mean + rng.normal(0.0, 1.0, n_games)
    spread_lines = -(np.floor(market_margin) + 0.5)  # half-point lines: no pushes
    base_cover = 1.0 - _phi((-spread_lines - sim_margin_mean) / margin_std)
    total_lines = np.floor(sim_total_mean + rng.normal(0.0, 1.5, n_games)) + 0.5
    base_over = 1.0 - _phi((total_lines - sim_total_mean) / margin_std)

    sim_home = _dampen(base_home, rng, FOOTBALL_SIM_DAMPENING, FOOTBALL_SIM_NOISE)
    sim_cover = _dampen(base_cover, rng, FOOTBALL_SIM_DAMPENING, FOOTBALL_SIM_NOISE)
    sim_over = _dampen(base_over, rng, FOOTBALL_SIM_DAMPENING, FOOTBALL_SIM_NOISE)

    # Rare-tie mass the drive-based plugin would report (NFL only): the
    # regulation zero-margin band, discounted by overtime resolution
    tie_band = _phi((0.5 - sim_margin_mean) / margin_std) - _phi((-0.5 - sim_margin_mean) / margin_std)
    sim_draw = np.where(is_nfl, tie_band * FOOTBALL_TIE_RATE_AFTER_OT, 0.0)

    # Actual finals sampled from the true means
    home_score = np.maximum(np.round(rng.normal(true_mu_home, FOOTBALL_TEAM_SCORE_STD)), 0.0)
    away_score = np.maximum(np.round(rng.normal(true_mu_away, FOOTBALL_TEAM_SCORE_STD)), 0.0)
    regulation_tie = home_score == away_score
    ot_home_win = rng.random(n_games) < _sigmoid(_FOOTBALL_OT_SLOPE * (true_mu_home - true_mu_away))
    stays_tied = regulation_tie & is_nfl & (rng.random(n_games) < FOOTBALL_TIE_RATE_AFTER_OT)
    resolve = regulation_tie & ~stays_tied
    home_score = home_score + np.where(resolve & ot_home_win, 3.0, 0.0)
    away_score = away_score + np.where(resolve & ~ot_home_win, 3.0, 0.0)
    margin = home_score - away_score
    total_points = home_score + away_score

    game_seasons = (np.arange(n_games) * n_seasons // n_games).astype(np.int64)

    features: list[FeatureMap] = []
    sim_probs: list[float] = []
    outcomes: list[float] = []
    seasons: list[int] = []

    for i in range(n_games):
        shared: FeatureMap = dict.fromkeys(FOOTBALL_FEATURES, None)
        ppg_home = float(base_points[i] + off_home_est[i] + rng.normal(0, 0.5))
        ppg_away = float(base_points[i] + off_away_est[i] + rng.normal(0, 0.5))
        pa_home = float(base_points[i] + def_home_est[i] + rng.normal(0, 0.5))
        pa_away = float(base_points[i] + def_away_est[i] + rng.normal(0, 0.5))
        shared.update(
            {
                "sim_margin_mean": float(sim_margin_mean[i] + rng.normal(0, 0.2)),
                "sim_total_mean": float(sim_total_mean[i] + rng.normal(0, 0.3)),
                "sim_draw_probability": float(sim_draw[i]),
                "sim_converged": 1.0,
                "home_points_per_game": ppg_home,
                "home_points_allowed_per_game": pa_home,
                "home_points_per_drive_off": float(ppg_home / FOOTBALL_DRIVES_PER_GAME + rng.normal(0, 0.05)),
                "home_points_per_drive_def": float(pa_home / FOOTBALL_DRIVES_PER_GAME + rng.normal(0, 0.05)),
                "home_turnover_margin_per_game": float(0.04 * (off_home[i] - def_home[i]) + rng.normal(0, 0.45)),
                "away_points_per_game": ppg_away,
                "away_points_allowed_per_game": pa_away,
                "away_points_per_drive_off": float(ppg_away / FOOTBALL_DRIVES_PER_GAME + rng.normal(0, 0.05)),
                "away_points_per_drive_def": float(pa_away / FOOTBALL_DRIVES_PER_GAME + rng.normal(0, 0.05)),
                "away_turnover_margin_per_game": float(0.04 * (off_away[i] - def_away[i]) + rng.normal(0, 0.45)),
                "home_rest_days": float(home_rest[i]),
                "away_rest_days": float(away_rest[i]),
                "home_bye_week": float(home_bye[i]),
                "away_bye_week": float(away_bye[i]),
                "league_is_nfl": 1.0 if is_nfl[i] else 0.0,
                "line_movement": float(rng.normal(0, 0.8)),
                "n_books_reporting": float(rng.integers(3, 9)),
                "line_consensus_std": float(abs(rng.normal(0.3, 0.2))),
            }
        )
        if is_nfl[i]:  # EPA measures the latents sharply; college has no EPA source
            shared["home_epa_per_play_off"] = float(
                off_home[i] / FOOTBALL_PLAYS_PER_GAME + rng.normal(0, FOOTBALL_EPA_NOISE)
            )
            shared["home_epa_per_play_def"] = float(
                def_home[i] / FOOTBALL_PLAYS_PER_GAME + rng.normal(0, FOOTBALL_EPA_NOISE)
            )
            shared["away_epa_per_play_off"] = float(
                off_away[i] / FOOTBALL_PLAYS_PER_GAME + rng.normal(0, FOOTBALL_EPA_NOISE)
            )
            shared["away_epa_per_play_def"] = float(
                def_away[i] / FOOTBALL_PLAYS_PER_GAME + rng.normal(0, FOOTBALL_EPA_NOISE)
            )
        else:  # SP+ measures overall strength sharply; the NFL has no SP+ source
            shared["home_sp_plus_rating"] = float(off_home[i] - def_home[i] + rng.normal(0, FOOTBALL_SP_NOISE))
            shared["away_sp_plus_rating"] = float(off_away[i] - def_away[i] + rng.normal(0, FOOTBALL_SP_NOISE))

        market_rows = [
            ("market_is_spread", float(sim_cover[i]), 1.0 if margin[i] + spread_lines[i] > 0 else 0.0),
            ("market_is_total", float(sim_over[i]), 1.0 if total_points[i] > total_lines[i] else 0.0),
        ]
        if not stays_tied[i]:  # a tied NFL final is a moneyline PUSH: no row
            market_rows.insert(0, ("market_is_moneyline", float(sim_home[i]), 1.0 if margin[i] > 0 else 0.0))
        for market_key, sim_prob, outcome in market_rows:
            row = dict(shared)
            row.update(
                {
                    "sim_probability": sim_prob,
                    "market_is_spread": 0.0,
                    "market_is_total": 0.0,
                    "market_is_moneyline": 0.0,
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


# Hockey generative model (independent Poisson goals with OT/SO resolution;
# single-league NHL): the hidden effects are goaltending (team save pct
# scales the opponent's conversion of the base rate), special teams (power
# play vs the opponent's penalty kill), and back-to-back fatigue -- the
# "simulation" sees only the attack/defense base rates, dampened per the
# established pattern. NHL finals have no draws: regulation-tie Poisson
# mass is reallocated to strength-weighted one-goal OT/SO wins (a shootout
# counts as one goal, per NHL convention), the exact machinery baseball
# uses for extra innings.
HOCKEY_BASE_GOALS = 3.0  # NHL average goals per team per game (totals ~6)
HOCKEY_HOME_ADVANTAGE = 0.05  # log-scale boost to home expected goals
HOCKEY_ATTACK_STD = 0.12  # latent attack spread (log scale)
HOCKEY_DEFENSE_STD = 0.10  # latent conceding spread (log scale)
# Goaltending and special teams are the hidden effects the base-rate sim
# ignores entirely, so they are the hockey model's dominant edge -- their
# spreads are sized to make that edge robust rather than coin-flip-thin
# (still within realistic season-to-date NHL ranges after the clips below).
HOCKEY_SAVE_STD = 0.016  # team save-pct spread around .900 -- goaltending
HOCKEY_BASE_SAVE_PCT = 0.900
HOCKEY_PP_OPPS_PER_GAME = 3.3  # power-play opportunities converting ST edges to goals
HOCKEY_PP_BASE, HOCKEY_PP_STD = 0.20, 0.040
HOCKEY_PK_BASE, HOCKEY_PK_STD = 0.79, 0.045
HOCKEY_B2B_RATE = 0.14  # NHL schedule density: second night of a back-to-back
HOCKEY_B2B_EFFECT = -0.18  # goals for the tired side
HOCKEY_SHOTS_PER_GAME = 30.0
HOCKEY_SIM_DAMPENING = 0.45  # the hockey "simulation" underreacts too
HOCKEY_SIM_NOISE = 0.14  # logit-scale noise on simulated probabilities
_HOCKEY_OT_SLOPE = 0.5  # logit slope of P(home wins OT/SO) on rate diff


def generate_hockey_synthetic_dataset(n_games: int = 3_000, seed: int = 19, n_seasons: int = 4) -> TrainingSet:
    """Per-selection binary rows for the NHL HOCKEY bootstrap (ADR-026).

    Each game emits three rows: a two-way HOME moneyline (OT/SO always
    produces a winner), a puck-line spread (the rate favorite laying -1.5,
    the dog getting +1.5), and a totals (OVER) row. Truth is independent
    Poisson goals from latent attack/conceding strengths plus goaltending,
    special teams, and back-to-back effects hidden from the "simulation".
    """
    rng = np.random.default_rng(seed)

    attack_home = rng.normal(0.0, HOCKEY_ATTACK_STD, n_games)
    attack_away = rng.normal(0.0, HOCKEY_ATTACK_STD, n_games)
    concede_home = rng.normal(0.0, HOCKEY_DEFENSE_STD, n_games)
    concede_away = rng.normal(0.0, HOCKEY_DEFENSE_STD, n_games)
    save_home = rng.normal(0.0, HOCKEY_SAVE_STD, n_games)  # save-pct deviation from .900
    save_away = rng.normal(0.0, HOCKEY_SAVE_STD, n_games)
    pp_home = rng.normal(HOCKEY_PP_BASE, HOCKEY_PP_STD, n_games)
    pp_away = rng.normal(HOCKEY_PP_BASE, HOCKEY_PP_STD, n_games)
    pk_home = rng.normal(HOCKEY_PK_BASE, HOCKEY_PK_STD, n_games)
    pk_away = rng.normal(HOCKEY_PK_BASE, HOCKEY_PK_STD, n_games)
    b2b_home = rng.random(n_games) < HOCKEY_B2B_RATE
    b2b_away = rng.random(n_games) < HOCKEY_B2B_RATE

    # Base rates: what the simulation is allowed to know about
    lambda_home = HOCKEY_BASE_GOALS * np.exp(attack_home + concede_away + HOCKEY_HOME_ADVANTAGE)
    lambda_away = HOCKEY_BASE_GOALS * np.exp(attack_away + concede_home)

    # True rates add goaltending, special teams, and fatigue
    st_home = HOCKEY_PP_OPPS_PER_GAME * ((pp_home - HOCKEY_PP_BASE) + (HOCKEY_PK_BASE - pk_away))
    st_away = HOCKEY_PP_OPPS_PER_GAME * ((pp_away - HOCKEY_PP_BASE) + (HOCKEY_PK_BASE - pk_home))
    goalie_home = (1.0 - HOCKEY_BASE_SAVE_PCT - save_away) / (1.0 - HOCKEY_BASE_SAVE_PCT)  # opp goalie vs home shots
    goalie_away = (1.0 - HOCKEY_BASE_SAVE_PCT - save_home) / (1.0 - HOCKEY_BASE_SAVE_PCT)
    true_home = np.clip(lambda_home * goalie_home + st_home + HOCKEY_B2B_EFFECT * b2b_home, 0.5, None)
    true_away = np.clip(lambda_away * goalie_away + st_away + HOCKEY_B2B_EFFECT * b2b_away, 0.5, None)

    # Simulation-visible outcome distribution from the base-rate Poisson grid
    joint = _poisson_pmf(lambda_home)[:, :, None] * _poisson_pmf(lambda_away)[:, None, :]
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    margin_pmf = np.stack(
        [np.trace(joint, offset=-m, axis1=1, axis2=2) for m in range(-_MAX_GOALS, _MAX_GOALS + 1)], axis=1
    )
    # NHL finals never tie: regulation-tie mass becomes strength-weighted
    # one-goal OT/SO wins (shootout = one goal), the baseball extras pattern.
    p_ot_home = _sigmoid(_HOCKEY_OT_SLOPE * (lambda_home - lambda_away))
    tie_mass = margin_pmf[:, _MAX_GOALS].copy()
    margin_pmf[:, _MAX_GOALS] = 0.0
    margin_pmf[:, _MAX_GOALS + 1] += tie_mass * p_ot_home
    margin_pmf[:, _MAX_GOALS - 1] += tie_mass * (1.0 - p_ot_home)

    base_home = margin_pmf[:, _MAX_GOALS + 1 :].sum(axis=1)
    puck_lines = np.where(lambda_home >= lambda_away, -1.5, 1.5)  # favorite lays the puck line
    cover_minus = margin_pmf[:, _MAX_GOALS + 2 :].sum(axis=1)  # -1.5: win by 2+
    cover_plus = margin_pmf[:, _MAX_GOALS - 1 :].sum(axis=1)  # +1.5: lose by <=1 or win
    base_cover = np.where(puck_lines < 0, cover_minus, cover_plus)

    total_pmf = np.stack(
        [np.trace(joint[:, ::-1, :], offset=t - _MAX_GOALS, axis1=1, axis2=2) for t in range(2 * _MAX_GOALS + 1)],
        axis=1,
    )
    exp_total = lambda_home + lambda_away
    total_lines = np.floor(exp_total + rng.uniform(-0.8, 0.8, n_games)) + 0.5
    over_idx = np.ceil(total_lines).astype(np.int64)
    total_cdf = total_pmf.cumsum(axis=1)
    base_over = 1.0 - np.take_along_axis(total_cdf, (over_idx - 1)[:, None], axis=1)[:, 0]

    sim_home = _dampen(base_home, rng, HOCKEY_SIM_DAMPENING, HOCKEY_SIM_NOISE)
    sim_cover = _dampen(base_cover, rng, HOCKEY_SIM_DAMPENING, HOCKEY_SIM_NOISE)
    sim_over = _dampen(base_over, rng, HOCKEY_SIM_DAMPENING, HOCKEY_SIM_NOISE)

    # Actual finals sampled from the true rates, OT/SO resolving ties
    home_goals = rng.poisson(true_home).astype(np.int64)
    away_goals = rng.poisson(true_away).astype(np.int64)
    ties = home_goals == away_goals
    ot_home_win = rng.random(n_games) < _sigmoid(_HOCKEY_OT_SLOPE * (true_home - true_away))
    home_goals = home_goals + (ties & ot_home_win)
    away_goals = away_goals + (ties & ~ot_home_win)
    margin = home_goals - away_goals
    total_goals = home_goals + away_goals

    game_seasons = (np.arange(n_games) * n_seasons // n_games).astype(np.int64)
    sim_margin_mean = lambda_home - lambda_away + rng.normal(0.0, 0.05, n_games)
    sim_total_mean = exp_total + rng.normal(0.0, 0.08, n_games)

    features: list[FeatureMap] = []
    sim_probs: list[float] = []
    outcomes: list[float] = []
    seasons: list[int] = []

    for i in range(n_games):
        shared: FeatureMap = dict.fromkeys(HOCKEY_FEATURES, None)
        shared.update(
            {
                "sim_margin_mean": float(sim_margin_mean[i]),
                "sim_total_mean": float(sim_total_mean[i]),
                "sim_converged": 1.0,
                "home_goals_for_per_game": float(HOCKEY_BASE_GOALS * np.exp(attack_home[i]) + rng.normal(0, 0.10)),
                "home_goals_against_per_game": float(HOCKEY_BASE_GOALS * np.exp(concede_home[i]) + rng.normal(0, 0.10)),
                "home_shots_for_per_game": float(HOCKEY_SHOTS_PER_GAME * np.exp(attack_home[i]) + rng.normal(0, 0.8)),
                "home_shots_against_per_game": float(
                    HOCKEY_SHOTS_PER_GAME * np.exp(concede_home[i]) + rng.normal(0, 0.8)
                ),
                "home_power_play_pct": float(np.clip(pp_home[i] + rng.normal(0, 0.004), 0.05, 0.40)),
                "home_penalty_kill_pct": float(np.clip(pk_home[i] + rng.normal(0, 0.004), 0.60, 0.95)),
                "home_team_save_pct": float(
                    np.clip(HOCKEY_BASE_SAVE_PCT + save_home[i] + rng.normal(0, 0.001), 0.85, 0.95)
                ),
                "away_goals_for_per_game": float(HOCKEY_BASE_GOALS * np.exp(attack_away[i]) + rng.normal(0, 0.10)),
                "away_goals_against_per_game": float(HOCKEY_BASE_GOALS * np.exp(concede_away[i]) + rng.normal(0, 0.10)),
                "away_shots_for_per_game": float(HOCKEY_SHOTS_PER_GAME * np.exp(attack_away[i]) + rng.normal(0, 0.8)),
                "away_shots_against_per_game": float(
                    HOCKEY_SHOTS_PER_GAME * np.exp(concede_away[i]) + rng.normal(0, 0.8)
                ),
                "away_power_play_pct": float(np.clip(pp_away[i] + rng.normal(0, 0.004), 0.05, 0.40)),
                "away_penalty_kill_pct": float(np.clip(pk_away[i] + rng.normal(0, 0.004), 0.60, 0.95)),
                "away_team_save_pct": float(
                    np.clip(HOCKEY_BASE_SAVE_PCT + save_away[i] + rng.normal(0, 0.001), 0.85, 0.95)
                ),
                "home_rest_days": 0.0 if b2b_home[i] else float(rng.integers(1, 4)),
                "away_rest_days": 0.0 if b2b_away[i] else float(rng.integers(1, 4)),
                "home_back_to_back": 1.0 if b2b_home[i] else 0.0,
                "away_back_to_back": 1.0 if b2b_away[i] else 0.0,
                "line_movement": float(rng.normal(0, 0.3)),
                "n_books_reporting": float(rng.integers(3, 9)),
                "line_consensus_std": float(abs(rng.normal(0.1, 0.08))),
            }
        )

        market_rows = (
            ("market_is_moneyline", float(sim_home[i]), 1.0 if margin[i] > 0 else 0.0),
            ("market_is_spread", float(sim_cover[i]), 1.0 if margin[i] + puck_lines[i] > 0 else 0.0),
            ("market_is_total", float(sim_over[i]), 1.0 if total_goals[i] > total_lines[i] else 0.0),
        )
        for market_key, sim_prob, outcome in market_rows:
            row = dict(shared)
            row.update(
                {
                    "sim_probability": sim_prob,
                    "market_is_spread": 0.0,
                    "market_is_total": 0.0,
                    "market_is_moneyline": 0.0,
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


# NCAA_BB generative model (clone of the NBA per-row structure at college
# scale: totals ~145 at pace ~68): the dominant honest signal is the CBBD
# adjusted efficiency margin, which measures the latent strength much more
# sharply than the noisy net-rating estimate -- mirroring reality, where
# opponent-adjusted college ratings dominate raw box-score aggregates.
# The same small rest/back-to-back effects as the NBA are embedded; there
# are no injury features (no reliable college injury source).
NCAA_BB_BASE_POINTS = 72.5  # college average points per team (totals ~145)
NCAA_BB_PACE = 68.0  # possessions per game
NCAA_BB_SIM_DAMPENING = 0.55  # the college "simulation" underreacts too
NCAA_BB_AEM_SCALE = 7.0  # adjusted efficiency margin per latent strength unit
NCAA_BB_AEM_NOISE = 0.5  # AEM measures strength sharply...
NCAA_BB_NET_NOISE = 3.0  # ...while raw net rating is much noisier


def generate_ncaa_bb_synthetic_dataset(n_rows: int = 6_000, seed: int = 23, n_seasons: int = 4) -> TrainingSet:
    """Per-row binary samples for the single-league NCAA_BB bootstrap.

    Follows the NBA generator's shape (one row per sample with a random
    market one-hot) at college scoring scale. The simulation's dampened
    response to latent strength plus the sharp adjusted-efficiency
    measurement give the model genuine correction signals; rest advantage
    and back-to-backs add the same small hidden effects as the NBA.
    """
    rng = np.random.default_rng(seed)

    strength_diff = rng.normal(0.0, 0.8, n_rows)
    sim_probs = np.clip(_sigmoid(NCAA_BB_SIM_DAMPENING * strength_diff + rng.normal(0.0, 0.18, n_rows)), 0.05, 0.95)

    rest_advantage = rng.integers(-3, 4, n_rows).astype(np.float64)
    home_b2b = (rng.random(n_rows) < 0.10).astype(np.float64)
    market_idx = rng.integers(0, len(_MARKETS), n_rows)

    contextual_adjustment = REST_EFFECT * (rest_advantage / 3.0) + BACK_TO_BACK_EFFECT * home_b2b
    true_probs = np.clip(_sigmoid(strength_diff) + contextual_adjustment, 0.02, 0.98)
    outcomes = (rng.random(n_rows) < true_probs).astype(np.float64)
    seasons = (np.arange(n_rows) * n_seasons // n_rows).astype(np.int64)

    features: list[FeatureMap] = []
    for i in range(n_rows):
        pace_home = rng.normal(NCAA_BB_PACE, 2.5)
        pace_away = rng.normal(NCAA_BB_PACE, 2.5)
        # AEM is a sharp measurement of the latent strength; net rating is noisy
        aem_home = strength_diff[i] * NCAA_BB_AEM_SCALE / 2 + rng.normal(0, NCAA_BB_AEM_NOISE)
        aem_away = -strength_diff[i] * NCAA_BB_AEM_SCALE / 2 + rng.normal(0, NCAA_BB_AEM_NOISE)
        net_home = strength_diff[i] * NCAA_BB_AEM_SCALE / 2 + rng.normal(0, NCAA_BB_NET_NOISE)
        net_away = -strength_diff[i] * NCAA_BB_AEM_SCALE / 2 + rng.normal(0, NCAA_BB_NET_NOISE)
        row: FeatureMap = dict.fromkeys(NCAA_BB_FEATURES, None)
        row.update(
            {
                "sim_probability": float(sim_probs[i]),
                "sim_margin_mean": float(strength_diff[i] * 5 + rng.normal(0, 1)),
                "sim_total_mean": float(rng.normal(2 * NCAA_BB_BASE_POINTS, 7)),
                "sim_converged": 1.0,
                "market_is_spread": 1.0 if market_idx[i] == 0 else 0.0,
                "market_is_total": 1.0 if market_idx[i] == 1 else 0.0,
                "market_is_moneyline": 1.0 if market_idx[i] == 2 else 0.0,
                "home_offensive_rating": float(105 + net_home / 2 + rng.normal(0, 1)),
                "home_defensive_rating": float(105 - net_home / 2 + rng.normal(0, 1)),
                "home_pace": float(pace_home),
                "home_net_rating": float(net_home),
                "home_adjusted_efficiency_margin": float(aem_home),
                "away_offensive_rating": float(105 + net_away / 2 + rng.normal(0, 1)),
                "away_defensive_rating": float(105 - net_away / 2 + rng.normal(0, 1)),
                "away_pace": float(pace_away),
                "away_net_rating": float(net_away),
                "away_adjusted_efficiency_margin": float(aem_away),
                "pace_differential": float(pace_home - pace_away),
                "net_rating_diff": float(net_home - net_away),
                "home_away_split_diff": float(rng.normal(3.0, 2.5)),
                "home_last5_ppg": float(NCAA_BB_BASE_POINTS + net_home / 2 + rng.normal(0, 3)),
                "home_last5_ppg_allowed": float(NCAA_BB_BASE_POINTS - net_home / 2 + rng.normal(0, 3)),
                "home_three_pct_last5": float(np.clip(rng.normal(0.34, 0.03), 0.22, 0.46)),
                "home_net_rating_last10": float(net_home + rng.normal(0, 2)),
                "away_last5_ppg": float(NCAA_BB_BASE_POINTS + net_away / 2 + rng.normal(0, 3)),
                "away_last5_ppg_allowed": float(NCAA_BB_BASE_POINTS - net_away / 2 + rng.normal(0, 3)),
                "away_three_pct_last5": float(np.clip(rng.normal(0.34, 0.03), 0.22, 0.46)),
                "away_net_rating_last10": float(net_away + rng.normal(0, 2)),
                "home_rest_days": float(max(rest_advantage[i], 0) + 1),
                "away_rest_days": float(max(-rest_advantage[i], 0) + 1),
                "rest_advantage": float(rest_advantage[i]),
                "home_back_to_back": float(home_b2b[i]),
                "away_back_to_back": 1.0 if rng.random() < 0.10 else 0.0,
                "line_movement": float(rng.normal(0, 1.0)),
                "n_books_reporting": float(rng.integers(3, 9)),
                "line_consensus_std": float(abs(rng.normal(0.3, 0.2))),
            }
        )
        features.append(row)

    return TrainingSet(features=features, sim_probs=sim_probs, outcomes=outcomes, seasons=seasons)


# --- Player-prop synthetic generators (Phase 7 Wave 3) -----------------------
#
# One unified prop bootstrap per sport following the established pattern:
# a latent per-player rate the "simulation" only sees a dampened, noisy,
# per-stat-biased view of, actual stat outcomes sampled from the true rate
# (Gamma-Poisson overdispersed for counts, Normal for yardage), and rows
# built as the OVER (or YES) perspective against varied half-step lines --
# the only perspective the predictor ever feeds the model (UNDER/NO are
# complements). The learnable correction signals are the dampening (via
# sim_prop_probability), the per-stat systematic bias (via the stat
# one-hots), and tail miscalibration of the Poisson-shaped sim view against
# the overdispersed truth (via prop_line vs sim_stat_mean/std geometry).
# A hidden matchup effect the features cannot recover keeps conformal
# widths honest.
PROP_SIM_DAMPENING = 0.65  # the prop "simulation" underreacts too
PROP_SIM_NOISE = 0.10  # logit-scale noise on simulated prop probabilities
PROP_MATCHUP_STD = 0.15  # hidden per-game matchup effect (log/logit scale)
PROP_DISPERSION = 8.0  # Gamma shape for count overdispersion (NegBinomial)
_PROP_MAX_COUNT = 40  # Poisson grid truncation for count-stat over grids


@dataclass(frozen=True)
class _PropStatSpec:
    """One prop stat's generative parameters.

    kind: "count" (Poisson-shaped, over/under a half-step line), "normal"
    (yardage-style, over/under), or "yes_no" (Bernoulli, no line).
    mean is the league-average rate (or yes-probability); skill_std the
    latent player spread (log scale for count, additive for normal, logit
    for yes_no); game_std the per-game noise (normal kind only); sim_bias a
    per-stat systematic logit bias in the sim's view the one-hots let the
    model learn away.
    """

    stat: str
    kind: str
    mean: float
    skill_std: float
    game_std: float = 0.0
    sim_bias: float = 0.0


_SOCCER_PROP_SPECS: tuple[_PropStatSpec, ...] = (
    _PropStatSpec("player_goal_scorer_anytime", "yes_no", 0.25, 0.80, sim_bias=0.15),
    _PropStatSpec("player_shots", "count", 2.2, 0.35, sim_bias=-0.10),
    _PropStatSpec("player_shots_on_target", "count", 0.9, 0.35),
)
_BASKETBALL_PROP_SPECS: tuple[_PropStatSpec, ...] = (
    _PropStatSpec("player_points", "count", 15.0, 0.45, sim_bias=0.08),
    _PropStatSpec("player_rebounds", "count", 6.0, 0.50),
    _PropStatSpec("player_assists", "count", 4.5, 0.60, sim_bias=-0.08),
    _PropStatSpec("player_threes", "count", 1.8, 0.50, sim_bias=0.12),
    _PropStatSpec("player_points_rebounds_assists", "count", 25.0, 0.40),
)
_BASEBALL_PROP_SPECS: tuple[_PropStatSpec, ...] = (
    _PropStatSpec("batter_hits", "count", 1.0, 0.30),
    _PropStatSpec("batter_total_bases", "count", 1.5, 0.35, sim_bias=0.08),
    _PropStatSpec("batter_home_runs", "count", 0.14, 0.50, sim_bias=0.12),
    _PropStatSpec("pitcher_strikeouts", "count", 5.5, 0.35, sim_bias=-0.08),
)
_FOOTBALL_PROP_SPECS: tuple[_PropStatSpec, ...] = (
    _PropStatSpec("player_pass_yds", "normal", 230.0, 35.0, game_std=55.0, sim_bias=0.08),
    _PropStatSpec("player_rush_yds", "normal", 55.0, 25.0, game_std=28.0),
    _PropStatSpec("player_reception_yds", "normal", 45.0, 20.0, game_std=25.0, sim_bias=-0.08),
    _PropStatSpec("player_receptions", "count", 3.8, 0.35),
    _PropStatSpec("player_anytime_td", "yes_no", 0.35, 0.70, sim_bias=0.12),
)


def _prop_sim_view(true_prob: float, bias: float, rng: np.random.Generator) -> float:
    """The prop "simulation" view: dampened logit response, per-stat bias, noise."""
    logit = float(np.log(true_prob / (1.0 - true_prob)))
    noisy = PROP_SIM_DAMPENING * logit + bias + float(rng.normal(0.0, PROP_SIM_NOISE))
    return float(np.clip(1.0 / (1.0 + np.exp(-noisy)), 0.02, 0.98))


def _poisson_over_probability(rate: float, line: float) -> float:
    """P(Poisson(rate) > line) for a half-step line."""
    pmf = _poisson_pmf(np.array([rate]), _PROP_MAX_COUNT)[0]
    return float(pmf[int(np.floor(line)) + 1 :].sum())


def _generate_prop_dataset(
    sport: str,
    specs: tuple[_PropStatSpec, ...],
    n_rows: int,
    seed: int,
    n_seasons: int,
) -> TrainingSet:
    """Shared prop-row machinery; per-sport wrappers own the spec tables."""
    rng = np.random.default_rng(seed)
    prop_features = get_prop_features(sport)
    stat_names = [spec.stat for spec in specs]

    features: list[FeatureMap] = []
    sim_probs: list[float] = []
    outcomes: list[float] = []
    seasons = (np.arange(n_rows) * n_seasons // n_rows).astype(np.int64)

    for _ in range(n_rows):
        spec = specs[int(rng.integers(0, len(specs)))]
        matchup = float(rng.normal(0.0, PROP_MATCHUP_STD))  # hidden from the sim AND the features

        if spec.kind == "yes_no":
            skill = float(rng.normal(0.0, spec.skill_std))
            base_logit = float(np.log(spec.mean / (1.0 - spec.mean)))
            known_prob = float(1.0 / (1.0 + np.exp(-(base_logit + skill))))
            true_prob = float(1.0 / (1.0 + np.exp(-(base_logit + skill + matchup))))
            sim_prob = _prop_sim_view(known_prob, spec.sim_bias, rng)
            outcome = 1.0 if rng.random() < true_prob else 0.0
            line = 0.0
            sim_mean = float(np.clip(known_prob + rng.normal(0.0, 0.01), 0.01, 0.99))
            sim_std = float(np.sqrt(sim_mean * (1.0 - sim_mean)))
            is_yes = True
        elif spec.kind == "count":
            rate = spec.mean * float(np.exp(rng.normal(0.0, spec.skill_std)))
            true_rate = rate * float(np.exp(matchup))
            line = max(0.5, float(np.floor(rate + rng.uniform(-1.0, 1.0) * max(0.6, 0.5 * np.sqrt(rate)))) + 0.5)
            base_over = float(np.clip(_poisson_over_probability(rate, line), 0.02, 0.98))
            sim_prob = _prop_sim_view(base_over, spec.sim_bias, rng)
            # Overdispersed actual (Gamma-Poisson = NegBinomial): real player
            # stats have heavier tails than the sim's Poisson view assumes.
            dispersed_rate = true_rate * float(rng.gamma(PROP_DISPERSION, 1.0 / PROP_DISPERSION))
            outcome = 1.0 if float(rng.poisson(dispersed_rate)) > line else 0.0
            sim_mean = float(rate * (1.0 + rng.normal(0.0, 0.02)))
            sim_std = float(np.sqrt(rate))
            is_yes = False
        else:  # "normal" (yardage)
            rate = spec.mean + float(rng.normal(0.0, spec.skill_std))
            true_mean = rate + matchup * spec.mean  # matchup scales with the stat's magnitude
            line = float(np.floor(rate + rng.uniform(-0.6, 0.6) * spec.game_std)) + 0.5
            base_over = float(np.clip(1.0 - _phi(np.array([(line - rate) / spec.game_std]))[0], 0.02, 0.98))
            sim_prob = _prop_sim_view(base_over, spec.sim_bias, rng)
            actual = max(0.0, float(rng.normal(true_mean, spec.game_std)))
            outcome = 1.0 if actual > line else 0.0
            sim_mean = float(rate + rng.normal(0.0, 0.02 * spec.mean))
            sim_std = float(spec.game_std * (1.0 + rng.normal(0.0, 0.02)))
            is_yes = False

        row: FeatureMap = dict.fromkeys(prop_features, None)
        row.update(
            {
                "sim_prop_probability": sim_prob,
                "prop_line": line,
                "sim_stat_mean": sim_mean,
                "sim_stat_std": sim_std,
                "side_is_over": 0.0 if is_yes else 1.0,
                "side_is_yes": 1.0 if is_yes else 0.0,
            }
        )
        for name in stat_names:
            row[f"prop_is_{name}"] = 1.0 if name == spec.stat else 0.0
        features.append(row)
        sim_probs.append(sim_prob)
        outcomes.append(outcome)

    return TrainingSet(
        features=features,
        sim_probs=np.array(sim_probs, dtype=np.float64),
        outcomes=np.array(outcomes, dtype=np.float64),
        seasons=seasons,
    )


def generate_soccer_prop_dataset(n_rows: int = 6_000, seed: int = 29, n_seasons: int = 4) -> TrainingSet:
    """Soccer prop bootstrap rows (anytime scorer yes/no, shots, shots on target)."""
    assert [spec.stat for spec in _SOCCER_PROP_SPECS] == list(SOCCER_PROP_STATS)
    return _generate_prop_dataset("SOCCER", _SOCCER_PROP_SPECS, n_rows, seed, n_seasons)


def generate_basketball_prop_dataset(n_rows: int = 6_000, seed: int = 31, n_seasons: int = 4) -> TrainingSet:
    """Basketball prop bootstrap rows (points/rebounds/assists/threes/PRA)."""
    assert [spec.stat for spec in _BASKETBALL_PROP_SPECS] == list(BASKETBALL_PROP_STATS)
    return _generate_prop_dataset("BASKETBALL", _BASKETBALL_PROP_SPECS, n_rows, seed, n_seasons)


def generate_baseball_prop_dataset(n_rows: int = 6_000, seed: int = 37, n_seasons: int = 4) -> TrainingSet:
    """Baseball prop bootstrap rows (dormant this wave; bootstrap-only-synthetic)."""
    assert [spec.stat for spec in _BASEBALL_PROP_SPECS] == list(BASEBALL_PROP_STATS)
    return _generate_prop_dataset("BASEBALL", _BASEBALL_PROP_SPECS, n_rows, seed, n_seasons)


def generate_football_prop_dataset(n_rows: int = 6_000, seed: int = 41, n_seasons: int = 4) -> TrainingSet:
    """Football prop bootstrap rows (dormant this wave; bootstrap-only-synthetic)."""
    assert [spec.stat for spec in _FOOTBALL_PROP_SPECS] == list(FOOTBALL_PROP_STATS)
    return _generate_prop_dataset("FOOTBALL", _FOOTBALL_PROP_SPECS, n_rows, seed, n_seasons)


# Sport -> player-prop synthetic bootstrap generator (Phase 7 Wave 3).
# SOCCER and BASKETBALL are live in v1; BASEBALL and FOOTBALL are
# registered but dormant (no live prop line coverage yet).
PROP_SYNTHETIC_GENERATORS: dict[str, Callable[[], TrainingSet]] = {
    "SOCCER": generate_soccer_prop_dataset,
    "BASKETBALL": generate_basketball_prop_dataset,
    "BASEBALL": generate_baseball_prop_dataset,
    "FOOTBALL": generate_football_prop_dataset,
}


def get_prop_synthetic_generator(sport: str) -> Callable[[], TrainingSet]:
    """Return the player-prop synthetic dataset generator for a sport."""
    try:
        return PROP_SYNTHETIC_GENERATORS[sport]
    except KeyError:
        raise ValueError(f"no player-prop synthetic generator registered for {sport}; added in its prop wave") from None


# Model key -> synthetic bootstrap generator. Each league wave registers
# its generator here (ADR-026); until then bootstrap fails loudly. Keys
# are sports for pooled models, the league name for single-league models
# (NCAA_BB; see core/leagues.py).
SYNTHETIC_GENERATORS: dict[str, Callable[[], TrainingSet]] = {
    "BASKETBALL": generate_synthetic_dataset,
    "SOCCER": generate_soccer_synthetic_dataset,
    "BASEBALL": generate_baseball_synthetic_dataset,
    "FOOTBALL": generate_football_synthetic_dataset,
    "HOCKEY": generate_hockey_synthetic_dataset,
    "NCAA_BB": generate_ncaa_bb_synthetic_dataset,
}


def get_synthetic_generator(sport: str) -> Callable[[], TrainingSet]:
    """Return the synthetic dataset generator for a sport."""
    try:
        return SYNTHETIC_GENERATORS[sport]
    except KeyError:
        raise ValueError(f"no synthetic generator registered for {sport}; added in its league wave") from None
