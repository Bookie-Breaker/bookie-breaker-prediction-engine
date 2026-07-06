"""Canonical ordered per-sport feature registries.

Each sport's model input vector follows its tuple's exact order;
feature_names stored in model_versions must match. The NBA tuple is
order-locked into the trained basketball model — never reorder it. Only
features honestly computable from the Phase 1 statistics-service and
lines-service APIs are included. Documented exclusions (no data source
yet): travel distance, altitude, timezone shift (no venue coordinates),
head-to-head history, public/sharp money split, weather (indoor sport).
"""

NBA_FEATURES: tuple[str, ...] = (
    # simulation-derived (per-market: sim_probability varies by market type)
    "sim_probability",
    "sim_margin_mean",
    "sim_total_mean",
    "sim_converged",
    # market type one-hot (single unified model, market as feature)
    "market_is_spread",
    "market_is_total",
    "market_is_moneyline",
    # season strength
    "home_offensive_rating",
    "home_defensive_rating",
    "home_pace",
    "home_net_rating",
    "away_offensive_rating",
    "away_defensive_rating",
    "away_pace",
    "away_net_rating",
    "pace_differential",
    "net_rating_diff",
    "home_away_split_diff",
    # recent form (rolling windows served by statistics-service)
    "home_last5_ppg",
    "home_last5_ppg_allowed",
    "home_three_pct_last5",
    "home_net_rating_last10",
    "away_last5_ppg",
    "away_last5_ppg_allowed",
    "away_three_pct_last5",
    "away_net_rating_last10",
    # situational
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "home_back_to_back",
    "away_back_to_back",
    # injuries (status-weighted proxy; no WAR data available)
    "home_injury_impact",
    "away_injury_impact",
    "injury_impact_diff",
    # market signal
    "line_movement",
    "n_books_reporting",
    "line_consensus_std",
)

# Soccer pools all competitions into one SOCCER model with the competition
# as a one-hot feature (ADR-026); EPL is the one-hot baseline. The sim block
# gains sim_draw_probability and moneyline rows gain selection_is_draw
# because the primary market is three-way (ADR-027). Documented exclusions
# (no honest data source in the Phase 6 statistics-service soccer block):
# injuries (no reliable soccer injury feed; always absent, unlike NBA),
# back-to-backs (soccer never schedules consecutive days — rest_days covers
# schedule density), home/away splits (FIFA_WC venues are neutral),
# possession/xG (not in the SoccerStats contract until a richer source
# lands), head-to-head history, and weather.
SOCCER_FEATURES: tuple[str, ...] = (
    # simulation-derived (per-market: sim_probability varies by market type)
    "sim_probability",
    "sim_margin_mean",
    "sim_total_mean",
    "sim_draw_probability",
    "sim_converged",
    # market type one-hot (single unified model, market as feature)
    "market_is_spread",
    "market_is_total",
    "market_is_moneyline",
    # selection (three-way moneyline rows: HOME/AWAY 0.0, DRAW 1.0)
    "selection_is_draw",
    # season strength (SoccerStats block: multiplicative vs competition avg)
    "home_attack_strength",
    "home_defense_strength",
    "away_attack_strength",
    "away_defense_strength",
    "home_goals_for_per_match",
    "home_goals_against_per_match",
    "away_goals_for_per_match",
    "away_goals_against_per_match",
    # recent form and sample size
    "home_form_points_last5",
    "away_form_points_last5",
    "home_matches_played",
    "away_matches_played",
    # situational
    "home_rest_days",
    "away_rest_days",
    "is_knockout",
    # competition one-hot (ADR-026 pooled model; EPL is the baseline)
    "competition_is_fifa_wc",
    # market signal
    "line_movement",
    "n_books_reporting",
    "line_consensus_std",
)

# Baseball pools MLB and NCAA_BSB into one BASEBALL model with the league
# as a one-hot feature (ADR-026); MLB is the dominant league, NCAA_BSB the
# zero level. No draw features: baseball games cannot tie (extra innings),
# so the moneyline stays two-way and the sim block has no
# sim_draw_probability. The probable-starter block is the sport's defining
# signal; when a starter is unannounced its stats are None (NaN to XGBoost)
# and the announced flag is 0.0, so the model can learn that unannounced
# games carry extra outcome noise. Documented exclusions from the
# BaseballStats contract block: team_obp and team_slg (wOBA is the linear
# combination of both -- keeping all three is pure collinearity),
# batting_walk_pct (also priced into wOBA), batting_strikeout_pct (weak
# marginal team-level signal once wOBA is known), team_era (FIP is the
# defense-independent skill estimate; ERA adds fielding noise on top of the
# same innings). Other exclusions (no honest data source): injuries (the
# NBA impact proxy is minutes-based and does not transfer to baseball, and
# the announced starter dominates the personnel signal -- see the builder),
# back-to-backs (baseball plays near-daily; rest_days carries the schedule
# signal), park factors, weather, platoon/handedness splits, day-night
# splits, and head-to-head history.
BASEBALL_FEATURES: tuple[str, ...] = (
    # simulation-derived (per-market: sim_probability varies by market type)
    "sim_probability",
    "sim_margin_mean",
    "sim_total_mean",
    "sim_converged",
    # market type one-hot (single unified model, market as feature)
    "market_is_spread",
    "market_is_total",
    "market_is_moneyline",
    # season strength (BaseballStats block)
    "home_runs_scored_per_game",
    "home_runs_allowed_per_game",
    "home_team_woba",
    "home_team_fip",
    "home_bullpen_era",
    "away_runs_scored_per_game",
    "away_runs_allowed_per_game",
    "away_team_woba",
    "away_team_fip",
    "away_bullpen_era",
    # probable starters (game-level; None + flag 0.0 when unannounced)
    "home_starter_fip",
    "home_starter_era",
    "home_starter_kbb",
    "home_starter_announced",
    "away_starter_fip",
    "away_starter_era",
    "away_starter_kbb",
    "away_starter_announced",
    "starter_fip_diff",
    # situational
    "home_rest_days",
    "away_rest_days",
    # league one-hot (ADR-026 pooled model; NCAA_BSB is the baseline)
    "league_is_mlb",
    # market signal
    "line_movement",
    "n_books_reporting",
    "line_consensus_std",
)

FeatureMap = dict[str, float | None]

# Sport -> ordered feature tuple. New sports register here in their league
# wave (ADR-026); until then get_features fails loudly for them.
FEATURES_BY_SPORT: dict[str, tuple[str, ...]] = {
    "BASKETBALL": NBA_FEATURES,
    "SOCCER": SOCCER_FEATURES,
    "BASEBALL": BASEBALL_FEATURES,
}


def get_features(sport: str) -> tuple[str, ...]:
    """Return the ordered feature tuple for a sport."""
    try:
        return FEATURES_BY_SPORT[sport]
    except KeyError:
        raise ValueError(f"no feature registry for {sport}; added in its league wave") from None
