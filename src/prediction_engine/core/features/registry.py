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

# Football pools NFL and NCAA_FB into one FOOTBALL model with the league
# as a one-hot feature (ADR-026); NCAA_FB is the zero level. The sim block
# keeps sim_draw_probability because NFL regular-season games can end tied
# (the drive-based plugin emits a rare ~0.3-1% tie mass, ADR-018) -- but
# there is NO selection_is_draw: the moneyline stays two-way and a tied
# final grades as PUSH (ADR-027), so tie risk is a per-game shading signal
# rather than a third selection. EPA metrics come from nflverse and exist
# for the NFL only (None for college); SP+ ratings come from CFBD and
# exist for NCAA_FB only (None for the NFL) -- the league one-hot lets the
# model branch on which block is populated. Bye-week flags are derived
# from rest days (> 10 full days off in a weekly sport means the bye);
# back-to-backs cannot happen in football, so the flag is excluded.
# Documented exclusions (no honest data source in the Phase 6
# statistics-service football block): injuries (no validated football
# injury-impact proxy; the NBA proxy is minutes-based), quarterback status
# (no starter feed equivalent to baseball's probable pitchers), weather,
# travel distance, and head-to-head history.
FOOTBALL_FEATURES: tuple[str, ...] = (
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
    # season strength (FootballStats block)
    "home_points_per_game",
    "home_points_allowed_per_game",
    "home_points_per_drive_off",
    "home_points_per_drive_def",
    "home_epa_per_play_off",
    "home_epa_per_play_def",
    "home_sp_plus_rating",
    "home_turnover_margin_per_game",
    "away_points_per_game",
    "away_points_allowed_per_game",
    "away_points_per_drive_off",
    "away_points_per_drive_def",
    "away_epa_per_play_off",
    "away_epa_per_play_def",
    "away_sp_plus_rating",
    "away_turnover_margin_per_game",
    # situational (weekly schedule: bye = > 10 full days off)
    "home_rest_days",
    "away_rest_days",
    "home_bye_week",
    "away_bye_week",
    # league one-hot (ADR-026 pooled model; NCAA_FB is the baseline)
    "league_is_nfl",
    # market signal
    "line_movement",
    "n_books_reporting",
    "line_consensus_std",
)

# Hockey is a single-league NHL model for now: NCAA_HKY remains gated on
# Odds API line coverage (ADR-026), so no league one-hot until it lands
# (it would join as league_is_nhl, mirroring FOOTBALL). No draw features:
# NHL finals include overtime/shootout resolution, so the hockey plugin's
# draw_probability is a constant 0.0 and a constant column carries no
# signal -- the moneyline stays two-way (ADR-027 hockey note).
# Back-to-backs are kept (the NBA situational module transfers directly):
# the NHL schedule is dense enough that they are common and meaningful.
# Documented exclusions (no honest data source in the Phase 6
# statistics-service hockey block): starting goaltender identity (no
# confirmed-starter feed; team_save_pct carries the aggregate goaltending
# signal), injuries (no validated hockey injury-impact proxy), and
# head-to-head history.
HOCKEY_FEATURES: tuple[str, ...] = (
    # simulation-derived (per-market: sim_probability varies by market type)
    "sim_probability",
    "sim_margin_mean",
    "sim_total_mean",
    "sim_converged",
    # market type one-hot (single unified model, market as feature)
    "market_is_spread",
    "market_is_total",
    "market_is_moneyline",
    # season strength (HockeyStats block)
    "home_goals_for_per_game",
    "home_goals_against_per_game",
    "home_shots_for_per_game",
    "home_shots_against_per_game",
    "home_power_play_pct",
    "home_penalty_kill_pct",
    "home_team_save_pct",
    "away_goals_for_per_game",
    "away_goals_against_per_game",
    "away_shots_for_per_game",
    "away_shots_against_per_game",
    "away_power_play_pct",
    "away_penalty_kill_pct",
    "away_team_save_pct",
    # situational (dense schedule: back-to-backs matter, as in the NBA)
    "home_rest_days",
    "away_rest_days",
    "home_back_to_back",
    "away_back_to_back",
    # market signal
    "line_movement",
    "n_books_reporting",
    "line_consensus_std",
)

# NCAA_BB trains its own single-league model rather than pooling into the
# NBA's order-locked BASKETBALL tuple (see core/leagues.py): the college
# scoring environment differs and CBBD supplies an opponent-adjusted
# efficiency margin the NBA block lacks. The tuple clones the NBA feature
# approach with two deliberate differences: no injury features
# (null-documented -- there is no reliable college injury source, so the
# columns would always be None), and adjusted_efficiency_margin per side
# (the college analytics community's dominant team-strength signal). The
# league stays implicit -- a single-league registry needs no one-hot,
# exactly like the NBA's.
NCAA_BB_FEATURES: tuple[str, ...] = (
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
    "home_adjusted_efficiency_margin",
    "away_offensive_rating",
    "away_defensive_rating",
    "away_pace",
    "away_net_rating",
    "away_adjusted_efficiency_margin",
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
    # market signal
    "line_movement",
    "n_books_reporting",
    "line_consensus_std",
)

# --- Player-prop registries (Phase 7 Wave 3) --------------------------------
#
# One unified prop model per sport with the prop stat type as a one-hot
# feature, mirroring the game models' "market type is a feature" design.
# Stat keys are the canonical Odds API market keys, matching the
# simulation-engine player-distributions contract exactly.
#
# The tuples are deliberately minimal: the simulation's P(stat over line)
# is the baseline feature (sim_prop_probability), the line and the sim's
# distribution mean/std give the model line-vs-rate geometry, and the
# one-hots let it learn per-stat systematic sim biases. The game models'
# market-signal block (line_movement / n_books_reporting /
# line_consensus_std) is a documented exclusion: lines-service best-lines
# serve game markets only -- there is no per-player prop line feed yet, so
# the columns would always be None.
#
# side_is_over / side_is_yes: only the OVER (or YES) perspective is ever
# fed to the model -- UNDER and NO are emitted as renormalized complements,
# mirroring the two-way game-market logic -- so the flags encode the market
# shape (count-line vs yes/no), not a per-row side choice.

# Canonical prop stat keys per sport (= Odds API market keys).
SOCCER_PROP_STATS: tuple[str, ...] = (
    "player_goal_scorer_anytime",
    "player_shots",
    "player_shots_on_target",
)
BASKETBALL_PROP_STATS: tuple[str, ...] = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
)
BASEBALL_PROP_STATS: tuple[str, ...] = (
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
    "pitcher_strikeouts",
)
FOOTBALL_PROP_STATS: tuple[str, ...] = (
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
)

# Yes/no markets (no line; the sim reports yes_probability instead of an
# over grid). prop_line is 0.0 for these rows.
YES_NO_PROP_STATS: frozenset[str] = frozenset({"player_goal_scorer_anytime", "player_anytime_td"})

SOCCER_PROP_FEATURES: tuple[str, ...] = (
    # simulation-derived baseline (P(over line) or yes_probability)
    "sim_prop_probability",
    "prop_line",
    "sim_stat_mean",
    "sim_stat_std",
    # stat one-hot (single unified prop model, stat type as a feature)
    "prop_is_player_goal_scorer_anytime",
    "prop_is_player_shots",
    "prop_is_player_shots_on_target",
    # market shape (modeled side is always OVER or YES; see module note)
    "side_is_over",
    "side_is_yes",
)

BASKETBALL_PROP_FEATURES: tuple[str, ...] = (
    "sim_prop_probability",
    "prop_line",
    "sim_stat_mean",
    "sim_stat_std",
    "prop_is_player_points",
    "prop_is_player_rebounds",
    "prop_is_player_assists",
    "prop_is_player_threes",
    "prop_is_player_points_rebounds_assists",
    "side_is_over",
    "side_is_yes",
)

# BASEBALL and FOOTBALL prop registries ship registered but dormant this
# wave (bootstrap-only-synthetic; no live prop line coverage yet).
BASEBALL_PROP_FEATURES: tuple[str, ...] = (
    "sim_prop_probability",
    "prop_line",
    "sim_stat_mean",
    "sim_stat_std",
    "prop_is_batter_hits",
    "prop_is_batter_total_bases",
    "prop_is_batter_home_runs",
    "prop_is_pitcher_strikeouts",
    "side_is_over",
    "side_is_yes",
)

FOOTBALL_PROP_FEATURES: tuple[str, ...] = (
    "sim_prop_probability",
    "prop_line",
    "sim_stat_mean",
    "sim_stat_std",
    "prop_is_player_pass_yds",
    "prop_is_player_rush_yds",
    "prop_is_player_reception_yds",
    "prop_is_player_receptions",
    "prop_is_player_anytime_td",
    "side_is_over",
    "side_is_yes",
)

PROP_STATS_BY_SPORT: dict[str, tuple[str, ...]] = {
    "SOCCER": SOCCER_PROP_STATS,
    "BASKETBALL": BASKETBALL_PROP_STATS,
    "BASEBALL": BASEBALL_PROP_STATS,
    "FOOTBALL": FOOTBALL_PROP_STATS,
}

PROP_FEATURES_BY_SPORT: dict[str, tuple[str, ...]] = {
    "SOCCER": SOCCER_PROP_FEATURES,
    "BASKETBALL": BASKETBALL_PROP_FEATURES,
    "BASEBALL": BASEBALL_PROP_FEATURES,
    "FOOTBALL": FOOTBALL_PROP_FEATURES,
}


def get_prop_features(sport: str) -> tuple[str, ...]:
    """Return the ordered prop feature tuple for a sport."""
    try:
        return PROP_FEATURES_BY_SPORT[sport]
    except KeyError:
        raise ValueError(f"no player-prop feature registry for {sport}; added in its prop wave") from None


def get_prop_stats(sport: str) -> tuple[str, ...]:
    """Return the canonical prop stat keys for a sport."""
    try:
        return PROP_STATS_BY_SPORT[sport]
    except KeyError:
        raise ValueError(f"no player-prop stat registry for {sport}; added in its prop wave") from None


FeatureMap = dict[str, float | None]

# Model key -> ordered feature tuple. Keys are sports for pooled models and
# the league name for single-league models (NCAA_BB; see core/leagues.py).
# New sports register here in their league wave (ADR-026); until then
# get_features fails loudly for them.
FEATURES_BY_SPORT: dict[str, tuple[str, ...]] = {
    "BASKETBALL": NBA_FEATURES,
    "SOCCER": SOCCER_FEATURES,
    "BASEBALL": BASEBALL_FEATURES,
    "FOOTBALL": FOOTBALL_FEATURES,
    "HOCKEY": HOCKEY_FEATURES,
    "NCAA_BB": NCAA_BB_FEATURES,
}


def get_features(sport: str) -> tuple[str, ...]:
    """Return the ordered feature tuple for a sport."""
    try:
        return FEATURES_BY_SPORT[sport]
    except KeyError:
        raise ValueError(f"no feature registry for {sport}; added in its league wave") from None
