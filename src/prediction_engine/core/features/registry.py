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

FeatureMap = dict[str, float | None]

# Sport -> ordered feature tuple. New sports register here in their league
# wave (ADR-026); until then get_features fails loudly for them.
FEATURES_BY_SPORT: dict[str, tuple[str, ...]] = {"BASKETBALL": NBA_FEATURES}


def get_features(sport: str) -> tuple[str, ...]:
    """Return the ordered feature tuple for a sport."""
    try:
        return FEATURES_BY_SPORT[sport]
    except KeyError:
        raise ValueError(f"no feature registry for {sport}; added in its league wave") from None
