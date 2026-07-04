"""Minimal odds math for the /edges convenience endpoint.

Deliberate small duplication: the canonical, full edge-detection module
(de-vig, EV, Kelly, quality) lives in bookie-breaker-agent per
algorithms/edge-detection.md and the data-ownership table. ADR-009 rejects
shared packages, so this module carries only what the convenience endpoint
needs: raw implied probability and the edge in percentage points.
"""


def american_to_implied_prob(odds: int) -> float:
    """Raw implied probability including vig (matches lines-service values)."""
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def edge_percentage(predicted_probability: float, implied_probability: float) -> float:
    """Edge in percentage points: (predicted - implied) * 100."""
    return round((predicted_probability - implied_probability) * 100, 2)
