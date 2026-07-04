"""Golden-value tests for the minimal edge math (canonical module lives in the agent repo)."""

import pytest

from prediction_engine.core.edges import american_to_implied_prob, edge_percentage


class TestImpliedProb:
    def test_favorite(self) -> None:
        assert american_to_implied_prob(-150) == pytest.approx(0.600)

    def test_underdog(self) -> None:
        assert american_to_implied_prob(150) == pytest.approx(0.400)

    def test_standard_juice(self) -> None:
        # contract example: -110 -> 0.5238 (raw, vig included)
        assert american_to_implied_prob(-110) == pytest.approx(0.5238, abs=1e-4)

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be 0"):
            american_to_implied_prob(0)


class TestEdgePercentage:
    def test_contract_example(self) -> None:
        # 0.5312 predicted vs 0.5238 implied -> 0.74 percentage points
        assert edge_percentage(0.5312, 0.5238) == pytest.approx(0.74)

    def test_negative_edge(self) -> None:
        assert edge_percentage(0.50, 0.5238) == pytest.approx(-2.38)
