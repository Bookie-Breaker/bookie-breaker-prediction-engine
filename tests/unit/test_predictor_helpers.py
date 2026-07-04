"""Predictor helper tests: grid interpolation and DSN translation."""

import pytest

from prediction_engine.core.predictor import _half_line, _interpolate
from prediction_engine.db.engine import _split_dsn


class TestInterpolate:
    grid = {"-4.5": 0.44, "-3.5": 0.48, "-2.5": 0.52}

    def test_exact_key(self) -> None:
        line, prob = _interpolate(self.grid, -3.5)
        assert line == -3.5
        assert prob == pytest.approx(0.48)

    def test_between_keys_interpolates(self) -> None:
        line, prob = _interpolate(self.grid, -3.0)
        assert line == -3.5  # nearest half-line (tie resolves to the lower)
        assert prob == pytest.approx(0.50)

    def test_outside_grid_clamps(self) -> None:
        _, prob = _interpolate(self.grid, -10.0)
        assert prob == pytest.approx(0.44)


class TestHalfLine:
    def test_snaps_to_half_points(self) -> None:
        assert _half_line(220.0) == 220.5
        assert _half_line(220.4) == 220.5
        assert _half_line(-2.6) == -2.5
        assert str(_half_line(3.2)).endswith(".5")


class TestDsnTranslation:
    def test_search_path_moved_to_server_settings(self) -> None:
        url, search_path = _split_dsn(
            "postgres://predictions_svc:localdev@localhost:5432/bookiebreaker?search_path=predictions,public"
        )
        assert url == "postgresql+asyncpg://predictions_svc:localdev@localhost:5432/bookiebreaker"
        assert search_path == "predictions,public"

    def test_default_search_path(self) -> None:
        _, search_path = _split_dsn("postgres://u:p@h:5432/db")
        assert search_path == "predictions,public"
