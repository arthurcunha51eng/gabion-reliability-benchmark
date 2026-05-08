"""Tests for YAML scenario serialization."""
from pathlib import Path

import pytest

from gabion.inputs import WallScenario


# Path to the shipped reference example, resolved relative to this file
# so the tests work regardless of where pytest is invoked from.
REFERENCE_YAML = Path(__file__).resolve().parent.parent / "examples" / "book_example_1.yaml"


class TestLoadShippedExample:
    """The example shipped under examples/ must load and equal the in-code reference."""

    def test_loads_without_error(self):
        sc = WallScenario.from_yaml(REFERENCE_YAML)
        assert sc is not None

    def test_matches_in_code_reference(self):
        # If the YAML drifts from outside_flat_reference(), one of the two
        # has changed silently — fail loudly so the discrepancy is caught.
        loaded = WallScenario.from_yaml(REFERENCE_YAML)
        coded = WallScenario.outside_flat_reference()

        assert loaded.gabion.gamma_g == coded.gabion.gamma_g
        assert loaded.gabion.n == coded.gabion.n
        assert loaded.gabion.geotex_reduction == coded.gabion.geotex_reduction
        assert loaded.geometry.layer_lengths == coded.geometry.layer_lengths
        assert loaded.geometry.beta == coded.geometry.beta
        assert loaded.backfill.gamma == coded.backfill.gamma
        assert loaded.backfill.phi == coded.backfill.phi
        assert loaded.backfill.c == coded.backfill.c
        assert loaded.foundation.phi == coded.foundation.phi
        assert loaded.q == coded.q
        assert loaded.q_adm == coded.q_adm


class TestRoundTrip:
    """Dump-then-load must be a no-op."""

    def test_round_trip_via_file(self, tmp_path):
        original = WallScenario.outside_flat_reference()
        path = tmp_path / "scenario.yaml"
        original.to_yaml(path)
        loaded = WallScenario.from_yaml(path)

        assert loaded.model_dump() == original.model_dump()

    def test_to_yaml_returns_text_even_without_path(self):
        sc = WallScenario.outside_flat_reference()
        text = sc.to_yaml()
        assert isinstance(text, str)
        assert "gabion" in text
        assert "layer_lengths" in text


class TestInvalidYAML:
    """Schema violations in the YAML must surface as ValidationError, not silent garbage."""

    def test_off_grid_layer_in_yaml_rejected(self, tmp_path):
        from pydantic import ValidationError

        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "gabion: {gamma_g: 25, n: 0.3, geotex_reduction: 0}\n"
            "geometry: {layer_lengths: [2.0, 1.7, 1.0], beta: 6}\n"  # 1.7 -> off-grid
            "backfill: {gamma: 18, phi: 30, c: 0}\n"
            "foundation: {gamma: 18, phi: 30, c: 0}\n"
            "q: 10\nq_adm: 200\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError, match="multiple of 0.5"):
            WallScenario.from_yaml(bad)
