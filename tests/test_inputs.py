"""Tests for the input data models.

These tests exercise the Pydantic validation layer: type constraints
on scalars and the half-meter grid rule for gabion layers. They run today.
"""
import pytest
from pydantic import ValidationError

from gabion.inputs import (
    GabionMaterial,
    SoilProperties,
    WallGeometry,
    WallScenario,
)


class TestReferenceScenario:
    """The canonical example must construct cleanly and have the expected shape."""

    def test_constructs(self):
        sc = WallScenario.outside_flat_reference()
        assert sc is not None

    def test_geometry_layers(self):
        sc = WallScenario.outside_flat_reference()
        assert sc.geometry.layer_lengths == [2.0, 1.5, 1.0]
        assert sc.geometry.n_layers == 3
        assert sc.geometry.H == 3.0
        assert sc.geometry.L_base == 2.0

    def test_materials(self):
        sc = WallScenario.outside_flat_reference()
        assert sc.gabion.gamma_g == 25.0
        assert sc.gabion.n == 0.30
        assert sc.gabion.geotex_reduction == 0.0
        assert sc.backfill.phi == 30.0
        assert sc.q == 10.0
        assert sc.q_adm == 200.0


class TestLayerValidation:
    """Layer widths must be positive and on the 0.5 m grid."""

    def test_negative_layer_rejected(self):
        with pytest.raises(ValidationError, match="must be > 0"):
            WallGeometry(layer_lengths=[2.0, -1.5, 1.0], beta=6.0)

    def test_off_grid_layer_rejected(self):
        # 1.7 is not a multiple of 0.5 — almost always a typo.
        with pytest.raises(ValidationError, match="multiple of 0.5"):
            WallGeometry(layer_lengths=[2.0, 1.7, 1.0], beta=6.0)

    def test_empty_layers_rejected(self):
        with pytest.raises(ValidationError):
            WallGeometry(layer_lengths=[], beta=6.0)


class TestScalarConstraints:
    """Out-of-range scalars must trigger ValidationError."""

    def test_friction_angle_above_50_rejected(self):
        with pytest.raises(ValidationError):
            SoilProperties(gamma=18.0, phi=55.0, c=0.0)

    def test_negative_unit_weight_rejected(self):
        with pytest.raises(ValidationError):
            SoilProperties(gamma=-1.0, phi=30.0, c=0.0)

    def test_porosity_above_one_rejected(self):
        with pytest.raises(ValidationError):
            GabionMaterial(gamma_g=25.0, n=1.5, geotex_reduction=0.0)

    def test_beta_above_45_rejected(self):
        with pytest.raises(ValidationError):
            WallGeometry(layer_lengths=[2.0, 1.5, 1.0], beta=46.0)
