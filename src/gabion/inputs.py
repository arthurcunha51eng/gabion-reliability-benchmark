"""Pydantic data models for gabion wall stability analysis inputs.

Scope of v0.1
-------------
This package targets a single, well-defined wall topology:

  * step face on the OUTSIDE (toe side); the back face is a single inclined plane
  * FLAT backfill surface (no upper slope above the wall)
  * dry conditions (no groundwater)
  * static loading (no seismic)

The data models below describe only this configuration; other
topologies are out of scope for the current project.
"""

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# -----------------------------------------------------------------------------
# Constrained scalar types
# -----------------------------------------------------------------------------
# Annotated types document units alongside validation. Constraint violations
# fail loudly at scenario construction, before any computation begins.

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]
FrictionAngle = Annotated[float, Field(ge=0, le=50, description="degrees")]


# -----------------------------------------------------------------------------
# Material and geometry
# -----------------------------------------------------------------------------
class GabionMaterial(BaseModel):
    """Stone fill of the gabion baskets."""
    model_config = ConfigDict(extra="forbid")

    gamma_g: PositiveFloat = Field(description="Stone unit weight γ_g [kN/m³]")
    n: float = Field(ge=0.0, le=1.0, description="Porosity (void fraction) [-]")
    geotex_reduction: float = Field(
        ge=0.0,
        lt=1.0,
        description="Reduction factor on δ due to geotextile [-]; "
                    "δ = (1 - r)·φ_backfill",
    )


class WallGeometry(BaseModel):
    """Stepped-trapezoidal wall cross-section.

    Each gabion layer is 1.0 m tall by convention; the cross-section is
    therefore fully described by the list of layer widths (base → top)
    and the wall batter angle β.
    """
    model_config = ConfigDict(extra="forbid")

    layer_lengths: list[float] = Field(
        min_length=1,
        description="Layer widths from base to top, each layer 1.0 m tall [m]",
    )
    beta: float = Field(
        ge=0.0,
        le=45.0,
        description="Wall batter angle from vertical [degrees]",
    )

    @field_validator("layer_lengths")
    @classmethod
    def _layers_on_half_meter_grid(cls, v: list[float]) -> list[float]:
        # Standard gabion baskets ship in 0.5 m width increments. Non-grid
        # values are almost certainly typos; reject them rather than silently
        # accept and confuse downstream calculations.
        for i, L in enumerate(v):
            if L <= 0:
                raise ValueError(f"layer_lengths[{i}] = {L}: must be > 0.")
            if abs((L * 2) - round(L * 2)) > 1e-9:
                raise ValueError(
                    f"layer_lengths[{i}] = {L}: must be a multiple of 0.5 m."
                )
        return v

    @property
    def n_layers(self) -> int:
        return len(self.layer_lengths)

    @property
    def H(self) -> float:
        """Total wall height [m] — each layer is 1.0 m tall."""
        return float(self.n_layers) * 1.0

    @property
    def L_base(self) -> float:
        """Base width B [m]."""
        return self.layer_lengths[0]


class SoilProperties(BaseModel):
    """Mohr-Coulomb soil parameters (drained / effective stress)."""
    model_config = ConfigDict(extra="forbid")

    gamma: PositiveFloat = Field(description="Unit weight γ [kN/m³]")
    phi: FrictionAngle = Field(description="Effective friction angle φ' [degrees]")
    c: NonNegFloat = Field(description="Effective cohesion c' [kPa]")


# -----------------------------------------------------------------------------
# Top-level scenario
# -----------------------------------------------------------------------------
class WallScenario(BaseModel):
    """Complete input for a deterministic stability check."""
    model_config = ConfigDict(extra="forbid")

    gabion: GabionMaterial
    geometry: WallGeometry
    backfill: SoilProperties
    foundation: SoilProperties

    q: NonNegFloat = Field(description="Surcharge on the backfill surface [kN/m²]")
    q_adm: PositiveFloat = Field(description="Allowable bearing capacity [kPa]")

    @classmethod
    def outside_flat_reference(cls) -> "WallScenario":
        """Canonical reference scenario used for validation and benchmarking.

        Three layers [2.0, 1.5, 1.0] m, β = 6°, dry conditions, surcharge q = 10.
        With the deterministic engine constrained to D = 2.5 m (GAWACWIN's
        critical wedge), this scenario reproduces GAWACWIN's published
        outputs to four significant figures. See
        ``tests/test_deterministic.py::TestGawacwinEquivalence``.
        """
        return cls(
            gabion=GabionMaterial(gamma_g=25.0, n=0.30, geotex_reduction=0.0),
            geometry=WallGeometry(layer_lengths=[2.0, 1.5, 1.0], beta=6.0),
            backfill=SoilProperties(gamma=18.0, phi=30.0, c=0.0),
            foundation=SoilProperties(gamma=18.0, phi=30.0, c=0.0),
            q=10.0,
            q_adm=200.0,
        )

    # ------------------------------------------------------------------
    # YAML serialization
    # ------------------------------------------------------------------
    # Lazy-import PyYAML so that ``inputs.py`` stays importable in
    # environments that don't have it (e.g., when only the deterministic
    # engine is used as a library, not the CLI).

    @classmethod
    def from_yaml(cls, path) -> "WallScenario":
        """Construct a scenario from a YAML file.

        The YAML structure mirrors the Pydantic model exactly:
        top-level keys ``gabion``, ``geometry``, ``backfill``,
        ``foundation``, ``q``, ``q_adm``. See
        ``examples/book_example_1.yaml``.
        """
        import yaml
        from pathlib import Path

        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path=None) -> str:
        """Serialize the scenario to YAML.

        If ``path`` is provided, the YAML is also written to that file.
        Returns the YAML text in either case so the result is testable
        without touching the filesystem.
        """
        import yaml
        from pathlib import Path

        text = yaml.safe_dump(
            self.model_dump(),
            sort_keys=False,    # preserve the natural top-down ordering
            default_flow_style=False,
        )
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text
