"""Active earth pressure on the wall — Coulomb's method of wedges.

Theory
------
For an OUTSIDE-stepped, FLAT-backfill gabion wall, the back face is a
single inclined plane of length ``A'B = H/cos(β)`` ascending from the
heel A at the origin to the corner B. Total wall height H is the sum
of the layer heights (1.0 m each); β is the wall batter from vertical.

A trial failure surface descends from the ground level at horizontal
distance D beyond the wall back to A. The triangular soil wedge so
defined weighs ``P = γ·area``; the surface surcharge q acts over the
horizontal distance D, contributing ``Q = q·D``.

Force equilibrium on the wedge — its weight P+Q balanced by friction
along the failure surface (at φ from its normal) and friction along
the wall back face (at δ = (1-r)·φ from its normal) — yields the
Coulomb equations::

    Eas = P · sin(ρ - φ) / sin(180° - α - ρ + φ + δ)
    Eaq = q·D · sin(ρ - φ) / sin(180° - α - ρ + φ + δ)

where ``α = 90°`` (back face perpendicular to the inclined wall base in
this geometry), ρ is the inclination of the failure surface from
horizontal, and the only free variable is D.

The textbook critical wedge maximizes Ea over D. The default
implementation searches a 0.5 m grid from 0.5 m to 5.0 m, matching the
GAWACWIN convention; ``d_override`` allows fixing D for cross-software
comparison.

References
----------
* Das, B. M. — *Principles of Foundation Engineering*, 8th ed., §14.4
  (active earth pressure, method of wedges).
* Craig, R. F. & Knappett, J. A. — *Soil Mechanics*, 9th ed., Ch. 12.
* Maccaferri — *Gabion Retaining Walls Design Manual* (GAWACWIN).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, degrees, radians, sin
from typing import TYPE_CHECKING, Optional

# WallScenario is only used as a type — not constructed here, just consumed
# by attribute access. Importing under TYPE_CHECKING keeps this module
# free of a runtime dependency on Pydantic, which makes it cheap to import
# from large Monte Carlo loops.
if TYPE_CHECKING:
    from gabion.inputs import WallScenario


# Default search grid for the critical wedge: matches GAWACWIN's
# convention of evaluating D in 0.5 m increments from 0.5 m to 5.0 m.
_D_SEARCH_GRID = tuple(k / 2.0 for k in range(1, 11))


@dataclass(frozen=True, slots=True)
class WedgeResult:
    """Output of the wedge analysis at one specific value of D.

    Geometry intermediates (``A_prime_B``, ``x_prime``, ``alpha``,
    ``delta``) are exposed because the kinematics module needs them to
    locate the application point of Ea without recomputing them.
    """
    D: float          # horizontal extent of the wedge from the heel [m]
    rho: float        # inclination of the failure surface from horizontal [deg]
    Ea: float         # total active thrust [kN/m]
    Eas: float        # ... due to soil self-weight [kN/m]
    Eaq: float        # ... due to surface surcharge [kN/m]
    P: float          # weight of the wedge [kN/m]

    # Geometry intermediates passed downstream
    A_prime_B: float  # length of the virtual back face [m]
    x_prime: float    # horizontal offset of B from A due to wall batter [m]
    alpha: float      # virtual back face inclination [deg]
    delta: float      # wall friction angle δ = (1-r)·φ [deg]


def _evaluate_at_D(scenario: "WallScenario", D: float) -> WedgeResult:
    """Compute Ea and the wedge geometry for a single value of D."""
    H = scenario.geometry.H
    beta_deg = scenario.geometry.beta
    phi_deg = scenario.backfill.phi
    gamma = scenario.backfill.gamma
    q = scenario.q
    r_geotex = scenario.gabion.geotex_reduction

    # Wall friction angle, possibly reduced by a geotextile interface.
    delta_deg = (1.0 - r_geotex) * phi_deg

    # For the OUTSIDE-stepped / FLAT-backfill geometry, the virtual
    # back face is perpendicular to the inclined wall base.
    alpha_deg = 90.0

    beta_rad = radians(beta_deg)
    A_prime_B = H / cos(beta_rad)
    x_prime = H * sin(beta_rad)

    # Wedge area: triangle from heel A to top B to ground point C.
    # Because α=90° the geometry reduces to (A'B · D) / 2.
    area = (A_prime_B * D) / 2.0
    P = gamma * area

    # Inclination of the failure surface from horizontal.
    rho_deg = degrees(atan(A_prime_B / (D + x_prime)))

    # Coulomb denominator (shared by Eas and Eaq).
    denom_angle_deg = 180.0 - alpha_deg - rho_deg + phi_deg + delta_deg
    denom = sin(radians(denom_angle_deg))
    num_factor = sin(radians(rho_deg - phi_deg))

    Eas = P * num_factor / denom
    # Surcharge contribution: q acts over the horizontal extent D only,
    # because A'A = A'B · sin(90°-α) = 0 when α=90°.
    Eaq = q * D * num_factor / denom

    return WedgeResult(
        D=D,
        rho=rho_deg,
        Ea=Eas + Eaq,
        Eas=Eas,
        Eaq=Eaq,
        P=P,
        A_prime_B=A_prime_B,
        x_prime=x_prime,
        alpha=alpha_deg,
        delta=delta_deg,
    )


def solve(
    scenario: "WallScenario",
    d_override: Optional[float] = None,
) -> WedgeResult:
    """Return the critical wedge for the given scenario.

    Parameters
    ----------
    scenario
        Wall, soil, and loading inputs.
    d_override
        If ``None`` (default), search for the wedge that maximizes Ea
        over a 0.5 m grid (textbook Coulomb). If a number, force the
        wedge to that D — used to compare against software (e.g.
        GAWACWIN) that selects the wedge by a different criterion.
    """
    if d_override is not None:
        return _evaluate_at_D(scenario, d_override)

    candidates = [_evaluate_at_D(scenario, D) for D in _D_SEARCH_GRID]
    return max(candidates, key=lambda w: w.Ea)
