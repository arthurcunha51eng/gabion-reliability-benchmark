"""Application point and direction of the active thrust on the wedge.

Coordinate system
-----------------
Origin at the heel A of the wall. x-axis points toward the toe (away
from the backfill), y-axis points up. Under this convention:

    A = (0, 0)                       — heel
    B = (x',  H·cos β)               — top corner of the back face
    C = (x' + D,  H·cos β)           — ground point at the wedge tip

where x' = H·sin β is the horizontal offset of B from A due to the
wall batter and D is the wedge horizontal extent (chosen by
``earth_pressure.solve``).

How application points are located
----------------------------------
1. The wedge centroid (Xg, Yg) is the average of A, B, and C.
2. Eas (soil-weight component) acts along the failure surface from
   the wedge centroid; we trace this line and intersect it with the
   wall back face. That intersection is (X_Eas, Y_Eas).
3. Eaq (surcharge component) acts at the midpoint of the back face
   — derivation is in the Maccaferri manual.
4. The combined Ea application point is the Eas/Eaq weighted average.
5. The direction θ of Ea above horizontal follows from the back-face
   inclination α and the wall friction angle δ::

       θ = 90° - α + δ - β

   For the OUTSIDE/FLAT case with α = 90°, this simplifies to θ = δ - β.

Convention note
---------------
This module uses ``Y_b = H · cos β`` for the vertical position of the
top corner. The earth-pressure module, in contrast, uses
``A'B = H / cos β`` for the *length* of the inclined back face along
its own axis. Both are correct for their respective purposes; they
diverge by O(β²) for small batter and the difference (≈1% at β=6°)
is preserved here to maintain bit-for-bit reproducibility with the
GAWACWIN-validated baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gabion.inputs import WallScenario
    from gabion.earth_pressure import WedgeResult


@dataclass(frozen=True, slots=True)
class Kinematics:
    """Application of the active thrust resultant Ea on the wedge."""
    X_Ea: float   # x-coordinate of Ea application point [m] (heel-origin frame)
    Y_Ea: float   # y-coordinate [m]
    theta: float  # angle of Ea above horizontal [degrees]
    Ag: float     # cross-sectional area of the wall [m²]


def solve(scenario: "WallScenario", wedge: "WedgeResult") -> Kinematics:
    """Locate Ea on the wedge and compute the wall section area."""
    H = scenario.geometry.H
    beta_rad = radians(scenario.geometry.beta)
    rho_rad = radians(wedge.rho)

    # --- Wedge vertices (heel-origin frame) -------------------------------
    X_a, Y_a = 0.0, 0.0
    X_b = wedge.x_prime
    Y_b = H * cos(beta_rad)
    X_c = wedge.x_prime + wedge.D
    Y_c = Y_b  # flat backfill -> C at the same height as B

    # --- Wedge centroid (centroid of triangle ABC) ------------------------
    X_g = (X_a + X_b + X_c) / 3.0
    Y_g = (Y_a + Y_b + Y_c) / 3.0

    # --- Slopes of the two relevant lines --------------------------------
    # m1: line from A through B (the wall back face).
    # m2: line through (Xg, Yg) parallel to AC (the failure surface).
    m1 = Y_b / X_b
    m2 = tan(rho_rad)

    # --- Application point of the soil-weight component Eas --------------
    # Through the centroid in the m2 direction, intersected with m1.
    X_Eas = (Y_g - m2 * X_g) / (m1 - m2)
    Y_Eas = X_Eas * m1

    # --- Application point of the surcharge component Eaq ----------------
    # Midpoint of the back face (Maccaferri manual).
    X_Eaq = X_b / 2.0
    Y_Eaq = Y_b / 2.0

    # --- Combined Ea application = Eas/Eaq weighted average -------------
    Ea_total = wedge.Eas + wedge.Eaq
    X_Ea = ((wedge.Eas * X_Eas) + (wedge.Eaq * X_Eaq)) / Ea_total
    Y_Ea = ((wedge.Eas * Y_Eas) + (wedge.Eaq * Y_Eaq)) / Ea_total

    # --- Direction of Ea above horizontal --------------------------------
    # θ = (90 - β) - (α - β) + δ - β   simplifies to   90 - α + δ - β.
    theta = 90.0 - wedge.alpha + wedge.delta - scenario.geometry.beta

    # --- Wall cross-sectional area --------------------------------------
    # Each layer is 1.0 m tall; total area is the sum of layer widths.
    Ag = sum(L * 1.0 for L in scenario.geometry.layer_lengths)

    return Kinematics(X_Ea=X_Ea, Y_Ea=Y_Ea, theta=theta, Ag=Ag)
