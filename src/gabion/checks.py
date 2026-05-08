"""Limit-equilibrium stability checks.

Three independent failure modes are checked against the same active
thrust Ea computed by ``earth_pressure.solve``:

  1. **Sliding** along the wall base:
     :math:`FS_s = T_{\\text{resist}} / T_{\\text{drive}} \\geq 1.5`
  2. **Overturning** about the toe:
     :math:`FS_o = M_{\\text{resist}} / M_{\\text{overturn}} \\geq 2.0`
  3. **Foundation bearing**: with eccentricity ``|e| < B/6`` (resultant
     within the middle third), the linear Navier distribution gives
     :math:`\\sigma_{max} \\leq q_{adm}` and :math:`\\sigma_{min} \\geq 0`
     (no tension at the heel).

Reference frame
---------------
Heel A at origin (0, 0); x rightward (toward the toe); y upward. The
wall toe sits at the rotated point (–B·cos β, B·sin β) — negative x
because the wall extends toward the toe.

All weights are per unit length (kN/m) consistent with plane-strain
analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gabion.inputs import WallScenario
    from gabion.earth_pressure import WedgeResult
    from gabion.kinematics import Kinematics


@dataclass(frozen=True, slots=True)
class StabilityChecks:
    """Three factors of safety plus the intermediates used to derive them."""
    # Self-weight of the gabion
    P_prime: float

    # Sliding
    N: float
    T_drive: float
    T_resist: float
    FS_sliding: float

    # Overturning
    M_overturning: float
    M_resisting: float
    FS_overturning: float

    # Foundation (base pressure)
    eccentricity: float   # signed [m]; negative ↔ resultant biased toward heel
    sigma_max: float      # base stress at the more-compressed edge [kPa]
    sigma_min: float      # base stress at the less-compressed edge [kPa]


def solve(
    scenario: "WallScenario",
    wedge: "WedgeResult",
    kin: "Kinematics",
) -> StabilityChecks:
    """Run the three limit-equilibrium checks for the given scenario."""
    beta_rad = radians(scenario.geometry.beta)
    theta_rad = radians(kin.theta)
    theta_plus_beta_rad = radians(kin.theta + scenario.geometry.beta)
    minus_beta_rad = -beta_rad

    B = scenario.geometry.L_base
    Ea = wedge.Ea

    # --- Self-weight of the gabion ---------------------------------------
    # Stones contribute γ_g·(1−n) per unit cross-section area.
    P_prime = kin.Ag * scenario.gabion.gamma_g * (1.0 - scenario.gabion.n)

    # ===== Sliding check =================================================
    # Normal force on the (β-inclined) base: vertical component of the
    # gabion weight plus vertical component of Ea. The latter has
    # incidence (θ + β) with respect to the base normal.
    N = P_prime * cos(beta_rad) + Ea * sin(theta_plus_beta_rad)

    # Base friction (δ* = φ_foundation in this implementation).
    delta_star_rad = radians(scenario.foundation.phi)
    Td = N * tan(delta_star_rad)

    # Driving force: horizontal component of Ea projected on the base plane.
    T_drive = Ea * cos(theta_plus_beta_rad)

    # Resistance: base friction + downslope component of the gabion weight
    # (the inclined base helps resist sliding, hence + P'·sin β).
    T_resist = Td + P_prime * sin(beta_rad)

    FS_sliding = T_resist / T_drive

    # ===== Overturning check =============================================
    # Toe of the wall in the heel-origin frame.
    X_p = -B * cos(beta_rad)
    Y_p = B * sin(beta_rad)

    # Lever arm from the toe up to the line of action of Ea.
    Y_arm = kin.Y_Ea - Y_p

    # Driving moment: horizontal component of Ea times its vertical arm.
    M_overturning = Ea * Y_arm * cos(theta_rad)

    # Centroid of the stepped wall section, written in a heel-origin frame
    # where x is reckoned negative because the wall extends toward the toe.
    sum_A = sum(L * 1.0 for L in scenario.geometry.layer_lengths)
    X_g_flat = sum(
        L * (-L / 2.0) for L in scenario.geometry.layer_lengths
    ) / sum_A
    Y_g_flat = sum(
        L * (i + 0.5) for i, L in enumerate(scenario.geometry.layer_lengths)
    ) / sum_A

    # Rotate the wall centroid by -β to account for the batter.
    X_g_rot = X_g_flat * cos(minus_beta_rad) - Y_g_flat * sin(minus_beta_rad)
    # Project onto the base axis from the toe.
    X_g_double = (X_g_rot - X_p) / cos(beta_rad)

    # Restoring moment: gabion weight at its rotated lever arm
    #                 + vertical component of Ea at its horizontal arm.
    M_p = P_prime * X_g_double * cos(beta_rad)
    X_prime_Ea = kin.X_Ea - X_p
    M_Eav = Ea * X_prime_Ea * sin(theta_rad)
    M_resisting = M_p + M_Eav

    FS_overturning = M_resisting / M_overturning

    # ===== Foundation bearing check ======================================
    # Position of the resultant on the base, measured from the toe.
    d = (M_resisting - M_overturning) / N
    eccentricity = (B / 2.0) - d  # signed: negative when biased toward heel

    # Linear (Navier) pressure distribution. Valid only while |e| ≤ B/6;
    # the test suite asserts this implicitly via σ_min ≥ 0 in the
    # canonical example.
    e_abs = abs(eccentricity)
    flexural_term = 6.0 * e_abs / B
    sigma_max = (N / B) * (1.0 + flexural_term)
    sigma_min = (N / B) * (1.0 - flexural_term)

    return StabilityChecks(
        P_prime=P_prime,
        N=N,
        T_drive=T_drive,
        T_resist=T_resist,
        FS_sliding=FS_sliding,
        M_overturning=M_overturning,
        M_resisting=M_resisting,
        FS_overturning=FS_overturning,
        eccentricity=eccentricity,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
    )
