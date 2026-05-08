"""Deterministic stability check — single entry point.

This module orchestrates the three calculation stages
(``earth_pressure``, ``kinematics``, ``checks``) and returns a flat
result object containing every quantity the test suite or the
probabilistic layers (FOSM, Monte Carlo) need.

The function ``run_check`` is *pure*: scenario in, result out, no I/O,
no globals, no random state. This makes it safe to call from inside
Monte Carlo loops with thousands of perturbed scenarios per second
without worrying about leaked state between iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gabion import checks, earth_pressure, kinematics
from gabion.inputs import WallScenario


@dataclass(frozen=True, slots=True)
class DeterministicResult:
    """Flattened output of a complete deterministic stability check.

    The fields below mirror the regression contract in
    ``tests/test_deterministic.py``: every name asserted by that file
    must appear here.
    """
    # --- Critical wedge -------------------------------------------------
    D_critical: float
    rho: float
    Ea: float

    # --- Kinematics -----------------------------------------------------
    theta: float
    X_Ea: float
    Y_Ea: float

    # --- Sliding --------------------------------------------------------
    N: float
    T_drive: float
    T_resist: float
    FS_sliding: float

    # --- Overturning ----------------------------------------------------
    M_overturning: float
    M_resisting: float
    FS_overturning: float

    # --- Foundation -----------------------------------------------------
    eccentricity: float
    sigma_max: float
    sigma_min: float


def run_check(
    scenario: WallScenario,
    d_override: Optional[float] = None,
) -> DeterministicResult:
    """Run the full deterministic stability check.

    Parameters
    ----------
    scenario
        Complete input describing the wall, soils, and loads.
    d_override
        If ``None`` (default), the engine selects the wedge that
        maximizes Ea over the half-meter D grid (textbook Coulomb).
        If a number, the engine forces the wedge to that D — useful
        for cross-software comparison (e.g., GAWACWIN reports its
        results at D = 2.5 m).

    Returns
    -------
    DeterministicResult
        All factors of safety plus the intermediate quantities the
        FOSM and Monte Carlo layers will eventually consume.
    """
    wedge = earth_pressure.solve(scenario, d_override=d_override)
    kin = kinematics.solve(scenario, wedge)
    chk = checks.solve(scenario, wedge, kin)

    return DeterministicResult(
        D_critical=wedge.D,
        rho=wedge.rho,
        Ea=wedge.Ea,
        theta=kin.theta,
        X_Ea=kin.X_Ea,
        Y_Ea=kin.Y_Ea,
        N=chk.N,
        T_drive=chk.T_drive,
        T_resist=chk.T_resist,
        FS_sliding=chk.FS_sliding,
        M_overturning=chk.M_overturning,
        M_resisting=chk.M_resisting,
        FS_overturning=chk.FS_overturning,
        eccentricity=chk.eccentricity,
        sigma_max=chk.sigma_max,
        sigma_min=chk.sigma_min,
    )
