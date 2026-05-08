"""Symbolic limit-state functions for FOSM/FORM analysis.

This module builds SymPy expressions for the three failure modes of an
OUTSIDE-stepped, FLAT-backfill gabion gravity wall:

    g_sliding(X)      = FS_sliding(X)     - 1
    g_overturning(X)  = FS_overturning(X) - 1
    g_bearing(X)      = q_adm / sigma_max(X) - 1

where ``X = (phi_backfill, gamma_backfill, phi_foundation, q, gamma_g)``
are the random variables; everything else in the WallScenario is held
fixed. The critical wedge ``D`` is also held constant — at the
deterministic value computed at the mean of X — under the standard
assumption that small perturbations of X do not switch the critical
wedge (validated empirically by Stage E sample-counts).

Why duplicate the math here?
----------------------------
The deterministic engine (``earth_pressure``, ``kinematics``, ``checks``)
uses ``math.sin/cos/tan`` which only accept numeric inputs; SymPy needs
``sp.sin``, ``sp.cos``, ``sp.tan``. Re-deriving in SymPy is one option;
refactoring the engine to be ``math``-agnostic is another. The first is
simpler and locally verifiable: the regression test asserts that the
symbolic g evaluated at the mean of X reproduces the deterministic FS
to better than 1e-9 relative tolerance — any drift between the two
implementations is caught immediately.

Subsequent FOSM/FORM stages
---------------------------
``build_limit_states`` is the only ingestion point for downstream
methods. ``mvfosm()`` (Stage D.3) consumes the means and gradients-at-
means; ``hl_fosm()`` (Stage D.4) iterates with the same gradient; FORM
(Stage F) reuses the gradient and adds a Nataf transformation on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from scipy.stats import norm

if TYPE_CHECKING:
    from gabion.inputs import WallScenario
    from gabion.random_variables import NatafTransformation, RandomVariableSpec


# =============================================================================
# Symbol declarations
# =============================================================================
# Module-level so SymPy keys ``subs``, ``diff``, and ``lambdify`` on the
# same identity. Names match ``random_variables.book_example_spec()``.
PHI_BACKFILL, GAMMA_BACKFILL, PHI_FOUNDATION, Q, GAMMA_G = sp.symbols(
    "phi_backfill gamma_backfill phi_foundation q gamma_g",
    positive=True,
    real=True,
)

# Canonical ordering — used by gradient and design-point vectors in
# Stages D.3, D.4, and beyond. Keep aligned with SYMBOL_NAMES.
SYMBOLS: tuple[sp.Symbol, ...] = (
    PHI_BACKFILL,
    GAMMA_BACKFILL,
    PHI_FOUNDATION,
    Q,
    GAMMA_G,
)
SYMBOL_NAMES: tuple[str, ...] = (
    "phi_backfill",
    "gamma_backfill",
    "phi_foundation",
    "q",
    "gamma_g",
)


# =============================================================================
# Limit-state container
# =============================================================================
@dataclass(frozen=True)
class LimitStates:
    """The three symbolic limit-state functions in margin form.

    For each ``g_i``: ``g_i(X) > 0`` ⇔ safe; ``g_i(X) < 0`` ⇔ failure.

    Per-mode resistance ``R`` and load ``S`` expressions are also
    exposed (``<mode>_R``, ``<mode>_S``) so the Stage D.5 invariance
    demonstration can reformulate ``g`` as ``R - S`` or ``ln(R/S)``
    without re-deriving the engine math.

    The signed eccentricity ``eccentricity`` is exposed so Stage E's
    Monte Carlo can count samples where ``e > 0`` (resultant moves
    to the toe side, violating the ``|e| = -e`` assumption used in
    the bearing limit-state formula).
    """
    sliding: sp.Expr
    overturning: sp.Expr
    bearing: sp.Expr
    # --- Resistance/load decompositions ---
    sliding_R: sp.Expr      # T_resist
    sliding_S: sp.Expr      # T_drive
    overturning_R: sp.Expr  # M_resist
    overturning_S: sp.Expr  # M_overturn
    bearing_R: sp.Expr      # q_adm (constant)
    bearing_S: sp.Expr      # σ_max
    # --- Diagnostic ---
    eccentricity: sp.Expr   # signed; assumption is e < 0 throughout


# =============================================================================
# Builder
# =============================================================================
def build_limit_states(scenario: "WallScenario", d_critical: float) -> LimitStates:
    """Assemble the symbolic g(X) for the three failure modes.

    Parameters
    ----------
    scenario
        Provides the deterministic geometry/material constants that get
        baked into the expressions.
    d_critical
        Critical wedge D [m]. Typically supplied by
        ``deterministic.run_check(scenario_at_means).D_critical``.

    Returns
    -------
    LimitStates with symbolic g_sliding, g_overturning, g_bearing.
    """
    # ------------------------------------------------------------ constants
    H = scenario.geometry.H
    beta_deg = scenario.geometry.beta
    r_geotex = scenario.gabion.geotex_reduction
    n_porosity = scenario.gabion.n
    layers = scenario.geometry.layer_lengths
    B = scenario.geometry.L_base
    q_adm = scenario.q_adm

    beta_rad = math.radians(beta_deg)

    # Earth-pressure geometry intermediates (numeric — D and β are fixed)
    A_prime_B = H / math.cos(beta_rad)
    x_prime = H * math.sin(beta_rad)
    area_wedge = (A_prime_B * d_critical) / 2.0
    rho_deg = math.degrees(math.atan(A_prime_B / (d_critical + x_prime)))
    rho_rad = math.radians(rho_deg)

    # Wall section area (constant)
    Ag = sum(L * 1.0 for L in layers)

    # Wedge centroid + Eas/Eaq application points (constants — D, β fixed)
    Y_b = H * math.cos(beta_rad)
    X_b = x_prime
    X_c = X_b + d_critical
    Y_c = Y_b
    X_g_wedge = (X_b + X_c) / 3.0
    Y_g_wedge = (Y_b + Y_c) / 3.0
    m1 = Y_b / X_b
    m2 = math.tan(rho_rad)
    X_Eas = (Y_g_wedge - m2 * X_g_wedge) / (m1 - m2)
    Y_Eas = X_Eas * m1
    X_Eaq = X_b / 2.0
    Y_Eaq = Y_b / 2.0

    # Wall centroid in the heel-origin frame, rotated by -β
    sum_A = sum(layers)
    X_g_flat = sum(L * (-L / 2.0) for L in layers) / sum_A
    Y_g_flat = sum(L * (i + 0.5) for i, L in enumerate(layers)) / sum_A
    X_g_rot = X_g_flat * math.cos(-beta_rad) - Y_g_flat * math.sin(-beta_rad)
    X_p = -B * math.cos(beta_rad)
    Y_p = B * math.sin(beta_rad)
    X_g_double = (X_g_rot - X_p) / math.cos(beta_rad)

    # ------------------------------------------------------------ symbolic
    # Wall friction angle, in degrees, depends on phi_backfill.
    delta = (1.0 - r_geotex) * PHI_BACKFILL

    # alpha = 90° in this geometry, so theta = 90 - alpha + delta - β
    # simplifies to (delta - β); theta + β simplifies to delta. We write
    # the algebra explicitly for readability and to avoid hidden algebraic
    # simplification that could mask a future bug.
    alpha_deg_const = 90.0
    theta = (90.0 - alpha_deg_const) + delta - beta_deg     # = delta - β
    theta_plus_beta = theta + beta_deg                       # = delta

    # Coulomb wedge — numerator and denominator factors
    denom_angle = 180.0 - alpha_deg_const - rho_deg + PHI_BACKFILL + delta
    denom = sp.sin(sp.rad(denom_angle))
    num_factor = sp.sin(sp.rad(rho_deg - PHI_BACKFILL))

    # Active thrust components
    P_wedge = GAMMA_BACKFILL * area_wedge
    Eas = P_wedge * num_factor / denom
    Eaq = Q * d_critical * num_factor / denom
    Ea = Eas + Eaq

    # Application point of the resultant Ea (weighted average; the
    # common factor num_factor/denom cancels, so X_Ea/Y_Ea depend on
    # gamma_backfill and q only — phi_backfill drops out at this step.)
    X_Ea = (Eas * X_Eas + Eaq * X_Eaq) / Ea
    Y_Ea = (Eas * Y_Eas + Eaq * Y_Eaq) / Ea

    # Self-weight of the gabion
    P_prime = Ag * GAMMA_G * (1.0 - n_porosity)

    # ------------------------------------------------------ Sliding mode
    # R_sliding = N tan(φ_f) + P' sin(β); S_sliding = Ea cos(θ + β).
    N = (
        P_prime * math.cos(beta_rad)
        + Ea * sp.sin(sp.rad(theta_plus_beta))
    )
    sliding_S = Ea * sp.cos(sp.rad(theta_plus_beta))
    sliding_R = (
        N * sp.tan(sp.rad(PHI_FOUNDATION))
        + P_prime * math.sin(beta_rad)
    )
    g_sliding = sliding_R / sliding_S - 1

    # ----------------------------------------------------- Overturning
    # R_overturn = M_p + M_Eav; S_overturn = Ea Y_arm cos(θ).
    Y_arm = Y_Ea - Y_p
    overturning_S = Ea * Y_arm * sp.cos(sp.rad(theta))
    M_p = P_prime * X_g_double * math.cos(beta_rad)
    X_prime_Ea = X_Ea - X_p
    M_Eav = Ea * X_prime_Ea * sp.sin(sp.rad(theta))
    overturning_R = M_p + M_Eav
    g_overturning = overturning_R / overturning_S - 1

    # ------------------------------------------- Bearing (sigma_max ≤ q_adm)
    # Resultant position from the toe; eccentricity signed (negative ↔
    # bias toward heel). At the canonical mean the eccentricity is
    # ≈ -0.131 m. Holding e < 0 lets us write |e| = -e and stay smooth
    # for SymPy differentiation. Stage E counts samples that violate
    # this assumption; if the count is non-trivial the formula must
    # be revisited.
    d_resultant = (overturning_R - overturning_S) / N
    eccentricity = (B / 2.0) - d_resultant
    sigma_max = (N / B) * (1.0 - 6.0 * eccentricity / B)  # uses |e| = -e
    bearing_R = sp.Float(q_adm)  # wrap as sp.Float so dataclass sees a SymPy expr
    bearing_S = sigma_max
    g_bearing = bearing_R / bearing_S - 1

    return LimitStates(
        sliding=g_sliding,
        overturning=g_overturning,
        bearing=g_bearing,
        sliding_R=sliding_R,
        sliding_S=sliding_S,
        overturning_R=overturning_R,
        overturning_S=overturning_S,
        bearing_R=bearing_R,
        bearing_S=bearing_S,
        eccentricity=eccentricity,
    )


# =============================================================================
# Convenience: numerical evaluation at means
# =============================================================================
def evaluate_at_means(expr: sp.Expr, spec: "RandomVariableSpec") -> float:
    """Numerically evaluate ``expr`` at the mean values declared in ``spec``.

    Used primarily by the cross-validation tests; downstream FOSM stages
    have their own (vectorized) substitution paths.
    """
    subs = {sym: spec[name].mean for sym, name in zip(SYMBOLS, SYMBOL_NAMES)}
    return float(expr.subs(subs))


# =============================================================================
# Result container
# =============================================================================
@dataclass(frozen=True)
class FosmResult:
    """Output of a FOSM analysis on a single failure mode.

    Methods MVFOSM and HL-FOSM share this container; ``design_point``
    and ``n_iter`` are populated only for HL-FOSM (Stage D.4).

    Notes
    -----
    * ``beta = mu_g / sigma_g`` for MVFOSM (Cornell, linearization at
      the mean). For HL-FOSM, ``beta`` is the distance from the origin
      to the design point in standard normal space.
    * ``pf = Φ(-beta)``. This is exact only when g is normally
      distributed — i.e., when g is linear in normal X. For non-linear
      g or non-normal X (handled by FORM in Stage F), it is an
      approximation; the error grows with both non-linearity and
      distribution skew.
    * ``sensitivity`` are the coefficients γ_i = (∂g/∂X_i · σ_i) / σ_g
      computed at the linearization point (mean for MVFOSM, design
      point for HL-FOSM). Sign convention: γ_i > 0 ↔ increasing X_i
      increases g (resistance-like); γ_i < 0 ↔ load-like. γ_i² sum
      to 1 only when X are uncorrelated; under correlation the sum
      can exceed 1 because a single physical X "uses" the variance
      it shares with other X's.
    """
    method: str            # "MVFOSM" or "HL-FOSM"
    mode: str              # "sliding", "overturning", "bearing"
    beta: float
    pf: float
    g_at_mean: float       # μ_g — value of g at the linearization point
    sigma_g: float         # σ_g — first-order standard deviation of g
    sensitivity: dict[str, float]
    design_point: dict[str, float] | None = None  # HL-FOSM only
    n_iter: int | None = None                     # HL-FOSM only
    alpha: dict[str, float] | None = None         # HL-FOSM only — Y-space
                                                   # directional cosines
                                                   # (Naccache eq. 5.41)
    # ----- Convergence diagnostics (HL-FOSM only) -----
    # MVFOSM is closed-form (no convergence concept); converged=True
    # always. HL-FOSM may fail when the design point lies in a region
    # where the limit-state model itself breaks down (e.g., bearing
    # whose σ_max formula is invalid beyond the base kern). When that
    # happens we return beta=pf=NaN with an informative note rather
    # than masking the failure. See Stage D.4 docstring + planned
    # docs/llm_failure_modes.md for the pedagogical discussion.
    converged: bool = True
    convergence_note: str | None = None


# =============================================================================
# MVFOSM (Cornell, 1969) — linearization at the mean
# =============================================================================
def _mvfosm_at_means(
    g: sp.Expr,
    spec: "RandomVariableSpec",
    mode_name: str,
) -> FosmResult:
    """Apply Cornell MVFOSM to a single symbolic limit state ``g``.

    Algorithm
    ---------
    1. μ_g = g(μ_X)                   — substitute means into g
    2. ∇g  = ∂g/∂X_i evaluated at μ_X — symbolic diff + subs
    3. σ_g² = ∇gᵀ · C_X · ∇g          — quadratic form with covariance
    4. β    = μ_g / σ_g
    5. P_f  = Φ(-β)

    The variance formula is the exact propagation of variance for a
    *linear* g — its application to nonlinear g is the first-order
    approximation that gives MVFOSM its name.
    """
    means_dict = {sym: spec[name].mean for sym, name in zip(SYMBOLS, SYMBOL_NAMES)}
    means_vec = np.array([spec[name].mean for name in SYMBOL_NAMES])
    stds_vec = np.array([spec[name].std for name in SYMBOL_NAMES])

    # μ_g
    g_at_mean = float(g.subs(means_dict))

    # ∇g at the mean — symbolic diff then numerical substitution.
    # diff is exact; subs is to floats. SymPy returns 0 (exact) for
    # variables not present in g, which is the desired behavior:
    # those components of the gradient are exactly zero, not
    # numerically tiny.
    grad_symbolic = [sp.diff(g, sym) for sym in SYMBOLS]
    grad_at_mean = np.array(
        [float(d.subs(means_dict)) for d in grad_symbolic]
    )

    # σ_g² via the full quadratic form (uses the off-diagonal covariance
    # contributions). Equivalent to ||Lᵀ ∇g||² where L Lᵀ = C_X — the
    # latter form is what HL-FOSM uses internally; stick to the direct
    # form here for clarity.
    C_X = spec.covariance_matrix()
    var_g = float(grad_at_mean @ C_X @ grad_at_mean)
    if var_g <= 0:
        raise ValueError(
            f"MVFOSM[{mode_name}]: σ_g² = {var_g} ≤ 0 — limit state has "
            f"no first-order uncertainty; check that gradient is nonzero "
            f"and covariance matrix is positive-definite."
        )
    sigma_g = math.sqrt(var_g)

    # Reliability index and probability of failure. Use survival
    # function (sf) instead of cdf(-β) for numerical accuracy when β
    # is large (cdf saturates to 0 around β ≈ 8 in double precision).
    beta = g_at_mean / sigma_g
    pf = float(norm.sf(beta))

    # X-space sensitivity coefficients γ_i = (∂g/∂X_i · σ_i) / σ_g.
    # Variables not in g get exactly 0 (we asked SymPy for an exact
    # zero, and we divide by the same σ_g for every i).
    sensitivity = {
        name: float(grad_at_mean[i] * stds_vec[i] / sigma_g)
        for i, name in enumerate(SYMBOL_NAMES)
    }

    return FosmResult(
        method="MVFOSM",
        mode=mode_name,
        beta=beta,
        pf=pf,
        g_at_mean=g_at_mean,
        sigma_g=sigma_g,
        sensitivity=sensitivity,
        design_point=None,
        n_iter=None,
    )


def mvfosm(
    scenario: "WallScenario",
    spec: "RandomVariableSpec",
) -> dict[str, FosmResult]:
    """Cornell MVFOSM for the three failure modes of a gabion wall.

    Workflow:
      1. Run the deterministic engine at the scenario's nominal values
         (which must coincide with ``spec.means`` — verified by the
         alignment test in Stage D.2) to fix the critical wedge D.
      2. Build symbolic limit-state functions with that D.
      3. Apply Cornell MVFOSM to each.

    Returns
    -------
    Dict keyed by mode name: ``{"sliding": ..., "overturning": ...,
    "bearing": ...}``. Each value is a :class:`FosmResult`.
    """
    # Local import keeps the module dependency-light at parse time;
    # ``run_check`` pulls in the whole engine chain.
    from gabion.deterministic import run_check

    det = run_check(scenario)
    ls = build_limit_states(scenario, d_critical=det.D_critical)

    return {
        "sliding":     _mvfosm_at_means(ls.sliding,     spec, "sliding"),
        "overturning": _mvfosm_at_means(ls.overturning, spec, "overturning"),
        "bearing":     _mvfosm_at_means(ls.bearing,     spec, "bearing"),
    }


# =============================================================================
# HL-FOSM (Hasofer-Lind, 1974) — iterative HLRF in X-space
# =============================================================================
def _build_lambdified(g: sp.Expr) -> tuple:
    """Compile g and its gradient to fast NumPy callables.

    Returned ``g_func(*x)`` evaluates to a scalar for scalar inputs;
    ``grad_funcs[i](*x)`` evaluates to ∂g/∂X_i. We compile once and
    reuse across all HLRF iterations — straight ``g.subs()`` would
    take ~50× longer per evaluation for our complex expressions.
    """
    g_func = sp.lambdify(SYMBOLS, g, modules="numpy")
    grad_funcs = [
        sp.lambdify(SYMBOLS, sp.diff(g, sym), modules="numpy")
        for sym in SYMBOLS
    ]
    return g_func, grad_funcs


def _hlrf_attempt(
    g_func,
    grad_funcs,
    spec: "RandomVariableSpec",
    mode_name: str,
    tol_x: float,
    tol_g: float,
    max_iter: int,
    relaxation: float,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """One attempt at HLRF iteration with the given relaxation factor.

    Returns (x_design, g_at_design, grad_at_design, n_iterations) on
    convergence. Raises RuntimeError if max_iter is exceeded.

    relaxation = 1.0 is classic HLRF (Naccache eq. 5.55). Smaller values
    apply a fixed step damping that helps when g is strongly nonlinear
    (e.g., bearing's 1/σ_max term causes classic HLRF to oscillate).
    """
    means = spec.means.copy()
    C_X = spec.covariance_matrix()

    x = means.copy()
    delta_x = np.inf
    g_k = float(g_func(*x))

    for it in range(1, max_iter + 1):
        g_k = float(g_func(*x))
        grad_k = np.array([float(gi(*x)) for gi in grad_funcs])

        Cgrad = C_X @ grad_k
        denom = float(grad_k @ Cgrad)
        if denom <= 0:
            raise ValueError(
                f"HL-FOSM[{mode_name}]: ‖∇g‖²_C_X = {denom} ≤ 0 at "
                f"iteration {it}. Gradient may be exactly zero or "
                f"covariance matrix ill-conditioned."
            )

        scalar = float(grad_k @ (x - means)) - g_k
        x_full = means + (scalar / denom) * Cgrad
        # Damped step: x_new = x + relaxation·(x_full - x).
        x_new = x + relaxation * (x_full - x)

        delta_x = float(np.linalg.norm(x_new - x))
        x = x_new

        if delta_x < tol_x and abs(g_k) < tol_g:
            return x, g_k, grad_k, it

    raise RuntimeError(
        f"HL-FOSM[{mode_name}]: did not converge in {max_iter} iterations "
        f"with relaxation = {relaxation} (final ‖Δx‖ = {delta_x:.2e}, "
        f"|g| = {abs(g_k):.2e})."
    )


def _hl_fosm_iterate(
    g: sp.Expr,
    spec: "RandomVariableSpec",
    mode_name: str,
    tol_x: float,
    tol_g: float,
    max_iter: int,
) -> FosmResult:
    """HLRF iteration with automatic relaxation fallback.

    The iteration runs in X-space using Naccache's eq. 5.55::

        x_{k+1} = μ_X + (1/||∇g||²_C_X) ·
                  [∇g(x_k)ᵀ (x_k − μ_X) − g(x_k)] · C_X · ∇g(x_k)

    where ``||∇g||²_C_X = ∇gᵀ C_X ∇g``. Geometrically equivalent to
    standard HLRF in standardized Y-space, but avoids one matrix-solve
    per iteration.

    Robustness: classic HLRF (relaxation = 1.0) can oscillate for
    strongly nonlinear g — the bearing limit state is the canonical
    offender here because ``g_bearing = q_adm/σ_max − 1`` becomes
    singular as σ_max → 0. We try relaxations
    [1.0, 0.5, 0.25, 0.1] in order, returning the first that converges.
    Reported ``n_iter`` is the iteration count of the successful attempt.
    """
    g_func, grad_funcs = _build_lambdified(g)
    means = spec.means.copy()
    C_X = spec.covariance_matrix()
    L = np.linalg.cholesky(C_X)

    g_at_mean = float(g_func(*means))

    # Try progressively smaller relaxations until one converges.
    last_error = None
    converged_attempt = None
    for relaxation in (1.0, 0.5, 0.25, 0.1):
        try:
            x, _, grad_design, iterations = _hlrf_attempt(
                g_func, grad_funcs, spec, mode_name,
                tol_x, tol_g, max_iter, relaxation,
            )
            converged_attempt = (x, grad_design, iterations, relaxation)
            break
        except RuntimeError as e:
            last_error = e
            continue

    # If no relaxation converged, return a non-convergent result rather
    # than raising. This is by design: some limit states (notably
    # bearing in the canonical scenario) have design points in regions
    # where the underlying physical model is no longer valid, so the
    # iteration cannot find a meaningful x*. Reporting failure honestly
    # is more useful than masking it.
    if converged_attempt is None:
        nan = float("nan")
        return FosmResult(
            method="HL-FOSM",
            mode=mode_name,
            beta=nan,
            pf=nan,
            g_at_mean=g_at_mean,
            sigma_g=nan,
            sensitivity={name: nan for name in SYMBOL_NAMES},
            design_point=None,
            n_iter=-1,
            alpha=None,
            converged=False,
            convergence_note=(
                f"HLRF did not converge with relaxations [1.0, 0.5, "
                f"0.25, 0.1]. Last attempt: {last_error}. The design "
                f"point may lie in a region where the limit-state "
                f"model is no longer physically valid (e.g., bearing's "
                f"rigid-foundation σ_max formula breaks down beyond "
                f"|e| > B/2). MVFOSM β provides a linearization-at-"
                f"mean estimate; FORM (Stage F) with proper "
                f"distribution transformation may behave better."
            ),
        )

    x, grad_design, iterations, _ = converged_attempt

    # ----------------------------- Final design-point quantities
    grad_Y = L.T @ grad_design                 # gradient in Y-space
    norm_grad_Y = float(np.linalg.norm(grad_Y))

    # β is the geometric distance from origin to design point in Y-space.
    # Sign convention: β > 0 ⇔ mean is in safe region (g(μ) > 0).
    y_star = np.linalg.solve(L, x - means)
    beta_magnitude = float(np.linalg.norm(y_star))
    beta = beta_magnitude if g_at_mean >= 0 else -beta_magnitude
    pf = float(norm.sf(beta))

    # α_i in Y-space (Naccache eq. 5.41): α = ∇g_Y / ‖∇g_Y‖.
    # Implies y* = -α · β (eq. 5.42). Sign:
    #   α_i > 0 ↔ Y_i is resistance-like at design point;
    #   α_i < 0 ↔ Y_i is load-like.
    # Squares Σ α_i² = 1 (unit normal).
    alpha = {
        name: float(grad_Y[i] / norm_grad_Y)
        for i, name in enumerate(SYMBOL_NAMES)
    }

    # X-space sensitivity at design point (same definition as MVFOSM,
    # gradient now at x* rather than μ).
    sensitivity = {
        name: float(grad_design[i] * spec.stds[i] / norm_grad_Y)
        for i, name in enumerate(SYMBOL_NAMES)
    }

    design_point = {name: float(x[i]) for i, name in enumerate(SYMBOL_NAMES)}

    return FosmResult(
        method="HL-FOSM",
        mode=mode_name,
        beta=beta,
        pf=pf,
        g_at_mean=g_at_mean,
        sigma_g=norm_grad_Y,
        sensitivity=sensitivity,
        design_point=design_point,
        n_iter=iterations,
        alpha=alpha,
    )


def hl_fosm(
    scenario: "WallScenario",
    spec: "RandomVariableSpec",
    tol_x: float = 1e-8,
    tol_g: float = 1e-8,
    max_iter: int = 50,
) -> dict[str, FosmResult]:
    """Hasofer-Lind FOSM via HLRF iteration for the three failure modes.

    Identifies the design point (closest point on g(X) = 0 to μ_X in
    the C_X metric) for each mode and reports the geometric reliability
    index β = ‖y*‖. Unlike Cornell MVFOSM, β here is INVARIANT to
    algebraic reformulation of g — that property is exercised in
    Stage D.5.

    Workflow mirrors :func:`mvfosm`: get D from the deterministic engine,
    build symbolic g, then iterate. Same scenario/spec alignment
    assumption as MVFOSM (verified by Stage D.2 tests).

    Parameters
    ----------
    tol_x, tol_g
        Convergence tolerances on ``‖Δx‖`` and ``|g|`` (both must hold).
    max_iter
        Hard cap on iterations; raises ``RuntimeError`` if exceeded.
    """
    from gabion.deterministic import run_check

    det = run_check(scenario)
    ls = build_limit_states(scenario, d_critical=det.D_critical)

    return {
        "sliding":     _hl_fosm_iterate(ls.sliding,     spec, "sliding",
                                         tol_x, tol_g, max_iter),
        "overturning": _hl_fosm_iterate(ls.overturning, spec, "overturning",
                                         tol_x, tol_g, max_iter),
        "bearing":     _hl_fosm_iterate(ls.bearing,     spec, "bearing",
                                         tol_x, tol_g, max_iter),
    }


# =============================================================================
# FORM (Stage F.2) — Nataf + HL-RF in Y-space
# =============================================================================
def _form_attempt(
    g_func,
    grad_funcs,
    nataf: "NatafTransformation",
    mode_name: str,
    tol_x: float,
    tol_g: float,
    max_iter: int,
    relaxation: float,
) -> tuple:
    """HL-RF in independent-normal Y-space with fixed relaxation.

    Returns ``(y_star, g_at_design, grad_x, grad_y, n_iter)`` on convergence.
    Raises ``RuntimeError`` if ``max_iter`` is exceeded.

    Update rule (Rackwitz–Fiessler in standardized space)::

        y_{k+1} = y_k + r · ([∇_y g · y_k − g_k] / ‖∇_y g‖² · ∇_y g − y_k)

    Y-space gradient via Nataf Jacobian (chain rule)::

        ∇_y g = L_Y^T · diag(φ(z) / f_X(x)) · ∇_x g    (z = L_Y y)
    """
    d = len(nataf.spec.variables)
    y = np.zeros(d)   # y=0 maps to marginal medians — safe interior start
    delta_y = np.inf
    g_k = float("nan")

    for it in range(1, max_iter + 1):
        x = nataf.y_to_x(y)
        g_k = float(g_func(*x))
        grad_x = np.array([float(gf(*x)) for gf in grad_funcs])

        z = nataf.L_Y @ y
        phi_z = norm.pdf(z)
        f_x = np.array([rv.pdf(xi) for rv, xi in zip(nataf.spec.variables, x)])

        if np.any(f_x < 1e-300):
            raise RuntimeError(
                f"FORM[{mode_name}]: marginal PDF ≈ 0 at iteration {it} "
                f"(iterate in deep distribution tail). "
                f"Try a smaller relaxation factor."
            )

        grad_y = nataf.L_Y.T @ ((phi_z / f_x) * grad_x)
        denom = float(grad_y @ grad_y)
        if denom <= 0:
            raise ValueError(
                f"FORM[{mode_name}]: ‖∇_y g‖² = {denom:.3e} ≤ 0 "
                f"at iteration {it}."
            )

        y_target = ((grad_y @ y) - g_k) / denom * grad_y
        y_new = y + relaxation * (y_target - y)
        delta_y = float(np.linalg.norm(y_new - y))
        delta_beta = abs(float(np.linalg.norm(y_new)) - float(np.linalg.norm(y)))
        y = y_new

        # Hybrid criterion: g must be near zero AND either the step in Y-space
        # is small (standard HL-RF convergence) OR β has stabilized (catches
        # modes with large β where the iterate slides along the limit surface
        # at fixed distance without delta_y ever reaching tol_x).
        if abs(g_k) < tol_g and (delta_y < tol_x or delta_beta < tol_x):
            return y, g_k, grad_x, grad_y, it

    raise RuntimeError(
        f"FORM[{mode_name}]: no convergence after {max_iter} iterations "
        f"(relaxation={relaxation}, ‖Δy‖={delta_y:.2e}, |g|={abs(g_k):.2e})."
    )


def _form_iterate(
    g: sp.Expr,
    spec: "RandomVariableSpec",
    nataf: "NatafTransformation",
    mode_name: str,
    tol_x: float,
    tol_g: float,
    max_iter: int,
) -> FosmResult:
    """FORM analysis with automatic relaxation fallback (mirrors _hl_fosm_iterate).

    Runs HL-RF in Y-space using the Nataf-corrected transformation.
    Correctly handles lognormal (φ-variables) and Gumbel (q) marginals
    through the marginal CDF route rather than the Gaussian assumption
    of HL-FOSM. Bearing may fail to converge (same physical singularity
    at the base kern as Stage D.4).
    """
    g_func, grad_funcs = _build_lambdified(g)
    means = spec.means.copy()
    g_at_mean = float(g_func(*means))

    last_error = None
    converged_attempt = None
    for relaxation in (1.0, 0.5, 0.25, 0.1):
        try:
            converged_attempt = _form_attempt(
                g_func, grad_funcs, nataf, mode_name,
                tol_x, tol_g, max_iter, relaxation,
            )
            break
        except RuntimeError as e:
            last_error = e

    if converged_attempt is None:
        nan = float("nan")
        return FosmResult(
            method="FORM",
            mode=mode_name,
            beta=nan,
            pf=nan,
            g_at_mean=g_at_mean,
            sigma_g=nan,
            sensitivity={name: nan for name in SYMBOL_NAMES},
            design_point=None,
            n_iter=-1,
            alpha=None,
            converged=False,
            convergence_note=(
                f"FORM HL-RF (Y-space) did not converge with relaxations "
                f"[1.0, 0.5, 0.25, 0.1]. Last error: {last_error}"
            ),
        )

    y_star, _g, grad_x_star, grad_y_star, iterations = converged_attempt
    norm_grad_y = float(np.linalg.norm(grad_y_star))

    beta = float(np.linalg.norm(y_star))
    if g_at_mean < 0:
        beta = -beta
    pf = float(norm.sf(beta))

    alpha = {
        name: float(grad_y_star[i] / norm_grad_y)
        for i, name in enumerate(SYMBOL_NAMES)
    }
    sensitivity = {
        name: float(grad_x_star[i] * spec.stds[i] / norm_grad_y)
        for i, name in enumerate(SYMBOL_NAMES)
    }
    x_star = nataf.y_to_x(y_star)
    design_point = {name: float(x_star[i]) for i, name in enumerate(SYMBOL_NAMES)}

    return FosmResult(
        method="FORM",
        mode=mode_name,
        beta=beta,
        pf=pf,
        g_at_mean=g_at_mean,
        sigma_g=norm_grad_y,
        sensitivity=sensitivity,
        design_point=design_point,
        n_iter=iterations,
        alpha=alpha,
    )


def form(
    scenario: "WallScenario",
    spec: "RandomVariableSpec",
    tol_x: float = 1e-6,
    tol_g: float = 1e-6,
    max_iter: int = 200,
) -> dict[str, FosmResult]:
    """FORM reliability analysis via Nataf transformation for three failure modes.

    Implements the First-Order Reliability Method (FORM) using the Nataf
    (Gaussian copula) joint distribution model. The HL-RF iteration runs
    in the independent standard-normal Y-space after marginal transformation
    (exact for each distribution family) and Liu–Der Kiureghian correlation
    correction, so lognormal φ-variables and the Gumbel q are handled
    correctly rather than approximated as normals.

    Returns ``dict[str, FosmResult]`` with keys ``"sliding"``,
    ``"overturning"``, ``"bearing"`` and ``method="FORM"`` — same interface
    as :func:`hl_fosm`.
    """
    from gabion.deterministic import run_check
    from gabion.random_variables import NatafTransformation

    det = run_check(scenario)
    ls = build_limit_states(scenario, d_critical=det.D_critical)
    nataf = NatafTransformation.from_spec(spec)

    return {
        "sliding":     _form_iterate(ls.sliding,     spec, nataf, "sliding",
                                      tol_x, tol_g, max_iter),
        "overturning": _form_iterate(ls.overturning, spec, nataf, "overturning",
                                      tol_x, tol_g, max_iter),
        "bearing":     _form_iterate(ls.bearing,     spec, nataf, "bearing",
                                      tol_x, tol_g, max_iter),
    }


# =============================================================================
# Stage D.5 — MVFOSM non-invariance demonstration
# =============================================================================
@dataclass(frozen=True)
class InvarianceComparison:
    """Side-by-side comparison of MVFOSM and HL-FOSM β values across
    three algebraically equivalent forms of the same limit state.

    The three forms (all zero exactly when R = S, hence equivalent
    failure conditions):

        g₁ = R/S − 1    (margin form, the canonical g)
        g₂ = R − S      (difference form)
        g₃ = ln(R/S)    (log-ratio form)

    MVFOSM linearizes at the mean, so its β depends on the algebraic
    form (Cornell's classic non-invariance). HL-FOSM measures geometric
    distance to the surface ``{g = 0}`` in standardized Y-space, which
    is the same surface for all three forms — its β is invariant.
    """
    mode: str
    form_labels: tuple[str, ...]
    mvfosm_betas: tuple[float, ...]
    hl_fosm_betas: tuple[float, ...]
    hl_converged: tuple[bool, ...]


def compare_mvfosm_invariance(
    scenario: "WallScenario",
    spec: "RandomVariableSpec",
    mode: str = "sliding",
) -> InvarianceComparison:
    """Apply MVFOSM and HL-FOSM to the three algebraically equivalent
    forms of a single limit state and report all six β values.

    The pedagogical pay-off: ``mvfosm_betas`` should disagree across
    forms, while ``hl_fosm_betas`` should coincide (modulo iteration
    tolerance). For our scenario, mode ``"bearing"`` will report
    ``hl_converged = (False, False, False)`` — see the Stage D.4
    discussion of why.
    """
    from gabion.deterministic import run_check

    if mode not in ("sliding", "overturning", "bearing"):
        raise ValueError(
            f"mode must be one of 'sliding', 'overturning', 'bearing'; "
            f"got {mode!r}"
        )

    det = run_check(scenario)
    ls = build_limit_states(scenario, d_critical=det.D_critical)

    R = getattr(ls, f"{mode}_R")
    S = getattr(ls, f"{mode}_S")

    forms: dict[str, sp.Expr] = {
        "g1_ratio_minus_1": R / S - 1,
        "g2_difference":    R - S,
        "g3_log_ratio":     sp.log(R / S),
    }

    mv_betas: list[float] = []
    hl_betas: list[float] = []
    hl_converged: list[bool] = []

    for label, g in forms.items():
        # MVFOSM: closed-form, always returns a finite β.
        mv = _mvfosm_at_means(g, spec, f"{mode}[{label}]")
        mv_betas.append(mv.beta)

        # HL-FOSM: iterative; may not converge (returns NaN β with
        # converged=False — preserved here as part of the demonstration).
        hl = _hl_fosm_iterate(
            g, spec, f"{mode}[{label}]",
            tol_x=1e-8, tol_g=1e-8, max_iter=50,
        )
        hl_betas.append(hl.beta)
        hl_converged.append(hl.converged)

    return InvarianceComparison(
        mode=mode,
        form_labels=tuple(forms.keys()),
        mvfosm_betas=tuple(mv_betas),
        hl_fosm_betas=tuple(hl_betas),
        hl_converged=tuple(hl_converged),
    )
