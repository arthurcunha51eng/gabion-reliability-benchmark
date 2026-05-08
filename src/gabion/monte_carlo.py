"""Monte Carlo reliability analysis for the gabion-stability project.

Stage E delivers a vectorized Monte Carlo over the same five-variable
spec used by the FOSM stages, with three additions over Stage D:

* **True marginal distributions.** Stage D treated all variables as
  Normal (Naccache §5.5.1 convention). Stage E samples from the
  declared marginals (Lognormal for friction angles, Gumbel for
  surcharge, Normal for unit weights) using an inverse-CDF transform
  on standard normals.
* **Cholesky-correlated samples.** The Gaussian copula approach: sample
  iid standard normals, Cholesky-correlate them, then transform each
  marginal independently. The X-space correlation matches the requested
  ρ to within copula distortion (small for our COVs ≤ 0.30; quantified
  by the empirical-correlation tests in `test_random_variables.py`).
* **System reliability.** Both the direct estimate ``P(F_1 ∪ F_2 ∪
  F_3) = (1/n) Σ I(any g_i < 0)`` and Ditlevsen's bi-modal bounds
  using only marginal and pairwise joint probabilities. Bounds should
  bracket the direct estimate.

Confidence intervals on each P_f are reported as Wilson score
intervals — preferred over normal-approximation intervals for small
P, which is our regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from scipy.stats import norm

from gabion.fosm import SYMBOLS, build_limit_states

if TYPE_CHECKING:
    from gabion.inputs import WallScenario
    from gabion.random_variables import RandomVariableSpec


# Modes in canonical order — used everywhere the dict layout doesn't
# already pin it. Keep aligned with build_limit_states' fields.
_MODES: tuple[str, ...] = ("sliding", "overturning", "bearing")


# =============================================================================
# Result container
# =============================================================================
@dataclass(frozen=True)
class MonteCarloResult:
    """Summary of a Monte Carlo reliability analysis.

    All probabilities are estimated from valid samples only — invalid
    samples (NaN or inf produced by the marginal transforms or the
    limit-state formulas) are excluded and counted in ``n_invalid``.
    """
    # ----- Setup -----
    n_samples: int        # Requested sample count
    n_valid: int          # Samples with finite g for all three modes
    n_invalid: int        # = n_samples − n_valid
    seed: int | None

    # ----- Per-mode P_f and Wilson 95% CIs -----
    pf_per_mode: dict[str, float]
    pf_ci_per_mode: dict[str, tuple[float, float]]

    # ----- System P_f via direct simulation (gold standard) -----
    pf_system: float
    pf_system_ci: tuple[float, float]

    # ----- Ditlevsen bi-modal bounds on system P_f -----
    pf_ditlevsen_lower: float
    pf_ditlevsen_upper: float

    # ----- Pairwise joint failure probabilities -----
    # Keys are sorted-tuple ("mode_a", "mode_b") with mode_a < mode_b
    # lexicographically — convention for stable dict access.
    pf_joint: dict[tuple[str, str], float]

    # ----- Diagnostic: bearing eccentricity sign-flip count -----
    # Number of samples where e > 0 (assumption ``|e| = -e`` violated).
    # Mean eccentricity is -0.131 m for the canonical scenario; if this
    # count exceeds a few percent of n_valid, the bearing limit-state
    # formula needs to be revisited.
    n_eccentricity_positive: int


# =============================================================================
# Helper: Wilson score confidence interval
# =============================================================================
def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score CI for binomial proportion ``p = k / n``.

    Preferred over the normal-approximation interval for small ``p``,
    which is our regime when reliability is high. For ``k = 0`` the
    Wilson interval is ``(0, z² / (n + z²))`` — non-degenerate even
    when no failures are observed, which is exactly what we want for
    interpreting "n samples, 0 failures, what can we say about P_f?".

    At the boundaries ``k = 0`` and ``k = n``, the analytic formula
    gives ``lo = 0`` and ``hi = 1`` respectively. In double precision
    the computed ``center − half`` may produce ~1e-19 instead of 0,
    so we short-circuit those endpoints to keep the contract clean.

    Reference: Wilson (1927); Brown, Cai, DasGupta (2001).
    """
    if n <= 0:
        return 0.0, 1.0
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    # Boundary cases are exact in analytic arithmetic; force exactness
    # in floating-point too so the test contract stays clean.
    lo = 0.0 if k == 0 else float(max(0.0, center - half))
    hi = 1.0 if k == n else float(min(1.0, center + half))
    return lo, hi


# =============================================================================
# Helper: Ditlevsen bi-modal bounds
# =============================================================================
def _ditlevsen_bounds(
    p_per_mode: dict[str, float],
    p_joint: dict[tuple[str, str], float],
) -> tuple[float, float]:
    """Ditlevsen bi-modal bounds on series-system probability of failure.

    Uses only marginal P_i and pairwise joint P_ij to bracket
    ``P(F_1 ∪ … ∪ F_n)``. Bounds are tightest when modes are ordered
    by decreasing P_i; this function does the ordering internally.

    Lower bound (Ditlevsen 1979)::

        P_F ≥ P_1 + Σ_{i=2}^{n} max(0, P_i − Σ_{j<i} P_ij)

    Upper bound::

        P_F ≤ P_1 + Σ_{i=2}^{n} (P_i − max_{j<i} P_ij)

    Equivalent to Naccache (2016) eq. 5.99 / 5.113–5.116.

    The pairwise joint dict can use either ordering of the tuple keys
    (``(a, b)`` or ``(b, a)``); this function tries both.
    """
    ordered = sorted(p_per_mode.keys(), key=lambda m: -p_per_mode[m])
    P = [p_per_mode[m] for m in ordered]
    n = len(ordered)

    def _Pij(i: int, j: int) -> float:
        m_i, m_j = ordered[i], ordered[j]
        key = (m_i, m_j) if (m_i, m_j) in p_joint else (m_j, m_i)
        return p_joint[key]

    lower = P[0]
    for i in range(1, n):
        sum_pij = sum(_Pij(i, j) for j in range(i))
        lower += max(0.0, P[i] - sum_pij)

    upper = P[0]
    for i in range(1, n):
        max_pij = max(_Pij(i, j) for j in range(i)) if i > 0 else 0.0
        upper += P[i] - max_pij

    return float(lower), float(upper)


# =============================================================================
# Public entry point
# =============================================================================
def run_monte_carlo(
    scenario: "WallScenario",
    spec: "RandomVariableSpec",
    n_samples: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run a Monte Carlo reliability analysis with Cholesky-correlated,
    mixed-marginal samples.

    Pipeline:
        1. Run the deterministic engine at the scenario's nominal values
           to fix the critical wedge ``D``. The same ``D`` is used for
           all MC samples — i.e., we do not re-do the wedge search per
           sample. (The agreed assumption from Stage D; Stage E does
           NOT yet count samples for which the optimal D would shift.)
        2. Build the symbolic limit-state functions for that D and
           lambdify them for vectorized evaluation.
        3. Sample correlated X using ``spec.sample_correlated`` (Stage
           E.1 infrastructure).
        4. Evaluate g_sliding, g_overturning, g_bearing on all samples;
           also evaluate the signed eccentricity for the ``|e| = -e``
           diagnostic.
        5. Aggregate: per-mode P_f with Wilson CIs, pairwise joint P_ij,
           direct system P_f, Ditlevsen bounds, and the eccentricity
           sign-flip count.

    Parameters
    ----------
    n_samples
        Number of MC samples. Default 100_000 gives ~ 0.3% std error on
        a ``P_f = 1e-3`` estimate. For ``P_f`` orders of magnitude
        smaller (overturning, bearing), expect zero failures and rely
        on the Wilson upper bound as the informative quantity.
    seed
        RNG seed for reproducibility. Pass ``None`` for OS entropy.
    """
    # Local import of run_check to keep monte_carlo.py's parse-time
    # import surface light (run_check pulls the whole engine chain).
    from gabion.deterministic import run_check

    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")

    # ----- Build symbolic g and lambdify -----
    det = run_check(scenario)
    ls = build_limit_states(scenario, d_critical=det.D_critical)

    g_funcs = {
        mode: sp.lambdify(SYMBOLS, getattr(ls, mode), modules="numpy")
        for mode in _MODES
    }
    e_func = sp.lambdify(SYMBOLS, ls.eccentricity, modules="numpy")

    # ----- Sample -----
    rng = np.random.default_rng(seed)
    X = spec.sample_correlated(n_samples, rng)  # shape (n, 5)

    # ----- Vectorized evaluation -----
    # X.T has shape (5, n); unpacking it gives 5 column-arrays as
    # positional arguments to the lambdified function.
    g_values = {
        mode: np.asarray(func(*X.T), dtype=float)
        for mode, func in g_funcs.items()
    }
    e_values = np.asarray(e_func(*X.T), dtype=float)

    # ----- Validity (NaN/inf in any mode → drop sample) -----
    valid = np.ones(n_samples, dtype=bool)
    for g in g_values.values():
        valid &= np.isfinite(g)
    valid &= np.isfinite(e_values)
    n_invalid = int((~valid).sum())
    n_valid = n_samples - n_invalid
    if n_valid == 0:
        raise RuntimeError(
            "All Monte Carlo samples produced invalid g — check the "
            "marginal transforms and the limit-state expressions."
        )

    # ----- Eccentricity diagnostic -----
    n_e_positive = int(((e_values > 0) & valid).sum())

    # ----- Failures per mode (valid samples only) -----
    failures = {
        mode: (g_values[mode] < 0) & valid for mode in _MODES
    }

    pf_per_mode = {
        mode: float(failures[mode].sum() / n_valid)
        for mode in _MODES
    }
    pf_ci_per_mode = {
        mode: _wilson_ci(int(failures[mode].sum()), n_valid)
        for mode in _MODES
    }

    # ----- Pairwise joint failures -----
    # Keys sorted lexicographically for stable access regardless of
    # iteration order.
    pf_joint: dict[tuple[str, str], float] = {}
    for i, m1 in enumerate(_MODES):
        for m2 in _MODES[i + 1:]:
            key = tuple(sorted((m1, m2)))
            pf_joint[key] = float((failures[m1] & failures[m2]).sum() / n_valid)

    # ----- Direct system P_f -----
    any_fail = failures["sliding"] | failures["overturning"] | failures["bearing"]
    n_any = int(any_fail.sum())
    pf_system = n_any / n_valid
    pf_system_ci = _wilson_ci(n_any, n_valid)

    # ----- Ditlevsen bi-modal bounds -----
    pf_lo, pf_hi = _ditlevsen_bounds(pf_per_mode, pf_joint)

    return MonteCarloResult(
        n_samples=n_samples,
        n_valid=n_valid,
        n_invalid=n_invalid,
        seed=seed,
        pf_per_mode=pf_per_mode,
        pf_ci_per_mode=pf_ci_per_mode,
        pf_system=pf_system,
        pf_system_ci=pf_system_ci,
        pf_ditlevsen_lower=pf_lo,
        pf_ditlevsen_upper=pf_hi,
        pf_joint=pf_joint,
        n_eccentricity_positive=n_e_positive,
    )
