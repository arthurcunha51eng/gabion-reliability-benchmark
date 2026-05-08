"""Random variable specifications for the probabilistic stability analysis.

The five quantities treated as uncertain (everything else in the
``WallScenario`` is held deterministic):

==============  ============  =========  ====    ============================
Variable        Distribution     Mean    COV     Source
==============  ============  =========  ====    ============================
phi_backfill    Lognormal     30°        0.10    Phoon & Kulhawy (1999)
gamma_backfill  Normal        18 kN/m³   0.05    Naccache (2016) Tab. 6.2
phi_foundation  Lognormal     30°        0.10    same as phi_backfill
q               Gumbel        10 kN/m²   0.30    live load surcharge
gamma_g         Normal        25 kN/m³   0.03    quarry-stone QC
==============  ============  =========  ====    ============================

Correlation matrix in the X space (the rest are zero):

* ρ(phi_backfill, phi_foundation) = +0.5  — same geological region
* ρ(phi_backfill, gamma_backfill) = +0.3  — denser soil tends to be stronger

This module defines the data structures only. Sampling lives in
``monte_carlo.py`` (Stage E); Nataf transformation lives in ``form.py``
(Stage F). Distribution parameters (lognormal λ/ξ, Gumbel u/α) are
exposed here so both downstream modules consume the same source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log, pi, sqrt
from typing import Literal

import numpy as np
from scipy.stats import gumbel_r, norm


DistributionType = Literal["normal", "lognormal", "gumbel"]
_ALLOWED = ("normal", "lognormal", "gumbel")

# Euler–Mascheroni constant — appears in Gumbel parameter conversion.
_EULER_MASCHERONI = 0.5772156649015329


# =============================================================================
# Single random variable
# =============================================================================
@dataclass(frozen=True, slots=True)
class RandomVariable:
    """A single random variable: distribution family + first two moments.

    The dataclass stores only the *user-facing* parameters: name,
    distribution, mean, and coefficient of variation. Distribution-
    specific parameters (Lognormal λ/ξ, Gumbel u/α) are derived on
    demand via methods.
    """

    name: str
    distribution: DistributionType
    mean: float
    cov: float  # coefficient of variation = std / mean

    def __post_init__(self):
        if self.distribution not in _ALLOWED:
            raise ValueError(
                f"{self.name}: distribution must be one of {_ALLOWED}, "
                f"got {self.distribution!r}"
            )
        if self.cov <= 0:
            raise ValueError(f"{self.name}: cov must be > 0, got {self.cov}")
        # Both Lognormal and Gumbel (as used here for surcharge) require
        # a positive mean — Lognormal is defined only on (0, ∞) and our
        # surcharge-Gumbel parameterization assumes positive intensity.
        if self.distribution in ("lognormal", "gumbel") and self.mean <= 0:
            raise ValueError(
                f"{self.name}: mean must be > 0 for {self.distribution}, "
                f"got {self.mean}"
            )

    # ------------------------------------------------------------------ moments
    @property
    def std(self) -> float:
        return self.mean * self.cov

    @property
    def variance(self) -> float:
        return self.std ** 2

    # --------------------------------------------------------- distribution params
    def lognormal_params(self) -> tuple[float, float]:
        """For Lognormal: return (λ, ξ) such that ln(X) ~ N(λ, ξ²).

        Derivation::

            mean = exp(λ + ξ²/2)
            var  = (exp(ξ²) − 1) · exp(2λ + ξ²)

        Solving:: ξ² = ln(1 + cov²);  λ = ln(mean) − ξ²/2.
        """
        if self.distribution != "lognormal":
            raise ValueError(
                f"{self.name}: lognormal_params() called on a "
                f"{self.distribution} variable"
            )
        xi_sq = log(1.0 + self.cov ** 2)
        xi = sqrt(xi_sq)
        lam = log(self.mean) - xi_sq / 2.0
        return lam, xi

    def gumbel_params(self) -> tuple[float, float]:
        """For Gumbel (maxima): return (u, α) — location and scale parameters.

        Convention used::

            f(x) = α · exp(−α(x−u) − exp(−α(x−u)))
            mean = u + γ_E / α   (γ_E ≈ 0.5772, Euler–Mascheroni)
            var  = π² / (6 α²)

        From these::

            α = π / (sqrt(6) · σ);  u = mean − γ_E / α.

        Note: NumPy's ``rng.gumbel(loc, scale)`` uses ``scale = 1/α``;
        sampling code must convert.
        """
        if self.distribution != "gumbel":
            raise ValueError(
                f"{self.name}: gumbel_params() called on a "
                f"{self.distribution} variable"
            )
        alpha = pi / (sqrt(6.0) * self.std)
        u = self.mean - _EULER_MASCHERONI / alpha
        return u, alpha

    # --------------------------------------------------------- sampling
    def transform_standard_normal(self, z: np.ndarray) -> np.ndarray:
        """Transform standard normal samples into this marginal distribution.

        This is the building block for both iid sampling (Stage E) and
        the Gaussian-copula correlated sampling (also Stage E). For
        Stage F's Nataf transformation, the same transform is applied
        to the correlated standard normals from FORM's Y-space; sharing
        the implementation here keeps Stages E and F numerically
        consistent on the marginal side.

        Mathematical content::

            Normal:    X = mean + std · Z
            Lognormal: X = exp(λ + ξ · Z)             (ln X ~ N(λ, ξ²))
            Gumbel:    X = u − ln(−ln(Φ(Z))) / α      (inverse-CDF route)

        For Gumbel, ``Φ(Z)`` is clipped away from {0, 1} by 1e-15 to
        protect ``gumbel_r.ppf`` from infinities at extreme tails. With
        finite ``z`` this clipping is rarely binding, but matters for
        very large samples.
        """
        z_arr = np.asarray(z, dtype=float)

        if self.distribution == "normal":
            return self.mean + self.std * z_arr

        if self.distribution == "lognormal":
            lam, xi = self.lognormal_params()
            return np.exp(lam + xi * z_arr)

        if self.distribution == "gumbel":
            u, alpha = self.gumbel_params()
            # Φ(z) gives uniform [0, 1]; clip away from {0, 1} for ppf safety.
            p = np.clip(norm.cdf(z_arr), 1e-15, 1.0 - 1e-15)
            return gumbel_r.ppf(p, loc=u, scale=1.0 / alpha)

        raise ValueError(
            f"{self.name}: cannot sample from distribution "
            f"{self.distribution!r}"
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate ``n`` iid samples from this marginal distribution.

        Routed through ``transform_standard_normal`` so the iid path
        and the Cholesky-correlated path of ``RandomVariableSpec``
        share a single implementation — single test target, no risk
        of drift between "rv.sample produces X" and "spec.sample_
        correlated transforms Z exactly the same way".
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        z = rng.standard_normal(n)
        return self.transform_standard_normal(z)

    # --------------------------------------------------------- CDF / PDF
    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Cumulative distribution function ``F_X(x) = P(X ≤ x)``.

        Used by the Nataf transformation (Stage F) to produce the
        equivalent standard-normal value ``Z = Φ⁻¹(F_X(X))`` for each
        marginal. Vectorized: accepts scalar or array input, preserves
        shape on output.
        """
        if self.distribution == "normal":
            return norm.cdf(x, loc=self.mean, scale=self.std)

        if self.distribution == "lognormal":
            lam, xi = self.lognormal_params()
            # F_X(x) = Φ((ln x − λ)/ξ); valid only for x > 0. For x ≤ 0
            # the lognormal puts no mass, so cdf = 0.
            x_arr = np.asarray(x, dtype=float)
            out = np.zeros_like(x_arr)
            positive = x_arr > 0
            out[positive] = norm.cdf(
                np.log(x_arr[positive]), loc=lam, scale=xi
            )
            return out if x_arr.shape else float(out)

        if self.distribution == "gumbel":
            u, alpha = self.gumbel_params()
            return gumbel_r.cdf(x, loc=u, scale=1.0 / alpha)

        raise ValueError(f"{self.name}: cdf not defined for {self.distribution}")

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Probability density function ``f_X(x)``.

        Used by the Nataf transformation to compute the equivalent
        normal standard deviation ``σ_eq = φ(Z) / f_X(X)`` at the
        current iterate.
        """
        if self.distribution == "normal":
            return norm.pdf(x, loc=self.mean, scale=self.std)

        if self.distribution == "lognormal":
            lam, xi = self.lognormal_params()
            # f_X(x) = (1/x) · φ((ln x − λ)/ξ) / ξ; zero for x ≤ 0.
            x_arr = np.asarray(x, dtype=float)
            out = np.zeros_like(x_arr)
            positive = x_arr > 0
            out[positive] = (
                norm.pdf(np.log(x_arr[positive]), loc=lam, scale=xi)
                / x_arr[positive]
            )
            return out if x_arr.shape else float(out)

        if self.distribution == "gumbel":
            u, alpha = self.gumbel_params()
            return gumbel_r.pdf(x, loc=u, scale=1.0 / alpha)

        raise ValueError(f"{self.name}: pdf not defined for {self.distribution}")


# =============================================================================
# Specification = collection of variables + correlation matrix
# =============================================================================
@dataclass(frozen=True, slots=True)
class RandomVariableSpec:
    """A complete probabilistic specification: variables + correlation matrix.

    The correlation matrix lives in the X space (the original variable
    space, before any transformation). It is reused by FOSM (Stage D)
    via the covariance matrix C_X, by Monte Carlo (Stage E) via Cholesky
    decomposition, and by FORM (Stage F) via Nataf transformation.
    """

    variables: tuple[RandomVariable, ...]
    correlation_matrix: np.ndarray = field(repr=False)

    def __post_init__(self):
        n = len(self.variables)
        if n == 0:
            raise ValueError("Spec must contain at least one variable.")

        # Names must be unique — otherwise index_of() and downstream
        # name-based access become ambiguous.
        names = [v.name for v in self.variables]
        if len(set(names)) != n:
            raise ValueError(f"Duplicate variable names in spec: {names}")

        rho = self.correlation_matrix
        if not isinstance(rho, np.ndarray):
            raise TypeError(
                f"correlation_matrix must be a NumPy array, got {type(rho).__name__}"
            )
        if rho.shape != (n, n):
            raise ValueError(
                f"correlation_matrix must be {n}×{n}, got shape {rho.shape}"
            )
        if not np.allclose(rho, rho.T, atol=1e-12):
            raise ValueError("correlation_matrix is not symmetric.")
        if not np.allclose(np.diag(rho), 1.0, atol=1e-12):
            raise ValueError("correlation_matrix diagonal must be all 1.0.")
        # Off-diagonals must lie in [-1, 1] for a valid correlation matrix.
        off = rho - np.eye(n)
        if (off > 1.0).any() or (off < -1.0).any():
            raise ValueError(
                "correlation_matrix off-diagonals must lie in [−1, 1]."
            )
        # Positive-definiteness — required for Cholesky downstream. We
        # use np.linalg.cholesky as the test because that is exactly the
        # operation Stage E will perform; if it fails here, fail loudly
        # at construction rather than mid-Monte Carlo.
        try:
            np.linalg.cholesky(rho)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"correlation_matrix is not positive-definite: {e}"
            ) from e

    # --------------------------------------------------------- shape / lookup
    @property
    def n(self) -> int:
        return len(self.variables)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables)

    def index_of(self, name: str) -> int:
        for i, v in enumerate(self.variables):
            if v.name == name:
                return i
        raise KeyError(f"No variable named {name!r} in spec; have {self.names}")

    def __getitem__(self, name: str) -> RandomVariable:
        return self.variables[self.index_of(name)]

    # --------------------------------------------------------- moment vectors
    @property
    def means(self) -> np.ndarray:
        return np.array([v.mean for v in self.variables], dtype=float)

    @property
    def stds(self) -> np.ndarray:
        return np.array([v.std for v in self.variables], dtype=float)

    def covariance_matrix(self) -> np.ndarray:
        """C_X = D · ρ · D, where D = diag(stds)."""
        D = np.diag(self.stds)
        return D @ self.correlation_matrix @ D

    # --------------------------------------------------------- sampling
    def sample_correlated(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample ``n`` rows of correlated values; returns shape ``(n, d)``.

        Pipeline (Gaussian copula via Cholesky):

        1. Sample iid standard normals ``Z`` with shape ``(n, d)``.
        2. Cholesky-correlate: ``Z_corr = Z @ Lᵀ`` where ``L Lᵀ = ρ``.
           Result: ``Z_corr`` rows are ``N(0, ρ)``-distributed (each
           row is a sample from the standard-normal-with-correlation-ρ
           distribution).
        3. Transform each marginal independently via inverse-CDF::

               X[:, i] = F_i^{-1}(Φ(Z_corr[:, i]))

           For Normal marginals this collapses to ``X = μ + σ · Z_corr``;
           for Lognormal and Gumbel the inverse-CDF differs.

        Correlation note
        ----------------
        After step 3, the X-space correlation is approximately but not
        exactly ``ρ`` — a known property of the Gaussian copula when
        marginals are non-Normal. For the COVs in this project (≤ 0.30),
        the discrepancy is small (~1–3% relative). Stage F's Nataf
        transformation solves for the equivalent normal-space
        correlation that yields ``ρ`` exactly in X-space.
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")

        d = self.n
        Z = rng.standard_normal((n, d))
        L = np.linalg.cholesky(self.correlation_matrix)
        Z_corr = Z @ L.T  # shape (n, d), correlated standard normals

        X = np.empty_like(Z_corr)
        for i, rv in enumerate(self.variables):
            X[:, i] = rv.transform_standard_normal(Z_corr[:, i])
        return X


# =============================================================================
# Canonical example
# =============================================================================
def book_example_spec() -> RandomVariableSpec:
    """The five-variable spec agreed for the gabion-stability portfolio.

    Variables and parameters are documented in the module docstring.
    The same spec is consumed by Stages D (FOSM), E (Monte Carlo), and
    F (FORM) so that all probabilistic results are directly comparable.
    """
    variables = (
        RandomVariable("phi_backfill",   "lognormal", mean=30.0, cov=0.10),
        RandomVariable("gamma_backfill", "normal",    mean=18.0, cov=0.05),
        RandomVariable("phi_foundation", "lognormal", mean=30.0, cov=0.10),
        RandomVariable("q",              "gumbel",    mean=10.0, cov=0.30),
        RandomVariable("gamma_g",        "normal",    mean=25.0, cov=0.03),
    )

    # Build the correlation matrix using index lookup so the code stays
    # robust to future reordering of `variables`.
    rho = np.eye(5)
    idx = {v.name: i for i, v in enumerate(variables)}
    rho[idx["phi_backfill"], idx["phi_foundation"]] = 0.5
    rho[idx["phi_foundation"], idx["phi_backfill"]] = 0.5
    rho[idx["phi_backfill"], idx["gamma_backfill"]] = 0.3
    rho[idx["gamma_backfill"], idx["phi_backfill"]] = 0.3

    return RandomVariableSpec(variables=variables, correlation_matrix=rho)


# =============================================================================
# Stage F.1 — Nataf transformation
# =============================================================================
def _nataf_factor(rv_i: RandomVariable, rv_j: RandomVariable, rho_X: float) -> float:
    """Liu & Der Kiureghian (1986) factor F such that ρ_Y = F · ρ_X
    yields the requested ρ_X in X-space after the Nataf transformation.

    Closed-form expressions for the supported distribution pairs:

    * Normal–Normal: F = 1 (no distortion).
    * Lognormal–Lognormal (exact)::

          ρ_X = [exp(ρ_Y · ξ_i · ξ_j) − 1]
              / sqrt[(exp(ξ_i²) − 1)(exp(ξ_j²) − 1)]

      Solved for ρ_Y, then F = ρ_Y / ρ_X.
    * Lognormal–Normal (Liu-DK)::

          F = cov_LN / sqrt(ln(1 + cov_LN²)) = cov_LN / ξ_LN

    Other pairs raise ``NotImplementedError``. For the canonical
    spec used in this project, only the two pairs above appear with
    non-zero ρ_X (others are independent), so no extension is needed.
    """
    types = (rv_i.distribution, rv_j.distribution)
    pair = tuple(sorted(types))

    if pair == ("normal", "normal"):
        return 1.0

    if pair == ("lognormal", "lognormal"):
        _, xi_i = rv_i.lognormal_params()
        _, xi_j = rv_j.lognormal_params()
        # Exact closed-form: solve for ρ_Y from ρ_X.
        rho_Y = np.log(
            1.0
            + rho_X * np.sqrt((np.exp(xi_i ** 2) - 1) * (np.exp(xi_j ** 2) - 1))
        ) / (xi_i * xi_j)
        return float(rho_Y / rho_X)

    if pair == ("lognormal", "normal"):
        # Liu-DK closed form. F depends only on the lognormal's COV.
        rv_ln = rv_i if rv_i.distribution == "lognormal" else rv_j
        cov_ln = rv_ln.cov
        _, xi_ln = rv_ln.lognormal_params()
        return float(cov_ln / xi_ln)

    raise NotImplementedError(
        f"Nataf factor for ({types[0]}, {types[1]}) not implemented. "
        f"Supported pairs: Normal–Normal, Lognormal–Lognormal, "
        f"Lognormal–Normal. Add a new branch in _nataf_factor() if "
        f"a future spec needs another pair."
    )


def _compute_nataf_correlation(spec: RandomVariableSpec) -> np.ndarray:
    """Compute the Y-space correlation matrix ρ_Y that yields the
    spec's X-space correlation ρ_X after the Nataf transformation.

    Diagonal entries are 1 (unit variance after standardization).
    Off-diagonal entries use ``_nataf_factor`` on a pair-by-pair
    basis. ρ_Y = ρ_X for any pair with ρ_X = 0 — no distortion to
    propagate.
    """
    n = spec.n
    rho_X = spec.correlation_matrix
    rho_Y = np.eye(n)

    for i in range(n):
        for j in range(i):
            if rho_X[i, j] == 0:
                continue  # independent stays independent
            factor = _nataf_factor(
                spec.variables[i], spec.variables[j], rho_X[i, j]
            )
            rho_Y[i, j] = rho_X[i, j] * factor
            rho_Y[j, i] = rho_Y[i, j]

    return rho_Y


@dataclass(frozen=True, slots=True)
class NatafTransformation:
    """Nataf transformation between original X-space (with arbitrary
    marginals and a specified correlation matrix) and a standardized,
    independent Y-space (Y ~ N(0, I_d)).

    The intermediate Z-space holds correlated standard normals with
    correlation matrix ``ρ_Y`` (the Liu-Der Kiureghian-corrected
    version of ρ_X); ``L_Y`` is its Cholesky factor.

    Pipelines used by :func:`form` (Stage F.2)::

        X-space → Z-space → Y-space  (backward, "x_to_y"):
            z_i = Φ⁻¹(F_X_i(x_i))    (marginal-by-marginal)
            y    = L_Y⁻¹ z

        Y-space → Z-space → X-space  (forward, "y_to_x"):
            z = L_Y y
            x_i = F_X_i⁻¹(Φ(z_i))    (marginal-by-marginal)

    Both transforms accept either a single point of shape ``(d,)``
    or a batch of shape ``(n, d)``; output shape mirrors input.
    """
    spec: RandomVariableSpec
    rho_Y: np.ndarray
    L_Y: np.ndarray

    @classmethod
    def from_spec(cls, spec: RandomVariableSpec) -> "NatafTransformation":
        """Build the transformation by computing ρ_Y and its Cholesky."""
        rho_Y = _compute_nataf_correlation(spec)
        try:
            L_Y = np.linalg.cholesky(rho_Y)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"Computed Nataf correlation matrix is not positive-"
                f"definite. This usually indicates an inconsistent ρ_X "
                f"in the spec (or a corner case the closed-form factors "
                f"don't capture). Original error: {e}"
            ) from e
        return cls(spec=spec, rho_Y=rho_Y, L_Y=L_Y)

    # ------------------------------------------------------ transforms
    def x_to_y(self, x: np.ndarray) -> np.ndarray:
        """Map X-space to standardized Y-space (independent N(0, I))."""
        x = np.asarray(x, dtype=float)
        single = (x.ndim == 1)
        x2 = x[np.newaxis, :] if single else x  # (n, d)

        # Stage 1: marginal-by-marginal, X → Z (correlated standard normals).
        z = np.empty_like(x2)
        for i, rv in enumerate(self.spec.variables):
            # Φ⁻¹(F_X_i(x_i))
            z[:, i] = norm.ppf(rv.cdf(x2[:, i]))

        # Stage 2: decorrelate via L_Y⁻¹. Solve Lz = y for each row;
        # equivalent to y = L_Y⁻¹ z but uses backsubstitution.
        # Since L_Y is lower triangular, np.linalg.solve_triangular
        # would be faster, but np.linalg.solve is more readable.
        y = np.linalg.solve(self.L_Y, z.T).T
        return y[0] if single else y

    def y_to_x(self, y: np.ndarray) -> np.ndarray:
        """Map standardized Y-space back to X-space."""
        y = np.asarray(y, dtype=float)
        single = (y.ndim == 1)
        y2 = y[np.newaxis, :] if single else y  # (n, d)

        # Stage 1: correlate via L_Y to get Z-space.
        z = (self.L_Y @ y2.T).T

        # Stage 2: marginal-by-marginal, Z → X via inverse CDF (which
        # we already have as transform_standard_normal — it's exactly
        # F_X_i⁻¹(Φ(Z_i)) routed through the marginal's parameters).
        x = np.empty_like(z)
        for i, rv in enumerate(self.spec.variables):
            x[:, i] = rv.transform_standard_normal(z[:, i])
        return x[0] if single else x
