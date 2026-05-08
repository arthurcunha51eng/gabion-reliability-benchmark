"""Tests for ``gabion.monte_carlo`` — Stage E.2.

Test strategy
-------------
Monte Carlo results are stochastic: even with a fixed RNG seed, exact
P_f values for very low-probability events (overturning, bearing) are
hyper-sensitive to which sample happened to land where. Rather than
pinning brittle numerical baselines, we test:

* **Structural properties** — return type, populated fields, shapes.
* **Invariants** — Wilson CI brackets the point estimate; Ditlevsen
  bounds bracket the direct estimate; valid + invalid = total.
* **Reproducibility** — same seed → same result.

A single moderate-sample-size regression test pins the eccentricity
sign-flip count, which is much more numerically stable than P_f at the
RNG seed used.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gabion.inputs import WallScenario
from gabion.monte_carlo import (
    MonteCarloResult,
    _ditlevsen_bounds,
    _wilson_ci,
    run_monte_carlo,
)
from gabion.random_variables import book_example_spec


# =============================================================================
# Module-level fixtures
# =============================================================================
@pytest.fixture(scope="module")
def scenario() -> WallScenario:
    return WallScenario.outside_flat_reference()


@pytest.fixture(scope="module")
def spec():
    return book_example_spec()


@pytest.fixture(scope="module")
def mc_small(scenario, spec):
    """10k-sample run — fast, used for most property tests."""
    return run_monte_carlo(scenario, spec, n_samples=10_000, seed=42)


# =============================================================================
# Wilson CI helper — unit tests independent of MC
# =============================================================================
class TestWilsonCI:
    def test_zero_failures_gives_valid_upper_bound(self):
        """Wilson is non-degenerate at k=0 (unlike normal approximation)."""
        lo, hi = _wilson_ci(0, 1000)
        assert lo == 0.0
        assert 0 < hi < 0.01  # roughly 4/1000 for 95% CI

    def test_n_failures_gives_valid_lower_bound(self):
        lo, hi = _wilson_ci(1000, 1000)
        assert hi == 1.0
        assert 0.99 < lo < 1.0

    def test_brackets_point_estimate(self):
        for k, n in [(5, 100), (50, 1000), (1, 10_000)]:
            lo, hi = _wilson_ci(k, n)
            p_hat = k / n
            assert lo <= p_hat <= hi

    def test_zero_n_returns_full_interval(self):
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0 and hi == 1.0

    def test_alpha_widens_interval(self):
        """Smaller alpha (higher confidence) → wider interval."""
        lo_95, hi_95 = _wilson_ci(50, 100, alpha=0.05)
        lo_99, hi_99 = _wilson_ci(50, 100, alpha=0.01)
        assert (hi_99 - lo_99) > (hi_95 - lo_95)


# =============================================================================
# Ditlevsen bounds helper — unit tests with hand-checkable cases
# =============================================================================
class TestDitlevsenBounds:
    def test_perfectly_independent_modes(self):
        """If ALL P_ij = 0 (impossible for actual events but stress-test):
        lower = Σ P_i ; upper = Σ P_i."""
        p = {"a": 0.10, "b": 0.05, "c": 0.02}
        joint = {("a", "b"): 0.0, ("a", "c"): 0.0, ("b", "c"): 0.0}
        lo, hi = _ditlevsen_bounds(p, joint)
        # Both bounds collapse to the inclusion-exclusion sum upper
        assert lo == pytest.approx(0.17)
        assert hi == pytest.approx(0.17)

    def test_modes_perfectly_dependent(self):
        """If F_2 ⊆ F_1, then P_12 = P_2; lower bound becomes P_1."""
        p = {"a": 0.10, "b": 0.05}
        joint = {("a", "b"): 0.05}
        lo, hi = _ditlevsen_bounds(p, joint)
        # P(F_1 ∪ F_2) = P_1 = 0.10 in this case
        assert lo == pytest.approx(0.10)
        assert hi == pytest.approx(0.10)

    def test_lower_le_upper(self):
        p = {"a": 0.10, "b": 0.05, "c": 0.02}
        joint = {("a", "b"): 0.02, ("a", "c"): 0.01, ("b", "c"): 0.005}
        lo, hi = _ditlevsen_bounds(p, joint)
        assert lo <= hi

    def test_dict_key_order_irrelevant(self):
        """Joint-prob dict can use either tuple ordering — test both."""
        p = {"a": 0.10, "b": 0.05}
        joint_ab = {("a", "b"): 0.02}
        joint_ba = {("b", "a"): 0.02}
        lo_ab, hi_ab = _ditlevsen_bounds(p, joint_ab)
        lo_ba, hi_ba = _ditlevsen_bounds(p, joint_ba)
        assert (lo_ab, hi_ab) == (lo_ba, hi_ba)


# =============================================================================
# Result structure
# =============================================================================
class TestMCStructure:
    def test_returns_MonteCarloResult(self, mc_small):
        assert isinstance(mc_small, MonteCarloResult)

    def test_populated_fields(self, mc_small):
        assert mc_small.n_samples == 10_000
        assert mc_small.seed == 42
        assert set(mc_small.pf_per_mode.keys()) == {
            "sliding", "overturning", "bearing"
        }
        assert set(mc_small.pf_ci_per_mode.keys()) == {
            "sliding", "overturning", "bearing"
        }

    def test_valid_invalid_sum(self, mc_small):
        assert mc_small.n_valid + mc_small.n_invalid == mc_small.n_samples
        assert mc_small.n_valid > 0


# =============================================================================
# Probability invariants
# =============================================================================
class TestProbabilityInvariants:
    def test_pf_in_unit_interval(self, mc_small):
        for pf in mc_small.pf_per_mode.values():
            assert 0.0 <= pf <= 1.0
        assert 0.0 <= mc_small.pf_system <= 1.0

    @pytest.mark.parametrize("mode", ["sliding", "overturning", "bearing"])
    def test_wilson_ci_brackets_pf(self, mc_small, mode):
        pf = mc_small.pf_per_mode[mode]
        lo, hi = mc_small.pf_ci_per_mode[mode]
        assert 0.0 <= lo <= pf <= hi <= 1.0

    def test_system_wilson_ci_brackets_system_pf(self, mc_small):
        lo, hi = mc_small.pf_system_ci
        assert 0.0 <= lo <= mc_small.pf_system <= hi <= 1.0

    def test_joint_pf_le_marginals(self, mc_small):
        """P(F_i ∩ F_j) ≤ min(P_i, P_j) — fundamental probability."""
        for (m1, m2), p_joint in mc_small.pf_joint.items():
            assert p_joint <= mc_small.pf_per_mode[m1]
            assert p_joint <= mc_small.pf_per_mode[m2]


# =============================================================================
# Ditlevsen bounds bracket the direct estimate
# =============================================================================
class TestDitlevsenBracketsMC:
    """The system P_f estimated directly from MC must lie within the
    Ditlevsen bounds — this is the cross-check between the two paths
    of computing system reliability."""

    def test_lower_le_system_le_upper(self, mc_small):
        assert mc_small.pf_ditlevsen_lower <= mc_small.pf_system
        assert mc_small.pf_system <= mc_small.pf_ditlevsen_upper

    def test_lower_le_upper(self, mc_small):
        assert mc_small.pf_ditlevsen_lower <= mc_small.pf_ditlevsen_upper

    def test_bounds_are_finite(self, mc_small):
        assert math.isfinite(mc_small.pf_ditlevsen_lower)
        assert math.isfinite(mc_small.pf_ditlevsen_upper)


# =============================================================================
# Reproducibility
# =============================================================================
class TestReproducibility:
    def test_same_seed_same_result(self, scenario, spec):
        r1 = run_monte_carlo(scenario, spec, n_samples=5_000, seed=123)
        r2 = run_monte_carlo(scenario, spec, n_samples=5_000, seed=123)
        for mode in ("sliding", "overturning", "bearing"):
            assert r1.pf_per_mode[mode] == r2.pf_per_mode[mode]
        assert r1.pf_system == r2.pf_system
        assert r1.n_eccentricity_positive == r2.n_eccentricity_positive

    def test_different_seed_likely_different(self, scenario, spec):
        """Two seeds give different eccentricity-flip counts almost
        surely. Picking the count rather than P_f because counts are
        non-degenerate even when failures are zero."""
        r1 = run_monte_carlo(scenario, spec, n_samples=5_000, seed=42)
        r2 = run_monte_carlo(scenario, spec, n_samples=5_000, seed=999)
        assert r1.n_eccentricity_positive != r2.n_eccentricity_positive


# =============================================================================
# Eccentricity diagnostic — moderately stable vs RNG
# =============================================================================
class TestEccentricityDiagnostic:
    def test_count_in_reasonable_range(self, mc_small):
        """For n=10_000 with seed=42, eccentricity flips happen for
        roughly 0.5-1% of samples (Stage E smoke shows 58/10000)."""
        frac = mc_small.n_eccentricity_positive / mc_small.n_valid
        # Bounds well outside the seed-driven jitter
        assert 0.001 < frac < 0.02, f"eccentricity-positive fraction = {frac:.4f}"

    def test_count_is_nonneg_integer(self, mc_small):
        assert isinstance(mc_small.n_eccentricity_positive, int)
        assert mc_small.n_eccentricity_positive >= 0
        assert mc_small.n_eccentricity_positive <= mc_small.n_valid


# =============================================================================
# Validation: rejected inputs
# =============================================================================
class TestInputValidation:
    def test_zero_n_samples_rejected(self, scenario, spec):
        with pytest.raises(ValueError, match="n_samples must be > 0"):
            run_monte_carlo(scenario, spec, n_samples=0)

    def test_negative_n_samples_rejected(self, scenario, spec):
        with pytest.raises(ValueError, match="n_samples must be > 0"):
            run_monte_carlo(scenario, spec, n_samples=-100)
