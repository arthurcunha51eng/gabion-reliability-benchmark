"""Tests for ``gabion.fosm`` — Stage D.2 (symbolic limit states only).

The only behavioral guarantee at this stage is *cross-validation*: the
SymPy expression evaluated at the mean of X must agree with the
deterministic engine's FS to better than 1e-9 relative tolerance.

Stages D.3 (MVFOSM) and D.4 (HL-FOSM) will add their own tests on top.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
from scipy.stats import norm

from gabion.deterministic import run_check
from gabion.fosm import (
    GAMMA_BACKFILL,
    GAMMA_G,
    PHI_BACKFILL,
    PHI_FOUNDATION,
    Q,
    SYMBOLS,
    SYMBOL_NAMES,
    FosmResult,
    InvarianceComparison,
    LimitStates,
    build_limit_states,
    compare_mvfosm_invariance,
    evaluate_at_means,
    form,
    hl_fosm,
    mvfosm,
)
from gabion.inputs import WallScenario
from gabion.random_variables import book_example_spec


# =============================================================================
# Module-level fixtures — built once, reused across tests
# =============================================================================
@pytest.fixture(scope="module")
def scenario() -> WallScenario:
    return WallScenario.outside_flat_reference()


@pytest.fixture(scope="module")
def deterministic(scenario):
    """Run the deterministic engine once; D_critical is the wedge we
    bake into the symbolic expressions."""
    return run_check(scenario)


@pytest.fixture(scope="module")
def spec():
    """The 5-variable spec; means must coincide with the scenario's
    deterministic values for the cross-validation to be meaningful."""
    return book_example_spec()


@pytest.fixture(scope="module")
def limit_states(scenario, deterministic) -> LimitStates:
    return build_limit_states(scenario, d_critical=deterministic.D_critical)


# =============================================================================
# Symbol-list invariants
# =============================================================================
class TestSymbolMetadata:
    def test_symbols_and_names_aligned(self):
        # Names must match the order of SYMBOLS so that gradient vectors
        # and design-point dicts stay coherent across stages.
        assert len(SYMBOLS) == len(SYMBOL_NAMES) == 5
        for sym, name in zip(SYMBOLS, SYMBOL_NAMES):
            assert sym.name == name

    def test_symbol_names_match_book_spec(self):
        spec = book_example_spec()
        assert set(SYMBOL_NAMES) == set(spec.names)


# =============================================================================
# Spec means align with scenario deterministic values (precondition)
# =============================================================================
class TestSpecScenarioAlignment:
    """If these break, the cross-validation that follows is meaningless."""

    def test_spec_means_match_scenario(self, spec, scenario):
        assert spec["phi_backfill"].mean   == scenario.backfill.phi
        assert spec["gamma_backfill"].mean == scenario.backfill.gamma
        assert spec["phi_foundation"].mean == scenario.foundation.phi
        assert spec["q"].mean              == scenario.q
        assert spec["gamma_g"].mean        == scenario.gabion.gamma_g


# =============================================================================
# CROSS-VALIDATION — the sole behavioral guarantee at Stage D.2
# =============================================================================
class TestCrossValidationAtMeans:
    """Each symbolic g(X) at the mean of X must reproduce the
    deterministic engine's FS - 1 (or q_adm/sigma_max - 1 for bearing)."""

    def test_g_sliding_at_means(self, limit_states, spec, deterministic):
        actual = evaluate_at_means(limit_states.sliding, spec)
        expected = deterministic.FS_sliding - 1.0
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_g_overturning_at_means(self, limit_states, spec, deterministic):
        actual = evaluate_at_means(limit_states.overturning, spec)
        expected = deterministic.FS_overturning - 1.0
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_g_bearing_at_means(self, limit_states, spec, deterministic):
        actual = evaluate_at_means(limit_states.bearing, spec)
        # The bearing limit state is q_adm/sigma_max - 1.
        expected_FS = 200.0 / deterministic.sigma_max  # q_adm hard-coded in scenario
        expected = expected_FS - 1.0
        assert actual == pytest.approx(expected, rel=1e-9)


# =============================================================================
# Free-symbol structure — encodes the physics of which RV affects which mode
# =============================================================================
class TestFreeSymbols:
    """phi_foundation enters only the sliding mode (it is the base
    friction angle). Overturning and bearing don't depend on it.
    This makes the importance factor α_phi_foundation exactly zero for
    the latter two modes — a useful guard against future regressions."""

    def test_sliding_uses_all_five_variables(self, limit_states):
        free = limit_states.sliding.free_symbols
        for sym in SYMBOLS:
            assert sym in free, f"Sliding should depend on {sym}"

    def test_overturning_does_not_use_phi_foundation(self, limit_states):
        free = limit_states.overturning.free_symbols
        assert PHI_FOUNDATION not in free
        # But all four others must appear
        for sym in (PHI_BACKFILL, GAMMA_BACKFILL, Q, GAMMA_G):
            assert sym in free

    def test_bearing_does_not_use_phi_foundation(self, limit_states):
        free = limit_states.bearing.free_symbols
        assert PHI_FOUNDATION not in free
        for sym in (PHI_BACKFILL, GAMMA_BACKFILL, Q, GAMMA_G):
            assert sym in free


# =============================================================================
# Sanity: g > 0 at the mean (the wall is safe in all three modes)
# =============================================================================
class TestSafeAtMeans:
    """Reference scenario is comfortably safe (FS ≈ 2.5 sliding,
    5.4 overturning, 3.1 bearing) — so g should be positive in every mode."""

    def test_all_g_positive_at_means(self, limit_states, spec):
        for name in ("sliding", "overturning", "bearing"):
            g_val = evaluate_at_means(getattr(limit_states, name), spec)
            assert g_val > 0, f"g_{name} = {g_val} ≤ 0 at means"


# =============================================================================
# Stage D.3 — MVFOSM (Cornell)
# =============================================================================
@pytest.fixture(scope="module")
def mvfosm_results(scenario, spec):
    """Run MVFOSM once for all three modes; reuse across tests."""
    return mvfosm(scenario, spec)


class TestMVFOSMShape:
    def test_returns_three_modes(self, mvfosm_results):
        assert set(mvfosm_results.keys()) == {"sliding", "overturning", "bearing"}

    def test_each_result_is_FosmResult(self, mvfosm_results):
        for r in mvfosm_results.values():
            assert isinstance(r, FosmResult)
            assert r.method == "MVFOSM"
            assert r.design_point is None
            assert r.n_iter is None

    def test_sensitivity_keys_match_spec(self, mvfosm_results):
        for r in mvfosm_results.values():
            assert set(r.sensitivity.keys()) == set(SYMBOL_NAMES)


class TestMVFOSMNumericalConsistency:
    """β > 0 (we are in the safe region), pf < 0.5, and pf = Φ(-β) exactly."""

    @pytest.mark.parametrize("mode", ["sliding", "overturning", "bearing"])
    def test_beta_positive(self, mvfosm_results, mode):
        assert mvfosm_results[mode].beta > 0

    @pytest.mark.parametrize("mode", ["sliding", "overturning", "bearing"])
    def test_pf_matches_phi_minus_beta(self, mvfosm_results, mode):
        r = mvfosm_results[mode]
        assert r.pf == pytest.approx(float(norm.sf(r.beta)), rel=1e-12)

    @pytest.mark.parametrize("mode", ["sliding", "overturning", "bearing"])
    def test_beta_equals_mu_g_over_sigma_g(self, mvfosm_results, mode):
        r = mvfosm_results[mode]
        assert r.beta == pytest.approx(r.g_at_mean / r.sigma_g, rel=1e-12)

    @pytest.mark.parametrize("mode", ["sliding", "overturning", "bearing"])
    def test_sigma_g_via_cholesky_path(self, mvfosm_results, mode, spec, limit_states):
        """σ_g computed via direct quadratic form must equal ||Lᵀ ∇g||
        where L Lᵀ = C_X. Cross-check between the MVFOSM internal path
        and the alternative Cholesky path that HL-FOSM will use."""
        g = getattr(limit_states, mode)
        means = {sym: spec[name].mean for sym, name in zip(SYMBOLS, SYMBOL_NAMES)}
        grad = np.array([float(sp.diff(g, sym).subs(means)) for sym in SYMBOLS])
        L = np.linalg.cholesky(spec.covariance_matrix())
        sigma_g_chol = float(np.linalg.norm(L.T @ grad))
        assert mvfosm_results[mode].sigma_g == pytest.approx(sigma_g_chol, rel=1e-12)


class TestMVFOSMPhysicalSigns:
    """Sensitivity signs must match physical intuition.

    Sliding (FS = T_resist / T_drive):
      - phi_backfill ↑ → less active pressure → FS ↑   → γ > 0
      - gamma_backfill ↑ → more active pressure → FS ↓ → γ < 0
      - phi_foundation ↑ → more base friction → FS ↑   → γ > 0
      - q ↑ → more surcharge thrust → FS ↓             → γ < 0
      - gamma_g ↑ → heavier wall, more N → FS ↑        → γ > 0
    """

    def test_sliding_signs(self, mvfosm_results):
        s = mvfosm_results["sliding"].sensitivity
        assert s["phi_backfill"]   > 0
        assert s["gamma_backfill"] < 0
        assert s["phi_foundation"] > 0
        assert s["q"]              < 0
        assert s["gamma_g"]        > 0

    def test_overturning_signs_and_zeros(self, mvfosm_results):
        s = mvfosm_results["overturning"].sensitivity
        # phi_foundation does not enter overturning → exactly 0
        assert s["phi_foundation"] == 0.0
        # Loads should still be negative, resistances positive
        assert s["gamma_backfill"] < 0
        assert s["q"]              < 0
        assert s["gamma_g"]        > 0  # heavier wall → more resisting moment

    def test_bearing_signs_and_zeros(self, mvfosm_results):
        s = mvfosm_results["bearing"].sensitivity
        assert s["phi_foundation"] == 0.0  # not in g_bearing


class TestMVFOSMReliabilityValues:
    """Regression baselines for β and Pf — locked once values are
    inspected. Future refactors can't silently change the answers."""

    # Values established by sandbox smoke run on 2025-XX-XX with the
    # canonical scenario + book_example_spec. Tolerance set to 1e-6
    # to leave headroom for harmless numerical reordering.

    EXPECTED_BETA = {
        "sliding":     2.69612,
        "overturning": 4.90839,
        "bearing":     7.07638,
    }

    @pytest.mark.parametrize("mode,expected", EXPECTED_BETA.items())
    def test_beta_baseline(self, mvfosm_results, mode, expected):
        assert mvfosm_results[mode].beta == pytest.approx(expected, rel=1e-6)


# =============================================================================
# Stage D.4 — HL-FOSM (Hasofer-Lind via HLRF iteration)
# =============================================================================
@pytest.fixture(scope="module")
def hl_results(scenario, spec):
    """Run HL-FOSM once for all three modes; reuse across tests."""
    return hl_fosm(scenario, spec)


class TestHLFOSMShape:
    def test_returns_three_modes(self, hl_results):
        assert set(hl_results.keys()) == {"sliding", "overturning", "bearing"}

    def test_each_result_is_FosmResult(self, hl_results):
        for r in hl_results.values():
            assert isinstance(r, FosmResult)
            assert r.method == "HL-FOSM"


class TestHLFOSMConvergent:
    """Sliding and overturning converge cleanly under HLRF iteration.
    Bearing does not — see TestHLFOSMBearingDoesNotConverge below."""

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_converged_flag(self, hl_results, mode):
        r = hl_results[mode]
        assert r.converged is True
        assert r.convergence_note is None
        assert r.n_iter > 0
        assert r.alpha is not None
        assert r.design_point is not None

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_g_at_design_point_near_zero(
        self, scenario, deterministic, hl_results, mode
    ):
        """The design point lies on g(x) = 0 — re-evaluate symbolic g
        at x* and require |g(x*)| ≪ 1 (independent of the iteration's
        own tol_g, since we want a property check, not a tautology)."""
        ls = build_limit_states(scenario, d_critical=deterministic.D_critical)
        g = getattr(ls, mode)
        dp = hl_results[mode].design_point
        subs = {sym: dp[name] for sym, name in zip(SYMBOLS, SYMBOL_NAMES)}
        g_at_design = float(g.subs(subs))
        assert abs(g_at_design) < 1e-6

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_alpha_squares_sum_to_one(self, hl_results, mode):
        """α_i are direction cosines of the unit normal at the design
        point — Σ α_i² = 1 exactly (HL-FOSM invariant)."""
        s = sum(a ** 2 for a in hl_results[mode].alpha.values())
        assert s == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_y_star_equals_minus_alpha_beta(self, hl_results, mode, spec):
        """Naccache eq. 5.42: y* = -α · β, with y* = L⁻¹(x* − μ) and
        L Lᵀ = C_X. Cross-check between the geometric (β = ||y*||) and
        gradient-direction (α = ∇g_Y / ||∇g_Y||) characterizations of
        the design point."""
        r = hl_results[mode]
        L = np.linalg.cholesky(spec.covariance_matrix())
        x_star = np.array([r.design_point[name] for name in SYMBOL_NAMES])
        y_star = np.linalg.solve(L, x_star - spec.means)
        alpha_vec = np.array([r.alpha[name] for name in SYMBOL_NAMES])
        np.testing.assert_array_almost_equal(
            y_star, -alpha_vec * r.beta, decimal=8
        )

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_pf_matches_phi_minus_beta(self, hl_results, mode):
        r = hl_results[mode]
        assert r.pf == pytest.approx(float(norm.sf(r.beta)), rel=1e-12)


class TestHLFOSMBearingDoesNotConverge:
    """Documented limitation: the bearing limit-state surface lies in a
    region where the rigid-foundation σ_max = (N/B)(1+6|e|/B) formula
    breaks down (resultant beyond the kern, possibly beyond the base
    edge). HLRF cannot find a meaningful design point.

    This is preserved as a test rather than fixed because it is the
    cleanest demonstration in the project of "MVFOSM gives a number;
    HL-FOSM tells you the number doesn't mean what you think". A future
    Stage F (FORM with proper distribution transformation + possibly a
    smoothed |e|) may handle it better; until then, the failure is
    deliberate."""

    def test_did_not_converge(self, hl_results):
        assert hl_results["bearing"].converged is False

    def test_returns_NaN(self, hl_results):
        r = hl_results["bearing"]
        assert math.isnan(r.beta)
        assert math.isnan(r.pf)
        assert math.isnan(r.sigma_g)

    def test_n_iter_sentinel(self, hl_results):
        assert hl_results["bearing"].n_iter == -1

    def test_design_point_and_alpha_are_None(self, hl_results):
        r = hl_results["bearing"]
        assert r.design_point is None
        assert r.alpha is None

    def test_convergence_note_is_informative(self, hl_results):
        note = hl_results["bearing"].convergence_note
        assert note is not None
        # Should mention the validity issue and point to FORM as the
        # next-stage remedy.
        for keyword in ("valid", "FORM"):
            assert keyword in note, f"Note should mention {keyword!r}"


class TestHLFOSMReliabilityValues:
    """Regression baselines for HL-FOSM β on the convergent modes.

    Tolerance is rel=1e-3 (looser than MVFOSM's 1e-6) because HLRF
    iteration with relaxation fallback can have small variation in
    convergence behavior across systems / NumPy versions."""

    EXPECTED_BETA = {
        "sliding":     4.0732,
        "overturning": 9.3743,
    }

    @pytest.mark.parametrize("mode,expected", EXPECTED_BETA.items())
    def test_beta_baseline(self, hl_results, mode, expected):
        assert hl_results[mode].beta == pytest.approx(expected, rel=1e-3)


class TestMVFOSMvsHLFOSM:
    """The 'MVFOSM not invariant' demo (Stage D.5) will exploit this
    gap. Here we record the qualitative observation: HL-FOSM β > MVFOSM
    β for our problem, meaning the limit-state surface curves AWAY
    from the origin and linear extrapolation from the mean
    underestimates the true geometric distance to g = 0."""

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_hl_beta_exceeds_mv_beta(self, mvfosm_results, hl_results, mode):
        bm = mvfosm_results[mode].beta
        bh = hl_results[mode].beta
        assert bh > bm, (
            f"HL β ({bh:.3f}) should exceed MV β ({bm:.3f}) for {mode} — "
            f"limit state curves away from origin"
        )


# =============================================================================
# Stage D.5 — MVFOSM non-invariance demonstration
# =============================================================================
@pytest.fixture(scope="module")
def invariance_results(scenario, spec):
    """Run compare_mvfosm_invariance once per mode; reuse across tests.

    Each call applies MVFOSM and HL-FOSM to three forms of the limit
    state, so caching matters — without it, parametrized tests would
    each re-do 6 SymPy gradient/iteration computations."""
    return {
        mode: compare_mvfosm_invariance(scenario, spec, mode)
        for mode in ("sliding", "overturning", "bearing")
    }


class TestInvarianceShape:
    def test_returns_InvarianceComparison(self, invariance_results):
        for cmp in invariance_results.values():
            assert isinstance(cmp, InvarianceComparison)

    def test_three_forms_per_mode(self, invariance_results):
        for cmp in invariance_results.values():
            assert len(cmp.form_labels) == 3
            assert len(cmp.mvfosm_betas) == 3
            assert len(cmp.hl_fosm_betas) == 3
            assert len(cmp.hl_converged) == 3

    def test_forms_in_canonical_order(self, invariance_results):
        # Order matters for the README/docs table — pin it via test.
        expected = ("g1_ratio_minus_1", "g2_difference", "g3_log_ratio")
        for cmp in invariance_results.values():
            assert cmp.form_labels == expected


class TestMVFOSMNotInvariant:
    """Cornell's classic non-invariance: MVFOSM β depends on the algebraic
    form of g. The three forms are mathematically equivalent — they share
    the failure surface ``{R = S}`` — but their gradients at the mean
    differ, producing different MVFOSM β values."""

    @pytest.mark.parametrize("mode", ["sliding", "overturning", "bearing"])
    def test_mvfosm_spread_is_substantial(self, invariance_results, mode):
        mv = invariance_results[mode].mvfosm_betas
        spread_pct = (max(mv) - min(mv)) / min(mv) * 100
        # Real spread is 54%, 356%, 210% for the three modes.
        # Require > 30% as the regression guard — well above any
        # plausible numerical-noise floor.
        assert spread_pct > 30.0, (
            f"{mode}: MVFOSM spread = {spread_pct:.2f}%; expected substantial "
            f"non-invariance"
        )


class TestHLFOSMInvariant:
    """HL-FOSM β is the geometric distance from origin to ``{g = 0}`` in
    standardized Y-space — the same surface for all algebraic forms of
    g, so the same β. Tested on the modes where all three forms
    converge (sliding, overturning)."""

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_all_forms_converge(self, invariance_results, mode):
        cmp = invariance_results[mode]
        assert all(cmp.hl_converged), (
            f"{mode}: HL convergence = {cmp.hl_converged}; expected all True"
        )

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_hl_betas_coincide_across_forms(self, invariance_results, mode):
        hl = invariance_results[mode].hl_fosm_betas
        spread_rel = (max(hl) - min(hl)) / min(hl)
        # Tolerance limited by HLRF termination criteria; different
        # relaxation paths can produce sub-tol differences.
        assert spread_rel < 1e-4, (
            f"{mode}: HL βs spread = {spread_rel*100:.4f}%, expected ≈ 0"
        )


class TestBearingPartialConvergence:
    """Pedagogical highlight: even when the underlying limit state is
    physically identical, the algorithm's ability to converge depends
    on which algebraic form is used.

    For bearing, the difference form ``g₂ = R − S`` converges where the
    ratio form ``g₁ = R/S − 1`` does not. The R/S form has a singularity
    near S = 0 (σ_max → 0) that the iteration approaches; the
    difference form has no such singularity in finite x. Same physics,
    different numerics."""

    def test_g2_converges(self, invariance_results):
        cmp = invariance_results["bearing"]
        idx = cmp.form_labels.index("g2_difference")
        assert cmp.hl_converged[idx] is True

    def test_g1_does_not_converge(self, invariance_results):
        cmp = invariance_results["bearing"]
        idx = cmp.form_labels.index("g1_ratio_minus_1")
        assert cmp.hl_converged[idx] is False


class TestInvarianceRegressionBaselines:
    """Regression-fixed MVFOSM β values across all (mode, form)
    combinations. These are the numbers that go into the README and
    the planned ``docs/llm_failure_modes.md`` table demonstrating
    non-invariance.

    Tolerance rel=1e-3: SymPy gradient ordering can induce small
    numerical-precision shifts that are below the pedagogical noise
    floor but exceed bit-exactness."""

    EXPECTED = [
        # (mode, form_label, expected MVFOSM β)
        ("sliding",     "g1_ratio_minus_1",   2.6961),
        ("sliding",     "g2_difference",      4.1636),
        ("sliding",     "g3_log_ratio",       4.1064),
        ("overturning", "g1_ratio_minus_1",   4.9084),
        ("overturning", "g2_difference",     22.4042),
        ("overturning", "g3_log_ratio",      10.1542),
        ("bearing",     "g1_ratio_minus_1",   7.0764),
        ("bearing",     "g2_difference",     21.9316),
        ("bearing",     "g3_log_ratio",      11.8176),
    ]

    @pytest.mark.parametrize("mode,label,expected", EXPECTED)
    def test_mvfosm_baseline(self, invariance_results, mode, label, expected):
        cmp = invariance_results[mode]
        idx = cmp.form_labels.index(label)
        assert cmp.mvfosm_betas[idx] == pytest.approx(expected, rel=1e-3)


class TestHLFOSMBaselines:
    """Regression-fixed HL-FOSM β for the convergent (mode, form)
    combinations. Used as the cross-form invariance check value."""

    EXPECTED = {
        "sliding":     4.0732,  # all three forms must agree
        "overturning": 9.3743,
    }

    @pytest.mark.parametrize("mode,expected", EXPECTED.items())
    def test_hl_baseline(self, invariance_results, mode, expected):
        cmp = invariance_results[mode]
        for i, label in enumerate(cmp.form_labels):
            if cmp.hl_converged[i]:
                assert cmp.hl_fosm_betas[i] == pytest.approx(
                    expected, rel=1e-3
                ), f"{mode}/{label}: HL β = {cmp.hl_fosm_betas[i]:.4f}"


# =============================================================================
# F.2 — FORM (Nataf + HL-RF in Y-space)
# =============================================================================
@pytest.fixture(scope="module")
def form_results(scenario, spec):
    """Run FORM once for all three modes; reused across F.2 tests."""
    return form(scenario, spec)


class TestFORMShape:
    def test_returns_three_modes(self, form_results):
        assert set(form_results.keys()) == {"sliding", "overturning", "bearing"}

    def test_each_result_is_FosmResult(self, form_results):
        for r in form_results.values():
            assert isinstance(r, FosmResult)

    def test_method_field(self, form_results):
        for mode, r in form_results.items():
            assert r.method == "FORM", f"{mode}: expected method='FORM', got {r.method!r}"


class TestFORMConvergent:
    """Sliding and overturning converge; bearing may or may not (documented)."""

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_converged_flag(self, form_results, mode):
        r = form_results[mode]
        assert r.converged is True
        assert r.convergence_note is None
        assert r.n_iter is not None and r.n_iter > 0
        assert r.alpha is not None
        assert r.design_point is not None

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_g_at_design_point_near_zero(self, scenario, deterministic, form_results, mode):
        """Design point must lie on the limit state surface g(x*) ≈ 0."""
        ls = build_limit_states(scenario, d_critical=deterministic.D_critical)
        g = getattr(ls, mode)
        dp = form_results[mode].design_point
        subs = {sym: dp[name] for sym, name in zip(SYMBOLS, SYMBOL_NAMES)}
        assert abs(float(g.subs(subs))) < 1e-5

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_pf_equals_phi_minus_beta(self, form_results, mode):
        r = form_results[mode]
        assert r.pf == pytest.approx(norm.sf(r.beta), rel=1e-9)

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_alpha_squares_sum_to_one(self, form_results, mode):
        r = form_results[mode]
        assert sum(v ** 2 for v in r.alpha.values()) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_beta_positive(self, form_results, mode):
        assert form_results[mode].beta > 0

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_design_point_and_alpha_keys(self, form_results, mode):
        r = form_results[mode]
        assert set(r.design_point.keys()) == set(SYMBOL_NAMES)
        assert set(r.alpha.keys()) == set(SYMBOL_NAMES)

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_sensitivity_keys(self, form_results, mode):
        assert set(form_results[mode].sensitivity.keys()) == set(SYMBOL_NAMES)


class TestFORMVsHLFOSM:
    """FORM should differ from HL-FOSM for non-normal variables."""

    @pytest.mark.parametrize("mode", ["sliding", "overturning"])
    def test_form_beta_differs_from_hl_fosm(self, form_results, hl_results, mode):
        beta_form = form_results[mode].beta
        beta_hl = hl_results[mode].beta
        # Non-normal marginals (lognormal phi, Gumbel q) cause a measurable
        # difference. A difference of less than 0.001 would indicate the
        # Nataf transformation is having no effect — likely a bug.
        assert abs(beta_form - beta_hl) > 0.001, (
            f"{mode}: FORM β = {beta_form:.4f}, HL-FOSM β = {beta_hl:.4f} "
            f"— difference too small, Nataf may not be applied."
        )


class TestFORMBearing:
    """Bearing capacity does not converge in FORM (same singularity as HL-FOSM)."""

    def test_bearing_not_converged(self, form_results):
        r = form_results["bearing"]
        assert r.converged is False
        assert r.n_iter == -1
        assert math.isnan(r.beta)
        assert math.isnan(r.pf)
        assert r.design_point is None
        assert r.alpha is None
        assert r.convergence_note is not None


# Regression baselines — freeze expected β values at rel=1e-3.
# Sliding and overturning converge. Bearing is documented non-convergent.
# FORM β > HL-FOSM β for both modes because the Nataf transformation
# correctly accounts for lognormal (φ) and Gumbel (q) marginals,
# whereas HL-FOSM approximates them as Gaussian in X-space.
class TestFORMRegressionBetas:
    _EXPECTED = {
        "sliding":     4.5377,
        "overturning": 9.6533,
    }

    @pytest.mark.parametrize("mode,expected", _EXPECTED.items())
    def test_form_beta_regression(self, form_results, mode, expected):
        r = form_results[mode]
        assert r.beta == pytest.approx(expected, rel=1e-3), (
            f"{mode}: FORM β = {r.beta:.4f}, expected ≈ {expected}"
        )

