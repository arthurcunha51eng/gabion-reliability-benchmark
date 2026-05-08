"""Tests for ``gabion.random_variables``."""
from math import exp, log, pi, sqrt

import numpy as np
import pytest

from gabion.random_variables import (
    NatafTransformation,
    RandomVariable,
    RandomVariableSpec,
    _nataf_factor,
    book_example_spec,
)


_EULER = 0.5772156649015329


# =============================================================================
# Single RandomVariable
# =============================================================================
class TestRandomVariableConstruction:
    def test_normal_constructs(self):
        rv = RandomVariable("x", "normal", mean=10.0, cov=0.1)
        assert rv.std == pytest.approx(1.0)
        assert rv.variance == pytest.approx(1.0)

    def test_unknown_distribution_rejected(self):
        with pytest.raises(ValueError, match="distribution must be one of"):
            RandomVariable("x", "weibull", mean=1.0, cov=0.1)  # type: ignore[arg-type]

    def test_nonpositive_cov_rejected(self):
        with pytest.raises(ValueError, match="cov must be > 0"):
            RandomVariable("x", "normal", mean=1.0, cov=0.0)
        with pytest.raises(ValueError, match="cov must be > 0"):
            RandomVariable("x", "normal", mean=1.0, cov=-0.1)

    def test_nonpositive_mean_rejected_for_lognormal(self):
        with pytest.raises(ValueError, match="mean must be > 0 for lognormal"):
            RandomVariable("x", "lognormal", mean=0.0, cov=0.1)
        with pytest.raises(ValueError, match="mean must be > 0 for lognormal"):
            RandomVariable("x", "lognormal", mean=-1.0, cov=0.1)

    def test_normal_with_negative_mean_allowed(self):
        # Normal can have negative mean (though we won't use it here).
        rv = RandomVariable("x", "normal", mean=-5.0, cov=0.2)
        assert rv.mean == -5.0


# =============================================================================
# Lognormal parameters
# =============================================================================
class TestLognormalParams:
    """Verify (λ, ξ) reconstruct the requested mean and variance."""

    def test_inverse_relationship(self):
        # Given (λ, ξ), the lognormal moments must satisfy:
        #   mean = exp(λ + ξ²/2)
        #   var  = (exp(ξ²) − 1) · exp(2λ + ξ²)
        rv = RandomVariable("phi", "lognormal", mean=30.0, cov=0.10)
        lam, xi = rv.lognormal_params()

        recovered_mean = exp(lam + xi ** 2 / 2)
        recovered_var = (exp(xi ** 2) - 1) * exp(2 * lam + xi ** 2)

        assert recovered_mean == pytest.approx(30.0, rel=1e-12)
        assert recovered_var == pytest.approx(rv.variance, rel=1e-12)

    def test_low_cov_limit(self):
        # For very small COV the lognormal approaches a normal and we
        # should have ξ ≈ COV, λ ≈ ln(mean).
        rv = RandomVariable("x", "lognormal", mean=100.0, cov=1e-6)
        lam, xi = rv.lognormal_params()
        assert xi == pytest.approx(1e-6, rel=1e-3)
        assert lam == pytest.approx(log(100.0), abs=1e-12)

    def test_rejected_for_non_lognormal(self):
        rv = RandomVariable("x", "normal", mean=1.0, cov=0.1)
        with pytest.raises(ValueError, match="lognormal_params"):
            rv.lognormal_params()


# =============================================================================
# Gumbel parameters
# =============================================================================
class TestGumbelParams:
    def test_inverse_relationship(self):
        # Given (u, α), the Gumbel moments must satisfy:
        #   mean = u + γ_E / α
        #   var  = π² / (6 α²)
        rv = RandomVariable("q", "gumbel", mean=10.0, cov=0.30)
        u, alpha = rv.gumbel_params()

        recovered_mean = u + _EULER / alpha
        recovered_var = pi ** 2 / (6 * alpha ** 2)

        assert recovered_mean == pytest.approx(10.0, rel=1e-12)
        assert recovered_var == pytest.approx(rv.variance, rel=1e-12)

    def test_alpha_formula(self):
        rv = RandomVariable("q", "gumbel", mean=10.0, cov=0.30)
        u, alpha = rv.gumbel_params()
        expected_alpha = pi / (sqrt(6) * 3.0)
        assert alpha == pytest.approx(expected_alpha, rel=1e-12)

    def test_rejected_for_non_gumbel(self):
        rv = RandomVariable("x", "normal", mean=1.0, cov=0.1)
        with pytest.raises(ValueError, match="gumbel_params"):
            rv.gumbel_params()


# =============================================================================
# RandomVariableSpec — happy path
# =============================================================================
class TestSpecBasics:
    def test_minimal_two_var_spec(self):
        spec = RandomVariableSpec(
            variables=(
                RandomVariable("a", "normal", mean=1.0, cov=0.1),
                RandomVariable("b", "normal", mean=2.0, cov=0.1),
            ),
            correlation_matrix=np.eye(2),
        )
        assert spec.n == 2
        assert spec.names == ("a", "b")

    def test_means_and_stds(self):
        spec = book_example_spec()
        assert spec.means.shape == (5,)
        np.testing.assert_array_equal(spec.means, [30.0, 18.0, 30.0, 10.0, 25.0])
        np.testing.assert_array_almost_equal(
            spec.stds, [3.0, 0.9, 3.0, 3.0, 0.75]
        )

    def test_covariance_matrix_diagonal_is_variance(self):
        spec = book_example_spec()
        C = spec.covariance_matrix()
        np.testing.assert_array_almost_equal(np.diag(C), spec.stds ** 2)

    def test_covariance_matrix_off_diagonal_uses_correlation(self):
        # C_X[i, j] = ρ_ij · σ_i · σ_j
        spec = book_example_spec()
        C = spec.covariance_matrix()
        i = spec.index_of("phi_backfill")
        j = spec.index_of("phi_foundation")
        expected = 0.5 * spec.stds[i] * spec.stds[j]
        assert C[i, j] == pytest.approx(expected, rel=1e-12)
        assert C[j, i] == pytest.approx(expected, rel=1e-12)

    def test_indexing_by_name(self):
        spec = book_example_spec()
        assert spec["phi_backfill"].mean == 30.0
        assert spec["q"].distribution == "gumbel"

    def test_index_of_unknown_raises(self):
        spec = book_example_spec()
        with pytest.raises(KeyError):
            spec.index_of("unknown_var")


# =============================================================================
# RandomVariableSpec — validation failures
# =============================================================================
def _two_var_spec_with_matrix(rho):
    return RandomVariableSpec(
        variables=(
            RandomVariable("a", "normal", mean=1.0, cov=0.1),
            RandomVariable("b", "normal", mean=2.0, cov=0.1),
        ),
        correlation_matrix=rho,
    )


class TestSpecValidation:
    def test_empty_variables_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            RandomVariableSpec(variables=(), correlation_matrix=np.eye(0))

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate variable names"):
            RandomVariableSpec(
                variables=(
                    RandomVariable("x", "normal", mean=1.0, cov=0.1),
                    RandomVariable("x", "normal", mean=2.0, cov=0.1),
                ),
                correlation_matrix=np.eye(2),
            )

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="must be 2×2"):
            _two_var_spec_with_matrix(np.eye(3))

    def test_non_array_rejected(self):
        with pytest.raises(TypeError, match="must be a NumPy array"):
            _two_var_spec_with_matrix([[1.0, 0.0], [0.0, 1.0]])

    def test_non_symmetric_rejected(self):
        rho = np.array([[1.0, 0.3], [0.4, 1.0]])
        with pytest.raises(ValueError, match="not symmetric"):
            _two_var_spec_with_matrix(rho)

    def test_diagonal_not_one_rejected(self):
        rho = np.array([[0.99, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="diagonal must be all 1"):
            _two_var_spec_with_matrix(rho)

    def test_off_diagonal_above_one_rejected(self):
        rho = np.array([[1.0, 1.5], [1.5, 1.0]])
        with pytest.raises(ValueError, match=r"off-diagonals must lie in"):
            _two_var_spec_with_matrix(rho)

    def test_off_diagonal_below_minus_one_rejected(self):
        rho = np.array([[1.0, -1.5], [-1.5, 1.0]])
        with pytest.raises(ValueError, match=r"off-diagonals must lie in"):
            _two_var_spec_with_matrix(rho)

    def test_positive_semi_definite_but_not_definite_rejected(self):
        # Singular matrix (perfectly correlated) — Cholesky will fail.
        rho = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="positive-definite"):
            _two_var_spec_with_matrix(rho)

    def test_indefinite_three_var_matrix_rejected(self):
        # Construct a clearly indefinite matrix. Cholesky must reject.
        rho = np.array(
            [[1.0,  0.9,  0.9],
             [0.9,  1.0, -0.9],
             [0.9, -0.9,  1.0]]
        )
        with pytest.raises(ValueError, match="positive-definite"):
            RandomVariableSpec(
                variables=(
                    RandomVariable("a", "normal", mean=1.0, cov=0.1),
                    RandomVariable("b", "normal", mean=1.0, cov=0.1),
                    RandomVariable("c", "normal", mean=1.0, cov=0.1),
                ),
                correlation_matrix=rho,
            )


# =============================================================================
# book_example_spec
# =============================================================================
class TestBookExample:
    def test_constructs(self):
        spec = book_example_spec()
        assert spec.n == 5

    def test_expected_distributions(self):
        spec = book_example_spec()
        expected = {
            "phi_backfill":   "lognormal",
            "gamma_backfill": "normal",
            "phi_foundation": "lognormal",
            "q":              "gumbel",
            "gamma_g":        "normal",
        }
        for name, dist in expected.items():
            assert spec[name].distribution == dist

    def test_expected_correlations(self):
        spec = book_example_spec()
        rho = spec.correlation_matrix
        ipb = spec.index_of("phi_backfill")
        ipf = spec.index_of("phi_foundation")
        igb = spec.index_of("gamma_backfill")
        igg = spec.index_of("gamma_g")
        iq  = spec.index_of("q")

        assert rho[ipb, ipf] == 0.5
        assert rho[ipb, igb] == 0.3
        # Everything else off-diagonal must be zero.
        assert rho[ipf, igb] == 0.0
        assert rho[ipb, iq] == 0.0
        assert rho[ipb, igg] == 0.0
        assert rho[ipf, iq] == 0.0

    def test_correlation_matrix_is_positive_definite(self):
        # Smoke test: construction itself triggers the Cholesky check;
        # this just makes the property explicit in the test report.
        spec = book_example_spec()
        np.linalg.cholesky(spec.correlation_matrix)


# =============================================================================
# Stage E.1 — sampling
# =============================================================================
class TestRandomVariableSampling:
    """iid marginal sampling via standard normal → marginal transform."""

    def test_normal_moments(self):
        rv = RandomVariable("x", "normal", mean=18.0, cov=0.05)
        rng = np.random.default_rng(42)
        samples = rv.sample(100_000, rng)
        assert samples.shape == (100_000,)
        assert np.mean(samples) == pytest.approx(18.0, abs=0.02)
        assert np.std(samples) == pytest.approx(0.9, abs=0.02)

    def test_lognormal_moments(self):
        rv = RandomVariable("x", "lognormal", mean=30.0, cov=0.10)
        rng = np.random.default_rng(42)
        samples = rv.sample(100_000, rng)
        assert (samples > 0).all()  # lognormal is positive-only
        assert np.mean(samples) == pytest.approx(30.0, rel=0.005)
        assert np.std(samples) == pytest.approx(3.0, rel=0.02)

    def test_gumbel_moments(self):
        rv = RandomVariable("x", "gumbel", mean=10.0, cov=0.30)
        rng = np.random.default_rng(42)
        samples = rv.sample(100_000, rng)
        assert np.mean(samples) == pytest.approx(10.0, abs=0.05)
        assert np.std(samples) == pytest.approx(3.0, rel=0.02)

    def test_sample_uses_transform_internally(self):
        """``rv.sample(n, rng)`` must equal
        ``rv.transform_standard_normal(rng.standard_normal(n))`` for the
        same RNG state — confirms the iid path and the copula path
        share a single implementation."""
        rv = RandomVariable("x", "lognormal", mean=30.0, cov=0.10)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = rv.sample(1000, rng1)
        s2 = rv.transform_standard_normal(rng2.standard_normal(1000))
        np.testing.assert_array_equal(s1, s2)

    def test_reproducibility(self):
        rv = RandomVariable("x", "gumbel", mean=10.0, cov=0.3)
        s1 = rv.sample(1000, np.random.default_rng(42))
        s2 = rv.sample(1000, np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)

    def test_zero_n_rejected(self):
        rv = RandomVariable("x", "normal", mean=1.0, cov=0.1)
        with pytest.raises(ValueError, match="n must be > 0"):
            rv.sample(0, np.random.default_rng(0))

    def test_negative_n_rejected(self):
        rv = RandomVariable("x", "normal", mean=1.0, cov=0.1)
        with pytest.raises(ValueError, match="n must be > 0"):
            rv.sample(-5, np.random.default_rng(0))

    def test_transform_handles_extreme_z(self):
        """For very large |z|, transform must remain finite (clipping
        in ``norm.cdf`` for Gumbel, no special handling for Normal/
        Lognormal which are smooth)."""
        rv_n = RandomVariable("a", "normal", mean=1.0, cov=0.1)
        rv_l = RandomVariable("b", "lognormal", mean=1.0, cov=0.1)
        rv_g = RandomVariable("c", "gumbel", mean=1.0, cov=0.3)
        z_extreme = np.array([-10.0, 10.0])
        for rv in (rv_n, rv_l, rv_g):
            x = rv.transform_standard_normal(z_extreme)
            assert np.isfinite(x).all(), f"{rv.distribution}: got {x}"


class TestRandomVariableSpecCorrelatedSampling:
    """Cholesky-based correlated sampling (Gaussian copula)."""

    def test_shape(self):
        spec = book_example_spec()
        rng = np.random.default_rng(42)
        samples = spec.sample_correlated(1000, rng)
        assert samples.shape == (1000, 5)

    def test_marginal_means_recovered(self):
        spec = book_example_spec()
        rng = np.random.default_rng(42)
        samples = spec.sample_correlated(100_000, rng)
        for i, rv in enumerate(spec.variables):
            empirical = float(np.mean(samples[:, i]))
            sem = rv.std / np.sqrt(100_000)
            assert abs(empirical - rv.mean) < 4 * sem, (
                f"{rv.name}: mean = {empirical:.4f}, expected ≈ {rv.mean}"
            )

    def test_marginal_stds_recovered(self):
        spec = book_example_spec()
        rng = np.random.default_rng(42)
        samples = spec.sample_correlated(100_000, rng)
        for i, rv in enumerate(spec.variables):
            empirical = float(np.std(samples[:, i]))
            assert empirical == pytest.approx(rv.std, rel=0.02), (
                f"{rv.name}: std = {empirical:.4f}, expected ≈ {rv.std:.4f}"
            )

    def test_correlations_recovered(self):
        """For our COVs (≤ 0.30), the Gaussian-copula distortion of the
        X-space correlation is much smaller than 1%. With n = 100_000
        the sampling-noise SE on a correlation estimate is ~ 1/√n ≈
        0.003. Tolerance 0.02 covers both."""
        spec = book_example_spec()
        rng = np.random.default_rng(42)
        samples = spec.sample_correlated(100_000, rng)
        emp = np.corrcoef(samples, rowvar=False)

        ipb = spec.index_of("phi_backfill")
        ipf = spec.index_of("phi_foundation")
        igb = spec.index_of("gamma_backfill")
        iq = spec.index_of("q")
        igg = spec.index_of("gamma_g")

        # Requested non-zero correlations
        assert emp[ipb, ipf] == pytest.approx(0.5, abs=0.02)
        assert emp[ipb, igb] == pytest.approx(0.3, abs=0.02)

        # Off-diagonals (zero correlation requested)
        for i, j in [(ipf, igb), (ipb, iq), (ipb, igg),
                     (ipf, iq), (ipf, igg), (igb, iq),
                     (igb, igg), (iq, igg)]:
            assert abs(emp[i, j]) < 0.02, (
                f"corr[{i},{j}] = {emp[i,j]:.4f}, expected ≈ 0"
            )

    def test_lognormal_samples_positive(self):
        """Lognormal marginals must produce X > 0 — sanity check that
        the transform was correctly applied (a missed exp would give
        Gaussian-like negative tails)."""
        spec = book_example_spec()
        rng = np.random.default_rng(42)
        samples = spec.sample_correlated(50_000, rng)
        ipb = spec.index_of("phi_backfill")
        ipf = spec.index_of("phi_foundation")
        assert (samples[:, ipb] > 0).all()
        assert (samples[:, ipf] > 0).all()

    def test_reproducibility(self):
        spec = book_example_spec()
        s1 = spec.sample_correlated(1000, np.random.default_rng(42))
        s2 = spec.sample_correlated(1000, np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_produce_different_samples(self):
        spec = book_example_spec()
        s1 = spec.sample_correlated(1000, np.random.default_rng(42))
        s2 = spec.sample_correlated(1000, np.random.default_rng(43))
        # Different seeds → almost certainly different samples
        assert not np.array_equal(s1, s2)

    def test_zero_n_rejected(self):
        spec = book_example_spec()
        with pytest.raises(ValueError, match="n must be > 0"):
            spec.sample_correlated(0, np.random.default_rng(0))


# =============================================================================
# Stage F.1 — CDF / PDF
# =============================================================================
class TestCDF:
    def test_normal_cdf_at_mean_is_half(self):
        rv = RandomVariable("x", "normal", mean=10.0, cov=0.1)
        assert rv.cdf(10.0) == pytest.approx(0.5, abs=1e-12)

    def test_lognormal_cdf_at_median_is_half(self):
        """Lognormal mean is ABOVE the median, so cdf(mean) > 0.5.
        cdf(median) = 0.5 exactly."""
        rv = RandomVariable("x", "lognormal", mean=30.0, cov=0.10)
        from math import exp
        lam, _ = rv.lognormal_params()
        median = exp(lam)
        assert rv.cdf(median) == pytest.approx(0.5, abs=1e-12)
        assert rv.cdf(30.0) > 0.5  # mean above median

    def test_gumbel_cdf_at_location_is_exp_minus_1(self):
        """Φ_Gumbel(u) = exp(−exp(0)) = e⁻¹ ≈ 0.368."""
        rv = RandomVariable("x", "gumbel", mean=10.0, cov=0.30)
        u, _ = rv.gumbel_params()
        from math import exp
        assert rv.cdf(u) == pytest.approx(exp(-1.0), rel=1e-9)

    def test_lognormal_cdf_at_zero_is_zero(self):
        rv = RandomVariable("x", "lognormal", mean=30.0, cov=0.10)
        assert rv.cdf(0.0) == 0.0
        assert rv.cdf(-5.0) == 0.0

    def test_cdf_is_monotonic_normal(self):
        rv = RandomVariable("x", "normal", mean=5.0, cov=0.2)  # std = 1.0
        xs = np.linspace(0, 10, 100)
        cdfs = np.array([rv.cdf(x) for x in xs])
        assert np.all(np.diff(cdfs) >= 0)

    def test_cdf_vectorized_returns_array(self):
        rv = RandomVariable("x", "normal", mean=10.0, cov=0.1)
        xs = np.array([8.0, 10.0, 12.0])
        cdfs = rv.cdf(xs)
        assert cdfs.shape == (3,)
        assert cdfs[1] == pytest.approx(0.5)


class TestPDF:
    def test_normal_pdf_at_mean(self):
        rv = RandomVariable("x", "normal", mean=10.0, cov=0.1)
        # 1 / (σ √(2π)) at the mean
        from math import sqrt, pi
        expected = 1.0 / (rv.std * sqrt(2 * pi))
        assert rv.pdf(10.0) == pytest.approx(expected, rel=1e-12)

    def test_lognormal_pdf_at_zero_is_zero(self):
        rv = RandomVariable("x", "lognormal", mean=30.0, cov=0.10)
        assert rv.pdf(0.0) == 0.0
        assert rv.pdf(-5.0) == 0.0

    @pytest.mark.parametrize(
        "dist,mean,cov,lo,hi",
        [
            ("normal",    18.0, 0.05,  13.5,  22.5),
            ("lognormal", 30.0, 0.10,  15.0,  45.0),
            ("gumbel",    10.0, 0.30,   0.0,  35.0),
        ],
    )
    def test_pdf_integrates_to_one(self, dist, mean, cov, lo, hi):
        """Sanity: the PDF integrates to ≈ 1 over its effective support.
        Tolerance allows for the truncation at finite ``hi``."""
        from scipy.integrate import quad
        rv = RandomVariable("x", dist, mean=mean, cov=cov)
        integral, _ = quad(rv.pdf, lo, hi)
        # 5σ tails are mostly captured but Gumbel's right tail is heavier.
        assert integral == pytest.approx(1.0, abs=2e-3)


# =============================================================================
# Stage F.1 — Nataf transformation
# =============================================================================
class TestNatafFactor:
    def test_normal_normal_factor_is_one(self):
        rv1 = RandomVariable("a", "normal", mean=1.0, cov=0.1)
        rv2 = RandomVariable("b", "normal", mean=2.0, cov=0.2)
        assert _nataf_factor(rv1, rv2, 0.5) == 1.0

    def test_lognormal_lognormal_factor_close_to_one_for_small_cov(self):
        """For COV → 0, Lognormal → Normal, so F → 1."""
        rv1 = RandomVariable("a", "lognormal", mean=1.0, cov=0.001)
        rv2 = RandomVariable("b", "lognormal", mean=1.0, cov=0.001)
        assert _nataf_factor(rv1, rv2, 0.5) == pytest.approx(1.0, abs=1e-6)

    def test_lognormal_lognormal_factor_canonical(self):
        """For our COV = 0.10 case, the analytical factor is ≈ 1.00249."""
        rv1 = RandomVariable("a", "lognormal", mean=30.0, cov=0.10)
        rv2 = RandomVariable("b", "lognormal", mean=30.0, cov=0.10)
        assert _nataf_factor(rv1, rv2, 0.5) == pytest.approx(1.00249, rel=1e-4)

    def test_lognormal_normal_factor_canonical(self):
        """Liu-DK factor F = cov_LN / sqrt(ln(1 + cov_LN²)).
        For cov = 0.10: F ≈ 0.10 / 0.0998 ≈ 1.00249."""
        rv_ln = RandomVariable("a", "lognormal", mean=30.0, cov=0.10)
        rv_n = RandomVariable("b", "normal", mean=18.0, cov=0.05)
        assert _nataf_factor(rv_ln, rv_n, 0.3) == pytest.approx(1.00249, rel=1e-4)

    def test_factor_symmetric_in_argument_order(self):
        """F(rv_i, rv_j) = F(rv_j, rv_i) — pair type is the same either way."""
        rv_ln = RandomVariable("a", "lognormal", mean=30.0, cov=0.10)
        rv_n = RandomVariable("b", "normal", mean=18.0, cov=0.05)
        f1 = _nataf_factor(rv_ln, rv_n, 0.3)
        f2 = _nataf_factor(rv_n, rv_ln, 0.3)
        assert f1 == pytest.approx(f2, rel=1e-12)

    def test_unsupported_pair_raises(self):
        rv_ln = RandomVariable("a", "lognormal", mean=10.0, cov=0.1)
        rv_g = RandomVariable("b", "gumbel", mean=10.0, cov=0.3)
        with pytest.raises(NotImplementedError, match="Lognormal–Lognormal"):
            _nataf_factor(rv_ln, rv_g, 0.5)


class TestNatafTransformationConstruction:
    def test_from_book_spec(self):
        spec = book_example_spec()
        nataf = NatafTransformation.from_spec(spec)
        assert nataf.rho_Y.shape == (5, 5)
        assert np.allclose(np.diag(nataf.rho_Y), 1.0)
        assert nataf.L_Y.shape == (5, 5)

    def test_rho_Y_close_to_rho_X_for_small_covs(self):
        """For our COVs ≤ 0.30, distortion should be < 1%."""
        spec = book_example_spec()
        nataf = NatafTransformation.from_spec(spec)
        diff = np.abs(nataf.rho_Y - spec.correlation_matrix)
        np.fill_diagonal(diff, 0)
        assert diff.max() < 0.01

    def test_zero_correlations_stay_zero(self):
        """ρ_X = 0 ⇒ ρ_Y = 0 regardless of marginal types."""
        spec = book_example_spec()
        nataf = NatafTransformation.from_spec(spec)
        # Most off-diagonals are 0 in the spec — they must stay 0.
        for i in range(5):
            for j in range(i):
                if spec.correlation_matrix[i, j] == 0:
                    assert nataf.rho_Y[i, j] == 0

    def test_all_normal_spec_has_rho_Y_equals_rho_X(self):
        """When all variables are Normal, Nataf is the identity transform
        on the correlation matrix."""
        all_normal = (
            RandomVariable("a", "normal", mean=10.0, cov=0.1),
            RandomVariable("b", "normal", mean=20.0, cov=0.2),
            RandomVariable("c", "normal", mean=30.0, cov=0.05),
        )
        rho = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
        spec = RandomVariableSpec(variables=all_normal, correlation_matrix=rho)
        nataf = NatafTransformation.from_spec(spec)
        np.testing.assert_array_almost_equal(nataf.rho_Y, rho, decimal=12)


class TestNatafTransformationRoundTrip:
    def test_round_trip_at_mean(self):
        """x → y → x must recover the original point at machine precision."""
        spec = book_example_spec()
        nataf = NatafTransformation.from_spec(spec)
        x_orig = spec.means
        y = nataf.x_to_y(x_orig)
        x_back = nataf.y_to_x(y)
        np.testing.assert_array_almost_equal(x_back, x_orig, decimal=10)

    def test_round_trip_at_random_points(self):
        spec = book_example_spec()
        nataf = NatafTransformation.from_spec(spec)
        rng = np.random.default_rng(42)
        # Sample plausible X by going through y_to_x first
        Y = rng.standard_normal((20, 5))
        X = nataf.y_to_x(Y)
        Y_back = nataf.x_to_y(X)
        X_back = nataf.y_to_x(Y_back)
        np.testing.assert_array_almost_equal(X_back, X, decimal=10)

    def test_batch_shape_preserved(self):
        """Shape (n, d) input ⇒ shape (n, d) output; shape (d,) ⇒ (d,)."""
        spec = book_example_spec()
        nataf = NatafTransformation.from_spec(spec)
        rng = np.random.default_rng(42)
        # Single
        y_single = rng.standard_normal(5)
        x_single = nataf.y_to_x(y_single)
        assert x_single.shape == (5,)
        # Batch
        y_batch = rng.standard_normal((100, 5))
        x_batch = nataf.y_to_x(y_batch)
        assert x_batch.shape == (100, 5)


class TestNatafEmpiricalCorrelation:
    """Sample Y ~ N(0, I), apply y_to_x, check empirical correlation in
    X matches ρ_X. This is the whole point of Nataf — better correlation
    fidelity than the bare Cholesky-on-ρ_X approach used in Stage E."""

    def test_empirical_correlation_matches_target(self):
        spec = book_example_spec()
        nataf = NatafTransformation.from_spec(spec)
        rng = np.random.default_rng(42)
        n = 100_000
        Y = rng.standard_normal((n, spec.n))
        X = nataf.y_to_x(Y)
        emp = np.corrcoef(X, rowvar=False)

        ipb = spec.index_of("phi_backfill")
        ipf = spec.index_of("phi_foundation")
        igb = spec.index_of("gamma_backfill")

        # Tolerance ≈ 3 × sampling SE (1/√n ≈ 0.003)
        assert emp[ipb, ipf] == pytest.approx(0.5, abs=0.01)
        assert emp[ipb, igb] == pytest.approx(0.3, abs=0.01)
