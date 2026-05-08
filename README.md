# gabion-stability

Probabilistic and deterministic analysis framework for gabion gravity wall 
stability, integrating limit equilibrium methods, reliability analysis 
(FOSM / FORM), and Monte Carlo simulation.

## Intended Audience and Project Positioning

This repository serves two distinct but related audiences.

## 1. Engineering / Geotechnical Reviewers

This project demonstrates a full pipeline of deterministic and probabilistic
stability analysis applied to a gabion gravity wall system, including:

- Limit equilibrium analysis (Coulomb wedge method)
- Sliding, overturning, and bearing capacity checks
- First-order reliability methods (MVFOSM, HL-FOSM, FORM)
- Monte Carlo simulation with correlated random variables (Cholesky + Nataf transformation)
- Direct comparison between deterministic safety factors and reliability indices (β)

The methodology extends standard deterministic geotechnical design by explicitly
quantifying uncertainty propagation and highlighting cases where deterministic 
safety factors and probabilistic reliability rankings diverge.

This work is directly aligned with my undergraduate thesis:

> Arthur Cunha da Silva. *Análise Determinística e Probabilística de uma Estrutura de Contenção do Tipo Gabião*. 2019. Undergraduate thesis, Civil Engineering, Universidade Federal de Viçosa. Advisor: Roberto Aguiar dos Santos.

## Interactive analysis pipeline

The repository includes an executable workflow for testing different wall 
configurations and probabilistic assumptions:

```bash
python run_analysis.py
```

Optional modes:

```bash
python run_analysis.py --defaults
python run_analysis.py --plots
python run_analysis.py --n-mc 100000
```

---

## 2. AI / Automation Evaluation Contexts

This project also includes a structured evaluation of how general-purpose
language models behave under constrained engineering reasoning tasks.

The `docs/llm_failure_modes.md` document systematically catalogs observed
failure modes in probabilistic geotechnical reasoning when LLMs are required to:

- Perform multi-stage physical computations  
- Maintain consistency across deterministic and probabilistic models  
- Handle correlated uncertainty propagation  
- Distinguish between algebraic equivalence and numerical equivalence  
- Respect domain-specific constraints (geotechnical limit states)

This section is intended as an empirical benchmark of constraint recognition 
and structured reasoning failure in applied STEM contexts.

## Why this project

Two motivations:

1. **Reliability methods are usually taught as recipes.** Most
   undergraduate references present FOSM as "compute β = μ_g/σ_g, done"
   and Monte Carlo as "draw samples, count failures". This project
   exercises each method on a single non-trivial problem, side by side,
   so the failure cases of each method are visible — the *non-invariance*
   of MVFOSM under algebraic reformulation (Stage D.5), the cost of the
   all-Normal simplification exposed by Stage E, and the distributional
   correction FORM provides in Stage F.
2. **AI tooling for engineering needs to be tested empirically.** The
   `docs/llm_failure_modes.md` companion document catalogs concrete,
   numerical examples of failure modes that current LLMs produce when
   asked geotechnical-reliability questions, with the gabion problem as
   the test bed. Fourteen distinct error patterns are documented with
   reproducible numbers; seven of them have been externally confirmed by
   a controlled red-team test against a production LLM (Gemini Pro,
   2025-05;Sonnet 4.6) using the standardized prompt in `docs/red_team_prompt.md`.

## Problem statement

A three-layer OUTSIDE-stepped gabion wall (steps face the backfill,
external face vertical) retains a cohesionless backfill with surcharge.
The active thrust is computed from a Coulomb wedge driven by the
critical distance D (chosen to maximize Ea, with the wall self-weight
treated through the wedge centroid). Three failure modes are checked:

| Mode        | R                                     | S                              |
|-------------|---------------------------------------|--------------------------------|
| Sliding     | T_resist = N tan(φ_f) + P′ sin(β)     | T_drive = Ea cos(θ + β)        |
| Overturning | M_resist = M_p + M_Eav                | M_overturn = Ea Y_arm cos(θ)   |
| Bearing     | q_adm                                 | σ_max = (N/B)(1 + 6\|e\|/B)    |

Notation follows Naccache (2016) and Bowles. The deterministic engine
is bit-exact against a hand-traced baseline and reproduces GAWACWIN
software output to four significant figures when constrained to
GAWACWIN's wedge selection (D = 2.5 m). The default mode uses the
textbook max-Ea wedge (D = 2.0 m for the canonical scenario), which
produces a 1–3% systematic gap in factors of safety — documented in
[`docs/baseline_capture.md`](docs/baseline_capture.md), not hidden.

## Random-variable specification

Five quantities are uncertain; everything else (geometry, geotextile
factor, porosity, allowable bearing pressure) is held fixed.

| Variable         | Distribution | Mean       | COV   | Source                       |
|------------------|--------------|------------|-------|------------------------------|
| φ_backfill       | Lognormal    | 30°        | 0.10  | Phoon & Kulhawy (1999)       |
| γ_backfill       | Normal       | 18 kN/m³   | 0.05  | Naccache (2016) Table 6.2    |
| φ_foundation     | Lognormal    | 30°        | 0.10  | same characterization        |
| q (surcharge)    | Gumbel       | 10 kN/m²   | 0.30  | live-load surcharge          |
| γ_g (stone)      | Normal       | 25 kN/m³   | 0.03  | quarry-stone QC              |

Correlation matrix (off-diagonals shown; everything else is zero):

* ρ(φ_backfill, φ_foundation) = +0.5 — same geological region
* ρ(φ_backfill, γ_backfill) = +0.3 — denser soils tend to be stronger

The same specification is consumed by Stages D, E, and F so the methods
are directly comparable.

## Methods implemented

### Deterministic engine (Stages B–C)

Coulomb wedge with kinematics in the gabion frame, max-Ea wedge
selection over a 10-point grid, FS computation for all three modes,
and a YAML/CLI front-end (`gabion-check`). Bit-exact regression
baseline maintained at `rtol = 1e-12`.

### Cornell MVFOSM (Stage D.3)

Limit-state functions g(X) are derived in SymPy and cross-validated at
the mean against the deterministic engine to machine precision (~1e-16).
For each mode, β = μ_g / σ_g with σ_g² = ∇gᵀ C_X ∇g; P_f = Φ(−β) via
`scipy.stats.norm.sf` (avoids underflow at large β). Sensitivity
coefficients γ_i = (∂g/∂X_i · σ_i) / σ_g are reported in X-space.
**All variables treated as Normal** at this stage (Naccache §5.5.1
convention).

### Hasofer-Lind FOSM (Stage D.4)

HLRF iteration in X-space (Naccache eq. 5.55) with automatic relaxation
fallback (1.0 → 0.5 → 0.25 → 0.1) when the unrelaxed iteration fails to
converge. Returns:

* β = ‖y*‖ as the geometric distance from origin to the design point
  in standardized Y-space (Y = L⁻¹(X − μ), L Lᵀ = C_X)
* design point x* in original X-space
* directional cosines α_i = (∇g_Y(y*))_i / ‖∇g_Y(y*)‖, satisfying
  Σα_i² = 1 and y* = −α β (Naccache eq. 5.42)

When all relaxations fail, the result has `converged = False`, β = NaN,
and an informative `convergence_note`. This deliberately preserves the
failure rather than masking it — see Stage D.4 discussion for bearing.
**All variables also treated as Normal** at this stage.

### MVFOSM non-invariance demonstration (Stage D.5)

Each limit-state function is reformulated in three algebraically
equivalent forms — `g₁ = R/S − 1`, `g₂ = R − S`, `g₃ = ln(R/S)` —
and both methods are applied to all three. MVFOSM β spreads by 54%
(sliding) and 356% (overturning). HL-FOSM β is identical across all
forms (invariant by construction). See `docs/llm_failure_modes.md`.

### Monte Carlo (Stage E)

Vectorized Monte Carlo with **Cholesky-correlated samples and the
actual marginals** (Lognormal for friction angles, Gumbel for surcharge,
Normal for unit weights) — the first stage where distributional shape
matters. Uses the Gaussian copula approach: sample iid standard normals,
Cholesky-correlate them, then transform each marginal independently
via inverse CDF.

System reliability via two paths:

* **Direct MC:** P(F₁ ∪ F₂ ∪ F₃) = (1/n) Σ I(any g_i < 0).
* **Ditlevsen bi-modal bounds:** P_F ∈ [lower, upper] using only
  per-mode P_i and pairwise joint P_ij (Naccache eq. 5.99).

Per-mode and system P_f are reported with Wilson 95% confidence
intervals — preferred over normal-approximation intervals for the
small-P regime.

### FORM via Nataf transformation (Stage F)

First-Order Reliability Method with the **Nataf transformation** to
handle the actual marginals (Liu & Der Kiureghian 1986). The three-space
pipeline:

1. **X-space** — original variables with their marginal distributions
2. **Z-space** — correlated standard normals via the marginal CDF route:
   z_i = Φ⁻¹(F_{X_i}(x_i))
3. **Y-space** — independent N(0, I) via Z = L_Y Y, where L_Y is the
   Cholesky factor of the Nataf-corrected correlation matrix

The HL-RF iteration runs in Y-space. The gradient is mapped back via the
chain rule: ∇_y g = L_Y^T ⊙ [φ(z_i) / f_{X_i}(x_i)] ⊙ ∇_x g. Hybrid
convergence criterion: |g| < ε and (‖Δy‖ < ε or |Δβ| < ε) — needed for
large-β modes where the iterate slides along the limit surface at
approximately constant β. Relaxation fallback identical to HL-FOSM.

FORM returns the design point in X-space, importance factors α_i² in
Y-space (unit-sum), and Pf = Φ(−β) with the actual distributional tails.

## Key results

For the canonical scenario, with n = 100,000 Monte Carlo samples
(seed = 42):

| Mode        | FS    | β_MVFOSM | β_HL-FOSM | β_FORM   | P_f_FORM  |
|-------------|-------|----------|-----------|----------|-----------|
| Sliding     | 2.484 | 2.696    | 4.073     | **4.538**| 2.8×10⁻⁶ |
| Overturning | 5.393 | 4.908    | 9.374     | **9.653**| 2.4×10⁻²²|
| Bearing     | —     | 7.076    | n.c.      | n.c.     | —         |

_MC with 100k samples observed 0 failures across all modes; 95% Wilson
upper bound is 3.8×10⁻⁵ for each mode. "n.c." = not converged — see
Honest limitations._

**Five key observations** (all reproducible from `tests/`):

1. **Deterministic-vs-probabilistic rank inversion.** Bearing has the
   lowest FS (3.10) but the highest β (7.08). Sliding is the critical
   mode probabilistically because φ_b and φ_f have COVs of 0.10 each.
2. **HL-FOSM β > MVFOSM β.** The limit-state surface is concave when
   viewed from the mean — Cornell's linearization underestimates the
   geometric distance to g = 0, by 51% for sliding and 91% for overturning.
3. **MVFOSM non-invariance.** Across the three algebraic forms, MVFOSM β
   spreads by 54% (sliding) and 356% (overturning). HL-FOSM β is
   identical across forms (spread < 0.0001%).
4. **FORM β > HL-FOSM β** — the distributional correction goes in the
   safe direction. Lognormal friction angles have thinner left tails than
   equivalent Normal distributions, so failure events (low φ) are rarer
   than the all-Normal HL-FOSM predicts. The gap is +0.46 β-units for
   sliding (+11%) and +0.28 β-units for overturning (+3%).
5. **FORM importance factors confirm physical intuition.** For sliding,
   φ_backfill (α²=0.44) and q (α²=0.36) dominate, with φ_foundation
   third (α²=0.15). For overturning, q dominates (α²=0.62), φ_backfill
   is second (α²=0.37), and φ_foundation is zero (correctly — it does
   not enter the overturning moment equation).

## Figures

Generated by `python run_analysis.py --plots` or
`from gabion.plots import generate_all_plots`.

| Figure | Description |
|--------|-------------|
| [`figures/beta_comparison.png`](figures/beta_comparison.png) | β values for all three methods side by side (sliding and overturning modes) |
| [`figures/form_importance.png`](figures/form_importance.png) | FORM importance factors α²_i for each mode — horizontal bars colored by sign |
| [`figures/mc_convergence.png`](figures/mc_convergence.png) | Monte Carlo P_f (sliding) vs log₁₀(n) with Wilson 95% CI and FORM/HL reference lines |
| [`figures/marginal_pdfs.png`](figures/marginal_pdfs.png) | Actual marginal PDFs vs equivalent Normal approximations for all 5 random variables |

## Honest limitations

* **All variables treated as Normal in Stages D.3 and D.4.** This is
  the Naccache §5.5.1 convention for HL-FOSM. The cost is quantified:
  HL-FOSM β_sliding = 4.073 vs FORM β_sliding = 4.538 — a factor-of-3
  difference in P_f. Stage F's FORM is the principled fix.
* **Bearing HL-FOSM and FORM do not converge** for the canonical
  scenario's ratio form (`g₁`). The design point lies in a region where
  the rigid-foundation σ_max formula breaks down (resultant beyond the
  base kern). The difference form (`g₂ = R − S`) converges to β = 16.99
  for HL-FOSM — meaning bearing failure is essentially impossible under
  the assumed distributions.
* **Critical wedge D is fixed** at the deterministic value computed
  at the mean of X. The Stage E MC does not re-do the wedge search
  per sample; this assumption is the standard practice in the
  literature but unquantified in this project.
* **|e| = −e in the bearing limit-state.** Valid while eccentricity
  remains negative (resultant on the heel side). Mean eccentricity
  is −0.131 m. Stage E counts the violation: 0.6% of the canonical-
  scenario MC samples produce e > 0. Small but non-zero, documented.
* **Gaussian copula distortion in Stage E.** The X-space correlation
  after marginal transformation differs from the requested ρ by ~1–3%
  for our COVs. Stage F's Nataf transformation corrects this exactly.

## Repository layout

```
src/gabion/
    inputs.py             # WallScenario data model + YAML I/O
    earth_pressure.py     # Coulomb wedge, max-Ea selection
    kinematics.py         # wedge centroid, application points
    checks.py             # FS for sliding, overturning, bearing
    deterministic.py      # run_check end-to-end
    cli.py                # gabion-check console script
    random_variables.py   # 5-variable spec + Nataf transformation
    fosm.py               # MVFOSM + HL-FOSM + FORM (Nataf) + invariance
    monte_carlo.py        # MC + Wilson CI + Ditlevsen bounds
    plots.py              # 4 comparison figures

tests/
    test_inputs.py
    test_deterministic.py
    test_yaml.py
    test_cli.py
    test_random_variables.py
    test_fosm.py
    test_monte_carlo.py

docs/
    baseline_capture.md   # GAWACWIN-vs-implementation comparison
    llm_failure_modes.md  # catalog of 14 LLM error patterns (7 externally confirmed)
    red_team_prompt.md    # standardized benchmark prompt + scoring rubric

figures/
    beta_comparison.png
    form_importance.png
    mc_convergence.png
    marginal_pdfs.png

examples/
    book_example_1.yaml   # canonical scenario for the CLI

run_analysis.py           # interactive full-report script
```

## Running

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run the full test suite (239 tests)
pytest -v

# Run the deterministic check on the canonical scenario
gabion-check examples/book_example_1.yaml

# Run the complete interactive report (all methods + figures)
python run_analysis.py

# Non-interactive with all outputs
python run_analysis.py --defaults --plots

# Programmatic API
python -c "
from gabion.inputs import WallScenario
from gabion.random_variables import book_example_spec
from gabion.fosm import mvfosm, hl_fosm, form
from gabion.monte_carlo import run_monte_carlo

scenario = WallScenario.outside_flat_reference()
spec     = book_example_spec()

mv   = mvfosm(scenario, spec)
hl   = hl_fosm(scenario, spec)
fr   = form(scenario, spec)
mc   = run_monte_carlo(scenario, spec, n_samples=100_000, seed=42)

for mode in ('sliding', 'overturning'):
    print(f'{mode:12}  '
          f'MV={mv[mode].beta:.3f}  '
          f'HL={hl[mode].beta:.3f}  '
          f'FORM={fr[mode].beta:.3f}')
"
```

## References

* **Naccache, E. A. K.** (2016). *Análise probabilística de tubulões à
  compressão*. Master's thesis, USP. — primary reliability reference;
  HL-FOSM iteration follows §5.5.1 (eq. 5.55), system reliability via
  Ditlevsen bi-modal bounds (§5.6).
* **Hasofer, A. M., & Lind, N. C.** (1974). Exact and invariant
  second-moment code format. *Journal of Engineering Mechanics*,
  100(EM1) — original HL-FOSM paper.
* **Liu, P.-L., & Der Kiureghian, A.** (1986). Multivariate distribution
  models with prescribed marginals and covariances. *Probabilistic
  Engineering Mechanics*, 1(2) — Nataf transformation and correlation
  correction used in Stage F.
* **Phoon, K. K., & Kulhawy, F. H.** (1999). Characterization of
  geotechnical variability. *Canadian Geotechnical Journal*, 36(4) —
  source for the COV values used in the random-variable spec.
* **Ditlevsen, O.** (1979). Narrow reliability bounds for structural
  systems. *Journal of Structural Mechanics*, 7(4) — original
  bi-modal bounds paper.
* **Wilson, E. B.** (1927). Probable inference, the law of succession,
  and statistical inference. *Journal of the American Statistical
  Association*, 22(158) — origin of the Wilson score interval.
* **Das, B. M.** *Principles of Foundation Engineering*.
* **Craig, R. F., & Knappett, J. A.** *Soil Mechanics*.
* **Bowles, J. E.** *Foundation Analysis and Design*.
* **Maccaferri.** *Gabion Retaining Walls Design Manual* (GAWACWIN) —
  software output used as cross-validation reference for the
  deterministic engine.
