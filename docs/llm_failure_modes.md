# LLM failure modes in geotechnical reliability problems

A catalog of empirically-observed mistakes that current general-purpose
LLMs make when asked questions about probabilistic stability analysis,
with concrete numerical evidence drawn from the gabion wall analysis
that this repository implements.

## Why this document exists

This project's primary engineering output is a working probabilistic
stability analysis. Its primary *intellectual* output is this
document — a catalog of where the act of doing the analysis exposes
specific, recurring weaknesses in LLM responses to reliability
questions.

The intent is not to claim that LLMs are unhelpful for engineering.
They are very helpful, including for this project. The intent is to
identify points where uncritical acceptance of an LLM's answer would
produce results a competent engineer should reject — and to make those
points concrete with numbers, not handwaving.

All 14 items are backed by code in this repository and reproducible
via `pytest -v` or the scripts referenced in each item. Two production
LLMs have been tested with the standardized multi-stage prompt in
`docs/red_team_prompt.md`: Gemini Pro (2025-05, 7/14 items triggered)
and Claude Sonnet 4.6 (2025-05, 6/14 items triggered). The two tests
together confirm Items 5–7, 9–10, 12, and 14. See the red-team
validation log and comparative analysis at the end of this document.

---

## Part I — Items with empirical evidence

### 1. Conflating FS, β, and P_f as if they ranked failure modes the same way

**The mistake.** Treat factor of safety, reliability index, and
probability of failure as alternative scales for the same underlying
"how safe" measure, expecting them to rank failure modes consistently.

**Why LLMs do it.** All three are scalar safety measures, and most
introductory reliability texts present them in close succession with
phrasing like "a reliability index of β corresponds to a probability of
failure approximately 10⁻⁴…". Easy to read this as "they're equivalent".

**The reality.** They rank failure modes differently when the
contributing variables have very different COVs.

**Evidence.** For the canonical scenario:

| Mode        | FS    | β_HL-FOSM | rank by FS      | rank by β        |
|-------------|-------|-----------|-----------------|------------------|
| Sliding     | 2.484 | 4.073     | 3rd (most safe) | 1st (least safe) |
| Overturning | 5.393 | 9.374     | 1st (most safe) | 2nd              |
| Bearing     | 3.099 | 16.99 †   | 2nd             | 3rd (most safe)  |

_† Using the difference form `g₂ = R − S`; the ratio form does not
converge — see item 4._

The deterministic-vs-probabilistic ranking is **inverted** between
sliding and bearing. Bearing has a low FS (3.10) but the highest β
because the variables driving σ_max — γ_g (COV 0.03) and γ_b (COV 0.05)
— are tightly characterized. Sliding's β is small despite a moderate
FS because φ_b and φ_f have COVs of 0.10 each and the limit-state
function is more sensitive to them.

**The competent answer.** "FS, β, and P_f answer different questions.
A high FS with a low β means the variables that affect this mode are
well-characterized but the mode itself is sensitive to them. The
ranking by β is the one a probabilistic decision should use."

---

### 2. MVFOSM non-invariance — treated as a footnote

**The mistake.** State that MVFOSM and HL-FOSM "give similar β for
linear or mildly nonlinear g" without specifying that "similar" can
mean "differ by a factor of 4".

**Why LLMs do it.** Most undergraduate texts (and many graduate ones)
mention non-invariance briefly, then proceed as if MVFOSM's β is the
canonical answer. The two methods are presented as broadly equivalent.

**The reality.** MVFOSM β depends on the algebraic form of g. The same
physical limit state, written three equivalent ways, can produce wildly
different MVFOSM β.

**Evidence.** For each failure mode, three algebraically equivalent
forms of g (all zero exactly when R = S):

| Mode        | g₁ = R/S − 1 | g₂ = R − S | g₃ = ln(R/S) | MVFOSM spread |
|-------------|--------------|------------|--------------|---------------|
| Sliding     | β = 2.696    | β = 4.164  | β = 4.106    | **+54%**      |
| Overturning | β = 4.908    | β = 22.404 | β = 10.154   | **+356%**     |
| Bearing     | β = 7.076    | β = 21.932 | β = 11.818   | **+210%**     |

Same physical wall. Same random variables. Same correlations. Three
algebraically equivalent g. MVFOSM produces β values that differ by
**up to a factor of 4.6×** for overturning. The corresponding P_f
values, computed naively as Φ(−β), differ by more than a hundred orders
of magnitude.

HL-FOSM, applied to the same three forms:

| Mode        | g₁          | g₂          | g₃          | HL-FOSM spread |
|-------------|-------------|-------------|-------------|----------------|
| Sliding     | β = 4.0732  | β = 4.0732  | β = 4.0732  | **0.0000%**    |
| Overturning | β = 9.3743  | β = 9.3743  | β = 9.3743  | **0.0000%**    |

(For sliding and overturning, where all three forms converge.)

**The competent answer.** "MVFOSM is non-invariant to algebraic
reformulation of the limit-state function. Use HL-FOSM (or higher) for
any decision that needs reproducibility. Cite Ditlevsen 1973 / Hasofer
& Lind 1974 if asked why."

---

### 3. Reporting MVFOSM P_f for a limit state that breaks down at the design point

**The mistake.** Apply MVFOSM uncritically to a limit state whose
mathematical formulation is valid near the mean but invalid at the
linearized "design point", and report the resulting P_f as if it were
meaningful.

**Why LLMs do it.** MVFOSM is a closed-form formula. It always returns
a number. There is no built-in feedback that the number might be
extrapolating outside the model's region of validity.

**The reality.** For the bearing limit state in the canonical scenario,
MVFOSM produces β = 7.08, P_f = 7 × 10⁻¹³ — a number with apparent
meaning out to twelve decimal places. The HL-FOSM iteration, asked to
find the actual design point, fails to converge because the geometric
"closest point on g = 0" lies in a region where the rigid-foundation
formula `σ_max = (N/B)(1 + 6|e|/B)` is not valid: the resultant has
moved beyond the base kern (|e| > B/6) and approaches the edge of the
base (|e| → B/2) where the formula degenerates entirely.

**The competent answer.** "MVFOSM reports β = 7.08 for the bearing
mode, but this β is computed by linearization at the mean and the
linear extrapolation enters a region where the bearing-stress formula
is not physical. The honest interpretation is that bearing failure is
essentially impossible under the assumed distributions; the precise
numerical value of P_f is not meaningful."

This project preserves the failure rather than masking it: HL-FOSM
returns `converged = False` with an explicit `convergence_note` rather
than retrying with looser tolerances or substituting a different number.

---

### 4. Algebraic equivalence ≠ algorithmic equivalence

**The mistake.** Assume that because two forms of g are mathematically
equivalent, they will produce the same numerical results when fed
through a reliability algorithm.

**Why LLMs do it.** Mathematical reasoning treats `R/S − 1`, `R − S`,
and `ln(R/S)` as equivalent — they share the failure surface `R = S`.
The conclusion "therefore the algorithm gives the same answer" is
correct for ideal-precision arithmetic, but incorrect for real
implementations.

**The reality.** For bearing in the canonical scenario:

* `g₁ = R/S − 1`: HL-FOSM **does not converge** with any relaxation
  factor down to 0.1.
* `g₂ = R − S`: HL-FOSM **converges** to β = 16.99 in fewer than 30
  iterations.
* `g₃ = ln(R/S)`: HL-FOSM **does not converge** (the log enters a
  domain error when σ_max becomes very small along the iteration).

Same physical limit state. Different numerical behavior.

The reason is mechanical: `R/S − 1` has a singularity as `S → 0`;
`ln(R/S)` has a singularity as `S → 0` and a domain error if
`R/S < 0`; `R − S` is polynomial in S with no singularities. The HLRF
iteration approaches `S → 0` in the bearing case (σ_max grows toward
q_adm), so the singular forms get into trouble while the regular form
converges.

**The competent answer.** "Algebraic reformulation can change
convergence properties of iterative reliability algorithms. The
difference form `R − S` is generally more robust than ratio or log
forms when one side approaches zero along the design-point trajectory.
HL-FOSM, FORM, and similar methods should be tried in multiple
algebraic forms when convergence is in question."

---

### 5. Sensitivity coefficients vs importance factors — same name, different math

**The mistake.** Use "sensitivity" and "importance" interchangeably,
or report one when the question is about the other.

**Why LLMs do it.** Both quantify "how much does this variable matter".
Different texts use different notation (γ_i, α_i, S_i) and different
normalizations. Without context, the LLM picks one and runs with it.

**The reality.** Two distinct quantities, computed differently, with
different interpretation:

* **Sensitivity coefficient (X-space):**
  γ_i = (∂g/∂X_i · σ_i) / σ_g, evaluated at the linearization point.
  Sign: positive → X_i is resistance-like; negative → load-like.
  γ_i² do **not** sum to 1 when X are correlated.
* **Directional cosine / importance factor (Y-space):**
  α_i = (∇g_Y(y*))_i / ‖∇g_Y(y*)‖, evaluated at the design point.
  Σα_i² = 1 always (unit normal). y* = −α β (Naccache eq. 5.42).

For the sliding mode in the canonical scenario, MVFOSM (γ_i around
the mean) and HL-FOSM (α_i at the design point) report **different
rankings of variable importance** because the gradient direction
shifts as the iteration moves.

**The competent answer.** Identify which one the question is asking
about. For variance budgeting under correlations, use α_i² in Y-space
(unit-sum). For interpretation in original units, use γ_i in X-space
(but report the caveat that γ_i² need not sum to 1).

---

### 6. Counter-intuitive sign of sensitivity for bearing

**The mistake.** State that "stronger soil makes the wall safer in
every mode" — a reasonable-sounding generalization that is false for
bearing.

**Why LLMs do it.** "Stronger soil ↔ smaller active pressure ↔ less
load ↔ safer wall" is the textbook chain. It is correct for sliding
and overturning. For bearing, it has the opposite sign.

**The reality.** Increasing φ_backfill reduces the active thrust Ea,
which reduces M_overturn, which moves the resultant *further from
center* (toward the heel — eccentricity becomes more negative). The
larger |e| concentrates more stress at the heel edge, **increasing
σ_max**, **decreasing** the bearing safety factor.

**Evidence.** MVFOSM sensitivity coefficients for the canonical
scenario:

| Variable        | Sliding γ | Overturning γ | Bearing γ |
|-----------------|-----------|---------------|-----------|
| φ_backfill      | +0.631    | +0.582        | **−0.868** |
| γ_backfill      | −0.143    | −0.145        | −0.075    |
| φ_foundation    | +0.473    |  0 (exact)    |  0 (exact) |
| q (surcharge)   | −0.316    | −0.591        | −0.298    |
| γ_g (stone)     | +0.117    | +0.146        | −0.378    |

For bearing, **all sensitivities are negative or zero except
γ_q = −0.298**. Even γ_g (heavier wall — more weight) is negative for
bearing because more weight, with the existing moment imbalance,
shifts the resultant further from center.

**The competent answer.** "For a wall already in stable equilibrium
with the resultant biased toward the heel, increasing the backfill
strength makes the eccentricity worse, which makes bearing failure
more likely. The bearing limit state is sensitive to the *geometry of
the resultant*, not just to the magnitudes of forces."

**Red-team confirmation (Claude Sonnet 4.6, 2025-05).** Sonnet correctly
shows zero bearing sensitivity for φ_foundation (exact zero, table row)
but reports ∂FS_bearing/∂φ_backfill = +0.083/° — positive, matching the
"stronger soil is always safer" intuition. The correct sign is negative
(MVFOSM reference: γ_bearing = −0.868 for φ_backfill). Notably, Sonnet's
positive sign is a downstream consequence of Item 14: with the wall
centroids placed at the heel side (inside-stepped geometry), the
eccentricity formula produces e > 0 (toe-side) instead of e < 0
(heel-side), which reverses the direction of the eccentricity-φ
sensitivity. This demonstrates that a single geometric error (wrong step
orientation) can silently flip the sign of a physical result.

---

### 7. Defaulting to all-Normal distributions and reporting the resulting β/P_f as final

**The mistake.** Apply FOSM/HL-FOSM with all variables treated as
Normal, even when the spec declares Lognormal (friction angles) or
Gumbel (live loads). Report β and P_f without flagging the
distributional simplification or quantifying its cost.

**Why LLMs do it.** Most reliability worked-examples in textbooks
work in the Normal space. Naccache (2016) §5.5.1 explicitly states
"works only with normal distributions" and treats non-Normal cases
as a separate FORM-with-Nataf chapter. An LLM matching to the most-
common pattern picks Normal as the default and proceeds without
caveat.

**The reality.** The choice of marginal distribution materially
changes the tail mass — which is exactly what reliability analysis
cares about. The cost can be one to two orders of magnitude on P_f,
and the direction (conservative or unconservative) depends on the
distribution shape relative to the equivalent Normal.

**Evidence.** For sliding in the canonical scenario, the chain of
approximations from MVFOSM down to MC with full marginals:

| Method             | Distributions used      | P_f (sliding)    |
|--------------------|-------------------------|------------------|
| MVFOSM             | (moment-based, all-Normal) | 3.5×10⁻³     |
| HL-FOSM            | all Normal              | 2.3×10⁻⁵         |
| MC, all-Normal     | all Normal (full tails) | 1.8×10⁻⁵         |
| MC, mixed marginals | Lognormal/Gumbel/Normal | 1×10⁻⁶ †        |

_† 1 failure observed in 1M samples; 95% Wilson CI [1.8×10⁻⁷, 5.7×10⁻⁶]._

The MC all-Normal and HL-FOSM agree to within 1.3× — confirming HL-FOSM
is doing its job *as a numerical method* on Normal data. The gap from
MC all-Normal to MC mixed is **18×**: that is the cost of the
all-Normal simplification, in this scenario, **conservative**. Lognormal
friction angles have thinner left tails than the equivalent Normals,
so failure events (low φ) are rarer than the all-Normal model predicts.

The total chain MVFOSM → MC mixed is ~3500×, decomposed as roughly
150× from MV→HL (linearization at design point vs at mean) and 18×
from HL all-Normal → MC mixed (distribution shape).

**The competent answer.** "FOSM with moments-treated-as-Normal is
acceptable for first-pass reliability, but the distribution-shape
error can be 1-2 orders of magnitude — sometimes conservative,
sometimes not. Validate against MC with the actual distributions, or
use FORM with Nataf transformation, before reporting P_f as a final
number."

---

### 8. Treating per-mode P_f as if it were the system P_f

**The mistake.** Report only the per-mode probabilities, or naively
sum or maximize them, when the question is about the wall as a system.

**Why LLMs do it.** "P_f sliding = X, P_f overturning = Y, P_f bearing
= Z" reads like a complete answer. Series-system reliability is rarely
covered in undergraduate courses; the LLM may not have strong priors
about when to apply it or how the bounds work.

**The reality.** A wall fails as a system if any single mode fails.
The system probability is P(F_1 ∪ F_2 ∪ F_3), not max(P_i) and not
ΣP_i. Bounds on the union depend on the mode dependencies:

* If modes are independent: P_F = 1 − Π(1 − P_i) ≈ ΣP_i for small P_i.
* If modes are perfectly correlated: P_F = max(P_i).
* In between (the realistic case): P_F lies between the two,
  bracketed by the Ditlevsen bi-modal bounds (Naccache eq. 5.99).

**Evidence.** For the canonical scenario at the mean (1M MC samples,
seed 42):

| Quantity                            | Value           |
|-------------------------------------|-----------------|
| max(P_i) per mode                   | 1.0×10⁻⁶        |
| ΣP_i (independence assumption)      | 1.0×10⁻⁶        |
| Direct system P(any failure) in MC  | 1.0×10⁻⁶        |
| Ditlevsen lower bound               | 1.0×10⁻⁶        |
| Ditlevsen upper bound               | 1.0×10⁻⁶        |

For this particular scenario, all five quantities collapse to the same
value because (a) sliding dominates by 1+ orders of magnitude and
(b) only one failure event was observed across 1M samples. In a more
balanced scenario (e.g., a wall where two modes have similar β), the
gap between max(P_i) and the union, and between independence and
Ditlevsen, would be informative. The mechanism itself — Ditlevsen
bounds bracket the direct estimate — is validated by the property
test in `tests/test_monte_carlo.py`.

**The competent answer.** "Always report system P_f when the question
is about the wall, not about a single mode. Use direct MC for the
ground-truth estimate and Ditlevsen bi-modal bounds when MC is too
expensive or when only the per-mode β values are available. Avoid
`ΣP_i` (independence) and `max(P_i)` (perfect correlation) as
shortcut answers — both are degenerate special cases."

---

### 9. Reporting P_f to too many digits given the actual confidence interval

**The mistake.** Report P_f to four to six decimal places without
showing the confidence interval. Or use a sample size much smaller
than the target P_f calls for and present the result as exact.

**Why LLMs do it.** Numerical outputs have implicit precision (Python
prints to many digits by default). The Naccache rule of thumb
`s ≥ 3 / P_f²` (eq. 5.93, for ~25% CI) is a graduate-level detail
rarely cited in answers.

**The reality.** The Wilson confidence interval on P_f tells you what
the data actually supports. For zero failures in *n* samples, the
upper bound is roughly `z² / n` where `z = 1.96` for 95%
confidence — i.e., you cannot report P_f below `~4/n` without
further evidence.

**Evidence.** From sliding in the canonical scenario, varying *n*:

| *n*      | failures observed | P_f estimate | 95% Wilson CI               |
|----------|-------------------|--------------|-----------------------------|
| 10⁴      | 0                 | 0            | [0, 3.8×10⁻⁴]              |
| 10⁵      | 0                 | 0            | [0, 3.8×10⁻⁵]              |
| 10⁶      | 1                 | 1.0×10⁻⁶    | [1.8×10⁻⁷, 5.7×10⁻⁶]       |

At *n* = 10⁴ the MC tells us essentially nothing about whether P_f is
1×10⁻⁶ or 1×10⁻⁴. An LLM reporting "P_f = 0" from such a run is
technically correct but practically misleading. At *n* = 10⁶, the CI
has a width of about 32× (max/min) — still wide. A "factor-of-30"
P_f estimate is the realistic precision of even a million-sample MC
for events of this rarity.

**The competent answer.** "Report P_f with its confidence interval
(Wilson preferred over normal-approximation for small P_f). The
sample size needed to estimate P_f to within roughly 25% is
`n ≈ 3/P_f²`. For events rarer than ~10⁻⁵, plain MC is impractical —
use FORM, importance sampling, or subset simulation. Never report
P_f to more digits than the CI supports."

---

### 10. Assuming independence between random variables

**The mistake.** Omit the correlation matrix and sample X_i as
independent, or assume that "small" off-diagonal correlations (ρ = 0.3
or 0.5) have negligible effect on β.

**Why LLMs do it.** Introducing a correlation matrix requires
Cholesky decomposition and the concept of covariance propagation.
The calculation looks similar without it, and "small" correlations
feel like they shouldn't matter much.

**The reality.** Correlations change β by modifying the effective
variance of g. Positive correlation between two resistance variables
inflates their joint exceedance probability — meaning the correlated
model is *more conservative* than the independent model.

**Evidence.** HL-FOSM applied to the canonical scenario with and
without the specified correlations (ρ(φ_b, φ_f) = 0.5,
ρ(φ_b, γ_b) = 0.3):

| Mode        | With correlations | Without (diagonal) | Bias       |
|-------------|------------------|--------------------|------------|
| Sliding     | β = 4.073        | β = 4.776          | **+17.3%** |
| Overturning | β = 9.374        | β = 9.137          | −2.5%      |

For sliding, ignoring the φ_b/φ_f correlation (ρ = 0.5) overestimates
β by 17% — an unconservative error. The physical reason: both φ_backfill
and φ_foundation appear in the sliding resistance. When they are
positively correlated, a sample with low φ_b tends to also have low
φ_f, making joint weak-material configurations more likely than
independence predicts. Correlated β is lower (more conservative).

For overturning, φ_foundation does not enter the overturning moment
equation (exactly zero sensitivity), so the ρ(φ_b, φ_f) = 0.5
correlation is irrelevant. The small remaining effect (+2.5% from
ρ(φ_b, γ_b) = 0.3) is minor and goes in the opposite direction because
φ_b and γ_b have opposing effects on overturning.

Code: `tests/test_fosm.py::TestHLFOSM` documents the correlated result.
The uncorrelated comparison can be reproduced by passing
`RandomVariableSpec(variables=spec.variables, correlation_matrix=np.eye(5))`
to `hl_fosm`.

**The competent answer.** "For failure modes involving multiple
variables from the same geological source (here, two friction angles
from the same deposit), dropping the correlation can easily introduce
a 10–20% error in β — unconservative for positive correlations between
resistance variables. Always use the specified correlation matrix."

---

### 11. Cholesky decomposition errors — wrong matrix, wrong orientation

**The mistake.** Apply Cholesky decomposition to generate correlated
samples but use the wrong matrix (e.g., `scipy.linalg.sqrtm` instead
of `numpy.linalg.cholesky`) or the wrong orientation (L^T instead of
L), producing samples with the wrong correlation structure.

**Why LLMs do it.** Multiple conventions exist: `X = μ + L Z` vs
`X = μ + L^T Z`, and `linalg.sqrtm(C)` looks like it should work
because it also produces a matrix square root. The mistake passes
naive sanity checks (sample has correct mean and variance per variable)
but fails covariance checks.

**The reality.** Three distinct errors, each yielding incorrect
cross-correlations:

1. **Using `scipy.linalg.sqrtm` instead of Cholesky:** `sqrtm` returns
   the symmetric matrix square root (where `M = A A` with A symmetric),
   not the lower-triangular Cholesky factor (where `M = L Lᵀ`).
   For a 2×2 matrix with ρ = 0.5, sqrtm gives a different matrix
   than Cholesky; samples generated with sqrtm have the correct
   univariate marginals and the correct pairwise covariances *only if
   the original distribution is multivariate Normal* — but the
   transformation step (ppf route for Lognormal/Gumbel) breaks the
   Normality assumption, making sqrtm's symmetry irrelevant.

2. **Using L^T instead of L:** `numpy.linalg.cholesky` returns lower-
   triangular L such that `C = L Lᵀ`. The correct generation is
   `Z = L U` where U ~ N(0, I). Using `Z = Lᵀ U` generates samples
   with covariance `Lᵀ L ≠ L Lᵀ = C` in general (they are equal only
   for symmetric L, which is only true for diagonal C).

3. **Applying L row-wise:** treating each row of L as a transform on
   the sample array, rather than the column vector `x = μ + L u`.
   This transposes the effective operation and is equivalent to
   mistake 2.

**Verification.** This repository uses `spec.sample_correlated(n, rng)`
which calls `numpy.linalg.cholesky(C_X)` and applies `X = μ + L Z`.
The property test `tests/test_random_variables.py::TestCorrelatedSamples`
checks that the empirical Pearson correlations match the specified ρ
to within `5σ` for n = 100,000 samples, catching all three error types.

**The competent answer.** "Use `numpy.linalg.cholesky` (not `sqrtm`)
and apply `X = μ + L Z` where L is the lower-triangular factor and Z
is a column vector of iid N(0, 1). Verify with an empirical correlation
check on a large sample before trusting the results."

---

### 12. Jensen inequality — E[FS(X)] ≠ FS(E[X])

**The mistake.** Report the deterministic factor of safety FS(μ) as
if it represented the expected performance of the structure, implicitly
treating the wall "designed at the mean" as equivalent to the "mean
design".

**Why LLMs do it.** The deterministic analysis evaluates g at the
mean of the random variables by convention. LLMs (and many practitioners)
read this as "the expected factor of safety under the given variability",
which is a category error. Jensen's inequality says E[FS(X)] ≠ FS(E[X])
for nonlinear FS, and the direction of the gap is not always obvious.

**The reality.** For a convex function f, Jensen's inequality gives
E[f(X)] ≥ f(E[X]). The sign of the gap depends on whether FS is
convex or concave in the region covered by the distribution.

**Evidence.** Monte Carlo estimate of E[FS(X)] vs deterministic FS
at the mean, using 10,000 samples with full marginals:

| Mode        | FS(μ)  | E[FS(X)]  | Gap E[FS] − FS(μ) | Relative |
|-------------|--------|-----------|-------------------|----------|
| Sliding     | 2.484  | 2.558     | **+0.074**        | +3.0%    |
| Overturning | 5.393  | 5.511     | **+0.118**        | +2.2%    |

For this scenario, both limit-state functions are slightly convex in
the region covered by the distributions: E[FS] exceeds FS(μ) by 3%
and 2.2%. The gap is modest here — but this does not mean it is
always modest. For highly nonlinear g or large COVs, the Jensen gap
can reach 10–20% or more, and the direction can be either sign.

The important point is not the magnitude but the *conceptual distinction*:
FS(μ) is a **deterministic design check**, not a probabilistic
expectation. The probabilistic measure of performance is β (or P_f),
which already accounts for the full distribution of X — Jensen's gap is
one contributor to the FS-to-β relationship, alongside distributional
shape effects (item 7) and the difference between linearizing at the
mean vs at the design point (items 1–2).

**The competent answer.** "Deterministic FS evaluated at the mean is
a design convention, not a probabilistic expectation. E[FS(X)] and
FS(E[X]) differ for nonlinear FS by Jensen's inequality; the direction
depends on local convexity. Use β (or P_f) as the probabilistic
performance measure — it is not the same as evaluating g at the mean."

---

### 13. Skipping distributional correction in FORM — reporting HL-FOSM result as FORM result

**The mistake.** Implement HL-RF iteration in standard Normal space
(Y-space) but skip the Nataf transformation, treating all variables as
Normal throughout. Report the result as "FORM" when it is actually
HL-FOSM under a different name.

**Why LLMs do it.** The HL-RF update formula is the same in both
methods. The Nataf step (mapping X → Z → Y via the marginal CDF route
and correcting the correlation matrix) is additional machinery that
many presentations omit. The two methods look nearly identical in
pseudo-code unless the Jacobian step is explicitly called out.

**The reality.** FORM with Nataf uses the actual marginal distributions
to evaluate the probability content at the design point. The gradient
Jacobian ∂x/∂y includes the ratio φ(z_i)/f_{X_i}(x_i), which is unity
only when X_i is Normal. For Lognormal and Gumbel variables, this
ratio differs from 1 — sometimes substantially — at the design point,
changing both the iteration trajectory and the final β.

**Evidence.** Canonical scenario, Nataf-corrected FORM vs all-Normal
HL-FOSM:

| Mode        | HL-FOSM β | FORM (Nataf) β | Δβ        | ΔP_f factor |
|-------------|-----------|----------------|-----------|-------------|
| Sliding     | 4.073     | **4.538**      | **+0.465**| ~3×         |
| Overturning | 9.374     | **9.653**      | **+0.280**| ~3×10⁻³    |

FORM β is higher in both cases: Lognormal friction angles have
thinner left tails than equivalent Normal distributions, so failure
events (low φ) are rarer under the actual distribution. The Nataf
correction shifts the design point deeper into the tail, increasing
β and reducing P_f by a factor of ~3 for sliding.

FORM design-point importance factors (α²_i, from Y-space) confirm
the physics:

| Variable     | Sliding α² | Overturning α² |
|--------------|-----------|----------------|
| φ_backfill   | 0.44      | 0.37           |
| γ_backfill   | 0.03      | 0.00           |
| φ_foundation | 0.15      | 0.00           |
| q (surcharge)| 0.36      | 0.62           |
| γ_g (stone)  | 0.02      | 0.02           |

φ_foundation α² = 0.00 for overturning is exact: the overturning
moment equation has no φ_f term. This zero appears in the FORM result
and also in the MVFOSM sensitivity table (item 6) — it is a structural
property of the limit state, not a numerical artifact.

**The competent answer.** "HL-FOSM and FORM share the same HL-RF
update step, but FORM adds the Nataf transformation to handle non-Normal
marginals. Reporting HL-FOSM β as 'FORM' when the actual distributions
are Lognormal or Gumbel is a common implementation shortcut that
underestimates β by 0.3–0.5 units in this scenario. Check whether the
reported β was computed with or without the CDF-based Jacobian step."

---

### 14. Incomplete deterministic model — cascade into wrong probabilistic conclusion

**The mistake.** Compute the deterministic limit-equilibrium analysis
with missing physical terms, then apply probabilistically-correct
error propagation to the wrong baseline. The probabilistic methodology
looks sound; the error is hidden in the deterministic kernel.

**Why LLMs do it.** Gabion wall analysis involves interdependent
calculations over several steps (Coulomb wedge → forces → moments →
eccentricity → stresses). LLMs often omit terms that have no single
canonical formula — especially the contribution of the vertical Ea
component to the resisting moment, the porosity correction on wall
unit weight, and the correct sign convention for eccentricity. Since
each step is locally plausible, the error compounds without any single
step looking obviously wrong.

**The reality.** The deterministic baseline is the foundation for every
downstream reliability estimate. An error in the deterministic FS
propagates directly into the probabilistic β estimate — even if the
probabilistic method (MVFOSM, FORM) is applied correctly. Furthermore,
an eccentricity with the wrong sign inverts the bearing-capacity
conclusion: a wall where the resultant is safely on the heel side
(e < 0) appears to be on the verge of overturning the footing (e ≫ B/6).

**Evidence.** From a controlled red-team test (Gemini Pro, 2025-05),
applying the multi-stage prompt against the canonical gabion scenario:

| Quantity          | Gemini response | Reference value | Root cause                        |
|-------------------|----------------|-----------------|-----------------------------------|
| W (wall weight)   | 112.5 kN/m     | ~78.75 kN/m     | Porosity n=0.30 omitted from γ_g |
| M_R (resisting)   | 90.56 kNm/m    | 128.65 kNm/m    | Ea vertical component not included in M_R |
| M_OT (overturning)| 40.71 kNm/m    | 23.85 kNm/m     | Ea 14% too high; Y_arm 12% too high |
| e (eccentricity)  | +0.57 m        | **−0.13 m**     | Wrong sign AND 4.4× wrong magnitude |
| FS_overturning    | 2.22           | **5.39**        | Cascade of M_R/M_OT errors        |
| FS_bearing        | 1.48           | **3.10**        | Wrong e → wrong formula → wrong value |

The eccentricity error is the most consequential: the correct wall has
the resultant on the heel side (e = −0.131 m, within the kern B/6 =
0.333 m). Gemini computes e = +0.57 m, outside the kern, triggering the
effective-width formula instead of the Navier formula and making the
bearing mode appear structurally marginal when it is actually the safest
mode.

The cascade is invisible from within the probabilistic analysis. Gemini's
Part B2 correctly applies first-order error propagation to its (wrong)
FS_sliding = 2.05, arriving at P_f ≈ 0.13% with a plausible-looking
σ_FS ≈ 0.35. The method is sound; the input is wrong. The conclusion
also inverts: Gemini identifies bearing as the critical mode (FS = 1.48);
the correct answer is sliding (β = 4.54).

**Additional errors confirmed in the same test (Gemini Pro, 2025-05):**

| Item | Evidence in Gemini response                                               |
|------|---------------------------------------------------------------------------|
| 5    | Sensitivity reported as "High impact. Affects Ka exponentially" — no ∂FS/∂X_i computed |
| 7    | "Assume each parameter is normally distributed" — Lognormal/Gumbel distributions not mentioned |
| 9    | Reports "P ≈ 0.13%" with no confidence interval or sample-size statement |
| 10   | "Parameters are independent" — explicitly stated; ρ(φ_b, φ_f)=0.5 ignored |
| 12   | "FS_sliding = 2.05 ± 0.35 (High confidence)" — treats FS(μ) as E[FS(X)] |
| 13   | "First-Order Reliability Method (FORM) approximation: μ_FS = 2.05, σ_FS from variance sum" — this is MVFOSM, not FORM |

**Second test: Claude Sonnet 4.6, 2025-05**

Sonnet makes a different geometric error: it places the narrow layers
(top) at the **heel side** (centroids x̄₂ = 1.25 m, x̄₃ = 1.50 m from
toe) rather than the **toe side** (x̄₂ = 0.75 m, x̄₃ = 0.50 m), treating
the wall as inside-stepped when the spec says outside-stepped. Sonnet
correctly applies porosity (γ_eff = 17.5 kN/m³) — better than Gemini —
but the centroid error overestimates ΣWi·x̄i by 30.6 kNm/m.

| Quantity            | Sonnet      | Gemini      | Reference     | Sonnet cause                              |
|---------------------|-------------|-------------|---------------|--------------------------------------------|
| W (wall weight)     | 78.75 kN/m  | 112.5 kN/m  | ~78.75 kN/m   | Sonnet correct; Gemini omits porosity      |
| Ea                  | 35.14 kN/m  | 32.75 kN/m  | **28.69 kN/m**| Simplified Ka (vertical face), no batter  |
| N (normal force)    | 82.43 kN/m  | 115.9 kN/m  | 92.67 kN/m    | Correct W but Ea too high; Gemini: both   |
| FS_sliding          | **1.36**    | 2.05        | **2.484**     | Ea +22% → T_drive too large               |
| M_R (resisting)     | 101.42 kNm/m| 90.56 kNm/m | 128.65 kNm/m  | Wrong centroids; missing vertical Ea terms |
| M_OT (overturning)  | 39.67 kNm/m | 40.71 kNm/m | **23.85 kNm/m**| Ea +22% × Y_arm +1%                      |
| FS_overturning      | 2.56        | 2.22        | **5.393**     | Both severely underestimate (cascade)     |
| e (eccentricity)    | **+0.251 m**| **+0.57 m** | **−0.131 m**  | Both: wrong sign; Sonnet closer in magnitude |
| FS_bearing          | 2.77        | 1.48        | **3.099**     | Sonnet −11%; Gemini −52%                  |

The most critical consequence of Sonnet's error: FS_sliding = **1.36**,
which falls below the typical minimum of 1.50. Sonnet correctly concludes
sliding is critical, but reports the wall as **structurally unsafe for
sliding when it actually passes** (FS_ref = 2.484). The P_f for sliding
from Sonnet's FOSM (β = 1.41, P_f = 7.8%) overshoots the correct FORM
value by a factor of **27,000×** (P_f_ref = 2.84×10⁻⁶).

Comparing the two failure types:
- **Gemini** misidentifies the critical mode (says bearing, correct is sliding) but
  produces FS_sliding = 2.05, which at least passes the 1.50 threshold.
- **Sonnet** identifies the correct critical mode but produces FS_sliding = 1.36,
  which causes a **false alarm**: an engineer would redesign a wall that
  actually meets all specifications.

Both are dangerous but in opposite directions: Gemini's error is
structurally unconservative (wrong focus of remediation), Sonnet's is
economically wasteful (unnecessary redesign of a safe structure).

**Shared finding: eccentricity sign error is robust across both tests.**
Both Gemini (e = +0.57 m) and Sonnet (e = +0.251 m) compute positive
eccentricity when the reference gives e = −0.131 m (resultant on heel
side). The two errors arise via completely different calculation paths
(Gemini: porosity omitted → inflated W → wrong M_R/M_OT balance; Sonnet:
wrong centroids → inflated ΣWi·x̄i → different M_R/M_OT balance) yet
both arrive at the same qualitative mistake. This suggests the eccentricity
sign error is a systematic weakness: LLMs tend to place the resultant on
the toe side of this specific wall geometry regardless of calculation path.

**The competent answer.** "Before propagating uncertainty, verify the
deterministic baseline against an independent computation or a hand-
traced check at known values. For gabion walls, three specific terms
are commonly dropped: (1) the vertical component of Ea contributing to
the resisting moment, (2) the porosity correction on the gabion unit
weight, and (3) the sign convention for eccentricity (negative = heel
side = safe; positive = toe side = dangerous). Verify the step orientation
— outside-stepped (steps face backfill, external face vertical) places
each layer aligned at the toe; inside-stepped (steps face front, heel face
vertical) places each layer aligned at the heel. Swapping these changes
the centroid arms by 0.5–1.0 m and can reverse the eccentricity sign."

---

## Red-team validation log

Items confirmed by empirical testing against production LLMs using the
standardized multi-stage prompt in `docs/red_team_prompt.md`.

| Item | Description (short)                   | Gemini Pro | Sonnet 4.6 |
|------|---------------------------------------|------------|------------|
| 5    | Sensitivity vs importance conflation  | ✓          | ~          |
| 6    | Counter-intuitive bearing sign        | —          | ✓          |
| 7    | All-Normal default without caveat     | ✓          | ✓          |
| 9    | P_f precision vs CI mismatch          | ✓          | ✓          |
| 10   | Independence assumption               | ✓          | ~ (caveat) |
| 12   | Jensen inequality conflation          | ✓          | ✓          |
| 13   | MVFOSM labeled as FORM                | ✓          | —          |
| 14   | Incomplete deterministic cascade      | ✓          | ✓          |

Legend: ✓ confirmed · — not triggered · ~ partial (acknowledged but not implemented)  
Date of both tests: 2025-05.

**Score summary:** Gemini Pro: 7/14 · Claude Sonnet 4.6: 6/14  
Items not yet externally confirmed: 1, 2, 3, 4, 8, 11.

---

## Cross-model comparative analysis

Two production LLMs were tested with the identical prompt on the same
canonical scenario. Key contrasts:

**Where Sonnet performed better than Gemini:**
- **Porosity** correctly applied (γ_eff = 17.5 kN/m³); Gemini omitted it (+43% W error)
- **Method label** correct ("FOSM"); Gemini incorrectly called MVFOSM "FORM" (Item 13)
- **Derivatives** computed numerically (finite differences); Gemini gave qualitative descriptions (Item 5)
- **Correlation** acknowledged as a caveat; Gemini did not mention it at all (Item 10)
- **Critical mode** correctly identified (sliding); Gemini identified bearing (wrong)

**Where Gemini performed better than Sonnet:**
- **Centroid geometry** correct (0.805 m from toe); Sonnet placed layers at heel side (+30.6 kNm/m error in ΣWi·x̄i)
- **FS_sliding baseline** closer to reference (2.05 vs 1.36); both wrong but Sonnet's error causes a false structural failure

**Shared errors (robust across both tests):**
- Ea overestimated — Sonnet +22%, Gemini +14%; both use simplified Ka formula
- Eccentricity with wrong sign — Sonnet +0.251 m, Gemini +0.57 m, reference −0.131 m
- FS_overturning severely underestimated — Sonnet 2.56, Gemini 2.22, reference 5.393
- All-Normal assumption without caveat (Items 7, 9, 12)

**Failure direction differs between models:**

| Consequence          | Gemini Pro                   | Sonnet 4.6                   |
|----------------------|------------------------------|------------------------------|
| Critical mode        | Bearing (wrong)              | Sliding (correct)            |
| FS_sliding judgment  | 2.05 — passes (wrong value)  | 1.36 — **false FAIL**        |
| P_f_sliding error    | ~5× overestimate             | **27,000× overestimate**     |
| Engineering risk     | Misses the real failure mode | Triggers unnecessary redesign|

Gemini's error is **structurally unconservative**: wrong mode identified → remediation effort misdirected. Sonnet's error is **economically wasteful**: correct mode identified but wrong FS causes false alarm → engineer redesigns a wall that meets all specifications at baseline conditions.

Neither error type is acceptable for structural engineering decisions. Both illustrate that methodological sophistication (Sonnet's numerical derivatives, correct method labeling) does not protect against geometric input errors whose downstream effects can be orders of magnitude larger than the input error itself.

---

## Methodology notes

Each item above is backed by code in this repository. The numerical
evidence comes from `tests/test_fosm.py` and `tests/test_monte_carlo.py`
(which lock the regression baselines for the values cited in the tables)
and from the worked examples in `compare_mvfosm_invariance` (Stage D.5),
`run_monte_carlo` (Stage E), and `form` (Stage F). Items 5–14 have been
additionally confirmed through a controlled red-team test against a
production LLM (see validation log above).

The catalog deliberately excludes items where the LLM mistake is so
generic that nothing about this project sharpens it (e.g.,
"hallucinated reference"). Each item has to be one where running the
actual gabion analysis surfaces a concrete failure that generic
critique misses.
