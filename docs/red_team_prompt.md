# Red-Team Prompt — Standardized Benchmark for LLM Geotechnical Reliability

A controlled, multi-stage prompt designed to expose the failure patterns
documented in `llm_failure_modes.md`. All input values are taken directly
from the canonical gabion wall scenario (`examples/book_example_1.yaml`).

**Purpose:** generate a single LLM response that can be scored against the
14-item failure-mode taxonomy. Each section of the prompt is engineered to
trigger a specific set of errors at the transition points between stages.

**Usage:** submit the prompt verbatim to any general-purpose LLM, collect
the full response, and score it against the Benchmark Scoring section below.

---

## Prompt

```
PART A — DETERMINISTIC STABILITY ANALYSIS
==========================================

You are a geotechnical engineer. A gabion wall has these fixed parameters:

GEOMETRY:
  Total height H = 3.0 m
  Base width B = 2.0 m
  Layer lengths (base to top) = [2.0, 1.5, 1.0] m  (OUTSIDE-stepped: steps
    face the backfill, external face is vertical)
  Batter angle β = 6° from vertical
  Gabion unit weight γ_g = 25 kN/m³  (stone fill — apply porosity n=0.30
    to get effective bulk weight)
  Porosity n = 0.30

SOIL / FOUNDATION:
  Backfill friction angle φ_b = 30°
  Backfill unit weight γ_b = 18 kN/m³
  Foundation friction angle φ_f = 30°
  Foundation cohesion c_f = 0
  Surcharge q = 10 kN/m²
  Allowable bearing pressure q_adm = 200 kPa

TASK A1: Compute the active earth pressure (Coulomb wedge analysis)
  — Find critical wedge depth D that maximizes Ea
  — Report: Ea (kN/m), wedge inclination θ, application point (X, Y)

TASK A2: Check sliding stability
  — Normal force N at base
  — Driving force T_drive
  — Resisting force T_resist = N·tan(φ_f)
  — Report: FS_sliding = T_resist / T_drive

TASK A3: Check overturning stability
  — Resisting moment M_R (include wall weight, surcharge, and vertical
    component of Ea)
  — Overturning moment M_OT from horizontal Ea component
  — Report: FS_overturning = M_R / M_OT

TASK A4: Check bearing capacity
  — Eccentricity e = B/2 − (M_R − M_OT)/N
    (note: negative e means resultant is on the heel side — safe)
  — Maximum stress σ_max using the Navier formula:
    σ_max = (N/B)·(1 + 6|e|/B)   [valid only if |e| ≤ B/6]
  — Report: FS_bearing = q_adm / σ_max

Provide step-by-step calculations for all three modes.
Show all intermediate values. Report final FS for each mode.


PART B — PARAMETRIC UNCERTAINTY ANALYSIS
=========================================

The above analysis used nominal values. Now consider that five parameters
are uncertain:

UNCERTAIN PARAMETERS:
  φ_backfill:   mean = 30°,  standard deviation = 3°
  γ_backfill:   mean = 18 kN/m³,  SD = 0.9 kN/m³
  φ_foundation: mean = 30°,  SD = 3°
  q (surcharge): mean = 10 kN/m²,  SD = 3 kN/m²
  γ_g (gabion): mean = 25 kN/m³,  SD = 0.75 kN/m³

All other parameters (geometry, q_adm, porosity) are deterministic.

TASK B1: Sensitivity analysis
  — For each of the 5 uncertain parameters, compute ∂FS/∂X_i at the mean
  — Identify which parameters most strongly affect each failure mode
  — Rank them by sensitivity magnitude

TASK B2: Probabilistic interpretation
  — For sliding only: estimate the probability that FS_sliding < 1.0
  — Show your method and all assumptions explicitly

TASK B3: Synthesis — rank the three failure modes by reliability
  — Consider both deterministic FS and the effect of parameter uncertainty
  — Which mode is most likely to govern long-term safety?
  — State whether your ranking changed from Part A and why.


PART C — JUDGMENT AND SUMMARY
==============================

Based on your full analysis:

1. Which failure mode is most critical? One sentence.
2. Estimate engineering confidence in each FS given the parameter
   uncertainty. (Express as FS ± something and explain what that ± means.)
3. What additional information would most improve this assessment?
```

---

## Reference Values (for scoring)

From `run_analysis.py --defaults`, canonical scenario:

| Quantity            | Reference value | Notes                                      |
|---------------------|-----------------|--------------------------------------------|
| Ea                  | 28.69 kN/m      | Coulomb wedge with max-Ea search           |
| θ (thrust inclination) | 24.0°        | Wall batter + wall friction                |
| Y_Ea (application) | 1.119 m         | Weighted centroid of triangle + rectangle  |
| N (normal force)    | 92.67 kN/m      | W + Ea·sin(θ)                              |
| FS_sliding          | **2.484**       |                                            |
| M_OT               | 23.85 kNm/m     |                                            |
| M_R                 | 128.65 kNm/m    | Includes Ea vertical component             |
| FS_overturning      | **5.393**       |                                            |
| e (eccentricity)    | **−0.131 m**    | Negative = heel side = safe; within kern   |
| σ_max               | 64.53 kPa       | Navier formula (|e| < B/6 = 0.333 m)      |
| FS_bearing          | **3.099**       |                                            |
| β_FORM sliding      | 4.538           | Lognormal φ, Gumbel q, ρ(φ_b,φ_f)=0.5    |
| β_FORM overturning  | 9.653           |                                            |
| P_f FORM sliding    | 2.84×10⁻⁶       |                                            |

**Critical mode ranking by β:** Sliding (4.54) < Overturning (9.65) << Bearing (n.c., very large)  
**Critical mode ranking by FS:** Sliding (2.48) < Bearing (3.10) < Overturning (5.39)  
FS and β rankings **invert between sliding and bearing** — see Item 1 in `llm_failure_modes.md`.

---

## Benchmark Scoring

Score each item 1 (error present) or 0 (correct or not triggered):

| Item | Trigger condition                                                                                          |
|------|------------------------------------------------------------------------------------------------------------|
| 1    | Ranks modes by FS in Part B3/C1, placing bearing or overturning as "most critical" instead of sliding     |
| 2    | Reports a single β value per mode without mentioning that MVFOSM β depends on algebraic form of g        |
| 3    | Accepts MVFOSM bearing P_f (7×10⁻¹³) as a meaningful number without convergence caveat                   |
| 4    | Claims that g₁=R/S−1 and g₂=R−S will produce identical numerical results in an iterative algorithm       |
| 5    | Describes sensitivity as "high/medium/low" without computing ∂FS/∂X_i; or conflates γ_i with α_i²        |
| 6    | States "higher φ_backfill makes the wall safer in every mode" (ignores bearing sign reversal)             |
| 7    | Estimates P_f in B2 with all variables Normal, without mentioning that the spec has Lognormal/Gumbel       |
| 8    | Reports per-mode P_f as the final answer without framing it as system P_f via union/Ditlevsen bounds      |
| 9    | Reports P_f to 2+ significant figures without a confidence interval or sample-size qualification          |
| 10   | Drops the correlation ρ(φ_b, φ_f)=0.5 without noting its ~17% effect on β_sliding                       |
| 11   | Sketches correlated sampling code using `sqrtm` or `L.T` instead of Cholesky lower-triangular factor L   |
| 12   | States that FS(μ) equals the "expected" or "average" factor of safety under uncertainty                   |
| 13   | Labels MVFOSM (variance propagation at the mean) as "FORM" or "First-Order Reliability Method"           |
| 14   | Produces wrong FS values due to omitted physical terms (porosity, incomplete moments, wrong eccentricity sign) |

**Scoring thresholds:**
- 0–3: Methodologically aware; errors are minor or isolated
- 4–6: Systematic gaps; response is usable with expert review
- 7–10: Pervasive misunderstanding; not usable without full recalculation
- 11–14: Fundamentally incorrect; dangerous if taken at face value

---

## Key Transition Points (where errors cluster)

| Transition              | Expected error class             | Items triggered    |
|-------------------------|----------------------------------|--------------------|
| A1 → A2 (wall weight)   | Porosity omission                | 14                 |
| A3 (moment balance)     | Incomplete M_R (missing Ea_v)    | 14                 |
| A4 (eccentricity sign)  | Wrong sign → wrong formula       | 14                 |
| A → B (det → prob)      | Jensen conflation                | 12                 |
| B1 (sensitivity)        | γ_i vs α_i² confusion           | 5                  |
| B2 (distribution)       | All-Normal default               | 7, 13              |
| B2 (precision)          | P_f without CI                   | 9                  |
| B2 (independence)       | Correlation matrix dropped       | 10                 |
| B3 (synthesis)          | FS-based ranking                 | 1, 6               |
| C2 (confidence)         | Symmetric ± on nonlinear g       | 12                 |

---

## Confirmed Test Results

| LLM                  | Date    | Items triggered          | Score | Critical error                                                        |
|----------------------|---------|--------------------------|-------|-----------------------------------------------------------------------|
| Gemini Flash Lite (v3.1)    | 2025-05 | 5, 7, 9, 10, 12, 13, 14  | 7/14  | Eccentricity sign inverted; bearing identified as critical mode (wrong) |
| Claude Sonnet 4.6    | 2025-05 | 6, 7, 9, 10~, 12, 14     | 6/14  | Inside-stepped geometry → FS_sliding = 1.36 (false FAIL vs correct 2.484) |

_~ partial credit: Item 10 acknowledged as caveat, not implemented._

---

## Cross-model error comparison

| Quantity            | Gemini Pro  | Sonnet 4.6  | Reference   |
|---------------------|-------------|-------------|-------------|
| Porosity applied    | No (γ=25.0) | Yes (γ=17.5)| Yes         |
| Layer centroids     | Correct     | Heel-biased | Toe-aligned |
| Ea (kN/m)           | 32.75 (+14%)| 35.14 (+22%)| 28.69       |
| FS_sliding          | 2.05 (−17%) | **1.36 (−45%) FAIL** | 2.484 |
| FS_overturning      | 2.22 (−59%) | 2.56 (−53%) | 5.393       |
| e (eccentricity)    | +0.57 ✗     | +0.251 ✗    | **−0.131**  |
| FS_bearing          | 1.48 (−52%) | 2.77 (−11%) | 3.099       |
| Critical mode ID    | Bearing ✗   | Sliding ✓   | Sliding     |
| Method label        | "FORM" ✗    | "FOSM" ✓    | —           |
| Sensitivity method  | Qualitative | ∂FS/∂Xi numerical ✓ | — |

**Shared failure across both models:**
- Ea overestimated (simplified Ka, no batter correction)
- Eccentricity with wrong sign (both +, correct is −)
- FS_overturning severely underestimated (~2.4 vs 5.4)
- All-Normal FOSM without Lognormal/Gumbel caveat
- No confidence interval on reported P_f

**Key asymmetry:** Gemini and Sonnet fail in opposite safety directions. Gemini misidentifies the critical mode (structurally risky). Sonnet correctly identifies the mode but declares the wall unsafe when it is not (economically wasteful). See `llm_failure_modes.md` comparative analysis for full discussion.
