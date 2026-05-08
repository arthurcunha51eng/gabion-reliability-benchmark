#!/usr/bin/env python3
"""
Gabion Wall Stability Analysis — Interactive Report Generator
=============================================================

Runs the complete analysis pipeline (deterministic → MVFOSM → HL-FOSM →
FORM → Monte Carlo) on a user-specified wall scenario and prints a
comprehensive engineering report.

Usage::

    python run_analysis.py                  # canonical scenario, prompts for n_MC
    python run_analysis.py --defaults       # fully non-interactive
    python run_analysis.py --plots          # also generate comparison figures

Report covers:
    • Deterministic limit equilibrium: Ea, T_drive, T_resist, FS per mode
    • MVFOSM / HL-FOSM / FORM: β and P_f per mode
    • FORM design points x* and importance factors α²_i
    • Monte Carlo system P_f with Wilson 95% CI and Ditlevsen bounds
    • Jensen inequality: E[FS(X)] vs FS(E[X])
"""
from __future__ import annotations

import argparse
import math
import sys
import textwrap

# Force UTF-8 output on Windows (cp1252 cannot encode β, →, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _ask(prompt: str, default, cast=float, non_interactive: bool = False):
    """Prompt the user; press Enter to accept the default."""
    if non_interactive:
        return default
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"    Invalid input — using default: {default}")
        return default


def _hr(char="=", width=72):
    print(char * width)


def _section(title: str):
    print()
    _hr()
    print(f"  {title}")
    _hr()


def _sub(title: str):
    print()
    print(f"  ── {title}")
    print()


def _row(label: str, value: str, unit: str = "", width: int = 42):
    unit_str = f" {unit}" if unit else ""
    print(f"  {label:<{width}s} {value}{unit_str}")


# ── Scenario setup ────────────────────────────────────────────────────────────

def _get_scenario(non_interactive: bool):
    from gabion.inputs import (
        WallScenario, GabionMaterial, WallGeometry, SoilProperties,
    )

    ref = WallScenario.outside_flat_reference()

    if non_interactive:
        return ref

    _section("WALL SCENARIO  — press Enter to accept each default")

    print("\n  Geometry:")
    raw = _ask(
        "Layer lengths from base to top, comma-separated (m)",
        ",".join(str(x) for x in ref.geometry.layer_lengths),
        cast=str,
    )
    try:
        layer_lengths = [float(v.strip()) for v in raw.split(",")]
    except ValueError:
        layer_lengths = list(ref.geometry.layer_lengths)
    beta_geom = _ask("Wall batter from vertical β (°)", ref.geometry.beta)

    print("\n  Gabion material:")
    gamma_g = _ask("Unit weight γ_g (kN/m³)", ref.gabion.gamma_g)
    n_por   = _ask("Porosity n (−)",           ref.gabion.n)
    geotex  = _ask("Geotextile reduction (−)", ref.gabion.geotex_reduction)

    print("\n  Backfill:")
    gamma_b = _ask("Unit weight γ_b (kN/m³)",    ref.backfill.gamma)
    phi_b   = _ask("Friction angle φ_b (°)",     ref.backfill.phi)
    c_b     = _ask("Cohesion c_b (kPa)",         ref.backfill.c)

    print("\n  Foundation:")
    gamma_f = _ask("Unit weight γ_f (kN/m³)",    ref.foundation.gamma)
    phi_f   = _ask("Friction angle φ_f (°)",     ref.foundation.phi)
    c_f     = _ask("Cohesion c_f (kPa)",         ref.foundation.c)

    print("\n  Loads:")
    q     = _ask("Surcharge q (kN/m²)",              ref.q)
    q_adm = _ask("Allowable bearing q_adm (kPa)",    ref.q_adm)

    return WallScenario(
        gabion=GabionMaterial(gamma_g=gamma_g, n=n_por, geotex_reduction=geotex),
        geometry=WallGeometry(layer_lengths=layer_lengths, beta=beta_geom),
        backfill=SoilProperties(gamma=gamma_b, phi=phi_b, c=c_b),
        foundation=SoilProperties(gamma=gamma_f, phi=phi_f, c=c_f),
        q=q,
        q_adm=q_adm,
    )


def _get_spec(scenario, non_interactive: bool):
    from gabion.random_variables import (
        RandomVariable, RandomVariableSpec, book_example_spec,
    )

    ref = book_example_spec()

    if non_interactive:
        return ref

    _section("RANDOM-VARIABLE SPEC — distributions are fixed; press Enter for defaults")
    print()
    print("  Distributions: φ → Lognormal,  γ → Normal,  q → Gumbel")
    print()

    rvs = []
    for rv in ref.variables:
        mean_new = _ask(f"{rv.name:<25s} mean", rv.mean)
        cov_new  = _ask(f"{rv.name:<25s} COV",  rv.cov)
        rvs.append(RandomVariable(
            name=rv.name, distribution=rv.distribution,
            mean=mean_new, cov=cov_new,
        ))

    print("\n  Correlations (off-diagonal; all others assumed zero):")
    rho_pb_pf = _ask("ρ(φ_backfill, φ_foundation)", ref.correlation_matrix[0, 2])
    rho_pb_gb = _ask("ρ(φ_backfill, γ_backfill)",   ref.correlation_matrix[0, 1])

    d = len(rvs)
    rho = np.eye(d)
    rho[0, 2] = rho[2, 0] = rho_pb_pf
    rho[0, 1] = rho[1, 0] = rho_pb_gb

    return RandomVariableSpec(variables=tuple(rvs), correlation_matrix=rho)


# ── Jensen inequality helper ─────────────────────────────────────────────────

def _compute_mean_fs(scenario, spec, n=50_000, seed=42):
    """E[FS(X)] via MC.  Uses lambdified g; FS = g + 1."""
    from gabion.deterministic import run_check
    from gabion.fosm import build_limit_states, _build_lambdified

    det = run_check(scenario)
    ls = build_limit_states(scenario, d_critical=det.D_critical)
    g_s_func, _ = _build_lambdified(ls.sliding)
    g_ot_func, _ = _build_lambdified(ls.overturning)

    rng = np.random.default_rng(seed)
    X = spec.sample_correlated(n, rng)   # shape (n, 5)

    g_s  = np.array([g_s_func(*X[i])  for i in range(n)], dtype=float)
    g_ot = np.array([g_ot_func(*X[i]) for i in range(n)], dtype=float)

    return {
        "E_FS_sliding":     float(np.nanmean(g_s)  + 1),
        "E_FS_overturning": float(np.nanmean(g_ot) + 1),
    }


# ── Report printer ────────────────────────────────────────────────────────────

def _nan_str(val, fmt=".3f"):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "  n.c."
    return format(val, fmt)


def print_report(scenario, det, mv, hl, fr, mc, mean_fs=None):
    from gabion.fosm import SYMBOL_NAMES
    from scipy.stats import norm as sci_norm

    W = 72
    print()
    _hr("=")
    print("  GABION WALL PROBABILISTIC STABILITY REPORT")
    _hr("=")

    # ── Deterministic ────────────────────────────────────────────────────
    _section("SECTION 1 — DETERMINISTIC LIMIT EQUILIBRIUM")

    _sub("Wall geometry")
    _row("Total height H",           f"{scenario.geometry.H:.2f}", "m")
    _row("Base width B",             f"{scenario.geometry.L_base:.2f}", "m")
    _row("Layer lengths (base→top)", str(scenario.geometry.layer_lengths), "m")
    _row("Batter angle β",           f"{scenario.geometry.beta:.1f}", "°")
    _row("Gabion unit weight γ_g",   f"{scenario.gabion.gamma_g:.1f}", "kN/m³")
    _row("Porosity n",               f"{scenario.gabion.n:.2f}", "")
    _row("Backfill φ_b / γ_b",
         f"{scenario.backfill.phi:.1f}° / {scenario.backfill.gamma:.1f}", "kN/m³")
    _row("Foundation φ_f",           f"{scenario.foundation.phi:.1f}", "°")
    _row("Surcharge q",              f"{scenario.q:.1f}", "kN/m²")
    _row("Allowable bearing q_adm",  f"{scenario.q_adm:.1f}", "kPa")

    _sub("Critical Coulomb wedge")
    _row("Wedge depth D_critical",   f"{det.D_critical:.3f}", "m")
    _row("Wedge angle ρ",            f"{det.rho:.3f}", "°")
    _row("Active thrust Ea",         f"{det.Ea:.3f}", "kN/m")
    _row("Thrust inclination θ",     f"{det.theta:.3f}", "°")
    _row("Application point (X, Y)", f"({det.X_Ea:.4f}, {det.Y_Ea:.4f})", "m")

    _sub("Sliding check  (FS ≥ 1.5)")
    _row("Normal force N",           f"{det.N:.3f}", "kN/m")
    _row("Driving force T_drive",    f"{det.T_drive:.3f}", "kN/m")
    _row("Resisting force T_resist", f"{det.T_resist:.3f}", "kN/m")
    ok_s  = "✓" if det.FS_sliding >= 1.5 else "✗"
    _row("FS_sliding",               f"{det.FS_sliding:.3f}  {ok_s}", "")

    _sub("Overturning check  (FS ≥ 2.0)")
    _row("Overturning moment M_OT",  f"{det.M_overturning:.3f}", "kN·m/m")
    _row("Resisting moment M_R",     f"{det.M_resisting:.3f}", "kN·m/m")
    ok_ot = "✓" if det.FS_overturning >= 2.0 else "✗"
    _row("FS_overturning",           f"{det.FS_overturning:.3f}  {ok_ot}", "")

    _sub("Bearing capacity check  (σ_max ≤ q_adm)")
    _row("Eccentricity e",           f"{det.eccentricity:.4f}", "m")
    _row("σ_max  (Navier formula)",  f"{det.sigma_max:.3f}", "kPa")
    _row("σ_min",                    f"{det.sigma_min:.3f}", "kPa")
    fs_b = scenario.q_adm / det.sigma_max
    ok_b  = "✓" if fs_b >= 1.0 else "✗"
    _row("FS_bearing  (q_adm / σ_max)", f"{fs_b:.3f}  {ok_b}", "")

    # ── β table ──────────────────────────────────────────────────────────
    _section("SECTION 2 — RELIABILITY INDICES β")

    fs_vals = {
        "sliding":     det.FS_sliding,
        "overturning": det.FS_overturning,
        "bearing":     scenario.q_adm / det.sigma_max,
    }

    print()
    hdr = f"  {'Mode':<14} {'FS':>6}  {'β_MVFOSM':>10}  {'β_HLFOSM':>10}  {'β_FORM':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for mode in ("sliding", "overturning", "bearing"):
        b_mv = _nan_str(mv[mode].beta)
        b_hl = _nan_str(hl[mode].beta)
        b_fr = _nan_str(fr[mode].beta)
        print(f"  {mode.capitalize():<14} {fs_vals[mode]:>6.3f}  "
              f"{b_mv:>10}  {b_hl:>10}  {b_fr:>10}")

    print()
    print(textwrap.fill(
        "  Note: HL-FOSM and FORM treat all variables as Normal and "
        "Lognormal/Gumbel respectively. n.c. = not converged (design point "
        "in physically invalid region — see llm_failure_modes.md §3–4).",
        width=70, initial_indent="  ", subsequent_indent="  ",
    ))

    # ── P_f table ─────────────────────────────────────────────────────────
    _section("SECTION 3 — FAILURE PROBABILITIES P_f")

    print()
    hdr2 = (f"  {'Mode':<14} {'P_f MVFOSM':>12}  {'P_f HL-FOSM':>12}  "
            f"{'P_f FORM':>12}  {'P_f MC (95% CI)':>24}")
    print(hdr2)
    print("  " + "─" * (len(hdr2) - 2))

    for mode in ("sliding", "overturning", "bearing"):
        def _pf_str(res):
            v = res[mode].pf
            if v is None or math.isnan(v):
                return "       n.c."
            return f"{v:.2e}"

        lo, hi = mc.pf_ci_per_mode[mode]
        pf_mc = mc.pf_per_mode[mode]
        if pf_mc > 0:
            mc_str = f"{pf_mc:.1e} [{lo:.1e},{hi:.1e}]"
        else:
            mc_str = f"  0  [0, {hi:.1e}]"

        print(f"  {mode.capitalize():<14}"
              f"  {_pf_str(mv):>12}"
              f"  {_pf_str(hl):>12}"
              f"  {_pf_str(fr):>12}"
              f"  {mc_str:>24}")

    # ── MC system ─────────────────────────────────────────────────────────
    _section("SECTION 4 — MONTE CARLO SYSTEM RELIABILITY")

    _row("MC sample count (n)",        f"{mc.n_samples:>10,}", "")
    _row("Valid samples",              f"{mc.n_valid:>10,}  ({100*mc.n_valid/mc.n_samples:.2f}%)")
    _row("P_f system (direct MC)",     f"{mc.pf_system:.3e}", "")
    _row("  95% Wilson CI",            f"[{mc.pf_system_ci[0]:.2e}, {mc.pf_system_ci[1]:.2e}]")
    _row("Ditlevsen lower bound",      f"{mc.pf_ditlevsen_lower:.3e}", "")
    _row("Ditlevsen upper bound",      f"{mc.pf_ditlevsen_upper:.3e}", "")
    _row("Bearing e>0 violations",
         f"{mc.n_eccentricity_positive} / {mc.n_valid}  "
         f"({100*mc.n_eccentricity_positive/max(mc.n_valid,1):.2f}%)")

    # ── Jensen inequality ─────────────────────────────────────────────────
    if mean_fs is not None:
        _section("SECTION 5 — JENSEN INEQUALITY  E[FS(X)] vs FS(E[X])")
        print()
        print("  FS is a nonlinear function of X.  By Jensen's inequality,")
        print("  E[FS(X)] ≠ FS(E[X]).  Computing both from MC samples (n=50k):")
        print()
        _row("  FS_sliding  at E[X] (deterministic)",
             f"{det.FS_sliding:.4f}", "(linearization point)")
        _row("  E[FS_sliding(X)]  (MC mean)",
             f"{mean_fs['E_FS_sliding']:.4f}", "(convex → higher)")
        delta_s = mean_fs["E_FS_sliding"] - det.FS_sliding
        _row("  Difference  E[FS] − FS(E[X])",
             f"{delta_s:+.4f}", f"({100*delta_s/det.FS_sliding:+.2f}%)")
        print()
        _row("  FS_overturning  at E[X] (deterministic)",
             f"{det.FS_overturning:.4f}", "")
        _row("  E[FS_overturning(X)]  (MC mean)",
             f"{mean_fs['E_FS_overturning']:.4f}", "")
        delta_ot = mean_fs["E_FS_overturning"] - det.FS_overturning
        _row("  Difference  E[FS] − FS(E[X])",
             f"{delta_ot:+.4f}", f"({100*delta_ot/det.FS_overturning:+.2f}%)")

    # ── FORM design points ────────────────────────────────────────────────
    _section("SECTION 6 — FORM DESIGN POINTS AND IMPORTANCE FACTORS")

    for mode in ("sliding", "overturning"):
        r = fr[mode]
        if r.design_point is None:
            print(f"\n  {mode.capitalize()}: not converged")
            continue

        _sub(f"{mode.capitalize()} mode — design point x*  "
             f"(β = {r.beta:.4f}, P_f = {r.pf:.2e})")

        NAMES = list(r.design_point.keys())
        means = {rv.name: rv.mean for rv in r.alpha.keys() if False} or \
                {n: m for n, m in zip(NAMES, [30., 18., 30., 10., 25.])}

        print(f"  {'Variable':<22} {'Mean':>8}  {'Design pt':>10}  "
              f"{'Δ/σ':>8}  {'α_i':>8}  {'α²_i':>8}  sign")
        print("  " + "─" * 68)

        spec_means = {rv.name: rv.mean for rv in
                      __import__("gabion.random_variables",
                                 fromlist=["book_example_spec"]
                                 ).book_example_spec().variables}
        spec_stds  = {rv.name: rv.std for rv in
                      __import__("gabion.random_variables",
                                 fromlist=["book_example_spec"]
                                 ).book_example_spec().variables}

        for name in NAMES:
            dp   = r.design_point[name]
            mu   = spec_means.get(name, 0)
            sig  = spec_stds.get(name, 1)
            d_sig = (dp - mu) / sig if sig else 0
            alpha = r.alpha[name]
            sign_str = "resist +" if alpha >= 0 else "load  −"
            print(f"  {name:<22} {mu:>8.3f}  {dp:>10.4f}  "
                  f"{d_sig:>+8.3f}  {alpha:>+8.4f}  {alpha**2:>8.4f}  {sign_str}")

    _hr("=")
    print("  END OF REPORT")
    _hr("=")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gabion stability — complete probabilistic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--defaults", action="store_true",
        help="Use canonical scenario non-interactively (no prompts)",
    )
    parser.add_argument(
        "--n-mc", type=int, default=0,
        help="MC sample count (0 = prompt the user)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for MC",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate and save comparison figures to figures/",
    )
    parser.add_argument(
        "--no-jensen", action="store_true",
        help="Skip Jensen-inequality computation (saves ~10 s)",
    )
    args = parser.parse_args()

    ni = args.defaults  # non-interactive flag

    print()
    _hr("=")
    print("  GABION WALL STABILITY — Complete Probabilistic Analysis")
    print("  Deterministic  |  MVFOSM  |  HL-FOSM  |  FORM (Nataf)  |  Monte Carlo")
    _hr("=")

    # ── Build scenario ───────────────────────────────────────────────────
    if not ni:
        use_ref = _ask("Use the canonical reference scenario? (y/n)", "y",
                       cast=str, non_interactive=ni)
        if use_ref.strip().lower().startswith("y"):
            from gabion.inputs import WallScenario
            scenario = WallScenario.outside_flat_reference()
            ni_spec = True
        else:
            scenario = _get_scenario(non_interactive=False)
            ni_spec = False
    else:
        from gabion.inputs import WallScenario
        scenario = WallScenario.outside_flat_reference()
        ni_spec = True

    if not ni:
        if ni_spec:
            use_ref_spec = _ask("Use the canonical random-variable spec? (y/n)", "y",
                                cast=str, non_interactive=False)
            if use_ref_spec.strip().lower().startswith("y"):
                from gabion.random_variables import book_example_spec
                spec = book_example_spec()
            else:
                spec = _get_spec(scenario, non_interactive=False)
        else:
            spec = _get_spec(scenario, non_interactive=False)
    else:
        from gabion.random_variables import book_example_spec
        spec = book_example_spec()

    # ── Monte Carlo settings ─────────────────────────────────────────────
    if args.n_mc > 0:
        n_mc = args.n_mc
    elif ni:
        n_mc = 100_000
    else:
        n_mc = int(_ask("Number of MC samples", 100_000, cast=float))
    seed = args.seed

    # ── Run analysis ─────────────────────────────────────────────────────
    from gabion.deterministic import run_check
    from gabion.fosm import mvfosm, hl_fosm, form
    from gabion.monte_carlo import run_monte_carlo

    print()
    print("  [1/5] Deterministic limit equilibrium...")
    det = run_check(scenario)
    print(f"        FS_sliding={det.FS_sliding:.3f}  "
          f"FS_overturning={det.FS_overturning:.3f}  "
          f"FS_bearing≈{scenario.q_adm/det.sigma_max:.3f}")

    print("  [2/5] MVFOSM (Cornell linearization at means)...")
    mv = mvfosm(scenario, spec)
    print(f"        β_sliding={mv['sliding'].beta:.3f}  "
          f"β_overturning={mv['overturning'].beta:.3f}")

    print("  [3/5] HL-FOSM (Hasofer-Lind, HLRF in X-space)...")
    hl = hl_fosm(scenario, spec)
    print(f"        β_sliding={hl['sliding'].beta:.3f}  "
          f"β_overturning={hl['overturning'].beta:.3f}")

    print("  [4/5] FORM (HL-RF in Y-space via Nataf transformation)...")
    fr = form(scenario, spec)
    print(f"        β_sliding={fr['sliding'].beta:.4f}  "
          f"β_overturning={fr['overturning'].beta:.4f}")

    print(f"  [5/5] Monte Carlo ({n_mc:,} samples, seed={seed})...")
    mc = run_monte_carlo(scenario, spec, n_samples=n_mc, seed=seed)
    print(f"        P_f_system={mc.pf_system:.2e}  "
          f"(sliding failures: {round(mc.pf_per_mode['sliding']*mc.n_valid)})")

    # ── Jensen inequality ─────────────────────────────────────────────────
    mean_fs = None
    if not args.no_jensen:
        print("  [+]   Jensen inequality E[FS(X)] (50k MC samples)...")
        try:
            mean_fs = _compute_mean_fs(scenario, spec, n=50_000, seed=seed)
            print(f"        E[FS_sliding]={mean_fs['E_FS_sliding']:.4f}  "
                  f"vs  FS(E[X])={det.FS_sliding:.4f}")
        except Exception as exc:
            print(f"        (Jensen computation failed: {exc})")
            mean_fs = None

    # ── Print report ──────────────────────────────────────────────────────
    print_report(scenario, det, mv, hl, fr, mc, mean_fs=mean_fs)

    # ── Optional plots ────────────────────────────────────────────────────
    do_plots = args.plots
    if not ni and not args.plots:
        ans = _ask("Generate comparison figures? (y/n)", "n", cast=str)
        do_plots = ans.strip().lower().startswith("y")

    if do_plots:
        print()
        print("  Generating figures...")
        try:
            from gabion.plots import generate_all_plots
            generate_all_plots(save_dir="figures", n_mc=min(n_mc, 100_000),
                               seed=seed, show=False)
        except Exception as exc:
            print(f"  Warning: figure generation failed — {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)
