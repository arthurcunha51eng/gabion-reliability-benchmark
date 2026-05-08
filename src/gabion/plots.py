"""Visualization for the gabion-stability probabilistic analysis.

Generates four publication-quality figures comparing all reliability methods:

    1. beta_comparison   — β per mode, three methods side by side
    2. form_importance   — FORM importance factors α²_i per mode
    3. mc_convergence    — P_f_sliding vs log10(n_samples) with Wilson CIs
    4. marginal_pdfs     — actual PDFs vs Normal approximation for each variable

Usage (standalone)::

    python -m gabion.plots               # saves to figures/
    python -m gabion.plots --show        # also opens interactive windows
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm as sci_norm

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "figure.dpi":      130,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ---------------------------------------------------------------------------
# Colour / label maps
# ---------------------------------------------------------------------------
_COLORS = {
    "MVFOSM":  "#2196F3",
    "HL-FOSM": "#FF9800",
    "FORM":    "#4CAF50",
    "MC":      "#9C27B0",
}
_VAR_COLORS = {
    "phi_backfill":   "#E53935",
    "gamma_backfill": "#FB8C00",
    "phi_foundation": "#43A047",
    "q":              "#1E88E5",
    "gamma_g":        "#8E24AA",
}
_VAR_LABELS = {
    "phi_backfill":   r"$\varphi_{b}$  (backfill friction)",
    "gamma_backfill": r"$\gamma_{b}$  (backfill weight)",
    "phi_foundation": r"$\varphi_{f}$  (foundation friction)",
    "q":              r"$q$  (surcharge)",
    "gamma_g":        r"$\gamma_{g}$  (stone weight)",
}
_VAR_LABELS_SHORT = {
    "phi_backfill":   r"$\varphi_{b}$",
    "gamma_backfill": r"$\gamma_{b}$",
    "phi_foundation": r"$\varphi_{f}$",
    "q":              r"$q$",
    "gamma_g":        r"$\gamma_{g}$",
}

_SYMBOL_NAMES = [
    "phi_backfill", "gamma_backfill", "phi_foundation", "q", "gamma_g"
]


# ---------------------------------------------------------------------------
# Figure 1 — β comparison
# ---------------------------------------------------------------------------
def plot_beta_comparison(mv, hl, fr, ax=None):
    """Grouped bar chart: β per mode × method.

    Only sliding and overturning are shown; bearing HL-FOSM and FORM do
    not converge (same physical singularity as HL-FOSM Stage D.4).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    modes = ["sliding", "overturning"]
    mode_labels = ["Sliding", "Overturning"]
    methods = [("MVFOSM", mv), ("HL-FOSM", hl), ("FORM", fr)]

    x = np.arange(len(modes))
    width = 0.22
    offsets = np.array([-1, 0, 1]) * width

    for (method, res), offset in zip(methods, offsets):
        betas = []
        for m in modes:
            b = res[m].beta
            betas.append(b if (b is not None and not np.isnan(b)) else 0.0)

        bars = ax.bar(x + offset, betas, width * 0.9,
                      label=method, color=_COLORS[method],
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar, beta in zip(bars, betas):
            if beta > 0.1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{beta:.2f}",
                    ha="center", va="bottom", fontsize=8.5, zorder=4,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=12)
    ax.set_ylabel("Reliability index β  (= ‖y*‖ in standard-normal space)")
    ax.set_title("Reliability index β by method and failure mode\n"
                 "(MVFOSM linearizes at means; HL-FOSM and FORM iterate to design point)")
    ax.legend(framealpha=0.9)
    max_beta = max(fr["overturning"].beta, hl["overturning"].beta)
    ax.set_ylim(0, max_beta * 1.13)
    ax.set_xlim(-0.5, len(modes) - 0.5)
    return ax


# ---------------------------------------------------------------------------
# Figure 2 — FORM importance factors
# ---------------------------------------------------------------------------
def plot_form_importance(fr, fig=None, axes=None):
    """Two-panel horizontal bar chart: FORM α²_i per mode.

    Sign annotation shows whether the variable is resistance-like (+) or
    load-like (−) at the design point.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, mode in zip(axes, ["sliding", "overturning"]):
        r = fr[mode]
        if r.alpha is None:
            ax.text(0.5, 0.5, "Not converged", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
            ax.set_title(mode.capitalize())
            continue

        alphas = [r.alpha[v] for v in _SYMBOL_NAMES]
        alpha_sq = [a ** 2 for a in alphas]
        labels = [_VAR_LABELS[v] for v in _SYMBOL_NAMES]
        colors = [_VAR_COLORS[v] for v in _SYMBOL_NAMES]

        y = np.arange(len(_SYMBOL_NAMES))
        h_bars = ax.barh(y, alpha_sq, color=colors, edgecolor="white",
                         linewidth=0.6, zorder=3)

        for bar, alpha in zip(h_bars, alphas):
            sign_str = "resistance ↑" if alpha >= 0 else "load ↑"
            x_pos = bar.get_width() + 0.012
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    "+" if alpha >= 0 else "−",
                    va="center", fontsize=13,
                    color="#1B5E20" if alpha >= 0 else "#B71C1C")
            ax.text(x_pos + 0.04, bar.get_y() + bar.get_height() / 2,
                    f"{alpha_sq[list(alphas).index(alpha)]:.3f}",
                    va="center", fontsize=9, color="gray")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(r"Importance factor $\alpha_i^2$  (sums to 1.0)")
        ax.set_title(
            f"FORM — {mode.capitalize()}\n"
            f"β = {r.beta:.4f},  P_f = {r.pf:.2e}",
            fontsize=11,
        )
        ax.set_xlim(0, max(alpha_sq) * 1.28)
        ax.axvline(0, color="black", linewidth=0.8)

        # Annotate total = 1.0
        total = sum(alpha_sq)
        ax.text(0.99, 0.02, f"Σα²_i = {total:.6f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="gray")

    if fig is not None:
        fig.suptitle(
            "FORM importance factors: contribution of each random variable to β\n"
            "(sign: + = resistance-like, − = load-like at design point)",
            fontsize=11, y=1.02,
        )
    return fig, axes


# ---------------------------------------------------------------------------
# Figure 3 — MC convergence
# ---------------------------------------------------------------------------
def plot_mc_convergence(scenario, spec, seed=42, ax=None):
    """P_f_sliding vs log10(n_samples) with Wilson 95% CI bands.

    Each point is an independent MC run at that sample size (same seed,
    growing the sample). Shows how the estimate and its uncertainty
    evolve with n. Reference lines from HL-FOSM and FORM are overlaid.
    """
    from gabion.monte_carlo import run_monte_carlo, _wilson_ci
    from gabion.fosm import hl_fosm, form as form_func

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    sample_sizes = [1_000, 3_000, 10_000, 30_000, 100_000]
    pf_estimates, ci_lo, ci_hi, failures = [], [], [], []

    print("    MC convergence runs:", end=" ", flush=True)
    for n in sample_sizes:
        result = run_monte_carlo(scenario, spec, n_samples=n, seed=seed)
        pf = result.pf_per_mode["sliding"]
        lo, hi = result.pf_ci_per_mode["sliding"]
        pf_estimates.append(pf)
        ci_lo.append(lo)
        ci_hi.append(hi)
        k = round(pf * result.n_valid)
        failures.append(k)
        print(f"n={n//1000}k({k}f)", end=" ", flush=True)
    print()

    log_n = np.log10(sample_sizes)

    # Wilson CI band
    ax.fill_between(log_n,
                    [np.log10(v) if v > 0 else -10 for v in ci_lo],
                    [np.log10(v) if v > 0 else -10 for v in ci_hi],
                    alpha=0.2, color=_COLORS["MC"], label="Wilson 95% CI")

    # Point estimates where failures > 0
    for idx, (lx, pf, k) in enumerate(zip(log_n, pf_estimates, failures)):
        if pf > 0:
            ax.plot(lx, np.log10(pf), "o", color=_COLORS["MC"], markersize=8, zorder=5)
        else:
            ax.annotate("0 failures\n(→ upper bound)",
                        xy=(lx, np.log10(ci_hi[idx])),
                        fontsize=8, color=_COLORS["MC"], ha="center", va="bottom")

    # Reference lines from FORM and HL-FOSM
    hl_res = hl_fosm(scenario, spec)
    fr_res = form_func(scenario, spec)
    ax.axhline(np.log10(hl_res["sliding"].pf), color=_COLORS["HL-FOSM"],
               linestyle="--", linewidth=1.5,
               label=f"HL-FOSM  P_f = {hl_res['sliding'].pf:.1e}")
    ax.axhline(np.log10(fr_res["sliding"].pf), color=_COLORS["FORM"],
               linestyle="-.", linewidth=1.5,
               label=f"FORM     P_f = {fr_res['sliding'].pf:.1e}")

    ax.set_xlabel("log₁₀(n_samples)")
    ax.set_ylabel("log₁₀(P_f_sliding)")
    ax.set_title("Monte Carlo convergence — sliding mode\n"
                 "(Wilson 95% CI shrinks as n grows; FORM/HL lines are closed-form estimates)")
    ax.set_xticks(log_n)
    ax.set_xticklabels([f"10^{int(x)}" for x in log_n])
    ax.legend(framealpha=0.9)
    return ax


# ---------------------------------------------------------------------------
# Figure 4 — Marginal PDFs
# ---------------------------------------------------------------------------
def plot_marginal_pdfs(spec, fig=None, axes=None):
    """Five-panel: actual marginal PDFs vs equivalent Normal.

    Highlights why distributional shape matters: Lognormal friction angles
    have thinner left tails than their Normal approximations, which is why
    the all-Normal HL-FOSM overestimates P_f by ~18× for sliding.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.8))

    var_axes = [
        ("phi_backfill",   r"$\varphi_b$ (°)",         None, None),
        ("gamma_backfill", r"$\gamma_b$ (kN/m³)",       None, None),
        ("phi_foundation", r"$\varphi_f$ (°)",          None, None),
        ("q",              r"$q$ (kN/m²)",              0.01, None),
        ("gamma_g",        r"$\gamma_g$ (kN/m³)",       None, None),
    ]

    for ax, (name, xlabel, x_lo_override, _) in zip(axes, var_axes):
        rv = spec[name]
        mu, sigma = rv.mean, rv.std
        x_lo = x_lo_override if x_lo_override else mu - 4 * sigma
        x_hi = mu + 4 * sigma
        x = np.linspace(x_lo, x_hi, 400)

        # Actual distribution PDF
        pdf_actual = rv.pdf(x)
        ax.plot(x, pdf_actual, color=_VAR_COLORS[name], linewidth=2.2,
                label="Actual", zorder=3)

        # Equivalent Normal PDF
        pdf_norm = sci_norm.pdf(x, loc=mu, scale=sigma)
        ax.plot(x, pdf_norm, "--", color="gray", linewidth=1.5,
                label="Normal approx.", alpha=0.85)

        # Mean line
        ax.axvline(mu, color="black", linestyle=":", linewidth=1.2, alpha=0.7)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)

        # Distribution label in corner
        dist_label = rv.distribution.capitalize()
        ax.text(0.97, 0.97, dist_label, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=_VAR_COLORS[name])

        if name == "phi_backfill":
            ax.legend(fontsize=8, loc="upper left")

    if fig is not None:
        fig.suptitle(
            "Marginal PDFs: actual distributions vs Normal approximation\n"
            "(Lognormal left tail is thinner → all-Normal FOSM over-estimates P_f)",
            fontsize=10, y=1.03,
        )
        plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Master driver
# ---------------------------------------------------------------------------
def generate_all_plots(
    save_dir: str = "figures",
    n_mc: int = 100_000,
    seed: int = 42,
    show: bool = False,
) -> None:
    """Run all methods on the canonical scenario and write 4 PNG figures."""
    from gabion.inputs import WallScenario
    from gabion.random_variables import book_example_spec
    from gabion.fosm import mvfosm, hl_fosm, form as form_func
    from gabion.monte_carlo import run_monte_carlo

    os.makedirs(save_dir, exist_ok=True)

    print("  Loading canonical scenario...")
    scenario = WallScenario.outside_flat_reference()
    spec = book_example_spec()

    print("  Running MVFOSM...")
    mv = mvfosm(scenario, spec)
    print("  Running HL-FOSM...")
    hl = hl_fosm(scenario, spec)
    print("  Running FORM (Nataf)...")
    fr = form_func(scenario, spec)
    print(f"    FORM beta_sliding = {fr['sliding'].beta:.4f}, "
          f"beta_overturning = {fr['overturning'].beta:.4f}")

    # ── Figure 1: β comparison ────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    plot_beta_comparison(mv, hl, fr, ax=ax1)
    fig1.tight_layout()
    _save(fig1, save_dir, "beta_comparison.png", show)

    # ── Figure 2: FORM importance factors ────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_form_importance(fr, fig=fig2, axes=axes2)
    fig2.tight_layout()
    _save(fig2, save_dir, "form_importance.png", show)

    # ── Figure 3: MC convergence ─────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    print("  Running MC convergence series...")
    plot_mc_convergence(scenario, spec, seed=seed, ax=ax3)
    fig3.tight_layout()
    _save(fig3, save_dir, "mc_convergence.png", show)

    # ── Figure 4: Marginal PDFs ────────────────────────────────────────
    fig4, axes4 = plt.subplots(1, 5, figsize=(14, 3.8))
    plot_marginal_pdfs(spec, fig=fig4, axes=axes4)
    _save(fig4, save_dir, "marginal_pdfs.png", show)

    print(f"\n  All figures saved to '{save_dir}/'")


def _save(fig, save_dir, name, show):
    path = os.path.join(save_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gabion-stability plots")
    parser.add_argument("--save-dir", default="figures", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display interactively")
    parser.add_argument("--n-mc", type=int, default=100_000, help="MC samples for convergence plot")
    args = parser.parse_args()
    generate_all_plots(save_dir=args.save_dir, show=args.show, n_mc=args.n_mc)
