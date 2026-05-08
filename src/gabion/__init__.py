"""gabion-stability — probabilistic stability analysis of gabion gravity walls.

Complete implementation covering four levels of analysis:

    Stage B-C  Deterministic limit equilibrium (Coulomb wedge, 3 FS modes)
    Stage D    FOSM: Cornell MVFOSM + Hasofer-Lind HL-FOSM + invariance demo
    Stage E    Monte Carlo with Cholesky-correlated mixed-marginal samples
    Stage F    FORM via Nataf transformation (HL-RF in Y-space)

Public API::

    from gabion.inputs        import WallScenario
    from gabion.random_variables import book_example_spec, NatafTransformation
    from gabion.deterministic import run_check
    from gabion.fosm          import mvfosm, hl_fosm, form, FosmResult
    from gabion.monte_carlo   import run_monte_carlo, MonteCarloResult
    from gabion.plots         import generate_all_plots

Quick start::

    scenario = WallScenario.outside_flat_reference()
    spec     = book_example_spec()
    det      = run_check(scenario)
    mv_res   = mvfosm(scenario, spec)
    hl_res   = hl_fosm(scenario, spec)
    form_res = form(scenario, spec)
    mc_res   = run_monte_carlo(scenario, spec, n_samples=100_000, seed=42)
"""

__version__ = "1.0.0"
