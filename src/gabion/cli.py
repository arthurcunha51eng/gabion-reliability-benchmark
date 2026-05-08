"""Command-line interface for gabion-stability.

Usage
-----
::

    gabion-check examples/book_example_1.yaml
    gabion-check examples/book_example_1.yaml --d-override 2.5
    gabion-check examples/book_example_1.yaml --format json

The CLI is intentionally thin: it parses arguments, loads a YAML
scenario, calls ``deterministic.run_check``, and formats the output.
All real work happens in the library; this module is here so that
``run_check`` becomes invocable from a terminal or a CI pipeline
without writing a Python driver.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from gabion.deterministic import DeterministicResult, run_check
from gabion.inputs import WallScenario


def _format_text_report(scenario: WallScenario, result: DeterministicResult,
                        d_override: Optional[float]) -> str:
    """Build a human-readable report mirroring GAWACWIN's section layout."""
    layers = scenario.geometry.layer_lengths
    mode = (
        f"AUTO (max Ea over D ∈ [0.5, 5.0] m)"
        if d_override is None
        else f"FORCED at D = {d_override:.2f} m"
    )

    lines = [
        "=" * 78,
        "  Gabion wall stability check — outside-stepped, flat backfill".ljust(78),
        "=" * 78,
        "",
        "Scenario",
        f"  Layers (base → top)        : {layers} m",
        f"  Wall height H              : {scenario.geometry.H:.2f} m",
        f"  Wall batter β              : {scenario.geometry.beta:.2f}°",
        f"  Backfill                   : "
        f"γ={scenario.backfill.gamma} kN/m³,  "
        f"φ'={scenario.backfill.phi}°,  "
        f"c'={scenario.backfill.c} kPa",
        f"  Foundation                 : "
        f"γ={scenario.foundation.gamma} kN/m³,  "
        f"φ'={scenario.foundation.phi}°,  "
        f"c'={scenario.foundation.c} kPa",
        f"  Surcharge q                : {scenario.q} kN/m²",
        f"  Allowable bearing q_adm    : {scenario.q_adm} kPa",
        "",
        f"Critical wedge selection: {mode}",
        f"  D                          : {result.D_critical:.3f} m",
        f"  ρ                          : {result.rho:.2f}°",
        f"  Ea                         : {result.Ea:.2f} kN/m",
        f"  θ (above horizontal)       : {result.theta:.2f}°",
        "",
        "Sliding check",
        f"  Normal force N             : {result.N:.2f} kN/m",
        f"  Driving force T_drive      : {result.T_drive:.2f} kN/m",
        f"  Resisting force T_resist   : {result.T_resist:.2f} kN/m",
        f"  Factor of safety FS_s      : {result.FS_sliding:.3f}",
        "",
        "Overturning check",
        f"  Driving moment             : {result.M_overturning:.2f} kN·m/m",
        f"  Resisting moment           : {result.M_resisting:.2f} kN·m/m",
        f"  Factor of safety FS_o      : {result.FS_overturning:.3f}",
        "",
        "Foundation (bearing pressure, Navier)",
        f"  Eccentricity e             : {result.eccentricity:+.3f} m   "
        f"(B/6 = {scenario.geometry.L_base / 6.0:.3f} m)",
        f"  σ_max                      : {result.sigma_max:.2f} kPa   "
        f"(allowable: {scenario.q_adm:.1f} kPa)",
        f"  σ_min                      : {result.sigma_min:.2f} kPa   "
        f"(must be ≥ 0)",
        "=" * 78,
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gabion-check",
        description=(
            "Run a deterministic stability check on a gabion gravity wall."
        ),
    )
    parser.add_argument(
        "scenario_file",
        type=Path,
        help="Path to a YAML scenario file (see examples/book_example_1.yaml).",
    )
    parser.add_argument(
        "--d-override",
        type=float,
        default=None,
        metavar="D",
        help=(
            "Force the critical wedge to this D [m] (for cross-software "
            "comparison). Default: maximize Ea over a 0.5 m grid."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    args = parser.parse_args(argv)

    scenario = WallScenario.from_yaml(args.scenario_file)
    result = run_check(scenario, d_override=args.d_override)

    if args.format == "text":
        print(_format_text_report(scenario, result, args.d_override))
    else:  # json
        # Use json.dumps so floats render as `28.69243604229485` rather
        # than the truncated `28.69` you would get from formatting them
        # in the report — JSON consumers want the full precision.
        print(json.dumps(asdict(result), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
