"""Tests for the gabion-check CLI.

Two flavors:

* In-process tests via ``cli.main(argv)`` are fast and let pytest
  capture stdout natively. They cover argument parsing, output format,
  and the happy path.
* Subprocess tests via ``python -m gabion.cli`` confirm the package is
  importable as a module and the entry point works end-to-end. We keep
  one of these as a smoke test; running everything as subprocess would
  triple test time for marginal extra coverage.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gabion.cli import main as cli_main


REFERENCE_YAML = Path(__file__).resolve().parent.parent / "examples" / "book_example_1.yaml"


# -----------------------------------------------------------------------------
# In-process: fast, native stdout capture, easy argument variations.
# -----------------------------------------------------------------------------
class TestCliInProcess:
    def test_runs_on_reference(self, capsys):
        rc = cli_main([str(REFERENCE_YAML)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Sliding check" in out
        assert "FS_s" in out

    def test_json_output_is_parseable(self, capsys):
        rc = cli_main([str(REFERENCE_YAML), "--format", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # Must expose the same field names asserted in test_deterministic.py.
        for key in ("D_critical", "Ea", "FS_sliding", "FS_overturning",
                    "sigma_max", "sigma_min"):
            assert key in data

    def test_json_full_precision(self, capsys):
        # The text report rounds; the JSON output must keep full precision
        # so that downstream tooling (notebooks, MC layers) is not lossy.
        cli_main([str(REFERENCE_YAML), "--format", "json"])
        data = json.loads(capsys.readouterr().out)
        # Baseline AUTO Ea = 28.69243604229485 — at least 10 sig figs in the dump.
        assert abs(data["Ea"] - 28.69243604229485) < 1e-10

    def test_d_override_changes_critical_wedge(self, capsys):
        cli_main([str(REFERENCE_YAML), "--d-override", "2.5", "--format", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["D_critical"] == pytest.approx(2.5, abs=1e-9)
        # And FS_sliding must match the D=2.5 baseline.
        assert data["FS_sliding"] == pytest.approx(2.5487747995767833, rel=1e-4)

    def test_text_report_marks_mode(self, capsys):
        # AUTO mode label
        cli_main([str(REFERENCE_YAML)])
        out_auto = capsys.readouterr().out
        assert "AUTO" in out_auto

        # Forced mode label
        cli_main([str(REFERENCE_YAML), "--d-override", "2.5"])
        out_forced = capsys.readouterr().out
        assert "FORCED" in out_forced or "forced" in out_forced.lower()


# -----------------------------------------------------------------------------
# Subprocess: smoke test that the entry point works as advertised.
# -----------------------------------------------------------------------------
class TestCliSubprocess:
    def test_invocable_as_python_module(self):
        proc = subprocess.run(
            [sys.executable, "-m", "gabion.cli",
             str(REFERENCE_YAML), "--format", "json"],
            capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data["D_critical"] == pytest.approx(2.0, abs=1e-9)
