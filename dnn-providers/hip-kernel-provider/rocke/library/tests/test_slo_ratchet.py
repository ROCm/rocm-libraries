# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Runs the SLO linter (tools/slo) against its committed baseline so that
# tools/run_checks.py, which auto-discovers library/tests, runs it too.
#
# WARN ONLY. Numbers that got worse than tools/slo/baseline.json, and new files
# over a target, are reported as pytest warnings. They never fail this test.
# The test fails only when the tool itself cannot run (exit code 2): a file that
# does not parse, or a broken baseline. Enforcement comes later, by switching
# FAIL_ON to "increase" once the rocKE owners agree the targets.

from __future__ import annotations

import importlib
import json
import sys
import warnings
from pathlib import Path

import pytest

_SLO = (
    Path(__file__).resolve().parents[2] / "tools" / "slo"
)  # tests -> library -> rocke
FAIL_ON = "never"


def test_slo_numbers_not_worse_than_baseline(tmp_path: Path) -> None:
    if not (_SLO / "slo.py").is_file():
        pytest.skip("tools/slo not present in this copy")
    sys.path.insert(0, str(_SLO))
    try:
        slo = importlib.import_module("slo")
    finally:
        sys.path.remove(str(_SLO))

    out = tmp_path / "slo.json"
    rc = slo.main(["check", "--fail-on", FAIL_ON, "--json", str(out)])
    assert rc != 2, "the SLO linter could not run; see its stderr above"

    findings = json.loads(out.read_text())["findings"]
    for f in findings:
        if f["kind"] in ("worse", "over_target"):
            warnings.warn(
                f"{f['check']} {f['path']}:{f['line']} {f['message']}", stacklevel=1
            )
    assert rc == 0
