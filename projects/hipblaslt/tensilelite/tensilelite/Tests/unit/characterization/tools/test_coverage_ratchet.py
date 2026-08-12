# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the coverage ratchet (AIHPBLAS-3878).

These lock the ratchet's decision logic: a per-file drop beyond tolerance is a
regression, a rise or an in-tolerance wobble is not, removed/added files are
handled, and malformed input is a setup error rather than a false "pass". The
ratchet is the enforcement mechanism, so its own behavior must be pinned.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent
_MODULE_PATH = _TOOLS_DIR / "coverage_ratchet.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("coverage_ratchet", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ratchet = _load_module()


def _cov_json(pcts: dict[str, float]) -> dict:
    """Build a minimal coverage.py-shaped JSON report from {path: percent}."""
    return {
        "meta": {"format": 3},
        "files": {
            path: {"summary": {"percent_covered": pct}} for path, pct in pcts.items()
        },
        "totals": {"percent_covered": sum(pcts.values()) / len(pcts) if pcts else 0.0},
    }


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# per_file_coverage                                                           #
# --------------------------------------------------------------------------- #
def test_per_file_coverage_extracts_percentages():
    cov = _cov_json({"a.py": 90.0, "b.py": 12.5})
    assert ratchet.per_file_coverage(cov) == {"a.py": 90.0, "b.py": 12.5}


def test_per_file_coverage_rejects_non_report():
    with pytest.raises(ratchet.RatchetError):
        ratchet.per_file_coverage({"not": "a coverage report"})


def test_per_file_coverage_rejects_malformed_entry():
    with pytest.raises(ratchet.RatchetError):
        ratchet.per_file_coverage({"files": {"a.py": {"summary": {}}}})


# --------------------------------------------------------------------------- #
# find_regressions                                                            #
# --------------------------------------------------------------------------- #
def test_rise_is_not_a_regression():
    baseline = {"a.py": 80.0}
    current = {"a.py": 95.0}
    assert ratchet.find_regressions(baseline, current, tolerance=0.1) == []


def test_equal_is_not_a_regression():
    baseline = {"a.py": 80.0}
    current = {"a.py": 80.0}
    assert ratchet.find_regressions(baseline, current, tolerance=0.1) == []


def test_drop_beyond_tolerance_is_flagged_with_values():
    baseline = {"a.py": 90.0}
    current = {"a.py": 84.0}
    result = ratchet.find_regressions(baseline, current, tolerance=0.1)
    assert result == [("a.py", 90.0, 84.0)]


def test_drop_within_tolerance_is_ignored():
    baseline = {"a.py": 90.0}
    current = {"a.py": 89.95}  # 0.05 pp drop, under the 0.1 pp tolerance
    assert ratchet.find_regressions(baseline, current, tolerance=0.1) == []


def test_removed_file_is_not_a_regression():
    # File deleted from the source tree: absent from the current report.
    baseline = {"gone.py": 100.0, "a.py": 80.0}
    current = {"a.py": 80.0}
    assert ratchet.find_regressions(baseline, current, tolerance=0.1) == []


def test_new_file_is_ignored_until_next_update():
    baseline = {"a.py": 80.0}
    current = {"a.py": 80.0, "new.py": 10.0}
    assert ratchet.find_regressions(baseline, current, tolerance=0.1) == []


def test_multiple_regressions_sorted_biggest_drop_first():
    baseline = {"a.py": 90.0, "b.py": 90.0}
    current = {"a.py": 88.0, "b.py": 70.0}  # b drops 20, a drops 2
    result = ratchet.find_regressions(baseline, current, tolerance=0.1)
    assert [row[0] for row in result] == ["b.py", "a.py"]


# --------------------------------------------------------------------------- #
# DEFAULT_TOLERANCE (the noise buffer)                                        #
# --------------------------------------------------------------------------- #
def test_default_tolerance_absorbs_sub_arc_noise():
    # The real case this buffer exists for: develop deleted 7 covered statements
    # from a 770-unit file, moving it 88.16 -> 88.05 with identical missed
    # statements and branch coverage. Nothing became less tested, so it must not
    # fail the gate.
    baseline = {"Tensile/Contractions.py": 88.16}
    current = {"Tensile/Contractions.py": 88.05}
    assert (
        ratchet.find_regressions(baseline, current, ratchet.DEFAULT_TOLERANCE) == []
    )


def test_default_tolerance_still_catches_a_real_regression():
    # The buffer is wide, not absent: a drop past it is still a failure.
    baseline = {"a.py": 90.0}
    current = {"a.py": 88.5}  # 1.5 pp, past the 1 pp buffer
    assert ratchet.find_regressions(
        baseline, current, ratchet.DEFAULT_TOLERANCE
    ) == [("a.py", 90.0, 88.5)]


def test_committed_baseline_tolerance_matches_the_default():
    # cmd_check reads the tolerance from the baseline while cmd_update writes
    # DEFAULT_TOLERANCE. If the two drift apart, the next `update` silently
    # retunes the gate, so pin them together.
    committed = json.loads(
        (_TOOLS_DIR.parent / "coverage-baseline.json").read_text(encoding="utf-8")
    )
    assert committed["tolerance"] == ratchet.DEFAULT_TOLERANCE


# --------------------------------------------------------------------------- #
# write_baseline / round-trip                                                 #
# --------------------------------------------------------------------------- #
def test_write_baseline_round_trips_and_rounds(tmp_path):
    out = tmp_path / "coverage-baseline.json"
    ratchet.write_baseline({"a.py": 90.126, "b.py": 12.5}, out, tolerance=0.1)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["tolerance"] == 0.1
    assert saved["files"] == {"a.py": 90.13, "b.py": 12.5}


def test_write_baseline_creates_missing_parent_dir(tmp_path):
    out = tmp_path / "nested" / "coverage-baseline.json"
    ratchet.write_baseline({"a.py": 50.0}, out, tolerance=0.1)
    assert out.is_file()


# --------------------------------------------------------------------------- #
# cmd_check / cmd_update (end-to-end via argparse namespaces)                 #
# --------------------------------------------------------------------------- #
def _args(**kw):
    return argparse.Namespace(**kw)


def test_cmd_check_passes_when_no_regression(tmp_path):
    baseline = _write(
        tmp_path / "base.json", {"tolerance": 0.1, "files": {"a.py": 80.0}}
    )
    current = _write(tmp_path / "cov.json", _cov_json({"a.py": 85.0}))
    rc = ratchet.cmd_check(
        _args(baseline=str(baseline), current=str(current), tolerance=None)
    )
    assert rc == 0


def test_cmd_check_fails_and_names_offender(tmp_path, capsys):
    baseline = _write(
        tmp_path / "base.json", {"tolerance": 0.1, "files": {"pkg/a.py": 90.0}}
    )
    current = _write(tmp_path / "cov.json", _cov_json({"pkg/a.py": 70.0}))
    rc = ratchet.cmd_check(
        _args(baseline=str(baseline), current=str(current), tolerance=None)
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "pkg/a.py" in err
    assert "coverage_ratchet.py" in err  # remediation command is printed


def test_cmd_check_missing_current_does_not_mask_upstream_failure(tmp_path, capsys):
    baseline = _write(
        tmp_path / "base.json", {"tolerance": 0.1, "files": {"a.py": 80.0}}
    )
    rc = ratchet.cmd_check(
        _args(
            baseline=str(baseline),
            current=str(tmp_path / "missing.json"),
            tolerance=None,
        )
    )
    assert rc == 0
    assert "no coverage report" in capsys.readouterr().err


def test_cmd_check_malformed_baseline_is_setup_error(tmp_path):
    baseline = _write(tmp_path / "base.json", {"no": "files key"})
    current = _write(tmp_path / "cov.json", _cov_json({"a.py": 85.0}))
    with pytest.raises(ratchet.RatchetError):
        ratchet.cmd_check(
            _args(baseline=str(baseline), current=str(current), tolerance=None)
        )


def test_cmd_check_cli_tolerance_overrides_baseline(tmp_path):
    # 5 pp drop: fails at tol=0.1, passes when the CLI widens tolerance to 10.
    baseline = _write(
        tmp_path / "base.json", {"tolerance": 0.1, "files": {"a.py": 90.0}}
    )
    current = _write(tmp_path / "cov.json", _cov_json({"a.py": 85.0}))
    assert (
        ratchet.cmd_check(
            _args(baseline=str(baseline), current=str(current), tolerance=None)
        )
        == 1
    )
    assert (
        ratchet.cmd_check(
            _args(baseline=str(baseline), current=str(current), tolerance=10.0)
        )
        == 0
    )


def test_cmd_update_then_check_is_green(tmp_path):
    # update pins the current numbers; an immediate check must pass.
    baseline = tmp_path / "base.json"
    current = _write(tmp_path / "cov.json", _cov_json({"a.py": 73.4, "b.py": 100.0}))
    assert (
        ratchet.cmd_update(
            _args(baseline=str(baseline), current=str(current), tolerance=None)
        )
        == 0
    )
    assert baseline.is_file()
    assert (
        ratchet.cmd_check(
            _args(baseline=str(baseline), current=str(current), tolerance=None)
        )
        == 0
    )


def test_main_check_regression_returns_one(tmp_path):
    baseline = _write(
        tmp_path / "base.json", {"tolerance": 0.1, "files": {"a.py": 90.0}}
    )
    current = _write(tmp_path / "cov.json", _cov_json({"a.py": 50.0}))
    rc = ratchet.main(["check", "--baseline", str(baseline), "--current", str(current)])
    assert rc == 1
