# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the coverage ratchet (AIHPBLAS-3878).

These lock the ratchet's decision logic: a per-file drop beyond tolerance is a
regression, a rise or an in-tolerance wobble is not, removed/added files are
handled, and malformed input is a setup error rather than a false "pass". They
also lock the direction ``update`` may move a floor: never down unless that file
was named, so moving one floor cannot reset the others. The ratchet is the
enforcement mechanism, so its own behavior must be pinned.
"""

import argparse
import importlib.util
import json
import shlex
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
    assert ratchet.find_regressions(baseline, current, ratchet.DEFAULT_TOLERANCE) == []


def test_default_tolerance_still_catches_a_real_regression():
    # The buffer is wide, not absent: a drop past it is still a failure.
    baseline = {"a.py": 90.0}
    current = {"a.py": 88.5}  # 1.5 pp, past the 1 pp buffer
    assert ratchet.find_regressions(baseline, current, ratchet.DEFAULT_TOLERANCE) == [
        ("a.py", 90.0, 88.5)
    ]


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
# ratchet_floors (update only ever strengthens the baseline)                   #
# --------------------------------------------------------------------------- #
def test_floor_rises_to_current():
    floors, refused = ratchet.ratchet_floors({"a.py": 80.0}, {"a.py": 91.0})
    assert floors == {"a.py": 91.0}
    assert refused == []


def test_floor_is_held_when_current_is_lower():
    floors, refused = ratchet.ratchet_floors({"a.py": 80.0}, {"a.py": 61.0})
    assert floors == {"a.py": 80.0}  # unchanged, not 61.0
    assert refused == [("a.py", 80.0, 61.0)]


def test_new_file_is_pinned_at_current():
    floors, refused = ratchet.ratchet_floors({}, {"new.py": 42.0})
    assert floors == {"new.py": 42.0}
    assert refused == []


def test_named_file_is_lowered():
    floors, refused = ratchet.ratchet_floors(
        {"a.py": 80.0}, {"a.py": 61.0}, allow_lower=["a.py"]
    )
    assert floors == {"a.py": 61.0}
    assert refused == []


def test_lowering_one_file_does_not_lower_another():
    # The regression this whole mechanism exists for: an update run to move one
    # file's floor must not quietly reset every other file to whatever the
    # coverage run on disk happened to measure.
    existing = {"named.py": 90.0, "bystander.py": 88.0}
    current = {"named.py": 70.0, "bystander.py": 60.0}
    floors, refused = ratchet.ratchet_floors(
        existing, current, allow_lower=["named.py"]
    )
    assert floors["named.py"] == 70.0  # lowered, because it was named
    assert floors["bystander.py"] == 88.0  # held, because it was not
    assert refused == [("bystander.py", 88.0, 60.0)]


def test_sub_precision_dip_is_not_a_fall():
    # The baseline is written to 2 dp, so a difference it cannot represent must
    # not count as lowering the floor (otherwise update refuses on float noise).
    floors, refused = ratchet.ratchet_floors({"a.py": 73.39}, {"a.py": 73.3899})
    assert floors == {"a.py": 73.39}
    assert refused == []


def test_multiple_refusals_sorted_biggest_drop_first():
    existing = {"a.py": 90.0, "b.py": 90.0}
    current = {"a.py": 88.0, "b.py": 70.0}
    _, refused = ratchet.ratchet_floors(existing, current)
    assert [row[0] for row in refused] == ["b.py", "a.py"]


def test_in_tolerance_dip_is_held_without_a_refusal():
    # Without a tolerance, ratchet_floors refuses every representable drop, even
    # one check's own tolerance would classify as noise rather than a real
    # regression. That mismatch is what makes the command check prints
    # insufficient for update: it names only the real regression, but update
    # still refuses the unnamed noisy file. Passing the same tolerance through
    # holds the noisy file at its existing floor instead of refusing it.
    floors, refused = ratchet.ratchet_floors(
        {"b.py": 80.0}, {"b.py": 79.5}, tolerance=1.0
    )
    assert floors == {"b.py": 80.0}  # held, not lowered to 79.5
    assert refused == []  # and not reported as something to authorize


def test_drop_beyond_tolerance_is_still_refused():
    # The tolerance absorbs noise, not real regressions: a drop past it is
    # refused exactly as before.
    floors, refused = ratchet.ratchet_floors(
        {"a.py": 90.0}, {"a.py": 88.0}, tolerance=1.0
    )
    assert floors == {"a.py": 90.0}
    assert refused == [("a.py", 90.0, 88.0)]


def test_named_file_beyond_tolerance_still_lowers():
    # allow_lower authorizes any magnitude of drop, tolerance or not.
    floors, refused = ratchet.ratchet_floors(
        {"a.py": 90.0}, {"a.py": 50.0}, allow_lower=["a.py"], tolerance=1.0
    )
    assert floors == {"a.py": 50.0}
    assert refused == []


# --------------------------------------------------------------------------- #
# remediation                                                                 #
# --------------------------------------------------------------------------- #
def test_remediation_names_every_offending_file():
    text = ratchet.remediation(["pkg/a.py", "pkg/b.py"])
    assert "--allow-lower=pkg/a.py" in text
    assert "--allow-lower=pkg/b.py" in text


def test_remediation_does_not_suggest_a_blanket_reset():
    # The printed command must be copy-pasteable without also lowering files the
    # developer never looked at, so it always carries an explicit file list.
    text = ratchet.remediation(["pkg/a.py"])
    assert "update --current coverage.json\n" not in text


def test_remediation_quotes_a_path_with_a_space():
    # Unquoted, "pkg/a b.py" would split into two shell arguments and the second
    # would be read as a positional, not part of the path.
    text = ratchet.remediation(["pkg/a b.py"])
    assert "--allow-lower=" + shlex.quote("pkg/a b.py") in text


def test_remediation_does_not_let_a_leading_hyphen_look_like_another_option():
    # "--allow-lower -leading.py" would make argparse treat "-leading.py" as a
    # separate (unknown) option rather than this option's value.
    text = ratchet.remediation(["-leading.py"])
    assert "--allow-lower=-leading.py" in text
    assert "--allow-lower -leading.py" not in text


def test_remediation_does_not_let_shell_metacharacters_execute():
    # If copy-pasted verbatim, an unquoted "$(...)" would run as a command
    # substitution instead of being treated as a literal filename.
    path = "pkg/$(echo injected).py"
    text = ratchet.remediation([path])
    assert "--allow-lower=" + shlex.quote(path) in text
    assert f"--allow-lower={path}" not in text


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
    assert "--allow-lower=pkg/a.py" in err  # and it names only this file


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
            _args(
                baseline=str(baseline),
                current=str(current),
                tolerance=None,
                allow_lower=None,
            )
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


def _update_args(baseline, current, allow_lower=None):
    return _args(
        baseline=str(baseline),
        current=str(current),
        tolerance=None,
        allow_lower=allow_lower,
    )


def test_cmd_update_refuses_to_lower_and_leaves_the_baseline_alone(tmp_path, capsys):
    baseline = _write(
        tmp_path / "base.json", {"tolerance": 1.0, "files": {"pkg/a.py": 90.0}}
    )
    before = baseline.read_bytes()
    current = _write(tmp_path / "cov.json", _cov_json({"pkg/a.py": 70.0}))

    assert ratchet.cmd_update(_update_args(baseline, current)) == 1
    assert baseline.read_bytes() == before  # nothing written on refusal

    err = capsys.readouterr().err
    assert "refusing to lower" in err
    assert "--allow-lower=pkg/a.py" in err  # the exact flag to add


def test_cmd_update_lowers_only_the_named_file(tmp_path):
    baseline = _write(
        tmp_path / "base.json",
        {"tolerance": 1.0, "files": {"pkg/a.py": 90.0, "pkg/b.py": 88.0}},
    )
    current = _write(
        tmp_path / "cov.json", _cov_json({"pkg/a.py": 70.0, "pkg/b.py": 95.0})
    )

    rc = ratchet.cmd_update(_update_args(baseline, current, allow_lower=["pkg/a.py"]))
    assert rc == 0

    saved = json.loads(baseline.read_text(encoding="utf-8"))["files"]
    assert saved == {"pkg/a.py": 70.0, "pkg/b.py": 95.0}


def test_cmd_update_warns_when_allow_lower_names_an_unknown_path(tmp_path, capsys):
    # A typo'd path authorizes nothing. Saying so out loud keeps it from looking
    # like the drop was reviewed and accepted.
    baseline = _write(
        tmp_path / "base.json", {"tolerance": 1.0, "files": {"pkg/a.py": 50.0}}
    )
    current = _write(tmp_path / "cov.json", _cov_json({"pkg/a.py": 60.0}))

    rc = ratchet.cmd_update(
        _update_args(baseline, current, allow_lower=["pkg/typo.py"])
    )
    assert rc == 0
    assert "pkg/typo.py" in capsys.readouterr().err


def test_main_update_refusal_returns_one(tmp_path):
    baseline = _write(
        tmp_path / "base.json", {"tolerance": 1.0, "files": {"a.py": 90.0}}
    )
    current = _write(tmp_path / "cov.json", _cov_json({"a.py": 50.0}))
    rc = ratchet.main(
        ["update", "--baseline", str(baseline), "--current", str(current)]
    )
    assert rc == 1


def test_main_update_accepts_repeated_allow_lower(tmp_path):
    baseline = _write(
        tmp_path / "base.json",
        {"tolerance": 1.0, "files": {"a.py": 90.0, "b.py": 90.0}},
    )
    current = _write(tmp_path / "cov.json", _cov_json({"a.py": 50.0, "b.py": 50.0}))
    rc = ratchet.main(
        [
            "update",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--allow-lower",
            "a.py",
            "--allow-lower",
            "b.py",
        ]
    )
    assert rc == 0
    saved = json.loads(baseline.read_text(encoding="utf-8"))["files"]
    assert saved == {"a.py": 50.0, "b.py": 50.0}


def test_main_check_regression_returns_one(tmp_path):
    baseline = _write(
        tmp_path / "base.json", {"tolerance": 0.1, "files": {"a.py": 90.0}}
    )
    current = _write(tmp_path / "cov.json", _cov_json({"a.py": 50.0}))
    rc = ratchet.main(["check", "--baseline", str(baseline), "--current", str(current)])
    assert rc == 1


def test_check_remediation_command_succeeds_against_update(tmp_path, capsys):
    # a.py regressed by more than the 1.0 pp tolerance; b.py dipped by 0.5 pp,
    # which the same tolerance treats as noise. check must name only a.py, and
    # running the exact command it prints must succeed: it lowers a.py to the
    # measured value and leaves b.py's floor untouched, rather than also
    # refusing on the unnamed, in-tolerance file.
    baseline = _write(
        tmp_path / "base.json",
        {"tolerance": 1.0, "files": {"a.py": 90.0, "b.py": 80.0}},
    )
    current = _write(
        tmp_path / "cov.json", _cov_json({"a.py": 88.0, "b.py": 79.5})
    )

    check_rc = ratchet.main(
        ["check", "--baseline", str(baseline), "--current", str(current)]
    )
    assert check_rc == 1
    err = capsys.readouterr().err
    assert "--allow-lower=a.py" in err
    assert "--allow-lower=b.py" not in err

    update_rc = ratchet.main(
        [
            "update",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--allow-lower",
            "a.py",
        ]
    )
    assert update_rc == 0

    saved = json.loads(baseline.read_text(encoding="utf-8"))["files"]
    assert saved["a.py"] == 88.0  # lowered, because it was named
    assert saved["b.py"] == 80.0  # held, the dip was within tolerance
