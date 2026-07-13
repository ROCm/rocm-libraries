################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``TensileLogic.Run`` (the CLI orchestrator).

``Run`` is Tier C: it wires argparse + toolchain + a ``ParallelMap2`` fan-out
over ``_runChecks``, with a background progress thread. The strategy
(per ``target.md``) is to pin the *orchestration* deterministically by
monkeypatching the heavy collaborators — the per-solution validators,
``validateToolchain`` / ``makeIsaInfoMap`` / ``assignGlobalParameters``,
``ParallelMap2``, and ``_setup`` — in the ``Run`` namespace. The validators
themselves are characterized in their own suites; here we pin how ``Run``
*combines* their results (keep / total / known-bug-skip / chip-id-failure
counts, batching, exit codes).

What is exercised for line coverage only (never snapshotted): ``_progress_loop``
(threads + ``time.time``) and ``main``'s background progress thread. ``main``'s
deterministic stdout (Total / Keep / Reject) is snapshotted with the progress
thread disabled (``Verbose >= 2``).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import Tensile.TensileLogic.Run as Run
from Tensile.TensileLogic.KnownBugs import normalize_logic_relative_path

pytestmark = pytest.mark.unit


# ===========================================================================
# Check NamedTuple
# ===========================================================================

def test_check_namedtuple(snapshot):
    assert Run.Check(OnlyCustomKernels=True, All=False)._asdict() == snapshot


# ===========================================================================
# _runChecks — validators injected via the Run namespace
# ===========================================================================

def _write_logic(path, solutions, problemType=None):
    """Write a minimal serialized library-logic YAML (data[4]=problemType,
    data[5]=solutions); the indices _runChecks does not read are placeholders."""
    if problemType is None:
        problemType = {"OperationType": "GEMM"}
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    path.write_text(
        yaml.safe_dump([
            {"MinimumRequiredVersion": "4.33.0"},
            "schedule",
            "gfx942",
            ["Device 74a0"],
            problemType,
            solutions,
        ]),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def passing_validators(monkeypatch):
    """Install validators that all accept; individual tests override as needed."""
    monkeypatch.setattr(Run, "_validateChipId",
                        lambda f, logic_relative_path=None, report_path=None: True)
    monkeypatch.setattr(Run, "handleCustomKernel", lambda s, iim: (s, False))
    monkeypatch.setattr(Run, "_validateMatrixInstruction", lambda s, iim, rel: True)
    monkeypatch.setattr(Run, "_validateWorkGroup", lambda s, rel: True)
    monkeypatch.setattr(Run, "_validateWorkGroupMappingXCC", lambda s, rel: True)
    monkeypatch.setattr(Run, "hasCustomKernel", lambda f: False)


_ALL = Run.Check(OnlyCustomKernels=False, All=True)
_CUSTOM = Run.Check(OnlyCustomKernels=True, All=False)


def _result(*tup):
    keep, total, kb, cid = tup
    return {"keep": keep, "total": total, "known_bug_skips": kb, "chip_id_failures": cid}


def test_runchecks_all_pass(tmp_path, passing_validators, snapshot):
    f = _write_logic(tmp_path / "lib" / "a.yaml",
                     [{"SolutionIndex": 0}, {"SolutionIndex": 1}])
    assert _result(*Run._runChecks(tmp_path, {}, _ALL, frozenset(), [f])) == snapshot


def test_runchecks_one_validator_fails(tmp_path, passing_validators, monkeypatch, snapshot):
    monkeypatch.setattr(Run, "_validateWorkGroup", lambda s, rel: False)
    f = _write_logic(tmp_path / "lib" / "a.yaml", [{"SolutionIndex": 0}])
    assert _result(*Run._runChecks(tmp_path, {}, _ALL, frozenset(), [f])) == snapshot


def test_runchecks_chip_id_failure(tmp_path, passing_validators, monkeypatch, snapshot):
    monkeypatch.setattr(Run, "_validateChipId",
                        lambda f, logic_relative_path=None, report_path=None: False)
    f = _write_logic(tmp_path / "lib" / "a.yaml", [{"SolutionIndex": 0}])
    # Chip-ID failure short-circuits the whole file: no per-solution counting.
    assert _result(*Run._runChecks(tmp_path, {}, _ALL, frozenset(), [f])) == snapshot


def test_runchecks_experimental_skipped(tmp_path, passing_validators, snapshot):
    f = _write_logic(tmp_path / "Experimental" / "a.yaml", [{"SolutionIndex": 0}])
    assert _result(*Run._runChecks(tmp_path, {}, _ALL, frozenset(), [f])) == snapshot


def test_runchecks_known_bug_skip(tmp_path, passing_validators, monkeypatch, snapshot):
    # idx 0 is a known bug (kept without validation); idx 1 would fail (validator
    # forced False) -> proves the known-bug path bypasses validation.
    monkeypatch.setattr(Run, "_validateWorkGroup", lambda s, rel: False)
    f = _write_logic(tmp_path / "foo" / "bar.yaml",
                     [{"SolutionIndex": 0}, {"SolutionIndex": 1}])
    kb = frozenset({(normalize_logic_relative_path(Path("foo/bar.yaml")), 0)})
    assert _result(*Run._runChecks(tmp_path, {}, _ALL, kb, [f])) == snapshot


def test_runchecks_only_custom_kernels_filters(tmp_path, passing_validators, monkeypatch, snapshot):
    # hasCustomKernel True so solutions are loaded; handleCustomKernel marks the
    # first solution custom and the second not -> only the custom one counts.
    monkeypatch.setattr(Run, "hasCustomKernel", lambda f: True)
    monkeypatch.setattr(
        Run, "handleCustomKernel",
        lambda s, iim: (s, s.get("SolutionIndex") == 0),
    )
    f = _write_logic(tmp_path / "lib" / "a.yaml",
                     [{"SolutionIndex": 0}, {"SolutionIndex": 1}])
    assert _result(*Run._runChecks(tmp_path, {}, _CUSTOM, frozenset(), [f])) == snapshot


def test_runchecks_only_custom_no_custom_in_file(tmp_path, passing_validators, snapshot):
    # check.OnlyCustomKernels but hasCustomKernel False -> no solutions loaded.
    f = _write_logic(tmp_path / "lib" / "a.yaml", [{"SolutionIndex": 0}])
    assert _result(*Run._runChecks(tmp_path, {}, _CUSTOM, frozenset(), [f])) == snapshot


def test_runchecks_single_file_relative_dot(tmp_path, passing_validators, snapshot):
    # When the file IS the logicPath, rel == Path(".") -> the chip_id_path uses
    # the file itself (the rel == "." branch).
    f = _write_logic(tmp_path / "solo.yaml", [{"SolutionIndex": 0}])
    assert _result(*Run._runChecks(f, {}, _ALL, frozenset(), [f])) == snapshot


# ===========================================================================
# _setup — heavy externals stubbed; argv driven
# ===========================================================================

@pytest.fixture
def stub_setup_externals(monkeypatch):
    """Replace the subprocess / capability externals _setup calls so it runs
    fast and deterministically; record the assignGlobalParameters config."""
    recorded = {}
    monkeypatch.setattr(Run, "validateToolchain", lambda c: "FAKE_CXX")
    monkeypatch.setattr(Run, "makeIsaInfoMap", lambda isa, cxx: {"isa": "fake"})
    monkeypatch.setattr(Run, "assignGlobalParameters",
                        lambda cfg, iim: recorded.__setitem__("gp_config", cfg))
    monkeypatch.setattr(Run, "setVerbosity", lambda v: None)
    return recorded


def _run_setup(monkeypatch, argv):
    import sys
    monkeypatch.setattr(sys, "argv", ["TensileLogic", *argv])
    return Run._setup()


def test_setup_happy_check_all(tmp_path, stub_setup_externals, monkeypatch, snapshot):
    _write_logic(tmp_path / "a.yaml", [{"SolutionIndex": 0}])
    jobs, iim, logicPath, files, check, args = _run_setup(
        monkeypatch, [str(tmp_path), "--check-all"]
    )
    assert {
        "jobs": jobs, "check": check._asdict(), "num_files": len(files),
        "gp_config": stub_setup_externals.get("gp_config"),
    } == snapshot


def test_setup_verbose2_sets_print_rejection(tmp_path, stub_setup_externals, monkeypatch, snapshot):
    _write_logic(tmp_path / "a.yaml", [{"SolutionIndex": 0}])
    _run_setup(monkeypatch, [str(tmp_path), "--check-all", "-v", "2"])
    # Verbose >= 2 -> assignGlobalParameters receives PrintSolutionRejectionReason.
    assert stub_setup_externals.get("gp_config") == snapshot


def test_setup_single_yaml_file(tmp_path, stub_setup_externals, monkeypatch, snapshot):
    f = _write_logic(tmp_path / "solo.yaml", [{"SolutionIndex": 0}])
    jobs, iim, logicPath, files, check, args = _run_setup(
        monkeypatch, [str(f), "--check-all"]
    )
    assert {"num_files": len(files), "is_solo": files == [f]} == snapshot


def test_setup_no_checks_exits_zero(tmp_path, stub_setup_externals, monkeypatch):
    _write_logic(tmp_path / "a.yaml", [{"SolutionIndex": 0}])
    with pytest.raises(SystemExit) as ei:
        _run_setup(monkeypatch, [str(tmp_path)])
    assert ei.value.code == 0


def test_setup_no_files_exits_one(tmp_path, stub_setup_externals, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as ei:
        _run_setup(monkeypatch, [str(empty), "--check-all"])
    assert ei.value.code == 1


# ===========================================================================
# _progress_loop — fake event, body coverage only (not snapshotted)
# ===========================================================================

def test_progress_loop_runs_body_then_clears(capsys):
    class _FakeEvent:
        def __init__(self):
            self.calls = 0

        def wait(self, timeout):
            # False on the first call (run the body once), True after (exit).
            self.calls += 1
            return self.calls > 1

    Run._progress_loop(_FakeEvent(), interval=0.0)
    out = capsys.readouterr().out
    assert "Validating library logic" in out


# ===========================================================================
# main — _setup / ParallelMap2 / load_known_bugs stubbed
# ===========================================================================

def _stub_main(monkeypatch, results, *, verbose=2, known_bugs=frozenset(),
               kb_raises=None, files=None):
    if files is None:
        files = [Path("lib/a.yaml")]
    args = SimpleNamespace(KnownBugs=None, Verbose=verbose)
    check = Run.Check(OnlyCustomKernels=False, All=True)
    monkeypatch.setattr(
        Run, "_setup",
        lambda: (1, {"isa": "fake"}, Path("lib"), files, check, args),
    )
    if kb_raises is not None:
        def _raise(_):
            raise kb_raises
        monkeypatch.setattr(Run, "load_known_bugs", _raise)
    else:
        monkeypatch.setattr(Run, "load_known_bugs", lambda p: known_bugs)
    monkeypatch.setattr(Run, "ParallelMap2",
                        lambda fn, batches, **kw: list(results))


def test_main_happy_no_rejects(monkeypatch, capsys, snapshot):
    _stub_main(monkeypatch, [(2, 2, 0, 0)], verbose=2)
    Run.main()  # no rejects -> returns normally
    assert capsys.readouterr().out == snapshot


def test_main_rejects_exit(monkeypatch, capsys, snapshot):
    _stub_main(monkeypatch, [(1, 2, 0, 0)], verbose=2)
    with pytest.raises(SystemExit) as ei:
        Run.main()
    assert {"exit_code": ei.value.code, "stdout": capsys.readouterr().out} == snapshot


def test_main_known_bugs_and_chip_id_failures(monkeypatch, capsys, snapshot):
    _stub_main(monkeypatch, [(1, 3, 1, 1)], verbose=2)
    with pytest.raises(SystemExit) as ei:
        Run.main()
    assert {"exit_code": ei.value.code, "stdout": capsys.readouterr().out} == snapshot


def test_main_verbose1_starts_progress_thread(monkeypatch, capsys):
    # Verbose < 2 -> the background progress thread path runs. stdout carries a
    # carriage-return progress artefact (nondeterministic), so we only assert
    # the run completed and printed the totals; no snapshot.
    _stub_main(monkeypatch, [(1, 1, 0, 0)], verbose=1)
    Run.main()
    assert "Total" in capsys.readouterr().out


def test_main_load_known_bugs_error_exits(monkeypatch):
    _stub_main(monkeypatch, [(1, 1, 0, 0)], verbose=2, kb_raises=ValueError("bad kb"))
    with pytest.raises(SystemExit) as ei:
        Run.main()
    assert ei.value.code == 1
