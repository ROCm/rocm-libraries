# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The top-level check runner must not silently drop a default host gate."""

import importlib.util
from pathlib import Path
import re
import subprocess
import sys

import pytest

from rocke.tests.lanes import SOURCE_TREE, skip_until


def _runner_path() -> Path:
    """Find the production runner in source and relocated test layouts."""
    test_file = Path(__file__).resolve()
    candidates = (
        test_file.parents[3] / "tools" / "run_checks.py",
        test_file.with_name("run_checks.py"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    skip_until(
        SOURCE_TREE,
        reason="top-level run_checks.py is absent from this platform-only tree",
    )


def _assert_guarded_runner_install(text: str) -> None:
    assignment = re.search(
        r'set\(\s*(\w+)\s+"\$\{CMAKE_CURRENT_SOURCE_DIR\}/\.\./tools/run_checks\.py"\s*\)',
        text,
    )
    assert assignment, "no variable names the top-level run_checks.py"
    variable = re.escape(assignment.group(1))
    guarded_install = re.search(
        rf'if\(\s*EXISTS\s+"\$\{{{variable}\}}"\s*\)\s*'
        rf'install\(\s*FILES\s+"\$\{{{variable}\}}"\s+'
        r'DESTINATION\s+"\$\{ROCKE_TEST_INSTALL_DIR\}/tests/tools"\s*\)',
        text,
    )
    assert guarded_install, "top-level runner installation must use its existence guard"


def test_platform_only_install_does_not_require_the_top_level_runner():
    platform = Path(__file__).resolve().parents[2]
    cmake = platform / "CMakeLists.txt"
    if not cmake.is_file():
        skip_until(
            SOURCE_TREE,
            reason="source CMake project is absent from the installed test artifact",
        )
    _assert_guarded_runner_install(cmake.read_text())


def _runner_install_text(variable: str, files_spacing: str) -> str:
    return (
        f'set({variable} "${{CMAKE_CURRENT_SOURCE_DIR}}/../tools/run_checks.py")\n'
        f'if(EXISTS "${{{variable}}}")\n'
        f'  install({files_spacing}FILES "${{{variable}}}"\n'
        '          DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/tools")\n'
        "endif()\n"
    )


@pytest.mark.parametrize("variable", ["_ROCKE_RUN_CHECKS", "_rocke_run_checks"])
@pytest.mark.parametrize("files_spacing", ["", "\n    "])
def test_runner_install_contract_accepts_cmake_formatting(variable, files_spacing):
    _assert_guarded_runner_install(_runner_install_text(variable, files_spacing))


@pytest.mark.parametrize(
    "old,new,message",
    [
        ('if(EXISTS "${_runner}")\n', "", "must use its existence guard"),
        (
            'if(EXISTS "${_runner}")',
            'if(EXISTS "${_other}")',
            "must use its existence guard",
        ),
        ('FILES "${_runner}"', 'FILES "${_other}"', "must use its existence guard"),
        ("../tools/run_checks.py", "../tools/other.py", "no variable names"),
        ("/tests/tools", "/tests/other", "must use its existence guard"),
    ],
    ids=[
        "unguarded",
        "wrong-guard",
        "wrong-installed-variable",
        "wrong-source",
        "wrong-destination",
    ],
)
def test_runner_install_contract_detects_broken_staging(old, new, message):
    text = _runner_install_text("_runner", "")
    assert old in text
    with pytest.raises(AssertionError, match=message):
        _assert_guarded_runner_install(text.replace(old, new))


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location("rocke_run_checks", _runner_path())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_has_gpu", lambda: False)
    monkeypatch.setattr(module, "_discover_pytest_dirs", lambda: [])
    monkeypatch.setattr(module, "_discover_pinned_reference_tests", lambda: [])
    return module


def _run_default(runner, monkeypatch, failing_label=""):
    calls = []

    def run(label, command, env, cwd=None):
        calls.append((label, command, cwd))
        return 9 if label == failing_label else 0

    monkeypatch.setattr(runner, "_run", run)
    monkeypatch.setattr(runner.sys, "argv", ["run_checks.py"])
    return runner.main(), calls


def test_runner_uses_every_qualified_compiler_lane(runner):
    from rocke.core.llvm_profile import LLVM_FLAVORS

    assert runner.FLAVORS is LLVM_FLAVORS
    assert "llvm24" in runner.FLAVORS


def test_default_runs_every_host_validation_gate_once(runner, monkeypatch):
    rc, calls = _run_default(runner, monkeypatch)
    assert rc == 0

    by_label = {}
    for label, command, cwd in calls:
        by_label.setdefault(label, []).append((command, cwd))

    expected_once = (
        "relative-path guard",
        "emitted-IR validity gate",
        "arch-domain warning gate",
        "configure arch-domain reproducibility CTest",
        "arch-domain reproducibility CTest",
    )
    for label in expected_once:
        assert len(by_label[label]) == 1, label

    ir_command, _ = by_label["emitted-IR validity gate"][0]
    assert Path(ir_command[1]).name == "check_ir_validity.py"
    domain_command, _ = by_label["arch-domain warning gate"][0]
    assert Path(domain_command[1]).name == "check_arch_domain.py"

    configure_command, _ = by_label["configure arch-domain reproducibility CTest"][0]
    assert configure_command[:2] == ["cmake", "-S"]
    ctest_command, _ = by_label["arch-domain reproducibility CTest"][0]
    assert ctest_command[0] == "ctest"
    assert ctest_command[ctest_command.index("-R") + 1] == (
        "^rocke_arch_domain_reproducibility$"
    )
    assert "--no-tests=error" in ctest_command


def test_the_guard_step_turns_off_every_other_run_all_stage(runner, monkeypatch):
    """The guard step runs platform/tests/run_all.py for its relative-path guard
    alone. Every stage run_all can turn off, read from its own --help, is turned
    off there, so a stage added later cannot slip into this fast step. The
    run_all.py beside these tests is the one the step runs in a source checkout,
    and its installed copy in the relocated suite."""
    run_all = Path(__file__).resolve().parents[1] / "run_all.py"
    usage = subprocess.run(
        [sys.executable, str(run_all), "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    stage_switches = set(re.findall(r"--no-[a-z-]+", usage)) - {"--no-guard"}
    assert stage_switches, usage

    _rc, calls = _run_default(runner, monkeypatch)
    [(command, _cwd)] = [
        (command, cwd)
        for label, command, cwd in calls
        if label == "relative-path guard"
    ]
    assert Path(command[1]).parts[-2:] == ("tests", "run_all.py")
    assert stage_switches <= set(command), sorted(stage_switches - set(command))


@pytest.mark.parametrize(
    "failing_label",
    (
        "emitted-IR validity gate",
        "arch-domain warning gate",
        "arch-domain reproducibility CTest",
    ),
)
def test_default_propagates_each_host_gate_failure(runner, monkeypatch, failing_label):
    rc, calls = _run_default(runner, monkeypatch, failing_label)
    assert rc == 1
    assert sum(label == failing_label for label, _command, _cwd in calls) == 1
