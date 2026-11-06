# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The top-level check runner must not silently drop a default host gate."""

import importlib.util
from pathlib import Path
import re

import pytest


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
    pytest.skip("top-level run_checks.py is absent from this platform-only tree")


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
        pytest.skip("source CMake project is absent from the installed test artifact")
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


def _main(runner, monkeypatch, *argv, rcs=None):
    """Run main() with every stage stubbed; ``rcs`` maps a stage label to its rc."""
    calls = []

    def run(label, command, env, cwd=None):
        calls.append((label, command, cwd))
        return (rcs or {}).get(label, 0)

    monkeypatch.setattr(runner, "_run", run)
    monkeypatch.setattr(runner.sys, "argv", ["run_checks.py", *argv])
    return runner.main(), calls


@pytest.fixture
def complete_host(runner, monkeypatch, tmp_path):
    """A host where every stage can run: a device, both test trees, the numeric lane."""
    library = tmp_path / "library"
    numeric = library / "tests" / "differential" / "numeric_attention.py"
    numeric.parent.mkdir(parents=True)
    numeric.touch()
    monkeypatch.setattr(runner, "LIBRARY", library)
    monkeypatch.setattr(runner, "_has_gpu", lambda: True)
    monkeypatch.setattr(
        runner,
        "_discover_pytest_dirs",
        lambda: [runner.PLATFORM / "tests", runner.LIBRARY / "tests"],
    )
    return runner


def _run_default(runner, monkeypatch, failing_label=""):
    return _main(runner, monkeypatch, rcs={failing_label: 9})


def test_runner_uses_every_qualified_compiler_lane(runner):
    from rocke.core.llvm_profile import LLVM_FLAVORS

    assert runner.FLAVORS is LLVM_FLAVORS
    assert "llvm24" in runner.FLAVORS


def test_pytest_stage_delegates_library_tier_policy_to_library_runner(
    runner, monkeypatch
):
    calls = []

    def run(label, command, env, cwd=None):
        calls.append((label, command, cwd))
        return 0

    monkeypatch.setattr(runner, "_run", run)
    monkeypatch.setattr(
        runner,
        "_discover_pytest_dirs",
        lambda: [runner.PLATFORM / "tests", runner.LIBRARY / "tests"],
    )
    monkeypatch.setattr(runner.sys, "argv", ["run_checks.py", "--steps", "pytest"])
    assert runner.main() == 0

    by_label = {label: (command, cwd) for label, command, cwd in calls}
    platform, _ = by_label["pytest [platform units]"]
    assert str(runner.PLATFORM / "tests") in platform
    assert str(runner.LIBRARY / "tests") not in platform

    library, library_cwd = by_label["pytest [library units]"]
    assert Path(library[1]) == runner.LIBRARY / "tests" / "run_all.py"
    assert "--unit-only" in library
    assert library_cwd == runner.LIBRARY
    assert "--gpu" not in library
    assert "pytest [library explicit gates]" in by_label


def test_pytest_stage_includes_library_gpu_files_when_a_device_is_visible(
    runner, monkeypatch
):
    calls = []

    def run(label, command, env, cwd=None):
        calls.append((label, command))
        return 0

    monkeypatch.setattr(runner, "_run", run)
    monkeypatch.setattr(runner, "_has_gpu", lambda: True)
    monkeypatch.setattr(
        runner,
        "_discover_pytest_dirs",
        lambda: [runner.PLATFORM / "tests", runner.LIBRARY / "tests"],
    )
    monkeypatch.setattr(runner.sys, "argv", ["run_checks.py", "--steps", "pytest"])
    assert runner.main() == 0
    library = next(
        command for label, command in calls if label == "pytest [library units]"
    )
    assert "--gpu" in library


def test_default_runs_every_host_validation_gate_once(
    complete_host, monkeypatch, capsys
):
    rc, calls = _run_default(complete_host, monkeypatch)
    assert rc == 0
    assert capsys.readouterr().out.rstrip().endswith("RESULT: GREEN")

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


@pytest.mark.parametrize(
    "failing_label",
    (
        "emitted-IR validity gate",
        "arch-domain warning gate",
        "arch-domain reproducibility CTest",
    ),
)
def test_default_propagates_each_host_gate_failure(
    complete_host, monkeypatch, failing_label
):
    rc, calls = _run_default(complete_host, monkeypatch, failing_label)
    assert rc == 1
    assert sum(label == failing_label for label, _command, _cwd in calls) == 1


def _result(capsys) -> str:
    return capsys.readouterr().out.rstrip().splitlines()[-1]


def test_step_subset_is_partial_not_green(runner, monkeypatch, capsys):
    rc, calls = _main(runner, monkeypatch, "--steps", "guard")
    assert rc == 0
    assert [label for label, _command, _cwd in calls] == ["relative-path guard"]
    assert _result(capsys) == "RESULT: PARTIAL"


def test_naming_every_stage_is_not_narrowing(complete_host, monkeypatch, capsys):
    rc, _calls = _main(
        complete_host, monkeypatch, "--steps", ",".join(reversed(complete_host.STAGES))
    )
    assert rc == 0
    assert _result(capsys) == "RESULT: GREEN"


@pytest.mark.parametrize(
    "argv", (("--op", "fmha_bwd"), ("--flavor", "llvm22")), ids=("op", "flavor")
)
def test_filters_are_partial_not_green(complete_host, monkeypatch, capsys, argv):
    rc, _calls = _main(complete_host, monkeypatch, *argv)
    assert rc == 0
    out = capsys.readouterr().out
    assert f"narrowed by: {' '.join(argv)}" in out
    assert out.rstrip().endswith("RESULT: PARTIAL")


def test_op_skipped_stages_are_narrowing_not_skips(complete_host, monkeypatch, capsys):
    rc, calls = _main(complete_host, monkeypatch, "--op", "fmha_bwd")
    assert rc == 0
    labels = {label for label, _command, _cwd in calls}
    assert "platform instance parity" not in labels
    assert "pytest [library explicit gates]" not in labels
    out = capsys.readouterr().out
    assert "parity has no operator filter" in out
    assert "library explicit gates not run" in out
    assert "skipped" not in out.split("run_checks stages:")[1]
    assert out.rstrip().endswith("RESULT: PARTIAL")


def test_implicit_numeric_without_a_device_is_incomplete(
    complete_host, monkeypatch, capsys
):
    monkeypatch.setattr(complete_host, "_has_gpu", lambda: False)
    rc, calls = _main(complete_host, monkeypatch)
    assert rc == 3
    assert "on-GPU numeric" not in {label for label, _command, _cwd in calls}
    out = capsys.readouterr().out
    assert "skipped           on-GPU numeric (no HIP device visible)" in out
    assert out.rstrip().endswith("RESULT: INCOMPLETE")


def test_explicit_numeric_without_a_device_is_red(runner, monkeypatch, capsys):
    rc, calls = _main(runner, monkeypatch, "--steps", "numeric", "--allow-incomplete")
    assert rc == 1
    assert calls == []
    assert _result(capsys) == "RESULT: RED"


def test_allow_incomplete_exits_zero_but_still_reports_incomplete(
    complete_host, monkeypatch, capsys
):
    monkeypatch.setattr(complete_host, "_has_gpu", lambda: False)
    rc, _calls = _main(complete_host, monkeypatch, "--allow-incomplete")
    assert rc == 0
    assert _result(capsys) == "RESULT: INCOMPLETE (allowed by --allow-incomplete)"


def test_failure_is_red_even_with_allow_incomplete(complete_host, monkeypatch, capsys):
    monkeypatch.setattr(complete_host, "_has_gpu", lambda: False)
    rc, _calls = _main(
        complete_host,
        monkeypatch,
        "--allow-incomplete",
        rcs={"emitted-IR validity gate": 1},
    )
    assert rc == 1
    assert _result(capsys) == "RESULT: RED"


def test_pytest_selecting_nothing_is_red(complete_host, monkeypatch, capsys):
    rc, _calls = _main(complete_host, monkeypatch, rcs={"pytest [platform units]": 5})
    assert rc == 1
    out = capsys.readouterr().out
    assert "failed            pytest [platform units] (selected no tests)" in out
    assert out.rstrip().endswith("RESULT: RED")


@pytest.mark.parametrize(
    "label",
    (
        "relative-path guard",
        "pytest [library units]",
        "pytest [library explicit gates]",
    ),
)
def test_delegated_runner_incomplete_makes_the_run_incomplete(
    complete_host, monkeypatch, capsys, label
):
    rc, _calls = _main(complete_host, monkeypatch, rcs={label: 3})
    assert rc == 3
    out = capsys.readouterr().out
    assert f"skipped           {label} (the delegated runner" in out
    assert out.rstrip().endswith("RESULT: INCOMPLETE")


def test_exit_three_from_a_non_runner_stage_is_a_failure(complete_host, monkeypatch):
    rc, _calls = _main(complete_host, monkeypatch, rcs={"pytest [platform units]": 3})
    assert rc == 1


def test_undiscovered_test_tree_is_incomplete(complete_host, monkeypatch, capsys):
    monkeypatch.setattr(
        complete_host,
        "_discover_pytest_dirs",
        lambda: [complete_host.PLATFORM / "tests"],
    )
    rc, calls = _main(complete_host, monkeypatch)
    assert rc == 3
    assert "pytest [library units]" not in {label for label, _command, _cwd in calls}
    out = capsys.readouterr().out
    assert "skipped           pytest [library units] (no test dir discovered)" in out


def test_unexercised_pinned_references_are_incomplete(
    complete_host, monkeypatch, capsys
):
    suite = complete_host.LIBRARY / "tests" / "test_conv_pinned_reference.py"
    monkeypatch.setattr(
        complete_host, "_discover_pinned_reference_tests", lambda: [suite]
    )
    monkeypatch.setattr(
        complete_host, "_unexercised_pinned", lambda junit, suites: ["conv"]
    )
    rc, calls = _main(complete_host, monkeypatch)
    assert rc == 3
    assert "pytest [pinned GPU references]" in {label for label, _c, _cwd in calls}
    out = capsys.readouterr().out
    assert "skipped           pinned GPU references (did not run for: conv" in out
    assert out.rstrip().endswith("RESULT: INCOMPLETE")


def test_exercised_pinned_references_stay_green(complete_host, monkeypatch, capsys):
    suite = complete_host.LIBRARY / "tests" / "test_conv_pinned_reference.py"
    monkeypatch.setattr(
        complete_host, "_discover_pinned_reference_tests", lambda: [suite]
    )
    monkeypatch.setattr(complete_host, "_unexercised_pinned", lambda junit, suites: [])
    rc, _calls = _main(complete_host, monkeypatch)
    assert rc == 0
    assert _result(capsys) == "RESULT: GREEN"


def test_failed_configure_marks_reproduction_not_run(
    complete_host, monkeypatch, capsys
):
    rc, calls = _main(
        complete_host,
        monkeypatch,
        "--allow-incomplete",
        rcs={"configure arch-domain reproducibility CTest": 1},
    )
    assert rc == 1
    labels = [label for label, _command, _cwd in calls]
    assert "arch-domain reproducibility CTest" not in labels
    out = capsys.readouterr().out
    assert (
        "not run           arch-domain reproducibility CTest (configure failed)" in out
    )


def test_unknown_step_is_a_usage_error(runner, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _main(runner, monkeypatch, "--steps", "bogus")
    assert exc.value.code == 2


def test_list_exits_zero_without_a_verdict(runner, monkeypatch, capsys):
    rc, calls = _main(runner, monkeypatch, "--list")
    assert rc == 0
    assert calls == []
    assert "RESULT:" not in capsys.readouterr().out


def test_default_build_root_is_per_user(runner, monkeypatch):
    from rocke.tests.runner_verdict import user_scratch_dir

    rc, calls = _main(runner, monkeypatch, "--steps", "reproduction")
    assert rc == 0
    configure = next(c for label, c, _cwd in calls if label.startswith("configure"))
    assert configure[configure.index("-B") + 1] == str(user_scratch_dir("rocke_checks"))
