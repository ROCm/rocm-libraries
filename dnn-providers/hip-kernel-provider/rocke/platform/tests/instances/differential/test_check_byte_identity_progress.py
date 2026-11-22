# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Progress reporting for the byte-identity gate."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize(
    ("flavor", "supports_target_ops", "expected_gate_calls"),
    [(None, False, 3), ("llvm23", True, 2), ("llvm24", True, 2)],
)
def test_differential_gate_streams_unbuffered_progress(
    monkeypatch, tmp_path, flavor, supports_target_ops, expected_gate_calls
):
    if flavor is None:
        monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    else:
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    tools = Path(__file__).resolve().parents[3] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    path = tools / "check_byte_identity.py"
    spec = importlib.util.spec_from_file_location("check_byte_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["cmake", "--build"]:
            (tmp_path / "librocke_core.a").touch()
            # Library families link their own archive (library/cpp) when present.
            if "rocke_library_core" in command:
                (tmp_path / "librocke_library.a").touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "_cxx", lambda: None)
    monkeypatch.setattr(
        module, "_selected_profile_supports_target_ops", lambda: supports_target_ops
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["check_byte_identity.py", "--build-root", str(tmp_path)],
    )

    assert module.main() == 0
    gate_calls = [call for call in calls if str(module.RUN_DIFF) in call[0]]
    assert len(gate_calls) == expected_gate_calls
    ll_calls = [call for call in gate_calls if _mode(call[0]) == "ll"]
    if not supports_target_ops:
        primary, supplemental = ll_calls
        exclude_index = primary[0].index("--exclude")
        excluded_families = primary[0][exclude_index + 1].split(",")
        assert set(module.TARGET_OP_PROFILE_FAMILIES) <= set(excluded_families)
        assert supplemental[1]["env"]["ROCKE_LLVM_FLAVOR"] == "llvm23"
        only_index = supplemental[0].index("--only")
        supplemental_families = supplemental[0][only_index + 1].split(",")
        assert module.GFX1250_FAMILY in supplemental_families
        assert set(module.TARGET_OP_PROFILE_FAMILIES) <= set(supplemental_families)
    for command, kwargs in gate_calls:
        assert command[:3] == [module.sys.executable, "-u", str(module.RUN_DIFF)]
        if supports_target_ops and _mode(command) == "ll":
            assert "--exclude" not in command
        assert "capture_output" not in kwargs
        assert "stdout" not in kwargs
        assert "stderr" not in kwargs


def _mode(command):
    """The --mode a run_diff command line selects."""
    return command[command.index("--mode") + 1]


def _load_gate(monkeypatch):
    tools = Path(__file__).resolve().parents[3] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "check_byte_identity", tools / "check_byte_identity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("library_builds", [False, True])
def test_library_cpp_without_its_archive_stops_before_the_gate(
    monkeypatch, tmp_path, capsys, library_builds
):
    """library/cpp carries C++ mirrors, so the build must produce
    librocke_library.a; a build that does not stops here instead of running a
    gate whose library families would all fail to link."""
    module = _load_gate(monkeypatch)
    instances = tmp_path / "library" / "cpp" / "instances"
    instances.mkdir(parents=True)
    (instances / "mirror.cpp").write_text("")
    monkeypatch.setattr(module, "LIBRARY_CPP_INSTANCES", instances)
    build_root = tmp_path / "build"
    build_root.mkdir()

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["cmake", "--build"]:
            (build_root / "librocke_core.a").touch()
            if library_builds:
                (build_root / "librocke_library.a").touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "_cxx", lambda: None)
    monkeypatch.setattr(module, "_selected_profile_supports_target_ops", lambda: True)
    monkeypatch.setattr(
        module.sys, "argv", ["check_byte_identity.py", "--build-root", str(build_root)]
    )

    code = module.main()
    build = next(c for c in calls if c[:2] == ["cmake", "--build"])
    assert "rocke_library_core" in build
    gate_calls = [c for c in calls if str(module.RUN_DIFF) in c]
    if library_builds:
        assert code == 0
        assert all(
            c[c.index("--library-archive") + 1]
            == str(build_root / "librocke_library.a")
            for c in gate_calls
        )
    else:
        assert code == 1
        assert gate_calls == []
        assert "librocke_library.a" in capsys.readouterr().err


def test_no_library_cpp_builds_and_links_no_library_archive(monkeypatch, tmp_path):
    module = _load_gate(monkeypatch)
    monkeypatch.setattr(module, "LIBRARY_CPP_INSTANCES", tmp_path / "absent")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["cmake", "--build"]:
            (tmp_path / "librocke_core.a").touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "_cxx", lambda: None)
    monkeypatch.setattr(module, "_selected_profile_supports_target_ops", lambda: True)
    monkeypatch.setattr(
        module.sys, "argv", ["check_byte_identity.py", "--build-root", str(tmp_path)]
    )

    assert module.main() == 0
    build = next(c for c in calls if c[:2] == ["cmake", "--build"])
    assert "rocke_library_core" not in build
    gate_calls = [c for c in calls if str(module.RUN_DIFF) in c]
    assert gate_calls and all("--library-archive" not in c for c in gate_calls)


def _run_gate_with_ir_lane_exit(monkeypatch, tmp_path, ir_exit, *extra_args):
    """Run the gate with every lane green except the serialized-IR lane, whose
    run_diff exits ``ir_exit`` (1 is run_diff's exit on a drifting family).
    Returns the gate's exit code and the run_diff command lines with their
    keyword arguments."""
    module = _load_gate(monkeypatch)
    monkeypatch.setattr(module, "LIBRARY_CPP_INSTANCES", tmp_path / "absent")
    calls = []

    def run(command, **kwargs):
        if command[:2] == ["cmake", "--build"]:
            (tmp_path / "librocke_core.a").touch()
        if str(module.RUN_DIFF) in command:
            calls.append((command, kwargs))
            if _mode(command) == "ir":
                return subprocess.CompletedProcess(command, ir_exit)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "_cxx", lambda: None)
    monkeypatch.setattr(module, "_selected_profile_supports_target_ops", lambda: True)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["check_byte_identity.py", "--build-root", str(tmp_path), *extra_args],
    )
    return module.main(), calls


def test_serialized_ir_drift_fails_the_gate(monkeypatch, tmp_path, capsys):
    """The serialized-IR lane gates: a drifting family there turns the whole
    run RED even though every .ll lane is green."""
    code, calls = _run_gate_with_ir_lane_exit(monkeypatch, tmp_path, 1)
    assert code != 0
    ir_calls = [command for command, _ in calls if _mode(command) == "ir"]
    assert len(ir_calls) == 1
    # Bytes, not the canonical form: equal bytes are the contract.
    assert "--canonical" not in ir_calls[0]
    assert "RESULT: RED" in capsys.readouterr().out


def test_serialized_ir_lane_runs_at_the_host_flavor(monkeypatch, tmp_path):
    """Whatever flavor the invocation selects, the serialized-IR lane runs once,
    with ROCKE_LLVM_FLAVOR unset so the engines resolve the host's flavor."""
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm22")
    code, calls = _run_gate_with_ir_lane_exit(monkeypatch, tmp_path, 0)
    assert code == 0
    ir_runs = [kwargs for command, kwargs in calls if _mode(command) == "ir"]
    assert len(ir_runs) == 1
    assert "ROCKE_LLVM_FLAVOR" not in ir_runs[0]["env"]


def test_no_ir_skips_the_serialized_ir_lane(monkeypatch, tmp_path, capsys):
    """--no-ir runs the .ll lanes only, and says the run was not a full gate."""
    code, calls = _run_gate_with_ir_lane_exit(monkeypatch, tmp_path, 1, "--no-ir")
    assert code == 0
    assert calls and all(_mode(command) == "ll" for command, _ in calls)
    assert "skipped by --no-ir" in capsys.readouterr().out
