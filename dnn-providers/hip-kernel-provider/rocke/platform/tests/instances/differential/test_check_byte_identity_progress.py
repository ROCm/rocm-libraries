# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Progress reporting for the byte-identity gate."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize(
    ("flavor", "supports_target_ops", "expected_gate_calls"),
    [(None, False, 2), ("llvm23", True, 1), ("llvm24", True, 1)],
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
    if not supports_target_ops:
        primary, supplemental = gate_calls
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
        if supports_target_ops:
            assert "--exclude" not in command
        assert "capture_output" not in kwargs
        assert "stdout" not in kwargs
        assert "stderr" not in kwargs


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
