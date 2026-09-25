# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The standard runner must supply native fixture coverage to both pytest passes."""

import importlib.util
import json
from pathlib import Path
import subprocess
import shutil

import pytest


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "run_all.py"
    spec = importlib.util.spec_from_file_location("rocke_run_all", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("ROCKE_STORAGE_TEST", raising=False)
    monkeypatch.delenv("ROCKE_BACKEND", raising=False)
    return module


@pytest.mark.parametrize(
    "filename", ["rocke_storage", "provider_rocke_storage_test.exe"]
)
def test_runner_passes_registered_fixture_to_both_backends(
    runner, monkeypatch, tmp_path, filename
):
    (tmp_path / "CMakeCache.txt").touch()
    (tmp_path / "CTestTestfile.cmake").touch()
    executable = tmp_path / "Debug" / filename
    executable.parent.mkdir()
    executable.touch()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        listing = {"tests": [{"name": "rocke_storage", "command": [str(executable)]}]}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(listing))

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_all.py",
            "--no-guard",
            "--no-gate",
            "--build-root",
            str(tmp_path),
            "--config",
            "Debug",
        ],
    )
    assert runner.main() == 0
    children = [
        (cmd, kw["env"]) for cmd, kw in calls if cmd[0] == runner.sys.executable
    ]
    assert len(children) == 3  # default pytest, extension import probe, both pytest
    assert all(env["ROCKE_STORAGE_TEST"] == str(executable) for _, env in children)
    assert "ROCKE_BACKEND" not in children[0][1]
    assert children[-1][1]["ROCKE_BACKEND"] == "both"
    assert calls[0][0] == [
        "cmake",
        "--build",
        str(tmp_path),
        "--config",
        "Debug",
    ]
    assert any(cmd[:3] == ["ctest", "-C", "Debug"] for cmd, _ in calls)


@pytest.mark.parametrize("built", ["none", "partial", "all"])
@pytest.mark.parametrize("ctest_rc", [0, 8])
def test_ctest_readiness_and_failure_propagation(
    runner, monkeypatch, tmp_path, capsys, built, ctest_rc
):
    (tmp_path / "CTestTestfile.cmake").touch()
    executable = tmp_path / "Debug" / "renamed.exe"
    executable.parent.mkdir()
    if built != "none":
        executable.touch()
    commands = [[str(executable)], [] if built != "all" else [str(executable)]]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        listing = {
            "tests": [
                {"name": str(i), "command": cmd} for i, cmd in enumerate(commands)
            ]
        }
        return subprocess.CompletedProcess(
            command,
            0 if "--show-only=json-v1" in command else ctest_rc,
            stdout=json.dumps(listing),
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_all.py",
            "--no-guard",
            "--no-gate",
            "--no-pytest",
            "--build-root",
            str(tmp_path),
            "--config",
            "Debug",
        ],
    )
    assert runner.main() == (0 if built == "none" else ctest_rc)
    executions = [cmd for cmd in calls if cmd[:3] == ["ctest", "-C", "Debug"]]
    assert len(executions) == (0 if built == "none" else 1)
    if built == "none":
        assert "ctest: SKIPPED" in capsys.readouterr().out


def test_ctest_discovery_failure_is_an_error(runner, monkeypatch, tmp_path):
    (tmp_path / "CTestTestfile.cmake").touch()

    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_all.py",
            "--no-guard",
            "--no-gate",
            "--no-pytest",
            "--build-root",
            str(tmp_path),
        ],
    )
    assert runner.main() == 1


def test_explicit_fixture_override_is_resolved_before_pytest_changes_directory(
    runner, monkeypatch, tmp_path
):
    executable = tmp_path / "fixture"
    executable.touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROCKE_STORAGE_TEST", "fixture")
    env = runner.native_pytest_env(tmp_path, "Release")
    assert env["ROCKE_STORAGE_TEST"] == str(executable)


def test_missing_build_reports_native_coverage_skip(runner, tmp_path, capsys):
    assert "ROCKE_STORAGE_TEST" not in runner.native_pytest_env(tmp_path, "Release")
    assert "native storage parity: SKIPPED" in capsys.readouterr().out


def test_invalid_override_is_an_error(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("ROCKE_STORAGE_TEST", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="does not exist"):
        runner.native_pytest_env(tmp_path, "Release")


@pytest.mark.parametrize("failure", ["build", "registration", "executable"])
def test_native_setup_failure_prevents_silently_skipped_pytest(
    runner, monkeypatch, tmp_path, failure
):
    (tmp_path / "CMakeCache.txt").touch()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "cmake" and failure == "build":
            raise subprocess.CalledProcessError(1, command)
        command_path = [str(tmp_path / "missing")] if failure == "executable" else []
        listing = {"tests": [{"name": "rocke_storage", "command": command_path}]}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(listing))

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        ["run_all.py", "--no-guard", "--no-gate", "--build-root", str(tmp_path)],
    )
    assert runner.main() == 1
    assert all(command[0] != runner.sys.executable for command in calls)


def test_fresh_build_prepares_entire_ctest_suite(runner, tmp_path):
    if not shutil.which("cmake") or not shutil.which("ctest"):
        pytest.skip("CMake and CTest required for fresh-build regression")
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.c").write_text("int main(void) { return 0; }\n")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(runner_fixture C)\n"
        "enable_testing()\n"
        "add_library(rocke_core STATIC main.c)\n"
        "add_executable(rocke_storage main.c)\n"
        "add_executable(rocke_dtypes main.c)\n"
        "add_test(NAME rocke_storage COMMAND rocke_storage)\n"
        "add_test(NAME rocke_dtypes COMMAND rocke_dtypes)\n"
    )
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True)
    # Reproduce the byte-identity gate's partial build before pytest setup.
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "rocke_core"], check=True
    )
    env = runner.native_pytest_env(build, "Release")
    assert Path(env["ROCKE_STORAGE_TEST"]).is_file()
    assert runner.ctest_ready(build, "Release")
    subprocess.run(
        ["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"],
        check=True,
    )
