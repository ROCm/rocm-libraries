# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The standard runner must supply native fixture coverage to both pytest passes.

Each test drives ``main()`` against a fake ``subprocess.run`` and isolates the
native pytest/ctest plumbing by opting out of every other step, so a new
default-on step needs its ``--no-...`` flag added to the argv lists below.
"""

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
    # Every native parity var: run_all.py exports them into its own pytest pass,
    # and a leaked value would let these tests pass without exercising CTest.
    for var, _, _ in module.NATIVE_PARITY_TESTS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("ROCKE_BACKEND", raising=False)
    monkeypatch.delenv("ROCKE_TEST_ENGINE_ARCHIVE", raising=False)
    monkeypatch.delenv("ROCKE_ONLINE_LIB", raising=False)
    monkeypatch.delenv("ROCKE_REPLAY_CLI", raising=False)
    monkeypatch.delenv("ROCKE_PROVENANCE_MANIFEST", raising=False)
    return module


@pytest.mark.parametrize(
    "filename", ["rocke_storage", "provider_rocke_storage_test.exe"]
)
@pytest.mark.parametrize("override", [False, True])
def test_runner_passes_registered_fixture_to_both_backends(
    runner, monkeypatch, tmp_path, filename, override
):
    (tmp_path / "CMakeCache.txt").touch()
    (tmp_path / "CTestTestfile.cmake").touch()
    executable = tmp_path / "Debug" / filename
    executable.parent.mkdir()
    executable.touch()
    ir_types = tmp_path / "Debug" / "rocke_ir_types"
    ir_types.touch()
    nontemporal = tmp_path / "Debug" / "rocke_nontemporal_hip"
    nontemporal.touch()
    unsigned_int_ops = tmp_path / "Debug" / "rocke_unsigned_int_ops"
    unsigned_int_ops.touch()
    target_identity = tmp_path / "Debug" / "rocke_arch_target_identity"
    target_identity.touch()
    shared = tmp_path / "librocke.so"
    replay = tmp_path / "tests" / "rocke_portable_ir_replay_cli"
    replay.parent.mkdir(exist_ok=True)
    shared.touch()
    replay.touch()
    buffer_rsrc = tmp_path / "Debug" / "rocke_buffer_rsrc_hip"
    buffer_rsrc.touch()
    selected = tmp_path / "override" if override else executable
    if override:
        selected.touch()
        monkeypatch.setenv("ROCKE_STORAGE_TEST", str(selected))
    archive = tmp_path / "Debug" / "librocke_core.a"
    archive.touch()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        listing = {
            "tests": [
                {"name": "rocke_storage", "command": [str(executable)]},
                {"name": "rocke_ir_types", "command": [str(ir_types)]},
                {"name": "rocke_nontemporal_hip", "command": [str(nontemporal)]},
                {
                    "name": "rocke_unsigned_int_ops",
                    "command": [str(unsigned_int_ops)],
                },
                {
                    "name": "rocke_arch_target_identity",
                    "command": [str(target_identity)],
                },
                {"name": "rocke_buffer_rsrc_hip", "command": [str(buffer_rsrc)]},
            ]
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(listing))

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_all.py",
            "--no-guard",
            "--no-gate",
            "--no-ir-validity",
            "--no-arch-domain",
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
    assert all(env["ROCKE_STORAGE_TEST"] == str(selected) for _, env in children)
    assert all(env["ROCKE_IR_TYPES_TEST"] == str(ir_types) for _, env in children)
    assert all(
        env["ROCKE_NONTEMPORAL_HIP_TEST"] == str(nontemporal) for _, env in children
    )
    assert all(env["ROCKE_TEST_ENGINE_ARCHIVE"] == str(archive) for _, env in children)
    assert all(
        env["ROCKE_UNSIGNED_INT_OPS_CLI"] == str(unsigned_int_ops)
        for _, env in children
    )
    assert all(
        env["ROCKE_ARCH_TARGET_TEST_EXE"] == str(target_identity) for _, env in children
    )
    assert all(env["ROCKE_ONLINE_LIB"] == str(shared) for _, env in children)
    assert all(env["ROCKE_REPLAY_CLI"] == str(replay) for _, env in children)
    assert all(
        env["ROCKE_BUFFER_RSRC_HIP_TEST"] == str(buffer_rsrc) for _, env in children
    )
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
    # An explicit override skips the storage lookup; other binaries still resolve.
    assert any("^rocke_storage$" in cmd for cmd, _ in calls) is not override


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
            "--no-ir-validity",
            "--no-arch-domain",
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


def test_source_script_alone_does_not_claim_the_native_suite_is_ready(
    runner, monkeypatch, tmp_path, capsys
):
    (tmp_path / "CTestTestfile.cmake").touch()

    def run(command, **kwargs):
        listing = {
            "tests": [
                {
                    "name": "rocke_arch_domain_reproducibility",
                    "command": [runner.sys.executable, "generator.py", "--check"],
                }
            ]
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(listing))

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert not runner.ctest_ready(tmp_path, "Release")
    assert "no built registered tests" in capsys.readouterr().out


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
            "--no-ir-validity",
            "--no-arch-domain",
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


def test_pytest_build_configuration_owns_shared_engine(runner, monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.configure_pytest_build(tmp_path, "Debug")

    assert calls == [
        [
            "cmake",
            "-S",
            str(runner.ROCKE),
            "-B",
            str(tmp_path),
            "-DCMAKE_BUILD_TYPE=Debug",
            f"-DPython3_EXECUTABLE={runner.sys.executable}",
            "-DROCKE_BUILD_SHARED_ENGINE=ON",
            "-DBUILD_TESTING=ON",
        ]
    ]


def test_arch_domain_ctest_registration_contract(runner, tmp_path):
    if not (runner.ROCKE / "CMakeLists.txt").is_file():
        pytest.skip("source CMake project is absent from the installed test artifact")
    if not all(shutil.which(tool) for tool in ("cmake", "ctest")):
        pytest.skip("CMake and CTest required for registration regression")

    build = tmp_path / "build"
    runner.configure_pytest_build(build, "Release")
    tests = runner.ctest_tests(build, "Release", "^rocke_arch_domain_reproducibility$")
    assert len(tests) == 1
    test = tests[0]
    assert test["name"] == "rocke_arch_domain_reproducibility"
    assert test["command"][:2] == [runner.sys.executable, "-u"]
    assert (
        Path(test["command"][2]).resolve()
        == (runner.ROCKE / "tools" / "gen_arch_domain.py").resolve()
    )
    assert test["command"][3:] == [
        "--check",
        "--unvalidated-exit-code",
        "77",
    ]
    properties = {item["name"]: item["value"] for item in test["properties"]}
    assert properties["SKIP_RETURN_CODE"] == 77
    assert properties["RUN_SERIAL"] is True
    assert properties["TIMEOUT"] == 1800.0
    assert "integration" in properties["LABELS"]
    assert "standard" in properties["LABELS"]

    # Provider CI runs the generated installed manifest, not this build-tree
    # registration. Pin both halves of the contract so a local run cannot be
    # green while packaging silently drops the gate.
    installed_manifest = (build / "CTestTestfile.cmake.install").read_text()
    assert "add_test(rocke_arch_domain_reproducibility" in installed_manifest
    assert "tests/arch_domain/gen_arch_domain.py" in installed_manifest
    assert "--check --unvalidated-exit-code 77" in installed_manifest
    assert "SKIP_RETURN_CODE 77" in installed_manifest
    assert "RUN_SERIAL TRUE" in installed_manifest
    assert "TIMEOUT 1800" in installed_manifest
    assert (
        'LABELS "ckc;python;host;integration;toolchain;generated-artifact;standard"'
        in installed_manifest
    )


def test_missing_build_reports_native_coverage_skip(runner, tmp_path, capsys):
    assert "ROCKE_STORAGE_TEST" not in runner.native_pytest_env(tmp_path, "Release")
    assert "native storage parity: SKIPPED" in capsys.readouterr().out


@pytest.mark.parametrize("subdir", ["", "Release"])
def test_existing_engine_archive_is_reused_by_pytest(runner, tmp_path, subdir):
    archive = tmp_path / subdir / "librocke_core.a"
    archive.parent.mkdir(exist_ok=True)
    archive.touch()
    env = runner.native_pytest_env(tmp_path, "Release")
    assert env["ROCKE_TEST_ENGINE_ARCHIVE"] == str(archive.resolve())


def test_configured_archive_wins_over_another_root_archive(runner, tmp_path):
    (tmp_path / "librocke_core.a").touch()
    archive = tmp_path / "Debug" / "librocke_core.a"
    archive.parent.mkdir()
    archive.touch()
    assert runner.native_pytest_env(tmp_path, "Debug")[
        "ROCKE_TEST_ENGINE_ARCHIVE"
    ] == str(archive)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_invalid_engine_archive_override_is_an_error(
    runner, monkeypatch, tmp_path, kind
):
    archive = tmp_path / "invalid.a"
    if kind == "directory":
        archive.mkdir()
    monkeypatch.setenv("ROCKE_TEST_ENGINE_ARCHIVE", str(archive))
    with pytest.raises(ValueError, match="native detector archive does not exist"):
        runner.native_pytest_env(tmp_path, "Release")


def test_configured_portable_ir_artifacts_replace_external_cache(
    runner, monkeypatch, tmp_path
):
    shared = tmp_path / "librocke.so"
    replay = tmp_path / "tests" / "rocke_portable_ir_replay_cli"
    replay.parent.mkdir()
    shared.touch()
    replay.touch()
    monkeypatch.setenv("ROCKE_ONLINE_LIB", str(tmp_path / "stale.so"))
    monkeypatch.setenv("ROCKE_REPLAY_CLI", str(tmp_path / "stale-cli"))

    env = runner.native_pytest_env(tmp_path, "Release")

    assert env["ROCKE_ONLINE_LIB"] == str(shared.resolve())
    assert env["ROCKE_REPLAY_CLI"] == str(replay.resolve())


def test_missing_owned_portable_artifacts_clear_external_values(
    runner, monkeypatch, tmp_path
):
    monkeypatch.setenv("ROCKE_ONLINE_LIB", str(tmp_path / "stale.so"))
    monkeypatch.setenv("ROCKE_REPLAY_CLI", str(tmp_path / "stale-cli"))
    monkeypatch.setenv("ROCKE_PROVENANCE_MANIFEST", str(tmp_path / "stale-manifest"))

    env = runner.native_pytest_env(tmp_path, "Release")

    assert "ROCKE_ONLINE_LIB" not in env
    assert "ROCKE_REPLAY_CLI" not in env
    assert "ROCKE_PROVENANCE_MANIFEST" not in env


def test_explicit_engine_archive_override_wins(runner, monkeypatch, tmp_path):
    built = tmp_path / "librocke_core.a"
    built.touch()
    override = tmp_path / "override.a"
    override.touch()
    monkeypatch.setenv("ROCKE_TEST_ENGINE_ARCHIVE", str(override))
    env = runner.native_pytest_env(tmp_path, "Release")
    assert env["ROCKE_TEST_ENGINE_ARCHIVE"] == str(override.resolve())


def test_invalid_override_is_an_error(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("ROCKE_STORAGE_TEST", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="does not exist"):
        runner.native_pytest_env(tmp_path, "Release")


@pytest.mark.parametrize(
    "failure", ["build", "build_override", "registration", "executable"]
)
def test_native_setup_failure_prevents_silently_skipped_pytest(
    runner, monkeypatch, tmp_path, failure
):
    (tmp_path / "CMakeCache.txt").touch()
    if failure == "build_override":
        executable = tmp_path / "override"
        executable.touch()
        monkeypatch.setenv("ROCKE_STORAGE_TEST", str(executable))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "cmake" and failure.startswith("build"):
            raise subprocess.CalledProcessError(1, command)
        command_path = [str(tmp_path / "missing")] if failure == "executable" else []
        listing = {"tests": [{"name": "rocke_storage", "command": command_path}]}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(listing))

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_all.py",
            "--no-guard",
            "--no-gate",
            "--no-ir-validity",
            "--no-arch-domain",
            "--build-root",
            str(tmp_path),
        ],
    )
    assert runner.main() == 1
    assert all(command[0] != runner.sys.executable for command in calls)


@pytest.mark.parametrize("override", [False, True])
def test_fresh_build_prepares_entire_ctest_suite(
    runner, monkeypatch, tmp_path, override
):
    if not all(shutil.which(tool) for tool in ("cmake", "ctest", "ninja")):
        pytest.skip("CMake, CTest, and Ninja required for fresh-build regression")
    source = tmp_path / "source"
    source.mkdir()
    # Exercise real build/discovery/execution without requiring a C toolchain.
    # Copy CMake itself as each executable: `-E true` supplies the trivial
    # success behavior, and the executable exists only after its target builds.
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(runner_fixture LANGUAGES NONE)\n"
        "enable_testing()\n"
        'add_custom_target(rocke_core COMMAND "${CMAKE_COMMAND}" -E touch core.ready)\n'
        'get_filename_component(tool_suffix "${CMAKE_COMMAND}" LAST_EXT)\n'
        "foreach(name rocke_storage rocke_ir_types rocke_nontemporal_hip\n"
        "             rocke_unsigned_int_ops rocke_arch_target_identity rocke_buffer_rsrc_hip\n"
        "             rocke_dtypes)\n"
        '  set(output "${CMAKE_CURRENT_BINARY_DIR}/fixtures/${name}${tool_suffix}")\n'
        '  add_custom_command(OUTPUT "${output}"\n'
        '    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/fixtures"\n'
        '    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_COMMAND}" "${output}"\n'
        "    VERBATIM)\n"
        '  add_custom_target(${name} ALL DEPENDS "${output}")\n'
        '  add_test(NAME ${name} COMMAND "${output}" -E true)\n'
        "endforeach()\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        ["cmake", "-G", "Ninja", "-S", str(source), "-B", str(build)], check=True
    )
    # Reproduce the byte-identity gate's partial build before pytest setup.
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "rocke_core"], check=True
    )
    if override:
        # An external fixture must not suppress the configured suite's build.
        executable = tmp_path / "override"
        executable.touch()
        monkeypatch.setenv("ROCKE_STORAGE_TEST", str(executable))
    env = runner.native_pytest_env(build, "Release")
    assert Path(env["ROCKE_STORAGE_TEST"]).is_file()
    assert Path(env["ROCKE_IR_TYPES_TEST"]).is_file()
    assert Path(env["ROCKE_NONTEMPORAL_HIP_TEST"]).is_file()
    assert Path(env["ROCKE_UNSIGNED_INT_OPS_CLI"]).is_file()
    assert Path(env["ROCKE_ARCH_TARGET_TEST_EXE"]).is_file()
    assert Path(env["ROCKE_BUFFER_RSRC_HIP_TEST"]).is_file()
    if override:
        assert env["ROCKE_STORAGE_TEST"] == str(executable)
    assert runner.ctest_ready(build, "Release")
    subprocess.run(
        ["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"],
        check=True,
    )


def test_multi_config_build_supplies_its_archive(runner, monkeypatch, tmp_path):
    if runner.sys.platform != "linux" or not all(
        shutil.which(tool) for tool in ("cmake", "ninja", "c++")
    ):
        pytest.skip("Linux, CMake, Ninja, and C++ compiler required")
    source = tmp_path / "source"
    source.mkdir()
    (source / "core.cpp").write_text("int fixture() { return 0; }\n")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(archive_fixture LANGUAGES CXX)\n"
        "add_library(rocke_core STATIC core.cpp)\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        ["cmake", "-G", "Ninja Multi-Config", "-S", str(source), "-B", str(build)],
        check=True,
    )
    # Native executable discovery is independent of the archive being tested.
    monkeypatch.setattr(runner, "NATIVE_PARITY_TESTS", ())
    env = runner.native_pytest_env(build, "Release")
    archive = build / "Release" / "librocke_core.a"
    assert archive.is_file()
    assert env["ROCKE_TEST_ENGINE_ARCHIVE"] == str(archive.resolve())
