# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Archive consumers carry the loader and thread dependencies of the lowerer."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rocke.portable_ir.src import online

PLATFORM = Path(__file__).resolve().parents[1]


def load_driver(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("consumer", ["differential", "micro_parity"])
def test_raw_archive_link_dependencies(monkeypatch, tmp_path, platform, consumer):
    def complete_compile(command, **_kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).touch()
        return SimpleNamespace(returncode=0, stdout=b"", stderr="")

    run = Mock(side_effect=complete_compile)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "platform", platform)
    if consumer == "differential":
        driver = load_driver(
            "link_test_run_diff",
            PLATFORM / "tests/instances/differential/run_diff.py",
        )
        monkeypatch.setattr(driver, "TMP", tmp_path)
        # Fingerprint invalidation has dedicated differential-runner coverage;
        # this test isolates link-command construction and cache publication.
        monkeypatch.setattr(
            driver, "_native_emitter_fingerprint", lambda *_args, **_kwargs: "fixture"
        )
        archive = tmp_path / "librocke_core.a"
        archive.touch()
        (tmp_path / "fixture_emit.c").write_text("int main() { return 0; }\n")
        assert driver.compile_c("fixture", archive, src_dir=tmp_path)[0]
    else:
        driver = load_driver(
            "link_test_run_parity", PLATFORM / "tests/instances/parity/run_parity.py"
        )
        archive = tmp_path / "librocke_core.a"
        archive.touch()
        monkeypatch.setattr(sys, "argv", ["run_parity", "--archive", str(archive)])
        monkeypatch.setattr(driver, "_cxx", lambda: "c++")
        monkeypatch.setattr(driver.tempfile, "gettempdir", lambda: str(tmp_path))
        assert driver.main() == 0
    commands = [
        call.args[0]
        for call in run.call_args_list
        if any(str(arg).endswith("librocke_core.a") for arg in call.args[0])
    ]
    assert len(commands) == 1
    command = commands[0]
    archive_index = next(
        i for i, arg in enumerate(command) if str(arg).endswith("librocke_core.a")
    )
    for flag in ("-ldl", "-pthread"):
        if platform == "linux":
            assert command.index(flag) > archive_index
        else:
            assert flag not in command


@pytest.mark.parametrize(
    ("platform", "library_name"),
    [("linux", "librocke.so"), ("darwin", "librocke.dylib"), ("win32", "rocke.dll")],
)
def test_online_build_uses_the_cmake_shared_target(
    monkeypatch, tmp_path, platform, library_name
):
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if "--build" in command:
            stage = tmp_path / f".stage-{os.getpid()}"
            (stage / library_name).touch()
        return SimpleNamespace(returncode=0, stdout=b"", stderr="")

    monkeypatch.setattr(online.subprocess, "run", run)
    monkeypatch.setattr(online.sys, "platform", platform)
    destination = tmp_path / library_name

    assert online.build_lib(str(destination)) == str(destination)
    assert destination.is_file()
    build = next(command for command in calls if "--build" in command)
    assert build[build.index("--target") + 1] == "rocke_shared"
    configure = next(command for command in calls if "-S" in command)
    assert "-DROCKE_BUILD_SHARED_ENGINE=ON" in configure
    assert "-DBUILD_TESTING=ON" in configure


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ctest") is None,
    reason="CMake and CTest required",
)
@pytest.mark.parametrize(
    "provider_subdir",
    ["", "hip_kernel_provider"],
    ids=["standalone", "provider"],
)
def test_installed_ctest_entries_get_scoped_environment_contracts(
    tmp_path, provider_subdir
):
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(PLATFORM),
            "-B",
            str(build),
            "-DBUILD_TESTING=ON",
            "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
            f"-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR={provider_subdir}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ctest_dir = tmp_path / "installed-ctest"
    ctest_dir.mkdir()
    shutil.copy2(
        build / "CTestTestfile.cmake.install", ctest_dir / "CTestTestfile.cmake"
    )
    executable_suffix = ".exe" if os.name == "nt" else ""
    test_ctest_rel = "./" if provider_subdir else "../"

    def native_artifact(target):
        if provider_subdir:
            target = f"{provider_subdir}_{target}_test"
        return f"../{target}{executable_suffix}"

    expected_artifacts = {
        "ROCKE_REPLAY_CLI": (
            f"{test_ctest_rel}tests/portable_ir/"
            f"rocke_portable_ir_replay_cli{executable_suffix}"
        ),
        "ROCKE_PROVENANCE_MANIFEST": (
            f"{test_ctest_rel}tests/portable_ir/rocke_portable_ir_provenance.txt"
        ),
        "ROCKE_STORAGE_TEST": native_artifact("rocke_storage"),
        "ROCKE_IR_TYPES_TEST": native_artifact("rocke_ir_types"),
        "ROCKE_NONTEMPORAL_HIP_TEST": native_artifact("rocke_nontemporal_hip"),
        "ROCKE_UNSIGNED_INT_OPS_CLI": native_artifact("rocke_unsigned_int_ops"),
        "ROCKE_ARCH_TARGET_TEST_EXE": native_artifact("rocke_arch_target_identity"),
        "ROCKE_BUFFER_RSRC_HIP_TEST": native_artifact("rocke_buffer_rsrc_hip"),
    }

    def environment_for(tests, test_name):
        test = next(test for test in tests if test["name"] == test_name)
        properties = {prop["name"]: prop["value"] for prop in test["properties"]}
        pairs = [item.split("=", 1) for item in properties["ENVIRONMENT"]]
        names = [name for name, _ in pairs]
        assert len(names) == len(set(names))
        return dict(pairs)

    for rocm_path in (None, tmp_path / "synthetic-rocm"):
        ctest_env = dict(os.environ)
        if rocm_path is None:
            ctest_env.pop("ROCM_PATH", None)
        else:
            ctest_env["ROCM_PATH"] = str(rocm_path)
        listed = subprocess.run(
            ["ctest", "--show-only=json-v1", "--test-dir", str(ctest_dir)],
            check=True,
            capture_output=True,
            text=True,
            env=ctest_env,
        )
        tests = json.loads(listed.stdout)["tests"]
        expected_platform_environment = {
            "PYTHONPATH": ".",
            **expected_artifacts,
        }
        expected_mla_environment = {"PYTHONPATH": f"{test_ctest_rel}tests/library"}
        if sys.platform == "linux" and rocm_path is not None:
            compiler_environment = {
                "PATH": os.pathsep.join(
                    [
                        str(rocm_path / "bin"),
                        str(rocm_path / "lib/llvm/bin"),
                        ctest_env.get("PATH", ""),
                    ]
                ),
                "HIP_DEVICE_LIB_PATH": str(rocm_path / "lib/llvm/amdgcn/bitcode"),
            }
            expected_platform_environment.update(compiler_environment)
            expected_mla_environment.update(compiler_environment)
            expected_mla_environment["ROCKE_REQUIRE_COMPILER_TOOLS"] = "1"

        assert environment_for(tests, "rocke_pytest") == expected_platform_environment
        assert (
            environment_for(tests, "rocke_library_mla_pytest")
            == expected_mla_environment
        )


def test_relocated_launcher_suite_has_its_test_assets(tmp_path):
    """Exercise installed tests without borrowing helpers from the source tree."""
    build = os.environ.get("ROCKE_LINK_TEST_BUILD_DIR")
    if not build:
        pytest.skip("set ROCKE_LINK_TEST_BUILD_DIR to a fresh engine build")
    prefix = tmp_path / "install"
    subprocess.run(
        ["cmake", "--install", build, "--prefix", str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )
    relocated = tmp_path / "relocated"
    prefix.rename(relocated)
    modules = list(relocated.rglob("tests/test_comgr_test_launcher.py"))
    assert len(modules) == 1
    test_file = modules[0]
    helper = test_file.parent / "tools" / "run_with_comgr.py"
    assert helper.read_bytes() == (PLATFORM / "tools/run_with_comgr.py").read_bytes()
    launchers = list(relocated.rglob("rocke/tests/subprocess_launcher.py"))
    assert len(launchers) == 1
    # Invoke the relocated launcher, not the parent package's helper: this lane
    # deliberately tests a separately selected, source-less package identity.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "ROCKE_"))
    }
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(launchers[0]),
            "-m",
            "pytest",
            str(test_file),
            "-q",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "7 passed" in result.stdout


def test_relocated_cmake_consumer_links_lowerer(tmp_path):
    """Run against a fresh build supplied by the host validation lane."""
    build = os.environ.get("ROCKE_LINK_TEST_BUILD_DIR")
    if not build:
        pytest.skip("set ROCKE_LINK_TEST_BUILD_DIR to a fresh engine build")
    prefix = tmp_path / "install"
    subprocess.run(["cmake", "--install", build, "--prefix", str(prefix)], check=True)
    relocated = tmp_path / "relocated"
    prefix.rename(relocated)
    source = tmp_path / "consumer"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(consumer LANGUAGES CXX)\n"
        "find_package(ckc CONFIG REQUIRED)\n"
        "if(NOT TARGET Threads::Threads)\n"
        '  message(FATAL_ERROR "missing transitive thread dependency")\n'
        "endif()\n"
        "add_executable(consumer main.cpp)\n"
        "target_link_libraries(consumer PRIVATE ckc::rocke_core)\n"
    )
    # Taking the API address forces archive extraction without needing a GPU or
    # a valid kernel. The executable must resolve the detector dependencies.
    (source / "main.cpp").write_text(
        '#include "rocke/lower_llvm.h"\n'
        "auto volatile lower = &rocke_lower_kernel_to_llvm_ex;\n"
        "int main() { return lower == nullptr; }\n"
    )
    output = tmp_path / "consumer-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(output),
            f"-DCMAKE_PREFIX_PATH={relocated}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(output)], check=True)
    executable = output / ("consumer.exe" if os.name == "nt" else "consumer")
    subprocess.run([str(executable)], check=True)
