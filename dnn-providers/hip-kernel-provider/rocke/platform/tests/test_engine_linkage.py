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
from rocke.tests.provider_categories import provider_category_labels

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


requires_cmake = pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ctest") is None,
    reason="CMake and CTest required",
)


def configure_installed_ctest(tmp_path, *cmake_arguments):
    """Configure the platform with BUILD_TESTING and return the build directory
    and a directory holding its installed CTestTestfile, as an install lays it
    out."""
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
            *cmake_arguments,
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
    return build, ctest_dir


def list_installed_ctests(ctest_dir, env=None):
    """The tests CTest lists for ``ctest_dir``, in its json-v1 form."""
    listed = subprocess.run(
        ["ctest", "--show-only=json-v1", "--test-dir", str(ctest_dir)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(listed.stdout)["tests"]


@requires_cmake
@pytest.mark.parametrize(
    "provider_subdir",
    ["", "hip_kernel_provider"],
    ids=["standalone", "provider"],
)
@pytest.mark.parametrize(
    "suffix_override", [None, ".test-bin"], ids=["host-suffix", "explicit-suffix"]
)
def test_installed_ctest_entries_get_scoped_environment_contracts(
    tmp_path, provider_subdir, suffix_override
):
    # Exercise filename construction with a real CMake suffix override, not a
    # substituted Python host platform. This is not a Windows execution test.
    project_include = tmp_path / "suffix.cmake"
    project_include.write_text(
        (
            f'set(CMAKE_EXECUTABLE_SUFFIX "{suffix_override}")\n'
            if suffix_override
            else ""
        )
        + 'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/emitter-name.txt" '
        'CONTENT "$<TARGET_FILE_NAME:rocke_optimization_barrier_emit>")\n'
    )
    build, ctest_dir = configure_installed_ctest(
        tmp_path,
        f"-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR={provider_subdir}",
        f"-DCMAKE_PROJECT_INCLUDE={project_include}",
    )
    executable_suffix = (
        suffix_override if suffix_override else (".exe" if os.name == "nt" else "")
    )
    emitter_filename = (build / "emitter-name.txt").read_text()
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
        "ROCKE_INTEXPR_CLI": native_artifact("rocke_intexpr"),
        "ROCKE_PARAM_FACTS_CLI": native_artifact("rocke_param_facts"),
        "ROCKE_GPU_CHECKED_VALUE_CLI": native_artifact("rocke_gpu_checked_value"),
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
        tests = list_installed_ctests(ctest_dir, env=ctest_env)
        pair = next(
            test
            for test in tests
            if test["name"] == "rocke_optimization_barrier_hip_pair"
        )
        # A mixed private-variable spelling can configure successfully while
        # appending this incoming registration to an unused CMake string.
        sdpa = next(
            test for test in tests if test["name"] == "rocke_library_sdpa_pytest"
        )
        assert f"{test_ctest_rel}tests/library/tests/sdpa" in sdpa["command"]
        assert environment_for(tests, "rocke_library_sdpa_pytest") == {
            "PYTHONPATH": f"{test_ctest_rel}tests/library"
        }
        if "--rocke-require-hipdnn-assets" in sdpa["command"]:
            install_script = (build / "cmake_install.cmake").read_text()
            assert "attention_sdpa_fwd.graph.json" in install_script
            assert "/tests/library/tests/sdpa/hipdnn" in install_script
        command = pair["command"]
        native = command[command.index("--native") + 1]
        assert native == native_artifact("rocke_optimization_barrier_emit")
        assert Path(native).name == emitter_filename
        assert command[command.index("--count") + 1] == "17"
        expected_platform_environment = {
            "PYTHONPATH": ".",
            **expected_artifacts,
        }
        expected_mla_environment = {"PYTHONPATH": f"{test_ctest_rel}tests/library"}
        # The tensor_transforms tests run from the installed package, under the
        # installed tests/pytest.ini, with the package root on PYTHONPATH.
        tensor_transforms = next(
            test for test in tests if test["name"] == "rocke_tensor_transforms_pytest"
        )
        package_root = environment_for(tests, "rocke_tensor_transforms_pytest")[
            "PYTHONPATH"
        ]
        tensor_transforms_command = tensor_transforms["command"]
        assert (
            tensor_transforms_command[tensor_transforms_command.index("-c") + 1]
            == f"{test_ctest_rel}tests/pytest.ini"
        )
        assert (
            f"{package_root}/rocke/tests/helpers/tiling/tensor_transforms"
            in tensor_transforms_command
        )
        expected_tensor_transforms_environment = {"PYTHONPATH": package_root}
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
            expected_tensor_transforms_environment.update(compiler_environment)
            expected_mla_environment.update(compiler_environment)
            expected_mla_environment["ROCKE_REQUIRE_COMPILER_TOOLS"] = "1"

        platform_pytest_parts = [
            test["name"] for test in tests if test["name"].startswith("rocke_pytest_")
        ]
        assert "rocke_pytest_other" in platform_pytest_parts
        for part in platform_pytest_parts:
            assert environment_for(tests, part) == expected_platform_environment
        assert (
            environment_for(tests, "rocke_tensor_transforms_pytest")
            == expected_tensor_transforms_environment
        )
        assert (
            environment_for(tests, "rocke_library_mla_pytest")
            == expected_mla_environment
        )


@requires_cmake
@pytest.mark.parametrize(
    "provider_subdir",
    ["", "hip_kernel_provider"],
    ids=["standalone", "provider"],
)
def test_installed_platform_pytest_parts_partition_the_suite(tmp_path, provider_subdir):
    """The parts run tests/ minus the library tests in disjoint pieces: each
    named part runs whole top-level directories of tests/ that exist, no
    directory runs in two parts, and rocke_pytest_other runs tests/ with the
    library tests and every named part's directories ignored. Every part keeps
    the suite's options, labels and time limit."""
    _, ctest_dir = configure_installed_ctest(
        tmp_path, f"-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR={provider_subdir}"
    )
    parts = [
        test
        for test in list_installed_ctests(ctest_dir)
        if test["name"].startswith("rocke_pytest_")
    ]
    tests_root = ("./" if provider_subdir else "../") + "tests"
    suite_options = ["-m", "not integration", "-v", "-rs", "--timeout=60"]
    selection_by_part = {}
    for part in parts:
        name, command = part["name"], part["command"]
        assert command[1:3] == ["-m", "pytest"], name
        assert command[-len(suite_options) :] == suite_options, name
        selection_by_part[name] = command[3 : -len(suite_options)]
        properties = {prop["name"]: prop["value"] for prop in part["properties"]}
        # One label string, as the single entry had: the generated CTest text
        # escapes the semicolons.
        assert properties["LABELS"] == ["ckc;python;quick"], name
        assert properties["TIMEOUT"] == 300, name

    other_selection = selection_by_part.pop("rocke_pytest_other")
    assert selection_by_part and all(selection_by_part.values())
    named_directories = [
        path for selection in selection_by_part.values() for path in selection
    ]
    assert len(named_directories) == len(set(named_directories))
    for path in named_directories:
        parent, _, directory = path.rpartition("/")
        assert parent == tests_root, path
        assert directory != "library", path
        assert (PLATFORM / "tests" / directory).is_dir(), path
    expected_other_selection = [tests_root, "--ignore", f"{tests_root}/library"]
    for path in named_directories:
        expected_other_selection += ["--ignore", path]
    assert other_selection == expected_other_selection


@requires_cmake
def test_installed_platform_pytest_parts_keep_the_suite_tier_labels(tmp_path):
    """TheRock CI selects provider tests by tier label, so each part needs every
    label of rocke_tensor_transforms_pytest, the other platform pytest entry,
    which the provider categories list in the same tiers."""
    _, ctest_dir = configure_installed_ctest(tmp_path)
    parts = [
        test["name"]
        for test in list_installed_ctests(ctest_dir)
        if test["name"].startswith("rocke_pytest_")
    ]
    anchor = "rocke_tensor_transforms_pytest"
    labels = provider_category_labels([anchor, *parts], tmp_path)
    assert labels.get(anchor), f"the provider categories no longer label {anchor}"
    for part in parts:
        missing = labels[anchor] - labels.get(part, set())
        assert not missing, (
            f"{part} lacks the {anchor} labels {sorted(missing)}: the provider "
            "categories do not list it"
        )


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
