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


def split_ignored_files(selection):
    """Split a part's pytest selection into its leading paths and the files its
    trailing ``--ignore`` pairs name."""
    first_ignore = (
        selection.index("--ignore") if "--ignore" in selection else len(selection)
    )
    flags = selection[first_ignore::2]
    assert flags == ["--ignore"] * len(flags), selection
    return selection[:first_ignore], selection[first_ignore + 1 :: 2]


# Files that two parts keep out through _ROCKE_PLATFORM_PYTEST_IGNORE_<part>,
# set from the configure line: a file in a named part's directory and a file at
# the top of tests/, which rocke_pytest_other collects.
_IGNORED_FILES = {"core": "core/test_intexpr.py", "other": "test_rocke.py"}


@requires_cmake
@pytest.mark.parametrize(
    ("provider_subdir", "ignored_files"),
    [("", {}), ("hip_kernel_provider", {}), ("hip_kernel_provider", _IGNORED_FILES)],
    ids=["standalone", "provider", "provider-ignored-files"],
)
def test_installed_platform_pytest_parts_partition_the_suite(
    tmp_path, provider_subdir, ignored_files
):
    """The parts run tests/ minus the library tests in disjoint pieces: each
    named part runs whole top-level directories of tests/ that exist, no
    directory runs in two parts, and rocke_pytest_other runs tests/ with the
    library tests and every named part's directories ignored. A file a part
    must not collect is ignored by the part that would collect it. Every part
    keeps the suite's options, labels and time limit."""
    _, ctest_dir = configure_installed_ctest(
        tmp_path,
        f"-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR={provider_subdir}",
        *(
            f"-D_ROCKE_PLATFORM_PYTEST_IGNORE_{part}={path}"
            for part, path in ignored_files.items()
        ),
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
        selection_by_part[name.removeprefix("rocke_pytest_")] = command[
            3 : -len(suite_options)
        ]
        properties = {prop["name"]: prop["value"] for prop in part["properties"]}
        # One label string, as the single entry had: the generated CTest text
        # escapes the semicolons.
        assert properties["LABELS"] == ["ckc;python;quick"], name
        assert properties["TIMEOUT"] == 300, name

    other_paths, other_ignored = split_ignored_files(selection_by_part.pop("other"))
    assert other_paths == [tests_root]
    directories_by_part = {}
    ignored_by_part = {"other": other_ignored}
    for part, selection in selection_by_part.items():
        directories_by_part[part], ignored_by_part[part] = split_ignored_files(
            selection
        )
    named_directories = [
        path for paths in directories_by_part.values() for path in paths
    ]
    assert directories_by_part and all(directories_by_part.values())
    assert len(named_directories) == len(set(named_directories))
    for path in named_directories:
        parent, _, directory = path.rpartition("/")
        assert parent == tests_root, path
        assert directory != "library", path
        assert (PLATFORM / "tests" / directory).is_dir(), path
    expected_directory_ignores = [f"{tests_root}/library", *named_directories]
    assert other_ignored[: len(expected_directory_ignores)] == (
        expected_directory_ignores
    )
    ignored_by_part["other"] = other_ignored[len(expected_directory_ignores) :]

    # A file is ignored by the part that would collect it: a named part, for a
    # file in one of its directories, or rocke_pytest_other, for any other file.
    for part, ignored in ignored_by_part.items():
        for path in ignored:
            assert (PLATFORM / "tests" / path.removeprefix(f"{tests_root}/")).is_file()
            owners = [
                owner
                for owner, directories in directories_by_part.items()
                if any(path.startswith(f"{directory}/") for directory in directories)
            ]
            assert owners == ([] if part == "other" else [part]), (part, path)
    for part, path in ignored_files.items():
        assert f"{tests_root}/{path}" in ignored_by_part[part], (part, path)


@requires_cmake
def test_installed_platform_pytest_parts_keep_the_suite_tier_labels(tmp_path):
    """TheRock CI selects provider tests by tier label. The provider categories
    tier the parts by their name prefix, so every part, and a part a new CMake
    group would add, gets the same labels, among them the quick, standard and
    ffm-quick tiers that the single entry ran in."""
    _, ctest_dir = configure_installed_ctest(tmp_path)
    parts = [
        test["name"]
        for test in list_installed_ctests(ctest_dir)
        if test["name"].startswith("rocke_pytest_")
    ]
    assert parts, "no rocke_pytest_ parts are registered"
    future_part = "rocke_pytest_fp8_group2"
    labels = provider_category_labels([*parts, future_part], tmp_path)
    expected = labels.get(parts[0], set())
    assert {"quick", "standard", "ffm-quick"} <= expected, expected
    for part in [*parts, future_part]:
        assert labels.get(part) == expected, part


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
