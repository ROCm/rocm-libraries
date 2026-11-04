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
@pytest.mark.parametrize(
    "suffix_override", [None, ".test-bin"], ids=["host-suffix", "explicit-suffix"]
)
def test_installed_ctest_entries_get_scoped_environment_contracts(
    tmp_path, provider_subdir, suffix_override
):
    build = tmp_path / "build"
    packs = tmp_path / "pack-headers"
    packs.mkdir()
    for header in (
        "Gfx950AttentionDenseGeometry.hpp",
        "Gfx950AttentionUnified3DGeometry.hpp",
    ):
        (packs / header).write_text("// header fixture\n")
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
            f"-DCMAKE_PROJECT_INCLUDE={project_include}",
            f"-DROCKE_HKP_PACKS_DIR={packs}",
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
        for test in tests:
            assert ("--rocke-require-hkp-pack-headers" in test.get("command", ())) == (
                test["name"] == "rocke_library_attention_cpu_pytest"
            )
        install_script = (build / "cmake_install.cmake").read_text()
        assert "/tests/library/tests/hkp_packs" in install_script
        assert "Gfx950AttentionUnified3DGeometry.hpp" in install_script
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


def _relocation_config(build: Path) -> str:
    """Honor the runner's selection or retain a manual build's default config."""
    config = os.environ.get("ROCKE_LINK_TEST_CONFIG")
    if config is not None:
        if not config:
            raise ValueError(
                "ROCKE_LINK_TEST_CONFIG must name a nonempty configuration"
            )
        return config
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith("CMAKE_BUILD_TYPE:") and "=" in line:
            build_type = line.split("=", 1)[1]
            if build_type:
                return build_type
    return "Release"


@pytest.mark.parametrize("cache_type", ["STRING", "UNINITIALIZED"])
@pytest.mark.parametrize(
    ("build_type", "explicit_config", "expected"),
    [("Debug", None, "Debug"), ("Debug", "Release", "Release"), ("", None, "Release")],
)
def test_relocation_config_preserves_manual_build_default(
    monkeypatch, tmp_path, cache_type, build_type, explicit_config, expected
):
    (tmp_path / "CMakeCache.txt").write_text(
        f"CMAKE_BUILD_TYPE:{cache_type}={build_type}\n"
    )
    monkeypatch.delenv("ROCKE_LINK_TEST_CONFIG", raising=False)
    if explicit_config is not None:
        monkeypatch.setenv("ROCKE_LINK_TEST_CONFIG", explicit_config)
    assert _relocation_config(tmp_path) == expected


def test_relocation_config_rejects_empty_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("ROCKE_LINK_TEST_CONFIG", "")
    with pytest.raises(ValueError, match="ROCKE_LINK_TEST_CONFIG must name a nonempty"):
        _relocation_config(tmp_path)


@pytest.mark.parametrize("consumer", ["cpp_lowerer", "c_mma"])
def test_relocated_cmake_consumer_links_lowerer(tmp_path, consumer):
    """Compile and run C/C++ consumers of a freshly built, relocated SDK."""
    build = os.environ.get("ROCKE_LINK_TEST_BUILD_DIR")
    if not build:
        pytest.skip("set ROCKE_LINK_TEST_BUILD_DIR to a fresh engine build")
    config = _relocation_config(Path(build))
    prefix = tmp_path / "install"
    subprocess.run(
        ["cmake", "--install", build, "--config", config, "--prefix", str(prefix)],
        check=True,
    )
    relocated = tmp_path / "relocated"
    prefix.rename(relocated)
    source = tmp_path / "consumer"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(consumer LANGUAGES C CXX)\n"
        "find_package(ckc CONFIG REQUIRED)\n"
        "if(NOT TARGET Threads::Threads)\n"
        '  message(FATAL_ERROR "missing transitive thread dependency")\n'
        "endif()\n"
        f"add_executable(consumer main.{'c' if consumer == 'c_mma' else 'cpp'})\n"
        "set_target_properties(consumer PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF LINKER_LANGUAGE CXX)\n"
        "target_link_libraries(consumer PRIVATE ckc::rocke_core)\n"
        'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/consumer-path-$<CONFIG>.txt" '
        'CONTENT "$<TARGET_FILE:consumer>")\n'
    )
    # Taking the API address forces archive extraction without needing a GPU or
    # a valid kernel. The executable must resolve the detector dependencies.
    if consumer == "cpp_lowerer":
        (source / "main.cpp").write_text(
            '#include "rocke/lower_llvm.h"\n'
            "auto volatile lower = &rocke_lower_kernel_to_llvm_ex;\n"
            "int main() { return lower == nullptr; }\n"
        )
    else:
        (source / "main.c").write_text(
            '#include "rocke/arch_target.h"\n'
            '#include "rocke/abi.h"\n'
            '#include "rocke/instance_conv_direct_fwd_wmma.h"\n'
            "#include <stddef.h>\n#include <string.h>\n"
            "int main(void) {\n"
            "  if (ROCKE_ABI_VERSION != 9 || rocke_abi_version() != ROCKE_ABI_VERSION) return 6;\n"
            "  if (rocke_recipe_abi_level() != 3) return 7;\n"
            "  if (sizeof(rocke_mma_op_t) != 216 || sizeof(rocke_direct_fwd_wmma_spec_t) != 152) return 8;\n"
            "  rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();\n"
            "  p.groups = p.cpg = p.kpg = 1 << 20;\n"
            "  if (rocke_direct_conv_problem_total_c(&p) != (1LL << 40)) return 9;\n"
            "  if (rocke_direct_conv_problem_total_k(&p) != (1LL << 40)) return 10;\n"
            '  const char* id = "wmma_gfx1250_bf16f32_16x16x32_bf16";\n'
            "  const char* c = rocke_arch_mma_op_id_c_dtype(id);\n"
            "  const char* d = rocke_arch_mma_op_id_dst_dtype(id);\n"
            '  if (!c || !d || strcmp(c, "fp32") || strcmp(d, "bf16")) return 1;\n'
            "  if (rocke_arch_mma_op_id_c_dtype(NULL) != NULL) return 2;\n"
            '  if (rocke_arch_mma_op_id_c_dtype("unknown") != NULL) return 3;\n'
            "  if (rocke_arch_mma_c_frag_len(id) != 8 || rocke_arch_mma_dst_frag_len(id) != 8) return 4;\n"
            '  const rocke_arch_target_t* t = rocke_arch_target_from_gfx("gfx1250");\n'
            '  const rocke_mma_op_t* op = rocke_mma_catalog_op_for_shape(&t->mma, "wmma", "bf16", "bf16", c, 16, 16, 32, NULL, d);\n'
            "  if (!op || strcmp(op->op_id, id)) return 5;\n"
            "  return 0;\n}\n"
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
            f"-DCMAKE_BUILD_TYPE={config}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(output), "--config", config], check=True)
    executable = (output / f"consumer-path-{config}.txt").read_text()
    subprocess.run([executable], check=True)


@pytest.mark.parametrize("generator", ["Ninja", "Ninja Multi-Config"])
@pytest.mark.parametrize("config", ["Release", "Debug"])
@pytest.mark.parametrize("explicit_config", [False, True])
def test_relocation_consumer_uses_selected_cmake_configuration(
    monkeypatch, tmp_path, generator, config, explicit_config
):
    """Isolate configuration handling with a real, minimal exported SDK."""
    if not all(shutil.which(tool) for tool in ("cmake", "ninja")):
        pytest.skip("CMake and Ninja required for relocation configuration regression")
    source = tmp_path / "sdk"
    source.mkdir()
    (source / "rocke").mkdir()
    (source / "rocke" / "lower_llvm.h").write_text(
        'extern "C" void rocke_lower_kernel_to_llvm_ex();\n'
    )
    (source / "engine.cpp").write_text(
        'extern "C" void rocke_lower_kernel_to_llvm_ex() {}\n'
    )
    (source / "ckcConfig.cmake").write_text(
        "include(CMakeFindDependencyMacro)\n"
        "find_dependency(Threads)\n"
        'include("${CMAKE_CURRENT_LIST_DIR}/ckcTargets.cmake")\n'
    )
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(relocation_fixture LANGUAGES CXX)\n"
        "find_package(Threads REQUIRED)\n"
        "add_library(rocke_core STATIC engine.cpp)\n"
        "target_link_libraries(rocke_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})\n"
        "target_include_directories(rocke_core PUBLIC\n"
        "  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include>)\n"
        "install(TARGETS rocke_core EXPORT ckcTargets ARCHIVE DESTINATION lib)\n"
        "install(DIRECTORY rocke DESTINATION include)\n"
        "install(EXPORT ckcTargets NAMESPACE ckc:: DESTINATION lib/cmake/ckc)\n"
        "install(FILES ckcConfig.cmake DESTINATION lib/cmake/ckc)\n"
    )
    build = tmp_path / "sdk-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            generator,
            f"-DCMAKE_BUILD_TYPE={config}",
        ],
        check=True,
    )
    # Build only the selected engine configuration so a wrong install cannot
    # accidentally succeed using artifacts left over from another lane.
    subprocess.run(["cmake", "--build", str(build), "--config", config], check=True)
    monkeypatch.setenv("CMAKE_GENERATOR", generator)
    monkeypatch.setenv("ROCKE_LINK_TEST_BUILD_DIR", str(build))
    monkeypatch.delenv("ROCKE_LINK_TEST_CONFIG", raising=False)
    if explicit_config:
        monkeypatch.setenv("ROCKE_LINK_TEST_CONFIG", config)
    consumer_root = tmp_path / "relocation"
    consumer_root.mkdir()
    test_relocated_cmake_consumer_links_lowerer(consumer_root, "cpp_lowerer")
