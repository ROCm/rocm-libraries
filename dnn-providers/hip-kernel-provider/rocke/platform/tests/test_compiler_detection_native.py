# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Native IR lowering consumes owner evidence; it never discovers a compiler."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parents[1]


def detector_archive(build_root):
    override = os.environ.get("ROCKE_TEST_ENGINE_ARCHIVE")
    installed = Path(__file__).parent / "librocke_core.a"
    if override or installed.is_file():
        archive = Path(override) if override else installed
        assert archive.is_file(), archive
        return archive.resolve()
    if not (PLATFORM / "cpp").is_dir():
        pytest.fail("installed native boundary test requires its matching archive")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(PLATFORM),
            "-B",
            str(build_root),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DROCKE_BUILD_PYBIND=OFF",
            "-DROCKE_INSTALL_TESTS=OFF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_root), "--target", "rocke_core", "-j", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    for name in ("librocke_core.a", "rocke_core.lib", "Release/rocke_core.lib"):
        if (build_root / name).is_file():
            return build_root / name
    pytest.fail("source build did not produce an engine archive")


def build_library(root, text, name="libamd_comgr.so.3", extra=()):
    root.mkdir(parents=True, exist_ok=True)
    windows = sys.platform == "win32"
    if windows and name == "libamd_comgr.so.3":
        name = "amd_comgr.dll"
    output = root / name
    compiler = os.environ.get("CXX") or ("clang" if windows else "cc")
    if windows:
        text = (
            "#include <stddef.h>\n"
            "__declspec(dllexport) void amd_comgr_get_version(size_t*,size_t*);\n"
            "__declspec(dllexport) int amd_comgr_do_action(int,void*,void*,void*);\n"
            "__declspec(dllexport) void LLVMGetVersion(unsigned*,unsigned*,unsigned*);\n"
        ) + text
    subprocess.run(
        [
            compiler,
            "-shared",
            *([] if windows else ["-fPIC"]),
            "-x",
            "c",
            "-",
            "-x",
            "none",
            *extra,
            "-o",
            str(output),
        ],
        input=text,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.fixture(scope="module")
def native_lowerer(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("raw archive boundary fixtures use the Linux C++ linker")
    include = PLATFORM / "cpp/include"
    if not include.is_dir():
        pytest.skip("source public headers required for boundary executable")
    root = tmp_path_factory.mktemp("native-owner-boundary")
    archive = detector_archive(root / "engine")
    source = root / "main.cpp"
    source.write_text(
        '#include "rocke/ir.h"\n#include "rocke/lower_llvm.h"\n'
        "#include <cstdio>\n#include <cstdlib>\n"
        "int main(int argc,char**argv){\n"
        'rocke_ir_builder_t b{};if(rocke_ir_builder_init(&b,"native_owner_probe")!=ROCKE_OK)return 2;\n'
        "char* text=nullptr;char error[4096]{};\n"
        "auto flavor=static_cast<rocke_llvm_flavor_t>(argc>1?std::atoi(argv[1]):0);\n"
        'auto status=rocke_lower_kernel_to_llvm_ex(rocke_ir_builder_kernel(&b),flavor,"gfx950",&text,error,sizeof(error));\n'
        'std::printf("STATUS=%d\\nDIAG=%s\\nIR:\\n%s",int(status),error,text?text:"");\n'
        "std::free(text);rocke_ir_builder_free(&b);return 0;}\n"
    )
    binary = root / "lower"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-I",
            str(include),
            str(source),
            str(archive),
            "-ldl",
            "-pthread",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def clean_env():
    env = dict(os.environ)
    for key in (
        "ROCKE_LLVM_FLAVOR",
        "ROCKE_COMGR_LIB",
        "ROCM_PATH",
        "ROCM_HOME",
        "PYTHONPATH",
    ):
        env.pop(key, None)
    return env


def lower(binary, flavor, env):
    result = subprocess.run(
        [str(binary), str(flavor)], env=env, check=True, capture_output=True, text=True
    )
    header, ir = result.stdout.split("IR:\n", 1)
    return int(header.splitlines()[0].split("=", 1)[1]), header, ir


@pytest.mark.parametrize(
    "flavor,number", [("llvm20", 1), ("llvm22", 2), ("llvm23", 3), ("llvm24", 4)]
)
def test_native_explicit_owner_flavor_matches_python(native_lowerer, flavor, number):
    from rocke.core.ir import IRBuilder
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    expected = _lower_kernel_to_llvm_python(
        IRBuilder("native_owner_probe").kernel, arch="gfx950", llvm_flavor=flavor
    )
    status, _, text = lower(native_lowerer, number, clean_env())
    assert status == 0 and text == expected
    env = clean_env()
    env["ROCKE_LLVM_FLAVOR"] = flavor
    status, _, text = lower(native_lowerer, 0, env)
    assert status == 0 and text == expected


def test_native_auto_does_not_load_or_discover_comgr(native_lowerer, tmp_path):
    compiler = build_library(
        tmp_path / "compiler",
        "#include <unistd.h>\n__attribute__((constructor))static void trip(){_exit(86);}\n",
    )
    build = tmp_path / "build"
    deployed = build / "deployment/bin/lower"
    deployed.parent.mkdir(parents=True)
    shutil.copy2(native_lowerer, deployed)
    (build / "CMakeCache.txt").write_text("source\n")
    (build / "rocke_sdk_site.txt").write_text(
        "rocke-sdk-site-absolute-v1\n" + str(compiler.parent) + "\n"
    )
    env = clean_env()
    env["ROCKE_COMGR_LIB"] = str(compiler)
    status, diagnostic, _ = lower(deployed, 0, env)
    assert status != 0
    assert "does not select or load a compiler" in diagnostic
    status, _, _ = lower(deployed, 1, env)
    assert status == 0


@pytest.mark.parametrize(
    "selection", ["package", "provider", "environment", "relative-environment"]
)
def test_source_native_ctest_uses_effective_build_comgr(tmp_path, selection):
    if not (PLATFORM / "CMakeLists.txt").is_file():
        pytest.skip("source CMake needed for owner forwarding test")
    library = build_library(
        tmp_path / "compiler",
        (
            "#include <stddef.h>\nvoid amd_comgr_get_version(size_t*a,size_t*b){*a=3;*b=4;}\n"
            "int amd_comgr_do_action(int a,void*b,void*c,void*d){return 0;}\n"
            "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){*a=23;*b=0;*c=1;}\n"
        ),
    )
    package = tmp_path / "package"
    package.mkdir()
    (package / "amd_comgr-config.cmake").write_text(
        "add_library(amd_comgr SHARED IMPORTED)\n"
        f'set_target_properties(amd_comgr PROPERTIES IMPORTED_LOCATION "{library.as_posix()}")\n'
    )
    build = tmp_path / "build"
    env = clean_env()
    command = [
        "cmake",
        "-S",
        str(PLATFORM),
        "-B",
        str(build),
        "-DHIPKERNELPROVIDER_ENABLE_ROCKE=ON",
        "-DHIPKERNELPROVIDER_ENABLE_TESTS=ON",
        "-DBUILD_TESTING=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DROCKE_INSTALL_TESTS=OFF",
        "-DROCKE_BUILD_PYBIND=OFF",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-Damd_comgr_DIR={package}",
    ]
    if selection == "provider":
        command.append(f"-DHIPKERNELPROVIDER_ROCKE_COMGR_LIB={library}")
    elif selection in ("environment", "relative-environment"):
        env["ROCKE_COMGR_LIB"] = (
            os.path.relpath(library)
            if selection == "relative-environment"
            else str(library)
        )
    configured = subprocess.run(command, env=env, capture_output=True, text=True)
    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert not list(build.rglob("rocke_sdk_*.txt"))
    inventory = json.loads(
        subprocess.run(
            ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["tests"]
    assert any(test["name"] == "rocke_compiler_discovery_source" for test in inventory)
    compiled = subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "--target",
            "rocke_ir_serialize_roundtrip",
            "-j",
            "2",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    env.pop("ROCKE_COMGR_LIB", None)
    ctest = [
        "ctest",
        "--build-config",
        "Release",
        "--test-dir",
        str(build),
        "-R",
        "^rocke_ir_serialize_roundtrip$",
        "--output-on-failure",
        "--no-tests=error",
    ]
    result = subprocess.run(ctest, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    env["ROCKE_COMGR_LIB"] = str(tmp_path / "missing.so")
    result = subprocess.run(ctest, env=env, capture_output=True, text=True)
    assert result.returncode != 0
