# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise the production archive's detector against loadable compiler fixtures."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parents[1]


def detector_archive(build_root):
    """Use the test artifact's archive, an explicit build, or a fresh source build."""
    override = os.environ.get("ROCKE_TEST_ENGINE_ARCHIVE")
    archive = Path(override) if override else Path(__file__).parent / "librocke_core.a"
    if override or archive.exists():
        if not archive.is_file():
            pytest.fail(f"native detector archive not found: {archive}")
        return archive.resolve()
    if not (PLATFORM / "cpp").is_dir():
        pytest.fail(f"installed native detector test requires {archive}")
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("source native detector tests require CMake to build the archive")
    subprocess.run(
        [
            cmake,
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
        [cmake, "--build", str(build_root), "--target", "rocke_core", "-j", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    archive = build_root / "librocke_core.a"
    if not archive.is_file():
        pytest.fail(f"native detector build did not produce {archive}")
    return archive


PROBE = r"""
import ctypes, json, os, sys
from dataclasses import asdict
from rocke.runtime.comgr import loaded_compiler_info

class Info(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint) for n in ('major', 'minor', 'patch')] + [
        (n, ctypes.c_char_p) for n in ('source', 'requested_comgr', 'comgr_path', 'query_library_path')]

native = ctypes.CDLL(sys.argv[1])
native.rocke_loaded_compiler_info.argtypes = []
native.rocke_loaded_compiler_info.restype = ctypes.POINTER(Info)
ptr = native.rocke_loaded_compiler_info()
assert ptr
info = ptr.contents
result = {'llvm_version': [info.major, info.minor, info.patch] if info.major else None}
for name in ('source', 'requested_comgr', 'comgr_path', 'query_library_path'):
    value = getattr(info, name)
    result[name] = value.decode() if value else None
print(json.dumps({'native': result, 'python': asdict(loaded_compiler_info())}))
"""


@pytest.fixture(scope="module")
def native_detector(tmp_path_factory):
    compiler = shutil.which("c++")
    if sys.platform != "linux" or not compiler or not shutil.which("cc"):
        pytest.skip("ELF loader fixtures require Linux and C/C++ compilers")
    root = tmp_path_factory.mktemp("native-detector")
    archive = detector_archive(root / "engine")
    output = root / "detector.so"
    # The return type is opaque here; ctypes below mirrors the private record.
    # Link the production implementation without installing its private headers.
    wrapper = (
        "namespace ckc { struct CompilerInfo; "
        "const CompilerInfo* candidate_compiler_info(); }\n"
        "static bool block_loads = false;\n"
        "static unsigned load_calls = 0;\n"
        'extern "C" void* __real_dlopen(const char*, int);\n'
        'extern "C" void* __wrap_dlopen(const char* path, int mode) {\n'
        "    ++load_calls;\n"
        "    return block_loads ? nullptr : __real_dlopen(path, mode);\n"
        "}\n"
        'extern "C" void fixture_block_loads(int block) { block_loads = block; }\n'
        'extern "C" unsigned fixture_load_calls() { return load_calls; }\n'
        'extern "C" const ckc::CompilerInfo* rocke_loaded_compiler_info() {\n'
        "    return ckc::candidate_compiler_info();\n"
        "}\n"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-shared",
            "-fPIC",
            "-Wl,--wrap=dlopen",
            "-x",
            "c++",
            "-",
            "-x",
            "none",
            str(archive),
            "-ldl",
            "-pthread",
            "-o",
            str(output),
        ],
        input=wrapper,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def build_library(root, text, name="libamd_comgr.so", extra=()):
    root.mkdir(parents=True, exist_ok=True)
    output = root / name
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
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


def run_probe(native, library, **overrides):
    env = dict(os.environ)
    for name in ("ROCKE_LLVM_FLAVOR", "ROCM_PATH", "ROCM_HOME"):
        env.pop(name, None)
    env.update(
        ROCKE_COMGR_LIB=str(library),
        PYTHONPATH=os.pathsep.join(str(path) for path in sys.path if path),
    )
    env.update(overrides)
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(native)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    assert info["native"] == info["python"]
    return info["native"]


def test_dependency_query_reports_the_library_that_supplied_version(
    native_detector, tmp_path
):
    root = tmp_path / "rocm-99.0" / "lib"
    dependency = build_library(
        root,
        "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){*a=23;*b=2;*c=1;}\n"
        "void fixture_dependency(void){}\n",
        "libfixtureLLVM.so",
    )
    library = build_library(
        root,
        "#include <stddef.h>\n"
        "extern void fixture_dependency(void);\n"
        "void amd_comgr_get_version(size_t*a,size_t*b){fixture_dependency();*a=3;*b=3;}\n",
        extra=(str(dependency), "-Wl,-rpath,$ORIGIN"),
    )
    metadata = root.parent / ".info"
    metadata.mkdir()
    (metadata / "version").write_text("99.0.0\n")
    info = run_probe(native_detector, library)
    assert info["llvm_version"] == [23, 2, 1]
    assert info["source"] == "LLVMGetVersion"
    assert info["comgr_path"] == str(library)
    assert info["query_library_path"] == str(dependency)


def test_unloadable_override_uses_first_loadable_library(native_detector, tmp_path):
    broken = tmp_path / "broken.so"
    broken.write_text("not a shared library")
    root = tmp_path / "versionless-install"
    library = build_library(
        root / "lib",
        "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){*a=21;*b=0;*c=0;}\n",
    )
    info = run_probe(native_detector, broken, ROCM_PATH=str(root))
    assert info["llvm_version"] == [21, 0, 0]
    assert info["requested_comgr"] == str(library)


def test_loaded_unqueryable_library_does_not_fall_through(native_detector, tmp_path):
    library = build_library(tmp_path, "void amd_comgr_get_version(void){}\n")
    info = run_probe(native_detector, library)
    assert info["llvm_version"] is None
    assert info["source"] == "unavailable"
    assert info["requested_comgr"] == str(library)


PREPROCESSOR = r"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>
typedef struct { uint64_t value; } handle;
static const char output[] = "# 1 \"version.cl\"\nROCKE_LLVM_VERSION 20 0 7\n";
void amd_comgr_get_version(size_t*a,size_t*b){*a=3;*b=3;}
int amd_comgr_create_data_set(handle*h){h->value=1;return 0;}
int amd_comgr_destroy_data_set(handle h){return 0;}
int amd_comgr_create_data(int kind,handle*h){h->value=2;return 0;}
int amd_comgr_release_data(handle h){return 0;}
int amd_comgr_set_data(handle h,size_t n,const char*p){return 0;}
int amd_comgr_set_data_name(handle h,const char*p){return 0;}
int amd_comgr_data_set_add(handle a,handle b){return 0;}
int amd_comgr_create_action_info(handle*h){h->value=3;return 0;}
int amd_comgr_destroy_action_info(handle h){return 0;}
int amd_comgr_get_isa_name(size_t i,const char**p){*p="amdgcn-amd-amdhsa--gfx900";return 0;}
int amd_comgr_action_info_set_isa_name(handle h,const char*p){return 0;}
int amd_comgr_action_info_set_language(handle h,int language){return language==1?0:1;}
int amd_comgr_do_action(int action,handle info,handle in,handle out){return action==0?0:1;}
int amd_comgr_action_data_get_data(handle set,int kind,size_t i,handle*h){h->value=4;return 0;}
int amd_comgr_get_data(handle h,size_t*n,char*p){if(p)memcpy(p,output,sizeof(output));*n=sizeof(output);return 0;}
"""


def test_preprocessing_fallback_and_provenance_agree(native_detector, tmp_path):
    library = build_library(tmp_path, PREPROCESSOR)
    info = run_probe(native_detector, library)
    assert info["llvm_version"] == [20, 0, 7]
    assert info["source"] == "COMGR preprocessing"
    assert info["query_library_path"] == info["comgr_path"] == str(library)


@pytest.mark.parametrize("queryable", [True, False])
def test_native_recovers_then_retains_loaded_candidate(
    native_detector, tmp_path, queryable
):
    text = "unsigned fixture_queries = 0;\n"
    if queryable:
        text += (
            "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c)"
            "{++fixture_queries;*a=23;*b=0;*c=0;}\n"
        )
    library = build_library(tmp_path, text)
    script = r"""
import ctypes, os, sys
class Info(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint) for n in ('major', 'minor', 'patch')] + [
        (n, ctypes.c_char_p) for n in
        ('source', 'requested_comgr', 'comgr_path', 'query_library_path')]
native = ctypes.CDLL(sys.argv[1])
native.rocke_loaded_compiler_info.restype = ctypes.POINTER(Info)
native.fixture_block_loads.argtypes = [ctypes.c_int]
native.fixture_load_calls.restype = ctypes.c_uint
native.fixture_block_loads(1)
assert not native.rocke_loaded_compiler_info()
failed_calls = native.fixture_load_calls()
assert failed_calls > 0
native.fixture_block_loads(0)
first = native.rocke_loaded_compiler_info()
assert first
assert first.contents.major == (23 if sys.argv[3] == 'True' else 0)
assert first.contents.requested_comgr.decode() == sys.argv[2]
successful_calls = native.fixture_load_calls()
assert successful_calls > failed_calls
os.environ['ROCKE_COMGR_LIB'] = 'replacement-that-must-not-be-loaded.so'
for _ in range(3):
    current = native.rocke_loaded_compiler_info()
    assert ctypes.addressof(current.contents) == ctypes.addressof(first.contents)
assert native.fixture_load_calls() == successful_calls
fixture = ctypes.CDLL(sys.argv[2])
assert ctypes.c_uint.in_dll(fixture, 'fixture_queries').value == (
    1 if sys.argv[3] == 'True' else 0)
"""
    env = dict(os.environ, ROCKE_COMGR_LIB=str(library))
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(native_detector),
            str(library),
            str(queryable),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


REJECTION_PROBE = r"""
import os
from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import serialize
from rocke.core import lower_llvm
from rocke.runtime import comgr
import rocke_engine

kernel = IRBuilder("compiler_requirement").kernel
serialized = serialize(kernel)
for lower in (
    lambda: lower_llvm.lower_kernel_to_llvm(kernel),
    lambda: rocke_engine.lower_serialized_ir(serialized, arch="gfx950"),
):
    try:
        lower()
    except RuntimeError as error:
        assert "Cannot determine LLVM flavor" in str(error), str(error)
    else:
        raise AssertionError("AUTO emitted IR without compiler evidence")

for flavor in lower_llvm.LLVM_FLAVORS:
    py = lower_llvm.lower_kernel_to_llvm(kernel, llvm_flavor=flavor)
    native = rocke_engine.lower_serialized_ir(serialized, arch="gfx950", flavor=flavor)
    assert py == native

os.environ["ROCKE_LLVM_FLAVOR"] = "llvm20"
assert lower_llvm.lower_kernel_to_llvm(kernel) == rocke_engine.lower_serialized_ir(
    serialized, arch="gfx950")
try:
    comgr.build_hsaco_from_llvm_ir(py)
except comgr.ComgrError as error:
    assert "Cannot determine LLVM flavor" in str(error), str(error)
else:
    raise AssertionError("explicit emission bypassed compiler evidence requirement")
"""


def test_unqueryable_compiler_rejects_auto_but_allows_explicit_emission(tmp_path):
    pytest.importorskip("rocke_engine")
    if sys.platform != "linux" or not shutil.which("cc"):
        pytest.skip("ELF loader fixture requires Linux and a C compiler")
    library = build_library(tmp_path, "void amd_comgr_get_version(void){}\n")
    env = dict(os.environ, ROCKE_COMGR_LIB=str(library), ROCKE_BACKEND="python")
    env.pop("ROCKE_LLVM_FLAVOR", None)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", REJECTION_PROBE],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
