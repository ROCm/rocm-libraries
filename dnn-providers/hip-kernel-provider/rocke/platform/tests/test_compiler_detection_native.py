# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise the native and Python queries against loadable compiler fixtures."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PLATFORM = Path(__file__).resolve().parents[1]

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
    output = tmp_path_factory.mktemp("native-detector") / "detector.so"
    wrapper = output.with_suffix(".cpp")
    wrapper.write_text(
        '#include "compiler_version.h"\n'
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
            "-pthread",
            str(PLATFORM / "cpp/core/lower_llvm/compiler_version.cpp"),
            str(wrapper),
            "-I",
            str(PLATFORM / "cpp/core/lower_llvm"),
            "-I",
            str(PLATFORM / "cpp/include"),
            "-ldl",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def build_library(root, text, name="libamd_comgr.so", extra=()):
    root.mkdir(parents=True, exist_ok=True)
    source = root / (name + ".c")
    source.write_text(text)
    output = root / name
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(source), *extra, "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def run_probe(native, library, **overrides):
    env = dict(os.environ)
    for name in ("ROCKE_LLVM_FLAVOR", "ROCM_PATH", "ROCM_HOME"):
        env.pop(name, None)
    env.update(ROCKE_COMGR_LIB=str(library), PYTHONPATH=str(PLATFORM / "python"))
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
