# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise the production archive's detector against loadable compiler fixtures."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    result[name] = value.decode() if value is not None else None
print(json.dumps({'native': result, 'python': asdict(loaded_compiler_info())}))
"""


@pytest.fixture(scope="module")
def native_detector(tmp_path_factory):
    compiler = shutil.which("c++")
    if sys.platform != "linux" or not compiler or not shutil.which("cc"):
        pytest.skip("ELF loader fixtures require Linux and C/C++ compilers")
    root = tmp_path_factory.mktemp("native-detector")
    archive = detector_archive(root / "engine")
    shutil.copyfile(archive, root / "librocke_core.a")
    output = root / "detector.so"
    # The return type is opaque here; ctypes below mirrors the private record.
    # Link the production implementation without installing its private headers.
    wrapper = (
        "namespace ckc { struct CompilerInfo; "
        "const CompilerInfo* candidate_compiler_info(); }\n"
        "static bool block_loads = false;\n"
        "static unsigned load_calls = 0;\n"
        "static unsigned invalid_path_calls = 0;\n"
        "static const char* fixture_root = nullptr;\n"
        "#include <cstring>\n"
        'extern "C" void* __real_dlopen(const char*, int);\n'
        'extern "C" void* __wrap_dlopen(const char* path, int mode) {\n'
        "    ++load_calls;\n"
        "    if(!*path || (*path != '/' && strchr(path, '/')))\n"
        "        ++invalid_path_calls;\n"
        "    if(fixture_root && *path == '/' &&\n"
        "       strncmp(path, fixture_root, strlen(fixture_root)) != 0)\n"
        "        return nullptr;\n"
        "    return block_loads ? nullptr : __real_dlopen(path, mode);\n"
        "}\n"
        'extern "C" void fixture_block_loads(int block) { block_loads = block; }\n'
        'extern "C" unsigned fixture_load_calls() { return load_calls; }\n'
        'extern "C" unsigned fixture_invalid_path_calls() { return invalid_path_calls; }\n'
        'extern "C" void fixture_limit_roots(const char* root) { fixture_root = root; }\n'
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


UNAVAILABLE_PROBE = r"""
import ctypes, os, sys
from rocke.runtime import comgr
native = ctypes.CDLL(sys.argv[1])
native.rocke_loaded_compiler_info.restype = ctypes.c_void_p
assert not native.rocke_loaded_compiler_info()
assert comgr.loaded_compiler_info() is None
try:
    comgr._resolve_lib()
except comgr.ComgrError:
    pass
else:
    raise AssertionError('unavailable selected compiler fell through')
"""


def sdk_fixture(site, suffix="", text=None):
    core = site / ("_rocm_sdk_core" + suffix)
    library = build_library(
        core / "lib",
        text
        or "void amd_comgr_get_version(void){}\n"
        "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){*a=23;*b=0;*c=1;}\n",
        "libamd_comgr.so.3",
    )
    (core / "__init__.py").write_text("raise AssertionError('SDK imported')\n")
    metadata = site / "rocm_sdk_core-7.9.0rc1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rocm-sdk-core\nVersion: 7.9.0rc1\n"
    )
    (metadata / "RECORD").write_text(
        f"{library.relative_to(site).as_posix()},,\n" f"{core.name}/__init__.py,,\n"
    )
    return library


def automatic_env(site):
    env = dict(os.environ)
    for name in ("ROCKE_LLVM_FLAVOR", "ROCKE_COMGR_LIB", "ROCM_PATH", "ROCM_HOME"):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site)] + [str(path) for path in sys.path if path]
    )
    return env


def test_unloadable_override_does_not_fall_through_to_sdk(native_detector, tmp_path):
    broken = tmp_path / "broken.so"
    broken.write_text("not a shared library")
    site = tmp_path / "site-packages"
    sdk_fixture(site)
    native = site / native_detector.name
    shutil.copyfile(native_detector, native)
    env = automatic_env(site)
    env["ROCKE_COMGR_LIB"] = str(broken)
    result = subprocess.run(
        [sys.executable, "-c", UNAVAILABLE_PROBE, str(native)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("variable", ["ROCKE_COMGR_LIB", "ROCM_PATH", "ROCM_HOME"])
def test_empty_discovery_values_match_unset(native_detector, tmp_path, variable):
    site = tmp_path / "site-packages"
    library = sdk_fixture(site, "_7_9_nonce")
    native = site / native_detector.name
    shutil.copyfile(native_detector, native)
    env = automatic_env(site)
    results = []
    for value in (None, ""):
        child_env = dict(env)
        if value is not None:
            child_env[variable] = value
        result = subprocess.run(
            [sys.executable, "-c", PROBE, str(native)],
            env=child_env,
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout)
        assert info["native"] == info["python"]
        assert info["native"]["llvm_version"] == [23, 0, 1]
        assert info["native"]["source"] == "LLVMGetVersion"
        assert info["native"]["comgr_path"] == str(library)
        assert info["native"]["query_library_path"] == str(library)
        assert info["native"]["requested_comgr"] == str(library)
        results.append(info)
    assert results[0] == results[1]


@pytest.mark.parametrize("suffix", ["", "_7_9_nonce"])
def test_sdk_selected_despite_competing_system_compiler(
    native_detector, tmp_path, suffix
):
    site = tmp_path / "site-packages"
    library = sdk_fixture(site, suffix)
    native = site / native_detector.name
    shutil.copyfile(native_detector, native)
    system = build_library(
        tmp_path / "system" / "lib",
        "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){*a=20;*b=0;*c=0;}\n",
    )
    env = automatic_env(site)
    env["ROCM_PATH"] = str(system.parent.parent)
    env["ROCM_HOME"] = str(system.parent.parent)
    env["LD_LIBRARY_PATH"] = str(system.parent)
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(native)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    assert info["native"] == info["python"]
    assert info["native"]["llvm_version"] == [23, 0, 1]
    assert info["native"]["requested_comgr"] == str(library)


@pytest.mark.parametrize(
    "kind", ["foreign", "deleted", "replaced", "multiple", "symlink", "hardlink"]
)
def test_sdk_selection_checks_mapped_file_identity(native_detector, tmp_path, kind):
    site = tmp_path / "site-packages"
    library = sdk_fixture(site, "_7_9_nonce")
    native = site / native_detector.name
    shutil.copyfile(native_detector, native)
    alias = kind in ("symlink", "hardlink")
    if alias:
        mapped = tmp_path / "alias" / library.name
        mapped.parent.mkdir()
        if kind == "symlink":
            mapped.symlink_to(library)
        else:
            os.link(library, mapped)
    else:
        mapped = build_library(
            tmp_path / "foreign",
            "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){*a=20;*b=0;*c=0;}\n",
            library.name,
        )
    script = "import ctypes,sys; retained=ctypes.CDLL(sys.argv[2])\n"
    if kind in ("deleted", "replaced"):
        script += "import os; os.unlink(sys.argv[2])\n"
        if kind == "replaced":
            # Recreating the same pathname must not identify the mapped old
            # inode as the selected SDK compiler, including NFS silly-renames.
            script += f"os.link({str(library)!r}, sys.argv[2])\n"
            script += "os.environ['ROCKE_COMGR_LIB']=sys.argv[2]\n"
    elif kind == "multiple":
        script += f"sdk_retained=ctypes.CDLL({str(library)!r})\n"
    if alias:
        script += PROBE
    else:
        script += UNAVAILABLE_PROBE
        script += (
            "native.fixture_load_calls.restype=ctypes.c_uint\n"
            "assert native.fixture_load_calls()==0\n"
        )
    result = subprocess.run(
        [sys.executable, "-c", script, str(native), str(mapped)],
        env=automatic_env(site),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    if alias:
        info = json.loads(result.stdout)
        assert info["native"] == info["python"]
        assert info["native"]["requested_comgr"] == str(library)


def test_missing_sdk_payload_does_not_use_system_compiler(native_detector, tmp_path):
    site = tmp_path / "site-packages"
    library = sdk_fixture(site)
    library.unlink()
    native = site / native_detector.name
    shutil.copyfile(native_detector, native)
    system = build_library(
        tmp_path / "system" / "lib",
        "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){*a=20;*b=0;*c=0;}\n",
    )
    env = automatic_env(site)
    env["ROCM_PATH"] = str(system.parent.parent)
    env["LD_LIBRARY_PATH"] = str(system.parent)
    result = subprocess.run(
        [sys.executable, "-c", UNAVAILABLE_PROBE, str(native)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_relocated_shared_engine_discovers_sdk_without_build_path(
    native_detector, tmp_path
):
    original = tmp_path / "original"
    site = original / "lib" / "python3.12" / "site-packages"
    library = sdk_fixture(site, "_7_9_nonce")
    native = original / "lib" / native_detector.name
    shutil.copyfile(native_detector, native)
    relocated = tmp_path / "relocated"
    original.rename(relocated)
    site = relocated / site.relative_to(original)
    library = relocated / library.relative_to(original)
    native = relocated / native.relative_to(original)
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(native)],
        env=automatic_env(site),
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    assert info["native"] == info["python"]
    assert info["native"]["requested_comgr"] == str(library)


def test_build_sidecar_tracks_selected_sdk_site_after_parent_relocation(
    native_detector, tmp_path, monkeypatch
):
    generator = PLATFORM / "tools" / "gen_sdk_comgr_location.py"
    if not generator.is_file():
        pytest.skip(
            "configure-time generator is not installed with native test artifacts"
        )
    original = tmp_path / "original"
    site = original / "env" / "site-packages"
    library = sdk_fixture(site, "_7_9_nonce")
    build = original / "build"
    build.mkdir()
    native = build / native_detector.name
    shutil.copyfile(native_detector, native)
    spec = importlib.util.spec_from_file_location("sdk_location_generator", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda relative: site),
    )
    module.write_location(build)
    descriptor = (build / "rocke_sdk_site.txt").read_text().splitlines()
    assert descriptor == [
        "rocke-sdk-site-v1",
        os.path.join("..", "env", "site-packages"),
    ]
    assert str(original) not in (build / "rocke_sdk_site.txt").read_text()
    relocated = tmp_path / "relocated"
    original.rename(relocated)
    site = relocated / site.relative_to(original)
    native = relocated / native.relative_to(original)
    library = relocated / library.relative_to(original)
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(native)],
        env=automatic_env(site),
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    assert info["native"] == info["python"]
    assert info["native"]["requested_comgr"] == str(library)


def test_cmake_build_sidecar_discovers_user_site_sdk(native_detector, tmp_path):
    if not (PLATFORM / "CMakeLists.txt").is_file():
        pytest.skip("source CMake project is not installed with native test artifacts")
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("configure-time SDK discovery requires CMake")

    # A normal virtualenv suppresses the user site. Enable it for this interpreter
    # so the SDK is visible to Python, but would disappear under isolated mode.
    interpreter_root = tmp_path / "interpreter"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            "--system-site-packages",
            str(interpreter_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    python = interpreter_root / "bin" / "python"
    env = dict(os.environ, PYTHONUSERBASE=str(tmp_path / "user-base"))
    for name in (
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "ROCKE_COMGR_LIB",
        "ROCKE_LLVM_FLAVOR",
    ):
        env.pop(name, None)
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import json, site; "
            "print(json.dumps([site.ENABLE_USER_SITE, site.getusersitepackages()]))",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    enabled, user_site = json.loads(result.stdout)
    assert enabled is True
    site = Path(user_site)
    library = sdk_fixture(site, "_7_9_nonce")
    build = tmp_path / "build"
    # Exercise the production CMake call site: importing write_location directly
    # would miss a regression that reinstates isolated Python invocation there.
    subprocess.run(
        [
            cmake,
            "-S",
            str(PLATFORM),
            "-B",
            str(build),
            f"-DPython3_EXECUTABLE={python}",
            "-DBUILD_TESTING=OFF",
            "-DROCKE_BUILD_PYBIND=OFF",
            "-DROCKE_INSTALL_TESTS=OFF",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    marker, relative = (build / "rocke_sdk_site.txt").read_text().splitlines()
    assert marker == "rocke-sdk-site-v1"
    assert not Path(relative).is_absolute()
    assert (build / relative).resolve() == site.resolve()
    native = build / native_detector.name
    shutil.copyfile(native_detector, native)
    env["PYTHONPATH"] = str(PLATFORM / "python")
    result = subprocess.run(
        [str(python), "-c", PROBE, str(native)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    assert info["native"] == info["python"]
    assert info["native"]["llvm_version"] == [23, 0, 1]
    assert info["native"]["requested_comgr"] == str(library)


def test_relocated_static_archive_consumer_discovers_sdk_without_python(
    native_detector, tmp_path
):
    original = tmp_path / "original"
    site = original / "lib" / "python3.12" / "site-packages"
    library = sdk_fixture(site, "_7_9_nonce")
    binary = original / "bin" / "native-detector"
    binary.parent.mkdir()
    wrapper = (
        "#include <cstdio>\n"
        "namespace ckc { struct CompilerInfo { unsigned major,minor,patch;"
        "const char *source,*requested,*comgr,*query; };"
        "const CompilerInfo* candidate_compiler_info(); }\n"
        "int main(){auto p=ckc::candidate_compiler_info();if(!p)return 1;"
        'std::printf("%u\\n%s\\n",p->major,p->requested);return 0;}\n'
    )
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-x",
            "c++",
            "-",
            "-x",
            "none",
            str(native_detector.parent / "librocke_core.a"),
            "-ldl",
            "-pthread",
            "-o",
            str(binary),
        ],
        input=wrapper,
        capture_output=True,
        text=True,
        check=True,
    )
    relocated = tmp_path / "relocated"
    original.rename(relocated)
    binary = relocated / binary.relative_to(original)
    library = relocated / library.relative_to(original)
    env = automatic_env(relocated / site.relative_to(original))
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [str(binary)], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == ["23", str(library)]


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
    lambda: lower_llvm.lower_kernel_to_llvm(kernel, arch="gfx950"),
    lambda: rocke_engine.lower_serialized_ir(serialized, arch="gfx950"),
):
    try:
        lower()
    except RuntimeError as error:
        assert "Cannot determine LLVM flavor" in str(error), str(error)
    else:
        raise AssertionError("AUTO emitted IR without compiler evidence")

for flavor in lower_llvm.LLVM_FLAVORS:
    py = lower_llvm.lower_kernel_to_llvm(kernel, arch="gfx950", llvm_flavor=flavor)
    native = rocke_engine.lower_serialized_ir(serialized, arch="gfx950", flavor=flavor)
    assert py == native

os.environ["ROCKE_LLVM_FLAVOR"] = "llvm20"
assert lower_llvm.lower_kernel_to_llvm(kernel, arch="gfx950") == rocke_engine.lower_serialized_ir(
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
