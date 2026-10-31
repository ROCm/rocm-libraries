# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fresh-process Linux wheel discovery, loader ownership and dependency tests.

These small ELF libraries are loader fixtures, not real SDK compiler evidence.
Distribution records locate the payload without executing its package initializer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def linux_loader():
    if sys.platform != "linux" or not shutil.which("cc"):
        pytest.skip("ELF fixtures require Linux and a C compiler")


def build_library(root, major=23, name="libamd_comgr.so.3", *, text=None, extra=()):
    root.mkdir(parents=True, exist_ok=True)
    library = root / name
    if text is None:
        text = (
            "void amd_comgr_get_version(void){}\n"
            "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c)"
            f"{{*a={major};*b=2;*c=1;}}\n"
        )
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
            str(library),
        ],
        input=text,
        text=True,
        check=True,
        capture_output=True,
    )
    return library


def sdk_install(site, suffix=""):
    package = site / ("_rocm_sdk_core" + suffix)
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('SDK imported')\n")
    metadata = site / ("rocm_sdk_core" + suffix + "-10.0.dist-info")
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: rocm-sdk-core{suffix}\nVersion: 10.0\n"
    )
    (metadata / "RECORD").write_text(f"{package.name}/__init__.py,,\n")
    return package / "lib"


def run_child(site, script, *arguments, **overrides):
    env = dict(os.environ)
    for name in (
        "ROCKE_COMGR_LIB",
        "ROCKE_LLVM_FLAVOR",
        "ROCM_PATH",
        "ROCM_HOME",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join([str(site), *filter(None, sys.path)])
    env.update(overrides)
    # Host installation tiers are controlled; dlopen and /proc maps are real.
    bootstrap = """
from rocke.runtime import runtime_coexistence as rc
from importlib import metadata
from importlib.machinery import PathFinder
import os
original_distributions = metadata.distributions
metadata.distributions = lambda: original_distributions(path=[os.getcwd()])
# Keep SDK fallback scoped to the fixture install.
from importlib import util
real_find_spec = util.find_spec
util.find_spec = lambda name, *args: (
    PathFinder.find_spec(name, [os.getcwd()]) if name == '_rocm_sdk_core'
    else real_find_spec(name, *args)
)
real_glob = rc.glob.glob
rc.glob.glob = lambda p: [] if p.startswith('/opt/rocm') else real_glob(p)
"""
    result = subprocess.run(
        [sys.executable, "-c", bootstrap + script, *map(str, arguments)],
        env=env,
        cwd=site,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout) if result.stdout.strip() else None


QUERY = """
import json, sys
from dataclasses import asdict
from rocke.runtime import comgr
hint = comgr.resolved_lib_path()
info = comgr.loaded_compiler_info()
print(json.dumps(dict(hint=hint, info=asdict(info),
                     imports=[n for n in sys.modules if n == 'torch' or
                              n == 'rocm_sdk' or n.startswith('_rocm_sdk_core')],
                     hip=rc._mapped_lib('amdhip64'))))
"""


@pytest.mark.parametrize("suffix", ["", "_gfx94X"])
def test_installed_wheel_without_torch_or_rocm_roots(tmp_path, suffix):
    library = build_library(sdk_install(tmp_path, suffix))
    result = run_child(tmp_path, QUERY)
    assert result["hint"] == result["info"]["requested_comgr"] == str(library)
    assert result["info"]["comgr_path"] == str(library)
    assert result["info"]["query_library_path"] == str(library)
    assert result["info"]["llvm_version"] == [23, 2, 1]
    assert result["imports"] == [] and result["hip"] is None


@pytest.mark.parametrize("override", [None, "", "valid", "missing", "text"])
def test_wheel_precedence_and_safe_override_fallback(tmp_path, override):
    wheel = build_library(sdk_install(tmp_path))
    system = build_library(tmp_path / "rocm-99.0" / "lib", 20)
    env = dict(ROCM_PATH=str(system.parent.parent))
    expected = wheel
    if override == "valid":
        expected = system
        env["ROCKE_COMGR_LIB"] = str(system)
    elif override == "missing":
        env["ROCKE_COMGR_LIB"] = str(tmp_path / "missing.so")
    elif override == "text":
        invalid = tmp_path / "invalid.so"
        invalid.write_text("not ELF")
        env["ROCKE_COMGR_LIB"] = str(invalid)
    elif override is not None:
        env["ROCKE_COMGR_LIB"] = override
    result = run_child(tmp_path, QUERY, **env)
    assert result["info"]["requested_comgr"] == str(expected)
    assert result["info"]["llvm_version"][0] == (20 if expected == system else 23)


@pytest.mark.parametrize("kind", ["cpu", "cuda", "rocm"])
def test_imported_torch_does_not_add_a_default_comgr_tier(tmp_path, kind):
    wheel = build_library(sdk_install(tmp_path))
    bundle = build_library(tmp_path / "torch" / "lib", 20, "libamd_comgr.so")
    script = (
        """
import sys
from types import SimpleNamespace
sys.modules['torch'] = SimpleNamespace(__file__=sys.argv[1],
    version=SimpleNamespace(hip='99.0' if sys.argv[2] == 'rocm' else None))
"""
        + QUERY
    )
    result = run_child(tmp_path, script, bundle.parent.parent / "__init__.py", kind)
    assert result["info"]["requested_comgr"] == str(wheel)


def test_default_does_not_load_a_mapped_torch_compiler_conflict(tmp_path):
    wheel = build_library(sdk_install(tmp_path))
    bundle = build_library(tmp_path / "torch" / "lib", 20, "libamd_comgr.so")
    script = """
import ctypes, sys
from rocke.runtime import comgr
retained = ctypes.CDLL(sys.argv[1])
try:
    comgr.loaded_compiler_info()
except comgr.ComgrError as error:
    assert sys.argv[1] in str(error) and sys.argv[2] in str(error)
    assert 'separate process' in str(error)
else:
    raise AssertionError('loaded a second compiler')
assert rc._mapped_lib('amd_comgr', require_unique=True) == sys.argv[1]
"""
    run_child(tmp_path, script, bundle, wheel)


def test_late_opt_in_does_not_import_torch_beside_retained_compiler(tmp_path):
    build_library(sdk_install(tmp_path))
    torch = tmp_path / "torch"
    torch.mkdir()
    (torch / "__init__.py").write_text("raise AssertionError('late torch import')")
    run_child(
        tmp_path,
        """
import sys
from rocke.runtime import comgr
first = comgr.loaded_compiler_info()
try:
    comgr.prefer_bundled_lib()
except comgr.ComgrError as error:
    assert 'retained compiler' in str(error) and 'before first' in str(error)
else:
    raise AssertionError('late opt-in imported torch')
assert 'torch' not in sys.modules
assert comgr.loaded_compiler_info() is first
""",
    )


@pytest.mark.parametrize("mode", ["live", "deleted", "multiple"])
def test_late_opt_in_rejects_external_compiler_before_torch_import(tmp_path, mode):
    external = build_library(tmp_path / "external")
    bundle = build_library(tmp_path / "torch" / "lib", 20, "libamd_comgr.so")
    (bundle.parent.parent / "__init__.py").write_text(
        "from pathlib import Path\nimport ctypes\n"
        "Path('torch-imported').write_text('imported')\n"
        "_compiler = ctypes.CDLL(str(Path(__file__).parent / 'lib/libamd_comgr.so'))\n"
    )
    other = build_library(tmp_path / "other", 22)
    run_child(
        tmp_path,
        """
import ctypes, sys
from pathlib import Path
from rocke.runtime import comgr
external = ctypes.CDLL(sys.argv[1])
if sys.argv[2] == 'deleted':
    Path(sys.argv[1]).unlink()
elif sys.argv[2] == 'multiple':
    other = ctypes.CDLL(sys.argv[3])
before = rc._mapped_comgr_files()
assert len(before) == (2 if sys.argv[2] == 'multiple' else 1)
assert comgr._lib is None and 'torch' not in sys.modules
try:
    comgr.prefer_bundled_lib()
except comgr.ComgrError as error:
    assert 'before first' in str(error) and 'separate process' in str(error)
else:
    raise AssertionError('late opt-in accepted an external compiler')
assert not Path('torch-imported').exists()
assert 'torch' not in sys.modules and comgr._lib is None
assert rc._mapped_comgr_files() == before
""",
        external,
        mode,
        other,
        ROCKE_COMGR_LIB=str(external),
    )


@pytest.mark.parametrize("maps", ["readable", "unavailable"])
def test_clean_opt_in_imports_torch_without_a_retained_compiler(tmp_path, maps):
    bundle = build_library(tmp_path / "torch" / "lib", 20, "libamd_comgr.so")
    (bundle.parent.parent / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "Path('torch-imported').write_text('imported')\n"
        "version = SimpleNamespace(hip='7.0')\n"
    )
    run_child(
        tmp_path,
        """
import sys
from pathlib import Path
from rocke.runtime import comgr
assert comgr._lib is None and not rc._mapped_comgr_files()
if sys.argv[2] == 'unavailable':
    rc._PROC_MAPS = 'unavailable-maps'
    assert rc._mapped_comgr_files() is None
assert comgr.prefer_bundled_lib() == (7, 0)
assert Path('torch-imported').exists() and 'torch' in sys.modules
info = comgr.loaded_compiler_info()
assert info.comgr_path == sys.argv[1] and info.llvm_version == (20, 2, 1)
rc._PROC_MAPS = '/proc/self/maps'
assert len(rc._mapped_comgr_files()) == 1
""",
        bundle,
        maps,
    )


@pytest.mark.parametrize("alias", [False, True])
def test_bundled_opt_in_accepts_retained_bare_name(tmp_path, alias):
    bundle = build_library(tmp_path / "torch" / "lib", 20, "libamd_comgr.so")
    linker_dir = bundle.parent
    if alias:
        linker_dir = tmp_path / "linker"
        linker_dir.mkdir()
        (linker_dir / bundle.name).hardlink_to(bundle)
    run_child(
        tmp_path,
        """
import os, sys
from types import SimpleNamespace
from rocke.runtime import comgr
sys.modules['torch'] = SimpleNamespace(__file__=sys.argv[1],
    version=SimpleNamespace(hip='7.0'))
assert rc._comgr_candidate_paths(['3']) == ['libamd_comgr.so']
first = comgr.loaded_compiler_info()
retained = comgr._lib
assert retained._name == 'libamd_comgr.so'
assert os.path.samefile(first.comgr_path, sys.argv[2])
assert os.path.samefile(rc._torch_bundled_lib('amd_comgr'), sys.argv[2])
comgr.prefer_bundled_lib()
assert comgr._lib is retained and comgr.loaded_compiler_info() is first
assert len(rc._mapped_comgr_files()) == 1
""",
        bundle.parent.parent / "__init__.py",
        bundle,
        LD_LIBRARY_PATH=str(linker_dir),
    )


@pytest.mark.parametrize("provenance", ["unavailable", "missing_symbol"])
@pytest.mark.parametrize("selection", ["same", "different"])
def test_bundled_opt_in_retains_name_fallback(tmp_path, provenance, selection):
    text = None
    if provenance == "missing_symbol":
        text = (
            "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c)"
            "{*a=20;*b=2;*c=1;}\n"
        )
    bundle = build_library(tmp_path / "torch" / "lib", 20, "libamd_comgr.so", text=text)
    retained = bundle
    if selection == "different":
        retained = build_library(tmp_path / "external", text=text)
    run_child(
        tmp_path,
        """
import os, sys
from types import SimpleNamespace
from rocke.runtime import comgr
sys.modules['torch'] = SimpleNamespace(__file__=sys.argv[1],
    version=SimpleNamespace(hip='7.0'))
first = comgr.loaded_compiler_info()
retained = comgr._lib
assert retained._name == sys.argv[2]
os.environ.pop('ROCKE_COMGR_LIB')
if sys.argv[3] == 'unavailable':
    comgr._library_for_symbol = lambda fn: None
else:
    assert not hasattr(retained, 'amd_comgr_get_version')
try:
    comgr.prefer_bundled_lib()
except comgr.ComgrError as error:
    assert sys.argv[4] == 'different'
    assert 'retained compiler' in str(error) and 'separate process' in str(error)
else:
    assert sys.argv[4] == 'same'
assert comgr._lib is retained and comgr.loaded_compiler_info() is first
assert len(rc._mapped_comgr_files()) == 1
""",
        bundle.parent.parent / "__init__.py",
        retained,
        provenance,
        selection,
        ROCKE_COMGR_LIB=str(retained),
    )


@pytest.mark.parametrize(
    "failure,entry",
    [
        (failure, entry)
        for failure in ("missing", "text", "dependency")
        for entry in ("opt_in", "default")
    ]
    + [("bundle", "opt_in")],
)
def test_repeated_opt_in_after_fallback_retains_compiler(tmp_path, entry, failure):
    bundle = build_library(tmp_path / "torch" / "lib", 20, "libamd_comgr.so")
    (bundle.parent.parent / "__init__.py").write_text(
        "from types import SimpleNamespace\nversion = SimpleNamespace(hip='7.0')\n"
    )
    failed = tmp_path / "failed" / "libamd_comgr.so"
    if failure == "text":
        failed.parent.mkdir()
        failed.write_text("not ELF")
    elif failure == "dependency":
        dependency = build_library(
            failed.parent, name="libfixture.so", text="void fixture(void){}"
        )
        build_library(
            failed.parent,
            name=failed.name,
            text="extern void fixture(void); void amd_comgr_get_version(void){fixture();}",
            extra=("-L" + str(failed.parent), "-lfixture", "-Wl,-rpath,$ORIGIN"),
        )
        dependency.unlink()
    expected = bundle
    env = dict(ROCKE_COMGR_LIB=str(failed))
    if failure == "bundle":
        bundle.write_text("not ELF")
        expected = build_library(sdk_install(tmp_path))
        env = {}
    elif entry == "default":
        # Default discovery and opt-in must lead to the same compiler.
        env["ROCM_PATH"] = str(bundle.parent.parent)
    different = build_library(tmp_path / "different", 22)
    run_child(
        tmp_path,
        """
import os, sys
from rocke.runtime import comgr
import torch
if sys.argv[1] == 'opt_in':
    comgr.prefer_bundled_lib()
first = comgr.loaded_compiler_info()
retained = comgr._lib
assert os.path.samefile(first.comgr_path, sys.argv[2])
real_cdll = comgr.ctypes.CDLL
def no_second_library(name, *args, **kwargs):
    if name is None:
        return real_cdll(name, *args, **kwargs)  # dladdr provenance lookup
    raise AssertionError('repeat opened another library')
comgr.ctypes.CDLL = no_second_library
for _ in range(2):
    comgr.prefer_bundled_lib()
    assert comgr._lib is retained and comgr.loaded_compiler_info() is first
    assert len(rc._mapped_comgr_files()) == 1
os.environ['ROCKE_COMGR_LIB'] = sys.argv[3]
try:
    comgr.prefer_bundled_lib()
except comgr.ComgrError as error:
    assert sys.argv[3] in str(error) and 'retained compiler' in str(error)
else:
    raise AssertionError('accepted a different compiler')
assert comgr._lib is retained and comgr.loaded_compiler_info() is first
assert len(rc._mapped_comgr_files()) == 1
""",
        entry,
        expected,
        different,
        **env,
    )


@pytest.mark.parametrize("change", ["repair", "cwd"])
def test_relative_failed_override_remains_bound_to_original_selection(tmp_path, change):
    bundle = build_library(tmp_path / "torch" / "lib", 20, "libamd_comgr.so")
    (bundle.parent.parent / "__init__.py").write_text(
        "from types import SimpleNamespace\nversion = SimpleNamespace(hip='7.0')\n"
    )
    new_cwd = tmp_path / "other"
    different = build_library(new_cwd / "failed", 22, "libamd_comgr.so")
    run_child(
        tmp_path,
        """
import os, sys
from pathlib import Path
from rocke.runtime import comgr
comgr.prefer_bundled_lib()
first = comgr.loaded_compiler_info()
retained = comgr._lib
if sys.argv[1] == 'repair':
    Path('failed').mkdir()
    Path('failed/libamd_comgr.so').symlink_to(sys.argv[2])
else:
    os.chdir(sys.argv[3])
real_cdll = comgr.ctypes.CDLL
def no_second_library(name, *args, **kwargs):
    if name is None:
        return real_cdll(name, *args, **kwargs)
    raise AssertionError('repeat opened another library')
comgr.ctypes.CDLL = no_second_library
try:
    comgr.prefer_bundled_lib()
except comgr.ComgrError as error:
    assert sys.argv[1] == 'cwd' and 'retained compiler' in str(error)
else:
    assert sys.argv[1] == 'repair'
assert comgr._lib is retained and comgr.loaded_compiler_info() is first
assert len(rc._mapped_comgr_files()) == 1
""",
        change,
        different,
        new_cwd,
        ROCKE_COMGR_LIB="failed/libamd_comgr.so",
    )


@pytest.mark.parametrize("mode", ["bundle", "override", "missing", "retained"])
def test_explicit_bundled_opt_in_and_retained_handle(tmp_path, mode):
    wheel = build_library(sdk_install(tmp_path))
    bundle = build_library(tmp_path / "torch" / "lib", 20, "libamd_comgr.so")
    if mode == "missing":
        bundle.unlink()
    script = """
import os, sys
from types import SimpleNamespace
from rocke.runtime import comgr
sys.modules['torch'] = SimpleNamespace(__file__=sys.argv[1],
    version=SimpleNamespace(hip='7.0'))
mode = sys.argv[2]
if mode == 'retained':
    first = comgr.loaded_compiler_info()
    try:
        comgr.prefer_bundled_lib()
    except comgr.ComgrError as error:
        assert 'retained compiler' in str(error) and sys.argv[3] in str(error)
    else:
        raise AssertionError('replaced retained compiler')
    assert comgr.loaded_compiler_info() is first
else:
    comgr.prefer_bundled_lib()
    info = comgr.loaded_compiler_info()
    expected = sys.argv[3] if mode in ('override', 'missing') else sys.argv[4]
    assert info.requested_comgr == expected
    assert info.llvm_version[0] == (23 if mode in ('override', 'missing') else 20)
    assert 'ROCKE_COMGR_LIB' not in os.environ or mode == 'override'
"""
    env = dict(ROCKE_COMGR_LIB=str(wheel)) if mode == "override" else {}
    run_child(
        tmp_path,
        script,
        bundle.parent.parent / "__init__.py",
        mode,
        wheel,
        bundle,
        **env,
    )


@pytest.mark.parametrize("failure", ["absent", "missing", "invalid", "dependency"])
def test_unusable_wheel_falls_back_before_query(tmp_path, failure):
    if failure != "absent":
        root = sdk_install(tmp_path)
        if failure == "invalid":
            root.mkdir()
            (root / "libamd_comgr.so.3").write_text("invalid ELF")
        elif failure == "dependency":
            dependency = build_library(
                root, name="libfixture.so", text="void fixture(void){}"
            )
            build_library(
                root,
                text="extern void fixture(void); void amd_comgr_get_version(void){fixture();}",
                extra=("-L" + str(root), "-lfixture", "-Wl,-rpath,$ORIGIN"),
            )
            dependency.unlink()
    fallback = build_library(tmp_path / "fallback" / "lib", 22)
    result = run_child(tmp_path, QUERY, ROCM_PATH=str(fallback.parent.parent))
    assert result["info"]["requested_comgr"] == str(fallback)
    assert result["info"]["llvm_version"] == [22, 2, 1]


def test_versioned_names_aliases_and_failed_candidate(tmp_path):
    root = sdk_install(tmp_path)
    root.mkdir()
    (root / "libamd_comgr.so").write_text("invalid ELF")
    valid = build_library(root, name="libamd_comgr.so.3.12.0")
    (root / "libamd_comgr.so.3").symlink_to(valid.name)
    (root / "libamd_comgr.so.debug").write_text("debug")
    (root / "libamd_comgr.so.2.backup").write_text("backup")
    (root / "libamd_comgr.so.1").mkdir()
    script = (
        """
import sys
from rocke.runtime import comgr
paths = rc._sdk_comgr_candidates()
assert paths == [sys.argv[1], sys.argv[2]], paths
"""
        + QUERY
    )
    result = run_child(
        tmp_path, script, root / "libamd_comgr.so", root / "libamd_comgr.so.3"
    )
    assert result["hint"] == str(root / "libamd_comgr.so")
    assert result["info"]["requested_comgr"] == str(root / "libamd_comgr.so.3")


@pytest.mark.parametrize("relative", [".", "rocm_sysdeps/lib"])
def test_wheel_dependency_provenance_survives_relocation(tmp_path, relative):
    site = tmp_path / "original"
    root = sdk_install(site, "_fixture")
    dependency = build_library(root / relative, name="libfixtureLLVM.so")
    library = build_library(
        root,
        text="extern void amd_comgr_get_version(void); void fixture_anchor(void){amd_comgr_get_version();}",
        extra=(
            "-L" + str(dependency.parent),
            "-lfixtureLLVM",
            "-Wl,-rpath,$ORIGIN/" + relative,
        ),
    )
    expected_query = str(dependency)
    result = run_child(site, QUERY)
    assert os.path.samefile(result["info"]["query_library_path"], expected_query)
    relocated = tmp_path / "relocated"
    site.rename(relocated)
    result = run_child(relocated, QUERY)
    assert result["info"]["requested_comgr"] == str(
        relocated / library.relative_to(site)
    )
    assert os.path.samefile(
        result["info"]["query_library_path"], relocated / dependency.relative_to(site)
    )
    assert result["info"]["llvm_version"] == [23, 2, 1]


def test_loaded_unqueryable_sdk_never_changes_compiler(tmp_path):
    wheel = build_library(
        sdk_install(tmp_path), text="void amd_comgr_get_version(void){}"
    )
    fallback = build_library(tmp_path / "fallback" / "lib", 20)
    script = """
import os, sys
from rocke.core import lower_llvm
from rocke.core.ir import IRBuilder
from rocke.runtime import comgr
first = comgr.loaded_compiler_info()
assert first.requested_comgr == sys.argv[1] and first.llvm_version is None
os.environ['ROCKE_COMGR_LIB'] = sys.argv[2]
assert comgr.loaded_compiler_info() is first
try:
    lower_llvm._resolve_llvm_flavor()
except RuntimeError as error:
    assert 'Cannot determine LLVM flavor' in str(error)
else:
    raise AssertionError('AUTO used a different compiler')
assert lower_llvm.lower_kernel_to_llvm(IRBuilder('offline').kernel, arch='gfx950', llvm_flavor='llvm20')
"""
    run_child(tmp_path, script, wheel, fallback, ROCM_PATH=str(fallback.parent.parent))


@pytest.mark.parametrize("mode", ["alias", "multiple", "deleted", "bare", "fallback"])
def test_mapped_compiler_safety_uses_all_files(tmp_path, mode):
    wheel = build_library(sdk_install(tmp_path))
    other = build_library(tmp_path / "other", 20)
    alias = tmp_path / "alias.so"
    alias.symlink_to(wheel)
    script = """
import ctypes, os, sys
from rocke.runtime import comgr
first = ctypes.CDLL(sys.argv[1])
mode = sys.argv[4]
handles = [first]
if mode == 'multiple':
    # Fixture libraries have no LLVM registration, so both can safely be mapped.
    handles.append(ctypes.CDLL(sys.argv[2]))
if mode == 'deleted':
    os.unlink(sys.argv[1])
    rc._sdk_comgr_candidates = lambda: [sys.argv[2]]
if mode == 'bare':
    rc._sdk_comgr_candidates = lambda: []
if mode == 'fallback':
    os.environ['ROCKE_COMGR_LIB'] = 'missing.so'
    rc._sdk_comgr_candidates = lambda: [sys.argv[2]]
if mode == 'alias':
    os.environ['ROCKE_COMGR_LIB'] = sys.argv[3]
    assert comgr.loaded_compiler_info().requested_comgr == sys.argv[3]
else:
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as error:
        assert 'mapped COMGR' in str(error) and 'separate process' in str(error)
    else:
        raise AssertionError('accepted a conflicting or unresolved compiler')
"""
    run_child(tmp_path, script, wheel, other, alias, mode)


def test_unreadable_maps_is_a_documented_detection_limit(tmp_path):
    wheel = build_library(sdk_install(tmp_path))
    result = run_child(
        tmp_path,
        "rc._PROC_MAPS = 'unavailable-maps'\nassert rc._mapped_comgr_files() is None\n"
        + QUERY,
    )
    assert result["info"]["requested_comgr"] == str(wheel)


def test_failure_diagnostics_retain_candidate_order(tmp_path):
    root = sdk_install(tmp_path)
    root.mkdir()
    broken = root / "libamd_comgr.so.3"
    broken.write_text("not ELF")
    script = """
from rocke.runtime import comgr
import sys
real = comgr.ctypes.CDLL
comgr.ctypes.CDLL = lambda p, *a, **kw: (_ for _ in ()).throw(OSError('fixture load failure'))
try:
    comgr._resolve_lib()
except comgr.ComgrError as error:
    text = str(error)
    assert text.index(sys.argv[1]) < text.index(sys.argv[2]) < text.rindex('libamd_comgr.so')
else:
    raise AssertionError('load did not fail')
"""
    run_child(
        tmp_path,
        script,
        tmp_path / "missing.so",
        broken,
        ROCKE_COMGR_LIB=str(tmp_path / "missing.so"),
    )
