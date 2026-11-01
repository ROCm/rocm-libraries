# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The cached HIP-header precompiled header of the helper-kernel compile
(timing_runs/DECISIONS.md topic 6, E): its key, its cache, and the fallback to a compile
without it."""

import os
import shutil
import stat
import time
from pathlib import Path

import pytest

import Tensile.Toolchain.Component as Component
import Tensile.Toolchain.HelperKernelPch as HKP
from Tensile.Common import SemanticVersion
from Tensile.Toolchain.Source import _compileHelperSource

pytestmark = pytest.mark.unit

# A stand-in for amdclang++: --version prints version.txt, -### prints a compile job whose
# header search path is hip_include.txt, a PCH build copies stdin to stdout, and a compile
# writes its -o file, or fails like clang on a stale PCH when stale_pch exists.
FAKE_DRIVER = r"""#!/bin/sh
dir=$(dirname "$0")
echo "$*" >> "$dir/calls.txt"
for a in "$@"; do
  case "$a" in
    --version) cat "$dir/version.txt"; exit 0;;
    "-###") echo "clang version 22" >&2
            inc=$(cat "$dir/hip_include.txt")
            printf ' "/x/clang" "-cc1" "-idirafter" "%s" "-include" "%s" "-x" "hip" "-"\n' \
              "$inc" "__clang_hip_runtime_wrapper.h" >&2
            exit 0;;
  esac
done
out=""; pch=""; prev=""
for a in "$@"; do
  if [ "$a" = "-emit-pch" ]; then
    [ -e "$dir/pch_build_fails" ] && { echo "pch error" >&2; exit 1; }
    cat; exit 0
  fi
  [ "$prev" = "-o" ] && out="$a"
  [ "$prev" = "-include-pch" ] && pch="$a"
  prev="$a"
done
if [ -n "$pch" ] && [ -e "$dir/stale_pch" ]; then
  echo "fatal error: file 'hip_runtime.h' has been modified since the precompiled header" >&2
  exit 1
fi
[ -e "$dir/compile_fails" ] && { echo "error: bad kernel" >&2; exit 1; }
echo obj > "$out"
"""


@pytest.fixture
def fakeDriver(tmp_path, monkeypatch):
    driverDir = tmp_path / "driver"
    driverDir.mkdir()
    driver = driverDir / "amdclang++"
    driver.write_text(FAKE_DRIVER)
    driver.chmod(driver.stat().st_mode | stat.S_IXUSR)
    (driverDir / "version.txt").write_text(
        "AMD clang version 22.0.0git (roc-7.2.1 26084 f58b06dce1f9)\n"
    )
    hipInclude = tmp_path / "rocm" / "include"
    hipInclude.mkdir(parents=True)
    (driverDir / "hip_include.txt").write_text(str(hipInclude))
    monkeypatch.setattr(Component, "_getVersion", lambda *_a: SemanticVersion(22, 0, 0))
    monkeypatch.delenv("Tensile_CXX_COMPILER_LAUNCHER", raising=False)
    monkeypatch.delenv("TENSILE_DISABLE_HELPER_PCH", raising=False)
    monkeypatch.setenv("TENSILE_HELPER_PCH_DIR", str(tmp_path / "pch_cache"))
    monkeypatch.setattr(HKP, "_rejections", {})
    return driverDir


def _compiler(driverDir, asan=False):
    return Component.Compiler(str(driverDir / "amdclang++"), "sha1", asan_build=asan)


def _calls(driverDir, flag):
    calls = driverDir / "calls.txt"
    return [c for c in calls.read_text().splitlines() if flag in c] if calls.exists() else []


def _key(compiler, targets):
    pchArgs = compiler.pchArgs(targets)
    return HKP._computePchKey(
        pchArgs, targets, HKP._compilerVersion(compiler.path), HKP._headerSearchPaths(pchArgs)
    )


# =========================================================================== #
# Key composition
# =========================================================================== #
BASE = dict(
    pchArgs=["amdclang++", "-O3", "--offload-arch=gfx950", "-"],
    targets=["gfx950"],
    compilerVersion="AMD clang version 22.0.0git (roc-7.2.1 26084 f58b06dce1f9)\n",
    headerSearchPaths=["-idirafter /opt/rocm-7.2.1/include"],
)


def test_key_is_deterministic():
    assert HKP._computePchKey(**BASE) == HKP._computePchKey(**dict(BASE))


@pytest.mark.parametrize(
    "field, value",
    [
        ("pchArgs", ["amdclang++", "-O3", "-DFOO=1", "--offload-arch=gfx950", "-"]),
        ("pchArgs", ["amdclang++", "-O2", "--offload-arch=gfx950", "-"]),
        ("pchArgs", ["amdclang++", "-O3", "--offload-arch=gfx950", "-fsanitize=address", "-"]),
        ("targets", ["gfx942"]),
        # Same major.minor.patch, another build: clang rejects a PCH across compiler builds.
        ("compilerVersion", "AMD clang version 22.0.0git (roc-7.2.0 25133 0123456789ab)\n"),
        ("headerSearchPaths", ["-idirafter /opt/rocm-7.2.0/include"]),
    ],
)
def test_every_input_changes_the_key(field, value):
    assert HKP._computePchKey(**{**BASE, field: value}) != HKP._computePchKey(**BASE)


def test_key_covers_the_pch_headers(monkeypatch):
    before = HKP._computePchKey(**BASE)
    monkeypatch.setattr(HKP, "PCH_SOURCE", HKP.PCH_SOURCE + b"#include <hip/hip_bf16.h>\n")
    assert HKP._computePchKey(**BASE) != before


def test_key_uses_the_full_version_output_and_the_resolved_hip_include_path(fakeDriver, tmp_path):
    compiler = _compiler(fakeDriver)
    key = _key(compiler, ["gfx950"])

    (fakeDriver / "version.txt").write_text(
        "AMD clang version 22.0.0git (roc-7.2.1 26085 0123456789ab)\n"
    )
    newBuild = _key(compiler, ["gfx950"])
    assert newBuild != key

    # /opt/rocm -> /opt/rocm-<version> style link: the key follows the link's target.
    otherRocm = tmp_path / "rocm-other" / "include"
    otherRocm.mkdir(parents=True)
    link = tmp_path / "rocm-link"
    link.symlink_to(tmp_path / "rocm-other")
    (fakeDriver / "hip_include.txt").write_text(str(link / "include"))
    assert HKP._headerSearchPaths(compiler.pchArgs(["gfx950"])) == [
        f"-idirafter {otherRocm}",
        "-include __clang_hip_runtime_wrapper.h",
    ]
    assert _key(compiler, ["gfx950"]) != newBuild


def test_key_covers_every_compile_argument(fakeDriver):
    assert _key(_compiler(fakeDriver), ["gfx950"]) != _key(
        _compiler(fakeDriver, asan=True), ["gfx950"]
    )
    assert _key(_compiler(fakeDriver), ["gfx950"]) != _key(_compiler(fakeDriver), ["gfx950:xnack+"])


# =========================================================================== #
# Compiler arguments
# =========================================================================== #
def test_pch_is_built_with_the_compile_arguments(fakeDriver, monkeypatch):
    compiler = _compiler(fakeDriver)
    calls = []
    monkeypatch.setattr(Component, "_invoke", lambda args, desc="": calls.append(args))
    compiler("inc", ["gfx950"], "Kernels.cpp", "Kernels.o")
    compiler("inc", ["gfx950"], "Kernels.cpp", "Kernels.o", pch_path="/c/hip.pch")
    plain, withPch = calls

    assert plain == [
        *compiler.default_args,
        "-I",
        "inc",
        "--offload-arch=gfx950",
        "Kernels.cpp",
        "-c",
        "-o",
        "Kernels.o",
    ]
    assert withPch == [
        *compiler.default_args,
        "-I",
        "inc",
        "--offload-arch=gfx950",
        "-include-pch",
        "/c/hip.pch",
        "-Xclang",
        "-fmodules-validate-system-headers",
        "Kernels.cpp",
        "-c",
        "-o",
        "Kernels.o",
    ]
    assert compiler.pchArgs(["gfx950"]) == [
        *compiler.default_args,
        "--offload-arch=gfx950",
        "-fpch-instantiate-templates",
        "-Xclang",
        "-emit-pch",
        "-fsyntax-only",
        "-",
    ]


# =========================================================================== #
# Cache
# =========================================================================== #
def test_miss_builds_and_stores_then_hits(fakeDriver, tmp_path):
    compiler = _compiler(fakeDriver)
    pchPath = HKP.HelperKernelPch().get(compiler, ["gfx950"])

    key = _key(compiler, ["gfx950"])
    assert pchPath == tmp_path / "pch_cache" / key / "hip.pch"
    assert pchPath.read_bytes() == HKP.PCH_SOURCE  # the fake driver's "PCH" is its stdin
    assert [p.name for p in (tmp_path / "pch_cache").iterdir()] == [key]

    assert HKP.HelperKernelPch().get(compiler, ["gfx950"]) == pchPath
    pchBuilds = [c for c in _calls(fakeDriver, "-emit-pch") if "-###" not in c]
    assert len(pchBuilds) == 1


def test_refreshes_the_entry_on_a_hit(fakeDriver):
    compiler = _compiler(fakeDriver)
    entry = HKP.HelperKernelPch().get(compiler, ["gfx950"]).parent
    old = time.time() - 10 * 24 * 3600
    os.utime(entry, (old, old))
    HKP.HelperKernelPch().get(compiler, ["gfx950"])
    assert entry.stat().st_mtime > old + 3600


def test_stale_entries_are_evicted(fakeDriver):
    compiler = _compiler(fakeDriver)
    entry = HKP.HelperKernelPch().get(compiler, ["gfx950"]).parent
    old = time.time() - (HKP.HelperKernelPch._MAX_AGE_DAYS + 1) * 24 * 3600
    os.utime(entry, (old, old))
    HKP.HelperKernelPch()
    assert not entry.exists()


@pytest.mark.parametrize("disable", ["1", "yes", "ON", "true"])
def test_disable_switch(fakeDriver, tmp_path, monkeypatch, disable):
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_PCH", disable)
    assert HKP.HelperKernelPch().get(_compiler(fakeDriver), ["gfx950"]) is None
    assert not (tmp_path / "pch_cache").exists()
    assert _calls(fakeDriver, "") == []


def test_no_pch_for_several_targets(fakeDriver, tmp_path):
    assert HKP.HelperKernelPch().get(_compiler(fakeDriver), ["gfx942", "gfx950"]) is None
    assert _calls(fakeDriver, "") == []


def test_no_pch_under_a_compiler_launcher(fakeDriver, monkeypatch):
    monkeypatch.setenv("Tensile_CXX_COMPILER_LAUNCHER", "ccache")
    assert HKP.HelperKernelPch().get(_compiler(fakeDriver), ["gfx950"]) is None
    assert _calls(fakeDriver, "") == []


def test_failed_pch_build_compiles_without_a_pch(fakeDriver, tmp_path):
    (fakeDriver / "pch_build_fails").touch()
    assert HKP.HelperKernelPch().get(_compiler(fakeDriver), ["gfx950"]) is None
    assert list((tmp_path / "pch_cache").iterdir()) == []  # no entry, no temporary left


def test_failed_compiler_query_compiles_without_a_pch(fakeDriver, tmp_path):
    compiler = _compiler(fakeDriver)
    (fakeDriver / "amdclang++").unlink()
    assert HKP.HelperKernelPch().get(compiler, ["gfx950"]) is None


def test_concurrent_store_keeps_the_first_entry(fakeDriver, tmp_path):
    """A build that loses the race to store its key leaves the stored entry in place."""
    compiler = _compiler(fakeDriver)
    key = _key(compiler, ["gfx950"])
    first = tmp_path / "pch_cache" / key
    first.mkdir(parents=True)
    (first / "hip.pch").write_bytes(b"first")
    assert HKP.HelperKernelPch()._build(compiler, ["gfx950"], key) == first / "hip.pch"
    assert (first / "hip.pch").read_bytes() == b"first"
    assert [p.name for p in (tmp_path / "pch_cache").iterdir()] == [key]


# =========================================================================== #
# Fallback
# =========================================================================== #
def _compileHelpers(compiler, tmp_path):
    pch = HKP.HelperKernelPch()
    pchPath = pch.get(compiler, ["gfx950"])
    objPath = tmp_path / "Kernels.o"
    _compileHelperSource(compiler, pch, pchPath, "inc", ["gfx950"], "Kernels.cpp", str(objPath))
    return pchPath, objPath


def test_compiles_with_the_pch(fakeDriver, tmp_path):
    pchPath, objPath = _compileHelpers(_compiler(fakeDriver), tmp_path)
    assert objPath.exists()
    compiles = _calls(fakeDriver, "Kernels.cpp")
    assert len(compiles) == 1 and f"-include-pch {pchPath}" in compiles[0]


def test_stale_pch_falls_back_and_is_rebuilt(fakeDriver, tmp_path):
    compiler = _compiler(fakeDriver)
    pchPath = HKP.HelperKernelPch().get(compiler, ["gfx950"])
    (fakeDriver / "stale_pch").touch()

    _, objPath = _compileHelpers(compiler, tmp_path)

    assert objPath.exists()
    compiles = _calls(fakeDriver, "Kernels.cpp")
    assert ["-include-pch" in c for c in compiles] == [True, False]
    assert not pchPath.parent.exists()  # discarded ...
    assert list((tmp_path / "pch_cache").iterdir()) == []
    (fakeDriver / "stale_pch").unlink()
    assert HKP.HelperKernelPch().get(compiler, ["gfx950"]) == pchPath  # ... and built again


def _pchBuilds(driverDir):
    return [c for c in _calls(driverDir, "-emit-pch") if "-###" not in c]


def test_a_pch_rejected_after_its_rebuild_is_not_used_again(fakeDriver, tmp_path, capsys):
    """A compiler that rejects every PCH (e.g. hosts sharing the cache whose HIP headers differ
    only in mtime) costs one rebuild, not a PCH build and a failed compile per helper compile."""
    compiler = _compiler(fakeDriver)
    (fakeDriver / "stale_pch").touch()

    for _ in range(5):
        _, objPath = _compileHelpers(compiler, tmp_path)
        assert objPath.exists()

    assert len(_pchBuilds(fakeDriver)) == 2
    compiles = _calls(fakeDriver, "Kernels.cpp")
    # Rejected, retried; rebuilt, rejected again, retried; then once each without the PCH.
    assert ["-include-pch" in c for c in compiles] == [
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert capsys.readouterr().out.count("PCH disabled for this run") == 1
    # Only this key: another target still gets its PCH.
    assert HKP.HelperKernelPch().get(compiler, ["gfx942"]) is not None


def test_source_error_is_raised_and_keeps_the_pch(fakeDriver, tmp_path):
    compiler = _compiler(fakeDriver)
    (fakeDriver / "compile_fails").touch()
    with pytest.raises(RuntimeError, match="bad kernel"):
        _compileHelpers(compiler, tmp_path)
    compiles = _calls(fakeDriver, "Kernels.cpp")
    assert ["-include-pch" in c for c in compiles] == [True, False]
    assert HKP.HelperKernelPch().get(compiler, ["gfx950"]).exists()


def test_no_pch_compiles_once(fakeDriver, tmp_path, monkeypatch):
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_PCH", "1")
    _, objPath = _compileHelpers(_compiler(fakeDriver), tmp_path)
    assert objPath.exists()
    compiles = _calls(fakeDriver, "Kernels.cpp")
    assert len(compiles) == 1 and "-include-pch" not in compiles[0]


# =========================================================================== #
# The real compiler
# =========================================================================== #
REAL_COMPILER = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin" / "amdclang++"


@pytest.mark.skipif(not REAL_COMPILER.exists(), reason="needs a ROCm amdclang++")
def test_real_compiler_rejects_a_pch_whose_hip_header_was_touched(tmp_path, monkeypatch):
    """Clang does not check the system headers a PCH was built from unless told to; the
    compile does (-fmodules-validate-system-headers), so a header changed after the PCH was
    built (a ROCm reinstall) fails the compile with the PCH, and the build falls back."""
    rocm = REAL_COMPILER.resolve().parents[3]  # <rocm>/lib/llvm/bin/clang++
    hip = tmp_path / "hip"
    shutil.copytree(rocm / "include" / "hip", hip / "include" / "hip")
    for entry in (rocm / "include").iterdir():
        if entry.name != "hip":
            (hip / "include" / entry.name).symlink_to(entry)
    shutil.copytree(rocm / "share" / "hip", hip / "share" / "hip")
    monkeypatch.setenv("HIP_PATH", str(hip))
    monkeypatch.setenv("TENSILE_HELPER_PCH_DIR", str(tmp_path / "pch_cache"))
    monkeypatch.setattr(HKP, "_rejections", {})
    monkeypatch.delenv("TENSILE_DISABLE_HELPER_PCH", raising=False)
    monkeypatch.delenv("Tensile_CXX_COMPILER_LAUNCHER", raising=False)
    (tmp_path / "Kernels.cpp").write_text(
        "#include <hip/hip_runtime.h>\n__global__ void k(int* p) { *p = 1; }\n"
    )
    compiler = Component.Compiler(str(REAL_COMPILER), "sha1")
    pch = HKP.HelperKernelPch()
    pchPath = pch.get(compiler, ["gfx950"])
    assert pchPath is not None
    assert f"-idirafter {hip / 'include'}" in HKP._headerSearchPaths(compiler.pchArgs(["gfx950"]))

    args = (str(tmp_path), ["gfx950"], str(tmp_path / "Kernels.cpp"), str(tmp_path / "k.o"))
    compiler(*args, pch_path=str(pchPath))

    header = hip / "include" / "hip" / "hip_runtime.h"
    mtime = header.stat().st_mtime + 60
    os.utime(header, (mtime, mtime))
    with pytest.raises(RuntimeError, match="modified since the precompiled header"):
        compiler(*args, pch_path=str(pchPath))

    _compileHelperSource(compiler, pch, pchPath, *args)
    assert (tmp_path / "k.o").stat().st_size > 0
    assert not pchPath.exists()
    assert HKP.HelperKernelPch().get(compiler, ["gfx950"]) == pchPath
    compiler(*args, pch_path=str(pchPath))
