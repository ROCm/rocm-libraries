# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The production comgr without importing torch, and torch-free compile-only
paths of the attention backward.

CPU only. A fake torch package whose ``__init__`` raises stands in for the
wheel: finding its library and its ROCm vintage must never run it.

* ``attention_bwd_comgr`` finds ``<torch>/lib`` comgr and pins it as
  ``ROCKE_COMGR_LIB`` (an explicit setting wins, ``ROCKE_BWD_NO_COMGR_PIN``
  opts out); the platform names that library's ROCm vintage from the
  package's ``version.py`` without importing it, so the toolchain identity
  follows it; and it gives comgr's own cache a directory of the process tree
  (made by the first process, inherited by its children, removed at its
  exit) unless one is set or the cache is off.
* The build / lower / compile / identity / resource-gate path, the
  pre-build's compile-worker start-up and the runner leave torch out of
  ``sys.modules`` (a fresh process each).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from kernels.common import attention_bwd_comgr as pc

_VERSION_PY = """\
from typing import Optional

__all__ = ['__version__', 'hip']
__version__ = '2.12.0+rocm7.2'
hip: Optional[str] = '{hip}'
"""


@pytest.fixture
def fake_torch(tmp_path, monkeypatch):
    """A torch package on ``sys.path`` that raises when imported."""
    site = tmp_path / "site"
    pkg = site / "torch"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise RuntimeError('torch was imported')\n")
    (pkg / "version.py").write_text(_VERSION_PY.format(hip="7.2.53211"))
    (pkg / "lib" / "libamd_comgr.so").write_bytes(b"\x7fELF")
    (pkg / "lib" / "amd_comgr0702.dll").write_bytes(b"MZ")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv(pc.COMGR_LIB_ENV, raising=False)
    monkeypatch.delenv(pc.HIP_LIB_ENV, raising=False)
    monkeypatch.delenv(pc.NO_PIN_ENV, raising=False)
    pc.torch_package_dir.cache_clear()
    yield pkg
    pc.torch_package_dir.cache_clear()


def _lib(pkg) -> str:
    name = "amd_comgr0702.dll" if sys.platform == "win32" else "libamd_comgr.so"
    return os.path.join(str(pkg), "lib", name)


def test_finds_the_bundled_comgr_and_vintage_without_importing_torch(fake_torch):
    from rocke.runtime import comgr

    assert pc.torch_package_dir() == fake_torch
    assert pc.torch_bundled_comgr() == _lib(fake_torch)
    assert comgr._torch_package_rocm_version(_lib(fake_torch)) == (7, 2)
    assert "torch" not in sys.modules


def test_pin_exports_the_library_and_keeps_an_explicit_one(fake_torch):
    env = {}
    assert pc.pin_production_comgr(env) == _lib(fake_torch)
    assert env == {pc.COMGR_LIB_ENV: _lib(fake_torch)}
    env = {pc.COMGR_LIB_ENV: "/explicit/libamd_comgr.so"}
    assert pc.pin_production_comgr(env) == "/explicit/libamd_comgr.so"
    assert env[pc.COMGR_LIB_ENV] == "/explicit/libamd_comgr.so"
    env = {pc.NO_PIN_ENV: "1"}
    assert pc.pin_production_comgr(env) is None and pc.COMGR_LIB_ENV not in env
    (fake_torch / "lib" / "libamd_comgr.so").unlink()
    (fake_torch / "lib" / "amd_comgr0702.dll").unlink()
    env = {}
    assert pc.pin_production_comgr(env) is None and env == {}
    assert "torch" not in sys.modules


def test_hip_runtime_pin_pairs_with_the_bundled_comgr(fake_torch):
    """The HIP runtime beside the pinned torch comgr is pinned with it, so a
    torch-free process maps the comgr it compiles with (Linux only)."""
    hip = os.path.join(str(fake_torch), "lib", "libamdhip64.so")
    with open(hip, "wb") as fh:
        fh.write(b"ELF")
    env = {}
    pc.pin_production_comgr(env)
    got = pc.pin_production_hip(env)
    if sys.platform == "win32":
        assert got is None and pc.HIP_LIB_ENV not in env
        return
    assert got == hip and env[pc.HIP_LIB_ENV] == hip
    env = {pc.COMGR_LIB_ENV: _lib(fake_torch), pc.HIP_LIB_ENV: "/explicit/hip.so"}
    assert pc.pin_production_hip(env) == "/explicit/hip.so"  # explicit wins
    env = {pc.COMGR_LIB_ENV: "/elsewhere/libamd_comgr.so"}  # not torch's: no pair
    assert pc.pin_production_hip(env) is None and pc.HIP_LIB_ENV not in env
    env = {pc.COMGR_LIB_ENV: _lib(fake_torch), pc.NO_PIN_ENV: "1"}
    assert pc.pin_production_hip(env) is None and pc.HIP_LIB_ENV not in env
    os.unlink(hip)  # a torch package without a HIP runtime: nothing to pin
    env = {pc.COMGR_LIB_ENV: _lib(fake_torch)}
    assert pc.pin_production_hip(env) is None and pc.HIP_LIB_ENV not in env
    assert "torch" not in sys.modules


def test_comgr_cache_is_owned_by_the_process_unless_set(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(pc.tempfile, "tempdir", None)  # re-read TMPDIR
    pc.process_comgr_cache.cache_clear()
    try:
        want = pc.process_comgr_cache()
        assert pc.process_comgr_cache() == want  # one directory per process
    finally:
        pc.process_comgr_cache.cache_clear()
    assert os.path.dirname(want) == str(tmp_path) and os.path.isdir(want)
    assert os.path.basename(want).startswith("rocke-comgr-cache-")
    assert os.listdir(want) == []  # new: nothing another process compiled
    monkeypatch.setattr(pc, "process_comgr_cache", lambda: want)
    env = {}
    assert pc.pin_comgr_cache(env) == want and env == {pc.COMGR_CACHE_DIR_ENV: want}
    env = {pc.COMGR_CACHE_DIR_ENV: "/explicit"}  # explicit or inherited: kept
    assert pc.pin_comgr_cache(env) == "/explicit"
    assert env == {pc.COMGR_CACHE_DIR_ENV: "/explicit"}
    env = {pc.COMGR_CACHE_ENV: "0"}  # the cache is off: nothing to place
    assert pc.pin_comgr_cache(env) is None and env == {pc.COMGR_CACHE_ENV: "0"}
    env = {pc.NO_PIN_ENV: "1"}
    assert pc.pin_comgr_cache(env) is None and pc.COMGR_CACHE_DIR_ENV not in env


_TWO_PROCESSES = textwrap.dedent(
    """
    import os, sys
    from kernels.common import attention_bwd_comgr as pc
    pc.ensure_production_comgr()
    print(os.environ[pc.COMGR_CACHE_DIR_ENV], os.path.isdir(os.environ[pc.COMGR_CACHE_DIR_ENV]))
    """
)


def test_each_process_tree_gets_its_own_comgr_cache_removed_at_exit(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), TMPDIR=str(tmp_path))
    for k in (pc.COMGR_CACHE_DIR_ENV, pc.COMGR_CACHE_ENV, pc.NO_PIN_ENV):
        env.pop(k, None)
    seen = []
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-c", _TWO_PROCESSES],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.returncode == 0, out.stderr[-3000:]
        path, existed = out.stdout.split()[-2:]
        assert existed == "True"
        assert not os.path.exists(path)  # the owner removed it at exit
        seen.append(path)
    assert seen[0] != seen[1]
    env[pc.COMGR_CACHE_DIR_ENV] = str(tmp_path / "inherited")
    out = subprocess.run(
        [sys.executable, "-c", _TWO_PROCESSES],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split()[-2] == str(tmp_path / "inherited")  # a child keeps it


def test_torch_free_vintage_follows_the_pinned_torch_library(fake_torch, monkeypatch):
    """The platform dates a library in a torch package's ``lib`` directory by
    that package's ``version.py`` while torch is not imported, so the
    toolchain identity of a torch-free process names the pinned library's
    vintage (the LLVM flavor itself follows the loaded compiler)."""
    from rocke.runtime import comgr

    monkeypatch.setattr(comgr, "resolved_lib_path", lambda: _lib(fake_torch))
    assert comgr.resolved_lib_rocm_version() == (7, 2)
    (fake_torch / "version.py").write_text(_VERSION_PY.format(hip="7.13.0"))
    assert comgr.resolved_lib_rocm_version() == (7, 13)
    assert "torch" not in sys.modules


_COMPILE_ONLY = textwrap.dedent(
    """
    import sys

    from benchmarks.common import attention_bwd_compile as bc
    from kernels.common.attention_bwd import (
        AttnBwdSpec, build_attn_bwd_main, validate_attn_bwd_spec,
    )
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
    from tests import _attention_bwd_prebuild, _attention_bwd_resources as res
    from tests import run_device_shards

    import os

    from kernels.common import attention_bwd_comgr as pc

    tc = bc.toolchain_identity()
    assert os.environ.get("AMD_COMGR_CACHE_DIR") == pc.process_comgr_cache()
    assert os.environ.get("ROCKE_COMGR_LIB") == pc.torch_bundled_comgr()
    bc._worker_init({})  # a compile-pool worker's start-up
    spec = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dtype="bf16"), "gfx942")
    ir = _lower_kernel_to_llvm_python(build_attn_bwd_main(spec, arch="gfx942"),
                                      arch="gfx942")
    assert "define" in ir
    compiled = "nocomgr"
    if tc.comgr_path and tc.comgr_sha:
        cache = bc.CompileCache(sys.argv[1], toolchain=tc, source_hash="h")
        r = bc.compile_batch([("main", spec)], arch="gfx942", cache=cache,
                             mode="serial")
        if r[0].ok:
            k = res.read_code_object_resources(r[0].entry.hsaco, arch="gfx942")
            assert k.scratch_bytes >= 0
            res.comgr_version()
            compiled = "compiled"
        else:
            compiled = r[0].error
    print(compiled, "torch" in sys.modules)
    """
)


def test_compile_only_paths_do_not_import_torch(tmp_path):
    """Identity (which pins the library and the comgr cache in a clean
    environment), a compile worker's start-up, build, lower, comgr compile,
    resource read and the gate / pre-build / runner modules: no torch."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    for k in (
        pc.COMGR_LIB_ENV,
        pc.HIP_LIB_ENV,
        pc.COMGR_CACHE_DIR_ENV,
        pc.COMGR_CACHE_ENV,
        pc.NO_PIN_ENV,
    ):
        env.pop(k, None)
    out = subprocess.run(
        [sys.executable, "-c", _COMPILE_ONLY, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    status, torch_loaded = out.stdout.split()[-2:]
    assert torch_loaded == "False", out.stdout
    assert status in ("compiled", "nocomgr") or status.startswith("compile:"), status
