# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The comgr code-object cache of the attention backward processes, and the
torch-free compile-only paths.

CPU only.

* ``attention_bwd_comgr`` gives comgr's own cache a directory of the process
  tree (made by the first process, inherited by its children, removed at its
  exit) unless one is set or the cache is off.
* The library test session and the toolchain identity leave the compiler to
  rocKE's discovery: with a torch package importable they pin neither its
  comgr nor its HIP runtime. A fake torch package whose ``__init__`` raises
  stands in for the wheel.
* The build / lower / compile / identity / resource-gate path, the
  pre-build's compile-worker start-up and the runner leave torch out of
  ``sys.modules`` (a fresh process each).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from kernels.common import attention_bwd_comgr as pc

_COMPILER_ENV = ("ROCKE_COMGR_LIB", "ROCKE_HIP_LIB")


def _fake_torch_site(root: Path) -> Path:
    """A site directory holding a torch package with a bundled comgr and HIP
    runtime, which raises when imported."""
    lib = root / "site" / "torch" / "lib"
    lib.mkdir(parents=True)
    (lib.parent / "__init__.py").write_text(
        "raise RuntimeError('torch was imported')\n"
    )
    for name in ("libamd_comgr.so", "libamdhip64.so", "amd_comgr0702.dll"):
        (lib / name).write_bytes(b"\x7fELF")
    return root / "site"


def _clean_environment(*paths: Path) -> dict[str, str]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([*map(str, paths), *sys.path]))
    for name in (
        *_COMPILER_ENV,
        pc.COMGR_CACHE_DIR_ENV,
        pc.COMGR_CACHE_ENV,
        pc.NO_PIN_ENV,
    ):
        env.pop(name, None)
    return env


_LIBRARY_ROOT = Path(__file__).resolve().parents[1]
_ENVIRONMENT_PROBE = textwrap.dedent(
    """
    import json, os

    def pytest_sessionfinish(session):
        names = ("ROCKE_COMGR_LIB", "ROCKE_HIP_LIB", "AMD_COMGR_CACHE_DIR")
        with open(os.environ["PROBE_OUT"], "w") as out:
            json.dump({name: os.environ.get(name) for name in names}, out)
    """
)


def test_a_library_session_leaves_the_compiler_to_discovery(tmp_path):
    """The library conftest places comgr's cache and pins no compiler: a torch
    package on the path does not make the session load its comgr."""
    (tmp_path / "environment_probe.py").write_text(_ENVIRONMENT_PROBE)
    out = tmp_path / "environment.json"
    env = _clean_environment(_fake_torch_site(tmp_path), tmp_path)
    env["PROBE_OUT"] = str(out)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "environment_probe",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            str(Path(__file__)),
        ],
        cwd=_LIBRARY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    seen = json.loads(out.read_text())
    assert seen["ROCKE_COMGR_LIB"] is None and seen["ROCKE_HIP_LIB"] is None, seen
    assert seen["AMD_COMGR_CACHE_DIR"], seen


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
    pc.pin_comgr_cache()
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
            check=False,
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
        check=False,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split()[-2] == str(tmp_path / "inherited")  # a child keeps it


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
    assert "ROCKE_COMGR_LIB" not in os.environ and "ROCKE_HIP_LIB" not in os.environ
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
    """Identity (which places the comgr cache in a clean environment and pins
    no compiler, though a torch package is importable), a compile worker's
    start-up, build, lower, comgr compile, resource read and the gate /
    pre-build / runner modules: no torch."""
    env = _clean_environment(_fake_torch_site(tmp_path))
    out = subprocess.run(
        [sys.executable, "-c", _COMPILE_ONLY, str(tmp_path / "cache")],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    status, torch_loaded = out.stdout.split()[-2:]
    assert torch_loaded == "False", out.stdout
    assert status in ("compiled", "nocomgr") or status.startswith("compile:"), status
