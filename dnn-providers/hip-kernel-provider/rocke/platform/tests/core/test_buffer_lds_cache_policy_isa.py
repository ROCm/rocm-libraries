# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Buffer cache-policy bits: policy names, and ``coherency=`` on
``async_buffer_load_lds_addr`` reaching the instruction.

On gfx942 / gfx950 the bits are SC0 (1) and NT (2), so CACHE_STREAM emits
``nt`` only and NON_TEMPORAL emits ``sc0 nt``.

No GPU: the ISA test uses COMGR + ``llvm-objdump`` and skips when either is
missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from rocke.core.ir import (
    CACHE_ALL,
    CACHE_GLOBAL,
    CACHE_POLICY_NAMES,
    CACHE_STREAM,
    F16,
    I32,
    NON_TEMPORAL,
    IRBuilder,
    PtrType,
    cache_policy_from_name,
)
from rocke.helpers.compile import compile_kernel

_EXPECTED_BITS = {
    CACHE_ALL: set(),
    CACHE_GLOBAL: {"sc0"},
    CACHE_STREAM: {"nt"},
    NON_TEMPORAL: {"sc0", "nt"},
}


def test_cache_policy_from_name():
    assert [cache_policy_from_name(n) for n in CACHE_POLICY_NAMES] == [
        CACHE_ALL,
        CACHE_GLOBAL,
        CACHE_STREAM,
        NON_TEMPORAL,
    ]
    with pytest.raises(ValueError, match="cache policy must be one of"):
        cache_policy_from_name("none")


def _objdump() -> str | None:
    for candidate in (
        os.environ.get("LLVM_OBJDUMP"),
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
        shutil.which("llvm-objdump"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _buffer_lds_kernel(coherency: int):
    b = IRBuilder("buffer_lds_policy")
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(F16, "global"))
    n = b.param("N_bytes", I32)
    lds = b.smem_alloc(F16, [64, 8], name_hint="stage")
    b.async_buffer_load_lds_addr(
        b.buffer_rsrc(x, n),
        b.smem_addr_of(lds),
        b.const_i32(0),
        b.const_i32(0),
        dwords=1,  # 4 B: the LDS-DMA size both archs implement (16 B is gfx950-only)
        coherency=coherency,
    )
    b.ret()
    return b.kernel


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("coherency", sorted(_EXPECTED_BITS))
def test_buffer_load_lds_cache_policy_bits(arch, coherency):
    from rocke.runtime import comgr

    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as e:
        pytest.skip(f"COMGR not loadable: {str(e)[:200]}")
    hsaco = compile_kernel(_buffer_lds_kernel(coherency), arch=arch).hsaco
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.hsaco"
        path.write_bytes(hsaco)
        isa = subprocess.run(
            [objdump, "-d", f"--mcpu={arch}", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    loads = [
        ln.split("//")[0].split()
        for ln in isa.splitlines()
        if "buffer_load_dword " in ln and " lds" in ln
    ]
    assert len(loads) == 1, loads
    assert set(loads[0]) & {"sc0", "sc1", "nt"} == _EXPECTED_BITS[coherency], loads[0]


def _buffer_load_vn_kernel(coherency: int | None):
    """One dwordx4 ``buffer_load_vN`` stored back; ``None`` omits the keyword."""
    from rocke.core.ir import VectorType

    b = IRBuilder("buffer_load_vn_policy")
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(I32, "global"))
    y = b.param("Y", PtrType(I32, "global"))
    n = b.param("N_bytes", I32)
    rsrc = b.buffer_rsrc(x, n)
    voff, c0 = b.const_i32(0), b.const_i32(0)
    kw = {} if coherency is None else {"coherency": coherency}
    v = b.buffer_load_vN(rsrc, voff, c0, I32, 4, **kw)
    assert v.type == VectorType(I32, 4)
    b.buffer_store_vN(b.buffer_rsrc(y, n), voff, c0, v)
    b.ret()
    return b.kernel


def _load_ops(kernel):
    return [op for op in kernel.body.ops if op.name == "tile.buffer_load_vN"]


@pytest.mark.parametrize("coherency", sorted(_EXPECTED_BITS))
def test_buffer_load_vn_coherency_attr_and_lowerings(coherency):
    """``coherency=`` on ``buffer_load_vN``: an ``aux`` attribute only when
    nonzero (0 leaves the IR as before), passed as the load intrinsic's aux
    operand by both Python lowerings."""
    from rocke.core.ir_serialize import serialize
    from rocke.core.lower_hip import lower_kernel_to_hip
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    k = _buffer_load_vn_kernel(coherency)
    (op,) = _load_ops(k)
    assert op.attrs.get("aux", 0) == coherency and ("aux" in op.attrs) == bool(coherency)
    if coherency == 0:
        assert serialize(k) == serialize(_buffer_load_vn_kernel(None))
    llvm = [
        ln
        for ln in _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22").splitlines()
        if "call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(" in ln
    ]
    assert len(llvm) == 1 and llvm[0].rstrip().endswith(f"i32 {coherency})"), llvm
    hip = [ln for ln in lower_kernel_to_hip(k, arch="gfx950").splitlines() if "__builtin_amdgcn_raw_buffer_load_b128(" in ln]
    assert len(hip) == 1 and hip[0].rstrip().endswith(f", {coherency});"), hip


@pytest.mark.parametrize("bad", [-1, 4, 16])
def test_buffer_load_vn_refuses_other_coherency(bad):
    with pytest.raises(ValueError, match="coherency must be 0..3"):
        _buffer_load_vn_kernel(bad)


@pytest.mark.parametrize("coherency", sorted(_EXPECTED_BITS))
def test_buffer_load_vn_cache_policy_bits(coherency):
    """``coherency=`` on ``buffer_load_vN`` reaching ``buffer_load_dwordx4``
    on gfx950 (LLVM through COMGR)."""
    from rocke.runtime import comgr

    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as e:
        pytest.skip(f"COMGR not loadable: {str(e)[:200]}")
    hsaco = compile_kernel(_buffer_load_vn_kernel(coherency), arch="gfx950").hsaco
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.hsaco"
        path.write_bytes(hsaco)
        isa = subprocess.run(
            [objdump, "-d", "--mcpu=gfx950", str(path)], check=True, capture_output=True, text=True
        ).stdout
    loads = [ln.split("//")[0].split() for ln in isa.splitlines() if "buffer_load_dwordx4" in ln]
    assert len(loads) == 1, loads
    assert set(loads[0]) & {"sc0", "sc1", "nt"} == _EXPECTED_BITS[coherency], loads[0]
