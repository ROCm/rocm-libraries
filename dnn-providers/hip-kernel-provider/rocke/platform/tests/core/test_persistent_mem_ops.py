# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python-engine-only memory ops for persistent kernels.

``int_to_ptr``, ``buffer_load_cp`` / ``buffer_store_cp`` (explicit cache-policy
bits), ``global_load_atomic`` / ``global_store_atomic`` and the
``tagged_poll`` leaf op (see ``rocke.core.ir.PYTHON_ONLY_OPCODES``).

CPU: builder validation, the LLVM text each op lowers to, serialization
round-trip, and the C++-engine refusal under ``ROCKE_CPP_STRICT=1`` (when the
binding is built). GPU (gfx942 / gfx950, skipped elsewhere): numerics through
every op, the cache bits in the final ISA, and a cross-CTA tagged poll.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from rocke.core.ir import (
    CPOL_NT,
    CPOL_SC0,
    CPOL_SC1,
    I32,
    I64,
    PYTHON_ONLY_OPCODES,
    IRBuilder,
    PtrType,
    VectorType,
    uses_python_only_ops,
)
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import lower_kernel_to_llvm

FLAVOR = "llvm22"


def _lower(k):
    """Lower through the Python engine (these opcodes have no C++ mirror)."""
    old = os.environ.get("ROCKE_BACKEND")
    os.environ["ROCKE_BACKEND"] = "python"
    try:
        return lower_kernel_to_llvm(k, arch="gfx950", llvm_flavor=FLAVOR)
    finally:
        if old is None:
            del os.environ["ROCKE_BACKEND"]
        else:
            os.environ["ROCKE_BACKEND"] = old


# ----------------------------------------------------------------- kernels


def _int_to_ptr_kernel():
    """dst[tid] = src[tid] + 1, with both pointers rebuilt from i64 params."""
    b = IRBuilder("pmo_int_to_ptr")
    s = b.param("src_addr", I64)
    d = b.param("dst_addr", I64)
    src = b.int_to_ptr(s, PtrType(I32, "global"))
    dst = b.int_to_ptr(d, PtrType(I32, "global"))
    tid = b.thread_id_x()
    b.global_store(dst, tid, b.add(b.global_load_i32(src, tid), b.const_i32(1)))
    b.ret()
    return b.kernel


def _buffer_cp_kernel(dwords: int, aux: int):
    """dst <- src through buffer_load_cp / buffer_store_cp of ``dwords``."""
    b = IRBuilder(f"pmo_bufcp_{dwords}_{aux}")
    src = b.param("src", PtrType(I32, "global"))
    dst = b.param("dst", PtrType(I32, "global"))
    n = b.const_i32(64 * 4 * dwords)
    rs, rd = b.buffer_rsrc(src, n), b.buffer_rsrc(dst, n)
    off = b.mul(b.thread_id_x(), b.const_i32(4 * dwords))
    v = b.buffer_load_cp(rs, off, b.const_i32(0), dwords=dwords, aux=aux)
    b.buffer_store_cp(v, rd, off, b.const_i32(0), aux=aux)
    b.ret()
    return b.kernel


def _atomic_kernel(scope: str):
    """dst[tid] = src[tid] (i64) through atomic monotonic load/store."""
    b = IRBuilder(f"pmo_atomic_{scope}")
    src = b.param("src", PtrType(I64, "global"))
    dst = b.param("dst", PtrType(I64, "global"))
    tid = b.thread_id_x()
    v = b.global_load_atomic(src, tid, scope=scope)
    b.global_store_atomic(dst, tid, v, scope=scope)
    b.ret()
    return b.kernel


def _poll_kernel(dwords: int, pause: str = "s_nop 0"):
    """CTA 0 sleeps, then writes the pairs; CTA 1 polls them with
    tagged_poll and copies the payloads out. ``tag`` is a param."""
    b = IRBuilder(f"pmo_poll_{dwords}")
    box = b.param("box", PtrType(I32, "global"))
    out = b.param("out", PtrType(I32, "global"))
    tag = b.readfirstlane(b.param("tag", I32))
    rs = b.buffer_rsrc(box, b.const_i32(4096))
    tid = b.thread_id_x()
    bid = b.block_id_x()
    off = b.mul(tid, b.const_i32(4 * dwords))
    with b.scf_if(b.cmp_eq(bid, b.const_i32(0))):
        b.inline_asm("s_sleep 127\ns_sleep 127\ns_sleep 127", "", [], None)
        words = []
        for q in range(dwords // 2):
            words += [b.add(b.mul(tid, b.const_i32(10)), b.const_i32(q)), tag]
        b.buffer_store_cp(b.vec_pack(words, I32), rs, off, b.const_i32(0), aux=CPOL_SC1)
    with b.scf_if(b.cmp_eq(bid, b.const_i32(1))):
        (v,) = b.tagged_poll(tag, [(rs, off, dwords)], pause=pause)
        for q in range(dwords // 2):
            idx = b.add(b.mul(tid, b.const_i32(dwords // 2)), b.const_i32(q))
            b.global_store(out, idx, b.vec_extract(v, 2 * q))
    b.ret()
    return b.kernel


# --------------------------------------------------------------------- CPU


def test_python_only_opcodes_are_detected():
    assert uses_python_only_ops(_int_to_ptr_kernel()) == ["tile.int_to_ptr"]
    assert uses_python_only_ops(_poll_kernel(2)) == [
        "tile.buffer_store_cp",
        "tile.tagged_poll",
    ]
    assert {
        "tile.int_to_ptr",
        "tile.buffer_load_cp",
        "tile.buffer_store_cp",
        "tile.global_load_atomic",
        "tile.global_store_atomic",
        "tile.tagged_poll",
        # gfx950 MX compute ops (mxcompute track).
        "tile.mfma_scale_mx",
        "arith.cvt_scalef32_pk_fp8_f32",
        "arith.cvt_scalef32_pk_bf8_f32",
        # gfx950 MX decode GEMV (head track) and natural exp: no C++ lowering.
        "arith.cvt_scalef32_pk_bf16_fp4",
        "tile.permlane16_swap",
        "math.exp",
    } == set(PYTHON_ONLY_OPCODES)


def test_builder_validation():
    b = IRBuilder("v")
    p = b.param("p", PtrType(I32, "global"))
    with pytest.raises(TypeError):
        b.int_to_ptr(b.const_i32(0), PtrType(I32, "global"))
    with pytest.raises(TypeError):
        b.int_to_ptr(b.const_i64(0), PtrType(I32, "lds"))
    r = b.buffer_rsrc(p, b.const_i32(64))
    with pytest.raises(ValueError):
        b.buffer_load_cp(r, b.const_i32(0), b.const_i32(0), dwords=3)
    with pytest.raises(ValueError):
        b.buffer_load_cp(r, b.const_i32(0), b.const_i32(0), dwords=1, aux=32)
    with pytest.raises(TypeError):
        b.buffer_store_cp(b.const_i64(0), r, b.const_i32(0), b.const_i32(0))
    with pytest.raises(TypeError):
        b.global_load_atomic(p, b.const_i32(0), dtype=VectorType(I32, 2))
    with pytest.raises(ValueError):
        b.global_load_atomic(p, b.const_i32(0), scope="cluster")
    with pytest.raises(ValueError):
        b.global_store_atomic(p, b.const_i32(0), b.const_i32(1), ordering="acquire")
    with pytest.raises(ValueError):
        b.tagged_poll(b.const_i32(1), [])
    with pytest.raises(ValueError):
        b.tagged_poll(b.const_i32(1), [(r, b.const_i32(0), 1)])
    with pytest.raises(TypeError):
        b.tagged_poll(b.const_i64(1), [(r, b.const_i32(0), 2)])


def test_int_to_ptr_lowering():
    ll = _lower(_int_to_ptr_kernel())
    assert re.search(r"= inttoptr i64 %\S+ to ptr addrspace\(1\)", ll)


@pytest.mark.parametrize("dwords", [1, 2, 4])
@pytest.mark.parametrize("aux", [0, CPOL_SC1, CPOL_SC0 | CPOL_SC1, CPOL_NT])
def test_buffer_cp_lowering_carries_aux(dwords, aux):
    ll = _lower(_buffer_cp_kernel(dwords, aux))
    sfx = "i32" if dwords == 1 else f"v{dwords}i32"
    assert re.search(
        rf"call .+ @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.{sfx}\(.*, i32 {aux}\)", ll
    )
    assert re.search(
        rf"call void @llvm\.amdgcn\.raw\.ptr\.buffer\.store\.{sfx}\(.*, i32 {aux}\)", ll
    )


@pytest.mark.parametrize("scope", ["agent", "system", "workgroup"])
def test_atomic_lowering(scope):
    ll = _lower(_atomic_kernel(scope))
    ss = "" if scope == "system" else f' syncscope("{scope}")'
    assert "load atomic i64, ptr addrspace(1) %agep" in ll
    assert f"{ss} monotonic, align 8" in ll
    assert re.search(r"store atomic i64 %\S+, ptr addrspace\(1\) %agep", ll)


def test_tagged_poll_lowering_is_a_self_contained_loop():
    ll = _lower(_poll_kernel(4, pause="s_sleep 1"))
    loop = re.search(r"^(poll\.loop\.\d+):", ll, re.MULTILINE).group(1)
    retry = re.search(r"^(poll\.retry\.\d+):", ll, re.MULTILINE).group(1)
    done = re.search(r"^(poll\.done\.\d+):", ll, re.MULTILINE).group(1)
    assert f"br label %{loop}" in ll
    assert re.search(rf"br i1 %\S+, label %{retry}, label %{done}", ll)
    assert 'call void asm sideeffect "s_sleep 1", ""()' in ll
    # Both tag words (1 and 3) of the 4-dword load are compared.
    assert ll.count("icmp ne i32 %poll.tag") == 2
    assert re.search(r"buffer\.load\.v4i32\(.*, i32 0, i32 16\)", ll)


@pytest.mark.parametrize(
    "build",
    [
        _int_to_ptr_kernel,
        lambda: _buffer_cp_kernel(4, CPOL_SC1),
        lambda: _atomic_kernel("agent"),
        lambda: _poll_kernel(2),
        lambda: _poll_kernel(4),
    ],
)
def test_serialize_round_trip(build):
    k = build()
    text = serialize(k)
    k2 = parse(text)
    assert serialize(k2) == text
    assert _lower(k2) == _lower(k)


def test_cpp_engine_refuses_under_strict(monkeypatch):
    if importlib.util.find_spec("rocke_engine") is None:
        pytest.skip("cpp engine needs the rocke_engine binding")
    monkeypatch.setenv("ROCKE_CPP_STRICT", "1")
    monkeypatch.setenv("ROCKE_BACKEND", "cpp")
    with pytest.raises(Exception, match="tile.tagged_poll|tile.buffer_store_cp"):
        lower_kernel_to_llvm(_poll_kernel(2), arch="gfx950", llvm_flavor=FLAVOR)


# --------------------------------------------------------------------- GPU


@pytest.fixture(scope="module")
def arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        target = get_device_arch(0)
    except Exception as e:  # noqa: BLE001 -- no HIP runtime at all
        pytest.skip(f"no HIP device: {str(e)[:120]}")
    if target not in ("gfx942", "gfx950"):
        pytest.skip(f"requires a gfx942 or gfx950 device; detected {target}")
    return target


def _isa(hsaco: bytes, arch: str) -> str:
    objdump = None
    for c in (
        os.environ.get("LLVM_OBJDUMP"),
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
        shutil.which("llvm-objdump"),
    ):
        if c and os.path.isfile(c):
            objdump = c
            break
    if objdump is None:
        pytest.skip("llvm-objdump not found")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "k.hsaco"
        p.write_bytes(hsaco)
        return subprocess.run(
            [objdump, "-d", f"--mcpu={arch}", str(p)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout


class _Dev:
    """Tiny torch-free device harness over ``rocke.runtime.hip_module``."""

    def __init__(self):
        from rocke.runtime.hip_module import Runtime

        self.rt = Runtime()
        self.bufs = []

    def put(self, arr: np.ndarray) -> int:
        p = self.rt.alloc(arr.nbytes)
        self.bufs.append(p)
        host = (ctypes.c_uint8 * arr.nbytes).from_buffer_copy(arr.tobytes())
        self.rt.memcpy_h2d(p, host, arr.nbytes)
        return p

    def zeros(self, nbytes: int) -> int:
        p = self.rt.alloc(nbytes)
        self.bufs.append(p)
        self.rt.memset(p, 0, nbytes)
        return p

    def get(self, p: int, nbytes: int, dtype) -> np.ndarray:
        out = (ctypes.c_uint8 * nbytes)()
        self.rt.memcpy_d2h(out, p, nbytes)
        return np.frombuffer(bytes(out), dtype=dtype).copy()

    def run(self, art, grid, block, args: bytes):
        mod = self.rt.load_module(art.hsaco)
        try:
            self.rt.launch(mod.get_function(art.kernel_name), grid, block, args)
            self.rt.sync()
        finally:
            mod.unload()

    def close(self):
        for p in self.bufs:
            self.rt.free(p)


@pytest.fixture()
def dev(arch):
    d = _Dev()
    yield d
    d.close()


def _compile(k, arch):
    from rocke.helpers.compile import compile_kernel

    return compile_kernel(k, arch=arch, backend="python")


def test_gpu_int_to_ptr(arch, dev):
    src = np.arange(64, dtype=np.int32) * 3
    ps, pd = dev.put(src), dev.zeros(256)
    dev.run(
        _compile(_int_to_ptr_kernel(), arch),
        (1, 1, 1),
        (64, 1, 1),
        struct.pack("<QQ", ps, pd),
    )
    np.testing.assert_array_equal(dev.get(pd, 256, np.int32), src + 1)


_BITS = {
    0: set(),
    CPOL_SC1: {"sc1"},
    CPOL_SC0 | CPOL_SC1: {"sc0", "sc1"},
    CPOL_NT: {"nt"},
}


@pytest.mark.parametrize("dwords", [1, 2, 4])
@pytest.mark.parametrize("aux", sorted(_BITS))
def test_gpu_buffer_cp(arch, dev, dwords, aux):
    n = 64 * dwords
    src = np.arange(n, dtype=np.int32) * 7 + 1
    ps, pd = dev.put(src), dev.zeros(4 * n)
    art = _compile(_buffer_cp_kernel(dwords, aux), arch)
    dev.run(art, (1, 1, 1), (64, 1, 1), struct.pack("<QQ", ps, pd))
    np.testing.assert_array_equal(dev.get(pd, 4 * n, np.int32), src)
    isa = _isa(art.hsaco, arch)
    w = "" if dwords == 1 else f"x{dwords}"
    for kind in ("load", "store"):
        lines = [
            ln.split("//")[0].replace(",", " ").split()
            for ln in isa.splitlines()
            if f"buffer_{kind}_dword{w} " in ln
        ]
        assert len(lines) == 1, (kind, lines)
        assert set(lines[0]) & {"sc0", "sc1", "nt"} == _BITS[aux], lines[0]


@pytest.mark.parametrize("scope,bits", [("agent", {"sc1"}), ("system", {"sc0", "sc1"})])
def test_gpu_atomic_pair(arch, dev, scope, bits):
    src = (np.arange(64, dtype=np.int64) << 32) | 0x1234
    ps, pd = dev.put(src), dev.zeros(512)
    art = _compile(_atomic_kernel(scope), arch)
    dev.run(art, (1, 1, 1), (64, 1, 1), struct.pack("<QQ", ps, pd))
    np.testing.assert_array_equal(dev.get(pd, 512, np.int64), src)
    isa = _isa(art.hsaco, arch)
    for kind in ("load", "store"):
        lines = [
            ln.split("//")[0].replace(",", " ").split()
            for ln in isa.splitlines()
            if f"global_{kind}_dwordx2 " in ln
        ]
        assert len(lines) == 1, (kind, lines)
        assert set(lines[0]) & {"sc0", "sc1", "nt"} == bits, lines[0]


@pytest.mark.parametrize("dwords", [2, 4])
def test_gpu_tagged_poll_cross_cta(arch, dev, dwords):
    tag = 0x5A5A0001
    box = dev.zeros(4096)
    out = dev.zeros(64 * 4 * (dwords // 2))
    art = _compile(_poll_kernel(dwords), arch)
    dev.run(art, (2, 1, 1), (64, 1, 1), struct.pack("<QQi", box, out, tag))
    got = dev.get(out, 64 * 4 * (dwords // 2), np.int32).reshape(64, dwords // 2)
    want = np.arange(64)[:, None] * 10 + np.arange(dwords // 2)[None, :]
    np.testing.assert_array_equal(got, want)
    isa = _isa(art.hsaco, arch)
    polls = [ln for ln in isa.splitlines() if f"buffer_load_dwordx{dwords} " in ln]
    assert polls and all(" sc1" in ln for ln in polls), polls
