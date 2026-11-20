# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU numeric verification of the addr64 (large-tensor forward-addressing)
primitives. Unlike the CPU-only golden-IR tests in
``platform/tests/core/test_make_tile_buffer_resource.py`` and
``test_addr64_m_decode.py`` (which pin the emitted LLVM text), this module
*launches* the primitives on a real GPU and checks their numeric result --
a microkernel reading from an origin > 4 GiB, and a numeric cross-check of
the M-decode pair against a plain-Python reference.

Requires a ROCm GPU; skips otherwise (see ``rocke.tests.gpu``).
"""

from __future__ import annotations

import ctypes
import struct

import pytest

from rocke.core.ir import I32, I64, IRBuilder, PtrType
from rocke.helpers.compile import compile_kernel
from rocke.helpers.tensor_view import (
    carry_decode_i32,
    make_tile_buffer_resource,
    tile_origin_decode_i64,
)
from rocke.runtime.hip_module import Runtime
from rocke.tests.gpu import probe_device_arch

pytestmark = pytest.mark.skipif(
    probe_device_arch()[0] is None, reason=probe_device_arch()[1]
)

_ARCH = probe_device_arch()[0] or "gfx950"


def _build_origin_probe_kernel():
    """``out[0] = buffer_load(make_tile_buffer_resource(A, origin, total))``."""
    b = IRBuilder("addr64_origin_probe")
    A = b.param("A", PtrType(I32, "global"))
    out = b.param("out", PtrType(I32, "global"))
    origin = b.param("origin", I64)
    total = b.param("total", I64)
    rsrc = make_tile_buffer_resource(b, A, origin_i64=origin, total_bytes_i64=total)
    val = b.buffer_load(rsrc.rsrc, b.const_i32(0), rsrc.soffset, I32)
    b.global_store(out, b.const_i32(0), val)
    b.ret()
    return b.kernel


def _launch_origin_probe(a_bytes: int, origin: int, total: int, *, write_at: int = 0):
    """Allocs a real device buffer of ``a_bytes``, writes ``0xCAFEBABE`` at
    byte offset ``write_at`` within it, launches the probe with ``origin``/
    ``total``, and returns the i32 the kernel read at voffset 0 (i.e. at
    device address ``A_ptr + origin``)."""
    kernel = _build_origin_probe_kernel()
    artifact = compile_kernel(kernel, arch=_ARCH)
    rt = Runtime()
    module = rt.load_module(artifact.hsaco)
    fn = module.get_function(artifact.kernel_name)

    a_ptr = rt.alloc(a_bytes)
    rt.memset(a_ptr, 0, a_bytes)
    marker = (ctypes.c_uint32 * 1)(0xCAFEBABE)
    rt.memcpy_h2d(a_ptr + write_at, marker, 4)
    out_ptr = rt.alloc(4)
    rt.memset(out_ptr, 0, 4)

    args = _pack_origin_probe_args(a_ptr, out_ptr, origin, total)
    rt.launch_blocking(fn, (1, 1, 1), (1, 1, 1), args)

    result = (ctypes.c_uint32 * 1)(0)
    rt.memcpy_d2h(result, out_ptr, 4)
    rt.free(a_ptr)
    rt.free(out_ptr)
    return result[0]


def _pack_origin_probe_args(a_ptr, out_ptr, origin, total):
    # Kernel param order: A (ptr), out (ptr), origin (i64), total (i64) --
    # all 8-byte aligned, so a flat little-endian pack needs no padding.
    return struct.pack("<QQqq", a_ptr, out_ptr, origin, total)


def test_in_bounds_read_at_a_small_origin():
    """origin=0, a real value at byte 0: the common case still reads through."""
    got = _launch_origin_probe(a_bytes=64, origin=0, total=64, write_at=0)
    assert got == 0xCAFEBABE


def test_in_bounds_read_at_an_origin_past_4_gib(request):
    """Plan section 5: a tile origin beyond 4 GiB that is still within the
    buffer's real extent reads the real value through -- not just a clamp to
    zero. Needs free VRAM comfortably past 4 GiB + 64 bytes; skip if short."""
    origin = (1 << 32) + 4096  # > 4 GiB
    a_bytes = origin + 64
    try:
        got = _launch_origin_probe(
            a_bytes=a_bytes, origin=origin, total=a_bytes, write_at=origin
        )
    except Exception as exc:  # noqa: BLE001 - hipMalloc OOM on a small GPU
        pytest.skip(f"could not allocate a >4 GiB buffer: {exc}")
    assert got == 0xCAFEBABE


def test_origin_past_4_gib_against_a_small_real_buffer_clamps_to_zero():
    """Plan section 5's clamp contract, proven on hardware (not just lowered
    IR text): an origin far beyond a small buffer's real extent makes
    ``num_records`` clamp to 0, so the hardware never dereferences the
    (unmapped) address -- the load returns 0 and the kernel does not fault."""
    origin = 5_000_000_000  # > 4 GiB, far past the 64-byte real allocation
    got = _launch_origin_probe(a_bytes=64, origin=origin, total=64, write_at=0)
    assert got == 0


def test_origin_exactly_at_total_clamps_to_zero_records():
    """remaining = total - origin == 0: still a valid (empty) buffer, not a
    negative-size one; the load returns 0."""
    got = _launch_origin_probe(a_bytes=64, origin=64, total=64, write_at=0)
    assert got == 0


# ---------------------------------------------------------------------
# M-decode pair: numeric cross-check against a plain-Python reference
# ---------------------------------------------------------------------


def _build_mdecode_kernel():
    """Decodes ``m0`` (i64, the tile origin) once, then ``d`` (i32, the
    per-lane delta) via the carry chain, and writes ``[n, ho, wo]`` (i32) to
    ``out``."""
    b = IRBuilder("addr64_mdecode_probe")
    out = b.param("out", PtrType(I32, "global"))
    m0 = b.param("m0", I64)
    ho_dim = b.param("Ho", I64)
    wo_dim = b.param("Wo", I64)
    d = b.param("d", I32)
    n0_i64, ho0_i64, wo0_i64 = tile_origin_decode_i64(b, m0, [ho_dim, wo_dim])
    n0 = b.trunc(n0_i64, I32)
    ho0 = b.trunc(ho0_i64, I32)
    wo0 = b.trunc(wo0_i64, I32)
    ho_dim32 = b.trunc(ho_dim, I32)
    wo_dim32 = b.trunc(wo_dim, I32)
    n, ho, wo = carry_decode_i32(b, d, [n0, ho0, wo0], [ho_dim32, wo_dim32])
    b.global_store(out, b.const_i32(0), n)
    b.global_store(out, b.const_i32(1), ho)
    b.global_store(out, b.const_i32(2), wo)
    b.ret()
    return b.kernel


def _launch_mdecode(m0: int, ho_dim: int, wo_dim: int, d: int):
    kernel = _build_mdecode_kernel()
    artifact = compile_kernel(kernel, arch=_ARCH)
    rt = Runtime()
    module = rt.load_module(artifact.hsaco)
    fn = module.get_function(artifact.kernel_name)

    out_ptr = rt.alloc(12)
    rt.memset(out_ptr, 0, 12)
    args = struct.pack("<Qqqqi", out_ptr, m0, ho_dim, wo_dim, d)
    rt.launch_blocking(fn, (1, 1, 1), (1, 1, 1), args)

    out = (ctypes.c_int32 * 3)()
    rt.memcpy_d2h(out, out_ptr, 12)
    rt.free(out_ptr)
    return tuple(out)


# ---------------------------------------------------------------------
# Epilogue addr64 bounds check: the exact formula added to
# DirectEpilogue._bounds_check / CShuffleEpilogue.store (helpers/epilogues.py)
# ---------------------------------------------------------------------


def _build_epilogue_bounds_probe():
    """Replicates ``_bounds_check``'s addr64 branch in isolation:
    ``ok = (origin + zext(m_local)) < M``. Writes 1/0 to ``out``."""
    b = IRBuilder("addr64_epilogue_bounds_probe")
    out = b.param("out", PtrType(I32, "global"))
    origin = b.param("origin", I64)
    m_local = b.param("m_local", I32)
    m_bound = b.param("M", I64)
    ok = b.cmp_lt(b.add(origin, b.zext(m_local, I64)), m_bound)
    result = b.select(ok, b.const_i32(1), b.const_i32(0))
    b.global_store(out, b.const_i32(0), result)
    b.ret()
    return b.kernel


def _launch_epilogue_bounds_probe(origin: int, m_local: int, m_bound: int) -> int:
    kernel = _build_epilogue_bounds_probe()
    artifact = compile_kernel(kernel, arch=_ARCH)
    rt = Runtime()
    module = rt.load_module(artifact.hsaco)
    fn = module.get_function(artifact.kernel_name)

    out_ptr = rt.alloc(4)
    rt.memset(out_ptr, 0, 4)
    # Kernel param order: out (ptr, @0), origin (i64, @8), m_local (i32, @16),
    # M (i64, needs 8-byte alignment -> 4 pad bytes, @24).
    args = struct.pack("<Qqi4xq", out_ptr, origin, m_local, m_bound)
    rt.launch_blocking(fn, (1, 1, 1), (1, 1, 1), args)

    result = (ctypes.c_int32 * 1)(0)
    rt.memcpy_d2h(result, out_ptr, 4)
    rt.free(out_ptr)
    return result[0]


@pytest.mark.parametrize(
    "origin,m_local,m_bound,want",
    [
        # Huge origin (> 4 GiB), small in-tile delta: a real row, in bounds.
        (6_000_000_000, 5, 6_000_000_100, 1),
        # Same huge origin, delta pushes past the bound: the tile's tail.
        (6_000_000_000, 150, 6_000_000_100, 0),
        # Origin exactly at the bound: every delta is out of bounds.
        (6_000_000_000, 0, 6_000_000_000, 0),
        # Small origin (the non-addr64-shaped case, still through the i64
        # branch): ordinary in-bounds / out-of-bounds rows.
        (0, 10, 128, 1),
        (0, 200, 128, 0),
    ],
)
def test_epilogue_addr64_bounds_check_matches_reference(origin, m_local, m_bound, want):
    got = _launch_epilogue_bounds_probe(origin, m_local, m_bound)
    assert got == want
    assert want == int(origin + m_local < m_bound)


@pytest.mark.parametrize(
    "n0_ref,ho0_ref,wo0_ref,ho_dim,wo_dim,d",
    [
        # The primary addr64 scenario: huge N, "normal"-sized Ho/Wo. n0 is
        # large enough that m0 = n0*Ho*Wo + ... comfortably exceeds i32, but
        # n0 itself still fits i32 (every realistic large-N case).
        (100_000_000, 0, 0, 14, 14, 0),
        (100_000_000, 3, 7, 14, 14, 0),
        (100_000_000, 13, 13, 14, 14, 1),  # carries wo -> ho -> n
        (100_000_000, 13, 13, 14, 14, 50),  # carries past a whole image too
        # Huge single-image spatial extent (Ho*Wo alone exceeds i32), n small.
        (1, 0, 0, 50_000, 50_000, 0),
        (1, 49_999, 49_998, 50_000, 50_000, 5),  # carries wo -> ho, no n carry
    ],
)
def test_mdecode_matches_python_reference(n0_ref, ho0_ref, wo0_ref, ho_dim, wo_dim, d):
    m0 = (n0_ref * ho_dim + ho0_ref) * wo_dim + wo0_ref
    n, ho, wo = _launch_mdecode(m0, ho_dim, wo_dim, d)

    m_abs = m0 + d
    want_wo = m_abs % wo_dim
    want_ho = (m_abs // wo_dim) % ho_dim
    want_n = m_abs // (ho_dim * wo_dim)

    assert (n, ho, wo) == (want_n, want_ho, want_wo)
