# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every LDS alignment claim in the emitted LLVM IR must be true.

All smem allocations of a kernel live in one 16-byte aligned pool global, each
at a byte offset chosen by the pool layout. An access through a segment emits
``align N``; LLVM treats a false alignment claim as undefined behaviour. These
tests build kernels that mix element types and vector widths with several
buffers live at once (so segments sit at non-zero offsets), lower them, and
walk the IR: for every addrspace(3) load or store, ``N`` must divide the
access's byte offset from the pool base, and the pool must be at least
``N``-aligned. The walk runs on the Python engine's IR; a separate test
requires the C++ engine's IR to be byte-identical when that engine is built.
"""

from __future__ import annotations

import re

import pytest

from rocke.core.ir import F16, F32, F64, FP8E4M3, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import (
    LLVM_FLAVOR_LLVM20,
    _lower_kernel_to_llvm_python,
)

_ARCH = "gfx942"

_POOL_RE = re.compile(
    r"^(@smem_pool\.\S+) = internal unnamed_addr addrspace\(3\) global "
    r"\[\d+ x i8\] poison, align (\d+)$"
)
_SEG_RE = re.compile(
    r"^\s*(%\S+) = getelementptr inbounds i8, ptr addrspace\(3\) (@\S+), i32 (\d+)$"
)
_GEP_RE = re.compile(
    r"^\s*(%\S+) = getelementptr inbounds (.+?), ptr addrspace\(3\) (\S+), (.*)$"
)
_ACCESS_RE = re.compile(r"ptr addrspace\(3\) (%\S+|@\S+), align (\d+)")

_SCALAR_BYTES = {"i8": 1, "half": 2, "bfloat": 2, "float": 4, "i32": 4, "double": 8}


def _array_type(ty: str) -> tuple[list[int], int]:
    """``[4 x [8 x half]]`` -> ([4, 8], 2): dimensions and element bytes."""
    dims = [int(d) for d in re.findall(r"\[(\d+) x ", ty)]
    elem = ty.replace("[", " ").replace("]", " ").split()[-1]
    return dims, _SCALAR_BYTES[elem]


def _gep_offset(ty: str, indices: str) -> int:
    """Byte offset of a typed GEP whose indices are all constants."""
    idx = [int(i) for i in re.findall(r"i32 (-?\d+)", indices)]
    assert len(idx) == indices.count("i32"), f"non-constant GEP index: {indices}"
    dims, elem_bytes = _array_type(ty)
    strides = []
    stride = elem_bytes
    for d in reversed(dims):
        strides.append(stride)
        stride *= d
    strides.append(stride)  # the leading index steps over the whole array
    strides.reverse()
    return sum(i * s for i, s in zip(idx, strides))


def _alignment_violations(ll: str) -> list[str]:
    """Each addrspace(3) access whose ``align N`` its address does not honour."""
    pool_align = 0
    offset: dict[str, int] = {}  # pointer name -> byte offset from the pool
    bad: list[str] = []
    for line in ll.splitlines():
        m = _POOL_RE.match(line)
        if m:
            offset[m.group(1)] = 0
            pool_align = int(m.group(2))
            continue
        m = _SEG_RE.match(line)
        if m:
            offset[m.group(1)] = offset[m.group(2)] + int(m.group(3))
            continue
        m = _GEP_RE.match(line)
        if m:
            name, ty, base, indices = m.groups()
            offset[name] = offset[base] + _gep_offset(ty, indices)
            continue
        m = _ACCESS_RE.search(line)
        if m:
            ptr, align = m.group(1), int(m.group(2))
            if offset[ptr] % align or pool_align % align:
                bad.append(
                    f"{line.strip()}  (byte offset {offset[ptr]}, pool align "
                    f"{pool_align})"
                )
    return bad


def _lower(kernel) -> str:
    """Lower with the Python engine (always available, so every check runs)."""
    return _lower_kernel_to_llvm_python(
        kernel, arch=_ARCH, llvm_flavor=LLVM_FLAVOR_LLVM20
    )


def _lower_cpp(kernel) -> str:
    """Lower with the C++ engine, skipping when it is not built."""
    from rocke.core.backend import BackendError, _lower_via_cpp_engine

    try:
        return _lower_via_cpp_engine(serialize(kernel), _ARCH, LLVM_FLAVOR_LLVM20)
    except BackendError as e:
        pytest.skip(f"C++ engine not importable: {str(e)[:200]}")


def _f16_then_f32x4():
    """A one-element f16 buffer, then an f32 buffer written as <4 x float>.

    The f16 store happens before and after the vector traffic, so both
    buffers are live together and the f32 one cannot share offset 0.
    """
    b = IRBuilder("lds_align_f16_f32x4")
    out = b.param("out", PtrType(F32, "global"))
    zero = b.const_i32(0)
    half_buf = b.smem_alloc(F16, [1], name_hint="h")
    float_buf = b.smem_alloc(F32, [8], name_hint="f")
    h = b.trunc_f32_to_f16(b.const_f32(1.0))
    b.smem_store_f16(half_buf, [zero], h)
    v = b.global_load_vN(out, zero, F32, 4, align=16)
    b.smem_store_vN(float_buf, [b.const_i32(4)], v, 4)
    w = b.smem_load_vN(float_buf, zero, dtype=F32, n=4)
    b.global_store_vN(out, zero, w, 4)
    b.smem_store_f16(half_buf, [zero], h)
    b.ret()
    return b.kernel


def _f16_then_f64x2():
    """A one-element f16 buffer, then an f64 buffer written as <2 x double>."""
    b = IRBuilder("lds_align_f16_f64x2")
    out = b.param("out", PtrType(F64, "global"))
    zero = b.const_i32(0)
    half_buf = b.smem_alloc(F16, [1], name_hint="h")
    double_buf = b.smem_alloc(F64, [4], name_hint="d")
    h = b.trunc_f32_to_f16(b.const_f32(1.0))
    b.smem_store_f16(half_buf, [zero], h)
    v = b.global_load_vN(out, zero, F64, 2, align=16)
    b.smem_store_vN(double_buf, [b.const_i32(2)], v, 2)
    w = b.smem_load_vN(double_buf, zero, dtype=F64, n=2)
    b.global_store_vN(out, zero, w, 2)
    b.smem_store_f16(half_buf, [zero], h)
    b.ret()
    return b.kernel


def _several_live_buffers():
    """Odd-sized f16 / f32 / fp8 buffers, all live, each read with a vector.

    Covers 8-wide f16, the f32 epilogue ops, and 8-wide fp8 loads, with
    buffer sizes that are not multiples of 16 bytes so the element rule
    alone would leave later segments under-aligned.
    """
    b = IRBuilder("lds_align_several_live")
    out = b.param("out", PtrType(F32, "global"))
    zero = b.const_i32(0)
    small = b.smem_alloc(F16, [3], name_hint="s")
    halves = b.smem_alloc(F16, [2, 8], name_hint="hv")
    floats = b.smem_alloc(F32, [5, 4], name_hint="fv")
    bytes_ = b.smem_alloc(FP8E4M3, [24], name_hint="bv")
    h = b.trunc_f32_to_f16(b.const_f32(1.0))
    b.smem_store_f16(small, [b.const_i32(2)], h)
    hv = b.smem_load_vN(halves, zero, zero, dtype=F16, n=8)
    b.smem_store_vN(halves, [b.const_i32(1), zero], hv, 8)
    fv = b.smem_load_vN_f32(floats, b.const_i32(1), zero, n=4)
    b.smem_store_vN_f32(floats, [b.const_i32(3), zero], fv, 4)
    bv = b.smem_load_vN(bytes_, b.const_i32(8), dtype=FP8E4M3, n=8)
    b.smem_store_vN(bytes_, [b.const_i32(16)], bv, 8)
    b.global_store_vN(out, zero, fv, 4)
    b.smem_store_f16(small, [zero], h)
    b.ret()
    return b.kernel


def _f16_then_f32x8():
    """A one-element f16 buffer, then an f32 buffer moved as <8 x float>.

    An 8-wide f32 access is 32 bytes, wider than the 16-byte pool, so it can
    claim only the pool alignment.
    """
    b = IRBuilder("lds_align_f16_f32x8")
    out = b.param("out", PtrType(F32, "global"))
    zero = b.const_i32(0)
    half_buf = b.smem_alloc(F16, [1], name_hint="h")
    float_buf = b.smem_alloc(F32, [16], name_hint="f")
    h = b.trunc_f32_to_f16(b.const_f32(1.0))
    b.smem_store_f16(half_buf, [zero], h)
    v = b.global_load_vN(out, zero, F32, 8, align=32)
    b.smem_store_vN(float_buf, [b.const_i32(8)], v, 8)
    w = b.smem_load_vN(float_buf, zero, dtype=F32, n=8)
    b.global_store_vN(out, zero, w, 8)
    b.smem_store_f16(half_buf, [zero], h)
    b.ret()
    return b.kernel


_KERNELS = [_f16_then_f32x4, _f16_then_f64x2, _several_live_buffers, _f16_then_f32x8]


@pytest.mark.parametrize("build", _KERNELS)
def test_every_lds_alignment_claim_holds(build):
    ll = _lower(build())
    assert "@smem_pool." in ll
    # The kernels place at least one segment at a non-zero offset, otherwise
    # the check below would only ever see offset 0.
    assert any(_SEG_RE.match(line) for line in ll.splitlines())
    assert _alignment_violations(ll) == []


@pytest.mark.parametrize("build", _KERNELS)
def test_both_engines_emit_the_same_ir(build):
    kernel = build()
    assert _lower_cpp(kernel) == _lower(kernel)


def test_vector_access_moves_its_segment_to_the_vector_alignment():
    """The f32 segment after a 2-byte f16 buffer lands at 16, not 4."""
    ll = _lower(_f16_then_f32x4())
    segs = [m.group(3) for m in map(_SEG_RE.match, ll.splitlines()) if m]
    assert segs and set(segs) == {"16"}
    assert "store <4 x float>" in ll and "align 16" in ll


def test_eight_wide_fp8_load_claims_eight_bytes():
    """An 8 x fp8 load is 8 bytes wide, so it claims 8-byte alignment."""
    ll = _lower(_several_live_buffers())
    assert re.search(
        r"load <8 x i8>, ptr addrspace\(3\) \S+, align 8$", ll, re.MULTILINE
    )


def test_access_wider_than_the_pool_claims_the_pool_alignment():
    """A 32-byte <8 x float> load or store claims 16 bytes, not 32.

    The pool is 16-byte aligned, so no segment can promise more; the claim
    is capped rather than the kernel refused (the attention backward kernel
    with the slices dK / dV sink reads its partials this way).
    """
    ll = _lower(_f16_then_f32x8())
    accesses = re.findall(
        r"(load|store) <8 x float>.*ptr addrspace\(3\) \S+, align (\d+)$",
        ll,
        re.MULTILINE,
    )
    assert sorted(accesses) == [("load", "16"), ("store", "16")]
    segs = [m.group(3) for m in map(_SEG_RE.match, ll.splitlines()) if m]
    assert segs == ["16"]


def _with_store_align(align: int):
    """The <8 x float> store kernel, with its store's ``align`` attribute set
    to ``align`` -- as serialized or hand-written IR may carry it."""
    kernel = _f16_then_f32x8()
    stores = [op for op in kernel.body.ops if op.name == "tile.smem_store_vN"]
    assert len(stores) == 1
    stores[0].attrs["align"] = align
    return kernel


@pytest.mark.parametrize("align", [24, 12, 0, -16])
def test_a_store_alignment_that_is_not_a_power_of_two_is_refused(align):
    """The pool cap must not turn an invalid claim (24 bytes) into a valid
    one (16): both engines refuse the store with the same message."""
    message = f"tile.smem_store_vN: align {align} is not a positive power of two"
    with pytest.raises(ValueError, match=re.escape(message)):
        _lower(_with_store_align(align))
    with pytest.raises(Exception, match=re.escape(message)):
        _lower_cpp(_with_store_align(align))


def test_a_valid_store_alignment_wider_than_the_pool_is_still_capped():
    ll = _lower(_with_store_align(32))
    assert re.search(r"store <8 x float>.*, align 16$", ll, re.MULTILINE)
    assert ll == _lower_cpp(_with_store_align(32))
