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
``N``-aligned. The walk runs on the Python engine's IR, so every check runs
without the C++ engine; where that engine is built, its IR must be
byte-identical and its refusals the same.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.ir import F16, F32, F64, FP8E4M3, I32, I64, IRBuilder, PtrType
from rocke.core.ir_serialize import parse
from rocke.core.lower_llvm import (
    LLVM_FLAVOR_LLVM20,
    _lower_kernel_to_llvm_python,
)
from rocke.tests.engines import (
    BACKEND_LANES,
    assert_backend_lane_rejects,
    assert_both_engines_reject,
    available_engines,
    lower_with_both_engines,
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


def _assert_engines_agree(kernel, python_ir: str) -> None:
    """Where the C++ engine is built, its IR must equal ``python_ir``."""
    if "cpp" in available_engines():
        assert lower_with_both_engines(kernel, _ARCH, LLVM_FLAVOR_LLVM20) == python_ir


def _assert_refused(kernel, error_type: type[Exception], message: str) -> None:
    """The Python engine refuses ``kernel`` with exactly ``error_type`` and
    ``message``; where the C++ engine is built, it refuses with the same text."""
    with pytest.raises(error_type) as refusal:
        _lower(kernel)
    assert str(refusal.value) == message
    if "cpp" in available_engines():
        assert_both_engines_reject(
            kernel, _ARCH, error_type, message, llvm_flavor=LLVM_FLAVOR_LLVM20
        )


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
    assert lower_with_both_engines(kernel, _ARCH, LLVM_FLAVOR_LLVM20) == _lower(kernel)


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


def _with_store_align(align: object):
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
    one (16): both engines refuse the store with the same message. (The
    engine binding reports every lowering failure as RuntimeError; the C API
    test in test_smem_access_alignment.cpp checks the C++ status is
    ROCKE_ERR_VALUE.)"""
    message = (
        f"tile.smem_store_vN failed - align={align}, expected a positive power of two"
    )
    _assert_refused(_with_store_align(align), ValueError, message)


@pytest.mark.parametrize(
    ("align", "kind"), [(True, "bool"), (24.0, "float"), (8.0, "float")]
)
def test_a_store_alignment_that_is_not_an_integer_is_refused(align, kind):
    """A bool or float ``align`` is the wrong kind of value: both engines
    refuse it with the same message, rather than read it as an integer
    (``True`` as 1, ``8.0`` as 8) or treat it as missing. Python raises
    TypeError; the C API test in test_smem_access_alignment.cpp checks the C++
    status is ROCKE_ERR_TYPE (the engine binding reports it as RuntimeError)."""
    message = f"tile.smem_store_vN failed - align={kind}, expected an integer"
    _assert_refused(_with_store_align(align), TypeError, message)


@pytest.mark.parametrize("lane", BACKEND_LANES)
@pytest.mark.parametrize(
    ("align", "error_type", "message"),
    [
        (
            24,
            ValueError,
            "tile.smem_store_vN failed - align=24, expected a positive power of two",
        ),
        (
            True,
            TypeError,
            "tile.smem_store_vN failed - align=bool, expected an integer",
        ),
    ],
)
def test_every_backend_lane_reports_an_invalid_store_alignment(
    monkeypatch, lane, align, error_type, message
):
    """Through ``lower_kernel_to_llvm``, each backend lane reports the refusal
    as its caller should see it: the C++ engine's error in the strict C++
    lane, the Python engine's exact exception everywhere else."""
    assert_backend_lane_rejects(
        monkeypatch,
        lane,
        _with_store_align(align),
        _ARCH,
        error_type,
        message,
        llvm_flavor=LLVM_FLAVOR_LLVM20,
    )


@pytest.mark.parametrize("align", [32, 2**32, 2**62])
def test_a_valid_store_alignment_wider_than_the_pool_is_still_capped(align):
    """Any power of two above the pool alignment is capped to 16, including
    values too wide for a 32-bit integer: the cap must apply before the value
    is narrowed, or 2**32 would wrap to 0 and emit ``align 0``."""
    kernel = _with_store_align(align)
    ll = _lower(kernel)
    assert re.search(r"store <8 x float>.*, align 16$", ll, re.MULTILINE)
    _assert_engines_agree(kernel, ll)


def _load_then_store(vector: str, vec: int) -> str:
    """Serialized IR that loads ``vec<{vector}>`` (``vec`` elements) from an
    LDS buffer and stores it to global memory -- a shape serialized or
    hand-written IR may carry even where the builder admits no such width."""
    elem = vector.split("x")[0]
    return f"""ckdsl.ir v1
kernel @lds_odd_width {{
  params {{
    %out : ptr<{elem},global>
  }}
  region @entry {{
    %c1 : i32 = arith.constant (  ) {{ ity = s:"i32", value = i:0 }}
    %buf : smem<{elem}, [16]> = tile.smem_alloc (  )
    %v : vec<{vector}> = tile.smem_load_vN ( %buf, %c1 ) {{ elem_type = s:"{elem}", rank = i:1, vec = i:{vec} }}
    memref.global_store_vN ( %out, %c1, %v ) {{ align = i:1, elem_type = s:"{elem}", vec = i:{vec} }}
    cf.return (  )
  }}
}}
"""


@pytest.mark.parametrize(
    ("vector", "vec", "llvm_vector", "claim"),
    [
        ("f32x3", 3, "<3 x float>", 4),
        ("f16x6", 6, "<6 x half>", 2),
        ("fp8e4m3x12", 12, "<12 x i8>", 1),
        # Widths no builder emits: a 6-byte and a 3-byte load still claim only
        # their element width, not a fraction of 12 bytes.
        ("f16x3", 3, "<3 x half>", 2),
        ("i8x3", 3, "<3 x i8>", 1),
        ("f32x6", 6, "<6 x float>", 4),
    ],
)
def test_a_load_whose_width_is_not_a_power_of_two_claims_its_element_width(
    vector, vec, llvm_vector, claim
):
    """A load whose byte width is not a power of two guarantees only element
    alignment, whatever its element type; both engines emit the same IR."""
    kernel = parse(_load_then_store(vector, vec))
    ll = _lower(kernel)
    assert re.search(
        rf"load {re.escape(llvm_vector)}, ptr addrspace\(3\) \S+, align {claim}$",
        ll,
        re.MULTILINE,
    )
    _assert_engines_agree(kernel, ll)


def test_a_distributed_store_claims_its_element_width():
    """smem_store_distributed writes one element at a time, so each store
    claims the stored element's width (4 bytes for f32), not a fixed 2."""
    b = IRBuilder("lds_align_distributed_f32")
    out = b.param("out", PtrType(F32, "global"))
    zero = b.const_i32(0)
    half_buf = b.smem_alloc(F16, [1], name_hint="h")
    float_buf = b.smem_alloc(F32, [4], name_hint="f")
    h = b.trunc_f32_to_f16(b.const_f32(1.0))
    b.smem_store_f16(half_buf, [zero], h)
    values = b.global_load_vN(out, zero, F32, 4, align=16)
    b.smem_store_distributed(float_buf, {}, values)
    reloaded = b.smem_load_vN(float_buf, zero, dtype=F32, n=4)
    b.global_store_vN(out, zero, reloaded, 4)
    b.smem_store_f16(half_buf, [zero], h)
    b.ret()
    ll = _lower(b.kernel)
    claims = re.findall(
        r"store float \S+, ptr addrspace\(3\) \S+, align (\d+)$", ll, re.MULTILINE
    )
    assert claims == ["4"] * 4
    assert _alignment_violations(ll) == []
    _assert_engines_agree(b.kernel, ll)


def test_an_eight_byte_element_segment_is_aligned_to_its_width():
    """A segment of 8-byte elements is 8-byte aligned even when no typed
    access claims it (here it is only address-taken), for i64 as for f64."""
    b = IRBuilder("lds_align_i64_address_taken")
    out = b.param("out", PtrType(I32, "global"))
    zero = b.const_i32(0)
    half_buf = b.smem_alloc(F16, [1], name_hint="h")
    wide_buf = b.smem_alloc(I64, [2], name_hint="w")
    h = b.trunc_f32_to_f16(b.const_f32(1.0))
    b.smem_store_f16(half_buf, [zero], h)
    address = b.trunc(b.smem_addr_of(wide_buf), I32)
    b.global_store_vN(out, zero, address, 1)
    b.smem_store_f16(half_buf, [zero], h)
    b.ret()
    ll = _lower(b.kernel)
    segs = [m.group(3) for m in map(_SEG_RE.match, ll.splitlines()) if m]
    assert segs == ["8"]
    _assert_engines_agree(b.kernel, ll)


def _f16_then_one_f32_vector_store(lanes: int):
    """A one-element f16 buffer, then an f32 buffer written by one
    ``smem_store_vN(..., n=1)`` of a ``<lanes x float>`` value. ``n=1`` routes
    the store through the whole-value ``tile.smem_store``."""
    b = IRBuilder(f"lds_align_f16_one_f32x{lanes}")
    out = b.param("out", PtrType(F32, "global"))
    zero = b.const_i32(0)
    half_buf = b.smem_alloc(F16, [1], name_hint="h")
    float_buf = b.smem_alloc(F32, [12], name_hint="f")
    h = b.trunc_f32_to_f16(b.const_f32(1.0))
    b.smem_store_f16(half_buf, [zero], h)
    vector = b.global_load_vN(out, zero, F32, lanes, align=4)
    b.smem_store_vN(float_buf, [zero], vector, 1)
    b.smem_store_f16(half_buf, [zero], h)
    b.ret()
    return b.kernel


@pytest.mark.parametrize(("lanes", "claim"), [(3, 4), (4, 16)])
def test_a_whole_value_store_claims_the_vector_access_alignment(lanes, claim):
    """A whole ``<4 x float>`` stored as one value claims its 16-byte width and
    moves its segment to byte 16; a 12-byte ``<3 x float>`` claims only its
    4-byte element (``align 12`` is not valid LLVM) and its segment stays at
    byte 4. Both engines emit the same IR."""
    kernel = _f16_then_one_f32_vector_store(lanes)
    ll = _lower(kernel)
    assert re.search(
        rf"store <{lanes} x float> \S+, ptr addrspace\(3\) \S+, align {claim}$",
        ll,
        re.MULTILINE,
    )
    segs = [m.group(3) for m in map(_SEG_RE.match, ll.splitlines()) if m]
    assert segs == [str(claim)]
    assert _alignment_violations(ll) == []
    _assert_engines_agree(kernel, ll)
