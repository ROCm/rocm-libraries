# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A Fragment holds its registers in the storage type of its dtype.

The fragment's ``dtype`` is the logical number format an author names; its ``value`` holds the
registers in ``storage_type``, derived from the dtype and never supplied:

- every arithmetic dtype (f16, bf16, f32, f64, i8, i16, i32, fp8, bf8) is its own logical vector;
- tf32 has no arithmetic, so its registers are held as i32;
- a sub-byte dtype (fp4, fp6, i4) has no value type, so its registers are packed into whole i32
  words, low bit first.

The memory verbs move storage units: a packed fragment moves whole 32-bit words, so a layout or
window that would split a word is refused, and its LDS buffer holds i32 words. A runtime origin
or bound on a packed tensor's stride-1 axis is refused unless the caller states it is on a
whole-word group (``packed_runs_aligned=True``). ``storage_slots`` is the one rule for which storage
registers hold which logical registers. These tests check the storage table and that rule, the
refusals and their messages, the alignment each packed access claims, and that each engine lowers these
kernels on the CPU, the C++ engine to the Python engine's IR, and on a gfx90a GPU that tf32
payloads and packed operand tiles (in gfx950's canonical operand layouts) round-trip bit-exactly
through global memory and LDS, and that the packed words a lane loads are exactly
``FragmentPacking.pack`` of the elements its layout places there.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

pytest.importorskip("rocke.core.ir", reason="rocke IR substrate required")

from rocke.core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    F64,
    FP4E2M1,
    FP6E2M3,
    FP6E3M2,
    FP8E4M3,
    I4,
    I8,
    I16,
    I32,
    TF32,
    U8,
    IRBuilder,
    PtrType,
    VectorType,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.storage import BitPacking, register_packing, storage_group
from rocke.helpers.tiling import (
    Fragment,
    TileMma,
    fill_fragment,
    lds_allocation,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
    transform_fragment,
)
from rocke.helpers.tiling.emit import emit_tensor_coordinates
from rocke.helpers.tiling.fragments import storage_slots
from rocke.helpers.tiling.lds_conflict import NumBuilder, addr_map
from rocke.tests.engines import ENGINES, lower_with_engine
from rocke.tests.gpu import launch_with_buffers

_WAVE = 64


def _flat_tile(registers: int):
    """A 1-D tile whose 64 lanes each hold `registers` consecutive elements."""
    return make_tile_desc(
        shape=[_WAVE * registers],
        thread_tile=[registers],
        thread_dist=[_WAVE],
        wave_size=_WAVE,
    )


def _operand_desc(dtype):
    """The canonical A-operand layout of gfx950's 16x16x128 f8f6f4 instruction for a 4-bit or 6-bit
    operand: each lane holds one run of 32 K elements."""
    return TileMma((16, 16, 128), a=dtype, b=dtype, c=F32, target="gfx950").a_desc


# ---- the storage type ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,registers,storage",
    [
        (F16, 8, "vec<f16x8>"),
        (BF16, 8, "vec<bf16x8>"),
        (F32, 4, "vec<f32x4>"),
        (F64, 2, "vec<f64x2>"),
        (I8, 16, "vec<i8x16>"),
        (U8, 16, "vec<u8x16>"),
        (I16, 8, "vec<i16x8>"),
        (I32, 4, "vec<i32x4>"),
        (FP8E4M3, 16, "vec<fp8e4m3x16>"),
        (BF8E5M2, 16, "vec<bf8e5m2x16>"),
        (TF32, 8, "vec<i32x8>"),
        (FP4E2M1, 32, "vec<i32x4>"),
        (I4, 32, "vec<i32x4>"),
        (FP6E2M3, 32, "vec<i32x6>"),
        (FP6E3M2, 16, "vec<i32x3>"),
    ],
)
def test_storage_type_is_derived_from_the_dtype(dtype, registers, storage) -> None:
    fragment = make_fragment(_flat_tile(registers), dtype)
    assert fragment.dtype == dtype
    assert fragment.tile_desc.register_count == registers
    assert fragment.storage_type.name == storage


def test_a_packed_fragment_must_fill_whole_words() -> None:
    # 4 fp4 elements are 16 bits: half a register, which no lane can own alone.
    with pytest.raises(ValueError, match="not a whole number of 32-bit registers"):
        make_fragment(_flat_tile(4), FP4E2M1)


def test_a_tf32_fragment_takes_its_i32_registers() -> None:
    b = IRBuilder("tf32_value")
    value = b.zero_vec(I32, 8)
    fragment = Fragment(_flat_tile(8), TF32, value)
    assert fragment.value is value


def test_a_tf32_value_must_be_reinterpreted_to_its_storage_type() -> None:
    tile = make_tile_desc(
        shape=[64, 8], thread_tile=[1, 8], thread_dist=[64, 1], wave_size=_WAVE
    )
    b = IRBuilder("tf32_logical_value")
    ptr = b.param("IN", PtrType(TF32, "global"))
    logical = b.global_load_vN(ptr, b.const_i32(0), TF32, 8)
    expected = (
        "Fragment value expected vec<i32x8> for TileDesc(shape=(64, 8)), got vec<tf32x8> -- "
        "tf32 registers are held as their storage type i32; reinterpret them with "
        "b.bitcast(value, VectorType(I32, 8))"
    )
    with pytest.raises(ValueError) as raised:
        Fragment(tile, TF32, logical)
    assert str(raised.value) == expected
    # The fix the message names is accepted.
    Fragment(tile, TF32, b.bitcast(logical, VectorType(I32, 8)))


def test_a_packed_value_must_be_its_words() -> None:
    a_desc = _operand_desc(FP4E2M1)
    b = IRBuilder("fp4_bytes_value")
    expected = (
        "Fragment value expected vec<i32x4> for TileDesc(shape=(16, 128)), got vec<i8x32> -- "
        "fp4e2m1 registers are held packed, 32 elements of 4 bits in 4 i32 words; build them "
        "with load_fragment or fill_fragment"
    )
    with pytest.raises(ValueError) as raised:
        Fragment(a_desc, FP4E2M1, b.zero_vec(I8, 32))
    assert str(raised.value) == expected


@pytest.mark.parametrize(
    "dtype,start,length,slots",
    [
        (F16, 3, 2, range(3, 5)),  # one register per slot
        (TF32, 4, 4, range(4, 8)),  # held as i32, still one register per slot
        (FP4E2M1, 16, 8, range(2, 3)),  # 8 registers per word
        (FP4E2M1, 0, 32, range(4)),
        (FP6E2M3, 16, 16, range(3, 6)),  # 16 registers per 3 words
    ],
)
def test_storage_slots_name_the_registers_that_hold_a_range(
    dtype, start, length, slots
) -> None:
    assert storage_slots(dtype, 32, start, length) == slots


@pytest.mark.parametrize(
    "dtype,start,length,message",
    [
        (
            FP4E2M1,
            4,
            8,
            (
                "elements [4, 12) of a fragment packed at 4 bits per element do not start and end "
                "on a register boundary (32 bits) -- start and length must be multiples of 8 "
                "elements"
            ),
        ),
        (
            F16,
            30,
            4,
            (
                "storage_slots failed - registers [30, 34), expected a non-empty range "
                "inside the fragment's 32 registers"
            ),
        ),
        (
            F16,
            0,
            0,
            (
                "storage_slots failed - registers [0, 0), expected a non-empty range "
                "inside the fragment's 32 registers"
            ),
        ),
    ],
)
def test_storage_slots_refuse_a_range_that_is_not_whole_slots(
    dtype, start, length, message
) -> None:
    with pytest.raises(ValueError) as raised:
        storage_slots(dtype, 32, start, length)
    assert str(raised.value) == message


@pytest.mark.parametrize(
    "dtype,group",
    [
        (F16, (1, 1)),
        (TF32, (1, 1)),
        (FP8E4M3, (1, 1)),
        (FP4E2M1, (8, 1)),
        (I4, (8, 1)),
        (FP6E2M3, (16, 3)),
    ],
)
def test_storage_group_is_the_fewest_registers_filling_whole_slots(
    dtype, group
) -> None:
    registers, slots = storage_group(dtype.name)
    assert (registers, slots) == group
    # The group's registers are exactly the slots storage_slots names for them.
    assert len(storage_slots(dtype, 64, 0, registers)) == slots


@pytest.mark.parametrize("dtype,words", [(FP4E2M1, 4), (FP6E2M3, 6), (TF32, 32)])
def test_a_zero_fill_is_all_zero_storage_words(dtype, words) -> None:
    b = IRBuilder("zero_fill")
    fragment = make_fragment(_operand_desc(FP4E2M1), dtype)
    fill_fragment(b, fragment, 0)
    assert fragment.value.type == VectorType(I32, words)
    (fill,) = [op for op in b.kernel.body.ops if op.name == "arith.constant_vec"]
    assert fill.attrs["elem"] == "i32" and fill.attrs["fill"] == 0.0


# ---- u8: a byte fragment like any other ----------------------------------------------------------------

_U8_TILE = make_tile_desc(
    shape=[64, 16], thread_tile=[1, 16], thread_dist=[64, 1], wave_size=_WAVE
)
_U8_TENSOR = make_tensor_desc((64, 16), (16, 1), U8)


def test_a_u8_fragment_is_filled_with_u8_zeros() -> None:
    b = IRBuilder("u8_fill")
    fragment = make_fragment(_U8_TILE, U8)
    fill_fragment(b, fragment, 0)
    assert fragment.value.type == VectorType(U8, 16)
    (fill,) = [op for op in b.kernel.body.ops if op.name == "arith.constant_vec"]
    assert fill.attrs["elem"] == "u8" and fill.attrs["fill"] == 0.0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("dest", ["global", "lds"])
def test_a_u8_fragment_loads_from_global_and_stores(dest, engine) -> None:
    b = IRBuilder(f"u8_to_{dest}")
    tid = b.thread_id_x()
    src = b.param("IN", PtrType(U8, "global"))
    window = make_window(_U8_TENSOR, (0, 0))
    fragment = load_fragment(b, src, window, _U8_TILE, tid)
    assert fragment.storage_type == VectorType(U8, 16)
    if dest == "global":
        dst = b.param("OUT", PtrType(U8, "global"))
    else:
        dst = b.smem_alloc(U8, [_U8_TENSOR.element_span], name_hint="lds")
    store_fragment(b, dst, window, fragment, tid)
    b.ret()
    text = lower_with_engine(b.kernel, "gfx90a", engine=engine)
    # The registers start from a u8 zero vector, which LLVM spells as i8.
    assert "<16 x i8> zeroinitializer" in text


def _u8_round_trip_kernel():
    """A u8 tile loaded from IN, staged through LDS and stored to OUT."""
    from rocke.helpers.spec import SignatureBuilder

    b = IRBuilder("u8_round_trip")
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    src = _param(b, "IN", U8, readonly=True)
    out = _param(b, "OUT", U8, writeonly=True)
    tid = b.thread_id_x()
    lds = b.smem_alloc(U8, [_U8_TENSOR.element_span], name_hint="lds")
    window = make_window(_U8_TENSOR, (0, 0))
    fragment = load_fragment(b, src, window, _U8_TILE, tid)
    assert fragment.storage_type == VectorType(U8, 16)
    store_fragment(b, lds, window, fragment, tid)
    b.sync()
    staged = load_fragment(b, lds, window, _U8_TILE, tid)
    store_fragment(b, out, window, staged, tid)
    b.ret()
    signature = SignatureBuilder().ptr("IN", "u8").ptr("OUT", "u8").build()
    return b.kernel, signature


@pytest.mark.parametrize("engine", ENGINES)
def test_a_u8_round_trip_tile_lowers_with_each_engine(engine) -> None:
    kernel, _ = _u8_round_trip_kernel()
    text = lower_with_engine(kernel, "gfx90a", engine=engine)
    # The registers start from a u8 zero vector, which LLVM spells as i8.
    assert "<16 x i8> zeroinitializer" in text


# ---- refusals: a packed access moves whole words --------------------------------------------------------


def _lds_buffer(b, td):
    """An LDS buffer for `td`, allocated as lds_allocation says."""
    element, count = lds_allocation(td)
    return b.smem_alloc(element, [count], name_hint="lds")


def _load(td, tile, *, origin=None, bounds=None, lds_elem=None):
    b = IRBuilder("packed_load")
    tid = b.thread_id_x()
    if lds_elem is None:
        src = b.param("IN", PtrType(I8, "global"))
    else:
        src = b.smem_alloc(lds_elem, [td.element_span], name_hint="lds")
    origin = (0,) * td.rank if origin is None else origin
    return load_fragment(b, src, make_window(td, origin, bounds), tile, tid)


def test_a_k_major_strided_packed_tensor_is_refused() -> None:
    # The operand layout's K runs lie along axis 1, but this tensor is M-contiguous, so each lane's
    # registers would be single 4-bit elements 16 apart.
    td = make_tensor_desc((16, 128), (1, 16), FP4E2M1)
    with pytest.raises(ValueError) as raised:
        _load(td, _operand_desc(FP4E2M1))
    assert str(raised.value) == (
        "load_fragment of fp4e2m1 needs each lane's registers to form runs of whole 32-bit "
        "words along the tensor's stride-1 axis (multiples of 8 elements, starting on an "
        "8-element boundary); this layout gives runs of 1 element -- choose a layout whose "
        "innermost register run lies along the stride-1 axis"
    )


def test_runs_shorter_than_a_word_group_are_refused() -> None:
    # Runs of 4 fp6 elements are 24 bits; whole words start every 16 fp6 elements.
    tile = make_tile_desc(
        shape=[32, 32], thread_tile=[4, 4], thread_dist=[8, 8], wave_size=_WAVE
    )
    td = make_tensor_desc((32, 32), (32, 1), FP6E2M3)
    with pytest.raises(ValueError) as raised:
        _load(td, tile)
    assert str(raised.value) == (
        "load_fragment of fp6e2m3 needs each lane's registers to form runs of whole 32-bit "
        "words along the tensor's stride-1 axis (multiples of 16 elements, starting on a "
        "16-element boundary); this layout gives runs of 4 elements -- choose a layout whose "
        "innermost register run lies along the stride-1 axis"
    )


def test_a_row_stride_that_splits_a_word_is_refused() -> None:
    td = make_tensor_desc((16, 128), (132, 1), FP4E2M1)
    with pytest.raises(ValueError, match="tensor axis 0 steps by 132 elements"):
        _load(td, _operand_desc(FP4E2M1))


def test_an_origin_inside_a_word_is_refused() -> None:
    td = make_tensor_desc((16, 256), (256, 1), FP4E2M1)
    with pytest.raises(ValueError, match="the window origin on the stride-1 axis is 4"):
        _load(td, _operand_desc(FP4E2M1), origin=(0, 4))


@pytest.mark.parametrize("verb", ["load", "store"])
def test_a_clipped_packed_window_needs_a_whole_word_bound(verb) -> None:
    a_desc = _operand_desc(FP4E2M1)
    td = make_tensor_desc((16, 128), (128, 1), FP4E2M1)
    window = make_window(td, (0, 0), (None, 100))
    b = IRBuilder("packed_clip")
    tid = b.thread_id_x()
    ptr = b.param("P", PtrType(I8, "global"))
    with pytest.raises(ValueError) as raised:
        if verb == "load":
            load_fragment(b, ptr, window, a_desc, tid)
        else:
            fragment = make_fragment(a_desc, FP4E2M1)
            fill_fragment(b, fragment, 0)
            store_fragment(b, ptr, window, fragment, tid)
    assert str(raised.value) == (
        f"{verb}_fragment of a clipped fp4e2m1 window needs its bound along the stride-1 axis "
        "to be a multiple of 8 elements (whole 32-bit words) -- got 100"
    )


_RUNTIME_REFUSAL = (
    "{verb}_fragment: the window {what} on the stride-1 axis is a runtime value, which "
    "{verb}_fragment cannot check is a multiple of 8 elements (whole 32-bit words of fp4e2m1) -- "
    "pass packed_runs_aligned=True if it always is"
)


def _packed_verb(verb, b, ptr, window, tile, tid, **flags):
    if verb == "load":
        load_fragment(b, ptr, window, tile, tid, **flags)
    else:
        fragment = make_fragment(tile, FP4E2M1)
        fill_fragment(b, fragment, 0)
        store_fragment(b, ptr, window, fragment, tid, **flags)


@pytest.mark.parametrize("verb", ["load", "store"])
@pytest.mark.parametrize("space", ["global", "lds"])
def test_a_runtime_packed_origin_needs_its_alignment_stated(verb, space) -> None:
    # A runtime origin off a word group would be rounded down to the group: refused unless stated.
    a_desc = _operand_desc(FP4E2M1)
    td = make_tensor_desc((16, 256), (256, 1), FP4E2M1)
    b = IRBuilder("packed_runtime_origin")
    tid = b.thread_id_x()
    if space == "global":
        ptr = b.param("P", PtrType(I8, "global"))
    else:
        ptr = _lds_buffer(b, td)
    k0 = b.param("K0", I32)
    window = make_window(td, (0, k0))
    with pytest.raises(ValueError) as raised:
        _packed_verb(verb, b, ptr, window, a_desc, tid)
    assert str(raised.value) == _RUNTIME_REFUSAL.format(verb=verb, what="origin")
    _packed_verb(verb, b, ptr, window, a_desc, tid, packed_runs_aligned=True)


@pytest.mark.parametrize("verb", ["load", "store"])
def test_a_runtime_packed_bound_needs_its_alignment_stated(verb) -> None:
    # A runtime bound inside a word group would leave the rest of the group unmasked.
    a_desc = _operand_desc(FP4E2M1)
    td = make_tensor_desc((16, 128), (128, 1), FP4E2M1)
    b = IRBuilder("packed_runtime_bound")
    tid = b.thread_id_x()
    ptr = b.param("P", PtrType(I8, "global"))
    window = make_window(td, (0, 0), (None, b.param("KB", I32)))
    with pytest.raises(ValueError) as raised:
        _packed_verb(verb, b, ptr, window, a_desc, tid)
    assert str(raised.value) == _RUNTIME_REFUSAL.format(verb=verb, what="bound")
    _packed_verb(verb, b, ptr, window, a_desc, tid, packed_runs_aligned=True)


def test_a_float_constant_packed_origin_is_a_runtime_origin() -> None:
    # Only an integer constant is a compile-time origin; a float constant is checked like any
    # runtime value, so it is refused unless its alignment is stated.
    a_desc = _operand_desc(FP4E2M1)
    td = make_tensor_desc((16, 256), (256, 1), FP4E2M1)
    b = IRBuilder("packed_float_origin")
    tid = b.thread_id_x()
    ptr = b.param("P", PtrType(I8, "global"))
    window = make_window(td, (0, b.const_f32(8.0)))
    with pytest.raises(ValueError) as raised:
        _packed_verb("load", b, ptr, window, a_desc, tid)
    assert str(raised.value) == _RUNTIME_REFUSAL.format(verb="load", what="origin")


@pytest.mark.parametrize("verb", ["load", "store"])
@pytest.mark.parametrize("dtype", [F16, I8, TF32])
def test_packed_runs_aligned_is_refused_for_a_dtype_that_is_not_packed(
    verb, dtype
) -> None:
    # The flag states a fact about whole words of a sub-byte dtype; with any other dtype nothing
    # would read it, so the caller is told to drop it.
    tile = make_tile_desc(
        shape=[64, 16], thread_tile=[1, 16], thread_dist=[64, 1], wave_size=_WAVE
    )
    td = make_tensor_desc((64, 16), (16, 1), dtype)
    b = IRBuilder("not_packed_flag")
    tid = b.thread_id_x()
    ptr = b.param("P", PtrType(dtype, "global"))
    window = make_window(td, (0, 0))
    with pytest.raises(ValueError) as raised:
        if verb == "load":
            load_fragment(b, ptr, window, tile, tid, packed_runs_aligned=True)
        else:
            fragment = make_fragment(tile, dtype)
            fill_fragment(b, fragment, 0)
            store_fragment(b, ptr, window, fragment, tid, packed_runs_aligned=True)
    assert str(raised.value) == (
        f"{verb}_fragment failed - packed_runs_aligned=True with {dtype.name}, expected a packed "
        f"sub-byte dtype (fp4, fp6, i4); {dtype.name} runtime origins and bounds need no "
        "statement, so drop the argument"
    )


def _accesses(kernel, arch: str, space: int) -> list[str]:
    """Every distinct load and store of address space `space`, as ``"load <type>, align <n>"``."""
    ir = _lower_kernel_to_llvm_python(kernel, arch=arch)
    loads = re.findall(
        rf"= load ([^,]+), ptr addrspace\({space}\) [^,]+, align (\d+)", ir
    )
    stores = re.findall(
        rf"store (<[^>]+>|\S+) \S+, ptr addrspace\({space}\) [^,]+, align (\d+)", ir
    )
    return sorted(
        {f"load {t}, align {a}" for t, a in loads}
        | {f"store {t}, align {a}" for t, a in stores}
    )


@pytest.mark.parametrize(
    "origin,access",
    [(8, "i32, align 4"), (16, "<2 x i32>, align 8"), (32, "<4 x i32>, align 16")],
)
def test_a_packed_lds_access_is_no_wider_than_the_origin_alignment(
    origin, access
) -> None:
    # Each lane moves a 32-element fp4 run (4 words). The run starts `origin` elements into the
    # buffer row, so an access may be only as wide as the word alignment that start has.
    td = make_tensor_desc((64, 64), (64, 1), FP4E2M1)
    tile = make_tile_desc(
        shape=[64, 32], thread_tile=[1, 32], thread_dist=[64, 1], wave_size=_WAVE
    )
    b = IRBuilder("packed_lds_origin")
    tid = b.thread_id_x()
    lds = _lds_buffer(b, td)
    window = make_window(td, (0, origin))
    fragment = load_fragment(b, lds, window, tile, tid)
    store_fragment(b, lds, window, fragment, tid)
    b.ret()
    assert _accesses(b.kernel, "gfx942", 3) == [f"load {access}", f"store {access}"]


@pytest.mark.parametrize(
    "dtype,run,access",
    [(FP4E2M1, 32, "<4 x i32>, align 16"), (FP6E2M3, 16, "<3 x i32>, align 4")],
)
def test_a_packed_global_load_claims_the_alignment_of_its_start(
    dtype, run, access
) -> None:
    # A 4-word fp4 run starts on 16 bytes; a 3-word fp6 run (one 16-element group) only on 4.
    td = make_tensor_desc((64, run), (run, 1), dtype)
    tile = make_tile_desc(
        shape=[64, run], thread_tile=[1, run], thread_dist=[64, 1], wave_size=_WAVE
    )
    b = IRBuilder("packed_global_align")
    tid = b.thread_id_x()
    src = b.param("IN", PtrType(I8, "global"))
    load_fragment(b, src, make_window(td, (0, 0)), tile, tid)
    b.ret()
    assert _accesses(b.kernel, "gfx942", 1) == [f"load {access}"]


def test_a_packed_lds_buffer_must_hold_words() -> None:
    td = make_tensor_desc((16, 128), (128, 1), FP4E2M1)
    with pytest.raises(ValueError) as raised:
        _load(td, _operand_desc(FP4E2M1), lds_elem=F16)
    assert str(raised.value) == (
        "load_fragment: LDS buffer element f16, expected i32 (the storage element of fp4e2m1 "
        "data) -- allocate the buffer with lds_allocation"
    )


def test_a_tf32_lds_buffer_must_hold_i32() -> None:
    # tf32 is held as i32 in LDS as in registers, so a buffer of tf32 elements is refused.
    td = make_tensor_desc((_WAVE * 8,), (1,), TF32)
    with pytest.raises(ValueError) as raised:
        _load(td, _flat_tile(8), lds_elem=TF32)
    assert str(raised.value) == (
        "load_fragment: LDS buffer element tf32, expected i32 (the storage element of tf32 "
        "data) -- allocate the buffer with lds_allocation"
    )


def test_a_swizzled_packed_lds_access_is_refused() -> None:
    td = make_tensor_desc((16, 128), (128, 1), FP4E2M1)
    b = IRBuilder("packed_swizzle")
    tid = b.thread_id_x()
    lds = _lds_buffer(b, td)
    with pytest.raises(NotImplementedError) as raised:
        load_fragment(
            b,
            lds,
            make_window(td, (0, 0)),
            _operand_desc(FP4E2M1),
            tid,
            lds_swizzle=True,
        )
    assert str(raised.value) == (
        "load_fragment: an LDS swizzle of packed fp4e2m1 data -- a swizzle moves elements, and "
        "fp4e2m1 elements are addressed in whole 32-bit words -- TODO: not implemented yet"
    )


def test_a_packed_lds_buffer_is_sized_in_words() -> None:
    # 16 x 128 fp4 elements are 256 words: a buffer of 255 words is refused, one of 256 accepted.
    td = make_tensor_desc((16, 128), (128, 1), FP4E2M1)
    a_desc = _operand_desc(FP4E2M1)
    for words, ok in ((255, False), (256, True)):
        b = IRBuilder("packed_lds_size")
        tid = b.thread_id_x()
        lds = b.smem_alloc(I32, [words], name_hint="lds")
        if ok:
            load_fragment(b, lds, make_window(td, (0, 0)), a_desc, tid)
        else:
            with pytest.raises(
                ValueError,
                match=(
                    "spans 2048 fp4e2m1 elements, held in 256 i32 elements, but the buffer "
                    "holds 255"
                ),
            ):
                load_fragment(b, lds, make_window(td, (0, 0)), a_desc, tid)


class _I32Arithmetic:
    """Evaluates the index arithmetic the memory verbs emit the way the GPU computes it: in
    two's-complement i32, so a product past the i32 range wraps instead of growing."""

    @staticmethod
    def _wrap(value: int) -> int:
        return (value + (1 << 31)) % (1 << 32) - (1 << 31)

    def const_i32(self, value: int) -> int:
        return self._wrap(value)

    def mul(self, a: int, b: int) -> int:
        return self._wrap(a * b)

    def div(self, a: int, b: int) -> int:
        return self._wrap(int(a / b))


@pytest.mark.parametrize("dtype,bits", [(FP6E2M3, 6), (FP4E2M1, 4), (I4, 4)])
def test_a_packed_word_index_is_exact_for_every_i32_element_index(dtype, bits) -> None:
    # The word an access starts at, for the largest whole-group element index i32 can hold: the
    # element index is in range, so its word index must be too. A 6-bit index multiplied by 3
    # before the division by 16 would wrap here (the product is about 6.4e9).
    from rocke.helpers.tiling.emit import _unit_index

    group = 32 // math.gcd(bits, 32)
    element_index = (2**31 - 1) // group * group
    assert (
        _unit_index(_I32Arithmetic(), element_index, dtype)
        == element_index * bits // 32
    )


# ---- the LDS bank model ----------------------------------------------------------------------------------


def test_the_lds_bank_model_refuses_packed_data() -> None:
    # The bank model addresses whole elements of a byte or more; fp4 elements share 32-bit words.
    td = make_tensor_desc((16, 128), (128, 1), FP4E2M1)
    with pytest.raises(NotImplementedError) as raised:
        addr_map(
            _operand_desc(FP4E2M1), td, origin=(0, 0), n_lanes=_WAVE, lds_swizzle=False
        )
    assert str(raised.value) == (
        "addr_map: the LDS bank model addresses whole elements of a byte or more, and fp4e2m1 "
        "elements are packed into 32-bit words -- TODO: not implemented yet"
    )


# ---- GPU (gfx90a) ---------------------------------------------------------------------------------------


def _run(kernel, signature, buffers: dict, scalars: dict | None = None) -> None:
    """Compile `kernel` for gfx90a and launch one wave on the host arrays `buffers` (and `scalars`),
    copying every buffer back."""
    from rocke.helpers.compile import compile_kernel

    launch_with_buffers(
        compile_kernel(kernel, arch="gfx90a"),
        signature,
        buffers,
        grid=(1, 1, 1),
        block=(_WAVE, 1, 1),
        scalars=scalars,
    )


def _param(b, name, elem, **flags):
    return b.param(name, PtrType(elem, "global"), noalias=True, align=16, **flags)


def _tf32_round_trip_kernel():
    """A tf32 tile loaded from IN, staged transposed through LDS and stored to OUT."""
    from rocke.helpers.spec import SignatureBuilder

    rows, cols = 16, 32
    global_td = make_tensor_desc((rows, cols), (cols, 1), TF32)
    lds_td = make_tensor_desc((rows, cols), (1, rows), TF32)
    load_tile = make_tile_desc(
        shape=[rows, cols], thread_tile=[1, 8], thread_dist=[16, 4], wave_size=_WAVE
    )
    read_tile = make_tile_desc(
        shape=[rows, cols], thread_tile=[8, 1], thread_dist=[2, 32], wave_size=_WAVE
    )
    b = IRBuilder("tf32_round_trip")
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    src = _param(b, "IN", TF32, readonly=True)
    out = _param(b, "OUT", TF32, writeonly=True)
    tid = b.thread_id_x()
    lds = _lds_buffer(b, lds_td)
    window = make_window(global_td, (0, 0))
    fragment = load_fragment(b, src, window, load_tile, tid)
    assert fragment.storage_type == VectorType(I32, 8)
    store_fragment(b, lds, make_window(lds_td, (0, 0)), fragment, tid)
    b.sync()
    staged = load_fragment(b, lds, make_window(lds_td, (0, 0)), read_tile, tid)
    store_fragment(b, out, window, staged, tid)
    b.ret()
    signature = SignatureBuilder().ptr("IN", "tf32").ptr("OUT", "tf32").build()
    return b.kernel, signature, (rows, cols)


@pytest.mark.usefixtures("gfx90a_gpu")
def test_tf32_payload_round_trips_through_lds() -> None:
    # A tf32 tile is loaded from global memory with 4-element runs, stored into an LDS buffer that
    # holds it transposed, read back with runs along the other axis and stored to OUT. tf32 has no
    # arithmetic, so every 32-bit pattern, NaN payloads included, must come back unchanged.
    kernel, signature, (rows, cols) = _tf32_round_trip_kernel()
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2**32, size=(rows, cols), dtype=np.uint64).astype(np.uint32)
    result = np.zeros_like(data)
    _run(kernel, signature, {"IN": data, "OUT": result})
    np.testing.assert_array_equal(result, data)


def _packed_bytes(patterns: np.ndarray, bits: int) -> np.ndarray:
    """Row-major `patterns` packed low-bit-first at `bits` per element."""
    packed = BitPacking(bits).pack([int(p) for p in patterns.reshape(-1)])
    return np.frombuffer(packed, dtype=np.uint8).copy()


def _expected_words(tile, patterns: np.ndarray, dtype_name: str) -> np.ndarray:
    """Per lane, the words ``FragmentPacking.pack`` makes of the elements the layout gives it."""
    packing = register_packing(dtype_name, tile.register_count)
    words = []
    for lane in range(_WAVE):
        elements = [
            int(
                patterns[
                    tuple(
                        emit_tensor_coordinates(NumBuilder(lane), tile.layout, lane, r)
                    )
                ]
            )
            for r in range(tile.register_count)
        ]
        words.append(packing.pack(elements))
    return np.array(words, dtype=np.uint32)


_PACKED = [("fp4e2m1", FP4E2M1, 4), ("fp6e2m3", FP6E2M3, 6), ("i4", I4, 4)]


def _packed_round_trip_kernel(name, dtype, bits):
    """An A-operand tile of a packed dtype loaded from IN, its lanes' words dumped to WORDS, and the
    fragment stored to OUT directly and to OUT_LDS through an LDS buffer of i32 words.
    """
    from rocke.helpers.spec import SignatureBuilder

    a_desc = _operand_desc(FP4E2M1 if bits == 4 else dtype)
    rows, cols = a_desc.shape
    td = make_tensor_desc((rows, cols), (cols, 1), dtype)
    words = register_packing(name, a_desc.register_count).carrier_count

    b = IRBuilder(f"packed_round_trip_{name}")
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    src = _param(b, "IN", I8, readonly=True)
    out = _param(b, "OUT", I8, writeonly=True)
    out_lds = _param(b, "OUT_LDS", I8, writeonly=True)
    dump = _param(b, "WORDS", I32, writeonly=True)
    tid = b.thread_id_x()
    lds = _lds_buffer(b, td)
    window = make_window(td, (0, 0))
    fragment = load_fragment(b, src, window, a_desc, tid)
    assert fragment.storage_type == VectorType(I32, words)
    lane_base = b.mul(tid, b.const_i32(words))
    for j in range(words):
        b.global_store(
            dump,
            b.add(lane_base, b.const_i32(j)),
            b.vec_extract(fragment.value, j),
            align=4,
        )
    store_fragment(b, out, window, fragment, tid)
    store_fragment(b, lds, window, fragment, tid)
    b.sync()
    staged = load_fragment(b, lds, window, a_desc, tid)
    store_fragment(b, out_lds, window, staged, tid)
    b.ret()
    signature = (
        SignatureBuilder()
        .ptr("IN", "i8")
        .ptr("OUT", "i8")
        .ptr("OUT_LDS", "i8")
        .ptr("WORDS", "i32")
        .build()
    )
    return b.kernel, signature, a_desc, words


@pytest.mark.usefixtures("gfx90a_gpu")
def test_a_u8_tile_round_trips_through_lds() -> None:
    # Every byte value, the high bit included, must come back unchanged through global memory and
    # LDS: a u8 fragment is a byte vector, never sign-extended.
    kernel, signature = _u8_round_trip_kernel()
    data = (np.arange(64 * 16, dtype=np.uint32) * 37 % 256).astype(np.uint8)
    out = np.zeros_like(data)
    _run(kernel, signature, {"IN": data, "OUT": out})
    np.testing.assert_array_equal(out, data)


@pytest.mark.usefixtures("gfx90a_gpu")
@pytest.mark.parametrize("name,dtype,bits", _PACKED, ids=[p[0] for p in _PACKED])
def test_packed_operand_tile_round_trips(name, dtype, bits) -> None:
    # An A-operand tile (16 x 128, K contiguous) in gfx950's canonical f8f6f4 layout is loaded from a
    # packed tensor. WORDS receives every lane's registers, which must be the packing of the
    # elements the layout places there. The fragment is stored back to OUT directly, and to OUT_LDS
    # through an LDS buffer of i32 words; both must equal the input bytes.
    kernel, signature, a_desc, words = _packed_round_trip_kernel(name, dtype, bits)
    rows, cols = a_desc.shape
    rng = np.random.default_rng(bits)
    patterns = rng.integers(0, 1 << bits, size=(rows, cols))
    data = _packed_bytes(patterns, bits)
    result, result_lds = np.zeros_like(data), np.zeros_like(data)
    lane_words = np.zeros((_WAVE, words), np.uint32)
    _run(
        kernel,
        signature,
        {"IN": data, "OUT": result, "OUT_LDS": result_lds, "WORDS": lane_words},
    )
    np.testing.assert_array_equal(lane_words, _expected_words(a_desc, patterns, name))
    np.testing.assert_array_equal(result, data)
    np.testing.assert_array_equal(result_lds, data)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", [p[0] for p in _PACKED] + ["tf32"])
def test_storage_word_kernels_lower_with_each_engine(name, engine) -> None:
    # The kernels above move packed words (whole i32 words, <N x i32> vectors) and tf32 held as
    # i32; each engine must lower them, and the C++ engine to the same IR as the Python engine.
    if name == "tf32":
        kernel = _tf32_round_trip_kernel()[0]
    else:
        kernel = _packed_round_trip_kernel(*next(p for p in _PACKED if p[0] == name))[0]
    lower_with_engine(kernel, "gfx90a", engine=engine)


@pytest.mark.usefixtures("gfx90a_gpu")
@pytest.mark.parametrize(
    "runtime_bound", [False, True], ids=["compile_time", "runtime"]
)
@pytest.mark.parametrize(
    "name,dtype,bits", _PACKED[:2], ids=[p[0] for p in _PACKED[:2]]
)
def test_clipped_packed_window_masks_whole_words(
    name, dtype, bits, runtime_bound
) -> None:
    # The window is bounded at K = 64, a compile-time bound or a kernel argument stated to be on a
    # word group. Its load reads zero words past the bound (OUT_FULL stores the whole tile
    # unclipped), and its store drops the words past the bound (OUT keeps its 0xff fill there);
    # before the bound both equal the input.
    from rocke.helpers.spec import SignatureBuilder

    a_desc = _operand_desc(dtype)
    rows, cols = a_desc.shape
    bound = 64
    td = make_tensor_desc((rows, cols), (cols, 1), dtype)
    b = IRBuilder(f"packed_clip_{name}")
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    src = _param(b, "IN", I8, readonly=True)
    out_full = _param(b, "OUT_FULL", I8, writeonly=True)
    out = _param(b, "OUT", I8, writeonly=True)
    kb = b.param("KB", I32) if runtime_bound else bound
    tid = b.thread_id_x()
    clipped = make_window(td, (0, 0), (None, kb))
    fragment = load_fragment(
        b, src, clipped, a_desc, tid, packed_runs_aligned=runtime_bound
    )
    store_fragment(b, out_full, make_window(td, (0, 0)), fragment, tid)
    whole = load_fragment(b, src, make_window(td, (0, 0)), a_desc, tid)
    store_fragment(b, out, clipped, whole, tid, packed_runs_aligned=runtime_bound)
    b.ret()
    signature = (
        SignatureBuilder().ptr("IN", "i8").ptr("OUT_FULL", "i8").ptr("OUT", "i8")
    )
    if runtime_bound:
        signature = signature.scalar("KB", "i32")
    signature = signature.build()

    rng = np.random.default_rng(bits + 1)
    patterns = rng.integers(1, 1 << bits, size=(rows, cols))
    data = _packed_bytes(patterns, bits)
    result_full = np.full_like(data, 0xFF)
    result = np.full_like(data, 0xFF)
    _run(
        b.kernel,
        signature,
        {"IN": data, "OUT_FULL": result_full, "OUT": result},
        {"KB": bound} if runtime_bound else None,
    )
    row_bytes = cols * bits // 8
    kept = bound * bits // 8
    expected = data.reshape(rows, row_bytes)
    result_full = result_full.reshape(rows, row_bytes)
    result = result.reshape(rows, row_bytes)
    np.testing.assert_array_equal(result_full[:, :kept], expected[:, :kept])
    np.testing.assert_array_equal(result_full[:, kept:], 0)
    np.testing.assert_array_equal(result[:, :kept], expected[:, :kept])
    np.testing.assert_array_equal(result[:, kept:], 0xFF)


# ---- transform_fragment moves storage registers ------------------------------------------------

# A (2, 2, 512) tile whose 64 lanes each hold 2 x 2 x 8 registers: register 16 * i + 8 * j + r holds
# element (i, j, 8 * lane + r). Making j the most significant register axis instead moves registers
# 8-15 to 16-23 and back, whole runs of 8; making the run axis r the middle one splits every run.
_REORDER_TILE = make_tile_desc(
    shape=[2, 2, 8 * _WAVE],
    thread_tile=[2, 2, 8],
    thread_dist=[1, 1, _WAVE],
    wave_size=_WAVE,
)
_RUNS_SWAPPED = _REORDER_TILE.reorder_registers((1, 0, 2))
_RUNS_SPLIT = _REORDER_TILE.reorder_registers((0, 2, 1))


def _filled(b, dtype):
    """A zero-filled `_REORDER_TILE` fragment of `dtype`; a u8 one is the i8 zero reinterpreted."""
    if dtype == U8:
        zero = b.bitcast(b.zero_vec(I8, 32), VectorType(U8, 32))
        return Fragment(_REORDER_TILE, U8, zero)
    fragment = make_fragment(_REORDER_TILE, dtype)
    fill_fragment(b, fragment, 0)
    return fragment


@pytest.mark.parametrize(
    "dtype", [TF32, U8, FP4E2M1, FP6E2M3], ids=["tf32", "u8", "fp4e2m1", "fp6e2m3"]
)
def test_transform_fragment_keeps_the_storage_type(dtype) -> None:
    # The identity transform and, for every dtype whose runs of registers stay whole, the swap of
    # runs return a fragment of the source's storage type: i32 for tf32, u8 for u8, packed words for
    # the sub-byte dtypes.
    b = IRBuilder("transform_storage")
    fragment = _filled(b, dtype)
    targets = [_REORDER_TILE] + ([_RUNS_SWAPPED] if dtype != FP6E2M3 else [])
    for target in targets:
        moved = transform_fragment(b, fragment, target)
        assert (moved.dtype, moved.storage_type) == (dtype, fragment.storage_type)
        assert moved.value.type == fragment.storage_type


@pytest.mark.parametrize(
    "dtype,target,message",
    [
        (
            FP4E2M1,
            _RUNS_SPLIT,
            (
                "transform_fragment cannot move fp4e2m1 registers 0..7 to [0, 2, 4, 6, 8, 10, "
                "12, 14]: fp4e2m1 registers are held packed, so a reorder must move each run of "
                "8 registers (1 i32 word) whole and in order onto another such run -- TODO: not "
                "implemented yet; repacking the bits of a packed fragment"
            ),
        ),
        (
            FP6E2M3,
            _RUNS_SWAPPED,
            (
                "transform_fragment cannot move fp6e2m3 registers 0..15 to [0, 1, 2, 3, 4, 5, 6, "
                "7, 16, 17, 18, 19, 20, 21, 22, 23]: fp6e2m3 registers are held packed, so a "
                "reorder must move each run of 16 registers (3 i32 words) whole and in order onto "
                "another such run -- TODO: not implemented yet; repacking the bits of a packed "
                "fragment"
            ),
        ),
    ],
    ids=["fp4-split-runs", "fp6-half-runs"],
)
def test_a_packed_reorder_inside_a_word_group_is_refused(
    dtype, target, message
) -> None:
    b = IRBuilder("transform_packed_refusal")
    with pytest.raises(NotImplementedError) as excinfo:
        transform_fragment(b, _filled(b, dtype), target)
    assert str(excinfo.value) == message


_TRANSFORMED = [
    ("tf32", TF32, "tf32"),
    ("u8", U8, "i8"),
    ("fp4e2m1", FP4E2M1, "i8"),
]


@pytest.mark.usefixtures("gfx90a_gpu")
@pytest.mark.parametrize(
    "name,dtype,carrier", _TRANSFORMED, ids=[t[0] for t in _TRANSFORMED]
)
def test_transform_fragment_round_trips_on_the_gpu(name, dtype, carrier) -> None:
    # The tile is loaded in `_REORDER_TILE`'s layout, transformed to `_RUNS_SWAPPED` (registers
    # 8-15 and 16-23 trade places) and stored in that layout: OUT equals IN only if every storage
    # register moved with its elements. A u8 tile is loaded as i8 and reinterpreted.
    from rocke.helpers.spec import SignatureBuilder

    shape = tuple(_REORDER_TILE.shape)
    load_dtype = I8 if dtype == U8 else dtype
    load_td = make_tensor_desc(shape, (2 * 8 * _WAVE, 8 * _WAVE, 1), load_dtype)
    store_td = make_tensor_desc(shape, (2 * 8 * _WAVE, 8 * _WAVE, 1), dtype)
    elem = {"tf32": TF32, "i8": I8}[carrier]
    b = IRBuilder(f"transform_round_trip_{name}")
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    src = _param(b, "IN", elem, readonly=True)
    out = _param(b, "OUT", U8 if dtype == U8 else elem, writeonly=True)
    tid = b.thread_id_x()
    fragment = load_fragment(
        b, src, make_window(load_td, (0, 0, 0)), _REORDER_TILE, tid
    )
    if dtype == U8:
        fragment = Fragment(
            _REORDER_TILE, U8, b.bitcast(fragment.value, VectorType(U8, 32))
        )
    moved = transform_fragment(b, fragment, _RUNS_SWAPPED)
    store_fragment(b, out, make_window(store_td, (0, 0, 0)), moved, tid)
    b.ret()
    signature = (
        SignatureBuilder()
        .ptr("IN", carrier)
        .ptr("OUT", "u8" if dtype == U8 else carrier)
        .build()
    )

    rng = np.random.default_rng(7)
    count = 2 * 2 * 8 * _WAVE
    if dtype == TF32:
        data = rng.integers(0, 2**32, size=count, dtype=np.uint64).astype(np.uint32)
    elif dtype == U8:
        data = rng.integers(0, 256, size=count, dtype=np.uint16).astype(np.uint8)
    else:
        data = _packed_bytes(rng.integers(0, 16, size=count), 4)
    result = np.zeros_like(data)
    _run(b.kernel, signature, {"IN": data, "OUT": result})
    np.testing.assert_array_equal(result, data)
