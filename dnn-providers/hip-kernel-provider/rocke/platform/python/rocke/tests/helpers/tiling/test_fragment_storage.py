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
window that would split a word is refused, and its LDS buffer holds i32 words. These tests check
the storage table, the refusals and their messages on the CPU, and on a gfx90a GPU that tf32
payloads and packed operand tiles (in gfx950's canonical operand layouts) round-trip bit-exactly
through global memory and LDS, and that the packed words a lane loads are exactly
``FragmentPacking.pack`` of the elements its layout places there.
"""

from __future__ import annotations

import math

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
from rocke.core.storage import BitPacking, register_packing
from rocke.helpers.tiling import (
    Fragment,
    TileMma,
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
    transform_fragment,
)
from rocke.helpers.tiling.analysis.geometry import _elem_bits, _elem_bytes
from rocke.helpers.tiling.emit import emit_tensor_coordinates
from rocke.helpers.tiling.lds_conflict import NumBuilder

_WAVE = 64


def _flat_tile(registers: int):
    """A 1-D tile whose 64 lanes each hold `registers` consecutive elements."""
    return make_tile_desc(
        shape=[_WAVE * registers],
        thread_tile=[registers],
        thread_dist=[_WAVE],
        wave_size=_WAVE,
    )


def _operand_desc(dtype_name: str):
    """The canonical A-operand layout of gfx950's 16x16x128 f8f6f4 instruction for a 4-bit or 6-bit
    operand: each lane holds one run of 32 K elements."""
    return TileMma(
        (16, 16, 128), a=dtype_name, b=dtype_name, c="f32", target="gfx950"
    ).a_desc


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
    a_desc = _operand_desc("fp4e2m1")
    b = IRBuilder("fp4_bytes_value")
    expected = (
        "Fragment value expected vec<i32x4> for TileDesc(shape=(16, 128)), got vec<i8x32> -- "
        "fp4e2m1 registers are held packed, 32 elements of 4 bits in 4 i32 words; build them "
        "with load_fragment or fill_fragment"
    )
    with pytest.raises(ValueError) as raised:
        Fragment(a_desc, FP4E2M1, b.zero_vec(I8, 32))
    assert str(raised.value) == expected


@pytest.mark.parametrize("dtype,words", [(FP4E2M1, 4), (FP6E2M3, 6), (TF32, 32)])
def test_a_zero_fill_is_all_zero_storage_words(dtype, words) -> None:
    b = IRBuilder("zero_fill")
    fragment = make_fragment(_operand_desc("fp4e2m1"), dtype)
    fill_fragment(b, fragment, 0)
    assert fragment.value.type == VectorType(I32, words)
    (fill,) = [op for op in b.kernel.body.ops if op.name == "arith.constant_vec"]
    assert fill.attrs["elem"] == "i32" and fill.attrs["fill"] == 0.0


# ---- u8: stored, but neither filled nor loaded -----------------------------------------------------------

_U8_TILE = make_tile_desc(
    shape=[64, 16], thread_tile=[1, 16], thread_dist=[64, 1], wave_size=_WAVE
)
_U8_TENSOR = make_tensor_desc((64, 16), (16, 1), U8)


def _u8_refusal(verb: str) -> str:
    return (
        f"{verb} cannot build a u8 fragment: core's zero_vec has no u8 form -- TODO: not "
        f"implemented yet; build an I8 fragment with {verb} and b.bitcast its value to "
        f"vec<u8 x n>"
    )


@pytest.mark.parametrize("source", ["global", "lds"])
def test_a_u8_fragment_is_refused_by_load_fragment(source) -> None:
    b = IRBuilder("u8_load")
    tid = b.thread_id_x()
    if source == "global":
        src = b.param("IN", PtrType(U8, "global"))
    else:
        src = b.smem_alloc(U8, [_U8_TENSOR.element_span], name_hint="lds")
    with pytest.raises(NotImplementedError) as raised:
        load_fragment(b, src, make_window(_U8_TENSOR, (0, 0)), _U8_TILE, tid)
    assert str(raised.value) == _u8_refusal("load_fragment")


def test_a_u8_fragment_is_refused_by_fill_fragment() -> None:
    b = IRBuilder("u8_fill")
    with pytest.raises(NotImplementedError) as raised:
        fill_fragment(b, make_fragment(_U8_TILE, U8), 0)
    assert str(raised.value) == _u8_refusal("fill_fragment")


@pytest.mark.parametrize("dest", ["global", "lds"])
def test_the_named_u8_route_builds_a_fragment_that_stores(dest) -> None:
    # An I8 fragment's value, bitcast to vec<u8 x n>, is a u8 fragment with the same bits; it stores
    # to global memory and to LDS.
    b = IRBuilder("u8_route")
    tid = b.thread_id_x()
    src = b.param("IN", PtrType(I8, "global"))
    i8_tensor = make_tensor_desc((64, 16), (16, 1), I8)
    loaded = load_fragment(b, src, make_window(i8_tensor, (0, 0)), _U8_TILE, tid)
    fragment = Fragment(_U8_TILE, U8, b.bitcast(loaded.value, VectorType(U8, 16)))
    if dest == "global":
        dst = b.param("OUT", PtrType(U8, "global"))
    else:
        dst = b.smem_alloc(U8, [_U8_TENSOR.element_span], name_hint="lds")
    store_fragment(b, dst, make_window(_U8_TENSOR, (0, 0)), fragment, tid)
    stores = [op for op in b.kernel.body.ops if "store" in op.name]
    assert len(stores) == 16


# ---- refusals: a packed access moves whole words --------------------------------------------------------


def _load(td, tile, *, origin=(0, 0), bounds=None, lds_elem=None):
    b = IRBuilder("packed_load")
    tid = b.thread_id_x()
    if lds_elem is None:
        src = b.param("IN", PtrType(I8, "global"))
    else:
        src = b.smem_alloc(lds_elem, [td.element_span], name_hint="lds")
    return load_fragment(b, src, make_window(td, origin, bounds), tile, tid)


def test_a_k_major_strided_packed_tensor_is_refused() -> None:
    # The operand layout's K runs lie along axis 1, but this tensor is M-contiguous, so each lane's
    # registers would be single 4-bit elements 16 apart.
    td = make_tensor_desc((16, 128), (1, 16), FP4E2M1)
    with pytest.raises(ValueError) as raised:
        _load(td, _operand_desc("fp4e2m1"))
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
        _load(td, _operand_desc("fp4e2m1"))


def test_an_origin_inside_a_word_is_refused() -> None:
    td = make_tensor_desc((16, 256), (256, 1), FP4E2M1)
    with pytest.raises(ValueError, match="the window origin on the stride-1 axis is 4"):
        _load(td, _operand_desc("fp4e2m1"), origin=(0, 4))


@pytest.mark.parametrize("verb", ["load", "store"])
def test_a_clipped_packed_window_needs_a_whole_word_bound(verb) -> None:
    a_desc = _operand_desc("fp4e2m1")
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


def test_a_packed_lds_buffer_must_hold_words() -> None:
    td = make_tensor_desc((16, 128), (128, 1), FP4E2M1)
    with pytest.raises(ValueError) as raised:
        _load(td, _operand_desc("fp4e2m1"), lds_elem=F16)
    assert str(raised.value) == (
        "LDS buffer for fp4e2m1 fragments must hold its storage words -- allocate "
        "smem_alloc(I32, [words]) sized from the descriptor; got smem<f16, [2048]>"
    )


def test_a_packed_lds_buffer_is_sized_in_words() -> None:
    # 16 x 128 fp4 elements are 256 words: a buffer of 255 words is refused, one of 256 accepted.
    td = make_tensor_desc((16, 128), (128, 1), FP4E2M1)
    a_desc = _operand_desc("fp4e2m1")
    for words, ok in ((255, False), (256, True)):
        b = IRBuilder("packed_lds_size")
        tid = b.thread_id_x()
        lds = b.smem_alloc(I32, [words], name_hint="lds")
        if ok:
            load_fragment(b, lds, make_window(td, (0, 0)), a_desc, tid)
        else:
            with pytest.raises(
                ValueError, match="256 i32 words, but the buffer holds 255"
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


# ---- the view widths ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,bits", [("f16", 16), ("tf32", 32), ("fp8e4m3", 8), ("fp4e2m1", 4)]
)
def test_elem_bits(name, bits) -> None:
    assert _elem_bits(name) == bits


def test_elem_bytes() -> None:
    assert _elem_bytes("f64") == 8
    assert _elem_bytes("f16") == 2


def test_byte_views_refuse_a_sub_byte_dtype() -> None:
    with pytest.raises(NotImplementedError) as raised:
        _elem_bytes("fp4e2m1")
    assert str(raised.value) == (
        "fp4e2m1 elements are 4 bits, not a whole number of bytes -- byte-addressed views "
        "(LDS bank model) of packed data -- TODO: not implemented yet"
    )


# ---- GPU (gfx90a) ---------------------------------------------------------------------------------------


def _gfx90a_present() -> bool:
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0) == "gfx90a"
    except Exception:  # noqa: BLE001
        # Any HIP or driver failure means there is no usable GPU here.
        return False


_needs_gfx90a = pytest.mark.skipif(not _gfx90a_present(), reason="needs a gfx90a GPU")


def _run(kernel, signature, inputs: dict, outputs: dict) -> None:
    """Compile `kernel` for gfx90a, launch one wave on host arrays and copy `outputs` back."""
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    art = compile_kernel(kernel, arch="gfx90a")
    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name, signature=signature
    )
    rt = Runtime()
    arrays = {**inputs, **outputs}
    dev = {name: DeviceMem(a.nbytes) for name, a in arrays.items()}
    for name, a in arrays.items():
        rt.memcpy_h2d(dev[name].ptr(), as_u8_buffer(a), a.nbytes)
    launcher(dev, config=LaunchConfig(grid=(1, 1, 1), block=(_WAVE, 1, 1)))
    synchronize_and_release()
    for name, a in outputs.items():
        rt.memcpy_d2h(as_u8_buffer(a), dev[name].ptr(), a.nbytes)


def _param(b, name, elem, **flags):
    return b.param(name, PtrType(elem, "global"), noalias=True, align=16, **flags)


@_needs_gfx90a
def test_tf32_payload_round_trips_through_lds() -> None:
    # A tf32 tile is loaded from global memory with 4-element runs, stored into an LDS buffer that
    # holds it transposed, read back with runs along the other axis and stored to OUT. tf32 has no
    # arithmetic, so every 32-bit pattern, NaN payloads included, must come back unchanged.
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
    lds = b.smem_alloc(TF32, [lds_td.element_span], name_hint="lds")
    window = make_window(global_td, (0, 0))
    fragment = load_fragment(b, src, window, load_tile, tid)
    assert fragment.storage_type == VectorType(I32, 8)
    store_fragment(b, lds, make_window(lds_td, (0, 0)), fragment, tid)
    b.sync()
    staged = load_fragment(b, lds, make_window(lds_td, (0, 0)), read_tile, tid)
    store_fragment(b, out, window, staged, tid)
    b.ret()
    signature = SignatureBuilder().ptr("IN", "tf32").ptr("OUT", "tf32").build()
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2**32, size=(rows, cols), dtype=np.uint64).astype(np.uint32)
    result = np.zeros_like(data)
    _run(b.kernel, signature, {"IN": data}, {"OUT": result})
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


@_needs_gfx90a
@pytest.mark.parametrize("name,dtype,bits", _PACKED, ids=[p[0] for p in _PACKED])
def test_packed_operand_tile_round_trips(name, dtype, bits) -> None:
    # An A-operand tile (16 x 128, K contiguous) in gfx950's canonical f8f6f4 layout is loaded from a
    # packed tensor. WORDS receives every lane's registers, which must be the packing of the
    # elements the layout places there. The fragment is stored back to OUT directly, and to OUT_LDS
    # through an LDS buffer of i32 words; both must equal the input bytes.
    from rocke.helpers.spec import SignatureBuilder

    a_desc = _operand_desc("fp4e2m1" if bits == 4 else name)
    rows, cols = a_desc.shape
    td = make_tensor_desc((rows, cols), (cols, 1), dtype)
    words = register_packing(name, a_desc.register_count).carrier_count
    lds_words = rows * cols * bits // 32

    b = IRBuilder(f"packed_round_trip_{name}")
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    src = _param(b, "IN", I8, readonly=True)
    out = _param(b, "OUT", I8, writeonly=True)
    out_lds = _param(b, "OUT_LDS", I8, writeonly=True)
    dump = _param(b, "WORDS", I32, writeonly=True)
    tid = b.thread_id_x()
    lds = b.smem_alloc(I32, [lds_words], name_hint="lds")
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

    rng = np.random.default_rng(bits)
    patterns = rng.integers(0, 1 << bits, size=(rows, cols))
    data = _packed_bytes(patterns, bits)
    result, result_lds = np.zeros_like(data), np.zeros_like(data)
    lane_words = np.zeros((_WAVE, words), np.uint32)
    _run(
        b.kernel,
        signature,
        {"IN": data},
        {"OUT": result, "OUT_LDS": result_lds, "WORDS": lane_words},
    )
    np.testing.assert_array_equal(lane_words, _expected_words(a_desc, patterns, name))
    np.testing.assert_array_equal(result, data)
    np.testing.assert_array_equal(result_lds, data)


@_needs_gfx90a
@pytest.mark.parametrize(
    "name,dtype,bits", _PACKED[:2], ids=[p[0] for p in _PACKED[:2]]
)
def test_clipped_packed_window_masks_whole_words(name, dtype, bits) -> None:
    # The window is bounded at K = 64. Its load reads zero words past the bound (OUT_FULL stores the
    # whole tile unclipped), and its store drops the words past the bound (OUT keeps its 0xff fill
    # there); before the bound both equal the input.
    from rocke.helpers.spec import SignatureBuilder

    a_desc = _operand_desc(name)
    rows, cols = a_desc.shape
    bound = 64
    td = make_tensor_desc((rows, cols), (cols, 1), dtype)
    b = IRBuilder(f"packed_clip_{name}")
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    src = _param(b, "IN", I8, readonly=True)
    out_full = _param(b, "OUT_FULL", I8, writeonly=True)
    out = _param(b, "OUT", I8, writeonly=True)
    tid = b.thread_id_x()
    clipped = make_window(td, (0, 0), (None, bound))
    fragment = load_fragment(b, src, clipped, a_desc, tid)
    store_fragment(b, out_full, make_window(td, (0, 0)), fragment, tid)
    whole = load_fragment(b, src, make_window(td, (0, 0)), a_desc, tid)
    store_fragment(b, out, clipped, whole, tid)
    b.ret()
    signature = (
        SignatureBuilder().ptr("IN", "i8").ptr("OUT_FULL", "i8").ptr("OUT", "i8")
    )
    signature = signature.build()

    rng = np.random.default_rng(bits + 1)
    patterns = rng.integers(1, 1 << bits, size=(rows, cols))
    data = _packed_bytes(patterns, bits)
    result_full = np.full_like(data, 0xFF)
    result = np.full_like(data, 0xFF)
    _run(b.kernel, signature, {"IN": data}, {"OUT_FULL": result_full, "OUT": result})
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


@_needs_gfx90a
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
    _run(b.kernel, signature, {"IN": data}, {"OUT": result})
    np.testing.assert_array_equal(result, data)
