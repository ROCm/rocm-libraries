# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A store never goes through a view that provably overlaps itself.

A tensor description whose strides map two positions to one element (a stride 0 broadcast, a
sliding-window view whose rows overlap) is a fine thing to read, but a store through it has two
lanes writing one element: whichever lands last wins, a race. ``store_fragment`` refuses such a
view at build, before it emits anything, when the overlap is PROVEN from the description: it names
two positions that reach one element. A view the build can decide neither way -- a runtime row
stride over a runtime row length with nothing stated between them -- is trusted, as CK and CUTLASS
trust their strides.

Two internal questions decide it: ``TensorDesc._strides_injective`` proves that no two positions
share an element, and ``TensorDesc._aliasing_positions`` finds two that do. These tests pin:

- the views kernels store through are proven not to alias: packed and padded tensors, permuted
  views, packed rows with runtime lengths, a BSHD tensor with a runtime sequence length, a batch
  stride over runtime rows, one THD sequence of a packed token buffer, axes of length 1, a reversed
  axis;
- the overlaps proven from the description: a stride 0, a sliding window, rows longer than their
  int stride (including a runtime row length stated long enough), two axes on one runtime stride;
- the views neither proven: a three-axis overlap no pair of axes shows, a runtime row stride over
  a runtime row length, a checked product that could wrap in i32, a product over another axis's
  length;
- ``store_fragment`` refuses a global and an LDS store through a proven overlap with one exact
  message and emits nothing, stores through a runtime-ldc padded C view, and a load through an
  overlapping view lowers identically in both engines;
- on the GPU, a load through a sliding-window view reads exactly what numpy's view of the same
  bytes holds.

The tracked demos store through packed views; their IR is pinned by ``test_demo_ir_snapshot.py``.
"""

from __future__ import annotations

import ctypes
import re
import struct

import numpy as np
import pytest

from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.helpers.tiling import (
    load_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)
from rocke.tests.engines import lower_with_both_engines

WAVE = 64
TM = TK = 16

# (M, K) tile, K-contiguous: 4 consecutive K elements per lane, 16 lanes on M, 4 on K.
TILE = make_tile_desc(
    shape=[TM, TK],
    thread_tile=[1, 4],
    thread_dist=[16, 4],
    thread_order=[1, 0],
    wave_size=WAVE,
)


def _builder(name: str = "aliasing_store") -> IRBuilder:
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = WAVE
    return b


def _refused(first: tuple[int, ...], second: tuple[int, ...]):
    message = (
        f"store_fragment: positions {first} and {second} of the tensor reach the same element "
        "-- a store through an aliasing view races; store through a view whose positions are all "
        "different elements"
    )
    return pytest.raises(ValueError, match=f"^{re.escape(message)}$")


def _loaded(b: IRBuilder, name: str):
    """A wave-uniform value read from memory: the GPU can see it, the host cannot."""
    source = b.param(name, PtrType(I32, "global"), readonly=True, noalias=True)
    return b.readfirstlane(b.global_load(source, b.block_id_x(), I32))


# --- views that do not alias -------------------------------------------------------------------


@pytest.mark.parametrize(
    "lengths,strides",
    [
        ((64, 32), (32, 1)),  # packed, K-contiguous
        ((64, 32), (40, 1)),  # padded rows
        ((64, 32), (1, 64)),  # M-contiguous: the layout lives in the strides
        ((4, 64, 32), (64 * 32, 32, 1)),  # batched, packed
        ((2, 3, 4), (4, 8, 1)),  # fits only in the order 1, 4, 8, not in axis order
        ((1, 32), (0, 1)),  # an axis of length 1 positions nothing, whatever its stride
        ((64, 32), (-32, 1)),  # rows read in reverse
    ],
)
def test_int_views_that_kernels_store_through_do_not_alias(lengths, strides):
    td = make_tensor_desc(lengths, strides, F16)
    assert td._strides_injective()
    assert td._aliasing_positions() is None


def test_a_packed_tensor_with_runtime_lengths_does_not_alias():
    # The same K Value as the length of the rows and their stride: rows are packed.
    b = _builder()
    m = b.param("M", I32, ge=0, le=1 << 20)
    k = b.param("K", I32, ge=1, le=1 << 20)
    assert make_tensor_desc((m, k), (k, 1), F16)._strides_injective()


def test_a_bshd_tensor_with_a_runtime_sequence_length_does_not_alias():
    # (B, S, H, D) packed. A batch stride declared at least the most a batch can reach clears the
    # sequence axis by the numbers; one checked as the product of the very sequence length the
    # axis holds clears it whatever that length is at run time.
    heads, dim = 8, 64
    b = _builder()
    s = b.param("S", I32, ge=1, le=1 << 16)
    batch_stride = b.param("batch_stride", I32, ge=(1 << 16) * heads * dim)
    td = make_tensor_desc((4, s, heads, dim), (batch_stride, heads * dim, dim, 1), F16)
    assert td._strides_injective()
    for operands in ("length first", "constant first"):
        b = _builder()
        seqlen = b.gpu_checked_value(_loaded(b, "SEQLENS"), ge=1, le=1 << 16)
        row = b.const_i32(heads * dim)
        product = (
            b.mul(seqlen, row) if operands == "length first" else b.mul(row, seqlen)
        )
        checked = b.gpu_checked_value(product, ge=1, le=(1 << 16) * heads * dim)
        td = make_tensor_desc(
            (4, seqlen, heads, dim), (checked, heads * dim, dim, 1), F16
        )
        assert td._strides_injective(), operands


def test_a_batch_stride_over_runtime_rows_does_not_alias():
    # (G, M, K) with runtime M and K: the batch stride is checked as M * K of both Values.
    b = _builder()
    m = b.gpu_checked_value(_loaded(b, "MS"), ge=1, le=1 << 10)
    k = b.gpu_checked_value(_loaded(b, "KS"), ge=1, le=1 << 10)
    batch_stride = b.gpu_checked_value(b.mul(m, k), ge=1, le=1 << 20)
    td = make_tensor_desc((3, m, k), (batch_stride, k, 1), F16)
    assert td._strides_injective()


def test_one_thd_sequence_of_a_packed_token_buffer_does_not_alias():
    # Tokens of all sequences packed back to back, H heads of D each: (head, token, D) with
    # strides (D, H * D, 1). One sequence has a runtime length loaded from memory; the heads
    # interleave inside one token, so no two (head, token, d) positions share an element.
    heads, dim = 4, 32
    b = _builder()
    seqlen = b.gpu_checked_value(_loaded(b, "SEQLENS"), ge=0, le=1 << 16)
    total = b.param("T", I32, ge=0, le=1 << 20)
    for tokens in (seqlen, total):
        td = make_tensor_desc((heads, tokens, dim), (dim, heads * dim, 1), F16)
        assert td._strides_injective()


def test_a_runtime_row_stride_whose_range_clears_the_row_is_proven():
    # No identity needed when the numbers do it: rows of at most 64 under a stride of at least 64.
    b = _builder()
    k = b.param("K", I32, ge=1, le=64)
    lda = b.param("LDA", I32, ge=64, le=1 << 20)
    assert make_tensor_desc((32, k), (lda, 1), F16)._strides_injective()


# --- overlaps proven from the description ----------------------------------------------------------


@pytest.mark.parametrize(
    "lengths,strides,first,second",
    [
        ((64, 32), (0, 1), (0, 0), (1, 0)),  # every row is the same row
        (
            (64, 32),
            (1, 1),
            (1, 0),
            (0, 1),
        ),  # sliding window: row m starts one element later
        (
            (64, 65),
            (64, 1),
            (1, 0),
            (0, 64),
        ),  # rows one element longer than their stride
        (
            (8, 64),
            (-32, 1),
            (0, 0),
            (1, 32),
        ),  # reversed rows of 64 under a stride of 32
    ],
)
def test_int_views_whose_positions_meet_are_proven_to_alias(
    lengths, strides, first, second
):
    td = make_tensor_desc(lengths, strides, F16)
    assert not td._strides_injective()
    assert td._aliasing_positions() == (first, second)


def test_a_runtime_row_length_stated_long_enough_proves_rows_overlap():
    # Rows of at least 65 elements under the int stride 64: row 1 starts inside row 0 at every
    # launch the facts allow.
    b = _builder()
    n = b.param("N", I32, ge=65, le=4096)
    td = make_tensor_desc((8, n), (64, 1), F16)
    assert td._aliasing_positions() == ((1, 0), (0, 64))
    # A row length that may be short proves nothing.
    short = b.param("N2", I32, ge=0, le=4096)
    assert make_tensor_desc((8, short), (64, 1), F16)._aliasing_positions() is None


def test_two_axes_on_one_runtime_stride_meet_after_one_step():
    b = _builder()
    s = b.param("S", I32, ge=1, le=4096)
    td = make_tensor_desc((4, 4, 16), (s, s, 1), F16)
    assert td._aliasing_positions() == ((1, 0, 0), (0, 1, 0))


# --- views neither proven ----------------------------------------------------------------------------


def test_an_overlap_no_pair_of_axes_shows_is_not_proven():
    # 1*4 - 1*7 + 3*1 == 0: positions (1, 0, 3) and (0, 1, 0) meet, through all three axes.
    td = make_tensor_desc((2, 3, 4), (4, 7, 1), F16)
    assert not td._strides_injective()
    assert td._aliasing_positions() is None


def test_a_runtime_row_stride_over_a_runtime_row_length_is_not_decided():
    b = _builder()
    m = b.param("M", I32, ge=0, le=1 << 16)
    n = b.param("N", I32, ge=0, le=1 << 16)
    ldc = b.param("ldc", I32, ge=1, le=1 << 16)
    td = make_tensor_desc((m, n), (ldc, 1), F16)
    assert not td._strides_injective()
    assert td._aliasing_positions() is None


def test_a_checked_product_that_could_wrap_in_i32_is_not_proven():
    # S up to 2^30 times 8 can wrap in i32, so the checked value need not be the product.
    b = _builder()
    s = b.gpu_checked_value(_loaded(b, "SEQLENS"), ge=1, le=1 << 30)
    batch_stride = b.gpu_checked_value(b.mul(s, b.const_i32(8)), ge=1, le=(1 << 31) - 1)
    td = make_tensor_desc((4, s, 8), (batch_stride, 8, 1), F16)
    assert not td._strides_injective()
    assert td._aliasing_positions() is None


def test_a_product_over_another_axis_length_is_not_proven():
    # The batch stride is S * 8 but the rows hold T: nothing ties it to the axes it must clear.
    b = _builder()
    s = b.gpu_checked_value(_loaded(b, "SEQLENS"), ge=1, le=1 << 16)
    t = b.gpu_checked_value(_loaded(b, "TS"), ge=1, le=1 << 16)
    batch_stride = b.gpu_checked_value(b.mul(s, b.const_i32(8)), ge=1, le=1 << 19)
    td = make_tensor_desc((4, t, 8), (batch_stride, 8, 1), F16)
    assert not td._strides_injective()


# --- the verbs ---------------------------------------------------------------------------------


def _fragment(b, ptr, lane):
    return load_fragment(
        b,
        ptr,
        make_window(make_tensor_desc((TM, TK), (TK, 1), F16), (0, 0)),
        TILE,
        lane,
    )


def test_a_global_store_through_an_aliasing_view_is_refused_and_a_load_is_not():
    b = _builder()
    ptr = b.param("X", PtrType(F16, "global"), align=16)
    lane = b.thread_id_x()
    sliding = make_window(make_tensor_desc((TM, 64), (1, 1), F16), (0, 0))
    fragment = load_fragment(b, ptr, sliding, TILE, lane)
    with _refused((1, 0), (0, 1)):
        store_fragment(b, ptr, sliding, fragment, lane)
    b.ret()
    lower_with_both_engines(b.kernel, "gfx90a")


def test_an_lds_store_through_an_aliasing_view_is_refused():
    b = _builder()
    ptr = b.param("X", PtrType(F16, "global"), align=16)
    lane = b.thread_id_x()
    fragment = _fragment(b, ptr, lane)
    smem = b.smem_alloc(F16, [TM * TK])
    broadcast = make_window(make_tensor_desc((TM, TK), (0, 1), F16), (0, 0))
    with _refused((0, 0), (1, 0)):
        store_fragment(b, smem, broadcast, fragment, lane)


def test_the_refusal_emits_nothing():
    b = _builder()
    ptr = b.param("X", PtrType(F16, "global"), align=16)
    lane = b.thread_id_x()
    fragment = _fragment(b, ptr, lane)
    before = len(b.kernel.body.ops)
    with _refused((0, 0), (1, 0)):
        store_fragment(
            b,
            ptr,
            make_window(make_tensor_desc((TM, TK), (0, 1), F16), (0, 0)),
            fragment,
            lane,
        )
    assert len(b.kernel.body.ops) == before


def test_a_store_through_a_runtime_ldc_padded_c_view_is_trusted():
    # C (M, N) with its own leading dimension ldc and nothing stated between ldc and N: the build
    # cannot decide whether the rows overlap, so it trusts the view and stores, in both engines.
    b = _builder()
    ptr = b.param("C", PtrType(F16, "global"), align=16)
    lane = b.thread_id_x()
    fragment = _fragment(b, ptr, lane)
    m = b.param("M", I32, ge=0, le=1 << 16)
    n = b.param("N", I32, ge=0, le=1 << 16)
    ldc = b.param("ldc", I32, ge=1, le=1 << 16)
    padded = make_window(make_tensor_desc((m, n), (ldc, 1), F16), (0, 0))
    before = len(b.kernel.body.ops)
    store_fragment(b, ptr, padded, fragment, lane)
    b.ret()
    assert len(b.kernel.body.ops) > before
    text = lower_with_both_engines(b.kernel, "gfx90a")
    assert "store" in text


# --- on the GPU: a load through a sliding-window view ---------------------------------------------


def _gpu_arch() -> str | None:
    try:
        from rocke.runtime.hip_module import current_device, get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch(current_device())
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


WIDTH = (
    TK + TM - 1
)  # the sliding view (TM, TK) with strides (1, 1) reaches TM + TK - 1 elements


def _sliding_copy_kernel():
    """``Y[m, k] = X[m + k]``: a load through the sliding view of ``X``, stored packed to ``Y``."""
    b = _builder("sliding_copy")
    x = b.param("X", PtrType(F16, "global"), align=16)
    y = b.param("Y", PtrType(F16, "global"), align=16)
    lane = b.thread_id_x()
    sliding = make_window(make_tensor_desc((TM, TK), (1, 1), F16), (0, 0))
    tile = load_fragment(b, x, sliding, TILE, lane)
    store_fragment(
        b, y, make_window(make_tensor_desc((TM, TK), (TK, 1), F16), (0, 0)), tile, lane
    )
    b.ret()
    return b.kernel


def _to_device(runtime, address: int, host: np.ndarray) -> None:
    runtime.memcpy_h2d(
        address, (ctypes.c_ubyte * host.nbytes).from_buffer(host), host.nbytes
    )


def _from_device(runtime, address: int, count: int) -> np.ndarray:
    host = np.empty(count, dtype=np.float16)
    runtime.memcpy_d2h(
        (ctypes.c_ubyte * host.nbytes).from_buffer(host), address, host.nbytes
    )
    return host


def test_a_load_through_a_sliding_window_view_reads_numpys_view():
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    kernel = _sliding_copy_kernel()
    lower_with_both_engines(kernel, "gfx90a")
    runtime = Runtime()
    function = runtime.load_module(
        compile_kernel(kernel, arch=arch).hsaco
    ).get_function(kernel.name)
    source = (np.arange(WIDTH) * 3 + 1).astype(np.float16)
    expected = np.lib.stride_tricks.sliding_window_view(source, TK)
    sentinel = np.full(TM * TK, np.nan, dtype=np.float16)
    x = runtime.alloc(source.nbytes)
    y = runtime.alloc(sentinel.nbytes)
    try:
        _to_device(runtime, x, source)
        _to_device(runtime, y, sentinel)
        runtime.launch_blocking(
            function, (1, 1, 1), (WAVE, 1, 1), struct.pack("QQ", x, y)
        )
        got = _from_device(runtime, y, TM * TK).reshape(TM, TK)
        assert np.array_equal(got, expected)
    finally:
        for address in (x, y):
            runtime.free(address)
