# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A global access claims only the alignment its address really has.

``load_fragment`` / ``store_fragment`` tell the compiler how aligned each access is. The claim is a
promise: a uniform address that claims 4 bytes or more may become a scalar load, which ignores the
low address bits, so a claim above the truth reads the wrong data. The claim is built from facts
only -- the pointer's declared ``align=``, every axis's position (int origins by value, runtime
origins by the multiple-of fact the IR range walk proves), a slice's begin offset and every stride
(a runtime stride by its stated ``multiple_of``) -- and is capped at the access's own width.

The width rule is unchanged: a wide access needs every other stride to be a multiple of its width
(a runtime stride by its stated ``multiple_of``), else the access is scalar.

These tests check, offline, that the claimed alignment equals the largest power of two dividing
every address the facts allow (times the element size, capped at the access width), over stride
classes ``K + {0, 1, 2, 4, 8}``, int and runtime origins, int and runtime begin offsets and runtime
strides; that the stated examples hold exactly; and that a uniform address at an odd runtime
offset is read by a vector-memory load. On the GPU present, GEMMs with padded strides are bit-exact
with NaN in every operand pad and the C pad left untouched.
"""

from __future__ import annotations

import itertools
import re
from math import gcd

import numpy as np
import pytest

from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType
from rocke.helpers.tiling import (
    exit_if_tile_outside,
    load_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.descriptors import TensorDesc

from .test_masked_global_access import (
    SENTINEL,
    _body,
    _gpu_arch,
    _guards_untouched,
    _launch,
)

WAVE = 64
ROWS, COLS = 32, 16
K = 64

# A (32, 16) tile, K-contiguous: 8 consecutive K elements per lane, 32 lanes on M, 2 on K. Its
# vectors start at every row and at the columns that are multiples of the access width.
TILE = make_tile_desc(
    shape=[ROWS, COLS],
    thread_tile=[1, 8],
    thread_dist=[32, 2],
    thread_order=[1, 0],
    wave_size=WAVE,
)

_ELEM_BYTES = {"f16": 2, "f32": 4}
_LOADS = ("memref.global_load_vN", "memref.global_load_typed")
_STORES = ("memref.global_store_vN", "memref.global_store_typed")


def _builder(name: str) -> IRBuilder:
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = WAVE
    return b


def _all_ops(b: IRBuilder) -> list:
    def walk(ops):
        for op in ops:
            yield op
            for region in op.regions:
                yield from walk(region.ops)

    return list(walk(b.kernel.body.ops))


def _claims(b: IRBuilder, names: tuple[str, ...]) -> list[tuple[int, int]]:
    """``(align, width)`` of every global access op named in ``names``."""
    return [
        (op.attrs["align"], op.attrs.get("vec") or 1)
        for op in _all_ops(b)
        if op.name in names
    ]


def _power_of_two(n: int) -> int:
    return n & -n


def _true_alignment(elem: int, pointer: int, offsets) -> int:
    """The largest power of two dividing the pointer's alignment and every element offset (in
    bytes)."""
    g = pointer
    for offset in offsets:
        g = gcd(g, offset * elem)
    return _power_of_two(g)


# --- the claim equals the truth, brute-forced over residues ---------------------------------------


class _Factor:
    """One input of the address: how the kernel gets it, and every value it can take at run time
    (one value for an int)."""

    def __init__(self, emit, values):
        self.emit = emit
        self.values = tuple(values)


def _int(n: int) -> _Factor:
    return _Factor(lambda b, name: n, (n,))


def _runtime_stride(multiple_of: int) -> _Factor:
    # A stride stated a multiple of ``multiple_of``, at least K: the values from K up.
    def emit(b, name):
        return b.param(name, I32, ge=K, le=1 << 16, multiple_of=multiple_of)

    first = -(-K // multiple_of) * multiple_of
    return _Factor(emit, range(first, first + 5 * multiple_of, multiple_of))


def _runtime_begin(multiple_of: int) -> _Factor:
    def emit(b, name):
        return b.param(name, I32, ge=0, le=1 << 16, multiple_of=multiple_of)

    return _Factor(emit, range(0, 5 * multiple_of, multiple_of))


def _origin_from_parameter() -> _Factor:
    # Guarded to [0, K - COLS] so the tile stays inside: the walk knows no multiple-of fact.
    def emit(b, name):
        return exit_if_tile_outside(b, b.param(name, I32), K - COLS + 1)

    return _Factor(emit, range(16))


def _origin_from_block(step: int, plus: int) -> _Factor:
    # block_id_x * step + plus: the walk proves ``plus mod step``.
    def emit(b, name):
        block = exit_if_tile_outside(b, b.block_id_x(), 2)
        return b.add(b.mul(block, b.const_i32(step)), b.const_i32(plus))

    return _Factor(emit, (plus, step + plus))


def _claim_and_truth(dtype, lda: _Factor, origin_k: _Factor, begin_k: _Factor, pointer):
    """Build a load of TILE through ``(32 + 32, K + 32)`` f16/f32 A at origin ``(0, origin_k)``
    with begin ``(0, begin_k)`` and stride ``lda``; return every load's claim and the truth for
    each width."""
    b = _builder("alignment")
    attrs = {} if pointer is None else {"align": pointer}
    x = b.param("X", PtrType(dtype, "global"), **attrs)
    td = TensorDesc(
        (ROWS + 32, K + 32),
        (lda.emit(b, "lda"), 1),
        dtype,
        begin=(0, begin_k.emit(b, "begin_k")),
    )
    window = make_window(td, (0, origin_k.emit(b, "origin_k")))
    load_fragment(b, x, window, TILE, b.thread_id_x())
    b.ret()
    elem = _ELEM_BYTES[dtype.name]
    claims = _claims(b, _LOADS)
    width = claims[0][1]
    offsets = [
        row * stride + column + origin + begin
        for stride, origin, begin in itertools.product(
            lda.values, origin_k.values, begin_k.values
        )
        for row in range(ROWS)
        for column in range(0, COLS, width)
    ]
    truth = _true_alignment(elem, pointer or elem, offsets)
    return claims, min(truth, width * elem, 16)


_STRIDE_CLASSES = [K + pad for pad in (0, 1, 2, 4, 8)]


@pytest.mark.parametrize("dtype", [F16, F32], ids=["f16", "f32"])
@pytest.mark.parametrize("lda", _STRIDE_CLASSES)
@pytest.mark.parametrize("origin", range(8))
@pytest.mark.parametrize("begin", [0, 1, 2, 4])
def test_int_strides_origins_and_begins_claim_exactly_the_true_alignment(
    dtype, lda, origin, begin
):
    claims, truth = _claim_and_truth(dtype, _int(lda), _int(origin), _int(begin), 16)
    assert claims and all(align == truth for align, _ in claims), (claims, truth)


_RUNTIME_CASES = {
    "runtime lda multiple of 1": (_runtime_stride(1), _int(0), _int(0)),
    "runtime lda multiple of 2": (_runtime_stride(2), _int(0), _int(0)),
    "runtime lda multiple of 4": (_runtime_stride(4), _int(0), _int(0)),
    "runtime lda multiple of 8": (_runtime_stride(8), _int(0), _int(0)),
    "runtime lda multiple of 8, odd origin": (_runtime_stride(8), _int(3), _int(0)),
    "origin from a parameter": (_int(K), _origin_from_parameter(), _int(0)),
    "origin block * 8": (_int(K), _origin_from_block(8, 0), _int(0)),
    "origin block * 8 + 2": (_int(K), _origin_from_block(8, 2), _int(0)),
    "origin block * 8 + 1, runtime lda": (
        _runtime_stride(8),
        _origin_from_block(8, 1),
        _int(0),
    ),
    "runtime begin multiple of 1": (_int(K), _int(0), _runtime_begin(1)),
    "runtime begin multiple of 4": (_int(K), _int(0), _runtime_begin(4)),
    "runtime begin multiple of 8, runtime lda": (
        _runtime_stride(8),
        _int(0),
        _runtime_begin(8),
    ),
}


@pytest.mark.parametrize("dtype", [F16, F32], ids=["f16", "f32"])
@pytest.mark.parametrize(
    "pointer", [16, 4, None], ids=["align16", "align4", "no-align"]
)
@pytest.mark.parametrize("case", list(_RUNTIME_CASES), ids=list(_RUNTIME_CASES))
def test_runtime_strides_origins_and_begins_claim_exactly_the_true_alignment(
    dtype, pointer, case
):
    lda, origin, begin = _RUNTIME_CASES[case]
    claims, truth = _claim_and_truth(dtype, lda, origin, begin, pointer)
    assert claims and all(align == truth for align, _ in claims), (claims, truth)


# --- the stated examples ----------------------------------------------------------------------------


def _copy_claims(lda, *, pointer=16, rows=ROWS, origin_k=0):
    """Load and store TILE through an f16 (rows, K) A with stride ``lda``; ``(load claims, store
    claims)``."""
    b = _builder("copy")
    attrs = {} if pointer is None else {"align": pointer}
    x = b.param("X", PtrType(F16, "global"), **attrs)
    y = b.param("Y", PtrType(F16, "global"), align=16)
    stride = lda(b) if callable(lda) else lda
    td = make_tensor_desc((rows, K), (stride, 1), F16)
    lane = b.thread_id_x()
    frag = load_fragment(b, x, make_window(td, (0, origin_k)), TILE, lane)
    out = make_tensor_desc((rows, K), (K, 1), F16)
    store_fragment(b, y, make_window(out, (0, 0)), frag, lane)
    b.ret()
    return _claims(b, _LOADS), _claims(b, _STORES)


def _runtime_lda(multiple_of):
    return lambda b: b.param("lda", I32, ge=K, le=1 << 16, multiple_of=multiple_of)


def test_an_odd_stride_keeps_the_access_scalar_at_the_element_alignment():
    loads, _ = _copy_claims(K + 1)
    assert set(loads) == {(2, 1)}


def test_a_runtime_stride_claims_its_stated_multiple_and_the_pointer_bounds_it():
    loads, _ = _copy_claims(_runtime_lda(1))
    assert set(loads) == {(2, 1)}
    loads, _ = _copy_claims(_runtime_lda(8))
    assert set(loads) == {(16, 8)}
    loads, _ = _copy_claims(_runtime_lda(8), pointer=4)
    assert set(loads) == {(4, 8)}


def test_a_pointer_with_no_declared_alignment_claims_the_element_size():
    loads, _ = _copy_claims(K, pointer=None)
    assert set(loads) == {(2, 8)}


def test_a_pointer_that_is_not_a_kernel_parameter_claims_the_element_size():
    b = _builder("derived_pointer")
    x = b.param("X", PtrType(F16, "global"), align=16)
    shifted = b.global_ptr_add(x, b.const_i64(32))
    td = make_tensor_desc((ROWS, K), (K, 1), F16)
    load_fragment(b, shifted, make_window(td, (0, 0)), TILE, b.thread_id_x())
    b.ret()
    assert set(_claims(b, _LOADS)) == {(2, 8)}


def test_an_odd_begin_claims_the_element_alignment():
    b = _builder("odd_begin")
    x = b.param("X", PtrType(F16, "global"), align=16)
    td = TensorDesc((ROWS, K), (K, 1), F16, begin=(0, 1))
    load_fragment(b, x, make_window(td, (0, 0)), TILE, b.thread_id_x())
    b.ret()
    assert set(_claims(b, _LOADS)) == {(2, 8)}


def test_an_aligned_tile_keeps_its_full_claim():
    # Every fact a multiple of the 8-element run: the claim is the whole 16-byte vector.
    loads, stores = _copy_claims(K)
    assert set(loads) == {(16, 8)}
    assert set(stores) == {(2, 1)}  # the one-index store form stores element by element


def test_a_checked_window_claims_the_same_and_its_edge_elements_their_own():
    # A ragged K (runtime length, no multiple-of fact): the window is checked and a vector may
    # cross the bound. Its wide accesses keep the claim of a proven window; each element of a
    # vector crossing the bound is accessed alone and claims its own size.
    b = _builder("checked")
    x = b.param("X", PtrType(F16, "global"), align=16)
    y = b.param("Y", PtrType(F16, "global"), align=16)
    ks = b.param("K", I32, ge=0, le=1 << 20)
    td = make_tensor_desc((ROWS, ks), (K, 1), F16)
    lane = b.thread_id_x()
    frag = load_fragment(b, x, make_window(td, (0, 0)), TILE, lane)
    store_fragment(b, y, make_window(td, (0, 0)), frag, lane)
    b.ret()
    assert set(_claims(b, _LOADS)) == {(16, 8), (2, 1)}
    assert set(_claims(b, _STORES)) == {(16, 8), (2, 1)}


# --- a uniform address at an odd offset is a vector-memory load -----------------------------------

# Every lane reads the same 8 halves: a uniform address the backend may serve with a scalar load.
_UNIFORM_ROW = make_tile_desc(
    shape=[8], thread_tile=[8], thread_dist=[1], thread_broadcast=WAVE, wave_size=WAVE
)


def _uniform_row_kernel(column):
    b = _builder("uniform_row")
    x = b.param("X", PtrType(F16, "global"), align=16, readonly=True, noalias=True)
    y = b.param("Y", PtrType(F16, "global"), align=16, noalias=True)
    p = b.param("p", I32)
    td = make_tensor_desc((K,), (1,), F16)
    lane = b.thread_id_x()
    frag = load_fragment(b, x, make_window(td, (column(b, p),)), _UNIFORM_ROW, lane)
    store_fragment(
        b, y, make_window(make_tensor_desc((8,), (1,), F16), (0,)), frag, lane
    )
    b.ret()
    return b


def _data_scalar_loads(body: str) -> int:
    """Scalar loads that do not read the kernel arguments: the first scalar load of the body reads
    the argument segment, and its base register pair is that segment's pointer."""
    loads = re.findall(r"^\s*s_load_\w+\s+[^,]+,\s*(s\[\d+:\d+\])", body, re.MULTILINE)
    if not loads:
        return 0
    return sum(base != loads[0] for base in loads)


def test_a_uniform_address_at_an_odd_runtime_offset_never_becomes_a_scalar_load():
    harness = pytest.importorskip(
        "rocke.tests.helpers.tiling.tensor_transforms._harness"
    )

    def odd(b, p):
        # Any column in [0, 57): the walk proves no multiple-of fact, so the claim is 2 bytes.
        return b.smin(b.smax(p, b.const_i32(0)), b.const_i32(K - 8))

    def aligned(b, p):
        # A multiple of 8: the 16-byte claim is true, and the backend may use a scalar load.
        return b.mul(b.smin(b.smax(p, b.const_i32(0)), b.const_i32(7)), b.const_i32(8))

    b = _uniform_row_kernel(odd)
    assert set(_claims(b, _LOADS)) == {(2, 8)}
    isa = harness.gfx90a_isa(b)
    body = isa[isa.index("<uniform_row>:") :]
    assert _data_scalar_loads(body) == 0
    assert harness.isa_count(body, r"global_load_\w+") == 1

    # The control: the same row at a proven multiple of 8 is read by a scalar load, so the count
    # above sees the case it guards against.
    b = _uniform_row_kernel(aligned)
    assert set(_claims(b, _LOADS)) == {(16, 8)}
    isa = harness.gfx90a_isa(b)
    assert _data_scalar_loads(isa[isa.index("<uniform_row>:") :]) >= 1


# --- GPU: padded-stride GEMMs are bit-exact -----------------------------------------------------------


def _padded(data: np.ndarray, stride: int, fill) -> np.ndarray:
    """``data`` (rows, cols) laid out with row stride ``stride``, the pad filled with ``fill``."""
    out = np.full((data.shape[0], stride), fill, data.dtype)
    out[:, : data.shape[1]] = data
    return out


def _check_gemm(kernel, block, grid, m, n, k, ld, *, c_n_major: bool):
    rng = np.random.default_rng(ld)
    a = rng.integers(-3, 4, size=(m, k)).astype(np.float16)
    bm = rng.integers(-3, 4, size=(n, k)).astype(np.float16)
    reference = a.astype(np.float32) @ bm.astype(np.float32).T
    c_rows, c_cols = (n, m) if c_n_major else (m, n)
    c = np.full((c_rows, ld), SENTINEL, np.float16)
    _, _, c_out = _launch(
        kernel,
        [_padded(a, ld, np.nan), _padded(bm, ld, np.nan), c, m, n, k],
        grid=grid,
        block=block,
    )
    assert _guards_untouched(c_out)
    result = _body(c_out, (c_rows, ld))
    assert np.all(result[:, c_cols:] == SENTINEL), "a C pad element was written"
    product = result[:, :c_cols].astype(np.float32)
    assert np.array_equal(product.T if c_n_major else product, reference)


@pytest.mark.parametrize("pad", [0, 1, 2, 4, 8])
def test_gpu_the_manual_gemm_is_bit_exact_at_every_padded_stride(pad):
    from rocke.helpers.tiling.kernels.tiling_gemm_manual_demo import build_manual_gemm

    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    m = n = k = 64
    ld = k + pad
    kernel, mma = build_manual_gemm(m, n, k, arch=arch, lda=ld, ldb=ld, ldc=ld)
    # C (M, N) is M-contiguous: stored as (N, ldc) rows.
    _check_gemm(
        kernel, mma.wave_size, (n // 16, m // 16, 1), m, n, k, ld, c_n_major=True
    )


@pytest.mark.parametrize("pad", [0, 1, 2, 4, 8])
def test_gpu_the_interleaved_gemm_is_bit_exact_at_every_padded_stride(pad):
    from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import (
        build_interleaved_gemm,
    )

    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    m = n = 128
    k = 64
    ld = n + pad  # one row stride for A, B (rows of K) and C (rows of N), padded alike
    kernel, mma = build_interleaved_gemm(m, n, k, arch=arch, lda=ld, ldb=ld, ldc=ld)
    _check_gemm(
        kernel, mma.wave_size, (n // 64, m // 64, 1), m, n, k, ld, c_n_major=False
    )


def test_gpu_a_wide_access_at_an_odd_runtime_offset_is_bit_exact():
    # An 8-wide f16 load at a column only the run time knows (claimed 2 bytes) reads exactly the
    # 16 columns from there; the copy writes nothing past its destination.
    b = _builder("odd_column_copy")
    x = b.param("X", PtrType(F16, "global"), align=16)
    y = b.param("Y", PtrType(F16, "global"), align=16)
    column = exit_if_tile_outside(b, b.param("column", I32), K - COLS + 1)
    td = make_tensor_desc((ROWS, K), (K, 1), F16)
    out = make_tensor_desc((ROWS, COLS), (COLS, 1), F16)
    lane = b.thread_id_x()
    frag = load_fragment(b, x, make_window(td, (0, column)), TILE, lane)
    store_fragment(b, y, make_window(out, (0, 0)), frag, lane)
    b.ret()
    assert set(_claims(b, _LOADS)) == {(2, 8)}
    src = np.random.default_rng(5).integers(-50, 50, size=(ROWS, K)).astype(np.float16)
    for start in (0, 1, 3, 8, K - COLS):
        _, y_out = _launch(
            b.kernel,
            [src, np.full((ROWS, COLS), SENTINEL, np.float16), start],
            grid=(1, 1, 1),
        )
        assert _guards_untouched(y_out)
        assert np.array_equal(_body(y_out, (ROWS, COLS)), src[:, start : start + COLS])
