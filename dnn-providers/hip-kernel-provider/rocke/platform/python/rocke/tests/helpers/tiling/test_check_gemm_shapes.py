# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for check_gemm_shapes: the build-time check that a GEMM's A, B and C agree on their extents
(offline, no GPU, no IR).

The check reads its tensors positionally -- A (batch..., M, K), B (batch..., N, K), C (batch..., M, N)
-- whatever their strides say about the memory order.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocke.core.ir import F16, F32
from rocke.helpers.tiling import check_gemm_shapes, make_tensor_desc

_READING = (
    "GEMM tensors are read as A (M, K), B (N, K), C (M, N), memory order in the strides; "
    "a B stored N-contiguous (stride 1 on N) is make_tensor_desc((N, K), (1, ldb), dtype)"
)


def _k_contiguous(lengths: tuple[int, ...], dtype=F16):
    """A densely packed tensor whose last axis is stride 1."""
    strides = []
    step = 1
    for length in reversed(lengths):
        strides.append(step)
        step *= length
    return make_tensor_desc(lengths, tuple(reversed(strides)), dtype)


def _message(a, b, c=None) -> str:
    with pytest.raises(ValueError) as excinfo:
        check_gemm_shapes(a, b, c)
    return str(excinfo.value)


def test_agreeing_rank2_gemm_passes_whatever_the_memory_order() -> None:
    # M=64, N=48, K=32. The memory order lives in the strides only, so a K-contiguous and an
    # N-contiguous B described (N, K) both pass, as does an M-contiguous C.
    a = _k_contiguous((64, 32))
    b_k_contiguous = _k_contiguous((48, 32))
    b_n_contiguous = make_tensor_desc((48, 32), (1, 48), F16)
    c_m_contiguous = make_tensor_desc((64, 48), (1, 64), F32)
    assert check_gemm_shapes(a, b_k_contiguous, c_m_contiguous) is None
    assert check_gemm_shapes(a, b_n_contiguous, c_m_contiguous) is None


def test_numpy_integer_extents_agree_with_python_int_extents() -> None:
    # Authors often compute sizes with numpy arithmetic. An extent of numpy.int64(64) is the same
    # extent as 64, so a GEMM mixing the two passes, and a real mismatch is still refused.
    a = make_tensor_desc((np.int64(64), 32), (32, 1), F16)
    b = make_tensor_desc((48, np.int32(32)), (32, 1), F16)
    c = make_tensor_desc((np.int64(64), 48), (1, 64), F32)
    assert check_gemm_shapes(a, b, c) is None
    c_wrong_m = make_tensor_desc((np.int64(65), 48), (1, 65), F32)
    assert "A's M (axis 0) is 64 but C's M (axis 0) is 65" in _message(a, b, c_wrong_m)


def test_different_k_is_refused_with_both_axes_named() -> None:
    assert _message(_k_contiguous((64, 128)), _k_contiguous((48, 64))) == (
        "check_gemm_shapes: A's K (axis 1) is 128 but B's K (axis 1) is 64 -- "
        + _READING
    )


def test_different_m_is_refused_with_both_axes_named() -> None:
    a, b = _k_contiguous((64, 32)), _k_contiguous((48, 32))
    assert _message(a, b, _k_contiguous((16, 48), F32)) == (
        "check_gemm_shapes: A's M (axis 0) is 64 but C's M (axis 0) is 16 -- "
        + _READING
    )


def test_different_n_is_refused_with_both_axes_named() -> None:
    a, b = _k_contiguous((64, 32)), _k_contiguous((48, 32))
    assert _message(a, b, _k_contiguous((64, 24), F32)) == (
        "check_gemm_shapes: B's N (axis 0) is 48 but C's N (axis 1) is 24 -- "
        + _READING
    )


def test_a_b_written_k_n_is_refused_when_its_extents_differ() -> None:
    # The anti-example: B written (K, N) = (32, 48). Read positionally its K is the N extent.
    assert _message(_k_contiguous((64, 32)), _k_contiguous((32, 48))) == (
        "check_gemm_shapes: A's K (axis 1) is 32 but B's K (axis 1) is 48 -- "
        + _READING
    )


def test_a_square_transposed_b_passes() -> None:
    # The documented limit: with M = N = K a B written (K, N) has the same extents as (N, K), so no
    # extent check can see the transpose. Asserted here so the limit stays visible; the bit-exact
    # tests are the backstop.
    a = _k_contiguous((32, 32))
    b_written_k_n = make_tensor_desc(
        (32, 32), (32, 1), F16
    )  # meant as (K, N), N-contiguous
    assert check_gemm_shapes(a, b_written_k_n, _k_contiguous((32, 32), F32)) is None


def test_structured_sparse_shaped_pair_is_refused() -> None:
    # A structured-sparse A stores half of K; the tiling layer issues dense MMAs only, so the
    # K comparison refuses the pair.
    assert _message(_k_contiguous((64, 32)), _k_contiguous((48, 64))) == (
        "check_gemm_shapes: A's K (axis 1) is 32 but B's K (axis 1) is 64 -- "
        + _READING
    )


def test_batched_gemm_passes_and_axes_are_counted_from_the_batch() -> None:
    # (G, M, K), (G, N, K), (G, M, N) with G=4: K is axis 2, M and N are axes 1 and 2.
    a, b = _k_contiguous((4, 64, 32)), _k_contiguous((4, 48, 32))
    assert check_gemm_shapes(a, b, _k_contiguous((4, 64, 48), F32)) is None
    assert _message(a, _k_contiguous((4, 48, 16))) == (
        "check_gemm_shapes: A's K (axis 2) is 32 but B's K (axis 2) is 16 -- "
        + _READING
    )
    assert _message(a, b, _k_contiguous((4, 64, 24), F32)) == (
        "check_gemm_shapes: B's N (axis 1) is 48 but C's N (axis 2) is 24 -- "
        + _READING
    )


def test_an_unlabelled_rank3_operand_is_read_as_batch_m_k() -> None:
    # Nothing labels the axes: a rank-3 A is read (batch, M, K), so its first axis is compared as a
    # batch axis, and its last as K.
    a = _k_contiguous((2, 64, 32))
    assert check_gemm_shapes(a, _k_contiguous((2, 48, 32))) is None
    assert _message(a, _k_contiguous((3, 48, 32))) == (
        "check_gemm_shapes: batch axis 0 is 2 in A but 3 in B -- every batch axis has one extent "
        "across A and B"
    )


def test_batch_extent_mismatch_with_c_is_refused() -> None:
    a, b = _k_contiguous((4, 64, 32)), _k_contiguous((4, 48, 32))
    assert _message(a, b, _k_contiguous((2, 64, 48), F32)) == (
        "check_gemm_shapes: batch axis 0 is 4 in A but 2 in C -- every batch axis has one extent "
        "across A, B and C"
    )


def test_two_batch_axes_are_each_checked() -> None:
    a, b = _k_contiguous((2, 3, 16, 8)), _k_contiguous((2, 3, 24, 8))
    assert check_gemm_shapes(a, b, _k_contiguous((2, 3, 16, 24), F32)) is None
    assert _message(a, _k_contiguous((2, 5, 24, 8))) == (
        "check_gemm_shapes: batch axis 1 is 3 in A but 5 in B -- every batch axis has one extent "
        "across A and B"
    )


def test_different_ranks_are_refused() -> None:
    a3, b2 = _k_contiguous((4, 64, 32)), _k_contiguous((48, 32))
    assert _message(a3, b2, _k_contiguous((4, 64, 48), F32)) == (
        "check_gemm_shapes: A, B and C have ranks 3, 2, 3 -- batched GEMM tensors carry the same "
        "leading batch axes: (G, M, K), (G, N, K), (G, M, N)"
    )
    assert _message(a3, b2) == (
        "check_gemm_shapes: A and B have ranks 3, 2 -- batched GEMM tensors carry the same "
        "leading batch axes: (G, M, K), (G, N, K), (G, M, N)"
    )


def test_a_rank1_operand_is_refused() -> None:
    assert _message(_k_contiguous((32,)), _k_contiguous((48, 32))) == (
        "check_gemm_shapes: A has rank 1 -- a GEMM operand has rank >= 2: (batch..., M, K)"
    )
    assert _message(_k_contiguous((64, 32)), _k_contiguous((32,))) == (
        "check_gemm_shapes: B has rank 1 -- a GEMM operand has rank >= 2: (batch..., N, K)"
    )


def test_c_none_checks_the_a_b_pair_of_a_batch_reduce_gemm() -> None:
    # A batch-reduce GEMM sums over G: C (M, N) has no G, so only its A / B pair is checked.
    a, b = _k_contiguous((8, 64, 32)), _k_contiguous((8, 48, 32))
    assert check_gemm_shapes(a, b) is None
    assert check_gemm_shapes(_k_contiguous((64, 32)), _k_contiguous((48, 32))) is None


def test_docstring_states_the_positional_reading_and_the_domain() -> None:
    doc = " ".join(check_gemm_shapes.__doc__.split())
    assert "positionally as A ``(batch..., M, K)``, B ``(batch..., N, K)``" in doc
    assert "C ``(batch..., M, N)``" in doc
    assert "Do not call it on tensors outside that family" in doc
    assert "transform graph" in doc
    assert "sums over a batch axis" in doc
