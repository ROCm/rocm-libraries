# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the SDPA length adapter.

``kernels.common._sdpa_seqlen`` validates hipDNN's SEQ_LEN_Q / SEQ_LEN_KV
tensors at the host boundary and normalizes them for the kernels: ``sdpa_seqlens``
for padded dense layouts (flat per-batch arrays), ``sdpa_cu_seqlens`` for ragged
packing (cumulative cu_seqlens). No GPU and no torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from kernels.common._sdpa_seqlen import sdpa_cu_seqlens, sdpa_seqlens


def test_valid_lengths_pass_through():
    q = np.array([3, 5, 2, 8], dtype=np.int32)
    kv = np.array([7, 7, 1, 8], dtype=np.int32)
    len_q, len_kv = sdpa_seqlens(q, kv, batch=4, s_q_max=8, s_kv_max=8)
    assert len_q.dtype == np.int32 and len_kv.dtype == np.int32
    assert len_q.shape == (4,) and len_kv.shape == (4,)
    assert len_q.flags["C_CONTIGUOUS"] and len_kv.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(len_q, q)
    np.testing.assert_array_equal(len_kv, kv)


def test_accepts_hipdnn_bshd_shape():
    # hipDNN supplies SEQ_LEN as [B, 1, 1, 1].
    q = np.array([2, 4], dtype=np.int32).reshape(2, 1, 1, 1)
    kv = np.array([4, 4], dtype=np.int32).reshape(2, 1, 1, 1)
    len_q, len_kv = sdpa_seqlens(q, kv, batch=2, s_q_max=4, s_kv_max=4)
    assert len_q.shape == (2,) and len_kv.shape == (2,)
    np.testing.assert_array_equal(len_q, [2, 4])


def test_boundary_lengths_zero_and_max_allowed():
    q = np.array([0, 6], dtype=np.int32)
    kv = np.array([6, 0], dtype=np.int32)
    len_q, len_kv = sdpa_seqlens(q, kv, batch=2, s_q_max=6, s_kv_max=6)
    np.testing.assert_array_equal(len_q, [0, 6])
    np.testing.assert_array_equal(len_kv, [6, 0])


def test_length_above_max_rejected():
    q = np.array([3, 9], dtype=np.int32)  # 9 > s_q_max
    kv = np.array([4, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="SEQ_LEN_Q"):
        sdpa_seqlens(q, kv, batch=2, s_q_max=8, s_kv_max=8)


def test_negative_length_rejected():
    q = np.array([3, 2], dtype=np.int32)
    kv = np.array([-1, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="SEQ_LEN_KV"):
        sdpa_seqlens(q, kv, batch=2, s_q_max=8, s_kv_max=8)


def test_wrong_batch_count_rejected():
    q = np.array([3, 2, 1], dtype=np.int32)  # 3 != batch
    kv = np.array([4, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="SEQ_LEN_Q"):
        sdpa_seqlens(q, kv, batch=2, s_q_max=8, s_kv_max=8)


def test_non_integer_dtype_rejected():
    q = np.array([3.0, 2.0], dtype=np.float32)
    kv = np.array([4, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="integer dtype"):
        sdpa_seqlens(q, kv, batch=2, s_q_max=8, s_kv_max=8)


def test_q_and_kv_bounded_independently():
    # Q valid up to s_q_max, KV up to a different s_kv_max.
    q = np.array([4, 4], dtype=np.int32)
    kv = np.array([16, 10], dtype=np.int32)
    len_q, len_kv = sdpa_seqlens(q, kv, batch=2, s_q_max=4, s_kv_max=16)
    np.testing.assert_array_equal(len_q, [4, 4])
    np.testing.assert_array_equal(len_kv, [16, 10])


# --- ragged / varlen: sdpa_cu_seqlens -------------------------------------


def test_cu_seqlens_exclusive_prefix_sum():
    q = np.array([3, 5, 2, 8], dtype=np.int32)
    kv = np.array([7, 1, 4, 6], dtype=np.int32)
    cu_q, cu_kv = sdpa_cu_seqlens(q, kv, batch=4)
    assert cu_q.dtype == np.int32 and cu_kv.dtype == np.int32
    assert cu_q.shape == (5,) and cu_kv.shape == (5,)
    assert cu_q.flags["C_CONTIGUOUS"] and cu_kv.flags["C_CONTIGUOUS"]
    # Leading 0, exclusive prefix sum, last element = packed token total.
    np.testing.assert_array_equal(cu_q, [0, 3, 8, 10, 18])
    np.testing.assert_array_equal(cu_kv, [0, 7, 8, 12, 18])
    assert cu_q[0] == 0 and int(cu_q[-1]) == 18


def test_cu_seqlens_zero_length_sequences_allowed():
    q = np.array([0, 4, 0, 2], dtype=np.int32)
    kv = np.array([4, 0, 0, 2], dtype=np.int32)
    cu_q, cu_kv = sdpa_cu_seqlens(q, kv, batch=4)
    np.testing.assert_array_equal(cu_q, [0, 0, 4, 4, 6])
    np.testing.assert_array_equal(cu_kv, [0, 4, 4, 4, 6])


def test_cu_seqlens_negative_length_rejected():
    q = np.array([3, 2], dtype=np.int32)
    kv = np.array([4, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="SEQ_LEN_KV"):
        sdpa_cu_seqlens(q, kv, batch=2)


def test_cu_seqlens_wrong_batch_count_rejected():
    q = np.array([3, 2, 1], dtype=np.int32)  # 3 != batch
    kv = np.array([4, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="SEQ_LEN_Q"):
        sdpa_cu_seqlens(q, kv, batch=2)


def test_cu_seqlens_non_integer_dtype_rejected():
    q = np.array([3.0, 2.0], dtype=np.float32)
    kv = np.array([4, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="integer dtype"):
        sdpa_cu_seqlens(q, kv, batch=2)
