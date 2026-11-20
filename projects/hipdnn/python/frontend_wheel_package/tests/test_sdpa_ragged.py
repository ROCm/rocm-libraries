# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Packed (THD, cu_seqlens) SDPA graphs built from Python through the ragged-offset bindings.

A ragged tensor holds its sequences back to back; its INT32 ragged-offset tensor
``[batch + 1, 1, 1, 1]`` holds each sequence's start (cu_seqlens), counted in tokens
when the ragged-offset multiplier is the token stride (RFC 0014, SEM-028). The
tests build such a graph and its padded equivalent (dense ``[B, max_s, H, D]`` with
``seq_len_q``/``seq_len_kv`` and the padding mask), check what validation and
serialization keep, and run both through hipDNN's CPU reference graph executor
when ``HIPDNN_REF_CAPI_LIB`` names dnn-benchmarking's ``libhipdnn_ref_capi.so``
(``tools/hipdnn_ref``), comparing them with each other and with a float64 oracle.
"""

import ctypes
import json
import os

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

Q_LENS = (5, 2, 3)
KV_LENS = (7, 4, 3)
HEADS = 2
HEAD_DIM = 8
SCALE = 0.4

UID_Q, UID_K, UID_V, UID_O, UID_STATS = 1, 2, 3, 4, 5
UID_Q_OFFSETS, UID_KV_OFFSETS, UID_SEQ_LEN_Q, UID_SEQ_LEN_KV = 6, 7, 8, 9


def _require_sdpa():
    if not hasattr(hipdnn.Graph, "sdpa"):
        pytest.skip("SDPA disabled")


def _cu_seqlens(lengths):
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)


def _tensor(uid, name, dims, strides, data_type=hipdnn.DataType.FLOAT):
    tensor = hipdnn.Tensor.create(dims, data_type)
    return tensor.set_uid(uid).set_name(name).set_stride(strides)


def _graph():
    graph = hipdnn.Graph()
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    return graph


def _attributes(causal):
    attributes = hipdnn.SdpaAttributes().set_attn_scale(SCALE).set_generate_stats(True)
    if causal:
        attributes.set_causal_mask_bottom_right(True)
    return attributes


def _finish(graph, outputs, o_dims, o_strides, stats_dims, stats_strides):
    o, stats = outputs
    o.set_uid(UID_O).set_name("o").set_dim(o_dims).set_stride(o_strides).set_output(
        True
    )
    stats.set_uid(UID_STATS).set_name("stats").set_data_type(hipdnn.DataType.FLOAT)
    stats.set_dim(stats_dims).set_stride(stats_strides).set_output(True)
    return o, stats


def build_packed_graph(causal=False):
    """Q/K/V/O packed token by token ([T, H, D]) and the LSE packed [T, H], all through
    cu_seqlens with the token stride as the multiplier: FlashAttention's varlen call."""
    batch, token = len(Q_LENS), HEADS * HEAD_DIM
    max_q, max_kv = max(Q_LENS), max(KV_LENS)
    graph = _graph()
    q_offsets = _tensor(
        UID_Q_OFFSETS,
        "cu_seqlens_q",
        [batch + 1, 1, 1, 1],
        [1, 1, 1, 1],
        hipdnn.DataType.INT32,
    )
    kv_offsets = _tensor(
        UID_KV_OFFSETS,
        "cu_seqlens_k",
        [batch + 1, 1, 1, 1],
        [1, 1, 1, 1],
        hipdnn.DataType.INT32,
    )

    def packed(uid, name, max_s, offsets):
        tensor = _tensor(
            uid,
            name,
            [batch, HEADS, max_s, HEAD_DIM],
            [max_s * token, HEAD_DIM, token, 1],
        )
        return tensor.set_ragged_offset(offsets).set_ragged_offset_multiplier(token)

    q = packed(UID_Q, "q", max_q, q_offsets)
    k = packed(UID_K, "k", max_kv, kv_offsets)
    v = packed(UID_V, "v", max_kv, kv_offsets)
    o, stats = _finish(
        graph,
        graph.sdpa(q, k, v, _attributes(causal)),
        [batch, HEADS, max_q, HEAD_DIM],
        [max_q * token, HEAD_DIM, token, 1],
        [batch, HEADS, max_q, 1],
        [max_q * HEADS, 1, HEADS, 1],
    )
    o.set_ragged_offset(q_offsets).set_ragged_offset_multiplier(token)
    stats.set_ragged_offset(q_offsets).set_ragged_offset_multiplier(HEADS)
    return graph


def build_padded_graph(causal=False):
    """The same attention on dense [B, max_s, H, D] buffers with per-batch lengths."""
    batch, token = len(Q_LENS), HEADS * HEAD_DIM
    max_q, max_kv = max(Q_LENS), max(KV_LENS)
    graph = _graph()

    def dense(uid, name, max_s):
        return _tensor(
            uid,
            name,
            [batch, HEADS, max_s, HEAD_DIM],
            [max_s * token, HEAD_DIM, token, 1],
        )

    attributes = _attributes(causal).set_padding_mask(True)
    attributes.set_seq_len_q(
        _tensor(
            UID_SEQ_LEN_Q,
            "seq_len_q",
            [batch, 1, 1, 1],
            [1, 1, 1, 1],
            hipdnn.DataType.INT32,
        )
    )
    attributes.set_seq_len_kv(
        _tensor(
            UID_SEQ_LEN_KV,
            "seq_len_kv",
            [batch, 1, 1, 1],
            [1, 1, 1, 1],
            hipdnn.DataType.INT32,
        )
    )
    _finish(
        graph,
        graph.sdpa(
            dense(UID_Q, "q", max_q),
            dense(UID_K, "k", max_kv),
            dense(UID_V, "v", max_kv),
            attributes,
        ),
        [batch, HEADS, max_q, HEAD_DIM],
        [max_q * token, HEAD_DIM, token, 1],
        [batch, HEADS, max_q, 1],
        [max_q * HEADS, 1, HEADS, 1],
    )
    return graph


class TestPackedSdpaGraph:
    """A packed THD graph built from Python validates and serializes with its offsets."""

    def test_validates_and_serializes_the_ragged_fields(self):
        _require_sdpa()
        graph = build_packed_graph()
        result = graph.validate()
        assert result.is_good(), result.get_message()
        document = json.loads(graph.to_json())
        tensors = {tensor["uid"]: tensor for tensor in document["tensors"]}
        token = HEADS * HEAD_DIM
        for uid in (UID_Q, UID_O):
            assert tensors[uid]["ragged_offset_tensor_uid"] == UID_Q_OFFSETS
            assert tensors[uid]["ragged_offset_multiplier"] == token
        for uid in (UID_K, UID_V):
            assert tensors[uid]["ragged_offset_tensor_uid"] == UID_KV_OFFSETS
            assert tensors[uid]["ragged_offset_multiplier"] == token
        assert tensors[UID_STATS]["ragged_offset_tensor_uid"] == UID_Q_OFFSETS
        assert tensors[UID_STATS]["ragged_offset_multiplier"] == HEADS
        # A ragged offset multiplier needs engine plugin API 1.4.0 (RFC 0014).
        version = document["min_required_engine_api_version"]
        assert (version["major"], version["minor"]) == (1, 4)

        restored = hipdnn.Graph()
        assert restored.from_json(json.dumps(document)).is_good()
        assert restored.to_json() == graph.to_json()
        assert restored.to_binary() == graph.to_binary()


# ---------------------------------------------------------------------------
# The CPU reference graph executor, through dnn-benchmarking's C interface
# ---------------------------------------------------------------------------

_KIND_CPU = 1
_MSG_LEN = 4096


def _reference_library():
    path = os.environ.get("HIPDNN_REF_CAPI_LIB")
    if not path:
        pytest.skip(
            "HIPDNN_REF_CAPI_LIB is not set (dnn-benchmarking tools/hipdnn_ref)"
        )
    if not os.path.isfile(path):
        pytest.fail(f"HIPDNN_REF_CAPI_LIB={path!r} is not a file")
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    lib.hipdnn_ref_is_applicable.restype = ctypes.c_int
    lib.hipdnn_ref_is_applicable.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.hipdnn_ref_execute.restype = ctypes.c_int
    lib.hipdnn_ref_execute.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    return lib


def _run_cpu_reference(lib, graph, buffers):
    assert graph.validate().is_good()
    blob = bytes(graph.to_binary())
    blob_buffer = ctypes.create_string_buffer(blob, len(blob))
    message = ctypes.create_string_buffer(_MSG_LEN)
    applicable = lib.hipdnn_ref_is_applicable(
        _KIND_CPU, blob_buffer, len(blob), message, _MSG_LEN
    )
    assert applicable == 1, message.value.decode(errors="replace")
    uids = (ctypes.c_int64 * len(buffers))(*buffers.keys())
    pointers = (ctypes.c_void_p * len(buffers))(
        *(array.ctypes.data for array in buffers.values())
    )
    status = lib.hipdnn_ref_execute(
        _KIND_CPU,
        blob_buffer,
        len(blob),
        len(buffers),
        uids,
        pointers,
        message,
        _MSG_LEN,
    )
    assert status == 0, message.value.decode(errors="replace")


def _sequences():
    rng = np.random.default_rng(7)
    q = [rng.uniform(-1, 1, (n, HEADS, HEAD_DIM)).astype(np.float32) for n in Q_LENS]
    k = [rng.uniform(-1, 1, (n, HEADS, HEAD_DIM)).astype(np.float32) for n in KV_LENS]
    v = [rng.uniform(-1, 1, (n, HEADS, HEAD_DIM)).astype(np.float32) for n in KV_LENS]
    return q, k, v


def _oracle(q, k, v, causal):
    """Per sequence, float64: softmax(scale * q k^T + bottom-right causal mask) v."""
    outputs, lses = [], []
    for qs, ks, vs in zip(q, k, v):
        scores = (
            np.einsum("ihd,jhd->hij", qs.astype(np.float64), ks.astype(np.float64))
            * SCALE
        )
        if causal:
            rows = np.arange(len(qs))[:, None] + len(ks) - len(qs)
            scores = np.where(np.arange(len(ks))[None, :] <= rows, scores, -np.inf)
        top = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - top)
        total = weights.sum(axis=-1, keepdims=True)
        outputs.append(
            np.einsum("hij,jhd->ihd", weights / total, vs.astype(np.float64))
        )
        lses.append((top + np.log(total))[..., 0].T)
    return outputs, lses


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_packed_graph_equals_its_padded_equivalent_on_the_cpu_reference(causal):
    """The packed graph and the padded one give the same O and LSE on every live row,
    and both equal the float64 oracle."""
    _require_sdpa()
    lib = _reference_library()
    q, k, v = _sequences()
    batch, max_q, max_kv = len(Q_LENS), max(Q_LENS), max(KV_LENS)

    packed_o = np.full((sum(Q_LENS), HEADS, HEAD_DIM), np.nan, np.float32)
    packed_lse = np.full((sum(Q_LENS), HEADS), np.nan, np.float32)
    _run_cpu_reference(
        lib,
        build_packed_graph(causal),
        {
            UID_Q: np.ascontiguousarray(np.concatenate(q)),
            UID_K: np.ascontiguousarray(np.concatenate(k)),
            UID_V: np.ascontiguousarray(np.concatenate(v)),
            UID_O: packed_o,
            UID_STATS: packed_lse,
            UID_Q_OFFSETS: _cu_seqlens(Q_LENS),
            UID_KV_OFFSETS: _cu_seqlens(KV_LENS),
        },
    )

    def pad(sequences, max_s):
        dense = np.zeros((batch, max_s, HEADS, HEAD_DIM), np.float32)
        for b, sequence in enumerate(sequences):
            dense[b, : len(sequence)] = sequence
        return dense

    padded_o = np.full((batch, max_q, HEADS, HEAD_DIM), np.nan, np.float32)
    padded_lse = np.full((batch, max_q, HEADS), np.nan, np.float32)
    _run_cpu_reference(
        lib,
        build_padded_graph(causal),
        {
            UID_Q: pad(q, max_q),
            UID_K: pad(k, max_kv),
            UID_V: pad(v, max_kv),
            UID_O: padded_o,
            UID_STATS: padded_lse,
            UID_SEQ_LEN_Q: np.array(Q_LENS, np.int32),
            UID_SEQ_LEN_KV: np.array(KV_LENS, np.int32),
        },
    )

    expected_o, expected_lse = _oracle(q, k, v, causal)
    starts = _cu_seqlens(Q_LENS)
    for b, n in enumerate(Q_LENS):
        rows = slice(starts[b], starts[b] + n)
        np.testing.assert_allclose(packed_o[rows], padded_o[b, :n], rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            packed_lse[rows], padded_lse[b, :n], rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(packed_o[rows], expected_o[b], rtol=0, atol=2e-5)
        np.testing.assert_allclose(packed_lse[rows], expected_lse[b], rtol=0, atol=2e-5)
