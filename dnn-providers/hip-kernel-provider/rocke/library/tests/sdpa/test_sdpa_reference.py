# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``sdpa_reference_from_request`` against independent oracles.

Two oracles, neither sharing the vectorised attention math it delegates to:

- a scalar fp64 loop that transliterates hipDNN's ``CpuFpReferenceSdpa::forward``
  (bias indexed the way its ``computeMaskIndex`` does, mask from ``keep``),
  extended per batch the way hipDNN's golden generator runs GROUP mode;
- PyTorch's math SDPA backend, which is what hipDNN's golden generator uses,
  run per batch with an additive mask (skipped without torch).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import replace

import numpy as np
import pytest

from dispatch.sdpa import SdpaBias, SdpaFp8, SdpaTensor
from rocke.numeric.references import keep, sdpa_reference_from_request
from rocke.numeric.sdpa_request_reference import _strided_view
from tests.sdpa.request_cases import (
    CASES,
    bhsd,
    bshd,
    make_request,
    make_tensors,
    packed,
)

NOT_IMPLEMENTED = {
    "alibi": "alibi",
    "dropout": "dropout",
    "paged": "paged",
    "block_mask": "block_mask",
    "sinks": "sinks",
    "fp8_s_o": "fp8_s_o",
    "max_sum_exp": "max_sum_exp",
}


def _lengths(req, tensors, name, ragged_key, extent):
    if getattr(req, name) is not None:
        return np.asarray(tensors[name]).reshape(-1)
    if ragged_key in tensors:
        return np.diff(np.asarray(tensors[ragged_key]).reshape(-1))
    return np.full(req.batch, extent)


def _unpack(req, tensors, name, lengths):
    t = getattr(req, name)
    if not t.ragged:
        return np.asarray(tensors[name], dtype=np.float64)
    out = np.zeros(t.dims)
    flat, row = np.asarray(tensors[name]).reshape(-1), 0
    _, h, _, d = t.dims
    for b, n in enumerate(lengths):
        out[b, :, :n] = (
            flat[row * h * d : (row + n) * h * d].reshape(n, h, d).transpose(1, 0, 2)
        )
        row += n
    return out


def loop_reference(req, tensors):
    """Scalar fp64 hipDNN forward; rows past a batch's length: O = 0, LSE = -inf."""
    batch, h_q, s_q, d = req.q.dims
    h_k, s_kv, h_v, d_v = req.k.dims[1], req.k.dims[2], req.v.dims[1], req.v.dims[3]
    len_q = _lengths(req, tensors, "seq_len_q", "ragged_offset_q", s_q)
    len_kv = _lengths(req, tensors, "seq_len_kv", "ragged_offset_k", s_kv)
    ragged_q = np.diff(tensors["ragged_offset_q"]) if req.q.ragged else None
    ragged_k = np.diff(tensors["ragged_offset_k"]) if req.k.ragged else None
    q = _unpack(req, tensors, "q", ragged_q)
    k = _unpack(req, tensors, "k", ragged_k)
    v = _unpack(req, tensors, "v", ragged_k)
    scale = (
        float(np.asarray(tensors["scale"]).reshape(-1)[0])
        if req.scale_source == "tensor"
        else req.scale
    )
    bias = None if req.bias is None else np.asarray(tensors["bias"], np.float64)

    def descale(name, b, kv_head, heads):
        if req.fp8 is None or getattr(req.fp8, name) is None:
            return 1.0
        x = np.asarray(tensors[name]).reshape(-1)
        return (
            float(x[0]) if x.size == 1 else float(x.reshape(batch, heads)[b, kv_head])
        )

    o = np.zeros((batch, h_q, s_q, d_v))
    lse = np.full((batch, h_q, s_q, 1), -np.inf)
    for b in range(batch):
        sq, skv = int(len_q[b]), int(len_kv[b])
        for h in range(h_q):
            kh, vh = h // (h_q // h_k), h // (h_q // h_v)
            dqk = descale("descale_q", b, kh, h_k) * descale("descale_k", b, kh, h_k)
            dv = descale("descale_v", b, vh, h_v)
            for i in range(sq):
                scores = []
                for j in range(skv):
                    s = float(np.dot(q[b, h, i], k[b, kh, j])) * scale * dqk
                    if bias is not None:
                        dims = bias.shape
                        idx = (b, h, i, j)[4 - len(dims) :]
                        s += bias[tuple(0 if n == 1 else x for n, x in zip(dims, idx))]
                    if not keep(
                        i, j, sq, skv, req.diagonal, req.left_bound, req.right_bound
                    ):
                        s = -math.inf
                    scores.append(s)
                m = max(scores, default=-math.inf)
                if m == -math.inf:
                    continue
                p = [math.exp(s - m) for s in scores]
                total = sum(p)
                o[b, h, i] = sum(p[j] / total * v[b, vh, j] for j in range(skv)) * dv
                lse[b, h, i, 0] = m + math.log(total)
    return o, lse


def _check(req, tensors, atol=2e-5):
    got = sdpa_reference_from_request(req, tensors)
    want_o, want_lse = loop_reference(req, tensors)
    np.testing.assert_allclose(got["o"], want_o, atol=atol, rtol=0)
    np.testing.assert_array_equal(np.isfinite(got["stats"]), np.isfinite(want_lse))
    finite = np.isfinite(want_lse)
    np.testing.assert_allclose(
        got["stats"][finite], want_lse[finite], atol=atol, rtol=0
    )
    return got


REFERENCE_CASES = sorted(set(CASES) - set(NOT_IMPLEMENTED))


@pytest.mark.parametrize("name", REFERENCE_CASES)
def test_case_table_matches_loop_oracle(name):
    req = CASES[name]
    # crc32, not hash(): str hashes are randomized per process, so a failure
    # here could not be replayed.
    _check(req, make_tensors(req, seed=zlib.crc32(name.encode()) % 1000))


@pytest.mark.parametrize("name", sorted(NOT_IMPLEMENTED))
def test_unsupported_features_raise_by_name(name):
    req = CASES[name]
    with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED[name]):
        sdpa_reference_from_request(req, make_tensors(req))


@pytest.mark.parametrize("diagonal", ["top_left", "bottom_right"])
@pytest.mark.parametrize(
    "left, right", [(-1, 0), (1, 0), (2, 1), (0, 3), (3, -1), (0, 0)]
)
@pytest.mark.parametrize("s_q, s_kv", [(5, 5), (3, 7), (7, 3)])
def test_mask_agrees_with_keep(diagonal, left, right, s_q, s_kv):
    # Q = 0 makes every kept score equal, and V = I_{s_kv} makes O[q] the
    # attention row itself, so O > 0 exactly where keep() is true.
    req = make_request(
        b=1,
        h_q=1,
        h_k=1,
        s_q=s_q,
        s_kv=s_kv,
        d=4,
        d_v=s_kv,
        diagonal=diagonal,
        left_bound=left,
        right_bound=right,
    )
    v = np.eye(s_kv, dtype=np.float32)[None, None]
    tensors = {"q": np.zeros((1, 1, s_q, 4)), "k": np.ones((1, 1, s_kv, 4)), "v": v}
    o = sdpa_reference_from_request(req, tensors)["o"][0, 0]
    want = np.array(
        [
            [keep(q, k, s_q, s_kv, diagonal, left, right) for k in range(s_kv)]
            for q in range(s_q)
        ]
    )
    np.testing.assert_array_equal(o > 0, want)
    rows = want.any(axis=1)
    np.testing.assert_allclose(o[rows].sum(axis=1), 1.0, atol=1e-6)


def test_fully_masked_row_is_zero_and_minus_inf():
    # Bottom-right causal with s_q > s_kv leaves the first s_q - s_kv rows empty.
    req = make_request(s_q=6, s_kv=3, right_bound=0, diagonal="bottom_right")
    out = _check(req, make_tensors(req))
    assert np.all(out["o"][:, :, :3] == 0)
    assert np.all(out["stats"][:, :, :3] == -np.inf)
    assert np.all(np.isfinite(out["stats"][:, :, 3:]))


def test_padding_uses_effective_lengths_for_the_diagonal():
    seq = packed((2, 1, 1, 1), "int32")
    req = make_request(
        s_q=8,
        s_kv=8,
        right_bound=0,
        diagonal="bottom_right",
        seq_len_q=seq,
        seq_len_kv=seq,
    )
    tensors = make_tensors(req)
    tensors["seq_len_q"] = np.array([3, 8])
    tensors["seq_len_kv"] = np.array([6, 2])
    out = _check(req, tensors)
    np.testing.assert_array_equal(out["row_valid"][0], np.arange(8) < 3)
    assert np.all(out["o"][0, :, 3:] == 0) and np.all(out["stats"][0, :, 3:] == -np.inf)
    # Batch 0 is a 3x6 bottom-right problem: every row sees at least 4 keys, so
    # it must equal running that batch alone, unpadded.
    alone = make_request(b=1, s_q=3, s_kv=6, right_bound=0, diagonal="bottom_right")
    solo = sdpa_reference_from_request(
        alone,
        {n: tensors[n][:1, :, : (3 if n == "q" else 6)] for n in ("q", "k", "v")},
    )
    np.testing.assert_allclose(out["o"][:1, :, :3], solo["o"], atol=1e-6)


@pytest.mark.parametrize(
    "bias_dims",
    [
        (2, 4, 8, 8),
        (1, 4, 8, 8),
        (2, 1, 8, 8),
        (8, 8),
        (8, 1),
        (1, 8),
        (8,),
        (2, 1, 1, 8),
    ],
)
def test_bias_broadcast(bias_dims):
    req = make_request(right_bound=0, bias=SdpaBias(packed(bias_dims)))
    tensors = make_tensors(req)
    tensors["bias"] = (
        np.random.default_rng(1).standard_normal(bias_dims).astype(np.float32)
    )
    _check(req, tensors)


def test_gqa_with_different_k_and_v_heads():
    req = make_request(h_q=8, h_k=2, h_v=4, d=16, d_v=8, left_bound=2, right_bound=1)
    _check(req, make_tensors(req))


@pytest.mark.parametrize("per_head", [False, True])
def test_fp8_descales_index_by_kv_head(per_head):
    dims = (2, 2, 1, 1) if per_head else (1,)
    descale = SdpaFp8(
        descale_q=packed(dims), descale_k=packed(dims), descale_v=packed(dims)
    )
    req = make_request(
        h_q=4, h_k=2, dtype="fp8_e4m3", o=bshd(2, 4, 8, 16, "bf16"), fp8=descale
    )
    _check(req, make_tensors(req))


def test_scale_tensor_value_is_used():
    req = CASES["scale_tensor"]
    tensors = make_tensors(req)
    base = sdpa_reference_from_request(req, tensors)["o"]
    tensors["scale"] = np.array([2.0], np.float32)
    assert not np.allclose(sdpa_reference_from_request(req, tensors)["o"], base)
    _check(req, tensors)


def _to_buffer(logical: np.ndarray, t: SdpaTensor, pad: int = 0) -> np.ndarray:
    size = 1 + sum((n - 1) * s for n, s in zip(t.dims, t.strides)) + pad
    flat = np.full(size, np.nan, dtype=np.float32)
    view = np.lib.stride_tricks.as_strided(
        flat, shape=t.dims, strides=tuple(s * flat.itemsize for s in t.strides)
    )
    view[...] = logical
    return flat


@pytest.mark.parametrize("case", ["dense_bhsd", "batch_stride", "strided", "d_stride"])
def test_flat_buffers_are_read_through_strides(case):
    # Unused buffer slots hold NaN, so reading one poisons the output.
    req = CASES[case]
    logical = make_tensors(req)
    buffers = {
        n: _to_buffer(logical[n], getattr(req, n), pad=7) for n in ("q", "k", "v")
    }
    want = sdpa_reference_from_request(req, logical)
    got = sdpa_reference_from_request(req, buffers)
    np.testing.assert_array_equal(got["o"], want["o"])


def test_ragged_equals_padded():
    ragged = CASES["ragged"]
    tensors = make_tensors(ragged, seed=3)
    out = _check(ragged, tensors)
    # The same problem as dense operands plus seq_len tensors.
    seq = packed((2, 1, 1, 1), "int32")
    dense = make_request(seq_len_q=seq, seq_len_kv=seq)
    lq = np.diff(tensors["ragged_offset_q"])
    lk = np.diff(tensors["ragged_offset_k"])
    padded = {
        n: _unpack(ragged, tensors, n, lq if n == "q" else lk).astype(np.float32)
        for n in ("q", "k", "v")
    }
    padded.update(seq_len_q=lq, seq_len_kv=lk)
    np.testing.assert_allclose(
        sdpa_reference_from_request(dense, padded)["o"], out["o"], atol=1e-6
    )


def test_ragged_offsets_must_be_whole_rows():
    req = CASES["ragged"]
    tensors = make_tensors(req)
    tensors["ragged_offset_q"] = tensors["ragged_offset_q"] * 2 + 1
    tensors["q"] = np.zeros(10_000, np.float32)
    req = replace(req, q=replace(req.q, ragged_offset_multiplier=1))
    with pytest.raises(ValueError, match="whole sequence rows"):
        sdpa_reference_from_request(req, tensors)


def test_ragged_k_and_v_lengths_must_match():
    """A V packed with fewer tokens than K would pair a key with a zero value row."""
    req = CASES["ragged"]
    tensors = make_tensors(req)
    lengths = np.diff(tensors["ragged_offset_k"])
    lengths[0] -= 1
    tensors["ragged_offset_v"] = np.concatenate([[0], np.cumsum(lengths)])
    with pytest.raises(ValueError, match="ragged K lengths .* and V lengths"):
        sdpa_reference_from_request(req, tensors)


def test_seq_lengths_are_read_through_strides():
    """A (2, 1, 1, 1) length tensor with batch stride 2 over [3, 0, 5] is [3, 5]."""
    strided = SdpaTensor("int32", (2, 1, 1, 1), (2, 1, 1, 1))
    req = make_request(seq_len_q=strided, seq_len_kv=strided)
    tensors = make_tensors(req)
    tensors["seq_len_q"] = tensors["seq_len_kv"] = np.array([3, 0, 5])
    got = sdpa_reference_from_request(req, tensors)
    logical = dict(tensors, seq_len_q=np.array([3, 5]), seq_len_kv=np.array([3, 5]))
    seq = packed((2, 1, 1, 1), "int32")
    want = _check(replace(req, seq_len_q=seq, seq_len_kv=seq), logical)
    np.testing.assert_array_equal(got["row_valid"], want["row_valid"])
    np.testing.assert_array_equal(got["o"], want["o"])


def test_fp8_descales_are_read_through_strides():
    """Flattening a (B, H_kv) descale stored head-major would swap batches and heads."""
    t = SdpaTensor("fp32", (2, 2, 1, 1), (1, 2, 0, 0))
    req = make_request(
        h_q=4,
        h_k=2,
        dtype="fp8_e4m3",
        o=bshd(2, 4, 8, 16, "bf16"),
        fp8=SdpaFp8(descale_q=t, descale_k=t, descale_v=t),
    )
    logical = make_tensors(req)
    want = _check(req, logical)
    buffers = dict(logical)
    for name in ("descale_q", "descale_k", "descale_v"):
        buffers[name] = _to_buffer(logical[name], t)
    got = sdpa_reference_from_request(req, buffers)
    np.testing.assert_array_equal(got["o"], want["o"])


# --- malformed operands raise instead of reading the wrong elements ---------------

_SEQ = packed((2, 1, 1, 1), "int32")
_FP8 = {"dtype": "fp8_e4m3", "o": bshd(2, 4, 8, 16, "bf16")}

# (id, request, tensors to override, match). Offsets are in tokens; the ragged
# case's multiplier (H * D = 64) turns them into elements.
MALFORMED_OPERANDS = [
    (
        "flat_buffer_too_short",
        CASES["dense_bhsd"],
        {"q": np.zeros(10)},
        "buffer has 10",
    ),
    (
        "shape_not_logical_or_flat",
        CASES["dense_bhsd"],
        {"q": np.zeros((3, 5))},
        "neither the logical dims",
    ),
    (
        "ragged_offset_count",
        CASES["ragged"],
        {"ragged_offset_q": np.array([0, 3])},
        "B \\+ 1 = 3 entries, got 2",
    ),
    (
        "ragged_length_negative",
        CASES["ragged"],
        {"ragged_offset_q": np.array([0, 5, 3])},
        "\\[5, -2\\] must be in \\[0, S_max=8\\]",
    ),
    (
        "ragged_length_above_extent",
        CASES["ragged"],
        {"ragged_offset_q": np.array([0, 9, 10])},
        "\\[9, 1\\] must be in \\[0, S_max=8\\]",
    ),
    (
        "seq_len_count",
        make_request(seq_len_q=packed((3, 1, 1, 1), "int32")),
        {"seq_len_q": np.array([1, 2, 3])},
        "one length per batch \\(2\\), got 3",
    ),
    (
        "seq_len_above_extent",
        make_request(seq_len_q=_SEQ),
        {"seq_len_q": np.array([9, 2])},
        "\\[9, 2\\] must be in \\[0, 8\\]",
    ),
    (
        "seq_len_negative",
        make_request(seq_len_kv=_SEQ),
        {"seq_len_kv": np.array([-1, 2])},
        "\\[-1, 2\\] must be in \\[0, 8\\]",
    ),
    (
        "descale_size",
        make_request(fp8=SdpaFp8(descale_q=packed((3,))), **_FP8),
        {},
        "descale_q: must be a scalar or \\(B, H_kv, 1, 1\\)",
    ),
    (
        "seq_len_above_ragged_length",
        replace(CASES["ragged"], seq_len_q=_SEQ),
        {
            "q": np.zeros(7 * 64, np.float32),
            "ragged_offset_q": np.array([0, 3, 7]),
            "seq_len_q": np.array([4, 4]),
        },
        "seq_len_q \\[4, 4\\] exceeds the ragged lengths \\[3, 4\\]",
    ),
]


@pytest.mark.parametrize(
    "req, overrides, match",
    [row[1:] for row in MALFORMED_OPERANDS],
    ids=[row[0] for row in MALFORMED_OPERANDS],
)
def test_malformed_operands_are_rejected(req, overrides, match):
    tensors = make_tensors(req)
    tensors.update(overrides)
    with pytest.raises(ValueError, match=match):
        sdpa_reference_from_request(req, tensors)


def test_ragged_operand_without_its_offsets_is_rejected():
    tensors = make_tensors(CASES["ragged"])
    del tensors["ragged_offset_q"]
    with pytest.raises(KeyError, match="needs tensors\\['ragged_offset_q'\\]"):
        sdpa_reference_from_request(CASES["ragged"], tensors)


def test_negative_strides_are_rejected():
    """``SdpaTensor`` rejects these first; the oracle reads requests by attribute,
    so a duck-typed request reaches this check."""
    with pytest.raises(ValueError, match="negative strides \\(-1,\\)"):
        _strided_view("q", np.zeros(8, np.float32), 0, (2,), (-1,))


# --- PyTorch math SDPA, the hipDNN golden generator's backend --------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"right_bound": 0},
        {"s_q": 5, "s_kv": 9, "right_bound": 0, "diagonal": "bottom_right"},
        {
            "s_q": 9,
            "s_kv": 5,
            "left_bound": 2,
            "right_bound": 0,
            "diagonal": "bottom_right",
        },
        {"h_q": 8, "h_k": 2, "left_bound": 3},
    ],
)
def test_matches_torch_math_sdpa_per_batch(overrides):
    torch = pytest.importorskip("torch")
    seq = packed((2, 1, 1, 1), "int32")
    req = make_request(
        layout=bhsd,
        bias=SdpaBias(packed((1, 1, 1, overrides.get("s_kv", 8)))),
        seq_len_q=seq,
        seq_len_kv=seq,
        stats=packed((2, overrides.get("h_q", 4), overrides.get("s_q", 8), 1)),
        **overrides,
    )
    tensors = make_tensors(req, seed=7)
    got = sdpa_reference_from_request(req, tensors)
    q, k, v = (torch.from_numpy(tensors[n]) for n in ("q", "k", "v"))
    bias = torch.from_numpy(tensors["bias"]).reshape(1, 1, 1, -1)
    group = req.h_q // req.h_k
    for b in range(req.batch):
        sq, skv = int(tensors["seq_len_q"][b]), int(tensors["seq_len_kv"][b])
        keep_mask = torch.tensor(
            [
                [
                    keep(i, j, sq, skv, req.diagonal, req.left_bound, req.right_bound)
                    for j in range(skv)
                ]
                for i in range(sq)
            ]
        )
        mask = bias[..., :skv] + torch.where(keep_mask, 0.0, float("-inf"))
        qb, kb, vb = q[b : b + 1, :, :sq], k[b : b + 1, :, :skv], v[b : b + 1, :, :skv]
        kb, vb = kb.repeat_interleave(group, 1), vb.repeat_interleave(group, 1)
        live = keep_mask.any(dim=1)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            o = torch.nn.functional.scaled_dot_product_attention(
                qb, kb, vb, attn_mask=mask, scale=req.scale
            )
        lse = torch.logsumexp(qb @ kb.transpose(-2, -1) * req.scale + mask, dim=-1)
        np.testing.assert_allclose(
            got["o"][b, :, :sq][:, live], o[0][:, live], atol=2e-5
        )
        np.testing.assert_allclose(got["stats"][b, :, :sq, 0], lse[0], atol=2e-5)
