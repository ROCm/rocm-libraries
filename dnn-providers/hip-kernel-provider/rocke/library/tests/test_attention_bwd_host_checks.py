# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host validation of the backward run APIs (CPU only).

The run functions take device pointers; these checks validate what a caller
holding the tensors knows (shapes, strides, dtypes, sizes) and the
host-visible device data (sequence lengths, offsets, block tables) against
the planned request, and reject a violation with a code of
``HOST_CHECK_CODES`` (never a clamp):

* general family: every tensor role of dense, padded and THD requests,
  ``SEQ_LEN`` values past ``S_max``, offsets that decrease, overrun the token
  bound or the tensor, or are not whole tokens;
* dense family: the compile-time shape against its tensors and batch;
* unified family: ``cu_seqlens_q`` (monotone, inside ``total_q``, spans),
  ``seq_lens_kv > max_seqlen_kv``, block-table ids outside ``[0,
  num_blocks)``, a non-injective table under the injectivity assertion, and
  every tensor (a cache smaller than ``num_blocks`` pages);
* every smoke-tier unified test case's real inputs pass.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kernels.common.attention_bwd_plan import AttnBwdRequest
from kernels.common.attention_bwd_run import (
    ELEM_BYTES,
    HOST_CHECK_CODES,
    TensorMeta,
    check_attn_bwd_lengths,
    check_attn_bwd_tensors,
    check_attn_dense_bwd_tensors,
    extent_elems,
)
from kernels.common.attention_bwd_unified_plan import AttnBwdUnifiedRequest
from kernels.common.attention_bwd_unified_run import check_attn_bwd_unified_inputs


def _code(code):
    assert code in HOST_CHECK_CODES
    return pytest.raises(ValueError, match=f"^{code}: ")


def _meta(shape, strides, dtype, slack=0):
    nbytes = extent_elems(shape, strides) * ELEM_BYTES[dtype] + slack
    return TensorMeta(tuple(shape), tuple(strides), dtype, nbytes)


# ---------------------------------------------------------------------------
# general family
# ---------------------------------------------------------------------------


def _req(**kw):
    base = dict(
        b=2, h_q=8, h_k=2, h_v=2, s_q=128, s_kv=96, d_qk=64, d_v=64, dtype="bf16"
    )
    base.update(kw)
    return AttnBwdRequest(**base)


def _general_tensors(req, tokens=(256, 192)):
    out = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        sb, sh, ss, sd = req.tensor_strides(n)
        h = req.tensor_heads(n)
        if req.is_thd:
            t = tokens[0] if n in ("q", "o", "do", "dq") else tokens[1]
            out[n] = _meta((t, h, req.d_qk), (ss, sh, sd), req.dtype)
        else:
            out[n] = _meta(
                (req.b, h, req.tensor_seq(n), req.d_qk), (sb, sh, ss, sd), req.dtype
            )
    if req.is_thd:
        out["lse"] = _meta((tokens[0], req.h_q), (req.h_q, 1), "fp32")
        for n in ("offsets_q", "offsets_kv"):
            out[n] = _meta((req.b + 1,), (1,), req.ragged_offset_dtype)
    else:
        out["lse"] = _meta(
            (req.b, req.h_q, req.s_q), (req.h_q * req.s_q, req.s_q, 1), "fp32"
        )
    if req.padding:
        for n in ("seq_len_q", "seq_len_kv"):
            out[n] = _meta((req.b,), (1,), "int32")
    return out


@pytest.mark.parametrize(
    "kw",
    [
        {},
        dict(padding=True, has_seq_len_q=True, has_seq_len_kv=True),
        dict(layout="thd", ragged_offset_dtype="int64"),
        dict(
            strides={n: (96 * 2 * 64 * 2, 64 * 2, 2 * 64 * 2, 1) for n in ("k", "dk")}
        ),
    ],
    ids=["dense", "padded", "thd", "strided_k"],
)
def test_general_valid_tensors_pass(kw):
    req = _req(**kw)
    check_attn_bwd_tensors(req, _general_tensors(req))


def test_general_tensor_violations():
    req = _req()
    good = _general_tensors(req)

    def bad(role, **change):
        t = dict(good)
        t[role] = dataclasses.replace(good[role], **change)
        return t

    with _code("TENSOR_EXTENT"):  # one byte short of the largest offset
        check_attn_bwd_tensors(req, bad("dk", nbytes=good["dk"].nbytes - 1))
    with _code("TENSOR_SHAPE"):
        check_attn_bwd_tensors(req, bad("q", shape=(2, 8, 64, 64)))
    with _code("TENSOR_STRIDES"):
        st = good["v"].strides
        check_attn_bwd_tensors(req, bad("v", strides=(st[0], st[1], st[2] * 2, 1)))
    with _code("TENSOR_DTYPE"):
        check_attn_bwd_tensors(req, bad("do", dtype="fp16"))
    with _code("TENSOR_DTYPE"):
        check_attn_bwd_tensors(req, bad("lse", dtype="bf16"))
    with _code("TENSOR_EXTENT"):
        check_attn_bwd_tensors(req, bad("lse", nbytes=good["lse"].nbytes - 4))
    with _code("MISSING_TENSOR"):
        check_attn_bwd_tensors(req, {k: v for k, v in good.items() if k != "dv"})
    padded = _req(padding=True, has_seq_len_q=True, has_seq_len_kv=True)
    t = _general_tensors(padded)
    t["seq_len_kv"] = dataclasses.replace(t["seq_len_kv"], nbytes=4)
    with _code("TENSOR_EXTENT"):
        check_attn_bwd_tensors(padded, t)
    thd = _req(layout="thd")
    t = _general_tensors(thd)
    t["offsets_q"] = dataclasses.replace(t["offsets_q"], dtype="int64")
    with _code("TENSOR_DTYPE"):
        check_attn_bwd_tensors(thd, t)


def test_general_lengths():
    req = _req(padding=True, has_seq_len_q=True, has_seq_len_kv=True)
    check_attn_bwd_lengths(req, seq_len_q=[128, 0], seq_len_kv=np.array([96, 5]))
    with _code("SEQ_LEN_RANGE"):  # longer than S_kv: rejected, never clamped
        check_attn_bwd_lengths(req, seq_len_kv=[97, 5])
    with _code("SEQ_LEN_RANGE"):
        check_attn_bwd_lengths(req, seq_len_q=[-1, 5])
    with _code("SEQ_LEN_RANGE"):
        check_attn_bwd_lengths(req, seq_len_q=[1, 2, 3])
    with _code("SEQ_LEN_RANGE"):
        check_attn_bwd_lengths(req, seq_len_q=[1.5, 2])


def test_thd_offsets():
    """THD offsets are element offsets: token = offset * multiplier / token
    stride (q: 8 heads * 64 = 512 elements per token; kv: 2 * 64 = 128)."""
    req = _req(layout="thd")
    q, kv = 512, 128
    check_attn_bwd_lengths(req, offsets_q=[0, 100 * q, 228 * q],
                           offsets_kv=[0, 96 * kv, 96 * kv], tokens_q=228, tokens_kv=96)  # fmt: skip
    for offs, phrase in (
        ([0, 100 * q, 90 * q], "decreases"),
        ([-q, 10 * q, 20 * q], "negative"),
        ([0, 129 * q, 130 * q], "spans 129"),
        ([0, 10 * q], "offsets for B"),
        ([0, 513, 1024], "whole token"),
    ):
        with pytest.raises(ValueError, match=f"^RAGGED_RANGE: .*{phrase}"):
            check_attn_bwd_lengths(req, offsets_q=offs)
    with pytest.raises(ValueError, match="past the tensor"):
        check_attn_bwd_lengths(req, offsets_q=[0, 100 * q, 228 * q], tokens_q=200)
    bounded = _req(layout="thd", max_total_q=150, max_total_kv=200)
    with pytest.raises(ValueError, match="bound 150"):
        check_attn_bwd_lengths(bounded, offsets_q=[0, 100 * q, 228 * q])
    # the multiplier scales element offsets before the division
    mult = _req(layout="thd", ragged_offset_multiplier=2)
    check_attn_bwd_lengths(mult, offsets_q=[0, 50 * q, 114 * q])
    with pytest.raises(ValueError, match="sequence 0 spans 130 > 128"):
        check_attn_bwd_lengths(mult, offsets_q=[0, 65 * q, 66 * q])


def test_tensor_meta_of_a_torch_like_tensor():
    class Storage:
        def nbytes(self):
            return 4096

    class T:
        shape = (2, 4, 8)
        dtype = "torch.bfloat16"

        def stride(self):
            return (32, 8, 1)

        def element_size(self):
            return 2

        def storage_offset(self):
            return 16

        def untyped_storage(self):
            return Storage()

    m = TensorMeta.of(T())
    assert m == TensorMeta((2, 4, 8), (32, 8, 1), "bf16", 4096 - 32)
    with pytest.raises(ValueError, match="TENSOR_DTYPE"):
        TensorMeta.of(type("U", (T,), {"dtype": "torch.float64"})())


def test_dense_spec_tensors():
    from kernels.common.attention_dense_bwd_spec import AttentionDenseBwdSpec

    spec = AttentionDenseBwdSpec(
        seqlen_q=128, seqlen_kv=256, num_query_heads=8, num_kv_heads=2, head_size=64
    )
    t = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        q_side = n in ("q", "o", "do", "dq")
        h, s = (8, 128) if q_side else (2, 256)
        sb, sh, st = spec.strides(n)
        t[n] = _meta((3, h, s, 64), (sb, sh, st, 1), "bf16")
    t["lse"] = _meta((3, 8, 128), (8 * 128, 128, 1), "fp32")
    check_attn_dense_bwd_tensors(spec, t, batch=3)
    with _code("TENSOR_SHAPE"):
        check_attn_dense_bwd_tensors(spec, t, batch=4)
    with _code("TENSOR_STRIDES"):  # a BHSD tensor where the spec bakes BSHD
        bad = dict(t, k=_meta((3, 2, 256, 64), (2 * 256 * 64, 256 * 64, 64, 1), "bf16"))
        check_attn_dense_bwd_tensors(spec, bad, batch=3)
    pinned = dataclasses.replace(spec, batch=2)
    with _code("TENSOR_SHAPE"):
        check_attn_dense_bwd_tensors(pinned, t, batch=3)


# ---------------------------------------------------------------------------
# unified family
# ---------------------------------------------------------------------------


def _ureq(**kw):
    base = dict(num_seqs=3, h_q=8, h_kv=2, d_qk=64, d_v=64, total_q=100,
                max_seqlen_q=60, max_seqlen_kv=300, page=16, num_blocks=50,
                max_blocks_per_seq=20)  # fmt: skip
    base.update(kw)
    return AttnBwdUnifiedRequest(**base)


def _unified_data(req):
    cu = [0, 30, 60, 100]
    lens = [300, 17, 0]
    bt = np.full((req.num_seqs, req.table_stride), -1, np.int32)  # unread padding
    ids = iter(range(req.num_blocks))
    for s, n in enumerate(lens):
        for j in range(-(-n // req.page)):
            bt[s, j] = next(ids)
    return cu, lens, bt


def _unified_tensors(req):
    t = {}
    for n in ("q", "o", "do", "dq"):
        t[n] = _meta((req.total_q, req.h_q, req.d_qk), req.tensor_strides(n), req.dtype)
    for role, n in (
        ("k_cache", "k"),
        ("v_cache", "v"),
        ("dk_cache", "dk"),
        ("dv_cache", "dv"),
    ):
        t[role] = _meta(req.tensor_dims(n), req.tensor_strides(n), req.dtype)
    t["lse"] = _meta((req.total_q, req.h_q), (req.h_q, 1), "fp32")
    t["cu_seqlens_q"] = _meta((req.num_seqs + 1,), (1,), "int32")
    t["seq_lens_kv"] = _meta((req.num_seqs,), (1,), "int32")
    t["block_table"] = _meta(
        (req.num_seqs, req.table_stride), (req.table_stride, 1), "int32"
    )
    return t


def test_unified_valid_inputs_pass():
    req = _ureq()
    cu, lens, bt = _unified_data(req)
    check_attn_bwd_unified_inputs(req, cu_seqlens_q=cu, seq_lens_kv=lens, block_table=bt,
                                  tensors=_unified_tensors(req))  # fmt: skip
    # the flat device layout of the same table
    check_attn_bwd_unified_inputs(req, seq_lens_kv=lens, block_table=bt.reshape(-1))


@pytest.mark.parametrize(
    "cu,phrase",
    [
        ([0, 30, 20, 100], "decreases"),
        ([0, 30, 60, 101], "total_q"),
        ([0, 61, 62, 100], "spans 61"),
        ([-1, 30, 60, 100], "negative"),
        ([0, 30, 60], "entries"),
    ],
)
def test_unified_cu_seqlens(cu, phrase):
    with pytest.raises(ValueError, match=f"^RAGGED_RANGE: .*{phrase}"):
        check_attn_bwd_unified_inputs(_ureq(), cu_seqlens_q=cu)


def test_unified_seq_lens_past_the_bound_are_rejected():
    with _code("SEQ_LEN_RANGE"):
        check_attn_bwd_unified_inputs(_ureq(), seq_lens_kv=[301, 0, 0])
    with _code("SEQ_LEN_RANGE"):
        check_attn_bwd_unified_inputs(_ureq(), seq_lens_kv=[-3, 0, 0])


@pytest.mark.parametrize("bad", [50, 51, -1, 2**31 - 1])
def test_unified_block_ids_out_of_range(bad):
    req = _ureq()
    cu, lens, bt = _unified_data(req)
    bt[1, 1] = bad  # a live block of sequence 1 (len 17: blocks 0 and 1)
    with pytest.raises(ValueError, match=r"^BLOCK_TABLE_RANGE: block_table\[1\]\[1\]"):
        check_attn_bwd_unified_inputs(req, seq_lens_kv=lens, block_table=bt)
    # an entry past the live blocks is never read: anything is accepted there
    cu, lens, bt = _unified_data(req)
    bt[1, 2] = bad
    check_attn_bwd_unified_inputs(req, seq_lens_kv=lens, block_table=bt)


def test_unified_injectivity_and_table_shape():
    req = _ureq(block_table_injective=True)
    cu, lens, bt = _unified_data(req)
    check_attn_bwd_unified_inputs(req, seq_lens_kv=lens, block_table=bt)
    bt[1, 0] = bt[0, 3]
    with _code("BLOCK_TABLE_NOT_INJECTIVE"):
        check_attn_bwd_unified_inputs(req, seq_lens_kv=lens, block_table=bt)
    # shared pages are legal without the assertion (atomic page mode)
    check_attn_bwd_unified_inputs(_ureq(), seq_lens_kv=lens, block_table=bt)
    with _code("BLOCK_TABLE_RANGE"):
        check_attn_bwd_unified_inputs(req, seq_lens_kv=lens, block_table=bt[:2])
    with pytest.raises(TypeError, match="needs seq_lens_kv"):
        check_attn_bwd_unified_inputs(req, block_table=bt)


def test_unified_tensor_violations():
    req = _ureq()
    good = _unified_tensors(req)
    small = dataclasses.replace(good["dk_cache"], nbytes=good["dk_cache"].nbytes - 2)
    with _code("TENSOR_EXTENT"):  # a gradient cache smaller than num_blocks pages
        check_attn_bwd_unified_inputs(req, tensors=dict(good, dk_cache=small))
    with _code("TENSOR_SHAPE"):
        bad = _meta((49, 16, 2, 64), req.tensor_strides("k"), "bf16")
        check_attn_bwd_unified_inputs(req, tensors=dict(good, k_cache=bad))
    with _code("TENSOR_EXTENT"):
        bt = dataclasses.replace(good["block_table"], nbytes=16)
        check_attn_bwd_unified_inputs(req, tensors=dict(good, block_table=bt))
    with _code("MISSING_TENSOR"):
        check_attn_bwd_unified_inputs(
            req, tensors={k: v for k, v in good.items() if k != "seq_lens_kv"}
        )


def test_unified_case_table_inputs_pass():
    """The real inputs of every smoke-tier unified device case (spare and
    trap pages, padded rows, shared prefixes) pass the checks."""
    from tests._attention_bwd_unified_harness import request_from_unified
    from tests.sdpa.bwd_unified_cases import materialize_unified, select_unified

    for case in select_unified("smoke"):
        inp = materialize_unified(case)
        req = request_from_unified(inp)
        check_attn_bwd_unified_inputs(
            req, cu_seqlens_q=inp.cu_seqlens_q, seq_lens_kv=inp.seq_lens_kv,
            block_table=inp.block_table,
        )  # fmt: skip
