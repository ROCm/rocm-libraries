# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shim mapping: AOTriton inputs -> the hipdnn_sdpa contract values (recording fake)."""

import subprocess
import sys

import pytest
import torch

import aotriton_flash as flash
import aotriton_layout
from aotriton_flash import CausalType


def dense(B=2, H=3, Sq=5, Sk=7, D=8, dtype=torch.float16):
    q, k, v = (torch.randn(B, H, S, D, dtype=dtype) for S in (Sq, Sk, Sk))
    o = torch.zeros(B, H, Sq, D, dtype=dtype)
    M = torch.zeros(B * H, Sq, dtype=torch.float32)
    return q, k, v, o, M


def fwd(
    q,
    k,
    v,
    M,
    o,
    *,
    b=None,
    causal=False,
    dropout_p=0.0,
    seed=None,
    off1=None,
    off2=0,
    seed_out=None,
    off_out=None,
    esm=None,
    scale=0.5
):
    empty = torch.empty(0, dtype=torch.uint64)
    return flash.attn_fwd(
        q,
        k,
        v,
        torch.empty(0, 0, 0, 0) if b is None else b,
        scale,
        M,
        o,
        dropout_p,
        empty if seed is None else seed,
        empty if off1 is None else off1,
        off2,
        empty if seed_out is None else seed_out,
        empty if off_out is None else off_out,
        esm,
        causal,
        torch.empty(0, dtype=torch.int32),
    )


def test_exports_cover_every_vendored_import():
    for name in (
        "attn_fwd",
        "attn_bwd",
        "attn_fwd_varlen",
        "attn_bwd_varlen",
        "debug_simulate_encoded_softmax",
        "lazy_dq_acc",
        "lazy_delta",
        "attn_options",
        "CausalType",
        "WindowValue",
        "IGNORE_BACKWARD_IMPORT",
        "AOTRITON_TORCH_ONLY_USE_CPU",
        "HipMemory",
        "hipError_t",
        "hipGetLastError",
    ):
        assert hasattr(flash, name), name
    assert (
        flash.IGNORE_BACKWARD_IMPORT is False
        and flash.AOTRITON_TORCH_ONLY_USE_CPU is False
    )
    assert flash.hipGetLastError() == flash.hipError_t.hipSuccess
    for name in (
        "hipSuccess",
        "hipErrorIllegalAddress",
        "hipErrorPeerAccessUnsupported",
    ):
        assert hasattr(flash.hipError_t, name)


def test_shim_is_what_the_vendored_suite_imports():
    """aotriton_flash resolves under shim/ even when the vendored tests directory leads the path."""
    code = "import aotriton_flash, pyaotriton; print(aotriton_flash.__file__); print(pyaotriton.get_name_suffix())"
    env = aotriton_layout.clean_env()
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=aotriton_layout.FLASH_TESTS,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[0].startswith(str(aotriton_layout.SHIM)) and out[1] == "hipdnn"
    assert not (
        aotriton_layout.VENDOR / "modules/flash/tests/aotriton_flash.py"
    ).exists()


@pytest.mark.parametrize(
    "causal, expected",
    [
        (False, None),
        (CausalType.NONE, None),
        (True, ("tl", None, 0)),
        (CausalType.TOP_LEFT, ("tl", None, 0)),
        (CausalType.BOTTOM_RIGHT, ("br", None, 0)),
        ((16, 3), ("tl", 17, 3)),
        ((0, 0), ("tl", 1, 0)),
    ],
)
def test_causal_forms_map_to_cudnn_masks(fake_sdpa, mode_record, causal, expected):
    q, k, v, o, M = dense()
    fwd(q, k, v, M, o, causal=causal)
    mask = fake_sdpa.calls[-1][1]["mask"]
    assert (
        (mask is None)
        if expected is None
        else (mask.alignment, mask.left, mask.right) == expected
    )


def test_window_is_inverse_of_aotriton_window_for(fake_sdpa, mode_record):
    """Query i sees keys i-l..i+r; cuDNN's left bound counts the diagonal, so left = l + 1."""
    q, k, v, o, M = dense()
    fwd(q, k, v, M, o, causal=(4, 2))
    mask = fake_sdpa.calls[-1][1]["mask"]
    i, j = 6, torch.arange(20)
    visible = (j - i <= mask.right) & (j - i > -mask.left)
    assert j[visible].tolist() == list(range(i - 4, i + 2 + 1))


def test_dense_stats_view_writes_through(fake_sdpa, mode_record):
    q, k, v, o, M = dense(B=2, H=3, Sq=5)
    assert fwd(q, k, v, M, o) == flash.hipError_t.hipSuccess
    call = fake_sdpa.calls[-1][1]
    stats = call["stats"]
    assert tuple(stats.shape) == (2, 3, 5, 1) and stats.stride() == (15, 5, 1, 1)
    assert stats.data_ptr() == M.data_ptr() and (M == 2.0).all()
    assert call["engine"] is None and call["varlen"] is None and call["dropout"] is None
    assert mode_record == []  # hipDNN calls are recorded by the plugin's wrappers


def test_engine_env_vars_pin_engines(fake_sdpa, mode_record, monkeypatch):
    monkeypatch.setenv("HIPDNN_FWD_ENGINE", "eng_f")
    monkeypatch.setenv("HIPDNN_BWD_ENGINE", "eng_b")
    q, k, v, o, M = dense()
    fwd(q, k, v, M, o)
    bwd(q, k, v, o, M)
    assert [c[1]["engine"] for c in fake_sdpa.calls] == ["eng_f", "eng_b"]


def test_fwd_no_stats_env_var_drops_stats_from_the_forward_only(
    fake_sdpa, mode_record, monkeypatch
):
    q, k, v, o, M = dense()
    fwd(q, k, v, M, o)
    assert fake_sdpa.calls[-1][1]["stats"] is not None
    monkeypatch.setenv("HIPDNN_SDPA_CONFORMANCE_FWD_NO_STATS", "1")
    fwd(q, k, v, M, o)
    assert fake_sdpa.calls[-1][1]["stats"] is None
    bwd(q, k, v, o, M)
    assert fake_sdpa.calls[-1][1]["stats"] is not None


def test_bias_forms(fake_sdpa, mode_record):
    q, k, v, o, M = dense()
    real = torch.randn(2, 3, 5, 7)
    fwd(q, k, v, M, o, b=real)
    assert fake_sdpa.calls[-1][1]["bias"] is real
    fwd(q, k, v, M, o, b=torch.empty(0, 0, 0, 0))
    assert fake_sdpa.calls[-1][1]["bias"] is None
    fwd(q, k, v, M, o)
    assert fake_sdpa.calls[-1][1]["bias"] is None


def bwd(
    q, k, v, o, L, *, b=None, db=None, causal=False, dropout_p=0.0, seed=None, off1=None
):
    empty = torch.empty(0, dtype=torch.uint64)
    return flash.attn_bwd(
        q,
        k,
        v,
        torch.empty(0, 0, 0, 0) if b is None else b,
        0.5,
        o,
        torch.zeros_like(o),
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        db,
        None,
        L,
        None,
        dropout_p,
        empty if seed is None else seed,
        empty if off1 is None else off1,
        0,
        causal,
    )


def test_backward_bias_gradient_rules(fake_sdpa, mode_record):
    q, k, v, o, M = dense()
    needs_grad = torch.randn(2, 3, 5, 7, requires_grad=True)
    no_grad = torch.randn(2, 3, 5, 7)
    db = torch.empty(2, 3, 5, 7)
    bwd(q, k, v, o, M, b=needs_grad, db=db)
    assert (
        fake_sdpa.calls[-1][1]["dbias"] is db
        and fake_sdpa.calls[-1][1]["bias"] is needs_grad
    )
    bwd(q, k, v, o, M, b=no_grad, db=None)
    assert fake_sdpa.calls[-1][1]["dbias"] is None
    bwd(q, k, v, o, M, b=no_grad, db=db)
    assert fake_sdpa.calls[-1][1]["dbias"] is None
    bwd(q, k, v, o, M, b=torch.empty(0, 0, 0, 0), db=None)
    assert (
        fake_sdpa.calls[-1][1]["bias"] is None
        and fake_sdpa.calls[-1][1]["dbias"] is None
    )


def test_backward_decline_is_recorded_and_reraised(fake_sdpa, mode_record):
    fake_sdpa.decline["bwd"] = "no engine"
    q, k, v, o, M = dense()
    with pytest.raises(fake_sdpa.EngineDeclined):
        bwd(q, k, v, o, M)
    assert mode_record == []  # the plugin's wrapper records the decline


def test_dropout_mapping_outputs_and_encoded_softmax(fake_sdpa, mode_record):
    q, k, v, o, M = dense()
    seed = torch.tensor([0x1BF52], dtype=torch.uint64)
    off1 = torch.tensor([0x1D4000], dtype=torch.uint64)
    seed_out, off_out = torch.zeros(1, dtype=torch.uint64), torch.zeros(
        1, dtype=torch.uint64
    )
    esm = torch.zeros(2, 3, 5, 7, dtype=torch.float16)
    fwd(
        q,
        k,
        v,
        M,
        o,
        dropout_p=0.25,
        seed=seed,
        off1=off1,
        off2=0xB42,
        seed_out=seed_out,
        off_out=off_out,
        esm=esm,
    )
    drop = fake_sdpa.calls[-1][1]["dropout"]
    assert (
        drop.p == 0.25
        and drop.seed.dtype == torch.int64
        and drop.offset.dtype == torch.int64
    )
    assert drop.seed.tolist() == [0x1BF52] and drop.offset.tolist() == [
        0x1D4000 + 0xB42
    ]
    assert seed_out.tolist() == [0x1BF52] and off_out.tolist() == [0x1D4000 + 0xB42]
    keep = fake_sdpa.philox_mask(2, 3, 5, 7, 0.25, 0x1BF52, 0x1D4000 + 0xB42, "cpu")
    assert torch.equal(esm >= 0, keep) and set(esm.unique().tolist()) <= {0.5, -0.5}


def test_preflight_forward_leaves_null_philox_outputs_alone(fake_sdpa, mode_record):
    q, k, v, o, M = dense()
    seed = torch.tensor([3], dtype=torch.uint64)
    off1 = torch.tensor([4], dtype=torch.uint64)
    assert (
        fwd(q, k, v, M, o, dropout_p=0.5, seed=seed, off1=off1)
        == flash.hipError_t.hipSuccess
    )


def test_backward_dropout_takes_forward_outputs(fake_sdpa, mode_record):
    q, k, v, o, M = dense()
    seed = torch.tensor([11], dtype=torch.uint64)
    offset = torch.tensor([0x1D4B42], dtype=torch.uint64)
    bwd(q, k, v, o, M, dropout_p=0.5, seed=seed, off1=offset)
    drop = fake_sdpa.calls[-1][1]["dropout"]
    assert drop.seed.tolist() == [11] and drop.offset.tolist() == [0x1D4B42]
    bwd(q, k, v, o, M)
    assert fake_sdpa.calls[-1][1]["dropout"] is None


# ------------------------------------------------------------------ varlen


def i32(*values):
    return torch.tensor(values, dtype=torch.int32)


def varlen_fwd(
    varlen_type,
    q,
    k,
    v,
    M,
    o,
    seq0q,
    seq0k,
    max_q,
    max_k,
    seq1q=None,
    seq1k=None,
    lse_layout="HT",
):
    empty = torch.empty(0, dtype=torch.uint64)
    return flash.attn_fwd_varlen(
        q,
        k,
        v,
        seq0q,
        seq0k,
        max_q,
        max_k,
        seq1q,
        seq1k,
        torch.empty(0, 0, 0, 0),
        0.5,
        M,
        o,
        0.0,
        empty,
        empty,
        0,
        empty,
        empty,
        None,
        False,
        torch.empty(0, dtype=torch.int32),
        varlen_type,
        lse_layout,
    )


def packed(H=2, T=12, D=8):
    q = torch.randn(1, H, T, D, dtype=torch.float16)
    return q, torch.randn_like(q), torch.randn_like(q), torch.zeros_like(q)


def test_varlen_compact(fake_sdpa, mode_record):
    q, k, v, o = packed()
    M = torch.zeros(2, 12)
    varlen_fwd("compact", q, k, v, M, o, i32(0, 5, 12), i32(0, 5, 12), 7, 7)
    vl = fake_sdpa.calls[-1][1]["varlen"]
    assert vl.seq_len_q.tolist() == [5, 7] and vl.seq_len_kv.tolist() == [5, 7]
    assert vl.pos_q.tolist() == [0, 5, 12] and vl.pos_kv.tolist() == [0, 5, 12]
    assert (vl.max_sq, vl.max_sk) == (7, 7) and vl.seq_len_q.dtype == torch.int32
    assert tuple(fake_sdpa.calls[-1][1]["stats"].shape) == (1, 2, 12, 1)


def test_varlen_strided_uses_slot_positions_and_slot_maxima(fake_sdpa, mode_record):
    q, k, v, o = packed()
    M = torch.zeros(2, 12)
    varlen_fwd(
        "strided",
        q,
        k,
        v,
        M,
        o,
        i32(0, 4, 9),
        i32(0, 3, 8),
        5,
        5,
        i32(0, 6, 12),
        i32(0, 5, 12),
    )
    vl = fake_sdpa.calls[-1][1]["varlen"]
    assert vl.seq_len_q.tolist() == [4, 5] and vl.seq_len_kv.tolist() == [3, 5]
    assert vl.pos_q.tolist() == [0, 6, 12] and vl.pos_kv.tolist() == [0, 5, 12]
    assert (vl.max_sq, vl.max_sk) == (6, 7)


def test_varlen_seqused_k_lengths_individual_positions_cumulative(
    fake_sdpa, mode_record
):
    q, k, v, o = packed()
    M = torch.zeros(2, 12)
    varlen_fwd(
        "seqused", q, k, v, M, o, i32(0, 5, 12), i32(3, 4), 7, 4, None, i32(0, 6, 12)
    )
    vl = fake_sdpa.calls[-1][1]["varlen"]
    assert vl.seq_len_q.tolist() == [5, 7] and vl.pos_q.tolist() == [0, 5, 12]
    assert vl.seq_len_kv.tolist() == [3, 4] and vl.pos_kv.tolist() == [0, 6, 12]
    assert (vl.max_sq, vl.max_sk) == (7, 6)


def test_varlen_padded_has_no_positions(fake_sdpa, mode_record):
    q = torch.randn(3, 2, 7, 8, dtype=torch.float16)
    k = torch.randn(3, 2, 6, 8, dtype=torch.float16)
    o = torch.zeros_like(q)
    M = torch.zeros(3 * 2, 7)
    varlen_fwd("padded", q, k, k, M, o, i32(0, 7, 12, 14), i32(0, 6, 9, 10), 7, 6)
    vl = fake_sdpa.calls[-1][1]["varlen"]
    assert vl.pos_q is None and vl.pos_kv is None
    assert vl.seq_len_q.tolist() == [7, 5, 2] and vl.seq_len_kv.tolist() == [6, 3, 1]
    assert (vl.max_sq, vl.max_sk) == (7, 6)


@pytest.mark.parametrize(
    "layout, padded", [("HT", True), ("TH", True), ("HT", False), ("TH", False)]
)
def test_lse_layouts_view_the_compact_buffer(fake_sdpa, mode_record, layout, padded):
    H, B, S = 2, 3, 6
    if padded:
        q = torch.randn(B, H, S, 8, dtype=torch.float16)
        shape = (B * H, S) if layout == "HT" else (B * S, H)
        args = (i32(0, 6, 12, 18), i32(0, 6, 12, 18), S, S, None, None)
    else:
        q = torch.randn(1, H, B * S, 8, dtype=torch.float16)
        shape = (H, B * S) if layout == "HT" else (B * S, H)
        args = (i32(0, 6, 12, 18), i32(0, 6, 12, 18), S, S, None, None)
    M = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    varlen_fwd(
        "padded" if padded else "compact",
        q,
        q,
        q,
        M,
        torch.zeros_like(q),
        *args,
        lse_layout=layout
    )
    stats = fake_sdpa.calls[-1][1]["stats"]
    assert stats.shape[0] == (B if padded else 1) and stats.shape[3] == 1
    for b in range(stats.shape[0]):
        for h in range(H):
            for s in range(stats.shape[2]):
                if padded:
                    want = M[b * H + h, s] if layout == "HT" else M[b * S + s, h]
                else:
                    want = M[h, s] if layout == "HT" else M[s, h]
                assert stats[b, h, s, 0] == want


def test_varlen_backward_passes_same_varlen_and_stats(fake_sdpa, mode_record):
    q, k, v, o = packed()
    L = torch.zeros(2, 12)
    empty = torch.empty(0, dtype=torch.uint64)
    flash.attn_bwd_varlen(
        q,
        k,
        v,
        i32(0, 5, 12),
        i32(0, 5, 12),
        7,
        7,
        None,
        None,
        torch.empty(0, 0, 0, 0),
        0.5,
        o,
        torch.zeros_like(o),
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty(0, 0, 0, 0),
        None,
        L,
        None,
        0.0,
        empty,
        empty,
        0,
        True,
        "compact",
        "HT",
    )
    call = fake_sdpa.calls[-1]
    assert call[0] == "bwd" and call[1]["bias"] is None and call[1]["dbias"] is None
    assert call[1]["mask"].alignment == "tl" and call[1]["varlen"].pos_q.tolist() == [
        0,
        5,
        12,
    ]
    assert tuple(call[1]["stats"].shape) == (1, 2, 12, 1)
    assert mode_record == []


# ------------------------------------------------------------------ reference fallback


def test_forward_decline_runs_reference_and_records_it(fake_sdpa, mode_record):
    fake_sdpa.decline["fwd"] = "no fwd engine"
    q, k, v, o, M = dense(dtype=torch.float32)
    fwd(q, k, v, M, o, causal=CausalType.TOP_LEFT)
    out, lse = flash.reference_attention(q, k, v, 0.5, fake_sdpa.Mask("tl", None, 0))
    assert torch.equal(o, out) and torch.equal(M.view(2, 3, 5), lse)
    assert mode_record == [("fwd", "ref", "no fwd engine")]


def test_reference_dropout_uses_the_philox_mask_and_fills_encoded_softmax(
    fake_sdpa, mode_record
):
    fake_sdpa.decline["fwd"] = "no dropout"
    q, k, v, o, M = dense(dtype=torch.float32)
    seed = torch.tensor([5], dtype=torch.uint64)
    off1 = torch.tensor([9], dtype=torch.uint64)
    esm = torch.zeros(2, 3, 5, 7)
    fwd(q, k, v, M, o, dropout_p=0.25, seed=seed, off1=off1, off2=1, esm=esm)
    keep = fake_sdpa.philox_mask(2, 3, 5, 7, 0.25, 5, 10, "cpu")
    out, _ = flash.reference_attention(q, k, v, 0.5, None, keep=keep, p=0.25)
    assert torch.equal(o, out) and torch.equal(esm >= 0, keep)


def test_varlen_reference_forward_matches_per_sequence_dense_reference(
    fake_sdpa, mode_record
):
    """Padded and slotted varlen under a declined forward: each sequence equals a dense run."""
    fake_sdpa.decline["fwd"] = "varlen unsupported"
    torch.manual_seed(1)
    H, D = 2, 8
    seq_q, seq_k, slot_q, slot_k = [3, 5], [4, 2], [4, 6], [5, 3]
    # strided: sequences sit in slots with unused tail rows
    q = torch.randn(1, H, sum(slot_q), D)
    k = torch.randn(1, H, sum(slot_k), D)
    o = torch.zeros_like(q)
    M = torch.zeros(H, sum(slot_q))
    varlen_fwd(
        "strided",
        q,
        k,
        k,
        M,
        o,
        i32(0, 3, 8),
        i32(0, 4, 6),
        5,
        4,
        i32(0, 4, 10),
        i32(0, 5, 8),
    )
    mask = fake_sdpa.Mask("tl", None, 0)
    for z, (sq, sk) in enumerate(zip(seq_q, seq_k)):
        qs, ks = sum(slot_q[:z]), sum(slot_k[:z])
        want, lse = flash.reference_attention(
            q[:, :, qs : qs + sq],
            k[:, :, ks : ks + sk],
            k[:, :, ks : ks + sk],
            0.5,
            None,
        )
        assert torch.allclose(o[:, :, qs : qs + sq], want)
        assert torch.allclose(M[:, qs : qs + sq], lse[0])
    assert (o[:, :, 3:4] == 0).all()  # a slot's unused tail is never written
    assert mode_record == [("fwd", "ref", "varlen unsupported")]

    # padded BHSD: rows past each sequence's length stay untouched
    qp = torch.randn(2, H, 5, D)
    kp = torch.randn(2, H, 4, D)
    op, Mp = torch.zeros_like(qp), torch.zeros(2 * H, 5)
    varlen_fwd(
        "padded", qp, kp, kp, Mp, op, i32(0, 3, 8), i32(0, 4, 6), 5, 4, lse_layout="HT"
    )
    for z, (sq, sk) in enumerate(zip(seq_q, seq_k)):
        want, _ = flash.reference_attention(
            qp[z : z + 1, :, :sq],
            kp[z : z + 1, :, :sk],
            kp[z : z + 1, :, :sk],
            0.5,
            None,
        )
        assert (
            torch.allclose(op[z : z + 1, :, :sq], want) and (op[z, :, sq:] == 0).all()
        )
