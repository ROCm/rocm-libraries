# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipDNN-backed replacement for AOTriton's ``aotriton_flash.py``.

The vendored suite reaches the GPU only through this module's ``attn_fwd``, ``attn_bwd``,
``attn_fwd_varlen`` and ``attn_bwd_varlen``. Each translates AOTriton's arguments into the
``hipdnn_sdpa`` contract (``Mask``, ``Varlen``, ``Dropout``, bias, stats views) and calls
``hipdnn_sdpa.sdpa_fwd`` / ``sdpa_bwd`` as module attributes at call time.

* A forward that hipDNN declines runs a pure-torch reference forward instead, so the hipDNN
  backward still receives O and the logsumexp (mode ``ref-fwd``).
* A backward decline propagates as ``EngineDeclined``; the plugin turns it into a skip.
* The plugin wraps ``sdpa_fwd`` / ``sdpa_bwd`` to record hipDNN calls and declines; the shim records
  only ``ref:`` entries (``hipdnn_aotriton_plugin.record``) for the reference fallback.

Engine pinning comes from ``HIPDNN_FWD_ENGINE`` / ``HIPDNN_BWD_ENGINE``. ``HIPDNN_FWD_ENGINE=reference``
(``conformance_contract.REFERENCE_PIN``) skips hipDNN for the forward: the reference forward runs and
the call records ``ref`` with the reason ``pinned reference``, so a backward can be tested on a forward
no hipDNN engine produced. With
``HIPDNN_SDPA_CONFORMANCE_FWD_NO_STATS`` set, the forward asks hipDNN for no Stats and fills the M buffer
with zeros: AOTriton's wrapper rejects NaN in M under ``is_testing``, and ``test_forward.py`` compares
only O. The reference fallback still fills M with the real LSE.
"""

import os
import sys

import conformance_contract as contract
import torch

from pyaotriton import hipError_t, hipGetLastError  # noqa: F401  (re-exported)

IGNORE_BACKWARD_IMPORT = False
AOTRITON_TORCH_ONLY_USE_CPU = False
HipMemory = None


class CausalType:
    NONE = 0
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2
    WINDOWED = 3


class WindowValue:
    NONE = 0
    TOP_LEFT_ALIGNED = -2147483647
    BOTTOM_RIGHT_ALIGNED = -2147483646


class attn_options:
    """Carries ``force_backend_index`` for callers that set it; hipDNN pins by engine name."""

    def __init__(self):
        self.force_backend_index = None


def _sdpa():
    import hipdnn_sdpa

    return hipdnn_sdpa


def _record(direction, kind, detail):
    plugin = sys.modules.get("hipdnn_aotriton_plugin")
    if plugin is not None:
        plugin.record(direction, kind, detail)


def _fwd_no_stats():
    return bool(os.environ.get("HIPDNN_SDPA_CONFORMANCE_FWD_NO_STATS"))


def _engine(direction):
    return os.environ.get(f"HIPDNN_{direction.upper()}_ENGINE") or None


# --------------------------------------------------------------------------- mapping


def _mask(causal):
    sdpa = _sdpa()
    if isinstance(causal, tuple):
        left, right = causal
        return sdpa.Mask("tl", int(left) + 1, int(right))
    if isinstance(causal, bool):
        return sdpa.Mask("tl", None, 0) if causal else None
    if causal == CausalType.NONE:
        return None
    if causal == CausalType.TOP_LEFT:
        return sdpa.Mask("tl", None, 0)
    if causal == CausalType.BOTTOM_RIGHT:
        return sdpa.Mask("br", None, 0)
    raise AssertionError(f"unsupported causal value {causal!r}")


def _bias(b):
    return None if b is None or b.numel() == 0 else b


def _dropout(dropout_p, philox_seed, philox_offset1, philox_offset2):
    if not dropout_p > 0.0:
        return None
    seed = philox_seed.view(torch.int64)
    offset = philox_offset1.view(torch.int64) + int(philox_offset2)
    return _sdpa().Dropout(float(dropout_p), seed, offset)


def _dense_stats(M, q):
    B, H, Sq, _ = q.shape
    assert M.shape == (B * H, Sq), f"unexpected M shape {tuple(M.shape)}"
    return M.as_strided((B, H, Sq, 1), (H * Sq, Sq, 1, 1), M.storage_offset())


def _varlen_stats(M, q, padded, lse_layout):
    """View AOTriton's compact logsumexp buffer as a 4-D FP32 stats tensor."""
    B_or_1, H, S, _ = q.shape
    if lse_layout == "HT":
        shape, strides = (B_or_1, H, S, 1), (H * S, S, 1, 1)
        want = (B_or_1 * H, S) if padded else (H, S)
    else:
        shape, strides = (B_or_1, H, S, 1), (S * H, 1, H, 1)
        want = (B_or_1 * S, H) if padded else (S, H)
    assert tuple(M.shape) == want, f"unexpected M shape {tuple(M.shape)}, wanted {want}"
    return M.as_strided(shape, strides, M.storage_offset())


def _slot_max(pos):
    return int(torch.diff(pos).max())


def _varlen(
    varlen_type,
    seqinfo_q0,
    seqinfo_k0,
    max_seqlen_q,
    max_seqlen_k,
    seqinfo_q1,
    seqinfo_k1,
    q,
    k,
):
    Varlen = _sdpa().Varlen
    if varlen_type == "compact":
        return Varlen(
            torch.diff(seqinfo_q0),
            torch.diff(seqinfo_k0),
            seqinfo_q0,
            seqinfo_k0,
            int(max_seqlen_q),
            int(max_seqlen_k),
        )
    if varlen_type == "padded":
        return Varlen(
            torch.diff(seqinfo_q0),
            torch.diff(seqinfo_k0),
            None,
            None,
            q.shape[2],
            k.shape[2],
        )
    if varlen_type == "strided":
        return Varlen(
            torch.diff(seqinfo_q0),
            torch.diff(seqinfo_k0),
            seqinfo_q1,
            seqinfo_k1,
            _slot_max(seqinfo_q1),
            _slot_max(seqinfo_k1),
        )
    if varlen_type == "seqused":
        return Varlen(
            torch.diff(seqinfo_q0),
            seqinfo_k0,
            seqinfo_q0,
            seqinfo_k1,
            int(max_seqlen_q),
            _slot_max(seqinfo_k1),
        )
    raise AssertionError(f"unknown varlen_type {varlen_type!r}")


# --------------------------------------------------------------------------- reference


def _band_keep(mask, sq, sk, device):
    """Boolean [sq, sk] of positions a ``Mask`` leaves visible (cuDNN band convention)."""
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(sk, device=device)[None, :]
    diagonal = j - i - (sk - sq if mask.alignment == "br" else 0)
    keep = torch.ones(sq, sk, dtype=torch.bool, device=device)
    if mask.right is not None:
        keep &= diagonal <= mask.right
    if mask.left is not None:
        keep &= diagonal > -mask.left
    return keep


def reference_attention(q, k, v, scale, mask=None, bias=None, keep=None, p=0.0):
    """Forward attention in FP32 (FP64 for FP32 inputs) over 4-D ``[B, H, S, D]`` tensors.

    Returns ``(O, LSE)`` with O in ``q.dtype`` and LSE ``[B, Hq, Sq]`` natural-log FP32. A query
    row with no visible key gets O = 0 and LSE = +inf. ``keep`` (``[B, Hq, Sq, Sk]`` bool) is
    the dropout keep mask; kept probabilities are scaled by ``1 / (1 - p)``.
    """
    work = (
        torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else torch.float64
    )
    qf, kf, vf = q.to(work), k.to(work), v.to(work)
    hq, hk = qf.shape[1], kf.shape[1]
    if hq != hk:
        kf = kf.repeat_interleave(hq // hk, dim=1)
        vf = vf.repeat_interleave(hq // hk, dim=1)
    s = (qf @ kf.transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.to(work)
    if mask is not None:
        visible = _band_keep(mask, qf.shape[2], kf.shape[2], q.device)
        s = s.masked_fill(~visible, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    dead = torch.isneginf(lse)
    probs = torch.exp(s - torch.where(dead, torch.zeros_like(lse), lse)[..., None])
    if keep is not None:
        probs = probs * keep / (1.0 - p)
    out = (probs @ vf).to(q.dtype)
    lse = torch.where(dead, torch.full_like(lse, float("inf")), lse)
    return out, lse.to(torch.float32)


def _reference_forward(q, k, v, bias, scale, mask, keep, p, o, stats, varlen):
    if varlen is None:
        out, lse = reference_attention(q, k, v, scale, mask, bias, keep, p)
        o.copy_(out)
        stats.copy_(lse[..., None])
        return
    seq_q = varlen.seq_len_q.tolist()
    seq_k = varlen.seq_len_kv.tolist()
    pos_q = None if varlen.pos_q is None else varlen.pos_q.tolist()
    pos_k = None if varlen.pos_kv is None else varlen.pos_kv.tolist()
    for z, (sq, sk) in enumerate(zip(seq_q, seq_k)):
        if pos_q is None:
            qs, ks = (slice(z, z + 1), slice(None), slice(0, sq)), (
                slice(z, z + 1),
                slice(None),
                slice(0, sk),
            )
        else:
            qs = (slice(None), slice(None), slice(pos_q[z], pos_q[z] + sq))
            ks = (slice(None), slice(None), slice(pos_k[z], pos_k[z] + sk))
        kz = None if keep is None else keep[z : z + 1, :, :sq, :sk]
        out, lse = reference_attention(q[qs], k[ks], v[ks], scale, mask, None, kz, p)
        o[qs] = out
        stats[qs] = lse[..., None]


# --------------------------------------------------------------------------- forward


def _keep_mask(dropout, B, Hq, max_sq, max_sk, device):
    seed, offset = (int(t.item()) for t in (dropout.seed, dropout.offset))
    return _sdpa().philox_mask(B, Hq, max_sq, max_sk, dropout.p, seed, offset, device)


def _forward(
    q,
    k,
    v,
    *,
    bias,
    scale,
    o,
    stats,
    mask,
    dropout,
    varlen,
    seed_out,
    offset_out,
    encoded_softmax,
):
    sdpa = _sdpa()
    no_stats = _fwd_no_stats()
    pin = _engine("fwd")
    try:
        if pin == contract.REFERENCE_PIN:  # a decline that never reaches hipDNN
            raise sdpa.EngineDeclined("fwd", "pinned reference")
        sdpa.sdpa_fwd(
            q,
            k,
            v,
            o=o,
            stats=None if no_stats else stats,
            scale=scale,
            bias=bias,
            mask=mask,
            dropout=dropout,
            varlen=varlen,
            engine=pin,
        )
        if no_stats:
            stats.zero_()
        ran_reference = False
    except sdpa.EngineDeclined as declined:
        _record("fwd", "ref", declined.reason)
        ran_reference = True
    keep = None
    if dropout is not None and (ran_reference or encoded_softmax is not None):
        batch = q.shape[0] if varlen is None else varlen.seq_len_q.shape[0]
        max_sq = q.shape[2] if varlen is None else varlen.max_sq
        max_sk = k.shape[2] if varlen is None else varlen.max_sk
        keep = _keep_mask(dropout, batch, q.shape[1], max_sq, max_sk, q.device)
    if ran_reference:
        _reference_forward(
            q,
            k,
            v,
            bias,
            scale,
            mask,
            keep,
            None if dropout is None else dropout.p,
            o,
            stats,
            varlen,
        )
    if dropout is not None:
        if encoded_softmax is not None:
            rows = min(encoded_softmax.shape[2], keep.shape[2])
            cols = min(encoded_softmax.shape[3], keep.shape[3])
            encoded_softmax[:, :, :rows, :cols] = torch.where(
                keep[:, :, :rows, :cols], 0.5, -0.5
            ).to(encoded_softmax.dtype)
        if seed_out is not None and seed_out.numel() != 0:
            seed_out.view(torch.int64).copy_(dropout.seed)
        if offset_out is not None and offset_out.numel() != 0:
            offset_out.view(torch.int64).copy_(dropout.offset)
    return hipError_t.hipSuccess


def attn_fwd(
    q,
    k,
    v,
    b,
    sm_scale,
    M,
    o,
    dropout_p,
    philox_seed,
    philox_offset1,
    philox_offset2,
    philox_seed_output,
    philox_offset_output,
    encoded_softmax,
    causal,
    atomic,
    extargs=None,
):
    return _forward(
        q,
        k,
        v,
        bias=_bias(b),
        scale=float(sm_scale),
        o=o,
        stats=_dense_stats(M, q),
        mask=_mask(causal),
        dropout=_dropout(dropout_p, philox_seed, philox_offset1, philox_offset2),
        varlen=None,
        seed_out=philox_seed_output,
        offset_out=philox_offset_output,
        encoded_softmax=encoded_softmax,
    )


def attn_fwd_varlen(
    q,
    k,
    v,
    seqinfo_q0,
    seqinfo_k0,
    max_seqlen_q,
    max_seqlen_k,
    seqinfo_q1,
    seqinfo_k1,
    b,
    sm_scale,
    M,
    o,
    dropout_p,
    philox_seed,
    philox_offset1,
    philox_offset2,
    philox_seed_output,
    philox_offset_output,
    encoded_softmax,
    causal,
    atomic,
    varlen_type,
    lse_layout="HT",
    extargs=None,
):
    varlen = _varlen(
        varlen_type,
        seqinfo_q0,
        seqinfo_k0,
        max_seqlen_q,
        max_seqlen_k,
        seqinfo_q1,
        seqinfo_k1,
        q,
        k,
    )
    return _forward(
        q,
        k,
        v,
        bias=_bias(b),
        scale=float(sm_scale),
        o=o,
        stats=_varlen_stats(M, q, varlen_type == "padded", lse_layout),
        mask=_mask(causal),
        dropout=_dropout(dropout_p, philox_seed, philox_offset1, philox_offset2),
        varlen=varlen,
        seed_out=philox_seed_output,
        offset_out=philox_offset_output,
        encoded_softmax=encoded_softmax,
    )


# --------------------------------------------------------------------------- backward


def _backward(
    q, k, v, o, do, stats, *, dq, dk, dv, scale, b, db, mask, dropout, varlen
):
    sdpa = _sdpa()
    bias = _bias(b)
    dbias = db if bias is not None and db is not None and b.requires_grad else None
    sdpa.sdpa_bwd(
        q,
        k,
        v,
        o,
        do,
        stats,
        dq=dq,
        dk=dk,
        dv=dv,
        scale=scale,
        dbias=dbias,
        bias=bias,
        mask=mask,
        dropout=dropout,
        varlen=varlen,
        engine=_engine("bwd"),
    )
    return hipError_t.hipSuccess


def attn_bwd(
    q,
    k,
    v,
    b,
    sm_scale,
    o,
    dout,
    dq,
    dk,
    dv,
    db,
    dq_acc,
    L,
    delta,
    dropout_p,
    philox_seed,
    philox_offset1,
    philox_offset2,
    causal,
    extargs=None,
):
    return _backward(
        q,
        k,
        v,
        o,
        dout,
        _dense_stats(L, q),
        dq=dq,
        dk=dk,
        dv=dv,
        scale=float(sm_scale),
        b=b,
        db=db,
        mask=_mask(causal),
        dropout=_dropout(dropout_p, philox_seed, philox_offset1, philox_offset2),
        varlen=None,
    )


def attn_bwd_varlen(
    q,
    k,
    v,
    seqinfo_q0,
    seqinfo_k0,
    max_seqlen_q,
    max_seqlen_k,
    seqinfo_q1,
    seqinfo_k1,
    b,
    sm_scale,
    o,
    dout,
    dq,
    dk,
    dv,
    db,
    dq_acc,
    L,
    delta,
    dropout_p,
    philox_seed,
    philox_offset1,
    philox_offset2,
    causal,
    varlen_type,
    lse_layout="HT",
    extargs=None,
):
    varlen = _varlen(
        varlen_type,
        seqinfo_q0,
        seqinfo_k0,
        max_seqlen_q,
        max_seqlen_k,
        seqinfo_q1,
        seqinfo_k1,
        q,
        k,
    )
    return _backward(
        q,
        k,
        v,
        o,
        dout,
        _varlen_stats(L, q, varlen_type == "padded", lse_layout),
        dq=dq,
        dk=dk,
        dv=dv,
        scale=float(sm_scale),
        b=b,
        db=db,
        mask=_mask(causal),
        dropout=_dropout(dropout_p, philox_seed, philox_offset1, philox_offset2),
        varlen=varlen,
    )


# --------------------------------------------------------------------------- misc


def debug_simulate_encoded_softmax(
    R, dropout_p, philox_seed, philox_offset1, philox_offset2
):
    dropout = _dropout(dropout_p, philox_seed, philox_offset1, philox_offset2)
    keep = _keep_mask(dropout, R.shape[0], R.shape[1], R.shape[2], R.shape[3], R.device)
    R.copy_(torch.where(keep, 0.5, -0.5).to(R.dtype))
    return hipError_t.hipSuccess


def lazy_dq_acc(dq):
    return torch.empty(0, device=dq.device)


def lazy_delta(L):
    return torch.empty(0, device=L.device)
