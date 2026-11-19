# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-device numerics of the RDNA WMMA prefill (kernels/gfx1151/wmma_fmha_fwd_v2).

Each case runs one compiled bucket object against a float64 reference over
every batch, head and row, and checks O and the natural-log LSE. Coverage:
GQA and a V head count different from the K head count, BSHD / BHSD and
padded strides, runtime head widths (multiples of 8) inside a bucket, the
causal class (top-left / bottom-right, sliding and two-sided windows, fully
masked rows), additive f32 / Q-dtype bias with broadcast strides, the wave
pair and gfx12's unsplit wide Q, large logits, an outlier key, and long
context. A separate test checks that a row's O is bit-independent of scores
only its wave neighbours see.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from rocke.core.arch import generic_arch_from_target_id
from rocke.runtime.hip_module import get_device_arch

_ARCH = generic_arch_from_target_id(get_device_arch() or "")
_NEEDS_RDNA = pytest.mark.skipif(
    _ARCH not in ("gfx11-generic", "gfx12-generic"),
    reason="needs an RDNA3/3.5 (gfx11) or RDNA4 (gfx12) GPU",
)
_BUCKETS = (64, 96, 128, 192, 256)
_LOG2E = 1.4426950408889634

# name: (B, Hq, Hkv, Hv, Sq, Sk, dq, dv, mask, bottom_right, wl, wr, options)
# options: layout ("bhsd"), pad (tokens, heads, elements), bias (kind,
# (b, h, q, k) full flags), bias_lead (elements between a dword boundary and
# the bias base pointer), bias_tail (every row's last key gets bias 4), scale,
# outlier, unsplit (gfx12 wave_pair=False), knobs (tuning-knob overrides the
# tuned candidates can select; B > 1 so the CTA order is observable).
CASES = {
    "none_tail_gqa_d64": (2, 3, 1, 1, 77, 45, 64, 64, "none", 0, -1, -1, {}),
    "none_bhsd_d128": (
        2,
        2,
        1,
        1,
        96,
        200,
        128,
        128,
        "none",
        0,
        -1,
        -1,
        {"layout": "bhsd"},
    ),
    "causal_tl_d64": (2, 4, 2, 2, 130, 130, 64, 64, "causal", 0, -1, 0, {}),
    "causal_br_d128": (2, 4, 1, 1, 77, 300, 128, 128, "causal", 1, -1, 0, {}),
    "window_br_bhsd_d128": (
        1,
        2,
        1,
        1,
        129,
        400,
        128,
        128,
        "causal",
        1,
        100,
        0,
        {"layout": "bhsd"},
    ),
    "band_both_d64": (1, 2, 1, 1, 200, 200, 64, 64, "causal", 0, 20, 9, {}),
    "rtdim_q40_v56": (2, 4, 2, 2, 70, 90, 40, 56, "none", 0, -1, -1, {}),
    "rtdim_q112_v128_window": (1, 3, 1, 1, 50, 99, 112, 128, "causal", 1, 33, 0, {}),
    "d96_none": (1, 2, 2, 2, 96, 160, 96, 96, "none", 0, -1, -1, {}),
    "pair_d256_causal": (1, 4, 2, 2, 200, 200, 256, 256, "causal", 0, -1, 0, {}),
    "pair_mla_q192_v128": (1, 4, 4, 4, 130, 130, 192, 128, "causal", 0, -1, 0, {}),
    "unsplit_mla_q192_v128": (
        1,
        4,
        4,
        4,
        130,
        130,
        192,
        128,
        "causal",
        0,
        -1,
        0,
        {"unsplit": True},
    ),
    "unsplit_q256_v128": (
        1,
        2,
        1,
        1,
        70,
        150,
        256,
        128,
        "none",
        0,
        -1,
        -1,
        {"unsplit": True},
    ),
    # V head count differs from the K head count.
    "hv4_hkv8_hq32_d128": (1, 32, 8, 4, 64, 96, 128, 128, "causal", 1, -1, 0, {}),
    "hv16_hkv8_hq32_d64": (1, 32, 8, 16, 70, 80, 64, 64, "none", 0, -1, -1, {}),
    "hv8_hkv8_hq32_d64": (1, 32, 8, 8, 70, 80, 64, 64, "none", 0, -1, -1, {}),
    "hv2_hkv4_pair_d256": (1, 8, 4, 2, 40, 70, 256, 256, "causal", 0, -1, 0, {}),
    # Additive bias.
    "bias_f32_full_d64": (
        2,
        3,
        3,
        3,
        77,
        90,
        64,
        64,
        "none",
        0,
        -1,
        -1,
        {"bias": ("f32", (1, 1, 1, 1))},
    ),
    "bias_q_alibi_d128": (
        1,
        4,
        4,
        4,
        130,
        130,
        128,
        128,
        "causal",
        0,
        -1,
        0,
        {"bias": ("q", (0, 1, 0, 1))},
    ),
    "bias_f32_pair_d256": (
        1,
        2,
        2,
        2,
        64,
        100,
        256,
        256,
        "causal",
        0,
        -1,
        0,
        {"bias": ("f32", (0, 1, 1, 1))},
    ),
    # Numerics edges.
    "fullmask_br_window": (1, 2, 1, 1, 300, 100, 64, 64, "causal", 1, 10, 0, {}),
    "fullmask_pair": (1, 2, 1, 1, 200, 40, 256, 256, "causal", 1, 5, 0, {}),
    "logits_1e4_causal": (
        1,
        2,
        1,
        1,
        200,
        200,
        128,
        128,
        "causal",
        0,
        -1,
        0,
        {"scale": 3500.0},
    ),
    "outlier_key_100x": (
        1,
        4,
        2,
        2,
        160,
        300,
        128,
        128,
        "causal",
        1,
        -1,
        0,
        {"outlier": 100.0},
    ),
    "width_q8_v8": (2, 2, 1, 1, 70, 90, 8, 8, "none", 0, -1, -1, {}),
    "width_q24_v40": (1, 3, 1, 1, 64, 77, 24, 40, "causal", 0, -1, 0, {}),
    "width_q136_v136": (1, 2, 2, 2, 70, 100, 136, 136, "causal", 0, -1, 0, {}),
    "pad_q40_v56": (
        2,
        2,
        2,
        2,
        50,
        70,
        40,
        56,
        "causal",
        1,
        -1,
        0,
        {"pad": (5, 2, 16)},
    ),
    "pad_pair_d256": (
        1,
        2,
        1,
        1,
        64,
        90,
        256,
        256,
        "causal",
        0,
        30,
        0,
        {"pad": (2, 1, 8)},
    ),
    "pad_bias_q": (
        1,
        2,
        2,
        2,
        60,
        70,
        64,
        64,
        "causal",
        0,
        -1,
        0,
        {"pad": (3, 1, 8), "bias": ("q", (0, 1, 0, 1))},
    ),
    # Q-dtype bias rows starting mid-dword (odd Sk), odd per-slice extents, and
    # a base pointer 2 bytes past a dword.
    "bias_q_odd_sk_full_d64": (
        2,
        3,
        1,
        1,
        37,
        53,
        64,
        64,
        "none",
        0,
        -1,
        -1,
        {"bias": ("q", (1, 1, 1, 1))},
    ),
    "bias_q_odd_sk_br_lead_d128": (
        2,
        2,
        2,
        2,
        37,
        53,
        128,
        128,
        "causal",
        1,
        -1,
        0,
        {"bias": ("q", (1, 0, 1, 1)), "bias_lead": 1},
    ),
    "bias_q_bcast_q_odd_lead_d64": (
        2,
        2,
        1,
        1,
        40,
        45,
        64,
        64,
        "none",
        0,
        -1,
        -1,
        {"bias": ("q", (1, 1, 0, 1)), "bias_lead": 1},
    ),
    "bias_q_pair_odd_lead_d256": (
        1,
        2,
        1,
        1,
        33,
        71,
        256,
        256,
        "causal",
        1,
        -1,
        0,
        {"bias": ("q", (0, 1, 1, 1)), "bias_lead": 1, "bias_tail": True},
    ),
    # Tuning-knob values only the opt-in tuned candidates select.
    "knob_hbq_causal_br_d128": (
        3,
        4,
        1,
        1,
        77,
        300,
        128,
        128,
        "causal",
        1,
        -1,
        0,
        {"knobs": {"cta_order": "hbq"}},
    ),
    "knob_bm32_band_d64": (
        2,
        2,
        1,
        1,
        200,
        200,
        64,
        64,
        "causal",
        0,
        20,
        9,
        {"knobs": {"block_m": 32}},
    ),
    "knob_eager_window_br_d128": (
        2,
        2,
        1,
        1,
        129,
        400,
        128,
        128,
        "causal",
        1,
        100,
        0,
        {"knobs": {"lazy_rescale": False}},
    ),
    "knob_pad0_tail_gqa_d64": (
        2,
        3,
        1,
        1,
        77,
        45,
        64,
        64,
        "none",
        0,
        -1,
        -1,
        {"knobs": {"k_pad": 0, "v_pad": 0}},
    ),
    "knob_pad16_causal_d128": (
        2,
        4,
        2,
        2,
        130,
        130,
        128,
        128,
        "causal",
        0,
        -1,
        0,
        {"knobs": {"k_pad": 16, "v_pad": 16}},
    ),
    "long_chunk_k32768": (1, 2, 1, 1, 256, 32768, 128, 128, "causal", 1, -1, 0, {}),
}


def _bucket(d):
    return next(b for b in _BUCKETS if d <= b)


def _host_bytes(array):
    return (ctypes.c_ubyte * array.nbytes).from_buffer(array.reshape(-1).view(np.uint8))


class _Tensor:
    """Device copy of a logical [B, S, H, D] array (optionally BHSD-ordered or
    padded with NaN), with its logical (batch, token, head) element strides."""

    def __init__(self, rt, array, layout="bshd", pad=None):
        self.rt, self.logical, self.layout, self.pad = rt, array, layout, pad
        if pad:
            B, S, H, D = array.shape
            phys = np.full(
                (B, S + pad[0], H + pad[1], D + pad[2]), np.nan, dtype=array.dtype
            )
            phys[:, :S, :H, :D] = array
            strides = phys.strides
            self.strides = tuple(s // phys.itemsize for s in strides[:3])
        elif layout == "bhsd":
            phys = np.ascontiguousarray(array.transpose(0, 2, 1, 3))
            st = [s // phys.itemsize for s in phys.strides]
            self.strides = (st[0], st[2], st[1])
        else:
            phys = np.ascontiguousarray(array)
            self.strides = tuple(s // phys.itemsize for s in phys.strides[:3])
        self.phys = phys
        self.ptr = rt.alloc(phys.nbytes)
        rt.memcpy_h2d(self.ptr, _host_bytes(phys), phys.nbytes)

    def read(self):
        out = np.empty_like(self.phys)
        self.rt.memcpy_d2h(_host_bytes(out), self.ptr, out.nbytes)
        if self.pad:
            B, S, H, D = self.logical.shape
            return out[:, :S, :H, :D]
        if self.layout == "bhsd":
            return out.transpose(0, 2, 1, 3)
        return out

    def free(self):
        self.rt.free(self.ptr)


def _reference(q, k, v, scale, bottom_right, wl, wr, causal, bias):
    """float64 attention; rows without keys -> O = 0, LSE = -inf."""
    B, Sq, Hq, _ = q.shape
    Sk, Hk, Hv = k.shape[1], k.shape[2], v.shape[2]
    out = np.zeros((B, Sq, Hq, v.shape[3]))
    lse = np.full((B, Hq, Sq), -np.inf)
    rows = np.arange(Sq)[:, None]
    keys = np.arange(Sk)[None, :]
    keep = np.ones((Sq, Sk), bool)
    if causal:
        diag = rows + ((Sk - Sq) if bottom_right else 0)
        if wr >= 0:
            keep &= keys <= diag + wr
        if wl >= 0:
            keep &= keys >= diag - wl
    for bi in range(B):
        for h in range(Hq):
            kf = k[bi, :, h // (Hq // Hk)].astype(np.float64)
            vf = v[bi, :, h // (Hq // Hv)].astype(np.float64)
            s = q[bi, :, h].astype(np.float64) @ kf.T * scale
            if bias is not None:
                s = s + bias[
                    min(bi, bias.shape[0] - 1), min(h, bias.shape[1] - 1)
                ].astype(np.float64)
            s = np.where(keep, s, -np.inf)
            m = s.max(axis=1, keepdims=True)
            live = np.isfinite(m[:, 0])
            p = np.exp(
                np.where(live[:, None], s - np.where(live[:, None], m, 0), -np.inf)
            )
            l = p.sum(axis=1, keepdims=True)
            out[bi, :, h] = np.where(
                live[:, None], (p / np.where(l == 0, 1, l)) @ vf, 0
            )
            lse[bi, h] = np.where(
                live, m[:, 0] + np.log(np.where(l[:, 0] == 0, 1, l[:, 0])), -np.inf
            )
    return out, lse


def _launch(
    spec,
    q,
    k,
    v,
    *,
    scale,
    bottom_right=0,
    wl=-1,
    wr=-1,
    layout="bshd",
    pad=None,
    write_lse=False,
    bias=None,
    bias_lead=0,
):
    """O, and the natural-log LSE [B, Hq, Sq] when ``write_lse`` (else None), of
    one launch of ``spec`` on logical [B, S, H, D] ``q`` / ``k`` / ``v`` stored
    in ``layout`` / ``pad``. ``bias`` is a logical [b, h, q, k] array whose
    size-1 axes broadcast, placed ``bias_lead`` NaN elements past a dword
    boundary (a misread surfaces)."""
    from kernels.gfx1151.wmma_fmha_fwd_v2 import (
        build_wmma_fmha_fwd_v2,
        wmma_fmha_fwd_v2_grid,
        wmma_fmha_fwd_v2_signature,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.packing import pack_args

    B, Sq, Hq, dq = q.shape
    Sk, Hk = k.shape[1], k.shape[2]
    Hv, dv = v.shape[2], v.shape[3]
    rt = Runtime()
    tensors = []
    module = None
    try:
        tq, tk, tv = (_Tensor(rt, x, layout, pad) for x in (q, k, v))
        to = _Tensor(rt, np.zeros((B, Sq, Hq, dv), q.dtype), layout, pad)
        tensors = [tq, tk, tv, to]
        values = dict(
            Q=tq.ptr, K=tk.ptr, V=tv.ptr, O=to.ptr, LSE=0,
            scale_log2=scale * _LOG2E, batch=B, seqlen_q=Sq, seqlen_k=Sk,
            num_query_heads=Hq, num_kv_heads=Hk, num_v_heads=Hv, head_dim_q=dq, head_dim_v=dv,
            bottom_right=bottom_right, window_left=wl, window_right=wr, write_lse=0,
            stride_lse_batch=0, stride_lse_token=0, stride_lse_head=0,
        )  # fmt: skip
        if write_lse:
            tl = _Tensor(rt, np.zeros((B, Hq, Sq, 1), np.float32))
            tensors.append(tl)
            values.update(
                LSE=tl.ptr,
                write_lse=1,
                stride_lse_batch=Hq * Sq,
                stride_lse_token=1,
                stride_lse_head=Sq,
            )
        for name, t in (("q", tq), ("k", tk), ("v", tv), ("o", to)):
            sb, st, sh = t.strides
            values.update(
                {
                    f"stride_{name}_batch": sb,
                    f"stride_{name}_token": st,
                    f"stride_{name}_head": sh,
                }
            )
        if bias is not None:
            flat = np.concatenate(
                [np.full(bias_lead, np.nan, bias.dtype), bias.reshape(-1)]
            )
            tb = _Tensor(rt, flat)
            tensors.append(tb)
            bs = [
                s // bias.itemsize if n > 1 else 0
                for s, n in zip(bias.strides, bias.shape)
            ]
            values.update(
                attn_bias_ptr=tb.ptr + bias_lead * bias.itemsize,
                bias_stride_b=bs[0],
                bias_stride_h=bs[1],
                bias_stride_q=bs[2],
            )
        artifact = compile_kernel(
            build_wmma_fmha_fwd_v2(spec, arch=_ARCH), arch=_ARCH, backend="python"
        )
        module = rt.load_module(artifact.hsaco)
        grid, block = wmma_fmha_fwd_v2_grid(
            spec, batch=B, seqlen_q=Sq, num_query_heads=Hq
        )
        rt.launch(
            module.get_function(artifact.kernel_name),
            grid,
            block,
            pack_args(wmma_fmha_fwd_v2_signature(spec), values),
        )
        rt.sync()
        return to.read(), (tl.read()[..., 0] if write_lse else None)
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        for t in tensors:
            t.free()


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_prefill_matches_reference(case, dtype):
    B, Hq, Hk, Hv, Sq, Sk, dq, dv, mask, br, wl, wr, opt = CASES[case]
    if opt.get("unsplit") and _ARCH != "gfx12-generic":
        pytest.skip("the unsplit wide-Q object is gfx12-only")
    if dtype == "bf16":
        npdt = pytest.importorskip("ml_dtypes").bfloat16
    else:
        npdt = np.float16
    from kernels.gfx1151.wmma_fmha_fwd_v2 import WmmaFmhaFwdV2Spec

    bias_opt = opt.get("bias")
    spec = WmmaFmhaFwdV2Spec(
        dtype=dtype,
        head_dim=_bucket(dq),
        head_dim_v=_bucket(dv),
        mask=mask,
        bias=bias_opt[0] if bias_opt else None,
        wave_pair=False if opt.get("unsplit") else None,
        **opt.get("knobs", {}),
    )
    rng = np.random.default_rng(11)

    def mk(s, h, d):
        return (rng.standard_normal((B, s, h, d), dtype=np.float32) * 0.5).astype(npdt)

    q, k, v = mk(Sq, Hq, dq), mk(Sk, Hk, dq), mk(Sk, Hv, dv)
    if "outlier" in opt:
        k[:, Sk // 3] = (k[:, Sk // 3].astype(np.float32) * opt["outlier"]).astype(npdt)
    bias = None
    if bias_opt:
        kind, flags = bias_opt
        shape = tuple(full if f else 1 for f, full in zip(flags, (B, Hq, Sq, Sk)))
        bias = rng.standard_normal(shape, dtype=np.float32).astype(
            np.float32 if kind == "f32" else npdt
        )
        if opt.get("bias_tail"):
            bias[..., -1] = 4.0
    scale = opt.get("scale", dq**-0.5)

    out, got_lse = _launch(
        spec,
        q,
        k,
        v,
        scale=scale,
        bottom_right=br,
        wl=wl,
        wr=wr,
        layout=opt.get("layout", "bshd"),
        pad=opt.get("pad"),
        write_lse=True,
        bias=bias,
        bias_lead=opt.get("bias_lead", 0),
    )
    out = out.astype(np.float64)
    ref, ref_lse = _reference(q, k, v, scale, br, wl, wr, mask == "causal", bias)
    tol = 3e-2 if dtype == "bf16" else 1e-2
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=0, atol=tol)
    fin = np.isfinite(ref_lse)
    assert np.all(np.isneginf(got_lse[~fin]))
    if fin.any():
        lse_tol = 1e-2 + 1e-5 * float(np.abs(ref_lse[fin]).max())
        np.testing.assert_allclose(got_lse[fin], ref_lse[fin], rtol=0, atol=lse_tol)


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("mask", ["causal", "bias_lowest"])
def test_prefill_rows_independent_of_wave_neighbours(mask, dtype):
    """A row's O must not depend on scores only other rows see. Under causal
    S = 72, key 71 is visible to row 71 alone; giving it huge K / V must leave
    rows 0..70 bit-equal, although row 71 shares a wave (rows 64..79) with
    them and its max jumps (the lazy-rescale decision is per row). The mask is
    the runtime causal bound, or a bias of the dtype's lowest finite value."""
    if dtype == "bf16":
        npdt = pytest.importorskip("ml_dtypes").bfloat16
        lowest = float(pytest.importorskip("ml_dtypes").finfo(npdt).min)
    else:
        npdt = np.float16
        lowest = float(np.finfo(np.float16).min)
    from kernels.gfx1151.wmma_fmha_fwd_v2 import WmmaFmhaFwdV2Spec

    B, S, H, D = 1, 72, 16, 64
    rng = np.random.default_rng(5)
    q, k, v = (
        rng.standard_normal((B, S, H, D), dtype=np.float32).astype(npdt)
        for _ in range(3)
    )
    kp, vp = k.copy(), v.copy()
    for t in (kp, vp):
        t[:, S - 1] = ((rng.integers(0, 2, (B, H, D)) * 2 - 1) * 1e4).astype(npdt)
    if mask == "causal":
        spec = WmmaFmhaFwdV2Spec(dtype=dtype, head_dim=D, mask="causal")
        launch = dict(wl=-1, wr=0)
    else:
        spec = WmmaFmhaFwdV2Spec(dtype=dtype, head_dim=D, mask="causal", bias="q")
        keep = np.arange(S)[None, :] <= np.arange(S)[:, None]
        bias = np.where(keep, 0.0, lowest).astype(npdt)[None, None]
        launch = dict(wl=-1, wr=-1, bias=bias)
    clean, _ = _launch(spec, q, k, v, scale=D**-0.5, **launch)
    poisoned, _ = _launch(spec, q, kp, vp, scale=D**-0.5, **launch)
    assert np.isfinite(poisoned.astype(np.float32)).all()
    rows = slice(0, S - 1)
    assert np.array_equal(
        clean[:, rows].view(np.uint16), poisoned[:, rows].view(np.uint16)
    )
