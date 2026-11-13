# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-device numerics of the RDNA WMMA prefill's optional features.

Every case runs one compiled ``kernels/gfx1151/wmma_fmha_fwd_v2`` object
against a float64 reference over every sequence, head and row, checking O and
the natural-log LSE. Features: softcap, attention sinks, ALiBi, QQ-bias, OCP
E4M3FN K/V with runtime scales, ragged (varlen) and paged K/V (per-tile and
per-row block-table lookups, shuffled pages, NaN-poisoned unused slots), each
alone and in model combinations (gpt-oss sinks + sliding window at D64 GQA,
Gemma-2 softcap + sliding window at D256, everything at once).

Reference order (natural domain): scale, softcap, ALiBi ``slope * (k -
ctx)``, QQ-bias ``qq[q, k - ctx]`` inside its bounds, additive bias, then the
causal / window masks; a sink appends one logit column with no value row.
``ctx = Sk - Sq`` with bottom-right alignment, else 0; positions are
sequence-local. A row with no visible key and no sink gives O = 0, LSE = -inf.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from rocke.core.arch import generic_arch_from_target_id
from rocke.runtime.hip_module import get_device_arch

_ARCH = generic_arch_from_target_id(get_device_arch() or "")
_NEEDS_RDNA = pytest.mark.skipif(
    _ARCH not in ("gfx11-generic", "gfx12-generic"),
    reason="needs an RDNA3/3.5 (gfx11) or RDNA4 (gfx12) GPU",
)
_LOG2E = 1.4426950408889634

_GPTOSS = dict(Hq=16, Hkv=2, dq=64, mask="causal", wl=127, wr=0)

# name -> options. Common keys: layout (dense / ragged / paged), B + Sq + Sk
# (dense) or lens_q + lens_k (packed), Hq, Hkv, Hv, dq, dv, mask, br, wl, wr,
# flags (c softcap, s sinks, a ALiBi, q QQ-bias, 8 FP8 K/V), cap, scale, ps
# (page size), bias ((kind, (b, h, q, k) full flags)), qq (rows, cols, pad),
# bn (block_n), unsplit (gfx12 wave_pair=False), ninf ((n, fill): the first n
# keys' bias / QQ-bias entries become fill, "inf" -> -inf, "finfo" -> the
# dtype's most negative finite value).
CASES = {
    "softcap_none_d128": dict(B=2, Sq=77, Sk=150, dq=128, flags="c", cap=5.0, scale=1.0),
    "softcap_large_logits_d64": dict(
        B=1, Sq=64, Sk=90, dq=64, flags="c", cap=30.0, scale=40.0
    ),
    "gemma2_softcap_swa_d256": dict(
        B=1, Sq=200, Sk=200, Hq=8, Hkv=4, dq=256, mask="causal", wl=63, wr=0,
        flags="c", cap=50.0, scale=1.0,
    ),
    "gptoss_sinks_swa_d64": dict(B=2, Sq=300, Sk=300, flags="s", **_GPTOSS),
    "sinks_none_d128": dict(B=2, Sq=70, Sk=130, dq=128, flags="s"),
    "sinks_fullmask_br_window": dict(
        B=1, Sq=300, Sk=100, dq=64, mask="causal", br=1, wl=10, wr=0, flags="s"
    ),
    "sinks_empty_kv": dict(B=1, Sq=40, Sk=0, dq=64, flags="s"),
    "alibi_causal_br_d128": dict(
        B=2, Sq=77, Sk=300, Hq=4, Hkv=1, dq=128, mask="causal", br=1, wr=0,
        flags="a",
    ),
    "alibi_none_d64": dict(B=1, Sq=90, Sk=120, dq=64, flags="a"),
    "qq_causal_d128": dict(
        B=1, Sq=100, Sk=100, dq=128, mask="causal", wr=0, flags="q",
        qq=(50, 60, 5),
    ),
    "qq_none_br_d64": dict(B=2, Sq=40, Sk=100, dq=64, br=1, flags="q", qq=(64, 64, 0)),
    "fp8_none_d128": dict(B=2, Sq=77, Sk=150, dq=128, flags="8"),
    "fp8_causal_d64": dict(B=1, Sq=130, Sk=130, dq=64, mask="causal", wr=0, flags="8"),
    "fp8_pair_d256": dict(
        B=1, Sq=64, Sk=100, Hq=4, Hkv=2, dq=256, mask="causal", wr=0, flags="8"
    ),
    "ragged_causal_br_d128": dict(
        layout="ragged", lens_q=(0, 17, 1, 130, 0, 64), lens_k=(9, 17, 5, 300, 0, 64),
        Hq=4, Hkv=2, dq=128, mask="causal", br=1, wr=0,
    ),
    "ragged_none_d64": dict(layout="ragged", lens_q=(33, 0, 70), lens_k=(0, 12, 140), dq=64),
    "paged16_causal_d128": dict(
        layout="paged", ps=16, lens_q=(20, 1, 77), lens_k=(150, 33, 77), dq=128,
        mask="causal", br=1, wr=0,
    ),
    "paged32_causal_d128": dict(
        layout="paged", ps=32, lens_q=(64, 3), lens_k=(64, 200), dq=128,
        mask="causal", br=1, wr=0,
    ),
    "paged64_none_d64": dict(layout="paged", ps=64, lens_q=(50, 9), lens_k=(130, 0), dq=64),
    "paged8_causal_d64": dict(
        layout="paged", ps=8, lens_q=(30, 5), lens_k=(71, 40), dq=64, mask="causal",
        br=1, wr=0,
    ),
    "paged16_pair_d256": dict(
        layout="paged", ps=16, lens_q=(40, 2), lens_k=(90, 60), Hq=4, Hkv=2,
        dq=256, mask="causal", br=1, wr=0,
    ),
    "gptoss_paged16_sinks_swa_d64": dict(
        layout="paged", ps=16, lens_q=(1, 5, 33), lens_k=(100, 300, 513), br=1,
        flags="s", **_GPTOSS,
    ),
    "fp8_ragged_causal_d128": dict(
        layout="ragged", lens_q=(17, 0, 90), lens_k=(40, 7, 90), dq=128,
        mask="causal", br=1, wr=0, flags="8",
    ),
    "fp8_paged32_causal_d128": dict(
        layout="paged", ps=32, lens_q=(20, 64), lens_k=(150, 64), dq=128,
        mask="causal", br=1, wr=0, flags="8",
    ),
    "sinks_alibi_paged16": dict(
        layout="paged", ps=16, lens_q=(10, 40), lens_k=(70, 40), dq=128,
        mask="causal", br=1, wr=0, flags="sa",
    ),
    "qq_window_ragged": dict(
        layout="ragged", lens_q=(64, 128, 32), lens_k=(64, 256, 256), dq=128,
        mask="causal", br=1, wl=40, wr=0, flags="q", qq=(32, 64, 3),
    ),
    "softcap_alibi_d128": dict(B=1, Sq=90, Sk=140, dq=128, flags="ca", cap=8.0, scale=1.0),
    "bias_f32_sinks_d64": dict(
        B=2, Sq=60, Sk=80, dq=64, mask="causal", wr=0, flags="s",
        bias=("f32", (1, 1, 1, 1)),
    ),
    # Q-dtype bias with odd key lengths: per-sequence extents end mid-dword.
    "bias_q_ragged_odd_k_d64": dict(
        layout="ragged", lens_q=(5, 17, 1, 25), lens_k=(9, 17, 5, 40), dq=64,
        mask="causal", br=1, wr=0, bias=("q", (1, 0, 1, 1)),
    ),
    "bias_q_softcap_odd_sk_d128": dict(
        B=2, Sq=37, Sk=53, dq=128, flags="c", cap=20.0, bias=("q", (1, 1, 1, 1)),
    ),
    "everything_paged16_fp8": dict(
        layout="paged", ps=16, lens_q=(20, 0, 50), lens_k=(90, 30, 50), Hq=4,
        Hkv=2, dq=128, mask="causal", br=1, wl=50, wr=0, flags="csaq8", cap=20.0,
        qq=(40, 48, 2), bias=("q", (1, 1, 1, 1)),
    ),
    "softcap_sinks_pair_d256": dict(
        B=1, Sq=80, Sk=120, dq=256, mask="causal", wr=0, flags="cs", cap=30.0
    ),
    "bn64_paged32_rows": dict(
        layout="paged", ps=32, lens_q=(30, 70), lens_k=(100, 70), dq=64,
        mask="causal", br=1, wr=0, bn=64,
    ),
    "bn64_paged64_sinks": dict(
        layout="paged", ps=64, lens_q=(30, 70), lens_k=(150, 70), dq=64,
        mask="causal", br=1, wr=0, bn=64, flags="s",
    ),
    "bn16_fp8_ragged": dict(
        layout="ragged", lens_q=(20, 50), lens_k=(33, 50), dq=128, mask="causal",
        br=1, wr=0, bn=16, flags="8",
    ),
    "unsplit_q192_v128_softcap_fp8": dict(
        B=1, Sq=70, Sk=100, dq=192, dv=128, mask="causal", wr=0, flags="c8",
        cap=20.0, unsplit=True,
    ),
    "unsplit_q256_v128_paged32_sinks": dict(
        layout="paged", ps=32, lens_q=(40, 9), lens_k=(70, 100), dq=256, dv=128,
        flags="s", unsplit=True,
    ),
    # A -inf bias / QQ-bias over a row's first whole key tile: the running max
    # stays -inf through an unmasked tile, which must not produce -inf - -inf.
    # bf16 finfo.min (the torch padding-mask idiom) overflows to -inf in log2.
    "bias_f32_ninf_tile_d64": dict(
        B=1, Sq=40, Sk=64, dq=64, bias=("f32", (0, 0, 0, 1)), ninf=(32, "inf")
    ),
    "bias_q_ninf_tile_d128": dict(
        B=1, Sq=40, Sk=96, dq=128, bias=("q", (0, 0, 0, 1)), ninf=(32, "inf")
    ),
    "bias_q_finfo_tile_d64": dict(
        B=1, Sq=40, Sk=64, dq=64, bias=("q", (0, 0, 0, 1)), ninf=(32, "finfo")
    ),
    "bias_f32_ninf_causal_br_d64": dict(
        B=1, Sq=8, Sk=200, dq=64, mask="causal", br=1, wr=0,
        bias=("f32", (0, 0, 0, 1)), ninf=(64, "inf"),
    ),
    "qq_ninf_tile_d64": dict(
        B=1, Sq=40, Sk=64, dq=64, flags="q", qq=(40, 64, 0), ninf=(32, "inf")
    ),
}  # fmt: skip

_K_SCALE, _V_SCALE = 0.3, 0.7


def _bucket(d):
    return next(b for b in (64, 96, 128, 192, 256) if d <= b)


def _host_bytes(array):
    return (ctypes.c_ubyte * array.nbytes).from_buffer(array.reshape(-1).view(np.uint8))


def _reference_seq(q, k, v, scale, opt, *, sinks, slopes, qq, bias_fn):
    """float64 attention of one sequence: q [Sq, Hq, D], k [Sk, Hkv, D],
    v [Sk, Hv, Dv] -> O [Sq, Hq, Dv], LSE [Hq, Sq]."""
    Sq, Hq = q.shape[:2]
    Sk, Hk, Hv, Dv = k.shape[0], k.shape[1], v.shape[1], v.shape[2]
    ctx = (Sk - Sq) if opt["br"] else 0
    rows = np.arange(Sq)[:, None]
    keys = np.arange(Sk)[None, :]
    keep = np.ones((Sq, Sk), bool)
    if opt["mask"] == "causal":
        diag = rows + ctx
        if opt["wr"] >= 0:
            keep &= keys <= diag + opt["wr"]
        if opt["wl"] >= 0:
            keep &= keys >= diag - opt["wl"]
    out = np.zeros((Sq, Hq, Dv))
    lse = np.full((Hq, Sq), -np.inf)
    for h in range(Hq):
        s = q[:, h] @ k[:, h // (Hq // Hk)].T * scale
        if "c" in opt["flags"]:
            s = opt["cap"] * np.tanh(s / opt["cap"])
        if slopes is not None:
            s = s + slopes[h] * (keys - ctx)
        if qq is not None:
            rel = np.broadcast_to(keys - ctx, (Sq, Sk))
            r = np.broadcast_to(rows, (Sq, Sk))
            ok = (r < qq.shape[0]) & (rel >= 0) & (rel < qq.shape[1])
            vals = qq[np.minimum(r, qq.shape[0] - 1), np.clip(rel, 0, qq.shape[1] - 1)]
            s = s + np.where(ok, vals, 0.0)
        if bias_fn is not None:
            s = s + bias_fn(h)
        s = np.where(keep, s, -np.inf)
        vh = v[:, h // (Hq // Hv)]
        if sinks is not None:
            s = np.concatenate([s, np.full((Sq, 1), sinks[h])], axis=1)
            vh = np.concatenate([vh, np.zeros((1, Dv))])
        if s.shape[1] == 0:
            continue
        m = s.max(axis=1, keepdims=True)
        live = np.isfinite(m[:, 0])
        p = np.exp(np.where(live[:, None], s - np.where(live[:, None], m, 0), -np.inf))
        l = p.sum(axis=1, keepdims=True)
        out[:, h] = np.where(live[:, None], (p / np.where(l == 0, 1, l)) @ vh, 0)
        lse[h] = np.where(
            live, m[:, 0] + np.log(np.where(l[:, 0] == 0, 1, l[:, 0])), -np.inf
        )
    return out, lse


class _Dev:
    """Device allocations of one launch."""

    def __init__(self, rt):
        self.rt, self.ptrs = rt, []

    def put(self, array):
        array = np.ascontiguousarray(array)
        ptr = self.rt.alloc(max(array.nbytes, 4))
        if array.nbytes:
            self.rt.memcpy_h2d(ptr, _host_bytes(array), array.nbytes)
        self.ptrs.append(ptr)
        return ptr

    def get(self, ptr, like):
        out = np.empty_like(like)
        if out.nbytes:
            self.rt.memcpy_d2h(_host_bytes(out), ptr, out.nbytes)
        return out

    def free(self):
        for ptr in self.ptrs:
            self.rt.free(ptr)


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_prefill_feature_matches_reference(case, dtype):
    opt = dict(
        layout="dense", Hq=4, Hkv=2, Hv=None, dv=None, mask="none", br=0, wl=-1,
        wr=-1, flags="", cap=None, scale=None, ps=0, bias=None, qq=None, bn=32,
        unsplit=False, ninf=None,
    )  # fmt: skip
    opt.update(CASES[case])
    if opt["unsplit"] and _ARCH != "gfx12-generic":
        pytest.skip("the unsplit wide-Q object is gfx12-only")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    npdt = ml_dtypes.bfloat16 if dtype == "bf16" else np.float16
    from kernels.gfx1151.wmma_fmha_fwd_v2 import (
        WmmaFmhaFwdV2Spec,
        build_wmma_fmha_fwd_v2,
        wmma_fmha_fwd_v2_grid,
        wmma_fmha_fwd_v2_signature,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.packing import pack_args

    layout, flags = opt["layout"], opt["flags"]
    Hq, Hk = opt["Hq"], opt["Hkv"]
    Hv = opt["Hv"] or Hk
    dq = opt["dq"]
    dv = opt["dv"] or dq
    fp8 = "8" in flags
    if layout == "dense":
        lens_q, lens_k = (opt["Sq"],) * opt["B"], (opt["Sk"],) * opt["B"]
    else:
        lens_q, lens_k = opt["lens_q"], opt["lens_k"]
    B, max_q, max_k = len(lens_q), max(lens_q), max(lens_k)
    scale = opt["scale"] or dq**-0.5
    spec = WmmaFmhaFwdV2Spec(
        dtype=dtype,
        head_dim=_bucket(dq),
        head_dim_v=_bucket(dv),
        mask=opt["mask"],
        bias=opt["bias"][0] if opt["bias"] else None,
        use_softcap="c" in flags,
        use_sinks="s" in flags,
        use_alibi="a" in flags,
        use_qq_bias="q" in flags,
        layout=layout,
        page_block_size=opt["ps"],
        kv_dtype="fp8e4m3" if fp8 else "",
        block_n=opt["bn"],
        wave_pair=False if opt["unsplit"] else None,
    )
    rng = np.random.default_rng(23)

    def rnd(*shape):
        return rng.standard_normal(shape, dtype=np.float32) * 0.5

    # Logical per-sequence tensors (float64 reference values).
    qs = [rnd(n, Hq, dq).astype(npdt) for n in lens_q]
    if fp8:
        f8 = ml_dtypes.float8_e4m3fn
        ks8 = [(rnd(n, Hk, dq) / _K_SCALE).astype(f8) for n in lens_k]
        vs8 = [(rnd(n, Hv, dv) / _V_SCALE).astype(f8) for n in lens_k]
        ks = [x.astype(np.float64) * _K_SCALE for x in ks8]
        vs = [x.astype(np.float64) * _V_SCALE for x in vs8]
        ks_dev, vs_dev = [x.view(np.uint8) for x in ks8], [
            x.view(np.uint8) for x in vs8
        ]
    else:
        ks_dev = [rnd(n, Hk, dq).astype(npdt) for n in lens_k]
        vs_dev = [rnd(n, Hv, dv).astype(npdt) for n in lens_k]
        ks, vs = ks_dev, vs_dev
    sinks = (rng.standard_normal(Hq) * 2).astype(npdt) if "s" in flags else None
    slopes = (
        np.array([2.0 ** (-8.0 * (i + 1) / Hq) for i in range(Hq)], np.float32)
        if "a" in flags
        else None
    )
    qq = qq_dev = None
    if "q" in flags:
        rows, cols, pad = opt["qq"]
        qq_dev = np.zeros((rows, cols + pad), np.float32)
        qq_dev[:, :cols] = rng.standard_normal((rows, cols))
        if opt["ninf"]:
            qq_dev[:, : opt["ninf"][0]] = -np.inf
        qq = qq_dev[:, :cols].astype(np.float64)
    bias = None
    if opt["bias"]:
        kind, full = opt["bias"]
        shape = tuple(n if f else 1 for f, n in zip(full, (B, Hq, max_q, max_k)))
        bias = rng.standard_normal(shape, dtype=np.float32).astype(
            np.float32 if kind == "f32" else npdt
        )
        if opt["ninf"]:
            n, fill = opt["ninf"]
            bias[..., :n] = (
                ml_dtypes.finfo(bias.dtype).min if fill == "finfo" else -np.inf
            )

    # Reference.
    refs, ref_lses = [], []
    for bi in range(B):
        bias_fn = None
        if bias is not None:
            sq, sk = lens_q[bi], lens_k[bi]

            def bias_fn(h, bi=bi, sq=sq, sk=sk):
                blk = bias[min(bi, bias.shape[0] - 1), min(h, bias.shape[1] - 1)]
                return np.broadcast_to(blk.astype(np.float64), (max_q, max_k))[:sq, :sk]

        o, l = _reference_seq(
            qs[bi].astype(np.float64),
            np.asarray(ks[bi], np.float64),
            np.asarray(vs[bi], np.float64),
            scale,
            opt,
            sinks=None if sinks is None else sinks.astype(np.float64),
            slopes=slopes,
            qq=qq,
            bias_fn=bias_fn,
        )
        refs.append(o)
        ref_lses.append(l)

    # Device layout.
    rt = Runtime()
    dev = _Dev(rt)
    module = None
    kv_np = np.uint8 if fp8 else npdt
    values = dict(
        scale_log2=scale * _LOG2E, batch=B, seqlen_q=max_q, seqlen_k=max_k,
        num_query_heads=Hq, num_kv_heads=Hk, num_v_heads=Hv, head_dim_q=dq,
        head_dim_v=dv, bottom_right=opt["br"], window_left=opt["wl"],
        window_right=opt["wr"], write_lse=1,
    )  # fmt: skip
    try:
        if layout == "dense":
            q_all = np.stack(qs)
            k_all, v_all = np.stack(ks_dev), np.stack(vs_dev)
            o_like = np.zeros((B, max_q, Hq, dv), npdt)
            lse_like = np.zeros((B, Hq, max_q), np.float32)
            strides = {
                "q": (max_q * Hq * dq, Hq * dq, dq),
                "k": (max_k * Hk * dq, Hk * dq, dq),
                "v": (max_k * Hv * dv, Hv * dv, dv),
                "o": (max_q * Hq * dv, Hq * dv, dv),
                "lse": (Hq * max_q, 1, max_q),
            }
        else:
            total_q = sum(lens_q)
            q_all = np.concatenate(qs)
            o_like = np.zeros((total_q, Hq, dv), npdt)
            lse_like = np.zeros((Hq, max(total_q, 1)), np.float32)
            cu_q = np.concatenate([[0], np.cumsum(lens_q)]).astype(np.int32)
            values["cu_seqlens_q"] = dev.put(cu_q)
            strides = {
                "q": (0, Hq * dq, dq),
                "o": (0, Hq * dv, dv),
                "lse": (0, 1, max(total_q, 1)),
            }
            if layout == "ragged":
                k_all, v_all = np.concatenate(ks_dev), np.concatenate(vs_dev)
                cu_k = np.concatenate([[0], np.cumsum(lens_k)]).astype(np.int32)
                values["cu_seqlens_k"] = dev.put(cu_k)
                strides["k"] = (0, Hk * dq, dq)
                strides["v"] = (0, Hv * dv, dv)
            else:
                ps = opt["ps"]
                pages = [-(-n // ps) for n in lens_k]
                max_pages = max(max(pages), 1)
                n_pages = sum(pages) + 3  # a few never-referenced pages
                order = rng.permutation(n_pages)
                nan = 0x7F if fp8 else np.nan  # NaN-poison every unused slot
                k_all = np.full((n_pages, ps, Hk, dq), nan, kv_np)
                v_all = np.full((n_pages, ps, Hv, dv), nan, kv_np)
                table = np.zeros((B, max_pages + 2), np.int32)  # padded rows
                nxt = 0
                for bi, n in enumerate(lens_k):
                    for pg in range(pages[bi]):
                        phys = order[nxt]
                        nxt += 1
                        table[bi, pg] = phys
                        t0, t1 = pg * ps, min(n, (pg + 1) * ps)
                        k_all[phys, : t1 - t0] = ks_dev[bi][t0:t1]
                        v_all[phys, : t1 - t0] = vs_dev[bi][t0:t1]
                values["seqused_k"] = dev.put(np.array(lens_k, np.int32))
                values["block_table"] = dev.put(table)
                values["block_table_stride"] = table.shape[1]
                values["stride_k_block"] = ps * Hk * dq
                values["stride_v_block"] = ps * Hv * dv
                values["num_blocks"] = n_pages
                strides["k"] = (0, Hk * dq, dq)
                strides["v"] = (0, Hv * dv, dv)
        values["Q"] = dev.put(q_all)
        values["K"] = dev.put(k_all)
        values["V"] = dev.put(v_all)
        values["O"] = o_ptr = dev.put(np.full_like(o_like, np.nan))
        values["LSE"] = lse_ptr = dev.put(lse_like)
        for t, (sb, st, sh) in strides.items():
            values.update(
                {
                    f"stride_{t}_batch": sb,
                    f"stride_{t}_token": st,
                    f"stride_{t}_head": sh,
                }
            )
        if "c" in flags:
            values["softcap"] = opt["cap"]
        if sinks is not None:
            values["sink_ptr"] = dev.put(sinks)
        if slopes is not None:
            values["alibi_slopes_ptr"] = dev.put(slopes)
        if qq_dev is not None:
            values.update(
                qq_bias_ptr=dev.put(qq_dev),
                qq_bias_rows=qq_dev.shape[0],
                qq_bias_cols=opt["qq"][1],
                qq_bias_stride=qq_dev.shape[1],
            )
        if fp8:
            values.update(k_scale=_K_SCALE, v_scale=_V_SCALE)
        if bias is not None:
            bs = [
                s // bias.itemsize if n > 1 else 0
                for s, n in zip(bias.strides, bias.shape)
            ]
            values.update(
                attn_bias_ptr=dev.put(bias),
                bias_stride_b=bs[0],
                bias_stride_h=bs[1],
                bias_stride_q=bs[2],
            )
        artifact = compile_kernel(
            build_wmma_fmha_fwd_v2(spec, arch=_ARCH), arch=_ARCH, backend="python"
        )
        module = rt.load_module(artifact.hsaco)
        grid, block = wmma_fmha_fwd_v2_grid(
            spec, batch=B, seqlen_q=max(max_q, 1), num_query_heads=Hq
        )
        rt.launch(
            module.get_function(artifact.kernel_name),
            grid,
            block,
            pack_args(wmma_fmha_fwd_v2_signature(spec), values),
        )
        rt.sync()
        out = dev.get(o_ptr, o_like).astype(np.float64)
        got_lse = dev.get(lse_ptr, lse_like)
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        dev.free()

    if layout == "dense":
        outs = [out[bi] for bi in range(B)]
        lses = [got_lse[bi] for bi in range(B)]
    else:
        off = np.concatenate([[0], np.cumsum(lens_q)])
        outs = [out[off[bi] : off[bi + 1]] for bi in range(B)]
        lses = [got_lse[:, off[bi] : off[bi + 1]] for bi in range(B)]
    tol = 4e-2 if dtype == "bf16" else 2e-2
    for bi in range(B):
        o, ref = outs[bi], refs[bi]
        assert np.isfinite(o).all(), f"sequence {bi}: non-finite output"
        np.testing.assert_allclose(o, ref, rtol=0, atol=tol, err_msg=f"sequence {bi}")
        got, want = lses[bi], ref_lses[bi]
        fin = np.isfinite(want)
        assert np.all(np.isneginf(got[~fin])), f"sequence {bi}: LSE of empty rows"
        if fin.any():
            lse_tol = 1e-2 + 1e-5 * float(np.abs(want[fin]).max())
            np.testing.assert_allclose(
                got[fin], want[fin], rtol=0, atol=lse_tol, err_msg=f"sequence {bi}"
            )
