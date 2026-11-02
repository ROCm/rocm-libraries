# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side driver for the probe shell: allocate, launch, compare to the oracle.

Shared by the device numeric tests.  Every tensor lives in a flat buffer with a
trailing canary region; K/V/Q padding is NaN-poisoned and the output starts as a
sentinel value, so out-of-range reads that reach the result and out-of-range
writes are both observable.
"""

from __future__ import annotations

import ctypes
import math
import struct

import numpy as np

POISON_TAIL = 4096  # canary elements appended after every buffer
OUT_SENTINEL = 7.0
LSE_SENTINEL = 5.0


def enc(x, dtype):
    x = np.asarray(x, np.float32)
    if dtype == "f16":
        return x.astype(np.float16)
    u = x.view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def dec(u, dtype):
    if dtype == "f16":
        return u.astype(np.float32)
    return (u.astype(np.uint32) << 16).view(np.float32)


def _raw_dtype(dtype):
    return np.float16 if dtype == "f16" else np.uint16


def _poison(dtype):
    return np.float16(np.nan) if dtype == "f16" else np.uint16(0x7FC0)


def block_threads(arch):
    return 32 if arch.startswith(("gfx11", "gfx12")) else 64


def run_case(
    arch,
    *,
    B=2,
    Hq=4,
    Hk=4,
    Sq=37,
    Sk=53,
    alloc_q=None,
    alloc_k=None,
    D=64,
    layout="BHSD",
    pad=0,
    dtype="f16",
    left=None,
    right=None,
    top_left=True,
    bias=None,
    v_lds=False,
    fold=False,
    band_range=False,
    return_valid=True,
    fill="neg_inf",
    seed=1,
    tag="x",
    guard_k=True,
    lse_hook_factory=None,
    lse_layout="bhs",
    lse_tag="",
):
    import _attention_fwd_ext_harness as H
    from rocke.helpers.compile import compile_kernel
    from rocke.numeric.sdpa_reference import _band_mask, sdpa_reference
    from rocke.runtime.hip_module import Runtime

    alloc_q = alloc_q or Sq
    alloc_k = alloc_k or Sk
    group = Hq // Hk
    if fold:
        assert layout == "BSHD" and pad == 0
    name = (
        f"probe_sdpa_{arch}_{layout}_{tag}_{dtype}_D{D}_g{group}"
        f"_b{int(bias is not None)}_f{int(fold)}_r{int(band_range)}"
        f"_v{int(return_valid)}_{fill}_l{int(v_lds)}_k{int(guard_k)}{lse_tag}"
    )
    kernel, params = H.make_probe_kernel(
        name,
        arch=arch,
        head_size=D,
        group=group,
        dtype=dtype,
        has_bias=bias is not None,
        v_lds_stage=v_lds,
        fold_batch=fold,
        return_valid=return_valid,
        fill=fill,
        use_band_range=band_range,
        raw_stats=True,
        guard_k=guard_k,
        lse_hook_factory=lse_hook_factory,
    )
    art = compile_kernel(kernel, arch=arch, capture_ir_text=False, backend="python")

    rng = np.random.default_rng(seed)
    udt = _raw_dtype(dtype)

    def make(S_alloc, Hh, fill_value, S_valid):
        """Flat poisoned buffer, its per-batch stride, logical view, valid data."""
        n = S_alloc * Hh * D
        flat = np.full(B * (n + pad) + POISON_TAIL, fill_value, dtype=udt)
        raw = flat[: B * (n + pad)].reshape(B, n + pad)
        if layout == "BHSD":
            lg = raw[:, :n].reshape(B, Hh, S_alloc, D)
        else:
            lg = raw[:, :n].reshape(B, S_alloc, Hh, D).transpose(0, 2, 1, 3)
        vals = rng.standard_normal((B, Hh, S_valid, D)).astype(np.float32) * 0.5
        lg[:, :, :S_valid, :] = enc(vals, dtype)
        return flat, n + pad, lg, vals

    qf, qb, _, qv = make(alloc_q, Hq, _poison(dtype), Sq)
    kf, kb, _, kv_ = make(alloc_k, Hk, _poison(dtype), Sk)
    vf, vb, _, vv = make(alloc_k, Hk, _poison(dtype), Sk)
    ob = alloc_q * Hq * D + pad
    seven = enc(np.float32(OUT_SENTINEL), dtype)
    of = np.full(B * ob + POISON_TAIL, seven, dtype=udt)
    if layout == "BHSD":
        qt_, qh_, kt_, kh_ = D, alloc_q * D, D, alloc_k * D
    else:
        qt_, qh_, kt_, kh_ = Hq * D, D, Hk * D, D
    n_lse = B * Hq * alloc_q
    lse = np.full(n_lse + POISON_TAIL, LSE_SENTINEL, np.float32)
    stats = np.full(2 * n_lse + POISON_TAIL, LSE_SENTINEL, np.float32)
    bias_arr = np.zeros((1,), np.float32)
    bsb = bsh = bsq = 0
    if bias is not None:
        bias_arr = np.ascontiguousarray(bias, dtype=np.float32)
        bsb, bsh, bsq = Hq * Sq * Sk, Sq * Sk, Sk
    scale = 1.0 / math.sqrt(D)
    left_arg = -1 if left is None else left
    right_arg = -1 if right is None else right
    diag = 0 if top_left else Sk - Sq

    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    bufs = dict(Q=qf, K=kf, V=vf, O=of, LSE=lse, BIAS=bias_arr, STATS=stats)
    dev = {}
    for k_, a in bufs.items():
        dev[k_] = rt.alloc(max(a.nbytes, 16))
        rt.memcpy_h2d(dev[k_], u8(a), a.nbytes)
    values = dict(
        scale_log2=scale * math.log2(math.e),
        seqlen_q=Sq, seqlen_k=Sk,
        q_b=qb, q_h=qh_, q_t=qt_,
        k_b=kb, k_h=kh_, k_t=kt_,
        v_b=vb, v_h=kh_, v_t=kt_,
        o_b=ob, o_h=qh_, o_t=qt_,
        lse_b=Hq * alloc_q, lse_h=alloc_q if lse_layout == "bhs" else Hq,
        stats_h=alloc_q,
        bias_b=bsb, bias_h=bsh, bias_q=bsq,
        left=left_arg, right=right_arg, diag=diag, q_fold=alloc_q,
    )  # fmt: skip
    packed = b""
    for n_, f_ in params:
        if f_ == "Q":
            packed += struct.pack("<Q", dev[n_])
        else:
            packed += struct.pack("<" + f_, values[n_])
    grid = ((Sq + 15) // 16, Hq, B)
    rt.launch(fn, grid, (block_threads(arch), 1, 1), packed)
    rt.sync()
    for k_, a in (("O", of), ("LSE", lse), ("STATS", stats)):
        rt.memcpy_d2h(u8(a), dev[k_], a.nbytes)
    for p in dev.values():
        rt.free(p)
    mod.unload()

    # ---- compare with the oracle ----
    q_in, k_in, v_in = (dec(enc(x, dtype), dtype) for x in (qv, kv_, vv))
    ref_o, ref_lse = sdpa_reference(
        q_in, k_in, v_in, scale=scale, bias=bias,
        left_bound=left, right_bound=right, top_left=top_left,
    )  # fmt: skip
    oraw = of[: B * ob].reshape(B, ob)
    n_o = alloc_q * Hq * D
    if layout == "BHSD":
        got_lg = oraw[:, :n_o].reshape(B, Hq, alloc_q, D)
    else:
        got_lg = oraw[:, :n_o].reshape(B, alloc_q, Hq, D).transpose(0, 2, 1, 3)
    go = dec(got_lg[:, :, :Sq, :], dtype).astype(np.float64)
    res = {"o_got": go}
    res["err_o"] = float(np.max(np.abs(go - ref_o)))
    res["nan_free"] = not bool(np.isnan(go).any())
    if lse_layout == "bhs":
        lse_bh = lse[:n_lse].reshape(B, Hq, alloc_q)
    else:  # [B, S_q, H_q] element order (packed-token style strides)
        lse_bh = lse[:n_lse].reshape(B, alloc_q, Hq).transpose(0, 2, 1)
    gl = lse_bh[:, :, :Sq].astype(np.float64)
    res["lse_got"] = gl
    res["lse_inf_pattern"] = bool(
        np.all(np.isinf(gl) == np.isinf(ref_lse)) and np.all(gl[np.isinf(gl)] < 0)
    )
    fin = np.isfinite(ref_lse)
    res["err_lse"] = float(np.max(np.abs(gl[fin] - ref_lse[fin]))) if fin.any() else 0.0
    res["dead"] = ~fin
    ok = bool(np.all(of[B * ob :] == seven))
    if alloc_q > Sq:
        ok &= bool(np.all(got_lg[:, :, Sq:, :] == seven))
    if pad:
        ok &= bool(np.all(oraw[:, n_o:] == seven))
    res["o_untouched"] = ok
    ok = bool(np.all(lse[n_lse:] == LSE_SENTINEL))
    ok &= bool(np.all(stats[2 * n_lse :] == LSE_SENTINEL))
    if alloc_q > Sq:
        ok &= bool(np.all(lse_bh[:, :, Sq:] == LSE_SENTINEL))
    res["lse_untouched"] = ok

    # epilogue statistics: running max (log2 domain) and running sum
    st = stats[: 2 * n_lse].reshape(B, Hq, alloc_q, 2)[:, :, :Sq, :]
    s_ref = np.einsum(
        "bhqd,bhkd->bhqk",
        q_in.astype(np.float64),
        np.repeat(k_in, group, axis=1).astype(np.float64),
    )
    s_ref *= scale
    if bias is not None:
        s_ref = s_ref + bias
    band = _band_mask(Sq, Sk, top_left, left_arg, right_arg)
    s2 = np.where(band[None, None], s_ref, -np.inf) * math.log2(math.e)
    live = np.isfinite(s2.max(axis=-1))
    m_ref = np.where(live, s2.max(axis=-1), 0.0)
    l_ref = np.where(live, np.exp2(s2 - m_ref[..., None]).sum(axis=-1), 1.0)
    res["err_m"] = (
        float(np.max(np.abs(st[..., 0][live] - m_ref[live]))) if live.any() else 0.0
    )
    res["err_l"] = (
        float(np.max(np.abs(st[..., 1][live] / l_ref[live] - 1.0)))
        if live.any()
        else 0.0
    )
    return res
