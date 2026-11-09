# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-device numerics of the RDNA split-KV WMMA decode (kernels/gfx1151/wmma_fmha_decode).

Every case compiles the decode kernel and its merge companion through
:func:`build_wmma_fmha_decode_module` into ONE code object, finds both symbols in
the loaded module, and checks O and the natural-log LSE of every row against a
float64 reference in each split mode: one split (no workspace), the fused merge
(the last split merges, ``WS_CNT`` counters; launched twice on the same
workspace to check the counters re-arm), the merge kernel, and the host policy's
own split count. The split partials (``WS_O`` / ``WS_LSE``) are NaN-poisoned
before every launch, as hipDNN's workspace is never cleared: every split, empty
ones included, must write its partials. Coverage: GQA packing (Sq 1 / 4 / 8 and
Hq / Hkv up to 8), a V head count different from the K head count (both
directions), runtime head widths inside a bucket, buckets 64 / 128 / 256, causal
top-left / bottom-right, sliding and two-sided windows, rows without visible keys,
key ranges shorter than the split count, softcap (both tanh forms), sinks, ALiBi,
QQ-bias, FP8 (OCP E4M3FN) K/V, the additive f32 / Q-dtype bias with broadcast,
ragged sequences (with empty queries and keys, a trailing empty query) and paged
caches (pages larger and smaller than a key tile), and the ``num_waves`` values
only the tuned candidates select.
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
_RQ = (1, 0, 3, 1, 2, 1)  # ragged / paged query lengths (an empty sequence included)
_RK = (1000, 50, 0, 1, 129, 4097)  # key lengths off every tile / page boundary
_XSPLITS = (8, 9, 13, 16, 33, 64)  # merge-only split counts past one 8-split group
_WS_POISON = 0xFF  # byte fill of the split partials before every launch (f32 NaN)


def _case(B, Hq, Hkv, Hv, Sq, Sk, dq, dv, br=1, wl=-1, wr=0, **opt):
    return dict(
        B=B,
        Hq=Hq,
        Hkv=Hkv,
        Hv=Hv,
        Sq=Sq,
        Sk=Sk,
        dq=dq,
        dv=dv,
        br=br,
        wl=wl,
        wr=wr,
        **opt,
    )


CASES = {
    # GQA packing, buckets, Sq 1 / 4 / 8, masks.
    "gqa8_d64_sq1": _case(2, 64, 8, 8, 1, 1000, 64, 64, xsplits=_XSPLITS),
    "gqa4_d128_sq4": _case(1, 32, 8, 8, 4, 777, 128, 128, xsplits=_XSPLITS),
    "gqa8_d256_sq8": _case(1, 16, 2, 2, 8, 600, 256, 256, xsplits=_XSPLITS),
    "mha_d128_sq1": _case(2, 8, 8, 8, 1, 1500, 128, 128, xsplits=_XSPLITS),
    "none_d128_sq3": _case(1, 16, 4, 4, 3, 500, 128, 128, br=0, wr=-1),
    "causal_topleft_d64_sq8": _case(1, 8, 2, 2, 8, 300, 64, 64, br=0),
    "window_d128_sq1": _case(1, 16, 4, 4, 1, 2000, 128, 128, wl=100, xsplits=_XSPLITS),
    "band_topleft_d64_sq4": _case(1, 8, 4, 4, 4, 300, 64, 64, br=0, wl=30, wr=7),
    "fullmask_rows_d64": _case(1, 8, 2, 2, 8, 40, 64, 64, br=1, wl=2, wr=0),
    # V head count different from the K head count; runtime widths.
    "hv4_hkv8_d128_sq2": _case(1, 32, 8, 4, 2, 900, 128, 128),
    "hv16_hkv8_d64_sq1": _case(2, 32, 8, 16, 1, 700, 64, 64),
    "rtdim_q40_v56": _case(1, 16, 4, 4, 2, 400, 40, 56),
    "rtdim_q112_v128": _case(1, 8, 2, 2, 1, 333, 112, 128),
    "rtdim_q200_v136": _case(1, 8, 4, 4, 3, 450, 200, 136),
    # Rows without a visible key (bottom-right diagonal): the merge sees only empty
    # splits for rows 0-3 and must store -inf LSE and zero O for them.
    "diag_masked_d64": _case(1, 8, 2, 2, 8, 4, 64, 64, br=1),
    "diag_masked_sinks_d64": _case(1, 8, 2, 2, 8, 4, 64, 64, br=1, sinks=True),
    # Score features.
    "softcap_d128": _case(2, 32, 8, 8, 1, 1200, 128, 128, softcap=30.0),
    "softcap_window_d256_sq3": _case(
        1, 16, 8, 8, 3, 900, 256, 256, wl=400, softcap=50.0
    ),
    "sinks_window_d64": _case(
        1, 64, 8, 8, 1, 2000, 64, 64, wl=127, sinks=True, xsplits=_XSPLITS
    ),
    "sinks_none_d128_sq4": _case(
        1, 16, 4, 4, 4, 300, 128, 128, br=0, wr=-1, sinks=True
    ),
    "alibi_topleft_d128_sq5": _case(1, 16, 2, 2, 5, 700, 128, 128, br=0, alibi=True),
    "qq_bias_d128_sq4": _case(2, 8, 2, 2, 4, 700, 128, 128, qq=(3, 500)),
    "bias_f32_full_d64_sq4": _case(
        2, 8, 2, 2, 4, 333, 64, 64, bias=("f32", (1, 1, 1, 1))
    ),
    "bias_q_bcast_d256_sq2": _case(
        1, 8, 4, 4, 2, 300, 256, 256, bias=("q", (0, 1, 0, 1))
    ),
    # Q-dtype bias rows starting mid-dword (odd Sk), odd extents, and a base pointer
    # 2 bytes past a dword (bias_lead elements, NaN-poisoned). bias_tail makes every
    # row's last key dominant, so a dropped final element moves the output.
    "bias_q_odd_sk_full_d64_sq4": _case(
        2, 8, 2, 2, 4, 53, 64, 64, bias=("q", (1, 1, 1, 1)), bias_tail=True
    ),
    "bias_q_odd_lead_d128_sq3": _case(
        1,
        8,
        4,
        4,
        3,
        37,
        128,
        128,
        bias=("q", (1, 0, 1, 1)),
        bias_lead=1,
        bias_tail=True,
    ),
    "bias_q_bcast_odd_lead_d256_sq1": _case(
        2,
        8,
        2,
        2,
        1,
        45,
        256,
        256,
        br=0,
        wr=-1,
        bias=("q", (1, 1, 0, 1)),
        bias_lead=1,
        bias_tail=True,
    ),
    "fp8_d64": _case(2, 64, 8, 8, 1, 1500, 64, 64, fp8=True),
    "fp8_d128_sq2": _case(1, 32, 8, 8, 2, 1000, 128, 128, fp8=True, xsplits=_XSPLITS),
    "fp8_d256_sq2": _case(1, 16, 8, 8, 2, 600, 256, 256, fp8=True),
    # Packed layouts.
    "ragged_d128": _case(
        6, 32, 8, 8, 0, 0, 128, 128, layout="ragged", xsplits=_XSPLITS
    ),
    "ragged_window_alibi_qq_d64": _case(
        6, 16, 2, 2, 0, 0, 64, 64, wl=100, layout="ragged", alibi=True, qq=(2, 40)
    ),
    "paged16_d64": _case(
        6, 64, 8, 8, 0, 0, 64, 64, layout="paged", ps=16, xsplits=_XSPLITS
    ),
    "paged8_d128": _case(6, 32, 8, 8, 0, 0, 128, 128, layout="paged", ps=8),
    "paged64_d256": _case(6, 16, 8, 8, 0, 0, 256, 256, layout="paged", ps=64),
    "paged16_sinks_fp8_d64": _case(
        6, 64, 8, 8, 0, 0, 64, 64, layout="paged", ps=16, sinks=True, fp8=True
    ),
    "paged32_all_features_d64": _case(
        6,
        16,
        4,
        4,
        0,
        0,
        64,
        64,
        wl=300,
        layout="paged",
        ps=32,
        softcap=30.0,
        sinks=True,
        alibi=True,
        qq=(2, 60),
        fp8=True,
        bias=("q", (0, 1, 0, 1)),
    ),
    "ragged_bias_f32_sinks_d128": _case(
        6,
        16,
        4,
        4,
        0,
        0,
        128,
        128,
        layout="ragged",
        sinks=True,
        bias=("f32", (1, 0, 1, 1)),
    ),
    "ragged_bias_q_lead_d64": _case(
        6,
        16,
        4,
        4,
        0,
        0,
        64,
        64,
        layout="ragged",
        bias=("q", (1, 0, 1, 1)),
        bias_lead=1,
    ),
    # Key ranges shorter than the split count: most splits see no key and must still
    # write their (empty) partials over the poisoned workspace.
    "empty_splits_sk17_d128": _case(2, 16, 4, 4, 1, 17, 128, 128, xsplits=_XSPLITS),
    "empty_splits_window3_d64": _case(1, 16, 2, 2, 1, 5000, 64, 64, wl=3),
    "empty_splits_sinks_window_d64": _case(
        1, 16, 4, 4, 2, 3000, 64, 64, wl=5, sinks=True
    ),
    "empty_splits_paged8_window_d256": _case(
        6, 16, 8, 8, 0, 0, 256, 256, wl=7, layout="paged", ps=8
    ),
    # The last sequence has no query rows (its first token is one past the end of Q).
    "ragged_trailing_empty_d64": _case(
        3, 8, 2, 2, 0, 0, 64, 64, layout="ragged", lens=((2, 1, 0), (100, 300, 50))
    ),
    # num_waves values only the tuned candidates select (8 fits D64 on both families).
    "knob_nw1_d128_sq4": _case(1, 32, 8, 8, 4, 777, 128, 128, knobs={"num_waves": 1}),
    "knob_nw2_window_d128_sq1": _case(
        1, 16, 4, 4, 1, 2000, 128, 128, wl=100, knobs={"num_waves": 2}
    ),
    "knob_nw8_d64_sq1": _case(2, 64, 8, 8, 1, 1000, 64, 64, knobs={"num_waves": 8}),
}


def _host_bytes(array):
    return (ctypes.c_ubyte * array.nbytes).from_buffer(array.reshape(-1).view(np.uint8))


def _strides(a):
    return [s // a.itemsize for s in a.strides]


class _Inputs:
    """Host inputs, per-sequence views and the float64 reference of one case."""

    def __init__(self, c, dtype):
        import ml_dtypes

        self.c = c
        self.npdt = ml_dtypes.bfloat16 if dtype == "bf16" else np.float16
        rng = np.random.default_rng(sum(map(ord, str(sorted(c.items())))) + len(dtype))
        layout = c.get("layout", "dense")
        B, Hq, Hk, Hv, dq, dv = c["B"], c["Hq"], c["Hkv"], c["Hv"], c["dq"], c["dv"]
        if layout == "dense":
            self.q_lens, self.k_lens = [c["Sq"]] * B, [c["Sk"]] * B
        else:
            self.q_lens, self.k_lens = (list(x) for x in c.get("lens", (_RQ, _RK)))
        self.sq_max = max(max(self.q_lens), 1)
        self.sk_max = max(self.k_lens)
        tq, tk = sum(self.q_lens), sum(self.k_lens)
        self.fp8 = bool(c.get("fp8"))
        self.k_scale, self.v_scale = (0.37, 0.61) if self.fp8 else (1.0, 1.0)

        def kv(shape, scale):
            x = rng.standard_normal(shape)
            if self.fp8:
                return np.ascontiguousarray((x / scale).astype(ml_dtypes.float8_e4m3fn))
            return np.ascontiguousarray(x.astype(self.npdt))

        if layout == "dense":
            self.q = rng.standard_normal((B, self.sq_max, Hq, dq)).astype(self.npdt)
            self.k = kv((B, self.sk_max, Hk, dq), self.k_scale)
            self.v = kv((B, self.sk_max, Hv, dv), self.v_scale)
            self.o = np.zeros((B, self.sq_max, Hq, dv), self.npdt)
            self.lse = np.zeros((B, Hq, self.sq_max), np.float32)
        else:
            self.q = rng.standard_normal((tq, Hq, dq)).astype(self.npdt)
            self.o = np.zeros((tq, Hq, dv), self.npdt)
            self.lse = np.zeros((Hq, tq), np.float32)
            self.cu_q = np.concatenate([[0], np.cumsum(self.q_lens)]).astype(np.int32)
            if layout == "ragged":
                self.k = kv((tk, Hk, dq), self.k_scale)
                self.v = kv((tk, Hv, dv), self.v_scale)
                self.cu_k = np.concatenate([[0], np.cumsum(self.k_lens)]).astype(
                    np.int32
                )
            else:
                ps = c["ps"]
                nblk = [-(-n // ps) for n in self.k_lens]
                pool = sum(nblk) + 5
                perm = rng.permutation(pool).astype(np.int32)
                self.block_table = np.zeros((B, max(max(nblk), 1)), np.int32)
                cur = 0
                for i, n in enumerate(nblk):
                    self.block_table[i, :n] = perm[cur : cur + n]
                    cur += n
                self.k = kv((pool, ps, Hk, dq), self.k_scale)
                self.v = kv((pool, ps, Hv, dv), self.v_scale)
                self.seqused_k = np.asarray(self.k_lens, np.int32)
        self.sinks = (
            (rng.standard_normal(Hq) * 2.0).astype(self.npdt)
            if c.get("sinks")
            else None
        )
        self.slopes = (
            (2.0 ** (-8.0 * (np.arange(Hq) + 1) / Hq)).astype(np.float32)
            if c.get("alibi")
            else None
        )
        self.qq = None
        if c.get("qq"):
            self.qq = (rng.standard_normal(c["qq"]) * 0.5).astype(np.float32)
        self.bias = None
        if c.get("bias"):
            kind, flags = c["bias"]
            shape = tuple(
                n if f else 1 for f, n in zip(flags, (B, Hq, self.sq_max, self.sk_max))
            )
            self.bias = np.ascontiguousarray(
                rng.standard_normal(shape).astype(
                    np.float32 if kind == "f32" else self.npdt
                )
            )
            if c.get("bias_tail"):
                self.bias[..., -1] = 4.0  # every row prefers its last key
        self.scale = c["dq"] ** -0.5

    def seq(self, i):
        """(q [sq,Hq,dq], k [sk,Hk,dq], v [sk,Hv,dv]) of sequence i, as float64."""
        layout = self.c.get("layout", "dense")
        f64 = lambda x: x.astype(np.float64)  # noqa: E731
        sq, sk = self.q_lens[i], self.k_lens[i]
        if layout == "dense":
            q, k, v = self.q[i, :sq], self.k[i, :sk], self.v[i, :sk]
        else:
            q = self.q[self.cu_q[i] : self.cu_q[i + 1]]
            if layout == "ragged":
                k = self.k[self.cu_k[i] : self.cu_k[i + 1]]
                v = self.v[self.cu_k[i] : self.cu_k[i + 1]]
            else:
                ps = self.c["ps"]
                pages = self.block_table[i, : -(-sk // ps)]
                k = self.k[pages].reshape(-1, *self.k.shape[2:])[:sk]
                v = self.v[pages].reshape(-1, *self.v.shape[2:])[:sk]
        return f64(q), f64(k) * self.k_scale, f64(v) * self.v_scale

    def reference(self):
        """O (like self.o) and natural-log LSE (like self.lse) in float64."""
        c = self.c
        Hq, Hk, Hv = c["Hq"], c["Hkv"], c["Hv"]
        out = np.zeros(self.o.shape)
        lse = np.full(self.lse.shape, -np.inf)
        dense = c.get("layout", "dense") == "dense"
        for i in range(len(self.q_lens)):
            q, k, v = self.seq(i)
            sq, sk = q.shape[0], k.shape[0]
            if sq == 0:
                continue
            ctx = sk - sq if c["br"] else 0
            diag = np.arange(sq)[:, None] + ctx
            keys = np.arange(sk)[None, :]
            first = (
                np.maximum(diag - c["wl"], 0) if c["wl"] >= 0 else np.zeros_like(diag)
            )
            end = (
                np.minimum(diag + c["wr"] + 1, sk)
                if c["wr"] >= 0
                else np.full_like(diag, sk)
            )
            keep = (keys >= first) & (keys < end)
            for h in range(Hq):
                s = q[:, h] @ k[:, h // (Hq // Hk)].T * self.scale
                if c.get("softcap"):
                    s = c["softcap"] * np.tanh(s / c["softcap"])
                if self.slopes is not None:
                    s = s + float(self.slopes[h]) * (keys - ctx)
                if self.qq is not None:
                    rows, cols = self.qq.shape
                    rel = np.arange(sk) - ctx
                    ok = (rel >= 0) & (rel < cols)
                    for r in range(min(sq, rows)):
                        s[r, ok] += self.qq[r, rel[ok]]
                if self.bias is not None:
                    bb = self.bias[
                        min(i, self.bias.shape[0] - 1), min(h, self.bias.shape[1] - 1)
                    ]
                    s = s + bb[: sq if bb.shape[0] > 1 else 1, :sk].astype(np.float64)
                s = np.where(keep, s, -np.inf)
                m = s.max(axis=1) if sk else np.full(sq, -np.inf)
                top = np.where(np.isneginf(m), 0.0, m)
                sink = None if self.sinks is None else float(self.sinks[h])
                if sink is not None:
                    top = np.maximum(top, sink)
                p = np.exp(s - top[:, None]) if sk else np.zeros((sq, 0))
                den = p.sum(axis=1) + (np.exp(sink - top) if sink is not None else 0.0)
                o = (p @ v[:, h // (Hq // Hv)]) / np.where(den > 0, den, 1.0)[:, None]
                ls = np.where(
                    den > 0, top + np.log(np.where(den > 0, den, 1.0)), -np.inf
                )
                if dense:
                    out[i, :sq, h] = o
                    lse[i, h, :sq] = ls
                else:
                    out[self.cu_q[i] : self.cu_q[i + 1], h] = o
                    lse[h, self.cu_q[i] : self.cu_q[i + 1]] = ls
        return out, lse


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_decode_matches_reference(case, dtype):
    pytest.importorskip("ml_dtypes")
    from kernels.gfx1151 import wmma_fmha_decode as dec
    from rocke.helpers import compile_module
    from rocke.runtime.hip_module import Runtime, get_device_num_cus
    from rocke.runtime.packing import pack_args

    c = CASES[case]
    inp = _Inputs(c, dtype)
    layout = c.get("layout", "dense")
    spec = dec.WmmaFmhaDecodeSpec(
        head_size=dec.wmma_fmha_decode_bucket(head_dim_q=c["dq"], head_dim_v=c["dv"]),
        dtype=dtype,
        bias=c["bias"][0] if c.get("bias") else "",
        softcap=bool(c.get("softcap")),
        sinks=bool(c.get("sinks")),
        alibi=bool(c.get("alibi")),
        qq_bias=bool(c.get("qq")),
        kv_dtype="fp8e4m3" if inp.fp8 else "",
        layout=layout,
        page_block_size=c.get("ps", 0),
        **c.get("knobs", {}),
    )
    rspec = dec.resolve_wmma_fmha_decode_spec(spec, arch=_ARCH)
    ref_o, ref_lse = inp.reference()

    B, Hq, Hk, Hv = c["B"], c["Hq"], c["Hkv"], c["Hv"]
    heads = dict(num_query_heads=Hq, num_kv_heads=Hk, num_v_heads=Hv)
    groups = dec.wmma_fmha_decode_pack_groups(**heads)
    row_tiles = dec.wmma_fmha_decode_row_tiles(seqlen_q=inp.sq_max, **heads)
    lo, hi = dec.wmma_fmha_decode_kv_range(
        seqlen_q=inp.sq_max,
        seqlen_k=inp.sk_max,
        bottom_right=bool(c["br"]),
        window_left=c["wl"],
        window_right=c["wr"],
    )
    auto = dec.wmma_fmha_decode_num_splits(
        rspec,
        batch=B,
        pack_groups=groups,
        row_tiles=row_tiles,
        kv_len=hi - lo,
        num_cus=get_device_num_cus() or 1,
    )
    modes = [
        (1, False),
        (3, True),
        (3, True),
        (7, False),
        (auto, dec.wmma_fmha_decode_fused_merge(auto)),
    ]
    # Merge kernel beyond its first 8-split group (running-max carry, rescale of l and
    # acc, partial tail group): the auto split count never gets there.
    modes += [(n, False) for n in c.get("xsplits", ())]

    rt = Runtime()
    bufs = {}
    module = None

    def upload(name, arr):
        arr = np.ascontiguousarray(arr)
        bufs[name] = rt.alloc(max(arr.nbytes, 4))
        if arr.nbytes:
            rt.memcpy_h2d(bufs[name], _host_bytes(arr), arr.nbytes)
        return bufs[name]

    try:
        artifact = compile_module(
            dec.build_wmma_fmha_decode_module(spec, arch=_ARCH),
            arch=_ARCH,
            backend="python",
        )
        module = rt.load_module(artifact.hsaco)  # one code object, both kernels
        f_dec = module.get_function(artifact.kernel_name)
        f_merge = module.get_function(artifact.companion_names["merge"])
        values = dict(
            Q=upload("q", inp.q), K=upload("k", inp.k), V=upload("v", inp.v), O=upload("o", inp.o),
            LSE=upload("lse", inp.lse), scale_log2=inp.scale * _LOG2E, seqlen_q=inp.sq_max, seqlen_k=inp.sk_max,
            head_dim_q=c["dq"], head_dim_v=c["dv"], bottom_right=c["br"], window_left=c["wl"],
            window_right=c["wr"], write_lse=1, **heads,
        )  # fmt: skip
        if layout == "dense":
            for n, a in (("q", inp.q), ("k", inp.k), ("v", inp.v), ("o", inp.o)):
                st = _strides(a)
                values.update(
                    {
                        f"stride_{n}_batch": st[0],
                        f"stride_{n}_token": st[1],
                        f"stride_{n}_head": st[2],
                    }
                )
            st = _strides(inp.lse)
            values.update(
                stride_lse_batch=st[0], stride_lse_head=st[1], stride_lse_token=st[2]
            )
        else:
            for n, a in (("q", inp.q), ("o", inp.o)) + (
                (("k", inp.k), ("v", inp.v)) if layout == "ragged" else ()
            ):
                st = _strides(a)
                values.update(
                    {
                        f"stride_{n}_batch": 0,
                        f"stride_{n}_token": st[0],
                        f"stride_{n}_head": st[1],
                    }
                )
            st = _strides(inp.lse)
            values.update(
                stride_lse_batch=0, stride_lse_head=st[0], stride_lse_token=st[1]
            )
            values["cu_seqlens_q"] = upload("cu_q", inp.cu_q)
            if layout == "ragged":
                values["cu_seqlens_k"] = upload("cu_k", inp.cu_k)
            else:
                for n, a in (("k", inp.k), ("v", inp.v)):
                    st = _strides(a)
                    values.update(
                        {
                            f"stride_{n}_batch": 0,
                            f"stride_{n}_token": st[1],
                            f"stride_{n}_head": st[2],
                            f"stride_{n}_block": st[0],
                        }
                    )
                values.update(
                    seqused_k=upload("seqused", inp.seqused_k),
                    block_table=upload("bt", inp.block_table),
                    block_table_stride=inp.block_table.shape[1],
                    num_blocks=inp.k.shape[0],
                )
        if spec.softcap:
            values["softcap"] = c["softcap"]
        if spec.sinks:
            values["sink_ptr"] = upload("sinks", inp.sinks)
        if spec.alibi:
            values["alibi_slopes_ptr"] = upload("slopes", inp.slopes)
        if spec.qq_bias:
            values.update(
                qq_bias_ptr=upload("qq", inp.qq),
                qq_bias_rows=inp.qq.shape[0],
                qq_bias_cols=inp.qq.shape[1],
                qq_bias_stride=inp.qq.shape[1],
            )
        if inp.fp8:
            values.update(k_scale=inp.k_scale, v_scale=inp.v_scale)
        if spec.bias:
            st = _strides(inp.bias)
            lead = c.get("bias_lead", 0)
            flat = np.concatenate(
                [np.full(lead, np.nan, inp.bias.dtype), inp.bias.reshape(-1)]
            )
            values.update(
                attn_bias_ptr=upload("bias", flat) + lead * inp.bias.itemsize,
                **{
                    f"bias_stride_{a}": (st[i] if inp.bias.shape[i] > 1 else 0)
                    for i, a in enumerate("bhq")
                },
            )
        mgrid = dec.wmma_fmha_decode_merge_grid(
            batch=B, num_query_heads=Hq, seqlen_q=inp.sq_max
        )
        ws_cache = {}
        for ns, fused in modes:
            lay = dec.wmma_fmha_decode_workspace(
                rspec,
                batch=B,
                seqlen_q=inp.sq_max,
                head_dim_v=c["dv"],
                num_splits=ns,
                **heads,
            )
            ws = 0
            if lay["bytes"]:
                # Allocated (counters zeroed) once per split count: the fused merge
                # re-arms its counters across launches.
                if ns not in ws_cache:
                    ws_cache[ns] = rt.alloc(lay["bytes"])
                    rt.memset(ws_cache[ns], 0, lay["bytes"])
                ws = ws_cache[ns]
                # WS_O and WS_LSE precede WS_CNT; poison them before every launch.
                rt.memset(ws, _WS_POISON, lay["ws_cnt"])
            rt.memset(bufs["o"], _WS_POISON, inp.o.nbytes)  # every element is written
            rt.memset(bufs["lse"], _WS_POISON, inp.lse.nbytes)
            values.update(
                WS_O=ws + lay["ws_o"] if ws else 0,
                WS_LSE=ws + lay["ws_lse"] if ws else 0,
                WS_CNT=ws + lay["ws_cnt"] if (ws and fused) else 0,
                num_splits=ns,
            )
            grid = dec.wmma_fmha_decode_grid(
                rspec, batch=B, pack_groups=groups, row_tiles=row_tiles, num_splits=ns
            )
            rt.launch(
                f_dec,
                grid,
                (rspec.block_size, 1, 1),
                pack_args(dec.wmma_fmha_decode_signature(spec), values),
            )
            if ns > 1 and not fused:
                mvals = {
                    k: values[k]
                    for k in values
                    if k.startswith(("stride_o_", "stride_lse_"))
                }
                mvals.update(
                    O=values["O"],
                    LSE=values["LSE"],
                    WS_O=values["WS_O"],
                    WS_LSE=values["WS_LSE"],
                    seqlen_q=inp.sq_max,
                    num_query_heads=Hq,
                    head_dim_v=c["dv"],
                    num_splits=ns,
                    write_lse=1,
                )
                if spec.packed:
                    mvals["cu_seqlens_q"] = values["cu_seqlens_q"]
                rt.launch(
                    f_merge,
                    mgrid,
                    (dec.MERGE_THREADS, 1, 1),
                    pack_args(dec.wmma_fmha_decode_merge_signature(spec), mvals),
                )
            rt.sync()
            got_o = np.empty_like(inp.o)
            got_lse = np.empty_like(inp.lse)
            rt.memcpy_d2h(_host_bytes(got_o), bufs["o"], got_o.nbytes)
            rt.memcpy_d2h(_host_bytes(got_lse), bufs["lse"], got_lse.nbytes)
            where = f"splits={ns} fused={fused}"
            got = got_o.astype(np.float64)
            assert np.isfinite(got).all(), where
            # The repository's documented per-dtype output bound for decode
            # (benchmarks/gfx1151/attention/cases.py): 2e-2 fp16, 4e-2 bf16.
            tol = 4e-2 if dtype == "bf16" else 2e-2
            np.testing.assert_allclose(got, ref_o, rtol=0, atol=tol, err_msg=where)
            fin = np.isfinite(ref_lse)
            assert np.array_equal(np.isfinite(got_lse), fin), where
            # A row without a visible key must store exactly -inf (not NaN, not left
            # at the poison).
            assert np.array_equal(np.isneginf(got_lse), np.isneginf(ref_lse)), where
            if fin.any():
                lse_tol = 1e-2 + 1e-5 * float(np.abs(ref_lse[fin]).max())
                np.testing.assert_allclose(
                    got_lse[fin], ref_lse[fin], rtol=0, atol=lse_tol, err_msg=where
                )
        for ptr in ws_cache.values():
            rt.free(ptr)
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        for ptr in bufs.values():
            rt.free(ptr)


@pytest.mark.gpu
@_NEEDS_RDNA
def test_module_is_one_code_object_with_both_symbols():
    from kernels.gfx1151 import wmma_fmha_decode as dec
    from rocke.helpers import compile_module
    from rocke.runtime.hip_module import Runtime

    spec = dec.WmmaFmhaDecodeSpec(head_size=128, dtype="bf16", layout="ragged")
    artifact = compile_module(
        dec.build_wmma_fmha_decode_module(spec, arch=_ARCH),
        arch=_ARCH,
        backend="python",
    )
    assert artifact.companion_names == {
        "merge": "rocke_wmma_fmha_decode_merge_bf16_packed"
    }
    assert (
        artifact.kernel_name
        == dec.resolve_wmma_fmha_decode_spec(spec, arch=_ARCH).kernel_name()
    )
    assert artifact.hsaco.count(artifact.kernel_name.encode()) >= 1
    assert artifact.hsaco.count(b"rocke_wmma_fmha_decode_merge_bf16_packed") >= 1
    rt = Runtime()
    module = rt.load_module(artifact.hsaco)
    try:
        assert module.get_function(artifact.kernel_name)
        assert module.get_function("rocke_wmma_fmha_decode_merge_bf16_packed")
    finally:
        module.unload()
