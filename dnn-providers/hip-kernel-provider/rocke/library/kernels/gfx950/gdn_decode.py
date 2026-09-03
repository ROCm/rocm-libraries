# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Gated DeltaNet (GDN) single-token decode kernel instance builder.

For each active sequence and value head, advance one linear-attention decode
step over a fixed-size recurrent state ``S`` (a ``head_v_dim x head_k_dim``
key->value matrix), per the gated delta rule:

    q_hat = l2norm(q) * head_k_dim**-0.5      # in-kernel, per key head
    k_hat = l2norm(k)
    decay = exp(-exp(A_log[vh]) * softplus(a[vh] + dt_bias[vh]))   # per value head
    beta  = sigmoid(b[vh])
    S     = decay * S                          # gated forget
    v_new = (v - S @ k_hat) * beta             # error-correcting delta value
    out   = S @ q_hat + v_new * dot(k_hat, q_hat)
    S     = S + outer(v_new, k_hat)            # rank-1 write

Only pages named by ``read_indices`` / ``write_indices`` are touched; negative
sentinel entries skip the block (continuous-batching padding lanes). Tensors are
assumed contiguous (row-major), matching the packed linear-attention decode
contract:

    query, key   : [B, 1, num_k_heads, head_k_dim]   dtype
    value, out   : [B, 1, num_v_heads, head_v_dim]   dtype
    a, b         : [B, 1, num_v_heads]               dtype
    dt_bias      : [num_v_heads]                      dtype
    A_log        : [num_v_heads]                      f32
    read/write_indices : [B]                          i32
    state        : [pool, num_v_heads, head_v_dim, head_k_dim]  state_dtype

**v1 mapping (correctness-first):** one workgroup per ``(sequence, value_head)``;
``block_size = head_v_dim`` threads; thread ``t`` owns state row ``t`` (the full
``head_k_dim``-wide key vector for value-dim ``t``) in registers, so every dot
product is thread-local and needs no cross-thread reduction. Q/K L2 norms and
``dot(k,q)`` are recomputed per thread (redundant but simple). This is
VGPR-heavy by design; the warp-tiled reduction is the first optimization pass,
not part of the correctness baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

from rocke.core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.io import (
    io_ir_type,
    load_scalar_as_f32,
    load_vec_as_f32,
    pack_f32_to,
    store_scalar_from_f32,
    store_vec,
)
from rocke.helpers.reduction import tree_reduce
from rocke.helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join

__all__ = [
    "GdnDecodeSpec",
    "is_valid_spec",
    "build_gdn_decode",
    "gdn_decode_grid",
    "gdn_decode_signature",
]

DType = Literal["f16", "bf16"]

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
NORM_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0
STATE_VEC = 8  # 16B bf16 vector load/store width


@dataclass(frozen=True)
class GdnDecodeSpec:
    """One GDN single-token decode instance."""

    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    dtype: DType = "bf16"
    state_dtype: DType = "bf16"
    use_qk_l2norm: bool = True
    wave_size: int = 64
    # Known-good fallback for direct callers. Production dispatch replaces
    # these values with a batch-tuned tile; callers that construct the spec
    # directly still get a valid general-purpose configuration.
    num_warps: int = 2
    warp_threads_k: int = 16
    blocks_per_v_dim: int = (
        8  # split a head's V-dim across this many CTAs (small-B fill)
    )
    simple: bool = False  # True => v1 one-thread-per-row reference path
    name: str = "rocke_gdn_decode"

    @property
    def block_size(self) -> int:
        return self.head_v_dim if self.simple else self.num_warps * self.wave_size

    @property
    def v_per_k_head(self) -> int:
        return self.num_v_heads // self.num_k_heads

    def kernel_name(self) -> str:
        # Every field that changes emitted code MUST appear here: this name is the
        # compile/launcher cache key, so two specs sharing a name means one of them
        # silently runs the other's kernel. Fields whose value equals the default
        # are folded into the deviation-only suffixes below to keep names stable.
        parts = (
            self.dtype,
            f"kh{self.num_k_heads}",
            f"vh{self.num_v_heads}",
            f"dk{self.head_k_dim}",
            f"dv{self.head_v_dim}",
            f"w{self.num_warps}k{self.warp_threads_k}b{self.blocks_per_v_dim}",
        )
        if self.state_dtype != self.dtype:
            parts += (f"st{self.state_dtype}",)
        if self.wave_size != 64:
            parts += (f"ws{self.wave_size}",)
        return kernel_name_join(
            self.name,
            *parts,
            flags={"l2": self.use_qk_l2norm, "s": self.simple},
        )


def is_valid_spec(spec: GdnDecodeSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Reject impossible/unsupported GDN decode configs before IR is built."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.dtype not in ("f16", "bf16") or spec.state_dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype}/{spec.state_dtype}"
    if spec.num_v_heads % spec.num_k_heads:
        return False, "num_v_heads must be divisible by num_k_heads"
    if spec.head_k_dim % STATE_VEC or spec.head_v_dim % STATE_VEC:
        return False, f"head dims must be multiples of {STATE_VEC}"
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )
    if not spec.simple:
        if spec.wave_size % spec.warp_threads_k:
            return False, "wave_size must be divisible by warp_threads_k"
        vpt = STATE_VEC
        warp_tile_k = spec.warp_threads_k * vpt
        wgroup_v = spec.num_warps * (spec.wave_size // spec.warp_threads_k)
        if spec.head_k_dim % warp_tile_k:
            return False, f"head_k_dim must be a multiple of {warp_tile_k}"
        if spec.head_v_dim % spec.blocks_per_v_dim:
            return False, "head_v_dim must be a multiple of blocks_per_v_dim"
        tile_v = spec.head_v_dim // spec.blocks_per_v_dim
        if tile_v % wgroup_v:
            return (
                False,
                f"head_v_dim/blocks_per_v_dim must be a multiple of {wgroup_v}",
            )
    return True, ""


def build_gdn_decode(spec: GdnDecodeSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one GDN single-token decode instance (dispatch)."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid gdn_decode spec for {arch}: {why}")
    return _build_simple(spec) if spec.simple else _build_warp_tiled(spec)


def _build_simple(spec: GdnDecodeSpec) -> KernelDef:
    """v1 reference: one thread per value-dim (state row); no cross-thread reduce."""

    HK, HV = spec.num_k_heads, spec.num_v_heads
    DK, DV = spec.head_k_dim, spec.head_v_dim
    G = spec.v_per_k_head
    BS = spec.block_size
    scale = 1.0 / math.sqrt(DK)

    # Contiguous row-major strides (element counts).
    Q_HN, Q_HK = HK * DK, DK  # query/key: [B,1,HK,DK]
    V_HN, V_HK = HV * DV, DV  # value/out: [B,1,HV,DV]
    S_POOL, S_HV, S_VR = HV * DV * DK, DV * DK, DK  # state: [pool,HV,DV,DK]

    io_ty = io_ir_type(spec.dtype)
    st_ty = io_ir_type(spec.state_dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    Q = b.param(
        "query", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    K = b.param("key", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Vv = b.param(
        "value", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Ag = b.param("a", PtrType(io_ty, "global"), noalias=True, readonly=True)
    Bg = b.param("b", PtrType(io_ty, "global"), noalias=True, readonly=True)
    DTB = b.param("dt_bias", PtrType(io_ty, "global"), noalias=True, readonly=True)
    ALOG = b.param("A_log", PtrType(F32, "global"), noalias=True, readonly=True)
    RIDX = b.param("read_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    WIDX = b.param("write_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    STATE = b.param("state", PtrType(st_ty, "global"), noalias=True, align=16)
    OUT = b.param(
        "out", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
    )
    _ = b.param("batch_size", I32)  # noqa: F841

    tid = b.thread_id_x()  # value-dim row this thread owns (0..DV-1)
    bidx = b.block_id_x()
    b_i = b.div(bidx, b.const_i32(HV))
    hv_i = b.mod(bidx, b.const_i32(HV))
    hk_i = b.div(hv_i, b.const_i32(G))

    read_pool = b.global_load_i32(RIDX, b_i)
    write_pool = b.global_load_i32(WIDX, b_i)

    # Skip continuous-batching padding lanes (negative sentinel index).
    active = b.land(
        b.cmp_ge(read_pool, b.const_i32(0)), b.cmp_ge(write_pool, b.const_i32(0))
    )
    with b.scf_if(active):
        # ---- exp/sigmoid/softplus composed from exp2/log2/rcp (no native) ----
        def exp_f32(x):
            return b.exp2_fast(b.fmul(x, b.const_f32(LOG2E)))

        def log1p_f32(x):
            return b.fmul(b.log2(b.fadd(b.const_f32(1.0), x)), b.const_f32(LN2))

        # ---- load q,k rows [DK] -> f32 registers ----
        q_base = b.add(b.mul(b_i, b.const_i32(Q_HN)), b.mul(hk_i, b.const_i32(Q_HK)))
        qv, kv = [], []
        for c in range(0, DK, STATE_VEC):
            off = b.add(q_base, b.const_i32(c))
            qv += load_vec_as_f32(b, Q, off, dtype=spec.dtype, n=STATE_VEC)
            kv += load_vec_as_f32(b, K, off, dtype=spec.dtype, n=STATE_VEC)

        # ---- L2 normalize q (and *scale), k ----
        if spec.use_qk_l2norm:
            sum_q2 = tree_reduce(b, b.fadd, [b.fmul(x, x) for x in qv])
            sum_k2 = tree_reduce(b, b.fadd, [b.fmul(x, x) for x in kv])
            inv_q = b.rsqrt(b.fadd(sum_q2, b.const_f32(NORM_EPS)))
            inv_k = b.rsqrt(b.fadd(sum_k2, b.const_f32(NORM_EPS)))
            sq = b.fmul(inv_q, b.const_f32(scale))
            qn = [b.fmul(x, sq) for x in qv]
            kn = [b.fmul(x, inv_k) for x in kv]
        else:
            qn = [b.fmul(x, b.const_f32(scale)) for x in qv]
            kn = kv

        # ---- gates (per value head) ----
        a_idx = b.add(b.mul(b_i, b.const_i32(HV)), hv_i)
        ra = load_scalar_as_f32(b, Ag, a_idx, dtype=spec.dtype)
        rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
        rdt = load_scalar_as_f32(b, DTB, hv_i, dtype=spec.dtype)
        ral = b.global_load_f32(ALOG, hv_i)  # A_log is fp32

        x = b.fadd(ra, rdt)
        sp = b.select(
            b.fcmp("ogt", x, b.const_f32(SOFTPLUS_THRESHOLD)),
            x,
            log1p_f32(exp_f32(x)),
        )
        decay = exp_f32(b.fneg(b.fmul(exp_f32(ral), sp)))
        beta = b.rcp_fast(b.fadd(b.const_f32(1.0), exp_f32(b.fneg(rb))))

        # ---- dot(k_hat, q_hat) (scalar, redundant per thread) ----
        dot_kq = tree_reduce(b, b.fadd, [b.fmul(kn[j], qn[j]) for j in range(DK)])

        # ---- load state row t=tid: state[read_pool, hv, tid, 0:DK] ----
        rs_base = b.add(
            b.add(
                b.mul(read_pool, b.const_i32(S_POOL)),
                b.mul(hv_i, b.const_i32(S_HV)),
            ),
            b.mul(tid, b.const_i32(S_VR)),
        )
        sv = []
        for c in range(0, DK, STATE_VEC):
            off = b.add(rs_base, b.const_i32(c))
            sv += load_vec_as_f32(b, STATE, off, dtype=spec.state_dtype, n=STATE_VEC)
        sv = [b.fmul(s, decay) for s in sv]  # gated forget

        # ---- S_row . k_hat  and  S_row . q_hat ----
        sum_hk = tree_reduce(b, b.fadd, [b.fmul(sv[j], kn[j]) for j in range(DK)])
        sum_hq = tree_reduce(b, b.fadd, [b.fmul(sv[j], qn[j]) for j in range(DK)])

        # ---- delta value + read-out for this value-dim ----
        v_idx = b.add(
            b.add(b.mul(b_i, b.const_i32(V_HN)), b.mul(hv_i, b.const_i32(V_HK))), tid
        )
        rv = load_scalar_as_f32(b, Vv, v_idx, dtype=spec.dtype)
        v_new = b.fmul(b.fsub(rv, sum_hk), beta)
        out_val = b.fadd(sum_hq, b.fmul(v_new, dot_kq))
        store_scalar_from_f32(b, OUT, v_idx, out_val, dtype=spec.dtype)

        # ---- rank-1 state write: S_row += k_hat * v_new ----
        ws_base = b.add(
            b.add(
                b.mul(write_pool, b.const_i32(S_POOL)),
                b.mul(hv_i, b.const_i32(S_HV)),
            ),
            b.mul(tid, b.const_i32(S_VR)),
        )
        new_s = [b.fma(kn[j], v_new, sv[j]) for j in range(DK)]
        for c in range(0, DK, STATE_VEC):
            vec = pack_f32_to(b, new_s[c : c + STATE_VEC], dtype=spec.state_dtype)
            store_vec(b, STATE, b.add(ws_base, b.const_i32(c)), vec, n=STATE_VEC)

    return b.kernel


def _build_warp_tiled(spec: GdnDecodeSpec) -> KernelDef:
    """v2: warp-tiled. One CTA per (seq, value-head); the
    ``head_v_dim x head_k_dim`` state is distributed across the block's warps
    (WTV x WTK lanes, VPT values/lane), and the K-reductions (L2 norms,
    dot(k,q), S.k, S.q) are folded once per WTK-lane group via shuffle-xor,
    eliminating v1's per-thread redundant compute and register pressure.
    """
    HK, HV = spec.num_k_heads, spec.num_v_heads
    DK, DV = spec.head_k_dim, spec.head_v_dim
    G = spec.v_per_k_head
    WAVE = spec.wave_size
    WTK = spec.warp_threads_k
    WTV = WAVE // WTK
    NW = spec.num_warps
    VPT = STATE_VEC
    BS = NW * WAVE
    WARP_TILE_K = WTK * VPT
    WTK_ITERS = DK // WARP_TILE_K
    WGROUP_V = NW * WTV
    BPV = spec.blocks_per_v_dim
    TILE_V = DV // BPV
    WTV_ITERS = TILE_V // WGROUP_V
    scale = 1.0 / math.sqrt(DK)
    shfl = [1 << i for i in range(WTK.bit_length() - 1)]  # [1,2,4] for WTK=8

    Q_HN, Q_HK = HK * DK, DK
    V_HN, V_HK = HV * DV, DV
    S_POOL, S_HV, S_VR = HV * DV * DK, DV * DK, DK

    io_ty = io_ir_type(spec.dtype)
    st_ty = io_ir_type(spec.state_dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    Q = b.param(
        "query", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    K = b.param("key", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Vv = b.param(
        "value", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Ag = b.param("a", PtrType(io_ty, "global"), noalias=True, readonly=True)
    Bg = b.param("b", PtrType(io_ty, "global"), noalias=True, readonly=True)
    DTB = b.param("dt_bias", PtrType(io_ty, "global"), noalias=True, readonly=True)
    ALOG = b.param("A_log", PtrType(F32, "global"), noalias=True, readonly=True)
    RIDX = b.param("read_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    WIDX = b.param("write_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    STATE = b.param("state", PtrType(st_ty, "global"), noalias=True, align=16)
    OUT = b.param(
        "out", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
    )
    _ = b.param("batch_size", I32)  # noqa: F841

    tid = b.thread_id_x()
    bidx = b.block_id_x()
    b_hv_i = b.div(bidx, b.const_i32(BPV))
    tile_v_start = b.mul(b.mod(bidx, b.const_i32(BPV)), b.const_i32(TILE_V))
    b_i = b.div(b_hv_i, b.const_i32(HV))
    hv_i = b.mod(b_hv_i, b.const_i32(HV))
    hk_i = b.div(hv_i, b.const_i32(G))
    w_tid = b.mod(tid, b.const_i32(WAVE))
    wid = b.div(tid, b.const_i32(WAVE))
    k_lane = b.mod(w_tid, b.const_i32(WTK))
    v_lane = b.div(w_tid, b.const_i32(WTK))
    warp_k_start = b.mul(k_lane, b.const_i32(VPT))
    gv_start = b.add(b.mul(wid, b.const_i32(WTV)), v_lane)
    k_lane0 = b.mul(v_lane, b.const_i32(WTK))  # first lane of this WTK group

    read_pool = b.global_load_i32(RIDX, b_i)
    write_pool = b.global_load_i32(WIDX, b_i)
    active = b.land(
        b.cmp_ge(read_pool, b.const_i32(0)), b.cmp_ge(write_pool, b.const_i32(0))
    )
    with b.scf_if(active):

        def exp_f32(x):
            return b.exp2_fast(b.fmul(x, b.const_f32(LOG2E)))

        def log1p_f32(x):
            return b.fmul(b.log2(b.fadd(b.const_f32(1.0), x)), b.const_f32(LN2))

        def wsum(v):  # xor-butterfly sum over the WTK-lane group (broadcast in-group)
            # xor 1/2 via quad_perm (VALU DPP: no LDS crossbar / no lgkmcnt(0) stall);
            # xor 4 crosses the 4-lane quad so it stays on ds_swizzle.
            for off in shfl:
                if off <= 2:
                    v = b.fadd(v, b.warp_shuffle_xor_quad(v, off))
                else:
                    v = b.fadd(v, b.warp_shuffle_xor(v, off))
            return v

        # gates (per value head)
        a_idx = b.add(b.mul(b_i, b.const_i32(HV)), hv_i)
        ra = load_scalar_as_f32(b, Ag, a_idx, dtype=spec.dtype)
        rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
        rdt = load_scalar_as_f32(b, DTB, hv_i, dtype=spec.dtype)
        ral = b.global_load_f32(ALOG, hv_i)
        x = b.fadd(ra, rdt)
        sp = b.select(
            b.fcmp("ogt", x, b.const_f32(SOFTPLUS_THRESHOLD)), x, log1p_f32(exp_f32(x))
        )
        decay = exp_f32(b.fneg(b.fmul(exp_f32(ral), sp)))
        beta = b.rcp_fast(b.fadd(b.const_f32(1.0), exp_f32(b.fneg(rb))))

        # load this lane's q,k K-chunks -> f32
        qk_base = b.add(b.mul(b_i, b.const_i32(Q_HN)), b.mul(hk_i, b.const_i32(Q_HK)))
        qn = [None] * WTK_ITERS
        kn = [None] * WTK_ITERS
        for ki in range(WTK_ITERS):
            off = b.add(qk_base, b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K)))
            qn[ki] = load_vec_as_f32(b, Q, off, dtype=spec.dtype, n=VPT)
            kn[ki] = load_vec_as_f32(b, K, off, dtype=spec.dtype, n=VPT)

        if spec.use_qk_l2norm:
            pq = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(qn[ki][i], qn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            pk = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(kn[ki][i], kn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            inv_q = b.rsqrt(b.fadd(pq, b.const_f32(NORM_EPS)))
            inv_k = b.rsqrt(b.fadd(pk, b.const_f32(NORM_EPS)))
            sq = b.fmul(inv_q, b.const_f32(scale))
            qn = [
                [b.fmul(qn[ki][i], sq) for i in range(VPT)] for ki in range(WTK_ITERS)
            ]
            kn = [
                [b.fmul(kn[ki][i], inv_k) for i in range(VPT)]
                for ki in range(WTK_ITERS)
            ]
        else:
            qn = [
                [b.fmul(qn[ki][i], b.const_f32(scale)) for i in range(VPT)]
                for ki in range(WTK_ITERS)
            ]

        dot_kq = wsum(
            tree_reduce(
                b,
                b.fadd,
                [
                    b.fmul(kn[ki][i], qn[ki][i])
                    for ki in range(WTK_ITERS)
                    for i in range(VPT)
                ],
            )
        )

        # load state tiles (decayed) into registers
        sv = {}
        for vi in range(WTV_ITERS):
            v_row = b.add(tile_v_start, b.add(gv_start, b.const_i32(vi * WGROUP_V)))
            rs_row = b.add(
                b.add(
                    b.mul(read_pool, b.const_i32(S_POOL)),
                    b.mul(hv_i, b.const_i32(S_HV)),
                ),
                b.mul(v_row, b.const_i32(S_VR)),
            )
            for ki in range(WTK_ITERS):
                off = b.add(rs_row, b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K)))
                vec = load_vec_as_f32(b, STATE, off, dtype=spec.state_dtype, n=VPT)
                sv[(vi, ki)] = [b.fmul(s, decay) for s in vec]

        for vi in range(WTV_ITERS):
            v_row = b.add(tile_v_start, b.add(gv_start, b.const_i32(vi * WGROUP_V)))
            phk = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(sv[(vi, ki)][i], kn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            phq = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(sv[(vi, ki)][i], qn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            v_idx = b.add(
                b.add(b.mul(b_i, b.const_i32(V_HN)), b.mul(hv_i, b.const_i32(V_HK))),
                v_row,
            )
            rv = load_scalar_as_f32(b, Vv, v_idx, dtype=spec.dtype)
            # v_new is in-group uniform (rv, broadcast phk, beta) - no bcast.
            v_new = b.fmul(b.fsub(rv, phk), beta)
            out_val = b.fadd(phq, b.fmul(v_new, dot_kq))
            with b.scf_if(b.cmp_eq(k_lane, b.const_i32(0))):
                store_scalar_from_f32(b, OUT, v_idx, out_val, dtype=spec.dtype)
            ws_row = b.add(
                b.add(
                    b.mul(write_pool, b.const_i32(S_POOL)),
                    b.mul(hv_i, b.const_i32(S_HV)),
                ),
                b.mul(v_row, b.const_i32(S_VR)),
            )
            for ki in range(WTK_ITERS):
                new = [b.fma(kn[ki][i], v_new, sv[(vi, ki)][i]) for i in range(VPT)]
                vec = pack_f32_to(b, new, dtype=spec.state_dtype)
                off = b.add(ws_row, b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K)))
                store_vec(b, STATE, off, vec, n=VPT)

    return b.kernel


def gdn_decode_grid(batch: int, spec: GdnDecodeSpec) -> Tuple[int, int, int]:
    """One workgroup per (sequence, value head, v-sub-block)."""
    bpv = 1 if spec.simple else spec.blocks_per_v_dim
    return ceil_div_grid((batch * spec.num_v_heads * bpv, 1))


def gdn_decode_signature(spec: GdnDecodeSpec):
    return (
        SignatureBuilder()
        .ptr("query", spec.dtype)
        .ptr("key", spec.dtype)
        .ptr("value", spec.dtype)
        .ptr("a", spec.dtype)
        .ptr("b", spec.dtype)
        .ptr("dt_bias", spec.dtype)
        .ptr("A_log", "f32")
        .ptr("read_indices", "i32")
        .ptr("write_indices", "i32")
        .ptr("state", spec.state_dtype)
        .ptr("out", spec.dtype)
        .scalar("batch_size", "i32")
        .build()
    )
