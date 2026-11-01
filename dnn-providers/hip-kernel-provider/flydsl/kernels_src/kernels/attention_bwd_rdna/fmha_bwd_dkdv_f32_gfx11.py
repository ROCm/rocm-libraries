# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); its sources are listed below.
#
# Provenance carried over from the sweep snapshot:
# Derived from: AITER aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py @ 8253efc4059516f492df597b80ce5311e9fda8bd
# via the attention_rdna gfx11 port: the runtime ABI, varlen / window / dropout / bias conventions are those of `fmha_bwd_dkdv_gfx11.py`; the algorithm is
# the fp32 VALU skeleton (AOTriton's fp32 `bwd_kernel_dk_dv`, MIT). compiler: flydsl 0.3.4

"""fp32 dK / dV backward attention for gfx11 (RDNA3 / 3.5, `gfx11-generic`): VALU FMAs, no WMMA (gfx11 has no fp32 WMMA).

The gfx950 runtime ABI of `fmha_bwd_dkdv_gfx950.py` (argument names, order and annotations of the launcher), exactly as the 16-bit gfx11 kernel; every
tensor is fp32 (Q, K, V, dO, dK, dV and the bias; LSE' and Delta from `fmha_bwd_prep_f32_gfx11`). The geometry, thread maps and the per-step algorithm are
documented in `fmha_tuning_bwd_dkdv_f32_gfx11.py` (one workgroup owns `block_kv` kv rows and loops the Q axis in `block_q`-row tiles; S, dP, P and dS through
LDS; owner accumulators in registers; fixed-order fp32 accumulation; `sm_scale` applied once to dK).

Scope: classes `plain`, `causal` and `bias`, each with baked dropout (`bk,1`) or without, exact head dims (hdim_qk = hdim_vo = tile) or padded
(`padded=1`: the runtime `hdim_qk` / `hdim_vo` <= tile), any sequence lengths (ragged Q and KV tiles), MHA / GQA and varlen addressing through the shared
decoders (`fmha.decode_addressing`: CUMULATIVE / INDIVIDUAL lengths (`seqused_k`), ARRAY positions, stacked layouts; LSE and Delta HT or TH). Other classes
and `dform=rt` raise `ValueError`. There is no dump object.

Padded head: a column at or past its runtime extent (Q, K and dK: `hdim_qk`; dO, V and dV: `hdim_vo`) is zero on every load (the address is clamped to
column 0 and the value selected away, so the caller's bytes are never read) and is never stored. Every store is one scalar, so a straddle needs no special
form: each element is its own predicated store.

Ragged tiles: a Q row or KV row past its sequence is a clamped duplicate of the sequence's last row; its P is selected to 0 after the exponential (never
multiplied), so it adds nothing; KV rows past the sequence are not stored.

Dropout (compile time): the gfx950 map (`philox.py`), one Philox call per (q, kv) pair of the thread (the kv column of a thread slot is fixed for the whole
kernel, so is its slot of each call). dV takes `keep * scale * P`, dS = P (keep * scale * dP - Delta) with the undropped P. A GQA group draws a distinct
plane per query head. Class `causal`: the runtime `(window_left, window_right)` pair with the causal sentinels resolved per sequence
(`fmha.resolve_window`), kv column c visible to q row i iff i - wl <= c <= i + wr; a KV tile visits only the Q tiles its band reaches (zero tiles
possible: the zeros are still stored). Class `bias`: P = exp2((sm_scale S + bias - lse') log2e), the bias `(batch, q head, q row, kv col)` at
`stride_b_batch / stride_b_head / stride_b_seq_q` (kv column contiguous; stride 0 broadcast), read at row and column clamped into the sequence; the
exponent chain carries no `nnan`/`ninf` (a `-inf` bias, "never attend", must reach `exp2`, which returns exactly 0). dB is the dQ kernel's store.
"""

import math as host_math


from kernels.attention_bwd_rdna import fmha_common_gfx1201 as fmha
from kernels.attention_rdna4.philox import Philox, keep_mask
from kernels.attention_bwd_rdna import fmha_common_gfx11 as cmn  # noqa: F401
from kernels.attention_bwd_rdna import fmha_tuning_bwd_dkdv_f32_gfx11 as tun
from kernels.attention_bwd_rdna.fmha_tuning_bwd_dkdv_f32_gfx11 import (
    BwdDkDvKnobs,
)  # noqa: F401

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

KERNEL_NAME = "fmha_bwd_dkdv_f32_gfx11_kernel"
_LOG2E = host_math.log2(host_math.e)

_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


def build_fmha_bwd_dkdv_f32_gfx11_module_primary(meta, knobs, runtime_axes=()):
    """Build the gfx11 fp32 dK/dV kernel for a resolved (meta, knobs) pair; `meta` is `variant_gfx11.Meta`, `knobs` the record of
    `fmha_tuning_bwd_dkdv_f32_gfx11.bwd_dkdv_knobs`."""
    if not knobs.legal:
        raise ValueError(f"the dK/dV knob record is not legal: {knobs.illegal_reason}")
    if tuple(runtime_axes):
        raise ValueError(
            f"runtime axes {tuple(runtime_axes)} are not built: dropout is baked (dform=bk)"
        )
    if meta.dtype != "fp32":
        raise ValueError(f"the f32 dK/dV supports fp32, got {meta.dtype!r}")
    if meta.cls not in ("plain", "causal", "bias"):
        raise ValueError(
            f"the gfx11 dK/dV builds classes 'plain', 'causal' and 'bias'; got cls={meta.cls!r}"
        )
    if not meta.padded and not (meta.hdim_qk == meta.hdim_vo == meta.tile):
        raise ValueError(
            f"an exact build needs hdim_qk == hdim_vo == tile ({meta.tile}); got ({meta.hdim_qk}, {meta.hdim_vo})"
        )
    if meta.hdim_qk > meta.tile or meta.hdim_vo > meta.tile:
        raise ValueError(
            f"hdim ({meta.hdim_qk}, {meta.hdim_vo}) exceeds the tile {meta.tile}"
        )
    if knobs.block_dmodel != meta.tile or knobs.block_dmodel_v != meta.tile:
        raise ValueError("the knob record was resolved for another tile")

    G = knobs.geometry
    BM, BN, T_, TPR, NPAIR, HD = G.block_q, G.block_kv, G.T, G.tpr, G.npair, meta.tile
    CH = G.chunks
    LY = G.lds()
    PITCH = tun.LDS_PITCH
    O_Q, O_DO, O_K, O_V, O_P, O_DS = LY.o_q, LY.o_do, LY.o_k, LY.o_v, LY.o_p, LY.o_ds
    chunk_col0 = [sum(CH[:c]) for c in range(len(CH))]
    chunk_acc0 = [
        sum(tun.cols_per_thread(w, TPR) for w in CH[:c]) for c in range(len(CH))
    ]
    NACC = G.acc_per_tensor()
    WAVES_PER_EU = knobs.waves_per_eu
    SCHED_STRATEGY = knobs.sched_strategy
    PADDED_HEAD = bool(meta.padded)
    CAUSAL = meta.cls == "causal"
    BIAS = meta.cls == "bias"
    DROPOUT = bool(meta.dropout)
    PHILOX = Philox.for_arch(get_rocm_arch())
    if DROPOUT and PHILOX.width != 32:
        raise ValueError(
            "the gfx11 dropout map is Philox 4x32 (the gfx950 map); the width table must fall back to 32 for this arch"
        )
    PHILOX_RN = PHILOX.randoms_per_offset

    # Exponent chain: no `nnan` / `ninf` (a -inf bias entry, a +inf LSE' must reach exp2); contract + reassoc only.
    fm_e = fmha.FastMath("safe")
    # Exponent chain of a bias object: contract only. No `nnan`/`ninf` (a -inf bias entry must reach exp2) and no `reassoc` (it would let the
    # compiler undo "combine in natural units, scale by log2 e last").
    fm_bias = fmha.FastMath("safe")
    fm_bias.flags = fx.arith.FastMathFlags.contract

    def _gptr(ptr):
        return fx.recast_iter(
            fx.PointerType.get(
                elem_ty=fx.Float32.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=4,
            ),
            ptr,
        )

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Float32, LY.total, 16]

    @flyc.kernel(known_block_size=[T_, 1, 1])
    def bwd_dkdv_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        Bias: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        rt_flags: fx.Int32,
    ):
        f32_ty = ir.F32Type.get()
        q_p, k_p, v_p, do_p = _gptr(Q), _gptr(K), _gptr(V), _gptr(DO)
        dk_p, dv_p, l_p, delta_p = _gptr(DK), _gptr(DV), _gptr(LSE), _gptr(Delta)
        if const_expr(BIAS):
            bias_p = _gptr(Bias)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek().buf.ptr

        # ---- Varlen prologue (the forward's, through the shared decoders) ----
        z_i32 = fx.Int32(gpu.block_idx.z)
        seqlen_q_i32, q_row_off, q_batch = fmha.decode_addressing(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32
        )
        seqlen_k_i32, k_row_off, k_batch = fmha.decode_addressing(
            varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32
        )
        lse_tokens = fmha.lse_token_pitch(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens
        )

        tid = fx.Int32(gpu.thread_idx.x)
        own_row, cpart = tun.owner_of(tid, TPR)

        kv_tile_idx = fx.Index(gpu.block_idx.x)
        head_k = fx.Index(gpu.block_idx.y)
        start_k_i32 = fx.Int32(kv_tile_idx) * fx.Int32(BN)
        # Under varlen the grid is sized from max_seqlen_k: whole workgroups can lie past this sequence's end.
        _alive = start_k_i32 < seqlen_k_i32
        q_last_i32 = fmha.common_utils.smax(seqlen_q_i32 - fx.Int32(1), fx.Int32(0))
        k_last_i32 = fmha.common_utils.smax(seqlen_k_i32 - fx.Int32(1), fx.Int32(0))

        # MQA/GQA: the KV head is shared by `group` query heads and dK/dV sum over all of them, folded into the Q-tile loop.
        group_i32 = num_head_q // num_head_k

        _q_batch_v = fx.Index(q_batch)
        _k_batch_v = fx.Index(k_batch)
        _q_row_off_v = fx.Index(q_row_off)
        _k_row_off_v = fx.Index(k_row_off)
        sm_log2e = sm_scale * fx.Float32(_LOG2E)
        hdq_i32 = fx.Int32(hdim_qk)
        hdv_i32 = fx.Int32(hdim_vo)

        # Tile bases (element offsets of row 0 of this sequence at head `head_k`) of K, V, dK, dV.
        k_base = (
            _k_batch_v * fx.Index(stride_k_batch)
            + head_k * fx.Index(stride_k_head)
            + _k_row_off_v * fx.Index(stride_k_seq)
        )
        v_base = (
            _k_batch_v * fx.Index(stride_v_batch)
            + head_k * fx.Index(stride_v_head)
            + _k_row_off_v * fx.Index(stride_v_seq)
        )
        dk_base = (
            _k_batch_v * fx.Index(stride_dk_batch)
            + head_k * fx.Index(stride_dk_head)
            + _k_row_off_v * fx.Index(stride_dk_seq)
        )
        dv_base = (
            _k_batch_v * fx.Index(stride_dv_batch)
            + head_k * fx.Index(stride_dv_head)
            + _k_row_off_v * fx.Index(stride_dv_seq)
        )

        def lds_ld(word):
            return fx.ptr_load(lds + fx.Int32(word), fx.Float32)

        def lds_st(word, val):
            fx.ptr_store(val, lds + fx.Int32(word))

        def stage(ptr, base, rstride, row0, row_last, rows, lds_off, c, bound):
            """VRAM -> LDS: a `rows` x `CH[c]` slab (head-dim columns `chunk_col0[c] ...`), row pitch PITCH; scalar loads, consecutive threads on consecutive
            columns. A row past the sequence is the last row (clamped); a column at or past `bound` (padded head) is zero and never addressed.
            """
            width = CH[c]
            for i in range_constexpr(rows * width // T_):
                r, cc = tun.stage_rc(tid + fx.Int32(i * T_), width)
                row = fmha.common_utils.smin(row0 + r, row_last)
                col = fx.Int32(chunk_col0[c]) + cc
                if const_expr(PADDED_HEAD):
                    ok = col < bound
                    colc = ok.select(col, fx.Int32(0))
                    val = ok.select(
                        fx.ptr_load(
                            ptr
                            + fx.Int64(base + fx.Index(row) * rstride + fx.Index(colc)),
                            fx.Float32,
                        ),
                        fx.Float32(0.0),
                    )
                else:
                    val = fx.ptr_load(
                        ptr + fx.Int64(base + fx.Index(row) * rstride + fx.Index(col)),
                        fx.Float32,
                    )
                lds_st(lds_off + r * fx.Int32(PITCH) + cc, val)

        # ---- Visited Q tiles: the whole sequence (plain), or the Q tiles the mask lets reach this KV tile (causal), times the GQA group ----
        if const_expr(CAUSAL):
            # kv column c is visible to q row i iff i - wl <= c <= i + wr (top-left: wl = Sq, wr = 0; bottom-right: wl = Sq, wr = Sk - Sq). The extents are
            # clamped to [-Sk, Sq] and [-Sq, Sk], which never changes a visible set and keeps every sum below in i32.
            _wl, _wr = fmha.resolve_window(
                window_left, window_right, seqlen_q_i32, seqlen_k_i32
            )
            _wl = fmha.common_utils.smin(
                fmha.common_utils.smax(fx.Int32(_wl), fx.Int32(0) - seqlen_k_i32),
                seqlen_q_i32,
            )
            _wr = fmha.common_utils.smin(
                fmha.common_utils.smax(fx.Int32(_wr), fx.Int32(0) - seqlen_q_i32),
                seqlen_k_i32,
            )
            _span_w = _wl + _wr + fx.Int32(1)
            # Q rows that see some key of this tile: [q_lo, q_hi) from the union of the band over the tile's live kv columns.
            _kv_end = fmha.common_utils.smin(start_k_i32 + fx.Int32(BN), seqlen_k_i32)
            _q_lo = fmha.common_utils.smax(start_k_i32 - _wr, fx.Int32(0))
            _q_hi = fmha.common_utils.smin(_kv_end + _wl, seqlen_q_i32)
            _first_blk = _q_lo // fx.Int32(BM)
            _any_live = _alive & (_span_w > fx.Int32(0)) & (_q_hi > _q_lo)
            _n_blocks_i32 = _any_live.select(
                (_q_hi + fx.Int32(BM - 1)) // fx.Int32(BM) - _first_blk, fx.Int32(0)
            )
            # Visiting order of the live tiles of one head: the tiles every row of which sees every key of this KV tile (full) first, then the leading
            # masked tiles, then the trailing masked tiles, each ascending. `_pre` leading masked tiles and `_nfull` full tiles, both clamped to
            # [0, n_blocks]; the order only permutes the accumulation, never which tiles are visited.
            _ff = fx.Int32(0) - (
                (fx.Int32(0) - (_kv_end - fx.Int32(1) - _wr)) // fx.Int32(BM)
            )  # first tile with q0 >= kv_end - 1 - wr (ceil)
            _lf = (start_k_i32 - fx.Int32(BM - 1) + _wl) // fx.Int32(
                BM
            )  # last tile with q0 <= k0 - (BM - 1) + wl (floor)
            _pre = fmha.common_utils.smin(
                fmha.common_utils.smax(_ff - _first_blk, fx.Int32(0)), _n_blocks_i32
            )
            _hi = fmha.common_utils.smin(
                fmha.common_utils.smax(_lf + fx.Int32(1) - _first_blk, _pre),
                _n_blocks_i32,
            )
            _nfull = _hi - _pre
            _nfull_pre = _nfull + _pre
            _neg_nfull = fx.Int32(0) - _nfull
        else:
            _n_blocks_i32 = _alive.select(
                (seqlen_q_i32 + fx.Int32(BM - 1)) // fx.Int32(BM), fx.Int32(0)
            )
        _nb_safe_i32 = fmha.common_utils.smax(_n_blocks_i32, fx.Int32(1))
        _total_i32 = group_i32 * _n_blocks_i32

        if const_expr(DROPOUT):
            # Plane stride from the launch maxima (the gfx950 / gfx1201 contract), never the sequence's own lengths.
            _ph_seed = fmha.philox_seed_value(philox_seed_ptr)
            _ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
            _ph_stride = PHILOX.grid_plane(
                _ph_off, fx.Int32(0), max_seqlen_q, max_seqlen_k
            )[1]

        # This thread's pair slots: the (q row, kv row) of each, fixed for the whole kernel.
        pair_m = []
        pair_n = []
        for i in range_constexpr(NPAIR):
            _m, _n = tun.pair_mn(tid + fx.Int32(i * T_), BN)
            pair_m.append(_m)
            pair_n.append(_n)
        pair_kv = [start_k_i32 + n for n in pair_n]  # absolute kv row of each pair
        if const_expr(BIAS):
            pair_kv_c = [fmha.common_utils.smin(kv, k_last_i32) for kv in pair_kv]
        if const_expr(DROPOUT):
            pair_slot_sel = [kv % fx.Int32(PHILOX_RN) for kv in pair_kv]

        def step(it, accs):
            _g = it // _nb_safe_i32
            _bi = it - _g * _nb_safe_i32
            if const_expr(CAUSAL):
                _bi = (
                    _first_blk
                    + _bi
                    + (_bi < _nfull).select(
                        _pre, (_bi < _nfull_pre).select(_neg_nfull, fx.Int32(0))
                    )
                )
            head_q_i32 = fx.Int32(head_k) * group_i32 + _g
            head_q = fx.Index(head_q_i32)
            q0 = _bi * fx.Int32(BM)  # first Q row of this tile
            q_base = (
                _q_batch_v * fx.Index(stride_q_batch)
                + head_q * fx.Index(stride_q_head)
                + _q_row_off_v * fx.Index(stride_q_seq)
            )
            do_base = (
                _q_batch_v * fx.Index(stride_do_batch)
                + head_q * fx.Index(stride_do_head)
                + _q_row_off_v * fx.Index(stride_do_seq)
            )
            q_rs, do_rs, k_rs, v_rs = (
                fx.Index(stride_q_seq),
                fx.Index(stride_do_seq),
                fx.Index(stride_k_seq),
                fx.Index(stride_v_seq),
            )
            lse_base, lse_pitch = fmha.lse_row_addressing(
                varlen_bits, _q_batch_v, head_q, num_head_q, lse_tokens, _q_row_off_v
            )
            if const_expr(BIAS):
                bias_base = (
                    _q_batch_v * fx.Index(stride_b_batch)
                    + head_q * fx.Index(stride_b_head)
                    + _q_row_off_v * fx.Index(stride_b_seq_q)
                )
            if const_expr(DROPOUT):
                _ph_plane = z_i32 * fx.Int32(num_head_q) + head_q_i32
                _ph_base = _ph_off + fx.Int64(_ph_plane) * fx.Int64(
                    max_seqlen_q
                ) * fx.Int64(_ph_stride)

            # (1) S = Q K^T and dP = dO V^T over the head dim, slab by slab
            s = [fx.Float32(0.0) for _ in range_constexpr(NPAIR)]
            dp = [fx.Float32(0.0) for _ in range_constexpr(NPAIR)]
            for c in range_constexpr(len(CH)):
                gpu.barrier()
                stage(q_p, q_base, q_rs, q0, q_last_i32, BM, O_Q, c, hdq_i32)
                stage(do_p, do_base, do_rs, q0, q_last_i32, BM, O_DO, c, hdv_i32)
                stage(k_p, k_base, k_rs, start_k_i32, k_last_i32, BN, O_K, c, hdq_i32)
                stage(v_p, v_base, v_rs, start_k_i32, k_last_i32, BN, O_V, c, hdv_i32)
                gpu.barrier()
                for i in range_constexpr(NPAIR):
                    qm = pair_m[i] * fx.Int32(PITCH)
                    kn = pair_n[i] * fx.Int32(PITCH)
                    for d in range_constexpr(CH[c]):
                        s[i] = s[i] + lds_ld(O_Q + qm + fx.Int32(d)) * lds_ld(
                            O_K + kn + fx.Int32(d)
                        )
                        dp[i] = dp[i] + lds_ld(O_DO + qm + fx.Int32(d)) * lds_ld(
                            O_V + kn + fx.Int32(d)
                        )

            # (2) P and dS to LDS
            for i in range_constexpr(NPAIR):
                q_row = q0 + pair_m[i]
                q_ok = q_row < seqlen_q_i32
                q_c = fmha.common_utils.smin(q_row, q_last_i32)
                row_off = lse_base + fx.Index(q_c) * lse_pitch
                if const_expr(BIAS):
                    # natural: the bias exponent scales by log2 e last
                    lse2 = fx.ptr_load(l_p + fx.Int64(row_off), fx.Float32)
                else:
                    lse2 = fx.ptr_load(
                        l_p + fx.Int64(row_off), fx.Float32
                    ) * fx.Float32(_LOG2E)
                delta = fx.ptr_load(delta_p + fx.Int64(row_off), fx.Float32)
                # p = exp2(sm_scale log2e S - lse log2e), or exp2((sm_scale S + bias - lse) log2e) for a bias object; a row no key reaches carries lse = +inf, which makes p exactly 0.
                if const_expr(BIAS):
                    b_off = (
                        bias_base
                        + fx.Index(q_c) * fx.Index(stride_b_seq_q)
                        + fx.Index(pair_kv_c[i])
                    )
                    bias_v = fx.ptr_load(bias_p + fx.Int64(b_off), fx.Float32)
                    e = fm_bias.mul(
                        fm_bias.sub(
                            fm_bias.add(fm_bias.mul(s[i], sm_scale), bias_v), lse2
                        ),
                        fx.Float32(_LOG2E),
                    )
                else:
                    e = fm_e.sub(fm_e.mul(s[i], sm_log2e), lse2)
                pv = fx.Float32(rocdl.exp2(f32_ty, fx.as_ir_value(e)))
                dead = (pair_kv[i] >= seqlen_k_i32) | (q_ok == fx.Boolean(False))
                if const_expr(CAUSAL):
                    rel = pair_kv[i] + _wl - q_row
                    dead = dead | (rel < fx.Int32(0)) | (rel >= _span_w)
                pv = dead.select(fx.Float32(0.0), pv)
                if const_expr(DROPOUT):
                    poff = PHILOX.grid_offset(_ph_base, _ph_stride, q_row, pair_kv[i])
                    rv = PHILOX.u32(_ph_seed, poff)
                    pick = fx.Int32(rv[0])
                    for j in range_constexpr(PHILOX_RN - 1):
                        pick = (pair_slot_sel[i] == fx.Int32(j + 1)).select(
                            fx.Int32(rv[j + 1]), pick
                        )
                    keep = keep_mask([pick], idropout_p)[0]
                    # dV takes the dropped, rescaled P; dS keeps the undropped P and a dropped, rescaled dP.
                    p_drop = keep.select(pv * dropout_scale, fx.Float32(0.0))
                    dpv = keep.select(dp[i] * dropout_scale, fx.Float32(0.0))
                else:
                    p_drop = pv
                    dpv = dp[i]
                ds = dead.select(fx.Float32(0.0), pv * (dpv - delta))
                slot = tun.pair_slot(pair_m[i], pair_n[i], BN)
                lds_st(O_P + slot, p_drop)
                lds_st(O_DS + slot, ds)

            # (3) owner accumulators += over the loop tile, slab by slab (loop-tile rows in order)
            new = list(accs)
            for c in range_constexpr(len(CH)):
                J = tun.cols_per_thread(CH[c], TPR)
                gpu.barrier()
                stage(q_p, q_base, q_rs, q0, q_last_i32, BM, O_Q, c, hdq_i32)
                stage(do_p, do_base, do_rs, q0, q_last_i32, BM, O_DO, c, hdv_i32)
                gpu.barrier()
                for m in range_constexpr(BM):
                    slot = fx.Int32(m * BN) + own_row
                    pvm = lds_ld(O_P + slot)
                    dsm = lds_ld(O_DS + slot)
                    for j in range_constexpr(J):
                        col = cpart + fx.Int32(j * TPR)
                        a = chunk_acc0[c] + j
                        new[a] = new[a] + pvm * lds_ld(
                            O_DO + fx.Int32(m * PITCH) + col
                        )  # dV
                        new[NACC + a] = new[NACC + a] + dsm * lds_ld(
                            O_Q + fx.Int32(m * PITCH) + col
                        )  # dK
            return [fx.as_ir_value(v) for v in new]

        accs = [fx.as_ir_value(fx.Float32(0.0)) for _ in range_constexpr(2 * NACC)]
        for it, inner in range(
            fx.Index(0), fx.Index(_total_i32), fx.Index(1), init=accs
        ):
            accs = yield step(fx.Int32(it), [fx.Float32(v) for v in inner])

        # ---- Epilogue: dV (V columns, hdim_vo) and dK (K columns, hdim_qk, times sm_scale); one predicated scalar store per element ----
        out_row = start_k_i32 + own_row
        row_live = out_row < seqlen_k_i32
        out_row_c = fmha.common_utils.smin(out_row, k_last_i32)
        for c in range_constexpr(len(CH)):
            for j in range_constexpr(tun.cols_per_thread(CH[c], TPR)):
                a = chunk_acc0[c] + j
                col = fx.Int32(chunk_col0[c]) + cpart + fx.Int32(j * TPR)
                if const_expr(PADDED_HEAD):
                    dk_ok = row_live & (col < hdq_i32)
                    dv_ok = row_live & (col < hdv_i32)
                else:
                    dk_ok = row_live
                    dv_ok = row_live
                if dk_ok:
                    fx.ptr_store(
                        fx.Float32(accs[NACC + a]) * sm_scale,
                        dk_p
                        + fx.Int64(
                            dk_base
                            + fx.Index(out_row_c) * fx.Index(stride_dk_seq)
                            + fx.Index(col)
                        ),
                    )
                if dv_ok:
                    fx.ptr_store(
                        fx.Float32(accs[a]),
                        dv_p
                        + fx.Int64(
                            dv_base
                            + fx.Index(out_row_c) * fx.Index(stride_dv_seq)
                            + fx.Index(col)
                        ),
                    )

    def _run(launcher, stream, num_kv_tiles, num_head_k_idx, nseq_idx):
        ctx = CompilationContext.get_current()
        if const_expr(WAVES_PER_EU is not None):
            _wpe = int(WAVES_PER_EU)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.WAVES_PER_EU"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        flat_wg_attr = ir.StringAttr.get(f"{T_},{T_}")
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["rocdl.FLAT_WORK_GROUP_SIZE"] = flat_wg_attr
        if const_expr(SCHED_STRATEGY):
            passthrough_entries = [
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("amdgpu-sched-strategy"),
                        ir.StringAttr.get(SCHED_STRATEGY),
                    ]
                )
            ]
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)
        launcher.launch(
            grid=(num_kv_tiles, num_head_k_idx, nseq_idx),
            block=(T_, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_fmha_bwd_dkdv_f32_gfx11(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        Bias: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        rt_flags: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        nseq_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        num_kv_tiles = (fx.Index(max_seqlen_k) + (BN - 1)) // BN
        _run(
            bwd_dkdv_kernel(
                Q,
                K,
                V,
                Bias,
                DO,
                DK,
                DV,
                LSE,
                Delta,
                seqinfo_q0,
                seqinfo_q1,
                seqinfo_k0,
                seqinfo_k1,
                varlen_bits,
                num_seqlens,
                max_seqlen_q,
                max_seqlen_k,
                window_left,
                window_right,
                philox_seed_ptr,
                philox_offset1,
                philox_offset2,
                idropout_p,
                dropout_scale,
                num_head_q,
                num_head_k,
                hdim_qk,
                hdim_vo,
                sm_scale,
                stride_q_batch,
                stride_q_head,
                stride_q_seq,
                stride_k_batch,
                stride_k_head,
                stride_k_seq,
                stride_v_batch,
                stride_v_head,
                stride_v_seq,
                stride_do_batch,
                stride_do_head,
                stride_do_seq,
                stride_dk_batch,
                stride_dk_head,
                stride_dk_seq,
                stride_dv_batch,
                stride_dv_head,
                stride_dv_seq,
                stride_b_batch,
                stride_b_head,
                stride_b_seq_q,
                rt_flags,
            ),
            stream,
            num_kv_tiles,
            fx.Index(num_head_k),
            nseq_idx,
        )

    def _launch(*args, **kwargs):
        return flyc.compile(launch_fmha_bwd_dkdv_f32_gfx11, *args, **kwargs)

    return _launch
