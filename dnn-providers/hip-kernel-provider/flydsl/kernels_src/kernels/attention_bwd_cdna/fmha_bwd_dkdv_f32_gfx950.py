# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""dK / dV of the fp32 backward for gfx950: tiles 32 .. 256 and t512 (every class, padded head dims), each (tile, geometry) on the schedule of the table.

The fp32 twin of `fmha_bwd_dkdv_gfx950` on the native-f32-MFMA core (`fmha_f32_common_gfx950`, `fmha_f32_dsplit_gfx950`): same argument names and
order, same grid (KV head, KV block, sequence), same math, with 4-byte elements and `v_mfma_f32_16x16x4_f32`.

--- Schedule ---------------------------------------------------------------------

`SCHEDULE_TABLE` (`fmha_tuning_bwd_dkdv_f32_gfx950`) assigns every (tile, geometry) its schedule and wave count; this module builds the tiles of
`BUILT_TILES`.

The two layers: the D-split (t512 below; also the other tiles whose table row says 'ds') and the per-wave (t64 below; the rows that say 'pw').

t512 (D-split, both geometries: block_q 32 `t512_32x16_ds_w4`, block_q 16 `t512_16x16_ds_w4`): a workgroup of 4 waves owns 16 KV rows, wave w owns
head-dim slice w (8 panels of 16 columns) of K, V, dK, dV, and stages only its slice of every q tile. S and dP contract the head dim, so each wave's accumulators are *partial* sums; `DSplitDkDv.exchange` adds the four partials in
wave order through LDS. Per q tile (`block_q` rows, `RB = block_q / 16` blocks of 16 rows) the phase order is that layer's:

    stage (global -> LDS DMA, own slice); sync; scores (partials); exchange (write, sync, sum)
    P, dS                                   (this module: softmax recompute from LSE, dS = P (dP - Delta))
    accumulate (dV += dO^T P, dK += Q^T dS); barrier

t64 (per-wave, both geometries: block_q 32 `t64_32x16_pw_w4`, block_q 16 `t64_16x16_pw_w4`): a workgroup of 4 waves owns 64 KV rows, wave w owns the 16 rows
`kv0 + 16 w ..` and the full head dim (4 panels): K, V, dK, dV resident per lane. The waves share the staged q tile (panels split over the waves), so the
loop is `stage; sync; scores; P, dS; accumulate; barrier` with no exchange: S and dP are complete in the wave's accumulators. A wave whose 16 rows start
past `seqlen_k` runs on clamped rows (it must reach every barrier) and stores nothing.

After `exchange`, lane (g, i) of block s holds S and dP of tile row `16 s + g + 4 i` at KV column m (= lane % 16): the accumulator layout the P / dS math
and both accumulations use unchanged.

--- Math (per tile row r, KV column m) -------------------------------------------

    P  = exp2(sm_scale log2e S - log2e LSE')        LSE' from the prep kernel (+inf for an empty row: P = 0)
    dS = P (dP - Delta)                             Delta = rowsum(O dO) from the prep kernel
    dV += P^T dO        dK += dS^T Q                fp32 MFMA accumulators, q ascending, GQA group-major then q
    dK *= sm_scale                                  once, after the last tile: dS is the gradient of the *scaled* logits, so the scale belongs to dK only

Rows of the last tile past `seqlen_q` are staged as copies of the last valid row (`stage_row_off` clamps); P and dS of those rows are forced to 0, which adds exactly
+0.0 to every accumulator. KV rows of the last block past `seqlen_k` read the last valid row and are not stored. LSE' and Delta are read through the shared
`lse_row_addressing` decode, the one the fp32 prep kernel writes with.

--- Scope ------------------------------------------------------------------------

Built: tiles 32 .. 256 and t512 with causal (top-left, bottom-right and window masks from the runtime `window_left` / `window_right`: `kv - right <= q <= kv + left`, the sentinels
resolved per sequence; the q tiles a KV block walks are the 16-bit family's `decompose_causal_regions` cut with the axes swapped), bias (an fp32 `(batch, q head, q row, kv col)`
element added to the scaled logits in the exponent, read directly), baked dropout (the 16-bit kernel's Philox plane `(sequence, q head)`, grid `max_seqlen_q x max_seqlen_k`; dP carries
keep and the survivor scale, dV takes keep P and the scale once at the end), varlen (the shared decode), GQA and padded head dims (`hdim_qk` / `hdim_vo` are launch values: resident K / V loads
and dK / dV stores are masked element-wise, the staged tiles are bounded buffer DMAs masked on the A-operand read; whole panels under `hdim_qk_floor` skip the mask). Everything else is refused by name in `_check_scope`. dB is stored by the dQ kernel, not here.

Objects that carry a feature pin their kernarg strides and the loop-invariant parts of their bases as opaque scalars in the prologue (`pin_sgpr`): left as kernarg products they keep the
`s_load_dwordx16` tuples live across the tile walk and the backend's lane spill of them trips the lane-spill check on the compiled object (ROCm/FlyDSL#1087). The plain exact objects are built without it.

--- Resources --------------------------------------------------------------------

LDS is the layer's (`lds_bytes`): t512 155,648 B at block_q 32 and 77,824 B at block_q 16; t64 17,408 B at block_q 32 and 8,704 B at block_q 16. The compile hint `unroll-max-count = 1` gives
one loop body per object, so the resource row is the loop's and not an LLVM unroll of it. No fast-math hints: the exchange order is the
documented wave order and `+inf` LSE' must survive to `exp2`.
"""


from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha  # noqa: E402
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    exp2_wait_state,
)  # noqa: E402
from kernels.attention_bwd_cdna.fmha_f32_common_gfx950 import (
    ColMask,
    DeviceOps,
    Lane,
    Philox,
    check_head_dim_floor,
    check_slab_span,
    launch_ints,
    slab_span,
)  # noqa: E402
from kernels.attention_bwd_cdna import philox  # noqa: E402
from kernels.attention_bwd_cdna.fmha_f32_dsplit_gfx950 import DSplitDkDv  # noqa: E402
from kernels.attention_bwd_cdna.fmha_f32_perwave_gfx950 import PerWaveDkDv  # noqa: E402
from kernels.attention_bwd_cdna.fmha_tuning_bwd_dkdv_f32_gfx950 import (
    BwdDkDvInputMetadata,
    bwd_dkdv_knobs,
)  # noqa: E402
from kernels.attention_bwd_cdna.gfx950_standalone import dualwave  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import range_constexpr, rocdl  # noqa: E402
from flydsl.expr import math as fmath  # noqa: E402
from kernels.attention_bwd_cdna.fmha_bwd_bias_exp_gfx950 import (
    FM_BIAS_EXP,
)  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402
from flydsl.expr.utils.arith import _to_raw as as_mlir_value  # noqa: E402

__all__ = [
    "BUILT_TILES",
    "KERNEL_NAME",
    "build_fmha_bwd_dkdv_f32_gfx950_module",
    "build_fmha_bwd_dkdv_f32_gfx950_module_primary",
    "check_launch_limits",
]

KERNEL_NAME = "fmha_bwd_dkdv_f32_gfx950_kernel"

# Tiles this module builds. The schedule of a (tile, geometry) is the table's (`SCHEDULE_TABLE`).
BUILT_TILES = (32, 64, 96, 128, 160, 192, 224, 256, 512)

_COMPILE_HINTS = {"llvm_options": {"unroll-max-count": 1}}


def _check_scope(meta, knobs, runtime_axes):
    """Refuse what this module does not compute, at the decision rather than at an address."""
    if runtime_axes:
        raise NotImplementedError(
            f"runtime axes {sorted(runtime_axes)}: the fp32 dK/dV kernel builds baked forms only"
        )
    if knobs.block_dmodel not in BUILT_TILES:
        raise NotImplementedError(
            f"fp32 dK/dV builds tiles {BUILT_TILES}, got {knobs.block_dmodel}"
        )
    if (
        meta.head_dim != knobs.block_dmodel
        or (meta.head_dim_v or meta.head_dim) != knobs.block_dmodel
    ):
        raise NotImplementedError(
            "fp32 dK/dV takes the tile as its head dim: the real head dims of a padded object are launch values"
        )


def check_launch_limits(knobs, **launch):
    """Refuse launch values this object cannot compute, by name. `launch` holds plain-int launch arguments by their `launch_fmha_bwd_dkdv_f32_gfx950` names; a
    value that is absent is not checked. Only padded objects have limits: the real head dims `hdim_qk` / `hdim_vo` must lie above the object's `hdim_qk_floor` (one floor
    covers the QK and the VO side), and the Q and dO slabs (`max_seqlen_q` rows at their row strides) must stay below `SLAB_SPAN_LIMIT` elements.
    """
    if not knobs.padded_head:
        return
    floor = int(knobs.hdim_qk_floor or 0)
    for name in ("hdim_qk", "hdim_vo"):
        if name in launch:
            check_head_dim_floor(name, launch[name], floor)
    for tensor, stride, hdim in (
        ("Q", "stride_q_seq", "hdim_qk"),
        ("DO", "stride_do_seq", "hdim_vo"),
    ):
        if all(k in launch for k in ("max_seqlen_q", stride, hdim)):
            check_slab_span(
                tensor, launch["max_seqlen_q"], launch[stride], launch[hdim]
            )


def build_fmha_bwd_dkdv_f32_gfx950_module_primary(meta, knobs, runtime_axes=()):
    """Build the fp32 dK/dV kernel for a resolved (meta, knobs) pair."""
    runtime_axes = frozenset(runtime_axes)
    if knobs.schedule is None:
        raise ValueError(
            "knobs must be resolved: call `bwd_dkdv_knobs(arch, ...).resolve(meta)` first"
        )
    _check_scope(meta, knobs, runtime_axes)
    if meta.dropout and philox.default_width("gfx950") != 32:
        raise ValueError(
            "the gfx950 dropout stream is Philox-4x32 (the forward's); philox.default_width('gfx950') must be 32"
        )
    fam = knobs.family()
    PW = fam.sched == "pw"
    layer = PerWaveDkDv(fam) if PW else DSplitDkDv(fam)
    W, RB, BQ, DSU = (
        fam.waves,
        fam.rb,
        fam.bq,
        fam.dsu,
    )  # DSU: resident panels per lane (the whole head dim for per-wave)
    BLOCK = fam.block
    BLOCK_KV = knobs.block_kv
    LDS_WORDS = layer.lds_words
    _cache_tag = (KERNEL_NAME, fam.tile, fam.bq, fam.sched, fam.waves)

    CAUSAL, WINDOW, BIAS, DROPOUT = (
        bool(meta.causal),
        bool(meta.window),
        bool(meta.bias),
        bool(meta.dropout),
    )
    PADDED = bool(knobs.padded_head)
    PIN = CAUSAL or BIAS or DROPOUT or PADDED
    FLOOR = (
        knobs.hdim_qk_floor or 0
    )  # head dims of the object are above it: whole panels under it skip the read-side column mask

    # The two layers differ in resident loading and in the partial-sum exchange; plain Python helpers keep the kernel body free of a build-time branch.
    def load_resident(
        ops, L, kv0, kv_row0, k_last, k_base, v_base, stride_k_seq, stride_v_seq, mq, mv
    ):
        if PW:
            return layer.load_resident(
                ops,
                L,
                kv_row0,
                k_last,
                k_base,
                v_base,
                stride_k_seq,
                stride_v_seq,
                mq,
                mv,
            )
        kv_row = fx.min(kv0 + L.m, k_last)
        K_res, V_res = layer.load_resident(
            ops,
            L,
            k_base + fx.Int64(kv_row) * stride_k_seq,
            v_base + fx.Int64(kv_row) * stride_v_seq,
            mq,
            mv,
        )
        layer.zero_pads(ops, L)
        return K_res, V_res

    def pin_sgpr(value):
        """An opaque scalar copy of a kernarg stride, computed in the prologue: the features' per-head bases and spans keep the kernarg
        `s_load_dwordx16` tuples live across the tile walk, which the backend lane-spills (ROCm/FlyDSL#1087); a side-effecting inline asm cannot be sunk, so the tuple dies here.
        The plain exact objects (t64, t512) are built without it."""
        if not PIN:
            return value
        ir_val = as_mlir_value(value)
        return fx.Int64(
            dualwave.llvm.inline_asm(
                ir_val.type, [ir_val], "", "=s,0", has_side_effects=True
            )
        )

    def exchange(ops, L, S, Pd):
        return (S, Pd) if PW else layer.exchange(ops, L, S, Pd)

    def tile_cut(kv0, k_len, q_len, window_left, window_right):
        """(first q tile, q tile count, `-right` bound, left bound) of the workgroup's KV rows. A (q, kv) element is live iff `kv - right <= q <= kv + left`
        (the sentinels resolve against this sequence's lengths); the tile range is the 16-bit family's cut with the axes swapped (a KV block's q range).
        """
        if not CAUSAL:
            return None, (q_len + fx.Int32(BQ - 1)) // fx.Int32(BQ), None, None
        left, right = fmha.resolve_window(window_left, window_right, q_len, k_len)
        reg = fmha.decompose_causal_regions(
            kv0, k_len, q_len, right, left, BLOCK_KV, BQ, fx.Boolean(True)
        )
        n = reg.n_left + reg.n_full + reg.n_right
        return (
            reg.left_col0 // fx.Int32(BQ),
            (n > fx.Int32(0)).select(n, fx.Int32(0)),
            fx.Int32(0) - fx.Int32(right),
            fx.Int32(left),
        )

    def tile_q0(it, t0):
        """First q row of tile `it` of the workgroup's walk."""
        return (
            fx.Int32(it) * fx.Int32(BQ)
            if not CAUSAL
            else (fx.Int32(it) + t0) * fx.Int32(BQ)
        )

    def row_live(row, kv_col, q_len, lo, left):
        """Whether q row `row` exists and, under a mask, sees KV column `kv_col`: `row - kv_col >= lo` (and `<= left` for a window)."""
        ok = row < q_len
        if CAUSAL:
            rel = row - kv_col
            ok = ok & (rel >= lo)
            if WINDOW:
                ok = ok & (rel <= left)
        return ok

    def keep_of(rng, seed, plane_base, row_stride, slot, row, kv_col, thr):
        """Philox keep flag of element (q row, KV column): the stream the 16-bit kernel draws (`philox_keep`), slot `kv_col % randoms_per_offset` of the call."""
        vals = rng.u32(
            seed,
            rng.grid_offset(plane_base, row_stride, fx.Int64(row), fx.Int64(kv_col)),
        )
        picked = fx.Int32(vals[0])
        for j in range(len(vals) - 1):
            picked = (slot == fx.Int32(j + 1)).select(fx.Int32(vals[j + 1]), picked)
        return picked > fx.Int32(thr)

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Float32, LDS_WORDS, 16]

    def _ptr_ty(align):
        return fx.PointerType.get(
            elem_ty=fx.Float32.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=align,
        )

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def fmha_bwd_dkdv_f32_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
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
        I32 = fx.Int32
        I64 = fx.Int64
        tid = I32(fx.thread_idx.x)
        wave = I32(rocdl.readfirstlane(T.i32, as_mlir_value(tid // I32(64))))
        lane = tid - wave * I32(64)
        L = Lane.of(lane, wave)
        kv_head = fx.Index(fx.block_idx.x)
        kv0 = I32(fx.block_idx.y) * I32(BLOCK_KV)
        z = I32(fx.block_idx.z)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        f32 = lambda P, align: fx.recast_iter(_ptr_ty(align), P)  # noqa: E731
        gptrs = {
            "Q": f32(Q, 16),
            "K": f32(K, 16),
            "V": f32(V, 16),
            "DO": f32(DO, 16),
            "DK": f32(DK, 16),
            "DV": f32(DV, 16),
            **({"B": f32(B, 4)} if BIAS else {}),
        }
        glse = f32(LSE, 4)
        gdel = f32(Delta, 4)
        ops = DeviceOps(lds.buf.ptr, gptrs)

        # sequence decode: Q side bits 7:0, K side bits 15:8 (shared helpers; dense = bits 0)
        q_len, q_row_off, q_batch = fmha.decode_addressing(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z
        )
        k_len, k_row_off, k_batch = fmha.decode_addressing(
            varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z
        )
        tokens = fmha.lse_token_pitch(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens
        )
        q_len = I32(q_len)
        k_len = I32(k_len)
        stride_q_batch, stride_q_head, stride_q_seq = (
            pin_sgpr(x) for x in (stride_q_batch, stride_q_head, stride_q_seq)
        )
        stride_do_batch, stride_do_head, stride_do_seq = (
            pin_sgpr(x) for x in (stride_do_batch, stride_do_head, stride_do_seq)
        )
        stride_dk_seq, stride_dv_seq = pin_sgpr(stride_dk_seq), pin_sgpr(stride_dv_seq)
        stride_b_batch, stride_b_head, stride_b_seq_q = (
            pin_sgpr(x) if BIAS else x
            for x in (stride_b_batch, stride_b_head, stride_b_seq_q)
        )
        q_last = (q_len > I32(0)).select(q_len - I32(1), I32(0))
        k_last = (k_len > I32(0)).select(k_len - I32(1), I32(0))

        # KV side: slab bases (elements) of K, V, dK, dV for this KV head
        def kv_base(s_batch, s_head, s_seq):
            return (
                I64(k_batch) * s_batch + I64(kv_head) * s_head + I64(k_row_off) * s_seq
            )

        k_base = pin_sgpr(kv_base(stride_k_batch, stride_k_head, stride_k_seq))
        v_base = pin_sgpr(kv_base(stride_v_batch, stride_v_head, stride_v_seq))
        dk_base = pin_sgpr(kv_base(stride_dk_batch, stride_dk_head, stride_dk_seq))
        dv_base = pin_sgpr(kv_base(stride_dv_batch, stride_dv_head, stride_dv_seq))

        # resident K, V of the lane's KV row (clamped to the last row): this wave's 16 rows (per-wave) or the workgroup's 16 rows (D-split)
        kv_row0 = kv0 + L.wave * I32(16) if PW else kv0
        # padded head dims: the real column counts (launch values) of the QK side (Q, K, dK) and the VO side (dO, V, dV); the staged tiles are bounded buffer DMAs over their slabs
        mq = ColMask(I32(hdim_qk), False, FLOOR) if PADDED else None
        mv = ColMask(I32(hdim_vo), False, FLOOR) if PADDED else None
        q_span = slab_span(ops, q_len, stride_q_seq, hdim_qk) if PADDED else None
        do_span = slab_span(ops, q_len, stride_do_seq, hdim_vo) if PADDED else None
        K_res, V_res = load_resident(
            ops,
            L,
            kv0,
            kv_row0,
            k_last,
            k_base,
            v_base,
            stride_k_seq,
            stride_v_seq,
            mq,
            mv,
        )

        c_scale_log2e = sm_scale * fx.Float32(dualwave._LOG2E)
        c_neg_log2e = fx.Float32(-dualwave._LOG2E)
        c_log2e = fx.Float32(dualwave._LOG2E)
        # the lane's KV column (unclamped: a mask and a Philox draw address the real column; a bias read clamps it)
        kv_col = kv_row0 + L.m if (CAUSAL or DROPOUT or BIAS) else None
        kv_col_in = fx.min(kv_col, k_last) if BIAS else None
        # a workgroup whose KV block starts past this sequence runs no tile (uniform) and stores nothing
        t0, n_tiles, lo, left = tile_cut(kv0, k_len, q_len, window_left, window_right)
        ntile = fx.Index((kv0 < k_len).select(n_tiles, I32(0)))
        rng = Philox.for_arch("gfx950") if DROPOUT else None
        seed = fmha.philox_seed_value(philox_seed_ptr) if DROPOUT else None
        offset_base = (
            fmha.philox_offset_base(philox_offset1, philox_offset2) if DROPOUT else None
        )
        slot = kv_col % I32(rng.randoms_per_offset) if DROPOUT else None
        # the loop-invariant part of the query-side bases (batch and row origin), computed here and not sunk into the group loop
        q_pre = (
            pin_sgpr(I64(q_batch) * stride_q_batch + I64(q_row_off) * stride_q_seq)
            if PIN
            else None
        )
        do_pre = (
            pin_sgpr(I64(q_batch) * stride_do_batch + I64(q_row_off) * stride_do_seq)
            if PIN
            else None
        )
        b_pre = (
            pin_sgpr(I64(q_batch) * stride_b_batch + I64(q_row_off) * stride_b_seq_q)
            if BIAS
            else None
        )
        # LSE / Delta addressing of the group's first query head and its per-head step: the decode (varlen bits, batch, token pitch) is computed once here, not per head in the loop
        group0 = kv_head * (fx.Index(num_head_q) // fx.Index(num_head_k))
        lse_b0, lse_pitch0 = (
            fmha.lse_row_addressing(
                varlen_bits,
                q_batch,
                group0,
                fx.Index(num_head_q),
                fx.Index(tokens),
                q_row_off,
            )
            if PIN
            else (None, None)
        )
        lse_base0 = fx.Index(pin_sgpr(fx.Int64(lse_b0))) if PIN else None
        lse_pitch0 = fx.Index(pin_sgpr(fx.Int64(lse_pitch0))) if PIN else None
        lse_step = (
            fx.Index(
                pin_sgpr(
                    fx.Int64(fmha.lse_row_step_per_head(varlen_bits, fx.Index(tokens)))
                )
            )
            if PIN
            else None
        )
        group = fx.Index(num_head_q) // fx.Index(num_head_k)

        init_args = [ops.zero(4) for _ in range_constexpr(2 * DSU)]
        group_results = init_args
        for g, group_args in range(fx.Index(0), group, fx.Index(1), init=init_args):
            # the query head g of this KV head's group; K, V, dK, dV and the accumulators do not move
            q_head = kv_head * group + g
            q_base = (
                (q_pre + I64(q_head) * stride_q_head)
                if PIN
                else I64(q_batch) * stride_q_batch
                + I64(q_head) * stride_q_head
                + I64(q_row_off) * stride_q_seq
            )
            # the bias row (q_batch, q_head, q row 0 of the sequence) and the Philox plane (sequence, q head) move with the query head
            b_base = b_pre + I64(q_head) * stride_b_head if BIAS else None
            plane_base, row_stride = (
                rng.grid_plane(
                    offset_base,
                    z * I32(num_head_q) + I32(q_head),
                    I32(max_seqlen_q),
                    I32(max_seqlen_k),
                )
                if DROPOUT
                else (None, None)
            )
            do_base = (
                (do_pre + I64(q_head) * stride_do_head)
                if PIN
                else I64(q_batch) * stride_do_batch
                + I64(q_head) * stride_do_head
                + I64(q_row_off) * stride_do_seq
            )
            lse_base, lse_pitch = (
                (lse_base0 + g * lse_step, lse_pitch0)
                if PIN
                else fmha.lse_row_addressing(
                    varlen_bits,
                    q_batch,
                    q_head,
                    fx.Index(num_head_q),
                    fx.Index(tokens),
                    q_row_off,
                )
            )
            loop_results = group_args
            for it, st in range(fx.Index(0), ntile, fx.Index(1), init=group_args):
                dK = [st[i] for i in range_constexpr(DSU)]
                dV = [st[DSU + i] for i in range_constexpr(DSU)]
                q0 = tile_q0(it, t0)
                layer.stage(
                    ops,
                    L,
                    q0,
                    q_last,
                    q_base,
                    do_base,
                    stride_q_seq,
                    stride_do_seq,
                    mq,
                    mv,
                    q_span,
                    do_span,
                )
                ops.sync()
                S, Pd = layer.scores(ops, L, K_res, V_res, mq, mv)
                S, Pd = exchange(ops, L, S, Pd)

                # P = exp2(c S - log2e LSE'), dS = P (dP - Delta); rows past the sequence end are zero. A bias build combines in natural units and scales last, P = exp2((sm_scale S + bias - LSE) log2e), so a lowest-finite bias cannot overflow before the LSE is subtracted
                rows = [
                    [q0 + I32(16 * s) + L.g + I32(4 * i) for i in range_constexpr(4)]
                    for s in range_constexpr(RB)
                ]
                offs = [
                    [
                        lse_base + fx.Index(fx.min(rows[s][i], q_last)) * lse_pitch
                        for i in range_constexpr(4)
                    ]
                    for s in range_constexpr(RB)
                ]
                lse = [
                    [
                        fx.ptr_load(glse + I64(offs[s][i]), fx.Float32)
                        for i in range_constexpr(4)
                    ]
                    for s in range_constexpr(RB)
                ]
                dlt = [
                    [
                        fx.ptr_load(gdel + I64(offs[s][i]), fx.Float32)
                        for i in range_constexpr(4)
                    ]
                    for s in range_constexpr(RB)
                ]
                bias_nat = (
                    [
                        [
                            ops.ldg(
                                "B",
                                b_base
                                + I64(fx.min(rows[s][i], q_last)) * stride_b_seq_q
                                + I64(kv_col_in),
                                1,
                            )
                            for i in range_constexpr(4)
                        ]
                        for s in range_constexpr(RB)
                    ]
                    if BIAS
                    else None
                )
                expo = exp2_wait_state(
                    [
                        dualwave.rocdl.exp2(
                            T.f32,
                            as_mlir_value(
                                ops.elem(S[s], i) * c_scale_log2e
                                + lse[s][i] * c_neg_log2e
                                if not BIAS
                                else (
                                    fmath.fma(
                                        ops.elem(S[s], i),
                                        sm_scale,
                                        bias_nat[s][i],
                                        fastmath=FM_BIAS_EXP,
                                    )
                                    - lse[s][i]
                                )
                                * c_log2e
                            ),
                        )
                        for s in range_constexpr(RB)
                        for i in range_constexpr(4)
                    ]
                )
                keep = (
                    [
                        [
                            keep_of(
                                rng,
                                seed,
                                plane_base,
                                row_stride,
                                slot,
                                rows[s][i],
                                kv_col,
                                idropout_p,
                            )
                            for i in range_constexpr(4)
                        ]
                        for s in range_constexpr(RB)
                    ]
                    if DROPOUT
                    else None
                )
                zero = fx.Float32(0.0)
                Pm = []
                dS = []
                for s in range_constexpr(RB):
                    ps = []
                    ds = []
                    for i in range_constexpr(4):
                        valid = row_live(rows[s][i], kv_col, q_len, lo, left)
                        p = valid.select(fx.Float32(expo[4 * s + i]), zero)
                        # under dropout dP carries the keep flag and the survivor scale, P (the undropped softmax) does not; dV takes the kept P
                        dp = (
                            keep[s][i].select(ops.elem(Pd[s], i) * dropout_scale, zero)
                            if DROPOUT
                            else ops.elem(Pd[s], i)
                        )
                        ps.append(keep[s][i].select(p, zero) if DROPOUT else p)
                        ds.append(valid.select(p * (dp - dlt[s][i]), zero))
                    Pm.append(ps)
                    dS.append(ds)

                layer.accumulate(ops, L, Pm, dS, dK, dV)
                ops.barrier()
                loop_results = yield dK + dV
            group_results = yield [loop_results[i] for i in range_constexpr(2 * DSU)]

        # dS is the gradient of the scaled logits: sm_scale belongs to dK; dropout's survivor scale belongs to dV (dK has it inside dS, on the dP term)
        dK = [
            ops.vec(
                [ops.elem(group_results[i], j) * sm_scale for j in range_constexpr(4)]
            )
            for i in range_constexpr(DSU)
        ]
        dV = [
            (
                ops.vec(
                    [
                        ops.elem(group_results[DSU + i], j) * dropout_scale
                        for j in range_constexpr(4)
                    ]
                )
                if DROPOUT
                else group_results[DSU + i]
            )
            for i in range_constexpr(DSU)
        ]
        layer.store(
            ops,
            L,
            kv_row0,
            k_len,
            dk_base,
            dv_base,
            stride_dk_seq,
            stride_dv_seq,
            dK,
            dV,
            mq,
            mv,
        )

    @flyc.jit
    def launch_fmha_bwd_dkdv_f32_gfx950(
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
        _ = _cache_tag
        num_kv_blocks = (fx.Index(max_seqlen_k) + fx.Index(BLOCK_KV - 1)) // fx.Index(
            BLOCK_KV
        )
        # the grid's z extent counts sequences: `num_seqlens` for a packed batch slot, else `batch_size`
        bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        fmha_bwd_dkdv_f32_gfx950_kernel(
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
        ).launch(
            grid=(fx.Index(num_head_k), num_kv_blocks, bs_idx),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    def _launch(*args, **kwargs):
        check_launch_limits(
            knobs, **launch_ints(launch_fmha_bwd_dkdv_f32_gfx950, args, kwargs)
        )
        return flyc.compile(launch_fmha_bwd_dkdv_f32_gfx950, *args, **kwargs)

    _launch.traits = fam
    _launch.knobs = knobs
    _launch.cache_tag = _cache_tag
    _launch.lds_bytes = layer.lds_bytes
    _launch.check_launch_limits = lambda **launch: check_launch_limits(knobs, **launch)
    return _launch


def build_fmha_bwd_dkdv_f32_gfx950_module(arch="gfx950", **kwargs):
    """Keyword front end: name a problem, get the table's schedule."""
    from dataclasses import fields as _fields

    meta_fields = {f.name for f in _fields(BwdDkDvInputMetadata)}
    meta = BwdDkDvInputMetadata(**{k: v for k, v in kwargs.items() if k in meta_fields})
    knob_kwargs = {k: v for k, v in kwargs.items() if k not in meta_fields}
    return build_fmha_bwd_dkdv_f32_gfx950_module_primary(
        meta, bwd_dkdv_knobs(arch, **knob_kwargs).resolve(meta)
    )
