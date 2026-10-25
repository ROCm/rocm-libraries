# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""dQ (and dB) of the fp32 backward for gfx950 (F.Q): the D-split dQ kernel at tiles 32 .. 256 and 512 (32, 64, 96, 128, 160, 192, 224, 256, 512), both geometries, every class.

The fp32 twin of `fmha_bwd_dq_gfx950` on the native-f32-MFMA core (`fmha_f32_common_gfx950`, `fmha_f32_dsplit_gfx950`): same argument names and order
(`parity/gaps/fp32/interfaces.md`), same grid (query head, q block, sequence), same math, with 4-byte elements and the fp32 MFMAs.

--- Schedule ---------------------------------------------------------------------

`DQ_TABLE` (`fmha_tuning_bwd_dq_f32_gfx950`) assigns every (tile, geometry) the D-split dQ layer (`block_m` 32 or 16, 2 or 4 waves). The workgroup owns `block_m` q rows (`RB = block_m / 16` blocks of 16 rows); wave w owns head-dim slice w of Q, dO (resident) and dQ
(accumulators); K and V are staged 16 KV rows at a time, each wave staging and reading back only its own slice. S^T = K Q^T and dP^T = V dO^T contract
the head dim, so each wave's accumulators are *partial* sums; `DSplitDq` adds the W partials in wave order through LDS. Per KV stage the phase order is
that layer's:

    stage_kv; wait_dma; scores (partials); write_partials (+ sync); ds_operands16 | ds_operands32 (wave-ordered sums)
    dS                                          (this module: softmax recompute from LSE, dS = P (dP - Delta))
    accumulate16 | accumulate32; barrier

After the exchange, lane (g, i) of a `block_m` 16 workgroup holds S, dP of KV row `g + 4 i` of the stage at q column m; at `block_m` 32 the sums come
back in the 32x32x2 B-operand layout (`ds_operands32`: step 4 h + i holds KV row `2 h + g32 + 4 i`, q row `lane % 32`), which is the layout the dQ
accumulation (`dQ^T += K^T dS^T`, A = the staged K chunk read as b32 columns) consumes with no repack.

--- Math (per q row r, KV row j) -------------------------------------------------

    P  = exp2(sm_scale log2e S - log2e LSE')        LSE' from the prep kernel (+inf for an empty row: P = 0)
    dS = P (dP - Delta)                             Delta = rowsum(O dO) from the prep kernel
    dQ += dS K                                      fp32 MFMA accumulators, KV ascending
    dQ *= sm_scale                                  once, after the last stage: dS is the gradient of the *scaled* logits, so the scale belongs to dQ only

Rows of the workgroup past `seqlen_q` read the last valid q row (`row_stats`, `load_resident`) and are not stored; KV rows of the last stage past `seqlen_k`
are staged as copies of the last valid row (`stage_row_off` clamps) and their P and dS are forced to 0, which adds exactly +0.0 to every accumulator. LSE' and Delta
are read through the shared `lse_row_addressing` decode, the one F.P's prep writes with.

--- Scope ------------------------------------------------------------------------

Classes (baked, one object per Spec): plain; causal, whose masks are runtime window arguments (`window_left`, `window_right`, sentinels resolved per sequence by
`fmha.resolve_window`: visible iff row - wl <= col <= row + wr; the KV loop walks only the stages the block's rows reach); bias (the bias input is read through its
strides, `bias * log2e` joins the scaled score) with `store_db` (dB = dS, fp32, in Q's batch and row layout; a runtime all-zero stride triple means "not requested");
baked dropout (the forward's Philox mask regenerated from (seed, offset, plane = sequence * Hq + head, max_seqlen_q x max_seqlen_k); `dP` is dropped and scaled,
`P` is not); varlen (the shared `decode_addressing` of both sides); GQA (`kv_head = q_head / group`); padded head dims (`hdim_qk` masks Q, K, dQ, `hdim_vo` masks
dO, V: element-wise runtime limits, staged K / V through bounded buffer DMA with read-side masking). The floor (`hdim_qk_floor`) is carried into the masks; the
D-split layer indexes panels by wave, so its read-side masks are not elided below it. dB is stored element-wise (the row pitch and `seqlen_k` are runtime values):
the W waves hold the same sums, so wave w stores the dS elements with `e % W == w`. `BUILT_TILES` and `_check_scope` refuse everything else (tile 384,
runtime axes) by name.

--- Resources --------------------------------------------------------------------

LDS is the layer's (`DSplitDq.lds_bytes`). Above tile `HOIST_ABOVE` (128) the SGPR pressure of the padded builds (the head-dim limits, two slab spans, the dB / bias / dropout addressing) is lowered
by two changes: the K / V buffer resources are built once before the loop (`DeviceOps.buf_rsrc`, `stage_kv(k_rsrc, v_rsrc)`) and dQ's slab base is derived after the loop instead of being carried through it. Without them
the padded no-dropout builds of tiles 160 .. 224 get a scratch frame (a dead private segment, no scratch access); the objects of tiles <= 128 are pinned and keep the original code. The compile hint `unroll-max-count = 1` is the SP-N skeleton's (`jobs/fp32-skel`): one loop body per object, so the
resource row is the loop's and not an LLVM unroll of it. No fast-math hints: the exchange order is the documented wave order and `+inf` LSE' must survive to `exp2`.

--- dBias reduction --------------------------------------------------------------

`dbias_reduce=True` (bias builds that store dB; every fp32 bias dQ stores dB) adds a second launch of `fmha_bwd_dbias_reduce_f32_gfx950`'s kernel after the dQ one,
so the object carries two kernel symbols: the dQ kernel and the reduce (`fmha_bwd_dbias_reduce_f32_gfx950_kernel_1`). The launcher's arguments gain the reduce launch's
(`DBias`, `reduce_batch`, `reduce_head`, `stride_ws_*`, `stride_out_*`; its `DB` is the dQ kernel's `DB`); they serve compilation only, since each kernel's kernarg is
packed from its own declaration. The gpu module is compiled under one set of LLVM options, this module's, so `unroll-max-count = 1` reaches the reduce kernel: its bytes
and resource row differ from the standalone reduce's and equal those of a standalone reduce built under this module's compile options.
"""


from kernels.attention_bwd_cdna import (  # noqa: E402
    fmha_bwd_dbias_reduce_f32_gfx950 as dbias_reduce_module,
)
from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha  # noqa: E402
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    exp2_wait_state,
)  # noqa: E402
from kernels.attention_bwd_cdna.fmha_f32_common_gfx950 import (
    ColMask,
    DeviceOps,
    Lane,
    Philox,
    gaddr,
    slab_span,
)  # noqa: E402
from kernels.attention_bwd_cdna.fmha_f32_dsplit_gfx950 import DSplitDq  # noqa: E402
from kernels.attention_bwd_cdna.fmha_tuning_bwd_dq_f32_gfx950 import (
    FmhaInputMetadata,
    bwd_dq_knobs,
)  # noqa: E402
from kernels.attention_bwd_cdna.gfx950_standalone import dualwave  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import const_expr, range_constexpr, rocdl  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402
from flydsl.expr.utils.arith import _to_raw as as_mlir_value  # noqa: E402

__all__ = [
    "BUILT_TILES",
    "KERNEL_NAME",
    "build_fmha_bwd_dq_f32_gfx950_module",
    "build_fmha_bwd_dq_f32_gfx950_module_primary",
]

KERNEL_NAME = "fmha_bwd_dq_f32_gfx950_kernel"

# Tiles this module builds (the D-split dQ layer of the table, both geometries).
BUILT_TILES = (32, 64, 96, 128, 160, 192, 224, 256, 512)

HOIST_ABOVE = 128

_COMPILE_HINTS = {"llvm_options": {"unroll-max-count": 1}}


def _carried(values, count):
    """An `scf.for`'s carried values as a list of `count`: a loop with exactly one carried value (one accumulator, `NACC` = 1) hands it back unwrapped."""
    if const_expr(isinstance(values, (list, tuple))):
        return list(values)
    if const_expr(count != 1):
        raise AssertionError(
            f"expected {count} carried values, got one unwrapped value"
        )
    return [values]


def _check_scope(meta, knobs, runtime_axes):
    """Refuse what this module does not compute, at the decision rather than at an address."""
    if runtime_axes:
        raise NotImplementedError(
            f"runtime axes {sorted(runtime_axes)}: the fp32 dQ kernel builds baked forms only"
        )
    if knobs.block_dmodel not in BUILT_TILES:
        raise NotImplementedError(
            f"fp32 dQ builds tiles {BUILT_TILES}, got {knobs.block_dmodel}"
        )
    if not knobs.padded_head and meta.head_dim != knobs.block_dmodel:
        raise NotImplementedError(
            "an exact fp32 dQ build needs head_dim == tile; a smaller head dim is a padded build"
        )
    if knobs.store_db and not meta.bias:
        raise NotImplementedError("store_db needs a bias build")


def build_fmha_bwd_dq_f32_gfx950_module_primary(
    meta, knobs, runtime_axes=(), dbias_reduce=False
):
    """Build the fp32 dQ kernel for a resolved (meta, knobs) pair; `dbias_reduce` also launches the fp32 dBias reduce (see the module docstring)."""
    runtime_axes = frozenset(runtime_axes)
    if knobs.num_waves is None:
        raise ValueError(
            "knobs must be resolved: call `bwd_dq_knobs(arch, ...).resolve(meta)` first"
        )
    _check_scope(meta, knobs, runtime_axes)
    fam = knobs.family()
    layer = DSplitDq(fam)
    W, RB, BQ, DSU = fam.waves, fam.rb, fam.bq, fam.dsu
    BLOCK = fam.block
    CAUSAL, BIAS, DROP, STORE_DB, PADDED = (
        bool(meta.causal),
        bool(meta.bias),
        bool(meta.dropout),
        bool(knobs.store_db),
        bool(knobs.padded_head),
    )
    TILE, FLOOR = fam.tile, int(knobs.hdim_qk_floor)
    if dbias_reduce and not (BIAS and STORE_DB):
        raise ValueError("dbias_reduce: a bias build that stores dB only")
    REDUCE_VEC = dbias_reduce_module.VEC
    REDUCE_BLOCK = dbias_reduce_module.BLOCK_THREADS
    reduce_kernel = (
        dbias_reduce_module.build_fmha_bwd_dbias_reduce_f32_gfx950_module_primary(
            "fp32"
        ).kernel
        if dbias_reduce
        else None
    )
    VO_FLOOR = (
        FLOOR if (meta.head_dim_v or TILE) > FLOOR else 0
    )  # the dO / V side promises its extent only when the V head dim lies above the floor
    NACC = (
        DSU if BQ == 16 else DSU // 2
    )  # accumulators per lane: one V4 per 16-column panel (block_m 16) or one V16 per 32-column unit (block_m 32)
    ACC_ELEMS = 4 if BQ == 16 else 16
    HOIST = (
        TILE > HOIST_ABOVE
    )  # SGPR pressure: K / V buffer resources built once, dQ's slab base derived past the loop (the objects of tiles <= HOIST_ABOVE are pinned without it)
    LDS_WORDS = layer.lds_words
    _cache_tag = (
        KERNEL_NAME,
        fam.tile,
        fam.bq,
        fam.sched,
        fam.waves,
        CAUSAL,
        BIAS,
        DROP,
        STORE_DB,
        PADDED,
        FLOOR,
        VO_FLOOR,
        bool(dbias_reduce),
    )

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
    def fmha_bwd_dq_f32_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
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
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
        rt_flags: fx.Int32,
    ):
        I32 = fx.Int32
        I64 = fx.Int64
        tid = I32(fx.thread_idx.x)
        wave = I32(rocdl.readfirstlane(T.i32, as_mlir_value(tid // I32(64))))
        lane = tid - wave * I32(64)
        L = Lane.of(lane, wave)
        q_head = fx.Index(fx.block_idx.x)
        q0 = I32(fx.block_idx.y) * I32(BQ)
        z = I32(fx.block_idx.z)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        f32 = lambda P, align: fx.recast_iter(_ptr_ty(align), P)  # noqa: E731
        gptrs = {
            "Q": f32(Q, 16),
            "K": f32(K, 16),
            "V": f32(V, 16),
            "DO": f32(DO, 16),
            "DQ": f32(DQ, 16),
            "LSE": f32(LSE, 4),
            "DELTA": f32(Delta, 4),
        }
        if const_expr(BIAS):
            gptrs["BIAS"] = f32(B, 4)
        if const_expr(STORE_DB):
            gptrs["DB"] = f32(DB, 4)
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
        q_last = (q_len > I32(0)).select(q_len - I32(1), I32(0))
        k_last = (k_len > I32(0)).select(k_len - I32(1), I32(0))

        # slab bases (elements): Q, dO, dQ of this query head; K, V of its KV head
        kv_head = q_head // (fx.Index(num_head_q) // fx.Index(num_head_k))
        q_base = (
            I64(q_batch) * stride_q_batch
            + I64(q_head) * stride_q_head
            + I64(q_row_off) * stride_q_seq
        )
        do_base = (
            I64(q_batch) * stride_do_batch
            + I64(q_head) * stride_do_head
            + I64(q_row_off) * stride_do_seq
        )
        if const_expr(not HOIST):
            dq_base = (
                I64(q_batch) * stride_dq_batch
                + I64(q_head) * stride_dq_head
                + I64(q_row_off) * stride_dq_seq
            )
        k_base = (
            I64(k_batch) * stride_k_batch
            + I64(kv_head) * stride_k_head
            + I64(k_row_off) * stride_k_seq
        )
        v_base = (
            I64(k_batch) * stride_v_batch
            + I64(kv_head) * stride_v_head
            + I64(k_row_off) * stride_v_seq
        )
        lse_base, lse_pitch = fmha.lse_row_addressing(
            varlen_bits,
            q_batch,
            q_head,
            fx.Index(num_head_q),
            fx.Index(tokens),
            q_row_off,
        )

        # resident Q, dO slices of the lane's q rows (clamped to the last row), LSE' and Delta of the lane's q column
        def q_off(s):
            return gaddr(
                ops, q_base, fx.min(q0 + I32(16 * s) + L.m, q_last), stride_q_seq, 0
            )

        def do_off(s):
            return gaddr(
                ops, do_base, fx.min(q0 + I32(16 * s) + L.m, q_last), stride_do_seq, 0
            )

        # padded head dim: the Q / K / dQ side ends at `hdim_qk`, the dO / V side at `hdim_vo` (element-wise masks: the head dims are runtime values)
        if const_expr(PADDED):
            mq = ColMask(I32(hdim_qk), False, FLOOR)
            mv = ColMask(I32(hdim_vo), False, VO_FLOOR)
            k_span = slab_span(ops, k_len, stride_k_seq, I32(hdim_qk))
            v_span = slab_span(ops, k_len, stride_v_seq, I32(hdim_vo))
            k_rsrc = v_rsrc = None
            if const_expr(HOIST):
                k_rsrc, v_rsrc = ops.buf_rsrc("K", k_base, k_span), ops.buf_rsrc(
                    "V", v_base, v_span
                )
        else:
            mq = mv = k_span = v_span = k_rsrc = v_rsrc = None

        Q_res, DO_res = layer.load_resident(ops, L, q_off, do_off, mq, mv)
        lse, dlt = layer.row_stats(ops, L, q0, q_last, lse_base, lse_pitch)
        layer.zero_pads(ops, L)

        c_scale_log2e = sm_scale * fx.Float32(dualwave._LOG2E)
        c_neg_log2e = fx.Float32(-dualwave._LOG2E)
        if const_expr(BIAS):
            c_log2e = fx.Float32(dualwave._LOG2E)
        neg_lse = lse * c_neg_log2e
        q_valid = (q0 + layer.q_row(L)) < q_len
        # a workgroup whose q block starts past this sequence runs no stage (uniform) and stores nothing
        nstage = fx.Index((q0 < q_len).select((k_len + I32(15)) // I32(16), I32(0)))
        zero = fx.Float32(0.0)
        if const_expr(CAUSAL or BIAS or STORE_DB or DROP):
            q_row = q0 + layer.q_row(L)  # the lane's q row (sequence-local)
        if const_expr(CAUSAL):
            # visible iff row - wl <= col <= row + wr (the window resolved against this sequence's lengths); the loop walks the KV stages the
            # block's rows can reach: [max(q0 - wl, 0) // 16, ceil((q0 + block_m + wr) / 16)), at most the sequence's stages
            wl, wr = fmha.resolve_window(window_left, window_right, q_len, k_len)
            end = q0 + I32(BQ) + wr
            end = (end > I32(0)).select(end, I32(0))
            hi = (end + I32(15)) // I32(16)
            nkv = (k_len + I32(15)) // I32(16)
            hi = (hi < nkv).select(hi, nkv)
            hi = (q0 < q_len).select(hi, I32(0))
            first = q0 - wl
            first = (first > I32(0)).select(first, I32(0))
            lo = first // I32(16)
            lo = (lo < hi).select(lo, hi)
            loop_lo = fx.Index(lo)
            loop_hi = fx.Index(hi)
            col_lo = q_row - wl
            col_hi = q_row + wr
        if const_expr(BIAS):
            # the bias input is read through its strides (a broadcast dim has stride 0), in Q's batch and row layout
            b_base = (
                I64(q_batch) * stride_b_batch
                + I64(q_head) * stride_b_head
                + I64(q_row_off) * stride_b_seq_q
            )
            b_row = gaddr(ops, b_base, fx.min(q_row, q_last), stride_b_seq_q, 0)
        if const_expr(STORE_DB):
            # dB = dS, fp32, in Q's batch and row layout; an all-zero stride triple means dB is not wanted (a null DB)
            db_base = (
                I64(q_batch) * stride_db_batch
                + I64(q_head) * stride_db_head
                + I64(q_row_off) * stride_db_seq_q
            )
            db_row = gaddr(ops, db_base, q_row, stride_db_seq_q, 0)
            db_on = (
                (stride_db_batch != I64(0))
                | (stride_db_head != I64(0))
                | (stride_db_seq_q != I64(0))
            )
        if const_expr(DROP):
            # the forward's mask, regenerated: plane = (sequence, query head), grid max_seqlen_q x max_seqlen_k; threshold / scale are launch values
            rng = Philox.for_arch("gfx950")
            seed = fmha.philox_seed_value(philox_seed_ptr)
            offset = fmha.philox_offset_base(philox_offset1, philox_offset2)
            plane = z * I32(num_head_q) + I32(q_head)
            plane_base, row_stride = rng.grid_plane(
                offset, plane, max_seqlen_q, max_seqlen_k
            )

        init_args = [ops.zero(ACC_ELEMS) for _ in range_constexpr(NACC)]
        results = init_args
        if const_expr(CAUSAL):
            loop_range = (loop_lo, loop_hi)
        else:
            loop_range = (fx.Index(0), nstage)
        for it, st in range(loop_range[0], loop_range[1], fx.Index(1), init=init_args):
            acc = _carried(st, NACC)
            k0 = I32(it) * I32(16)
            layer.stage_kv(
                ops,
                L,
                k0,
                k_last,
                k_base,
                v_base,
                stride_k_seq,
                stride_v_seq,
                mq,
                mv,
                k_span,
                v_span,
                k_rsrc,
                v_rsrc,
            )
            ops.wait_dma()
            S, Pd = layer.scores(ops, L, Q_res, DO_res, mq, mv)

            # (S sum, dP sum) per group of four KV rows and the KV row of each element: block_m 16 one group, block_m 32 the two `h` pairs
            if const_expr(BQ == 16):
                groups = [layer.ds_operands16(ops, L, S, Pd)]
                kv_rows = [[k0 + L.g + I32(4 * i) for i in range_constexpr(4)]]
            else:
                groups = layer.ds_operands32(ops, L, S, Pd)
                kv_rows = [
                    [
                        k0 + I32(2 * h) + L.lane // I32(32) + I32(4 * i)
                        for i in range_constexpr(4)
                    ]
                    for h in range_constexpr(2)
                ]

            # P = exp2(c S - log2e LSE'), dS = P (dP - Delta); q rows and KV rows past the sequence ends are zero
            if const_expr(BIAS):
                # the stage's bias elements, in the element order of the dS list below
                bias_v = layer.load_elems(
                    ops, "BIAS", b_row, [c for rows in kv_rows for c in rows], k_last
                )
            # (bias: `bias * log2e` joins the scaled score, ahead of the exponential)
            exp_args = []
            for h, (ss, _) in enumerate(groups):
                for i in range_constexpr(4):
                    x = ops.elem(ss, i) * c_scale_log2e + neg_lse
                    if const_expr(BIAS):
                        x = x + bias_v[4 * h + i] * c_log2e
                    exp_args.append(dualwave.rocdl.exp2(T.f32, as_mlir_value(x)))
            expo = exp2_wait_state(exp_args)
            if const_expr(DROP):
                # u32 randoms of the lane's q row: call i covers KV columns k0 + 4 i .. +3; the lane's element sits at slot col % 4
                calls = [
                    rng.u32(
                        seed,
                        rng.grid_offset(
                            plane_base, row_stride, I64(q_row), k0 + I32(4 * i)
                        ),
                    )
                    for i in range_constexpr(4)
                ]
            dS = []
            for h, (_, dp) in enumerate(groups):
                for i in range_constexpr(4):
                    valid = q_valid & (kv_rows[h][i] < k_len)
                    if const_expr(CAUSAL):
                        valid = (
                            valid
                            & (kv_rows[h][i] >= col_lo)
                            & (kv_rows[h][i] <= col_hi)
                        )
                    p = valid.select(fx.Float32(expo[4 * h + i]), zero)
                    dpv = ops.elem(dp, i)
                    if const_expr(DROP):
                        if const_expr(BQ == 16):
                            u = (L.g == I32(0)).select(
                                calls[i][0],
                                (L.g == I32(1)).select(
                                    calls[i][1],
                                    (L.g == I32(2)).select(calls[i][2], calls[i][3]),
                                ),
                            )
                        else:
                            u = (L.lane // I32(32) == I32(0)).select(
                                calls[i][2 * h], calls[i][2 * h + 1]
                            )
                        keep = fx.Int32(u) > I32(idropout_p)
                        dpv = keep.select(dpv * dropout_scale, zero)
                    dS.append(valid.select(p * (dpv - dlt), zero))

            if const_expr(STORE_DB):
                db_cols = [c for rows in kv_rows for c in rows]
                layer.store_ds(
                    ops, L, "DB", db_row, q_valid & db_on, db_cols, dS, k_len
                )

            if const_expr(BQ == 16):
                layer.accumulate16(ops, L, acc, dS)
            else:
                layer.accumulate32(ops, L, acc, dS)
            ops.barrier()
            results = yield acc

        # dS is the gradient of the scaled logits: sm_scale belongs to dQ
        results = _carried(results, NACC)
        dq = [
            ops.vec(
                [ops.elem(results[i], j) * sm_scale for j in range_constexpr(ACC_ELEMS)]
            )
            for i in range_constexpr(NACC)
        ]
        if const_expr(HOIST):
            _, dq_row_off, dq_batch = fmha.decode_addressing(
                varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z
            )
            dq_base = (
                I64(dq_batch) * stride_dq_batch
                + I64(q_head) * stride_dq_head
                + I64(dq_row_off) * stride_dq_seq
            )
        layer.store(ops, L, q0, q_len, dq_base, stride_dq_seq, dq, mq)

    @flyc.jit
    def launch_fmha_bwd_dq_f32_gfx950(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        Bias: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
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
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
        rt_flags: fx.Int32,
        DBias: fx.Pointer,
        reduce_batch: fx.Int32,
        reduce_head: fx.Int32,
        stride_ws_batch: fx.Int64,
        stride_ws_head: fx.Int64,
        stride_ws_seq: fx.Int64,
        stride_out_batch: fx.Int64,
        stride_out_head: fx.Int64,
        stride_out_seq: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        num_q_blocks = (fx.Index(max_seqlen_q) + fx.Index(BQ - 1)) // fx.Index(BQ)
        # the grid's z extent counts sequences: `num_seqlens` for a packed batch slot, else `batch_size`
        bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        fmha_bwd_dq_f32_gfx950_kernel(
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DB,
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
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
            stride_db_batch,
            stride_db_head,
            stride_db_seq_q,
            rt_flags,
        ).launch(
            grid=(fx.Index(num_head_q), num_q_blocks, bs_idx),
            block=(BLOCK, 1, 1),
            stream=stream,
        )
        if const_expr(dbias_reduce):
            nvec = (fx.Index(max_seqlen_k) + fx.Index(REDUCE_VEC - 1)) // fx.Index(
                REDUCE_VEC
            )
            nblk = (
                fx.Index(max_seqlen_q) * nvec + fx.Index(REDUCE_BLOCK - 1)
            ) // fx.Index(REDUCE_BLOCK)
            out_b = (reduce_batch != fx.Int32(0)).select(fx.Int32(1), batch_size)
            out_h = (reduce_head != fx.Int32(0)).select(fx.Int32(1), num_head_q)
            reduce_kernel(
                DB,
                DBias,
                batch_size,
                num_head_q,
                max_seqlen_q,
                max_seqlen_k,
                reduce_batch,
                reduce_head,
                stride_ws_batch,
                stride_ws_head,
                stride_ws_seq,
                stride_out_batch,
                stride_out_head,
                stride_out_seq,
            ).launch(
                grid=(nblk, fx.Index(out_b) * fx.Index(out_h), 1),
                block=(REDUCE_BLOCK, 1, 1),
                stream=stream,
            )

    def _launch(*args, **kwargs):
        return flyc.compile(launch_fmha_bwd_dq_f32_gfx950, *args, **kwargs)

    _launch.traits = fam
    _launch.knobs = knobs
    _launch.cache_tag = _cache_tag
    _launch.lds_bytes = layer.lds_bytes
    _launch.jit = launch_fmha_bwd_dq_f32_gfx950
    return _launch


def build_fmha_bwd_dq_f32_gfx950_module(arch="gfx950", **kwargs):
    """Keyword front end: name a problem, get the table's wave count."""
    from dataclasses import fields as _fields

    meta_fields = {f.name for f in _fields(FmhaInputMetadata)}
    meta = FmhaInputMetadata(**{k: v for k, v in kwargs.items() if k in meta_fields})
    knob_kwargs = {k: v for k, v in kwargs.items() if k not in meta_fields}
    return build_fmha_bwd_dq_f32_gfx950_module_primary(
        meta, bwd_dq_knobs(arch, **knob_kwargs).resolve(meta)
    )
