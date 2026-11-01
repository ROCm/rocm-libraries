# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); its sources are listed below.
#
# Provenance carried over from the sweep snapshot:
# Derived from: AITER aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py @ 8253efc4059516f492df597b80ce5311e9fda8bd
# via the gfx11 backward dQ `fmha_bwd_dq_gfx11.py` (the runtime ABI, masks, bias / dB, dropout, varlen / GQA and padded-head semantics) and the
# VALU fp32 skeleton (the loop structure and LDS idioms). The loop dataflow follows AOTriton's `bwd_kernel_dq` (reference only).
# compiler: flydsl 0.3.4

"""FlyDSL fp32 backward dQ (and dB) for `gfx11-generic`, on the VALU (RDNA has no fp32 WMMA; Triton lowers an fp32 `tl.dot` to FMAs, and so does this kernel).

    P  = exp2(qk_scale * S - lse2')       S  = Q.K^T,  qk_scale = sm_scale * log2e,  lse2' = LSE' * log2e (LSE' = +inf for an empty row)
    dP = dO . V^T
    dS = P * (dP - delta)                 delta = rowsum(dO * O) from prep
    dQ = sm_scale * sum_j dS . K          dB = dS (bias builds; unscaled, fp32)

ABI = the gfx11 16-bit dQ launcher (`fmha_bwd_dq_gfx11.py`): Q, K, V, Bias, DO, DQ, DB, LSE, Delta, ..., rt_flags, stream; every tensor is fp32, strides in
4-byte elements, unit column stride. A `dump=1` object has one extra `dump: fx.Pointer` immediately before `stream` and is a separate nested launcher.

Algorithm (one workgroup per (query head, `block_m` query rows, sequence)): the workgroup owns `block_m` query rows and loops the kv axis in `block_n`-row
tiles. A kv step is

    (1) S and dP: one (q, kv) pair per thread slot (`pairs = block_m * block_n / T`), accumulated over the head dim in `D_CHUNK`-wide fp32 LDS slabs of Q, dO, K
        and V (row pitch D_CHUNK + 1);
    (2) P and dS per pair (scale, bias, visibility, exp2, dropout), dS to LDS; the dB element of a pair is stored by the pair's one thread;
    (3) the owner accumulators (thread -> row `tid / TPR`, `D_CHUNK / TPR` columns per slab) += dS . K over the kv tile, slab by slab.

Every accumulation is fp32 in a fixed order (no atomics, no cross-thread combine), so dQ and dB are bitwise repeatable; `sm_scale` is applied once at the store.

Rows past `seqlen_q` read zero Q/dO rows, `LSE' = +inf` and `Delta = 0`: P = dS = 0. A kv row past `seqlen_k` reads zeros and its P is selected to 0.
Visibility (one rule for every class): plain / bias keep kv `< seqlen_k`; causal keeps `q - window_left <= kv <= q + window_right`, `kv < seqlen_k`
(TOPLEFT / BOTRIGHT resolved per sequence, `fmha.resolve_window`), and the kv loop walks only the tile range the Q block can see. P is selected to 0 after the
exponential, so no -inf is formed before `exp2`.

Padded head (`padded=1`): Q, K and the dQ store use `hdim_qk <= tile`, V and dO use `hdim_vo <= tile`; a column past its runtime extent loads 0 and the dQ store is
element-wise and guarded per column, so the region past `hdim_qk` (NaN-poisoned by the caller) is never read or written; a straddling chunk needs no special case.

Class `bias`: `score = sm_scale * S + bias` (natural units; the exponent is `(score - LSE') * log2e`); `bias[batch, head, q, kv]` strides `stride_b_*` (zero = broadcast; `stride_b_seq_q == 0` is a row-broadcast bias),
read per element, a kv column past `seqlen_k` reads column 0 and a q row past `seqlen_q` the Q tile's first row. dB (`store_db` builds only):
`dB[batch, head, q, kv] = dS`, stored only when `stride_db_seq_q != 0`, the q row is `< seqlen_q` and the kv column `< seqlen_k` (the null form: all dB strides
zero never touches the DB pointer). One thread owns each (q, kv) pair of a workgroup and workgroups own disjoint query rows, so every dB element has one writer.

Dropout (`meta.dropout`): `dP <- keep ? dP * dropout_scale : 0` on the f32 dP before dS (P stays undropped; Delta already carries the mask through the forward's O).
The mask is the shared stream of `philox.py` (`Philox.for_arch(arch)`; plane = `block_idx.z * num_head_q + head_q`, grid extents the max lengths, element (q, kv) at
counter `grid_offset(.., q, kv)` slot `kv % 4`), as the 16-bit kernels and the forward: a varlen mask is a function of the element coordinates and the two maxima.
Varlen and GQA are as the 16-bit dQ: the Q and K sides decode independently, LSE and Delta share one HT or TH layout, query head `h` reads KV head `h // (Hq / Hk)`.

Dump (`dump=1`, debug; version 2, the fp32 pair-ordered layout, not the version-1 WMMA lane layout): workgroup (0, 0, 0) stores, at its kv iterations 0, 1 and last,
S, P, dP, dS as `[block_m][block_n]` f32 and the dQ accumulators after the iteration as `[block_m][tile]` f32 (columns past `hdim_qk` are unspecified) into `dump`:
header of 64 u32 words (magic, version, kernel, block_m, block_n, n_iter, hdim_qk, 0, kv block of iteration 0, 1, last; -1 when absent), then slots 0 / 1 / last of
`4 * block_m * block_n + block_m * tile` words. S is -inf and dP 0 outside the visible region.
"""

from types import SimpleNamespace as _WireArgs


from kernels.attention_bwd_rdna import fmha_common_gfx1201 as fmha
from kernels.attention_bwd_rdna.fmha_common_gfx11 import _if_then
from kernels.attention_bwd_rdna.fmha_tuning_bwd_dq_f32_gfx11 import (
    BwdDqF32KnobsGfx11,
    bwd_dq_knobs,
)  # noqa: F401
from kernels.attention_rdna4.philox import Philox

from kernels.attention_bwd_rdna import flyc_polyfill as common_utils
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf as _scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

KERNEL_NAME = "fmha_bwd_dq_f32_gfx11_kernel"
_LOG2E = 1.4426950408889634
WARP_SIZE = 32
DUMP_HEADER_WORDS = 64
DUMP_MAGIC = 0x52444E41
DUMP_VERSION = 2
DUMP_KERNEL_DQ = 1

# fast_fp_math keeps the FMA contraction of the fp32 VALU skeleton; the +inf LSE' of an empty row goes through `fmha.FastMath("noninf")` (no `ninf` flag).
_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


def build_fmha_bwd_dq_f32_gfx11_module_primary(meta, knobs, runtime_axes=()):
    """Build the gfx11 fp32 dQ kernel for `meta` (`variant_gfx11.Meta`) and its resolved knob record (`bwd_dq_knobs`); returns the host launch function
    closing over the one `JitFunction`."""
    runtime_axes = frozenset(runtime_axes)
    if runtime_axes:
        raise NotImplementedError(
            f"runtime axes {sorted(runtime_axes)}: gfx11 builds dropout baked (dform=bk) only"
        )
    if not knobs.legal:
        raise ValueError(f"the knob record is not legal: {knobs.why}")
    if (meta.dtype, meta.tile) != (knobs.dtype, knobs.tile):
        raise ValueError(
            f"meta ({meta.dtype}, {meta.tile}) and knobs ({knobs.dtype}, {knobs.tile}) disagree"
        )
    if meta.dtype != "fp32":
        raise ValueError(f"dtype {meta.dtype!r}: this kernel builds fp32 only")
    if meta.cls not in ("plain", "causal", "bias"):
        raise NotImplementedError(
            f"class {meta.cls!r}: the gfx11 fp32 dQ builds plain, causal and bias"
        )
    HD = meta.tile
    PADDED_HEAD = bool(meta.padded)
    if not PADDED_HEAD and not (meta.hdim_qk == meta.hdim_vo == HD):
        raise ValueError(
            f"an exact build needs hdim_qk == hdim_vo == tile ({HD}); got {meta.hdim_qk}, {meta.hdim_vo}"
        )
    if meta.hdim_qk > HD or meta.hdim_vo > HD:
        raise ValueError(f"hdim ({meta.hdim_qk}, {meta.hdim_vo}) exceeds the tile {HD}")

    BM = knobs.block_m
    BN = knobs.block_n
    NUM_WAVES = knobs.num_waves
    T = NUM_WAVES * WARP_SIZE
    DC = knobs.d_chunk
    DCP = DC + 1
    NCH = HD // DC
    TPR = T // BM
    J = DC // TPR
    NPAIR = BM * BN // T
    NACC = NCH * J
    assert HD % DC == 0 and T % BM == 0 and (BM * BN) % T == 0 and DC % TPR == 0, (
        HD,
        DC,
        T,
        BM,
        BN,
    )
    assert BM & (BM - 1) == 0 and BN & (BN - 1) == 0 and DC & (DC - 1) == 0, (
        BM,
        BN,
        DC,
    )
    BN_LOG2 = BN.bit_length() - 1
    CAUSAL = meta.cls == "causal"
    BIAS = meta.cls == "bias"
    DROPOUT = bool(meta.dropout)
    STORE_DB = BIAS and bool(knobs.store_db)
    LPT = CAUSAL and knobs.lpt_tile_order
    DUMP = knobs.dump
    PHILOX = Philox.for_arch(knobs.arch)
    fastmath = fmha.FastMath(knobs.fp_mode)
    # Exponent chain of a bias object: contract only. No `nnan`/`ninf` (a -inf bias entry must reach exp2) and no `reassoc` (it would let the
    # compiler undo "combine in natural units, scale by log2 e last").
    fastmath_e = fmha.FastMath("safe")
    fastmath_e.flags = fx.arith.FastMathFlags.contract

    # ---- LDS layout (f32 words) ----
    O_Q = 0
    O_DO = O_Q + BM * DCP
    O_K = O_DO + BM * DCP
    O_V = O_K + BN * DCP
    O_DS = O_V + BN * DCP
    LDS_WORDS = O_DS + BM * BN
    assert LDS_WORDS * 4 == knobs.lds_bytes, (LDS_WORDS * 4, knobs.lds_bytes)

    DUMP_TILE_WORDS = BM * BN
    DUMP_SLOT_WORDS = 4 * DUMP_TILE_WORDS + BM * HD

    _cache_tag = (
        KERNEL_NAME,
        HD,
        PADDED_HEAD,
        BM,
        BN,
        NUM_WAVES,
        knobs.waves_per_eu,
        DUMP,
    )
    if CAUSAL:
        _cache_tag += ("causal", LPT)
    if BIAS:
        _cache_tag += ("bias", STORE_DB)
    if DROPOUT:
        _cache_tag += ("dropout", PHILOX.width)

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Float32, LDS_WORDS, 16]

    def _gptr(ptr, ty=fx.Float32):
        return fx.recast_iter(
            fx.PointerType.get(
                elem_ty=ty.ir_type, address_space=fx.AddressSpace.Global, alignment=4
            ),
            ptr,
        )

    @flyc.jit
    def emit_dq_body(a):
        """The kernel body over one namespace `a` of the wire arguments (plus `a.dump` in a dump object)."""
        varlen_bits, num_seqlens, max_seqlen_q, max_seqlen_k = (
            a.varlen_bits,
            a.num_seqlens,
            a.max_seqlen_q,
            a.max_seqlen_k,
        )
        num_head_q, num_head_k, sm_scale = a.num_head_q, a.num_head_k, a.sm_scale
        f32_ty = ir.F32Type.get()

        q_p, k_p, v_p, do_p, dq_p = (
            _gptr(a.Q),
            _gptr(a.K),
            _gptr(a.V),
            _gptr(a.DO),
            _gptr(a.DQ),
        )
        lse_p, delta_p = _gptr(a.LSE), _gptr(a.Delta)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek().buf.ptr

        tid = fx.Int32(gpu.thread_idx.x)

        # ---- varlen prologue: VarlenBits -> sequence scalars ----
        z_i32 = fx.Int32(gpu.block_idx.z)
        seqlen_q, q_row_off, q_batch = fmha.decode_addressing(
            varlen_bits, 0, max_seqlen_q, a.seqinfo_q0, a.seqinfo_q1, z_i32
        )
        seqlen_k, k_row_off, k_batch = fmha.decode_addressing(
            varlen_bits, 8, max_seqlen_k, a.seqinfo_k0, a.seqinfo_k1, z_i32
        )
        lse_tokens = fmha.lse_token_pitch(
            varlen_bits, 0, max_seqlen_q, a.seqinfo_q0, a.seqinfo_q1, num_seqlens
        )

        head_q = fx.Index(gpu.block_idx.x)
        q_tile_idx = fx.Index(gpu.block_idx.y)
        if const_expr(LPT):
            # Longest-processing-time order: the heaviest (last) Q tile of the sequence is dispatched first; tiles past the sequence stay where they are.
            _n_qblk = (fx.Index(seqlen_q) + fx.Index(BM - 1)) // fx.Index(BM)
            q_tile_idx = fx.Index(
                _n_qblk - q_tile_idx - fx.Index(1)
                if q_tile_idx < _n_qblk
                else q_tile_idx
            )
        start_q = fx.Int32(q_tile_idx * fx.Index(BM))
        alive = start_q < seqlen_q
        head_k = head_q // (fx.Index(num_head_q) // fx.Index(num_head_k))

        # ---- element bases: (batch, head, row_off) of each tensor, 64-bit ----
        q_batch_v, k_batch_v = fx.Index(q_batch), fx.Index(k_batch)
        q_row_off_v, k_row_off_v = fx.Index(q_row_off), fx.Index(k_row_off)
        q_sseq, k_sseq, v_sseq = (
            fx.Index(a.stride_q_seq),
            fx.Index(a.stride_k_seq),
            fx.Index(a.stride_v_seq),
        )
        do_sseq, dq_sseq = fx.Index(a.stride_do_seq), fx.Index(a.stride_dq_seq)
        q_base = (
            q_batch_v * fx.Index(a.stride_q_batch)
            + head_q * fx.Index(a.stride_q_head)
            + q_row_off_v * q_sseq
        )
        do_base = (
            q_batch_v * fx.Index(a.stride_do_batch)
            + head_q * fx.Index(a.stride_do_head)
            + q_row_off_v * do_sseq
        )
        dq_base = (
            q_batch_v * fx.Index(a.stride_dq_batch)
            + head_q * fx.Index(a.stride_dq_head)
            + q_row_off_v * dq_sseq
        )
        k_base = (
            k_batch_v * fx.Index(a.stride_k_batch)
            + head_k * fx.Index(a.stride_k_head)
            + k_row_off_v * k_sseq
        )
        v_base = (
            k_batch_v * fx.Index(a.stride_v_batch)
            + head_k * fx.Index(a.stride_v_head)
            + k_row_off_v * v_sseq
        )

        sm_log2e = fastmath.mul(sm_scale, fx.Float32(_LOG2E))
        c_zero = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))

        def lds_ld(word):
            return fx.ptr_load(lds + fx.Int32(word), fx.Float32)

        def lds_st(word, val):
            fx.ptr_store(val, lds + fx.Int32(word))

        def stage(ptr, base, sseq, row0, rows, row_limit, col_limit, lds_off, chunk):
            """VRAM -> LDS: `rows` x DC slab of columns `chunk * DC ..` (row pitch DCP). A row at or past `row_limit` and a column at or past `col_limit` (None: no
            column mask) load 0; their addresses are clamped to element 0 of the row / sequence, so no read leaves the tensor.
            """
            for i in range_constexpr(rows * DC // T):
                idx = tid + fx.Int32(i * T)
                r = idx // fx.Int32(DC)
                c = idx % fx.Int32(DC)
                grow = row0 + r
                gcol = fx.Int32(chunk * DC) + c
                ok = grow < row_limit
                rr = ok.select(grow, fx.Int32(0))
                if const_expr(col_limit is not None):
                    okc = gcol < col_limit
                    cc = okc.select(gcol, fx.Int32(0))
                    ok = ok & okc
                else:
                    cc = gcol
                g = base + fx.Index(rr) * sseq + fx.Index(cc)
                val = fx.ptr_load(ptr + fx.Int64(g), fx.Float32)
                lds_st(lds_off + r * fx.Int32(DCP) + c, ok.select(val, c_zero))

        qk_lim = fx.Int32(a.hdim_qk) if PADDED_HEAD else None
        vo_lim = fx.Int32(a.hdim_vo) if PADDED_HEAD else None

        # ---- per-pair constants: the (q row, kv column) slot of each of this thread's pairs ----
        pair_m = [
            (tid + fx.Int32(i * T)) // fx.Int32(BN) for i in range_constexpr(NPAIR)
        ]
        pair_n = [
            (tid + fx.Int32(i * T)) % fx.Int32(BN) for i in range_constexpr(NPAIR)
        ]
        pair_q = [start_q + pair_m[i] for i in range_constexpr(NPAIR)]
        pair_in = [pair_q[i] < seqlen_q for i in range_constexpr(NPAIR)]

        # LSE' and Delta of each pair's row: an out-of-range row is an empty row (LSE' = +inf, Delta = 0), so P = dS = 0
        lse2 = []
        delta = []
        for i in range_constexpr(NPAIR):
            tok = common_utils.smin(q_row_off + pair_q[i], lse_tokens - fx.Int32(1))
            tok_row = fx.Index(common_utils.smax(tok, fx.Int32(0)))
            lse_off, _ = fmha.lse_row_addressing(
                varlen_bits, q_batch_v, head_q, num_head_q, lse_tokens, tok_row
            )
            lse_v = fx.ptr_load(lse_p + fx.Int64(lse_off), fx.Float32)
            delta_v = fx.ptr_load(delta_p + fx.Int64(lse_off), fx.Float32)
            if const_expr(BIAS):
                # natural: the bias exponent scales by log2 e last
                lse2.append(pair_in[i].select(lse_v, fx.Float32(float("inf"))))
            else:
                lse2.append(
                    fastmath.mul(
                        pair_in[i].select(lse_v, fx.Float32(float("inf"))),
                        fx.Float32(_LOG2E),
                    )
                )
            delta.append(pair_in[i].select(delta_v, c_zero))

        if const_expr(BIAS):
            b_p = _gptr(a.B)
            b_base = (
                q_batch_v * fx.Index(a.stride_b_batch)
                + head_q * fx.Index(a.stride_b_head)
                + q_row_off_v * fx.Index(a.stride_b_seq_q)
            )
            b_sseq = fx.Index(a.stride_b_seq_q)
        if const_expr(STORE_DB):
            db_p = _gptr(a.DB)
            db_base = (
                q_batch_v * fx.Index(a.stride_db_batch)
                + head_q * fx.Index(a.stride_db_head)
                + q_row_off_v * fx.Index(a.stride_db_seq_q)
            )
            db_sseq = fx.Index(a.stride_db_seq_q)
            db_requested = fx.Int64(a.stride_db_seq_q) != fx.Int64(0)
        if const_expr(DROPOUT):
            ph_seed = fmha.philox_seed_value(a.philox_seed_ptr)
            ph_off = fmha.philox_offset_base(a.philox_offset1, a.philox_offset2)
            ph_base, ph_stride = PHILOX.grid_plane(
                ph_off,
                z_i32 * fx.Int32(num_head_q) + fx.Int32(head_q),
                max_seqlen_q,
                max_seqlen_k,
            )
            drop_scale = fx.Float32(a.dropout_scale)
            drop_thr = fx.Int32(a.idropout_p)

        # ---- the kv range [kv_lo, kv_hi) the Q block walks, in rows (a multiple of BN) ----
        if const_expr(CAUSAL):
            # key c is visible to row q iff q - wl <= c <= q + wr and c < seqlen_k, the TOPLEFT / BOTRIGHT sentinels resolved against this sequence.
            # The clamps keep every derived i32 in range without changing the visible set.
            wl, wr = fmha.resolve_window(
                a.window_left, a.window_right, seqlen_q, seqlen_k
            )
            wl = common_utils.smin(
                common_utils.smax(wl, fx.Int32(0) - seqlen_k), seqlen_q
            )
            wr = common_utils.smin(
                common_utils.smax(wr, fx.Int32(0) - seqlen_q), seqlen_k
            )
            q_last = common_utils.smin(start_q + fx.Int32(BM), seqlen_q) - fx.Int32(1)
            blk_last = common_utils.sdiv_rd_pow2(seqlen_k - fx.Int32(1), BN)
            v_lo = common_utils.smax(
                common_utils.sdiv_rd_pow2(start_q - wl, BN), fx.Int32(0)
            )
            v_hi = common_utils.smin(
                blk_last, common_utils.sdiv_rd_pow2(q_last + wr, BN)
            )
            v_hi = common_utils.ssel(alive, v_hi, v_lo - fx.Int32(1))
            kv_lo_i32 = v_lo * fx.Int32(BN)
            kv_hi_i32 = common_utils.smax(
                (v_hi + fx.Int32(1)) * fx.Int32(BN), kv_lo_i32
            )
            pair_first = [
                common_utils.smax(pair_q[i] - wl, fx.Int32(0))
                for i in range_constexpr(NPAIR)
            ]
            pair_span = [
                common_utils.smax(
                    common_utils.smin(pair_q[i] + wr + fx.Int32(1), seqlen_k)
                    - pair_first[i],
                    fx.Int32(0),
                )
                for i in range_constexpr(NPAIR)
            ]
        else:
            kv_lo_i32 = fx.Int32(0)
            kv_hi_i32 = common_utils.ssel(
                alive,
                ((seqlen_k + fx.Int32(BN - 1)) >> fx.Int32(BN_LOG2))
                << fx.Int32(BN_LOG2),
                fx.Int32(0),
            )

        own_row = tid // fx.Int32(TPR)
        cpart = tid % fx.Int32(TPR)

        if const_expr(DUMP):
            dump_p = _gptr(a.dump)
            dump_i = _gptr(a.dump, fx.Int32)
            n_iter = (kv_hi_i32 - kv_lo_i32) >> fx.Int32(BN_LOG2)
            if const_expr(LPT):
                is_wg0 = (
                    (fx.Int32(gpu.block_idx.x) == fx.Int32(0))
                    & (fx.Int32(q_tile_idx) == fx.Int32(0))
                    & (z_i32 == fx.Int32(0))
                )
            else:
                is_wg0 = (
                    (fx.Int32(gpu.block_idx.x) == fx.Int32(0))
                    & (fx.Int32(gpu.block_idx.y) == fx.Int32(0))
                    & (z_i32 == fx.Int32(0))
                )
            kv_blk0 = kv_lo_i32 >> fx.Int32(BN_LOG2)
            hdr = [
                fx.Int32(v) for v in (DUMP_MAGIC, DUMP_VERSION, DUMP_KERNEL_DQ, BM, BN)
            ] + [
                n_iter,
                fx.Int32(meta.hdim_qk),
                fx.Int32(0),
                (n_iter >= fx.Int32(1)).select(kv_blk0, fx.Int32(-1)),
                (n_iter >= fx.Int32(2)).select(kv_blk0 + fx.Int32(1), fx.Int32(-1)),
                (n_iter >= fx.Int32(3)).select(
                    kv_blk0 + n_iter - fx.Int32(1), fx.Int32(-1)
                ),
            ]
            with _if_then(_scf.IfOp(fx.as_ir_value(is_wg0 & (tid == fx.Int32(0))))):
                for w in range_constexpr(len(hdr)):
                    fx.ptr_store(fx.Int32(hdr[w]), dump_i + fx.Int64(w))

        def step(kv0_i32, accs):
            """One kv tile at row `kv0_i32` (the tile's first kv row): returns the new owner accumulators."""
            s = [c_zero for _ in range_constexpr(NPAIR)]
            dp = [c_zero for _ in range_constexpr(NPAIR)]

            # (1) S = Q.K^T and dP = dO.V^T over the head dim, slab by slab
            for c in range_constexpr(NCH):
                gpu.barrier()
                stage(q_p, q_base, q_sseq, start_q, BM, seqlen_q, qk_lim, O_Q, c)
                stage(do_p, do_base, do_sseq, start_q, BM, seqlen_q, vo_lim, O_DO, c)
                stage(k_p, k_base, k_sseq, kv0_i32, BN, seqlen_k, qk_lim, O_K, c)
                stage(v_p, v_base, v_sseq, kv0_i32, BN, seqlen_k, vo_lim, O_V, c)
                gpu.barrier()
                for i in range_constexpr(NPAIR):
                    qm = pair_m[i] * fx.Int32(DCP)
                    kn = pair_n[i] * fx.Int32(DCP)
                    for d in range_constexpr(DC):
                        s[i] = s[i] + lds_ld(O_Q + qm + fx.Int32(d)) * lds_ld(
                            O_K + kn + fx.Int32(d)
                        )
                        dp[i] = dp[i] + lds_ld(O_DO + qm + fx.Int32(d)) * lds_ld(
                            O_V + kn + fx.Int32(d)
                        )

            # (2) P and dS per pair; dS to LDS, dB to global
            for i in range_constexpr(NPAIR):
                kv_col = kv0_i32 + pair_n[i]
                if const_expr(BIAS):
                    sc = fastmath_e.mul(s[i], sm_scale)
                    # added after the scale: `sm_scale * S + bias` (natural units); a column past seqlen_k reads column 0, a row past seqlen_q the tile's first row
                    b_row = pair_in[i].select(pair_q[i], start_q)
                    b_col = (kv_col < seqlen_k).select(kv_col, fx.Int32(0))
                    b_val = fx.ptr_load(
                        b_p
                        + fx.Int64(b_base + fx.Index(b_row) * b_sseq + fx.Index(b_col)),
                        fx.Float32,
                    )
                    sc = fastmath_e.add(sc, b_val)
                    ex = fastmath_e.mul(fastmath_e.sub(sc, lse2[i]), fx.Float32(_LOG2E))
                else:
                    sc = fastmath.mul(s[i], sm_log2e)
                    ex = fastmath.sub(sc, lse2[i])
                e = fx.Float32(rocdl.exp2(f32_ty, fx.as_ir_value(ex)))
                if const_expr(CAUSAL):
                    vis = fx.Uint32(kv_col - pair_first[i]) < fx.Uint32(pair_span[i])
                else:
                    vis = kv_col < seqlen_k
                p = vis.select(e, c_zero)
                dpi = dp[i]
                if const_expr(DROPOUT):
                    # dP <- keep ? dP * scale : 0, before dS. Element (q, kv) is slot `kv % 4` of the Philox call at `grid_offset(.., q, kv)`
                    w = PHILOX.u32(
                        ph_seed,
                        PHILOX.grid_offset(ph_base, ph_stride, pair_q[i], kv_col),
                    )
                    slot = kv_col & fx.Int32(3)
                    word = (slot == fx.Int32(0)).select(
                        fx.Int32(w[0]),
                        (slot == fx.Int32(1)).select(
                            fx.Int32(w[1]),
                            (slot == fx.Int32(2)).select(
                                fx.Int32(w[2]), fx.Int32(w[3])
                            ),
                        ),
                    )
                    keep = word > drop_thr
                    dpi = keep.select(fastmath.mul(dpi, drop_scale), c_zero)
                ds = fastmath.mul(p, fastmath.sub(dpi, delta[i]))
                lds_st(O_DS + pair_m[i] * fx.Int32(BN) + pair_n[i], ds)
                if const_expr(STORE_DB):
                    # one writer per element: this pair's thread; only a requested dB (`stride_db_seq_q != 0`), a live q row and a live kv column
                    with _if_then(
                        _scf.IfOp(
                            fx.as_ir_value(
                                db_requested & pair_in[i] & (kv_col < seqlen_k)
                            )
                        )
                    ):
                        fx.ptr_store(
                            ds,
                            db_p
                            + fx.Int64(
                                db_base
                                + fx.Index(pair_m[i] + start_q) * db_sseq
                                + fx.Index(kv_col)
                            ),
                        )
                if const_expr(DUMP):
                    it = (kv0_i32 - kv_lo_i32) >> fx.Int32(BN_LOG2)
                    slot_i = (it == fx.Int32(0)).select(
                        fx.Int32(0),
                        (it == fx.Int32(1)).select(fx.Int32(1), fx.Int32(2)),
                    )
                    do_dump = is_wg0 & (
                        (it == fx.Int32(0))
                        | (it == fx.Int32(1))
                        | ((it >= fx.Int32(2)) & (it == n_iter - fx.Int32(1)))
                    )
                    sbase = (
                        fx.Int64(DUMP_HEADER_WORDS)
                        + fx.Int64(slot_i) * fx.Int64(DUMP_SLOT_WORDS)
                        + fx.Int64(pair_m[i] * fx.Int32(BN) + pair_n[i])
                    )
                    ok = vis & pair_in[i]
                    sd = fastmath.mul(s[i], sm_scale)
                    if const_expr(BIAS):
                        sd = fastmath.add(sd, b_val)
                    with _if_then(_scf.IfOp(fx.as_ir_value(do_dump))):
                        fx.ptr_store(ok.select(sd, c_neg_inf), dump_p + sbase)
                        fx.ptr_store(p, dump_p + sbase + fx.Int64(DUMP_TILE_WORDS))
                        fx.ptr_store(
                            ok.select(dpi, c_zero),
                            dump_p + sbase + fx.Int64(2 * DUMP_TILE_WORDS),
                        )
                        fx.ptr_store(ds, dump_p + sbase + fx.Int64(3 * DUMP_TILE_WORDS))

            # (3) owner accumulators += dS . K over the kv tile, slab by slab
            new = list(accs)
            for c in range_constexpr(NCH):
                gpu.barrier()
                stage(k_p, k_base, k_sseq, kv0_i32, BN, seqlen_k, qk_lim, O_K, c)
                gpu.barrier()
                for m in range_constexpr(BN):
                    dsv = lds_ld(O_DS + own_row * fx.Int32(BN) + fx.Int32(m))
                    for j in range_constexpr(J):
                        col = cpart + fx.Int32(j * TPR)
                        aidx = c * J + j
                        new[aidx] = new[aidx] + dsv * lds_ld(
                            O_K + fx.Int32(m * DCP) + col
                        )

            if const_expr(DUMP):
                with _if_then(_scf.IfOp(fx.as_ir_value(do_dump))):
                    abase = (
                        fx.Int64(DUMP_HEADER_WORDS)
                        + fx.Int64(slot_i) * fx.Int64(DUMP_SLOT_WORDS)
                        + fx.Int64(4 * DUMP_TILE_WORDS)
                    )
                    for c in range_constexpr(NCH):
                        for j in range_constexpr(J):
                            col = fx.Int32(c * DC + j * TPR) + cpart
                            fx.ptr_store(
                                new[c * J + j],
                                dump_p + abase + fx.Int64(own_row * fx.Int32(HD) + col),
                            )
            return [fx.as_ir_value(v) for v in new]

        accs = [fx.as_ir_value(c_zero) for _ in range_constexpr(NACC)]
        loop_results = accs
        for kv_block_start, inner in range(
            fx.Index(kv_lo_i32), fx.Index(kv_hi_i32), BN, init=accs
        ):
            loop_results = yield step(
                fx.Int32(kv_block_start),
                [fx.Float32(inner[i]) for i in range_constexpr(NACC)],
            )
        # an scf.for with one carried value hands it back unwrapped
        if const_expr(NACC == 1):
            loop_results = [loop_results]

        # ---- scale and store dQ (sm_scale applied once): element-wise, so a column past hdim_qk is never written ----
        own_q = start_q + own_row
        q_live = own_q < seqlen_q
        for c in range_constexpr(NCH):
            for j in range_constexpr(J):
                col = fx.Int32(c * DC + j * TPR) + cpart
                cond = q_live
                if const_expr(PADDED_HEAD):
                    cond = cond & (col < qk_lim)
                with _if_then(_scf.IfOp(fx.as_ir_value(cond))):
                    val = fx.Float32(loop_results[c * J + j]) * sm_scale
                    fx.ptr_store(
                        val,
                        dq_p
                        + fx.Int64(dq_base + fx.Index(own_q) * dq_sseq + fx.Index(col)),
                    )

    # ---- kernel + launcher -------------------------------------------------------------------------------------------------------
    _value_attrs = (
        {"rocdl.waves_per_eu": knobs.waves_per_eu} if knobs.waves_per_eu else {}
    )

    @flyc.kernel(known_block_size=[T, 1, 1])
    def fmha_bwd_dq_f32_gfx11_kernel(
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
        emit_dq_body(
            _WireArgs(
                Q=Q,
                K=K,
                V=V,
                DO=DO,
                DQ=DQ,
                LSE=LSE,
                Delta=Delta,
                seqinfo_q0=seqinfo_q0,
                seqinfo_q1=seqinfo_q1,
                seqinfo_k0=seqinfo_k0,
                seqinfo_k1=seqinfo_k1,
                varlen_bits=varlen_bits,
                num_seqlens=num_seqlens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                window_left=window_left,
                window_right=window_right,
                philox_seed_ptr=philox_seed_ptr,
                philox_offset1=philox_offset1,
                philox_offset2=philox_offset2,
                idropout_p=idropout_p,
                dropout_scale=dropout_scale,
                num_head_q=num_head_q,
                num_head_k=num_head_k,
                hdim_qk=hdim_qk,
                hdim_vo=hdim_vo,
                sm_scale=sm_scale,
                stride_q_batch=stride_q_batch,
                stride_q_head=stride_q_head,
                stride_q_seq=stride_q_seq,
                stride_k_batch=stride_k_batch,
                stride_k_head=stride_k_head,
                stride_k_seq=stride_k_seq,
                stride_v_batch=stride_v_batch,
                stride_v_head=stride_v_head,
                stride_v_seq=stride_v_seq,
                stride_do_batch=stride_do_batch,
                stride_do_head=stride_do_head,
                stride_do_seq=stride_do_seq,
                stride_dq_batch=stride_dq_batch,
                stride_dq_head=stride_dq_head,
                stride_dq_seq=stride_dq_seq,
                B=B,
                DB=DB,
                stride_b_batch=stride_b_batch,
                stride_b_head=stride_b_head,
                stride_b_seq_q=stride_b_seq_q,
                stride_db_batch=stride_db_batch,
                stride_db_head=stride_db_head,
                stride_db_seq_q=stride_db_seq_q,
            )
        )

    @flyc.jit
    def launch_fmha_bwd_dq_f32_gfx11(
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
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        num_q_blocks = (fx.Index(max_seqlen_q) + BM - 1) // BM
        fmha_bwd_dq_f32_gfx11_kernel(
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
            value_attrs=_value_attrs,
        ).launch(
            grid=(num_head_q, num_q_blocks, bs_idx), block=(T, 1, 1), stream=stream
        )

    _entry = launch_fmha_bwd_dq_f32_gfx11

    if DUMP:

        @flyc.kernel(known_block_size=[T, 1, 1])
        def fmha_bwd_dq_f32_gfx11_dump_kernel(
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
            dump: fx.Pointer,
        ):
            emit_dq_body(
                _WireArgs(
                    Q=Q,
                    K=K,
                    V=V,
                    DO=DO,
                    DQ=DQ,
                    LSE=LSE,
                    Delta=Delta,
                    seqinfo_q0=seqinfo_q0,
                    seqinfo_q1=seqinfo_q1,
                    seqinfo_k0=seqinfo_k0,
                    seqinfo_k1=seqinfo_k1,
                    varlen_bits=varlen_bits,
                    num_seqlens=num_seqlens,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    window_left=window_left,
                    window_right=window_right,
                    philox_seed_ptr=philox_seed_ptr,
                    philox_offset1=philox_offset1,
                    philox_offset2=philox_offset2,
                    idropout_p=idropout_p,
                    dropout_scale=dropout_scale,
                    num_head_q=num_head_q,
                    num_head_k=num_head_k,
                    hdim_qk=hdim_qk,
                    hdim_vo=hdim_vo,
                    sm_scale=sm_scale,
                    stride_q_batch=stride_q_batch,
                    stride_q_head=stride_q_head,
                    stride_q_seq=stride_q_seq,
                    stride_k_batch=stride_k_batch,
                    stride_k_head=stride_k_head,
                    stride_k_seq=stride_k_seq,
                    stride_v_batch=stride_v_batch,
                    stride_v_head=stride_v_head,
                    stride_v_seq=stride_v_seq,
                    stride_do_batch=stride_do_batch,
                    stride_do_head=stride_do_head,
                    stride_do_seq=stride_do_seq,
                    stride_dq_batch=stride_dq_batch,
                    stride_dq_head=stride_dq_head,
                    stride_dq_seq=stride_dq_seq,
                    B=B,
                    DB=DB,
                    stride_b_batch=stride_b_batch,
                    stride_b_head=stride_b_head,
                    stride_b_seq_q=stride_b_seq_q,
                    stride_db_batch=stride_db_batch,
                    stride_db_head=stride_db_head,
                    stride_db_seq_q=stride_db_seq_q,
                    dump=dump,
                )
            )

        @flyc.jit
        def launch_fmha_bwd_dq_f32_gfx11_dump(
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
            dump: fx.Pointer,
            stream: fx.Stream = fx.Stream(None),
        ):
            _ = _cache_tag
            bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
            num_q_blocks = (fx.Index(max_seqlen_q) + BM - 1) // BM
            fmha_bwd_dq_f32_gfx11_dump_kernel(
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
                dump,
                value_attrs=_value_attrs,
            ).launch(
                grid=(num_head_q, num_q_blocks, bs_idx), block=(T, 1, 1), stream=stream
            )

        _entry = launch_fmha_bwd_dq_f32_gfx11_dump

    def _launch(*packed):
        """`packed` = the launcher's arguments in wire order (the harness's `gate2.arg_values` order), stream last."""
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return _entry(*packed)

    _launch.knobs = knobs
    _launch.block_m = BM
    _launch.cache_tag = _cache_tag
    return _launch
