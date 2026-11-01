# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); its sources are listed below.
#
# Provenance carried over from the sweep snapshot:
# Adapted from: https://github.com/ROCm/aiter
#   commit:   8253efc4059516f492df597b80ce5311e9fda8bd
#   path:     aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py
# via the gfx11 forward `attention_rdna/flash_attn_func_rdna.py` (ROCm/rocm-libraries PR #14111 @ d850254d45e),
# and `fmha_bwd_dq_gfx950.py` (the runtime ABI). The loop dataflow follows AOTriton's gfx1201 `bwd_kernel_dq` port
# (`fmha_bwd_dq_gfx1201_kernel.py`, MIT; reference only), re-expressed over `rdna_layout_gfx11`.

"""FlyDSL backward dQ (and dB) for the generic RDNA targets (`gfx11-generic`; the WMMA ABI switch also covers `gfx12-generic`).

    P  = exp2(qk_scale * S - lse2')       S  = Q.K^T,  qk_scale = sm_scale * log2e,  lse2' = LSE' * log2e (LSE' = +inf for an empty row)
    dP = dO . V^T
    dS = P * (dP - delta)                 delta = rowsum(dO * O) from prep
    dQ = sm_scale * sum_j dS . K          dB = dS (bias builds; unscaled, rounded to the 16-bit I/O type like the GEMM3 operand)

ABI = the gfx950 dQ launcher (`tools/test_abi_parity.py` enforces it): Q, K, V, Bias, DO, DQ, DB, LSE, Delta, ..., rt_flags, stream. The launcher
keeps its gfx950 name. A `dump=1` object has one extra `dump: fx.Pointer` immediately before `stream` and is a separate nested launcher.

Orientation (`rdna_layout_gfx11.KV_ROWS`, S^T form): every accumulator tile has rows = kv, columns = q, so LSE' and Delta are per lane (column
`l16`), one value for all 8 elements:

    GEMM1  S^T[kv][q]  = K[kv][:] . Q[q][:]      A = K (LDS, row-major), B = Q (register preload)
    GEMM2  dP^T[kv][q] = V[kv][:] . dO[q][:]     A = V (LDS, row-major), B = dO (register preload)
    GEMM3  dQ^T[d][q] += K^T[d][kv] . dS^T[kv][q]   A = K^T (`kt_lds_layout`), B = dS^T through `acc_to_operand` (xor-16 exchange on gfx11)

The dQ accumulator holds dQ^T (rows = head dim, cols = q); the epilogue trades it once (`acc_to_rows8`) into 8 contiguous head-dim values per lane.
Every operand width, accumulator coordinate, exchange and epilogue goes through `rdna_layout_gfx11` with `V8 = wmma_v8_for_arch(arch)`: gfx11 passes
16 K values per lane (`v16`), gfx12 passes 8. The gfx12 path is validated only by the numpy lane model until a gfx12 device job exists.

Head-dim shards (`contraction_shards` Qs): Qs waves cooperate on one 16-row Q tile, each holding `tile / Qs` columns of Q, dO and the dQ
accumulator. The partial S and dP are summed across the shards through LDS (`fmha.reduce_s_across_shards`, f32 slots, arch-neutral); the shards'
dQ stores are disjoint. `kt_lds_layout`: `scalar` reads K^T out of the row-major K tile with one 16-bit LDS read per element; `staged` writes a
transposed K^T copy to LDS at staging time (scalar 16-bit stores through registers; gfx11 has no `global_load_tr`) and reads one vector per operand.

P and dS are narrowed to the 16-bit operand type for GEMM3 (`dS`): f16 by conversion, bf16 by truncation (round toward zero,
`fmha.bf16_trunc_pack_v8`) or, with the `rne` knob, round to nearest even (`fmha_common_gfx11.bf16_rne_pack_v8`; the dB store narrows the same
way); both go through the f32 exchange first. Dropout: `rt_flags` bit 2 selects AOTriton's 32-bit plane product (see the dK/dV module).

Rows past `seqlen_q` read zero Q/dO packs, `LSE' = +inf` and `Delta = 0`: P = exp2(s - inf) = 0 and dS = 0 without a per-element select.
Columns past `seqlen_k` are masked to -inf in the ragged tail tile only (the other masks live in that same masked region).

Dump (`dump=1`, debug): workgroup (0, 0, 0) stores S, P, dP, dS and the raw dQ^T accumulators, lane-ordered f32, at its iterations 0, 1 and last
into `dump`, in the version-1 layout of `h/arms_rdna_dump.py` (header of 64 u32 words, then slots 0 / 1 / last x tiles S, P, dP, dS, dQ_acc, each
`[BLOCK_M/16, BLOCK_N/16 (accumulators: ceil16(hd)/16), 32 lanes, 8]`). Only predicated stores are added to the production index code.

Milestones A-E build classes `plain`, `causal` and `bias`, exact or padded head, MHA/GQA, dense and packed varlen addressing, baked dropout (`dform=bk`;
the runtime `dform=rt` axis is not built on gfx11).

Dropout (`meta.dropout`): `dP <- keep ? dP * dropout_scale : 0` on the f32 dP before dS (P stays undropped, Delta already carries the
mask through the forward's O: gfx950 and gfx1201 order). The mask is the shared stream of `philox.py` (`Philox.for_arch(arch)`: width 32, four
randoms per call, 64-bit counter): plane = `block_idx.z * num_head_q + head_q` (the grid's sequence, not the packed batch index),
`Philox.grid_plane(offset, plane, max_seqlen_q, max_seqlen_k)` (the max lengths, not the sequence's own, so a varlen mask is a function of the
element coordinates and the call's two maxima), element (q, kv) at counter `grid_offset(.., q, kv)` slot `kv % 4`. A `KV_ROWS` tile holds a lane's
8 kv rows of one q column as `rdna_layout_gfx11.dropout_kv_rows_call` says: on gfx11 four Philox calls per 16-kv accumulator, each serving the
lane's two elements of one counter (the klane pair draws the same calls and keeps different slots, so the work is duplicated 2x); on gfx12 two
calls. The counter offset of kv column `kv_block_start` is a 64-bit quantity formed once per row (`grid_offset(.., q, 0)`); a block start is a multiple of
`block_n` (>= 16), so `kv // 4` splits exactly.

Varlen and GQA (runtime paths of the one object): the Q and K sides decode independently (`fmha.decode_addressing`: STACKED, CUMULATIVE /
INDIVIDUAL lengths, REUSE / ARRAY positions); LSE and Delta share one HT or TH layout (`fmha.lse_row_addressing`); query head `h` reads KV head
`h // (num_head_q / num_head_k)` (GQA and MQA); dQ is per query head, so no group sum and no atomics.

Class `bias` (no mask; the exponent is `(sm_scale * S + bias - LSE') * log2e`, the bias added before the ragged-tail mask):
    bias load    `bias_stage == "direct"`: every accumulator element reads its own 16-bit bias entry `B[batch, head, q, kv]` from global memory
                 (`rdna_layout_gfx11.bias_offset`: row stride `stride_b_seq_q`, contiguous kv). No LDS tile, so no alignment or stride contract
                 (there is no `bias_chunk_contract_ok`). A row-broadcast bias (`stride_b_seq_q == 0`) reads the one `(1, Sk)` row for every q; a head or
                 batch broadcast is a zero stride. In the ragged-tail tile a column past `seqlen_k` reads column 0 (the mask discards it), so no read leaves
                 the bias allocation; a row past `seqlen_q` reads the tile's first row.
    dB store     `store_db` builds only. `dB[batch, head, q, kv] = dS` (unscaled, narrowed exactly as the GEMM3 operand): the lane's 8 contiguous kv of
                 its q column (rows of the `KV_ROWS` tile after the xor-16 trade, `kv0 + 8 klane + j`) are 16-bit stores, one per element, because
                 `stride_db_seq_q` has no alignment contract. A store is issued only when `stride_db_seq_q != 0`, the q row is `< seqlen_q` and the kv column
                 is `< seqlen_k`: every issued store lies in `[0, (seqlen_q - 1) stride_db_seq_q + seqlen_k)` of the (batch, head) slab, and a dB that is
                 not requested (all dB strides zero, or the `store_db=0` twin, which has no store at all) is never touched; the DB pointer is not read.
                 gfx950 expresses the same rule as a buffer descriptor bound and a drop sentinel that are both 0 for a zero row stride.
"""

from types import SimpleNamespace as _WireArgs


from kernels.attention_bwd_rdna import fmha_common_gfx1201 as fmha
from kernels.attention_bwd_rdna import rdna_layout_gfx11 as L
from kernels.attention_bwd_rdna.fmha_common_gfx11 import (
    _guarded_batches,
    _if_then,
    _stage,
    _write_v8,
)  # noqa: F401  (the guarded helpers kernels_common lacks)
from kernels.attention_bwd_rdna import fmha_common_gfx11 as cmn
from kernels.attention_bwd_rdna.fmha_tuning_bwd_dq_gfx11 import (  # noqa: F401
    BwdDqKnobsGfx11,
    LDS_PAD,
    bwd_dq_knobs,
    reduction_aliases_v,
)
from kernels.attention_rdna4.philox import Philox
from kernels.common import buffer_ops
from kernels.common import kernels_common as common_kernels

from kernels.attention_bwd_rdna import flyc_polyfill as common_utils
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf as _scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

KERNEL_NAME = "fmha_bwd_dq_gfx11_kernel"
_LOG2E = 1.4426950408889634
WARP_SIZE = 32
WMMA = 16
VEC_WIDTH = 8  # elements per cooperative global load (16 bytes)
DUMP_HEADER_WORDS = 64
DUMP_MAGIC = 0x52444E41
DUMP_VERSION = 1
DUMP_KERNEL_DQ = 1

_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
        "amdgpu-use-amdgpu-trackers": True,
    },
}


def _pointer_load(result_type, ptr):
    return _llvm.LoadOp(result_type, fmha.llvm_value(ptr)).result


def _pointer_store(value, ptr):
    return _llvm.StoreOp(fmha.llvm_value(value), fmha.llvm_value(ptr))


def _cat(parts):
    """Concatenate equal-width vectors (v4 -> v8 -> v16)."""
    parts = [Vec(p) for p in parts]
    while len(parts) > 1:
        parts = [
            parts[i].shuffle(parts[i + 1], list(range(2 * parts[i].numel)))
            for i in range(0, len(parts), 2)
        ]
    return parts[0].ir_value()


def _lds_load_vec(lds_ptr, lds_idx, width, v4_type):
    """`width` (8 or 16) contiguous 16-bit elements from LDS as honest 8-byte accesses (rows are only guaranteed 8-byte aligned)."""
    return _cat(
        [
            fx.ptr_load(lds_ptr + fx.Int32(lds_idx + 4 * i), result_type=v4_type)
            for i in range_constexpr(width // 4)
        ]
    )


def _ceil16(n):
    return (n + 15) // 16 * 16


def build_fmha_bwd_dq_gfx11_module_primary(meta, knobs, runtime_axes=()):
    """Build the gfx11 dQ kernel for `meta` (`variant_gfx11.Meta`) and its resolved knob record (`bwd_dq_knobs`); returns the host launch function
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
    if meta.cls not in ("plain", "causal", "bias"):
        raise NotImplementedError(
            f"class {meta.cls!r}: the gfx11 dQ builds plain, causal and bias"
        )
    dtype_str = meta.dtype
    BLOCK_DMODEL = meta.tile
    PADDED_HEAD = bool(meta.padded)
    if not PADDED_HEAD and not (meta.hdim_qk == meta.hdim_vo == BLOCK_DMODEL):
        raise ValueError(
            f"an exact build needs hdim_qk == hdim_vo == tile ({BLOCK_DMODEL}); got {meta.hdim_qk}, {meta.hdim_vo}"
        )
    if meta.hdim_qk > BLOCK_DMODEL or meta.hdim_vo > BLOCK_DMODEL:
        raise ValueError(
            f"hdim ({meta.hdim_qk}, {meta.hdim_vo}) exceeds the tile {BLOCK_DMODEL}"
        )

    V8 = L.wmma_v8_for_arch(knobs.arch)
    OW = L.op_width(V8)  # K values a lane passes per operand
    NUM_WAVES = knobs.num_waves
    QK_SHARDS = knobs.contraction_shards
    BLOCK_M = knobs.BLOCK_M
    BLOCK_N = knobs.block_n
    QK_SLICE = knobs.slice
    Q_TILES_PER_BLOCK = NUM_WAVES // QK_SHARDS
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE
    KT_STAGED = knobs.kt_lds_layout == "staged"
    DUMP = knobs.dump
    RNE = bool(knobs.rne)
    CAUSAL = meta.cls == "causal"
    BIAS = meta.cls == "bias"
    DROPOUT = bool(meta.dropout)
    PHILOX = Philox.for_arch(knobs.arch)
    assert (
        PHILOX.randoms_per_offset == 4
    ), "dropout_kv_rows_call models four randoms per Philox call"
    STORE_DB = BIAS and bool(knobs.store_db)
    if BIAS and knobs.bias_stage != "direct":
        raise NotImplementedError(
            f"bias_stage {knobs.bias_stage!r}: the gfx11 dQ reads its bias directly from global memory"
        )
    LPT = CAUSAL and knobs.lpt_tile_order
    fastmath = fmha.FastMath(knobs.fp_mode)
    # Exponent chain of a bias object: contract only. No `nnan`/`ninf` (a -inf bias entry must reach exp2) and no `reassoc` (it would let the
    # compiler undo "combine in natural units, scale by log2 e last").
    fastmath_e = fmha.FastMath("safe")
    fastmath_e.flags = fx.arith.FastMathFlags.contract
    NUM_S_ACCS = BLOCK_N // WMMA
    NUM_S_VALS = NUM_S_ACCS * 8
    K_STEPS = QK_SLICE // WMMA
    D_CHUNKS = QK_SLICE // WMMA
    assert BLOCK_M == WMMA * Q_TILES_PER_BLOCK

    # ---- LDS layout (elements) ----
    K_STRIDE = BLOCK_DMODEL + LDS_PAD
    V_STRIDE = BLOCK_DMODEL + LDS_PAD
    KT_STRIDE = BLOCK_N + LDS_PAD
    K_TPR_LOAD, K_ROWS_PER_BATCH, NUM_BATCHES_K, K_NEEDS_GUARD = fmha.load_geom(
        BLOCK_DMODEL, VEC_WIDTH, BLOCK_SIZE, BLOCK_N
    )
    V_TPR_LOAD, V_ROWS_PER_BATCH, NUM_BATCHES_V, V_NEEDS_GUARD = fmha.load_geom(
        BLOCK_DMODEL, VEC_WIDTH, BLOCK_SIZE, BLOCK_N
    )
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_BASE = LDS_K_TILE_SIZE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_KT_BASE = LDS_V_BASE + LDS_V_TILE_SIZE
    LDS_KT_TILE_SIZE = BLOCK_DMODEL * KT_STRIDE if KT_STAGED else 0
    RED_F32_PER_WAVE = 2 * NUM_S_VALS * WARP_SIZE
    RED_F32_TOTAL = NUM_WAVES * RED_F32_PER_WAVE
    RED_ALIASES_V = reduction_aliases_v(BLOCK_DMODEL, NUM_WAVES, QK_SHARDS, BLOCK_N)
    LDS_RED_SIZE = 0 if RED_ALIASES_V else RED_F32_TOTAL * 2
    LDS_TOTAL_SIZE = LDS_KT_BASE + LDS_KT_TILE_SIZE + LDS_RED_SIZE
    RED_BYTE0 = (LDS_V_BASE if RED_ALIASES_V else LDS_KT_BASE + LDS_KT_TILE_SIZE) * 2
    assert LDS_TOTAL_SIZE * 2 == knobs.lds_bytes, (LDS_TOTAL_SIZE * 2, knobs.lds_bytes)

    # ---- dump geometry (words of f32) ----
    DUMP_NG = BLOCK_M // WMMA
    DUMP_ND = _ceil16(meta.hdim_qk) // WMMA
    DUMP_S_WORDS = DUMP_NG * NUM_S_ACCS * WARP_SIZE * 8
    DUMP_ACC_WORDS = DUMP_NG * DUMP_ND * WARP_SIZE * 8
    DUMP_SLOT_WORDS = 4 * DUMP_S_WORDS + DUMP_ACC_WORDS
    BLOCK_N_LOG2 = BLOCK_N.bit_length() - 1

    _cache_tag = (
        KERNEL_NAME,
        dtype_str,
        BLOCK_DMODEL,
        PADDED_HEAD,
        V8,
        NUM_WAVES,
        QK_SHARDS,
        BLOCK_N,
        knobs.kt_lds_layout,
        knobs.waves_per_eu,
        DUMP,
    )
    if RNE:
        _cache_tag += ("rne",)
    if CAUSAL:
        _cache_tag += ("causal", LPT)
    if BIAS:
        _cache_tag += ("bias", STORE_DB)
    if DROPOUT:
        _cache_tag += ("dropout", PHILOX.width)
    elem_numeric_cls = common_kernels.dtype_to_elem_type(dtype_str)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem_numeric_cls, LDS_TOTAL_SIZE, 16]

    @flyc.jit
    def emit_dq_body(a):
        """The kernel body over one namespace `a` of the wire arguments (plus `a.dump` in a dump object)."""
        Q, K, V, DO, DQ = a.Q, a.K, a.V, a.DO, a.DQ
        LSE, Delta = a.LSE, a.Delta
        seqinfo_q0, seqinfo_q1, seqinfo_k0, seqinfo_k1 = (
            a.seqinfo_q0,
            a.seqinfo_q1,
            a.seqinfo_k0,
            a.seqinfo_k1,
        )
        varlen_bits, num_seqlens, max_seqlen_q, max_seqlen_k = (
            a.varlen_bits,
            a.num_seqlens,
            a.max_seqlen_q,
            a.max_seqlen_k,
        )
        num_head_q, num_head_k, hdim_qk, hdim_vo, sm_scale = (
            a.num_head_q,
            a.num_head_k,
            a.hdim_qk,
            a.hdim_vo,
            a.sm_scale,
        )
        window_left, window_right = a.window_left, a.window_right

        elem_dtype = elem_numeric_cls
        elem_type = elem_numeric_cls.ir_type
        vx_type = Vec.make_type(VEC_WIDTH, elem_dtype)
        v8_type = Vec.make_type(8, elem_dtype)
        v4_type = Vec.make_type(4, elem_dtype)
        f32_ty = ir.F32Type.get()

        q_ptr = fmha.pointer_to_llvm_ptr(Q)
        k_ptr = fmha.pointer_to_llvm_ptr(K)
        v_ptr = fmha.pointer_to_llvm_ptr(V)
        do_ptr = fmha.pointer_to_llvm_ptr(DO)
        dq_ptr = fmha.pointer_to_llvm_ptr(DQ)

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA, WMMA, WMMA, elem_dtype, fx.Float32)
        )

        def wmma_acc(a_op, b_op, c8):
            # gfx11: 16 K values per lane for A and B; gfx12: 8. Eight f32 accumulators either way.
            a_frag = fx.make_rmem_tensor(OW, elem_dtype)
            b_frag = fx.make_rmem_tensor(OW, elem_dtype)
            c_frag = fx.make_rmem_tensor(8, fx.Float32)
            a_frag.store(Vec(a_op))
            b_frag.store(Vec(b_op))
            c_frag.store(Vec(c8))
            fx.gemm(wmma_atom, c_frag, a_frag, b_frag, c_frag)
            return Vec(c_frag.load()).ir_value()

        def exchange(v):
            return fx.Float32(v).shuffle_xor(fx.Int32(16), fx.Int32(WARP_SIZE))

        # ---- varlen prologue: VarlenBits -> sequence scalars ----
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
        seqlen_k_v = fx.Index(seqlen_k_i32)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_kv = lds.kv.ptr
        _lds_byte_base = fx.as_ir_value(fx.ptrtoint(lds_kv))

        tid, wave_id, lane, lane16, klane = fmha.wave_lanes(WARP_SIZE)
        klane_is_zero = klane == fx.Index(0)

        q_tile_in_block = wave_id // QK_SHARDS
        shard_id = wave_id % QK_SHARDS
        wave_q_offset = q_tile_in_block * WMMA
        shard_qk_off = shard_id * fx.Index(QK_SLICE)

        head_q = fx.Index(gpu.block_idx.x)
        q_tile_idx = fx.Index(gpu.block_idx.y)
        if const_expr(LPT):
            # Longest-processing-time order: the heaviest (last) Q tile of the sequence is dispatched first; tiles past the sequence stay where they are.
            _n_qblk = (fx.Index(seqlen_q_i32) + fx.Index(BLOCK_M - 1)) // fx.Index(
                BLOCK_M
            )
            q_tile_idx = fx.Index(
                _n_qblk - q_tile_idx - fx.Index(1)
                if q_tile_idx < _n_qblk
                else q_tile_idx
            )
        start_q = q_tile_idx * BLOCK_M
        _alive = fx.Int32(start_q) < seqlen_q_i32
        _q_start_addr = fx.Index(start_q if _alive else fx.Index(0))
        head_k = head_q // (fx.Index(num_head_q) // fx.Index(num_head_k))

        load_row_in_batch = tid // K_TPR_LOAD
        load_col_base = (tid % K_TPR_LOAD) * VEC_WIDTH
        v_row_in_batch = tid // V_TPR_LOAD
        v_col_base = (tid % V_TPR_LOAD) * VEC_WIDTH

        _slast_i32 = common_utils.smax(seqlen_k_i32 - fx.Int32(1), fx.Int32(0))
        seq_last = fx.Index(_slast_i32)

        q_st = (
            fx.Index(a.stride_q_batch),
            fx.Index(a.stride_q_head),
            fx.Index(a.stride_q_seq),
        )
        k_st = (
            fx.Index(a.stride_k_batch),
            fx.Index(a.stride_k_head),
            fx.Index(a.stride_k_seq),
        )
        v_st = (
            fx.Index(a.stride_v_batch),
            fx.Index(a.stride_v_head),
            fx.Index(a.stride_v_seq),
        )
        do_st = (
            fx.Index(a.stride_do_batch),
            fx.Index(a.stride_do_head),
            fx.Index(a.stride_do_seq),
        )
        dq_st = (
            fx.Index(a.stride_dq_batch),
            fx.Index(a.stride_dq_head),
            fx.Index(a.stride_dq_seq),
        )
        sm_log2e = fastmath.mul(sm_scale, fx.Float32(_LOG2E))

        _q_batch_v = fx.Index(q_batch)
        _k_batch_v = fx.Index(k_batch)
        _q_row_off_v = fx.Index(q_row_off)
        _k_row_off_v = fx.Index(k_row_off)
        _addr_kw = dict(seqlen_k=seqlen_k_v, seq_last=seq_last, hoist=False, clamp=True)
        q_tbase, q_toff, _ = fmha.make_addr_pair(
            q_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw
        )
        do_tbase, do_toff, _ = fmha.make_addr_pair(
            do_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw
        )
        dq_tbase, dq_toff, _ = fmha.make_addr_pair(
            dq_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw
        )
        _, _, k_addr = fmha.make_addr_pair(
            k_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw
        )
        _, _, v_addr = fmha.make_addr_pair(
            v_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw
        )

        # ---- padded-head masks: an element is valid iff its column < the runtime extent ----
        qk_cols = fmha.MaskedAxis(
            fx.Index(hdim_qk), active=PADDED_HEAD, elem_dtype=elem_dtype
        )
        vo_cols = fmha.MaskedAxis(
            fx.Index(hdim_vo), active=PADDED_HEAD, elem_dtype=elem_dtype
        )
        k_ap = fmha.Aperture(
            qk_cols,
            lds_base=0,
            lds_stride=K_STRIDE,
            vec_width=VEC_WIDTH,
            threads_per_row=K_TPR_LOAD,
            rows_per_batch=K_ROWS_PER_BATCH,
            num_batches=NUM_BATCHES_K,
            needs_guard=K_NEEDS_GUARD,
        )
        v_ap = fmha.Aperture(
            vo_cols,
            lds_base=LDS_V_BASE,
            lds_stride=V_STRIDE,
            vec_width=VEC_WIDTH,
            threads_per_row=V_TPR_LOAD,
            rows_per_batch=V_ROWS_PER_BATCH,
            num_batches=NUM_BATCHES_V,
            needs_guard=V_NEEDS_GUARD,
        )

        def load_global_vx(base_ptr, base64, off32):
            return _pointer_load(
                vx_type, fmha.split_ptr(base_ptr, base64, off32, elem_type)
            )

        def load_global_v8(base_ptr, base64, off32):
            return _pointer_load(
                v8_type, fmha.split_ptr(base_ptr, base64, off32, elem_type)
            )

        def store_global_v8(base_ptr, base64, off32, val):
            _pointer_store(val, fmha.split_ptr(base_ptr, base64, off32, elem_type))

        fetch_k = fmha.reader(k_addr, lambda b, o: load_global_vx(k_ptr, b, o))
        fetch_v = fmha.reader(v_addr, lambda b, o: load_global_vx(v_ptr, b, o))

        # ---- Q and dO preload: one 16-row tile per wave (rows past seqlen_q are zero) ----
        q_row = start_q + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)
        q_row_in_tile = wave_q_offset + lane16
        q_rows_axis = fmha.MaskedAxis(fx.Index(seqlen_q_i32))
        q_ap = fmha.Aperture(qk_cols, rows=q_rows_axis)
        do_ap = fmha.Aperture(vo_cols, rows=q_rows_axis)
        q_tile_base = q_tbase(_q_start_addr)
        do_tile_base = do_tbase(_q_start_addr)

        def fetch_q(row, col):
            return load_global_v8(q_ptr, q_tile_base, q_toff(row, col))

        def fetch_do(row, col):
            return load_global_v8(do_ptr, do_tile_base, do_toff(row, col))

        _q_in, _q_safe = q_ap.rows.gate(q_row, q_row_in_tile)

        if const_expr(BIAS):
            # The bias is (batch, head_q, q row, kv col) with a contiguous kv axis; a zero stride is a broadcast, and a row-broadcast bias
            # (`stride_b_seq_q == 0`) makes every q row read the one (1, Sk) row. A row past `seqlen_q` reads the tile's first row.
            b_ptr = fmha.pointer_to_llvm_ptr(a.B)
            b_st = (
                fx.Index(a.stride_b_batch),
                fx.Index(a.stride_b_head),
                fx.Index(a.stride_b_seq_q),
            )
            b_tbase, b_toff, _ = fmha.make_addr_pair(
                b_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw
            )
            b_tile_base = b_tbase(_q_start_addr)
            b_row_off = b_toff(_q_safe, fx.Index(0))
            kl_off = fx.Int32(klane) * fx.Int32(8 if V8 else 1)
        if const_expr(STORE_DB):
            # dB is stored only for a requested dB (`stride_db_seq_q != 0`; all three strides zero is "not requested") and a live q row.
            db_ptr = fmha.pointer_to_llvm_ptr(a.DB)
            db_st = (
                fx.Index(a.stride_db_batch),
                fx.Index(a.stride_db_head),
                fx.Index(a.stride_db_seq_q),
            )
            db_tbase, db_toff, _ = fmha.make_addr_pair(
                db_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw
            )
            db_tile_base = db_tbase(_q_start_addr)
            db_row_off = db_toff(q_row_in_tile, fx.Index(0))
            db_on = _q_in & (fx.Int64(a.stride_db_seq_q) != fx.Int64(0))
            if const_expr(QK_SHARDS > 1):
                # Every shard wave of a Q tile holds the whole reduced dS tile, and the shards' copies need not be bit-identical (each wave sums the
                # partials in its own order, and fused multiply-adds after the sum see different operands), so two waves storing one dB element
                # race with different values. One writer per element: the shard `L.DB_WRITER_SHARD`.
                db_on = db_on & (shard_id == fx.Index(L.DB_WRITER_SHARD))

        q_packs = []
        do_packs = []
        for ks in range_constexpr(K_STEPS):
            _c = L.op_origin(shard_qk_off + fx.Index(ks * WMMA), klane, V8)
            q_packs.append(
                _cat(
                    [
                        q_ap.read_v8(fetch_q, _q_safe, _c + fx.Index(8 * h), _q_in)
                        for h in range_constexpr(OW // 8)
                    ]
                )
            )
            do_packs.append(
                _cat(
                    [
                        do_ap.read_v8(fetch_do, _q_safe, _c + fx.Index(8 * h), _q_in)
                        for h in range_constexpr(OW // 8)
                    ]
                )
            )

        # ---- LSE' and Delta: one scalar per Q row (per lane, `KV_ROWS`) ----
        _tok_i32 = lse_tokens
        _lse_tok = common_utils.smin(q_row_off + q_row_i32, _tok_i32 - fx.Int32(1))
        _tok_row = fx.Index(common_utils.smax(_lse_tok, fx.Int32(0)))
        _lse_base, _lse_pitch = fmha.lse_row_addressing(
            varlen_bits, _q_batch_v, head_q, num_head_q, _tok_i32, _tok_row
        )
        _lse_off = _lse_base

        def _load_row_f32(ptr):
            return _pointer_load(
                f32_ty,
                buffer_ops.get_element_ptr(
                    fmha.pointer_to_llvm_ptr(ptr), fx.Int64(_lse_off), elem_type=f32_ty
                ),
            )

        # A row past seqlen_q is an empty row: LSE' = +inf, Delta = 0 give P = dS = 0.
        _lse_nat = fx.Float32(
            fx.Float32(_load_row_f32(LSE)) if _q_in else fx.Float32(float("inf"))
        )
        delta_i = fx.Float32(
            fx.Float32(_load_row_f32(Delta)) if _q_in else fx.Float32(0.0)
        )
        if const_expr(BIAS):
            lse2_i = _lse_nat  # natural: the bias exponent scales by log2 e last
        else:
            lse2_i = fastmath.mul(_lse_nat, fx.Float32(_LOG2E))

        if const_expr(DROPOUT):
            # The offset scheme is `Philox.grid_plane` / `grid_offset` of the forward and of gfx950 / gfx1201: the plane is the grid's sequence times
            # num_head_q plus the query head, the grid extents are the max lengths. `_ph_row` = counter of element (q_row, 0) of this lane's column.
            _ph_seed = fmha.philox_seed_value(a.philox_seed_ptr)
            _ph_off = fmha.philox_offset_base(a.philox_offset1, a.philox_offset2)
            _ph_plane = z_i32 * fx.Int32(num_head_q) + fx.Int32(head_q)
            _ph_base, _ph_stride = PHILOX.grid_plane(
                _ph_off, _ph_plane, max_seqlen_q, max_seqlen_k
            )
            _ph_base = cmn.philox_plane_base(
                _ph_off, _ph_plane, max_seqlen_q, _ph_stride, a.rt_flags
            )
            _ph_row = PHILOX.grid_offset(_ph_base, _ph_stride, q_row_i32, fx.Int64(0))
            _drop_scale = fx.Float32(a.dropout_scale)
            _drop_thr = fx.Int32(a.idropout_p)

        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        sm_scale_vec = (
            Vec.from_elements([sm_scale], fx.Float32).broadcast_to(8).ir_value()
        )

        # ---- the KV range: full tiles, then the ragged tail tile ----
        _full_end = (seqlen_k_v // fx.Index(BLOCK_N)) * fx.Index(BLOCK_N)
        kv_upper = fx.Index(
            ((seqlen_k_v + fx.Index(BLOCK_N - 1)) // fx.Index(BLOCK_N))
            * fx.Index(BLOCK_N)
        )
        _full_end = fx.Index(_full_end if _alive else fx.Index(0))
        kv_upper = fx.Index(kv_upper if _alive else fx.Index(0))

        if const_expr(CAUSAL):
            # Mask semantics of the gfx950 kernels: key c is visible to row q iff q - wl <= c <= q + wr and c < seqlen_k, with the TOPLEFT / BOTRIGHT
            # sentinels resolved against this sequence. The clamps keep every derived i32 in range without changing the visible set.
            _wl, _wr = fmha.resolve_window(
                window_left, window_right, seqlen_q_i32, seqlen_k_i32
            )
            _wl = common_utils.smin(
                common_utils.smax(_wl, fx.Int32(0) - seqlen_k_i32), seqlen_q_i32
            )
            _wr = common_utils.smin(
                common_utils.smax(_wr, fx.Int32(0) - seqlen_q_i32), seqlen_k_i32
            )
            # The visible kv column run of this lane's q row is [_first, _first + _span) (`rdna_layout_gfx11.kv_window`).
            _first = common_utils.smax(q_row_i32 - _wl, fx.Int32(0))
            _span = common_utils.smax(
                common_utils.smin(q_row_i32 + _wr + fx.Int32(1), seqlen_k_i32) - _first,
                fx.Int32(0),
            )
            _rel_lane = fx.Int32(klane) * fx.Int32(8 if V8 else 1) - _first
            # The Q tile's KV block runs: [masked][full][masked]; a block no row reaches is not walked, a Q tile with no visible key walks none.
            _reg = fmha.decompose_causal_regions(
                start_q, seqlen_q_i32, seqlen_k_i32, _wl, _wr, BLOCK_M, BLOCK_N, _alive
            )
            _n_l, _n_f, _n_r = _reg.n_left, _reg.n_full, _reg.n_right
            _col_l, _col_f, _col_r = _reg.left_col0, _reg.full_col0, _reg.right_col0
            _runs = [(_col_l, _n_l, True), (_col_f, _n_f, False), (_col_r, _n_r, True)]
        else:
            _runs = [(None, None, False), (None, None, True)]

        if const_expr(DUMP):
            dump_ptr = fmha.pointer_to_llvm_ptr(a.dump)
            if const_expr(CAUSAL):
                _n_iter = _n_l + _n_f + _n_r
            else:
                _n_iter = (seqlen_k_i32 + fx.Int32(BLOCK_N - 1)) >> fx.Int32(
                    BLOCK_N_LOG2
                )
            if const_expr(LPT):
                _is_wg0 = (
                    (fx.Int32(gpu.block_idx.x) == fx.Int32(0))
                    & (fx.Int32(q_tile_idx) == fx.Int32(0))
                    & (z_i32 == fx.Int32(0))
                )
            else:
                _is_wg0 = (
                    (fx.Int32(gpu.block_idx.x) == fx.Int32(0))
                    & (fx.Int32(gpu.block_idx.y) == fx.Int32(0))
                    & (z_i32 == fx.Int32(0))
                )

            def _dump_header():
                if const_expr(CAUSAL):
                    _blk0 = _col_l >> fx.Int32(BLOCK_N_LOG2)
                    _b0 = (_n_iter >= fx.Int32(1)).select(_blk0, fx.Int32(-1))
                    _b1 = (_n_iter >= fx.Int32(2)).select(
                        _blk0 + fx.Int32(1), fx.Int32(-1)
                    )
                    _b2 = (_n_iter >= fx.Int32(3)).select(
                        _blk0 + _n_iter - fx.Int32(1), fx.Int32(-1)
                    )
                else:
                    _b0 = fx.Int32(0)
                    _b1 = (_n_iter >= fx.Int32(2)).select(fx.Int32(1), fx.Int32(-1))
                    _b2 = (_n_iter >= fx.Int32(3)).select(
                        _n_iter - fx.Int32(1), fx.Int32(-1)
                    )
                _hdr = [
                    fx.Int32(v)
                    for v in (
                        DUMP_MAGIC,
                        DUMP_VERSION,
                        DUMP_KERNEL_DQ,
                        BLOCK_M,
                        BLOCK_N,
                    )
                ] + [
                    _n_iter,
                    fx.Int32(meta.hdim_qk),
                    fx.Int32(0),
                    _b0,
                    _b1,
                    _b2,
                ]
                for _w in range_constexpr(len(_hdr)):
                    _pointer_store(
                        fx.as_ir_value(_hdr[_w]),
                        buffer_ops.get_element_ptr(
                            dump_ptr, fx.Int64(_w), elem_type=T.i32
                        ),
                    )

            def _dump_s_tiles(slot_base, tiles):
                """`tiles` = (S, P, dP, dS), each NUM_S_VALS f32; one v8 store per (tile, 16-kv group) at the lane-ordered slot."""
                for _t in range_constexpr(4):
                    for _a in range_constexpr(NUM_S_ACCS):
                        _idx = (
                            slot_base
                            + fx.Int64(_t * DUMP_S_WORDS)
                            + (
                                (
                                    fx.Int64(q_tile_in_block) * fx.Int64(NUM_S_ACCS)
                                    + fx.Int64(_a)
                                )
                                * fx.Int64(WARP_SIZE)
                                + fx.Int64(lane)
                            )
                            * fx.Int64(8)
                        )
                        _pointer_store(
                            Vec.from_elements(
                                tiles[_t][_a * 8 : _a * 8 + 8], fx.Float32
                            ).ir_value(),
                            buffer_ops.get_element_ptr(
                                dump_ptr, _idx, elem_type=f32_ty
                            ),
                        )

            def _dump_acc(slot_base, accs):
                for _dc in range_constexpr(D_CHUNKS):
                    _dcg = shard_id * fx.Index(D_CHUNKS) + fx.Index(_dc)
                    _idx = (
                        slot_base
                        + fx.Int64(4 * DUMP_S_WORDS)
                        + (
                            (
                                fx.Int64(q_tile_in_block) * fx.Int64(DUMP_ND)
                                + fx.Int64(_dcg)
                            )
                            * fx.Int64(WARP_SIZE)
                            + fx.Int64(lane)
                        )
                        * fx.Int64(8)
                    )
                    _pointer_store(
                        fx.as_ir_value(accs[_dc]),
                        buffer_ops.get_element_ptr(dump_ptr, _idx, elem_type=f32_ty),
                    )

            if _is_wg0 & (tid == fx.Index(0)):
                _dump_header()

        def _stage_k(start):
            """K rows -> the K tile (and, `staged`, the transposed K^T tile through scalar 16-bit stores)."""
            read = fetch_k(start)

            def body(_batch, row):
                vec = k_ap.read_vec(read, row, load_col_base)
                k_ap.to_lds(lds_kv, vec, row, load_col_base, VEC_WIDTH)
                if const_expr(KT_STAGED):
                    for j in range_constexpr(VEC_WIDTH):
                        fx.ptr_store(
                            Vec(vec)[j],
                            lds_kv
                            + fx.Int32(
                                LDS_KT_BASE
                                + (load_col_base + fx.Index(j)) * KT_STRIDE
                                + row
                            ),
                        )

            _guarded_batches(k_ap, load_row_in_batch, fx.Index(BLOCK_N), body)

        def kv_loop_body(kv_block_start, inner_iter_args, _MASK_STEPS):
            """One KV tile. `_MASK_STEPS` is a trace-time bool: the ragged-tail region emits the column mask, the full region none."""
            dq_accs = [inner_iter_args[i] for i in range_constexpr(D_CHUNKS)]

            gpu.barrier()
            _stage_k(kv_block_start)
            _stage(
                v_ap,
                lds_kv,
                fetch_v(kv_block_start),
                v_row_in_batch,
                v_col_base,
                fx.Index(BLOCK_N),
            )
            gpu.barrier()

            # ==== GEMM1: S^T = K @ Q^T (A = K row-major, lane16 = kv row; B = Q preload) ====
            s_accs = [fx.as_ir_value(c_zero_v8f32) for _ in range_constexpr(NUM_S_ACCS)]
            for ks in range_constexpr(K_STEPS):
                for a_idx in range_constexpr(NUM_S_ACCS):
                    _kidx = L.lds_rowmajor_operand(
                        0,
                        K_STRIDE,
                        lane16 + fx.Index(a_idx * WMMA),
                        shard_qk_off + fx.Index(ks * WMMA),
                        klane,
                        V8,
                    )
                    s_accs[a_idx] = wmma_acc(
                        _lds_load_vec(lds_kv, _kidx, OW, v4_type),
                        q_packs[ks],
                        s_accs[a_idx],
                    )

            # ==== GEMM2: dP^T = V @ dO^T ====
            dp_accs = [
                fx.as_ir_value(c_zero_v8f32) for _ in range_constexpr(NUM_S_ACCS)
            ]
            for ks in range_constexpr(K_STEPS):
                for a_idx in range_constexpr(NUM_S_ACCS):
                    _vidx = L.lds_rowmajor_operand(
                        LDS_V_BASE,
                        V_STRIDE,
                        lane16 + fx.Index(a_idx * WMMA),
                        shard_qk_off + fx.Index(ks * WMMA),
                        klane,
                        V8,
                    )
                    dp_accs[a_idx] = wmma_acc(
                        _lds_load_vec(lds_kv, _vidx, OW, v4_type),
                        do_packs[ks],
                        dp_accs[a_idx],
                    )

            # ==== cross-shard reduction of the partial S and dP ====
            if const_expr(QK_SHARDS > 1):
                if const_expr(RED_ALIASES_V):
                    gpu.barrier()  # the partials land on the V tile: every wave must be past its GEMM2 reads
                _red = fmha.reduce_s_across_shards(
                    s_accs + dp_accs,
                    lds_byte_base=_lds_byte_base,
                    byte0=RED_BYTE0,
                    wave_id=wave_id,
                    lane=lane,
                    shard_id=shard_id,
                    q_tile_in_block=q_tile_in_block,
                    num_shards=QK_SHARDS,
                    f32_per_wave=RED_F32_PER_WAVE,
                    warp_size=WARP_SIZE,
                    fastmath=fastmath,
                )
                s_accs = _red[:NUM_S_ACCS]
                dp_accs = _red[NUM_S_ACCS:]

            s_raw = []
            dp_raw = []
            for st in range_constexpr(NUM_S_ACCS):
                for r in range_constexpr(8):
                    s_raw.append(Vec(s_accs[st])[r])
                    dp_raw.append(Vec(dp_accs[st])[r])
            s_acc_raw = list(s_raw)  # the unscaled accumulators (the dump's S)

            # Scaled before the mask, so the mask's -inf cannot be folded away by a fast-math multiply.
            if const_expr(BIAS):
                s_raw = [fastmath_e.mul(v, sm_scale) for v in s_raw]
            else:
                s_raw = [fastmath.mul(v, sm_log2e) for v in s_raw]

            # Element `st * 8 + r` is kv `kv_block_start + st * 16 + acc_row(r, klane)` of q column `lane16`.
            _kv_i32 = fx.Int32(kv_block_start)
            if const_expr(BIAS):
                # Added after the scale and before the ragged-tail mask: `sm_scale * S + bias` (natural units; `log2e` scales the exponent). A column past `seqlen_k` (tail tile) reads column 0.
                b_f32 = []
                for _i in range_constexpr(NUM_S_VALS):
                    _bcol = (
                        _kv_i32
                        + fx.Int32((_i // 8) * WMMA + L.acc_row(_i % 8, 0, V8))
                        + kl_off
                    )
                    if const_expr(_MASK_STEPS):
                        _bcol_ld = (_bcol < seqlen_k_i32).select(_bcol, fx.Int32(0))
                    else:
                        _bcol_ld = _bcol
                    _bp = fmha.split_ptr(
                        b_ptr, b_tile_base, b_row_off + fx.Index(_bcol_ld), elem_type
                    )
                    _bv = fx.Float32(
                        _arith.ExtFOp(f32_ty, _pointer_load(elem_type, _bp)).result
                    )
                    b_f32.append(_bv)
                    s_raw[_i] = fastmath_e.add(fx.Float32(s_raw[_i]), _bv)
            if const_expr(CAUSAL and (_MASK_STEPS or DUMP)):
                # One unsigned compare per element: `0 <= kv - first < span`, the Sk bound included.
                _rel0 = _kv_i32 + _rel_lane
                _vis = [
                    fx.Uint32(
                        _rel0 + fx.Int32((_i // 8) * WMMA + L.acc_row(_i % 8, 0, V8))
                    )
                    < fx.Uint32(_span)
                    for _i in range_constexpr(NUM_S_VALS)
                ]
            if const_expr(_MASK_STEPS and not CAUSAL):
                for _i in range_constexpr(NUM_S_VALS):
                    _col = (
                        _kv_i32
                        + fx.Int32((_i // 8) * WMMA + L.acc_row(_i % 8, 0, V8))
                        + fx.Int32(klane) * fx.Int32(8 if V8 else 1)
                    )
                    _dead = _col >= seqlen_k_i32
                    s_raw[_i] = c_neg_inf if _dead else s_raw[_i]

            p_vals = []
            for _i in range_constexpr(NUM_S_VALS):
                if const_expr(BIAS):
                    _ex = fastmath_e.mul(
                        fastmath_e.sub(fx.Float32(s_raw[_i]), lse2_i),
                        fx.Float32(_LOG2E),
                    )
                else:
                    _ex = fastmath.sub(fx.Float32(s_raw[_i]), lse2_i)
                _e = rocdl.exp2(f32_ty, fx.as_ir_value(_ex))
                if const_expr(CAUSAL and _MASK_STEPS):
                    # Selected after the exponential: a masked score may be anything against any LSE', so no -inf is formed before `exp2`.
                    _e = _vis[_i].select(fx.Float32(_e), fx.Float32(0.0))
                p_vals.append(_e)

            if const_expr(DROPOUT):
                # dP <- keep ? dP * scale : 0, before dS. One Philox call per (accumulator, counter group); `dropout_kv_rows_call` gives each element
                # its group (counter offset from `kv_block_start // 4`) and slot at klane 0 and 1: a klane-dependent group is an offset, a
                # klane-dependent slot a select.
                _tile_ctr = _ph_row + (fx.Int64(kv_block_start) >> fx.Int64(2))
                for _st in range_constexpr(NUM_S_ACCS):
                    _calls = {}
                    for _r in range_constexpr(8):
                        _g0, _s0 = L.dropout_kv_rows_call(_r, 0, V8)
                        _g1, _s1 = L.dropout_kv_rows_call(_r, 1, V8)
                        if const_expr((_g0, _g1) not in _calls):
                            _grp = fx.Int64(_st * (WMMA // 4) + _g0)
                            if const_expr(_g1 != _g0):
                                _grp = _grp + fx.Int64(klane) * fx.Int64(_g1 - _g0)
                            _calls[(_g0, _g1)] = PHILOX.u32(_ph_seed, _tile_ctr + _grp)
                        _w = _calls[(_g0, _g1)]
                        if const_expr(_s0 == _s1):
                            _word = fx.Int32(_w[_s0])
                        else:
                            _word = klane_is_zero.select(
                                fx.Int32(_w[_s0]), fx.Int32(_w[_s1])
                            )
                        _keep = fx.Int32(_word) > _drop_thr
                        _i = _st * 8 + _r
                        dp_raw[_i] = _keep.select(
                            fastmath.mul(fx.Float32(dp_raw[_i]), _drop_scale),
                            fx.Float32(0.0),
                        )

            # dS = P * (dP - delta); delta belongs to the Q row = lane16.
            ds_vals = []
            for _i in range_constexpr(NUM_S_VALS):
                ds_vals.append(
                    fastmath.mul(
                        fx.Float32(p_vals[_i]),
                        fastmath.sub(fx.Float32(dp_raw[_i]), delta_i),
                    )
                )

            if const_expr(DUMP):
                # Predicated stores only: iteration 0, 1 and the last (label -1 needs n_iter >= 3), workgroup (0, 0, 0).
                # S is -inf and dP is 0 outside (seqlen_q, seqlen_k): the kernel reads clamped real rows there and discards them.
                _s_dump = []
                _dp_dump = []
                for _i in range_constexpr(NUM_S_VALS):
                    if const_expr(CAUSAL):
                        _kv_ok = _vis[_i]
                    else:
                        _colv = (
                            _kv_i32
                            + fx.Int32((_i // 8) * WMMA + L.acc_row(_i % 8, 0, V8))
                            + fx.Int32(klane) * fx.Int32(8 if V8 else 1)
                        )
                        _kv_ok = _colv < seqlen_k_i32
                    _sd = fastmath.mul(fx.Float32(s_acc_raw[_i]), sm_scale)
                    if const_expr(BIAS):
                        _sd = fastmath.add(_sd, b_f32[_i])
                    _s_dump.append((_kv_ok & _q_in).select(_sd, c_neg_inf))
                    _dp_dump.append(
                        (_kv_ok & _q_in).select(fx.Float32(dp_raw[_i]), fx.Float32(0.0))
                    )
                _tiles = (
                    _s_dump,
                    [fx.Float32(p_vals[_i]) for _i in range_constexpr(NUM_S_VALS)],
                    _dp_dump,
                    [fx.Float32(ds_vals[_i]) for _i in range_constexpr(NUM_S_VALS)],
                )
                _it = (
                    (_kv_i32 - _col_l >> fx.Int32(BLOCK_N_LOG2))
                    if CAUSAL
                    else (_kv_i32 >> fx.Int32(BLOCK_N_LOG2))
                )
                _c0 = _it == fx.Int32(0)
                _c1 = _it == fx.Int32(1)
                _c2 = (_it >= fx.Int32(2)) & (_it == _n_iter - fx.Int32(1))
                _slot = fx.Int32(0) if _c0 else (fx.Int32(1) if _c1 else fx.Int32(2))
                _do = (_c0 | _c1 | _c2) & _is_wg0
                _slot_base = fx.Int64(DUMP_HEADER_WORDS) + fx.Int64(_slot) * fx.Int64(
                    DUMP_SLOT_WORDS
                )
                if _do & (shard_id == fx.Index(0)):
                    _dump_s_tiles(_slot_base, _tiles)
            # ==== dS^T as the B operand of GEMM3: the xor-16 exchange (gfx11), then the 16-bit narrowing ====
            ds_packs = []
            ds_halves = []
            for _acc in range_constexpr(NUM_S_ACCS):
                _own = [fx.Float32(ds_vals[_acc * 8 + j]) for j in range_constexpr(8)]
                _full = L.acc_to_operand(_own, exchange, klane_is_zero, V8)
                _halves = []
                for _h in range_constexpr(OW // 8):
                    _sl = _full[_h * 8 : (_h + 1) * 8]
                    if const_expr(dtype_str == "bf16"):
                        _halves.append(
                            (cmn.bf16_rne_pack_v8 if RNE else fmha.bf16_trunc_pack_v8)(
                                _sl, elem_dtype
                            )
                        )
                    else:
                        _halves.append(
                            Vec.from_elements(
                                [
                                    fx.Float32(_sl[j]).to(elem_dtype)
                                    for j in range_constexpr(8)
                                ],
                                elem_dtype,
                            ).ir_value()
                        )
                ds_packs.append(_cat(_halves))
                ds_halves.append(_halves)

            if const_expr(STORE_DB):
                # dB = dS, narrowed exactly as the GEMM3 operand (the packs). `_halves[0]` / `[1]` hold tile rows 0-7 / 8-15 of the lane's q column
                # (gfx11; gfx12 holds rows 8 klane + j in `_halves[0]`), so the lane's 8 contiguous kv are `kv0 + 8 klane + j`.
                with _if_then(_scf.IfOp(fx.as_ir_value(db_on))):
                    for _acc in range_constexpr(NUM_S_ACCS):
                        _hh = ds_halves[_acc]
                        if const_expr(V8):
                            _rows8 = _hh[0]
                        else:
                            _rows8 = klane_is_zero.select(Vec(_hh[0]), Vec(_hh[1]))
                        for _j in range_constexpr(8):
                            _dcol = (
                                _kv_i32
                                + fx.Int32(_acc * WMMA + _j)
                                + fx.Int32(klane) * fx.Int32(8)
                            )
                            _dptr = fmha.split_ptr(
                                db_ptr,
                                db_tile_base,
                                db_row_off + fx.Index(_dcol),
                                elem_type,
                            )
                            if const_expr(_MASK_STEPS):
                                with _if_then(
                                    _scf.IfOp(fx.as_ir_value(_dcol < seqlen_k_i32))
                                ):
                                    _pointer_store(Vec(_rows8)[_j], _dptr)
                            else:
                                _pointer_store(Vec(_rows8)[_j], _dptr)

            def _load_kt(dc_val, acc_idx):
                """One K^T A operand: row d = lane16, K = the kv values of the 16-kv step `acc_idx`."""
                d_pos = shard_qk_off + fx.Index(dc_val * WMMA) + lane16
                if const_expr(KT_STAGED):
                    return _lds_load_vec(
                        lds_kv,
                        L.lds_rowmajor_operand(
                            LDS_KT_BASE,
                            KT_STRIDE,
                            d_pos,
                            fx.Index(acc_idx * WMMA),
                            klane,
                            V8,
                        ),
                        OW,
                        v4_type,
                    )
                _offs = L.lds_transposed_operand(
                    0, K_STRIDE, d_pos, fx.Index(acc_idx * WMMA), klane, V8
                )
                return Vec.from_elements(
                    [fx.ptr_load(lds_kv + fx.Int32(_o)) for _o in _offs], elem_dtype
                ).ir_value()

            # ==== GEMM3: dQ^T[d][q] += K^T[d][kv] . dS^T[kv][q] ====
            for dc in range_constexpr(D_CHUNKS):
                for a_idx in range_constexpr(NUM_S_ACCS):
                    dq_accs[dc] = wmma_acc(
                        _load_kt(dc, a_idx), ds_packs[a_idx], dq_accs[dc]
                    )

            if const_expr(DUMP):
                if _do:
                    _dump_acc(_slot_base, dq_accs)

            return [dq_accs[i] for i in range_constexpr(D_CHUNKS)]

        init_args = [fx.as_ir_value(c_zero_v8f32) for _ in range_constexpr(D_CHUNKS)]
        loop_results = init_args
        # Plain: the full tiles, then the ragged tail tile (the column mask). Causal: the three block runs of the Q tile (masked, full, masked).
        for _run in range_constexpr(len(_runs)):
            _r0, _rn, _rmask = _runs[_run]
            if const_expr(CAUSAL):
                _start = common_utils.smax(_r0, fx.Int32(0))
                _run_lo = fx.Index(_start)
                _run_hi = fx.Index(_start + _rn * fx.Int32(BLOCK_N))
            else:
                _run_lo = fx.Index(0) if _run == 0 else _full_end
                _run_hi = _full_end if _run == 0 else kv_upper
            for kv_block_start, inner_iter_args in range(
                _run_lo, _run_hi, BLOCK_N, init=loop_results
            ):
                loop_results = yield kv_loop_body(
                    kv_block_start, inner_iter_args, _rmask
                )
            # An scf.for with one carried value hands it back unwrapped.
            if const_expr(D_CHUNKS == 1):
                loop_results = [loop_results]

        # ---- scale and store dQ ----
        # sm_scale is applied once here: the exponent already carries sm_scale * log2e and dS feeds a 16-bit operand.
        dq_tile_base = dq_tbase(start_q)

        def write_dq(row, col, val):
            store_global_v8(dq_ptr, dq_tile_base, dq_toff(row, col), val)

        def write_dq_elem(row, col, val):
            _pointer_store(
                val, fmha.split_ptr(dq_ptr, dq_tile_base, dq_toff(row, col), elem_type)
            )

        def store_dq_chunk(row, col, val):
            """One 8-column dQ chunk. Padded: a chunk wholly inside `hdim_qk` is one 16-byte store and a chunk that straddles it stores only its
            live elements, so the region past `hdim_qk` (NaN-poisoned by the caller) is never written.
            """
            if const_expr(not PADDED_HEAD):
                write_dq(row, col, val)
                return
            with _if_then(_scf.IfOp(fx.as_ir_value(qk_cols.valid(col + fx.Index(7))))):
                write_dq(row, col, val)
            straddle = qk_cols.valid(col) & (
                fx.Int64(col + fx.Index(7)) >= fx.Int64(hdim_qk)
            )
            with _if_then(_scf.IfOp(fx.as_ir_value(straddle))):
                for j in range_constexpr(8):
                    with _if_then(
                        _scf.IfOp(fx.as_ir_value(qk_cols.valid(col + fx.Index(j))))
                    ):
                        write_dq_elem(row, col + fx.Index(j), Vec(val)[j])

        if _q_in:
            for dc in range_constexpr(D_CHUNKS):
                _own = [
                    fx.Float32(Vec(loop_results[dc])[r]) for r in range_constexpr(8)
                ]
                _rows = L.acc_to_rows8(_own, exchange, klane_is_zero, V8)
                _trunc = Vec.from_elements(
                    [
                        fx.Float32(fastmath.mul(_rows[j], sm_scale)).to(elem_dtype)
                        for j in range_constexpr(8)
                    ],
                    elem_dtype,
                ).ir_value()
                _col = shard_qk_off + fx.Index(dc * WMMA) + L.rows8_origin(klane)
                store_dq_chunk(q_row_in_tile, _col, _trunc)

    # ---- kernel + launcher -------------------------------------------------------------------------------------------------------
    _passthrough = [["denormal-fp-math-f32", "preserve-sign,preserve-sign"]]
    _value_attrs = {
        "rocdl.waves_per_eu": knobs.waves_per_eu,
        "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
        "passthrough": _passthrough,
    }

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dq_gfx11_kernel(
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
                rt_flags=rt_flags,
            )
        )

    @flyc.jit
    def launch_fmha_bwd_dq_gfx950(
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
        num_q_blocks = (fx.Index(max_seqlen_q) + BLOCK_M - 1) // BLOCK_M
        fmha_bwd_dq_gfx11_kernel(
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
            grid=(num_head_q, num_q_blocks, bs_idx),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _entry = launch_fmha_bwd_dq_gfx950

    if DUMP:

        @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
        def fmha_bwd_dq_gfx11_dump_kernel(
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
                    rt_flags=rt_flags,
                    dump=dump,
                )
            )

        @flyc.jit
        def launch_fmha_bwd_dq_gfx11_dump(
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
            num_q_blocks = (fx.Index(max_seqlen_q) + BLOCK_M - 1) // BLOCK_M
            fmha_bwd_dq_gfx11_dump_kernel(
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
                grid=(num_head_q, num_q_blocks, bs_idx),
                block=(BLOCK_SIZE, 1, 1),
                stream=stream,
            )

        _entry = launch_fmha_bwd_dq_gfx11_dump

    def _launch(*packed):
        """`packed` = the launcher's arguments in wire order (the harness's `gate2.arg_values` order), stream last."""
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return _entry(*packed)

    _launch.knobs = knobs
    _launch.block_m = BLOCK_M
    _launch.cache_tag = _cache_tag
    return _launch
