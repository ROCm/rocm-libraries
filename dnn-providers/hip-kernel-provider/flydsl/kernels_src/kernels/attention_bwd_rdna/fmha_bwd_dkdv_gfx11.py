# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); its sources are listed below.
#
# Provenance carried over from the sweep snapshot:
# Adapted from: https://github.com/ROCm/aiter
#   commit:   8253efc4059516f492df597b80ce5311e9fda8bd
#   path:     aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py
# via the gfx11 forward `attention_rdna/flash_attn_func_rdna.py` (ROCm/rocm-libraries PR #14111 @ d850254d45e: WMMA ABI split, xor-16
# exchange) and the AOTriton gfx1201 dK/dV kernel `fmha_bwd_dkdv_gfx1201_kernel.py` (MIT, reference for the transposed loop only;
# its notice is kept in `fmha_tuning_bwd_dkdv_gfx11.py`, where `lds_bytes` and the legality walk-down are re-expressed).

"""dK / dV backward attention for gfx11 (RDNA3 / 3.5, `gfx11-generic`) and gfx12 (`WMMA_V8`): the transposed loop of the forward.

The gfx950 runtime ABI (`fmha_bwd_dkdv_gfx950.py`: argument names, order and annotations of the launcher); the launcher keeps its gfx950
name. K and V are resident (one 16-row KV tile per team of waves), Q and dO stream through LDS, dK^T and dV^T accumulate in registers.

Layout (`rdna_layout_gfx11.py`, orientation `Q_ROWS`). WMMA 16x16x16, wave32, `WMMA_V8 = False` on gfx11: A and B operands carry 16 K
values per lane (lanes 16-31 duplicate lanes 0-15), accumulator element r of a lane is row `2r + klane`, column `lane % 16`.

    S    [q][kv]  = Q . K^T      A = Q rows (K = d, row-major LDS)         B = K pack (resident, lane l16 = kv)
    dP   [q][kv]  = dO . V^T     A = dO rows                               B = V pack
    dV^T [d][kv] += dO^T . P     A = dO^T (K = q; LDS transposed tile)     B = P through `acc_to_operand` (xor-16 exchange)
    dK^T [d][kv] += Q^T . dS     A = Q^T                                   B = dS through `acc_to_operand`

S, P, dP and dS have rows = q, so LSE and Delta are per accumulator element (row `2r + klane`, staged once per Q block in LDS); the
accumulators have rows = head dim, so the epilogue (`acc_to_rows8`, one xor-16 trade) stores 8 contiguous head-dim elements per lane.
The softmax scale is applied to dK at the epilogue.

Residency (role-owned). A team of waves owns 16 KV rows: 1 wave unsplit (K and V packs, dK and dV accumulators), 2 under
`split_head_dim`, `2 * contraction_shards` (C) with the shards. Wave (shard c, role r): the K role holds the K pack of columns
`[c hd/C, (c+1) hd/C)` and the dK accumulator of the same columns, the V role the V pack and the dV accumulator of its columns. S
(dP) is the sum of the K (V) role waves' partials: every wave writes its partial to its LDS slot, one barrier, every wave reads the
slots it needs (C = 1: the partner's, a relay; C > 1: the 2C partials, summed per role), so every wave of a team has S and dP and
computes the same elementwise P and dS. The accumulating GEMM of a wave contracts q with its own role's operand: dS (dK role) or P
(dV role) as the B operand, Q^T or dO^T as the A operand, its shard's columns only. The only column constraint is that hd / C is a
multiple of 16.

LDS: Q, dO row-major (pitch hd + 4), Q^T, dO^T (`transposed_source="tile"`, written by the staging thread from the registers it loaded:
gfx11 has no `global_load_tr`; or `"derived"`: the A operand is read strided out of the row-major tile and the transposed tiles do not
exist), the Q block's LSE (premultiplied by log2 e) and Delta, and the team exchange.

bf16: P and dS are narrowed by truncation (`fmha.bf16_trunc_pack_v8`, round toward zero) unless the `rne` knob is set (`rne=1`: round to nearest
even, `fmha_common_gfx11.bf16_rne_pack_v8`); truncation leaves the bf16 dK/dV error at 1.6-1.7x a round-to-nearest reference.

Dropout counter: the plane base is `Philox.grid_plane`'s 64-bit product by default; `rt_flags` bit 2 (`fmha_common_gfx11.RT_FLAG_AOTRITON_CTR`)
selects AOTriton's sign-extended 32-bit plane product, for a backward that follows an AOTriton forward.

Scope: classes `plain`, `causal` and `bias`, each with baked dropout (`bk,1`) or without, exact head dims (hdim_qk = hdim_vo = tile) or
padded (`padded=1`: the runtime `hdim_qk` / `hdim_vo` <= tile), any sequence lengths, MHA / GQA and varlen addressing through the shared
decoders. Other classes and `dform=rt` raise `ValueError`.

Padded head: a column at or past its runtime extent is zero on every load (K, V, Q, dO: the products and the transposed tiles carry
zeros, never the caller's bytes) and is never stored. Q, K and dK run to `hdim_qk`, dO, V and dV to `hdim_vo`; under `split_head_dim`
each wave masks at its own role's bound (K role `hdim_qk`, V role `hdim_vo`).

Dropout (compile time, `meta.dropout`): the mask is the gfx950 map (`philox.py`, Philox 4x32-10 with a 64-bit counter; the width table falls
back to 32 for gfx11). Element (q, kv) of sequence z and query head h draws counter `offset + (z Hq + h) max_seqlen_q ceil(max_seqlen_k / 4) +
q ceil(max_seqlen_k / 4) + kv / 4`, slot `kv % 4`, `offset = *philox_offset1 + philox_offset2`, seed `*philox_seed_ptr`; `idropout_p` is the i32
keep threshold (`philox.dropout_threshold`), `dropout_scale` = 1 / (1 - p). The lane holds one kv column and eight q rows, so each element is
its own Philox call (4x redundant; the slot is a loop-invariant select). dV takes `keep * scale * P`, dS = P (keep * scale * dP - Delta) with the
undropped P (the softmax denominator is undropped and baked into LSE'). A GQA group draws a distinct plane per query head.

GQA / varlen: the workgroup of (kv tile, kv head, sequence) loops over its group's query heads and their Q blocks in one flat
loop (iteration -> (group member, block)), accumulating dK / dV of the KV head in registers in a fixed order: deterministic, no atomics.
Q and K sides decode independently (`fmha.decode_addressing`: dense, CUMULATIVE, INDIVIDUAL + ARRAY positions); LSE and Delta are HT or TH.

Class `causal` (compile time) takes the mask at run time through `window_left` / `window_right`, the gfx950 ABI's pair: the causal
sentinels resolve per sequence (`fmha.resolve_window`: top-left wl = Sq, wr = 0; bottom-right wl = Sq, wr = Sk - Sq), and kv column c is
visible to q row i iff i - wl <= c <= i + wr (two-sided windows, `rdna_layout_gfx11.kv_window` with the `win` kind). A KV tile visits only
the Q blocks the union of its columns' bands reaches (`n_blocks` may be 0, the GQA group loop and the dump header count executed
blocks); the mask is applied to P and dS after the exponential (a row no key reaches carries LSE' = +inf and every element of it is
dead), never to S. A workgroup whose tile sees no query runs no block and still stores its zero accumulators, so every dK/dV element of
the sequence is written.

Class `bias` (compile time; mutually exclusive with `causal`, as on gfx950) adds the caller's bias, in natural units, to the scaled
score: P = exp2((sm_scale S + bias - lse) log2e) with the natural lse. The bias is `(batch, q head, q row, kv col)`
at `stride_b_batch` / `stride_b_head` / `stride_b_seq_q` (the varlen q token origin rides `stride_b_seq_q`), kv column contiguous; a
broadcast dim has stride 0 (`rdna_layout_gfx11.bias_offset`). The lane of S element r reads the bias at (q row `acc_row(r)` of the
sub-tile, its kv column): 8 scalar loads, each a plain global load at a q row and kv column clamped into the sequence (a row or column
past it is dead and its P is selected to 0, so the clamped value never reaches the answer). There is no byte bound derived from
`seqlen_q * stride_b_seq_q`: a row-broadcast bias (`stride_b_seq_q == 0`, shape `(B, H, 1, Sk)`) is read in full, every q row at the
same `seqlen_k` entries, where a bound pinned to 0 would read zeros. The exponent chain drops `nnan` (as
gfx950's `fm_bias` drops `ninf`: a `-inf` bias, the caller's "never attend", must reach `exp2`, which returns exactly 0). dK/dV
does not store dB; dB = dS is the dQ kernel's store.

`dump=1` object (debug): one extra launcher argument `dump: fx.Pointer` before `stream`; workgroup (0, 0, 0) stores its S, P, dP, dS
and accumulator tiles, lane ordered, at iterations 0, 1 and last, in the layout of `h/arms_rdna_dump.py` (version 1). Dump code is a
predicated store; every index expression is shared with the production object.
"""

import math as host_math
import os
import sys


from kernels.attention_bwd_rdna import fmha_common_gfx1201 as fmha
from kernels.attention_rdna4.philox import Philox, keep_mask
from kernels.attention_bwd_rdna import fmha_common_gfx11 as cmn
from kernels.attention_bwd_rdna import rdna_layout_gfx11 as lay
from kernels.attention_bwd_rdna.fmha_tuning_bwd_dkdv_gfx11 import (
    LDS_PAD,
    ROWS_PER_WAVE,
    BwdDkDvKnobs,
    lds_layout,
)  # noqa: F401
from kernels.common import buffer_ops

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf as _scf
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

KERNEL_NAME = "fmha_bwd_dkdv_gfx11_kernel"
_LOG2E = host_math.log2(host_math.e)

# `amdgpu-use-amdgpu-trackers`: the option the register-pressure skeletons were measured with and the generic-target build path sets.
_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
        "amdgpu-use-amdgpu-trackers": True,
    },
}

# Dump buffer (`h/arms_rdna_dump.py`, version 1).
_DUMP_MAGIC = 0x52444E41
_DUMP_NOT_STORED = -1  # 0xFFFFFFFF as a signed word
_DUMP_HEADER_F32 = 64  # 256 bytes
_DUMP_TILE_F32 = 256  # 32 lanes x 8 elements


def _pointer_load(result_type, ptr):
    return _llvm.LoadOp(result_type, fmha.llvm_value(ptr)).result


def _pointer_store(value, ptr):
    return _llvm.StoreOp(fmha.llvm_value(value), fmha.llvm_value(ptr))


def _cat(parts):
    """One vector from adjacent vectors (element order kept)."""
    parts = [Vec(p) for p in parts]
    while len(parts) > 1:
        merged = []
        for i in range(0, len(parts), 2):
            n = parts[i].numel + parts[i + 1].numel
            merged.append(parts[i].shuffle(parts[i + 1], list(range(n))))
        parts = merged
    return parts[0]


def _when(cond, body):
    """`body()` under an `scf.if` on `cond` (a helper outside the kernel body must build the IfOp itself)."""
    with cmn._if_then(_scf.IfOp(fx.as_ir_value(cond))):
        body()


def build_fmha_bwd_dkdv_gfx11_module_primary(meta, knobs, runtime_axes=()):
    """Build the gfx11 dK/dV kernel for a resolved (meta, knobs) pair; `meta` is `variant_gfx11.Meta`, `knobs` the record of
    `fmha_tuning_bwd_dkdv_gfx11.bwd_dkdv_knobs`."""
    if not knobs.legal:
        raise ValueError(f"the dK/dV knob record is not legal: {knobs.illegal_reason}")
    if tuple(runtime_axes):
        raise ValueError(
            f"runtime axes {tuple(runtime_axes)} are not built: dropout is baked (dform=bk)"
        )
    dtype_str = meta.dtype
    if dtype_str not in ("f16", "bf16"):
        raise ValueError(f"dK/dV supports f16/bf16, got {dtype_str!r}")
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

    BLOCK_DMODEL = knobs.block_dmodel
    BLOCK_DMODEL_V = knobs.block_dmodel_v
    BLOCK_M = knobs.block_q
    NUM_WAVES = knobs.num_waves
    SPLIT_HEAD_DIM = knobs.split_head_dim
    CONTRACTION_SHARDS = knobs.contraction_shards
    TRANSPOSED_SOURCE = knobs.transposed_source
    KEEP_TR_TILES = TRANSPOSED_SOURCE == "tile"
    WAVES_PER_EU = knobs.waves_per_eu
    SCHED_STRATEGY = knobs.sched_strategy
    DUMP = bool(knobs.dump)
    RNE = bool(knobs.rne)
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

    WARP_SIZE = 32
    WMMA_M = WMMA_N = WMMA_K = 16
    # `WMMA_V8` of the vendored forward: gfx12 passes 8 K values per lane, gfx11 all 16.
    WMMA_V8 = lay.wmma_v8_for_arch(get_rocm_arch())
    OPW = lay.op_width(WMMA_V8)

    TEAM_WAVES = knobs.team_waves
    NUM_TEAMS = NUM_WAVES // TEAM_WAVES
    BLOCK_N = ROWS_PER_WAVE * NUM_TEAMS  # BLOCK_KV
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE
    D_STEPS = BLOCK_DMODEL // WMMA_K
    DV_STEPS = BLOCK_DMODEL_V // WMMA_K
    Q_SUBTILES = BLOCK_M // WMMA_M
    # Role-owned residency: a wave of the split holds D_STEPS_SHARD packs and accumulators.
    D_STEPS_SHARD = D_STEPS // CONTRACTION_SHARDS
    N_ACC = D_STEPS_SHARD if SPLIT_HEAD_DIM else D_STEPS + DV_STEPS
    assert BLOCK_M % WMMA_M == 0 and BLOCK_M > 0
    assert D_STEPS % CONTRACTION_SHARDS == 0 and (
        not SPLIT_HEAD_DIM or BLOCK_DMODEL == BLOCK_DMODEL_V
    )

    # ---- LDS layout (one definition, shared with `lds_bytes`) ----
    LY = lds_layout(
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_DMODEL_V,
        NUM_WAVES,
        SPLIT_HEAD_DIM,
        TRANSPOSED_SOURCE,
    )
    QRM_BASE, QRM_STRIDE = LY.qrm_base, LY.qrm_stride
    QTR_BASE, QTR_STRIDE = LY.qtr_base, LY.qtr_stride
    DORM_BASE, DORM_STRIDE = LY.dorm_base, LY.dorm_stride
    DOTR_BASE, DOTR_STRIDE = LY.dotr_base, LY.dotr_stride
    ROWQ_BYTE0 = LY.rowq_base * 2
    PDS_BYTE0 = LY.pds_base * 2
    PDS_SLOT_F32 = LY.pds_slot_f32
    PDS_TEAM_F32 = TEAM_WAVES * PDS_SLOT_F32
    PDS_BUF_F32 = NUM_WAVES * PDS_SLOT_F32
    LDS_TOTAL = LY.total
    VEC_WIDTH = 8
    _ROWQ_BATCHES = (BLOCK_M + BLOCK_SIZE - 1) // BLOCK_SIZE

    QRM_TPR, QRM_RPB, QRM_BATCHES, QRM_GUARD = fmha.load_geom(
        BLOCK_DMODEL, VEC_WIDTH, BLOCK_SIZE, BLOCK_M
    )
    DORM_TPR, DORM_RPB, DORM_BATCHES, DORM_GUARD = fmha.load_geom(
        BLOCK_DMODEL_V, VEC_WIDTH, BLOCK_SIZE, BLOCK_M
    )

    # ---- Dump buffer geometry ----
    N_RES = BLOCK_N // 16
    N_STR = BLOCK_M // 16
    N_DK = BLOCK_DMODEL // 16
    N_DV = BLOCK_DMODEL_V // 16
    _DUMP_S_F32 = N_RES * N_STR * _DUMP_TILE_F32
    DUMP_OFF_S, DUMP_OFF_P, DUMP_OFF_DP, DUMP_OFF_DS = (
        0,
        _DUMP_S_F32,
        2 * _DUMP_S_F32,
        3 * _DUMP_S_F32,
    )
    DUMP_OFF_DVACC = 4 * _DUMP_S_F32
    DUMP_OFF_DKACC = DUMP_OFF_DVACC + N_RES * N_DV * _DUMP_TILE_F32
    DUMP_SLOT_F32 = DUMP_OFF_DKACC + N_RES * N_DK * _DUMP_TILE_F32

    _DUMP_HEADER_WORDS = [
        (0, _DUMP_MAGIC),
        (1, 1),
        (3, BLOCK_N),
        (4, BLOCK_M),
        (6, BLOCK_DMODEL),
        (7, BLOCK_DMODEL_V),
        (8, _DUMP_NOT_STORED),
        (9, _DUMP_NOT_STORED),
        (10, _DUMP_NOT_STORED),
    ]

    elem_numeric_cls = {"f16": fx.Float16, "bf16": fx.BFloat16}[dtype_str]
    fastmath = fmha.FastMath("noninf")
    # Exponent chain of a bias object: contract only. No `nnan`/`ninf` (a -inf bias entry must reach exp2) and no `reassoc` (it would let the
    # compiler undo "combine in natural units, scale by log2 e last").
    fastmath_e = fmha.FastMath("safe")
    fastmath_e.flags = fx.arith.FastMathFlags.contract
    if not BIAS:
        fastmath_e = fastmath

    @fx.struct
    class SharedStorage:
        qdo: fx.Array[elem_numeric_cls, LDS_TOTAL, 16]

    def _body(
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
        DUMP_PTR,
    ):
        elem_type = elem_numeric_cls.ir_type
        elem_dtype = elem_numeric_cls
        f32_ty = ir.F32Type.get()
        i32_ty = ir.IntegerType.get_signless(32)

        def _to_global_ptr_i64(ptr):
            return fx.as_ir_value(fx.Int64(fx.ptrtoint(ptr)))

        q_ptr = fmha.pointer_to_llvm_ptr(Q)
        do_ptr = fmha.pointer_to_llvm_ptr(DO)
        k_ptr = fmha.pointer_to_llvm_ptr(K)
        v_ptr = fmha.pointer_to_llvm_ptr(V)
        dk_ptr = fmha.pointer_to_llvm_ptr(DK)
        dv_ptr = fmha.pointer_to_llvm_ptr(DV)
        l_ptr = fmha.pointer_to_llvm_ptr(LSE)
        delta_ptr = fmha.pointer_to_llvm_ptr(Delta)
        if const_expr(DUMP):
            dump_ptr = fmha.pointer_to_llvm_ptr(DUMP_PTR)
        if const_expr(BIAS):
            bias_ptr = fmha.pointer_to_llvm_ptr(Bias)

        v4_ty = Vec.make_type(4, elem_dtype)
        vxf16_type = Vec.make_type(VEC_WIDTH, elem_dtype)

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

        seqlen_q_v = fx.Index(seqlen_q_i32)
        seqlen_k_v = fx.Index(seqlen_k_i32)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_qdo = lds.qdo.ptr
        _lds_byte_base = fx.as_ir_value(fx.ptrtoint(lds_qdo))

        tid, wave_id, lane, lane16, klane = fmha.wave_lanes(WARP_SIZE)
        klane_is_zero = fx.Int32(klane) == fx.Int32(0)
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)

        def _exchange(v):
            return fx.Float32(v).shuffle_xor(fx.Int32(16), fx.Int32(WARP_SIZE))

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_dtype, fx.Float32)
        )

        def wmma_acc(a_op, b_op, c_v8):
            a_frag = fx.make_rmem_tensor(OPW, elem_dtype)
            b_frag = fx.make_rmem_tensor(OPW, elem_dtype)
            c_frag = fx.make_rmem_tensor(8, fx.Float32)
            a_frag.store(Vec(a_op))
            b_frag.store(Vec(b_op))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, a_frag, b_frag, c_frag)
            return Vec(c_frag.load())

        # One KV tile per workgroup column: grid (kv_tile, head_k, batch).
        kv_tile_idx = fx.Index(gpu.block_idx.x)
        head_k = fx.Index(gpu.block_idx.y)
        start_k_i32 = fx.Int32(kv_tile_idx) * fx.Int32(BLOCK_N)
        start_k = fx.Index(start_k_i32)

        # Under varlen the grid is sized from max_seqlen_k: whole workgroups can lie past this sequence's end.
        _alive = start_k_i32 < seqlen_k_i32
        _k_start_addr = fx.Index(start_k if _alive else fx.Index(0))

        _q_last_i32 = fmha.common_utils.smax(seqlen_q_i32 - fx.Int32(1), fx.Int32(0))
        _k_last_i32 = fmha.common_utils.smax(seqlen_k_i32 - fx.Int32(1), fx.Int32(0))
        q_seq_last = fx.Index(_q_last_i32)
        k_seq_last = fx.Index(_k_last_i32)

        # MQA/GQA: the KV head is shared by `group` query heads and dK/dV sum over all of them, folded into the Q-block loop.
        group_i32 = num_head_q // num_head_k

        q_st = (
            fx.Index(stride_q_batch),
            fx.Index(stride_q_head),
            fx.Index(stride_q_seq),
        )
        k_st = (
            fx.Index(stride_k_batch),
            fx.Index(stride_k_head),
            fx.Index(stride_k_seq),
        )
        v_st = (
            fx.Index(stride_v_batch),
            fx.Index(stride_v_head),
            fx.Index(stride_v_seq),
        )
        do_st = (
            fx.Index(stride_do_batch),
            fx.Index(stride_do_head),
            fx.Index(stride_do_seq),
        )
        dk_st = (
            fx.Index(stride_dk_batch),
            fx.Index(stride_dk_head),
            fx.Index(stride_dk_seq),
        )
        dv_st = (
            fx.Index(stride_dv_batch),
            fx.Index(stride_dv_head),
            fx.Index(stride_dv_seq),
        )

        _q_batch_v = fx.Index(q_batch)
        _k_batch_v = fx.Index(k_batch)
        _q_row_off_v = fx.Index(q_row_off)
        _k_row_off_v = fx.Index(k_row_off)

        sm_log2e = fastmath.mul(sm_scale, fx.Float32(_LOG2E))

        # Column masks: an element is valid iff its column is below the runtime extent (inactive for an exact object).
        qk_cols = fmha.MaskedAxis(
            fx.Index(hdim_qk), active=PADDED_HEAD, elem_dtype=elem_dtype
        )
        vo_cols = fmha.MaskedAxis(
            fx.Index(hdim_vo), active=PADDED_HEAD, elem_dtype=elem_dtype
        )

        def load_global_f16xN(base_ptr, base64, off32):
            return _pointer_load(
                vxf16_type, fmha.split_ptr(base_ptr, base64, off32, elem_type)
            )

        def store_global_v8f16(base_ptr, base64, off32, val):
            _pointer_store(val, fmha.split_ptr(base_ptr, base64, off32, elem_type))

        def store_global_elem(base_ptr, base64, off32, val):
            _pointer_store(val, fmha.split_ptr(base_ptr, base64, off32, elem_type))

        def store_chunk(cols, write, write_elem, row, col, val):
            """One 8-column dK / dV chunk. Padded: a chunk wholly inside the extent is one 16-byte store and a chunk that straddles it
            stores only its live elements, so the region past the extent (NaN-poisoned by the caller) is never written.
            """
            if const_expr(not PADDED_HEAD):
                write(row, col, val)
                return
            with cmn._if_then(_scf.IfOp(fx.as_ir_value(cols.valid(col + fx.Index(7))))):
                write(row, col, val)
            straddle = cols.valid(col) & (
                fx.Int64(col + fx.Index(7)) >= fx.Int64(cols._bound())
            )
            with cmn._if_then(_scf.IfOp(fx.as_ir_value(straddle))):
                for j in range_constexpr(8):
                    with cmn._if_then(
                        _scf.IfOp(fx.as_ir_value(cols.valid(col + fx.Index(j))))
                    ):
                        write_elem(row, col + fx.Index(j), Vec(val)[j])

        def load_global_f32(base_ptr, off64):
            return _pointer_load(
                f32_ty,
                buffer_ops.get_element_ptr(base_ptr, fx.Int64(off64), elem_type=f32_ty),
            )

        # ---- Team decomposition ----
        #   group_of_wave: the 16 KV rows (team) this wave belongs to;  slot = 2 * shard + role inside the team.
        group_of_wave = wave_id // fx.Index(TEAM_WAVES)
        _slot = wave_id % fx.Index(TEAM_WAVES)
        role = (
            fx.Int32(_slot % fx.Index(2)) if const_expr(SPLIT_HEAD_DIM) else fx.Int32(0)
        )
        _role0 = role == fx.Int32(0)
        kv_row_in_tile = group_of_wave * fx.Index(ROWS_PER_WAVE) + lane16
        kv_row_abs = start_k + kv_row_in_tile
        kv_row_abs_i32 = (
            start_k_i32
            + fx.Int32(group_of_wave) * fx.Int32(ROWS_PER_WAVE)
            + fx.Int32(lane16)
        )
        if const_expr(SPLIT_HEAD_DIM):
            _pds_team = fx.Int32(group_of_wave) * fx.Int32(PDS_TEAM_F32) + fx.Int32(
                lane
            )
            _pds_mine = _pds_team + fx.Int32(_slot) * fx.Int32(PDS_SLOT_F32)
            _pds_theirs = _pds_team + (fx.Int32(1) - role) * fx.Int32(PDS_SLOT_F32)
            _rm_base_off = fx.Index(role) * fx.Index(DORM_BASE - QRM_BASE)
            _tr_base_off = fx.Index(role) * fx.Index(DOTR_BASE - QTR_BASE)
            # First K step this wave's shard covers.
            _shard_k0 = fx.Int32(_slot // fx.Index(2)) * fx.Int32(D_STEPS_SHARD)

        # ---- K and V: register-resident packs, one 16-row KV tile per team ----
        _addr_kw_k = dict(
            seqlen_k=seqlen_k_v, seq_last=k_seq_last, hoist=False, clamp=True
        )
        kv_rows = fmha.MaskedAxis(seqlen_k_v)
        k_ap = fmha.Aperture(qk_cols, rows=kv_rows)
        v_ap = fmha.Aperture(vo_cols, rows=kv_rows)
        q_ap = fmha.Aperture(qk_cols, vec_width=VEC_WIDTH)
        do_ap = fmha.Aperture(vo_cols, vec_width=VEC_WIDTH)
        _kv_in, _kv_safe = k_ap.rows.gate(kv_row_abs, kv_row_in_tile)

        if const_expr(BIAS):
            # Bias column of this lane's kv row, clamped into the sequence (a column past it is dead).
            _bias_col = fx.Index(
                (kv_row_abs_i32 < seqlen_k_i32).select(kv_row_abs_i32, _k_last_i32)
            )
            _bias_pitch = fx.Index(stride_b_seq_q)

            def load_bias(base64, q_row_i32, q_ok):
                """The bias entry (f32, natural units) of absolute q row `q_row_i32` at this lane's kv column; a row past the sequence
                is clamped to the last (dead)."""
                _qc = fx.Index(q_ok.select(q_row_i32, _q_last_i32))
                return _pointer_load(
                    elem_type,
                    fmha.split_ptr(
                        bias_ptr, base64, _qc * _bias_pitch + _bias_col, elem_type
                    ),
                )

        def _read_pack(ap, fetch, k_origin):
            """The resident B operand of one K step: `OPW` contiguous head-dim values of this lane's KV row."""
            c0 = lay.op_origin(k_origin, klane, WMMA_V8)
            return _cat(
                [
                    ap.read_v8(fetch, _kv_safe, c0 + fx.Index(8 * i), _kv_in)
                    for i in range_constexpr(OPW // 8)
                ]
            )

        if const_expr(SPLIT_HEAD_DIM):
            # Role 0 holds K, role 1 holds V, selected once on plain values (the dynamic-branch trap of `fmha_common`).
            _my_addr = fx.Int64(fx.ptrtoint(K)) if _role0 else fx.Int64(fx.ptrtoint(V))
            my_ptr = _llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<1>"), fx.as_ir_value(_my_addr)
            ).result
            _my_st = tuple(_a if _role0 else _b for _a, _b in zip(k_st, v_st))
            my_tbase, my_toff, _ = fmha.make_addr_pair(
                _my_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k
            )
            my_tile_base = my_tbase(_k_start_addr)
            # The role's own head-dim bound: the K role runs to hdim_qk, the V role to hdim_vo.
            if const_expr(PADDED_HEAD):
                my_cols = fmha.MaskedAxis(
                    fx.Index(_role0.select(fx.Int32(hdim_qk), fx.Int32(hdim_vo))),
                    elem_dtype=elem_dtype,
                )
            else:
                my_cols = fmha.MaskedAxis(
                    fx.Index(hdim_qk), active=False, elem_dtype=elem_dtype
                )
            my_ap = fmha.Aperture(my_cols, rows=kv_rows)

            def fetch_my(row, col):
                return load_global_f16xN(my_ptr, my_tile_base, my_toff(row, col))

            packs = []
            for ks in range_constexpr(D_STEPS_SHARD):
                if const_expr(CONTRACTION_SHARDS > 1):
                    _kc = fx.Index((_shard_k0 + fx.Int32(ks)) * fx.Int32(WMMA_K))
                else:
                    _kc = fx.Index(ks * WMMA_K)
                packs.append(_read_pack(my_ap, fetch_my, _kc))

            # The output side of the role: dK for role 0, dV for role 1.
            _out_addr = (
                fx.Int64(fx.ptrtoint(DK)) if _role0 else fx.Int64(fx.ptrtoint(DV))
            )
            out_ptr = _llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<1>"), fx.as_ir_value(_out_addr)
            ).result
            _out_st = tuple(_a if _role0 else _b for _a, _b in zip(dk_st, dv_st))
            out_tbase, out_toff, _ = fmha.make_addr_pair(
                _out_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k
            )
        else:
            k_tbase, k_toff, _ = fmha.make_addr_pair(
                k_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k
            )
            v_tbase, v_toff, _ = fmha.make_addr_pair(
                v_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k
            )
            k_tile_base = k_tbase(_k_start_addr)
            v_tile_base = v_tbase(_k_start_addr)

            def fetch_k(row, col):
                return load_global_f16xN(k_ptr, k_tile_base, k_toff(row, col))

            def fetch_v(row, col):
                return load_global_f16xN(v_ptr, v_tile_base, v_toff(row, col))

            k_packs = [
                _read_pack(k_ap, fetch_k, fx.Index(ks * WMMA_K))
                for ks in range_constexpr(D_STEPS)
            ]
            v_packs = [
                _read_pack(v_ap, fetch_v, fx.Index(ks * WMMA_K))
                for ks in range_constexpr(DV_STEPS)
            ]

        # ---- Staging geometry: one cooperative load per tensor, row-major and (written from registers) transposed ----
        q_row_in_batch = tid // QRM_TPR
        q_col_base = (tid % QRM_TPR) * VEC_WIDTH
        do_row_in_batch = tid // DORM_TPR
        do_col_base = (tid % DORM_TPR) * VEC_WIDTH

        def _stage(ap, fetch, row, col, rm_base, rm_stride, tr_base, tr_stride):
            """One thread's 8 elements at (row, col) of the Q (dO) block into LDS: row-major, and transposed when the tile exists. Columns
            past the aperture's runtime extent are zeroed (a padded head)."""
            vec = ap.read_vec(fetch, row, col)
            fmha.lds_store_vx(lds_qdo, vec, rm_base + row * rm_stride + col, VEC_WIDTH)
            if const_expr(KEEP_TR_TILES):
                for j in range_constexpr(VEC_WIDTH):
                    fx.ptr_store(
                        Vec(vec)[j],
                        lds_qdo
                        + fx.Int32(tr_base + (col + fx.Index(j)) * tr_stride + row),
                    )

        def _lds_vec(idx, n):
            """`n` contiguous 16-bit elements from LDS element index `idx`, in 8-byte pieces (rows are 8-byte aligned only)."""
            return _cat(
                [
                    fx.ptr_load(
                        lds_qdo + fx.Int32(idx + fx.Index(4 * i)), result_type=v4_ty
                    )
                    for i in range_constexpr(n // 4)
                ]
            )

        def _tr_operand(rm_base, rm_stride, tr_base, tr_stride, d_col, q0):
            """A operand contracted over q: Q^T / dO^T row `d_col` (head dim), K values q0 ... from the transposed tile or, derived,
            strided out of the row-major tile."""
            if const_expr(KEEP_TR_TILES):
                return _lds_vec(
                    lay.lds_rowmajor_operand(
                        tr_base, tr_stride, d_col, q0, klane, WMMA_V8
                    ),
                    OPW,
                )
            offs = lay.lds_transposed_operand(
                rm_base, rm_stride, d_col, q0, klane, WMMA_V8
            )
            return Vec.from_elements(
                [fx.ptr_load(lds_qdo + fx.Int32(o)) for o in offs], elem_dtype
            )

        def _pack(vals):
            """f32 values to one 16-bit WMMA operand; bf16 by truncation (`fmha.bf16_trunc_pack_v8`) or, with the `rne` knob, round to nearest even."""
            if const_expr(dtype_str == "bf16"):
                _pk = cmn.bf16_rne_pack_v8 if RNE else fmha.bf16_trunc_pack_v8
                return _cat(
                    [
                        Vec(_pk(vals[i * 8 : (i + 1) * 8], elem_dtype))
                        for i in range_constexpr(len(vals) // 8)
                    ]
                )
            return Vec.from_elements(
                [fx.Float32(v).to(elem_dtype) for v in vals], elem_dtype
            )

        # ---- Visited Q blocks: the whole sequence (plain), or the Q blocks the mask lets reach this KV tile (causal), times the GQA group ----
        if const_expr(CAUSAL):
            # The mask is the runtime (window_left, window_right) pair with the causal sentinels resolved per sequence: kv column c is
            # visible to q row i iff i - wl <= c <= i + wr (top-left: wl = Sq, wr = 0; bottom-right: wl = Sq, wr = Sk - Sq). The
            # extents are clamped to [-Sk, Sq] and [-Sq, Sk], which never changes a visible set and keeps every sum below in i32.
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
            # Columns per row (one unsigned compare of `c - (i - wl)` against it, as `rdna_layout_gfx11.kv_window`); <= 0: no row sees a key.
            _span_w = _wl + _wr + fx.Int32(1)
            # Q rows that see some key of this tile: [q_lo, q_hi) from the union of the band over the tile's live kv columns.
            _kv_end = fmha.common_utils.smin(
                start_k_i32 + fx.Int32(BLOCK_N), seqlen_k_i32
            )
            _q_lo = fmha.common_utils.smax(start_k_i32 - _wr, fx.Int32(0))
            _q_hi = fmha.common_utils.smin(_kv_end + _wl, seqlen_q_i32)
            _first_blk = _q_lo // fx.Int32(BLOCK_M)
            _any_live = _alive & (_span_w > fx.Int32(0)) & (_q_hi > _q_lo)
            _n_blocks_i32 = _any_live.select(
                (_q_hi + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M) - _first_blk,
                fx.Int32(0),
            )
            _kv_wl = kv_row_abs_i32 + _wl
        else:
            _n_blocks_i32 = (
                (seqlen_q_i32 + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
                if _alive
                else fx.Int32(0)
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
            # This lane's kv column is fixed for the whole kernel, so is its slot of each Philox call.
            _ph_slot = kv_row_abs_i32 % fx.Int32(PHILOX_RN)

        # ---- Dump preamble: header words, written by the thread whose id is the word index ----
        if const_expr(DUMP):
            _dump_wg = (
                (fx.Int32(gpu.block_idx.x) == fx.Int32(0))
                & (fx.Int32(gpu.block_idx.y) == fx.Int32(0))
                & (z_i32 == fx.Int32(0))
            )
            _tid_i32 = fx.Int32(tid)
            _hdr = fx.Int32(0)
            for _wi in range_constexpr(len(_DUMP_HEADER_WORDS)):
                _hdr = (_tid_i32 == fx.Int32(_DUMP_HEADER_WORDS[_wi][0])).select(
                    fx.Int32(_DUMP_HEADER_WORDS[_wi][1]), _hdr
                )
            _hdr = (_tid_i32 == fx.Int32(5)).select(_total_i32, _hdr)
            if _dump_wg & (_tid_i32 < fx.Int32(_DUMP_HEADER_F32)):
                _pointer_store(
                    fx.as_ir_value(_hdr),
                    buffer_ops.get_element_ptr(
                        dump_ptr, fx.Int64(_tid_i32), elem_type=i32_ty
                    ),
                )

        def _dump_store8(off, vals):
            """Eight f32 (element r of this lane) at f32 index `off`."""
            for _r in range_constexpr(8):
                _pointer_store(
                    fx.as_ir_value(fx.Float32(vals[_r])),
                    buffer_ops.get_element_ptr(
                        dump_ptr, fx.Int64(off + fx.Int32(_r)), elem_type=f32_ty
                    ),
                )

        # ---- Loop-carried state: the accumulators ----
        init_args = [fx.as_ir_value(c_zero_v8f32) for _ in range_constexpr(N_ACC)]

        loop_results = init_args
        for _it, _iter_args in range(
            fx.Index(0), fx.Index(_total_i32), 1, init=init_args
        ):
            accs = [_iter_args[i] for i in range_constexpr(N_ACC)]

            _it_i32 = fx.Int32(_it)
            _g_i32 = _it_i32 // _nb_safe_i32
            _bi_i32 = _it_i32 - _g_i32 * _nb_safe_i32
            if const_expr(CAUSAL):
                _bi_i32 = _bi_i32 + _first_blk
            head_q = fx.Index(fx.Int32(head_k) * group_i32 + _g_i32)
            if const_expr(DROPOUT):
                _ph_plane = fx.Int32(z_i32) * fx.Int32(num_head_q) + fx.Int32(head_q)
                _ph_base = cmn.philox_plane_base(
                    _ph_off, _ph_plane, max_seqlen_q, _ph_stride, rt_flags
                )
            _q_row_start_i32 = _bi_i32 * fx.Int32(BLOCK_M)
            _q_row_start = fx.Index(_q_row_start_i32)

            _addr_kw_q = dict(
                seqlen_k=seqlen_q_v, seq_last=q_seq_last, hoist=False, clamp=True
            )
            _, _, q_addr = fmha.make_addr_pair(
                q_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw_q
            )
            _, _, do_addr = fmha.make_addr_pair(
                do_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw_q
            )
            _fetch_q = fmha.reader(q_addr, lambda b, o: load_global_f16xN(q_ptr, b, o))(
                _q_row_start
            )
            _fetch_do = fmha.reader(
                do_addr, lambda b, o: load_global_f16xN(do_ptr, b, o)
            )(_q_row_start)

            # VRAM -> LDS. Rows past seqlen_q are clamped by the address closure (a duplicate of a real row); the P mask below
            # keeps them out of the answer.
            for _b in range_constexpr(QRM_BATCHES):
                _qrow = q_row_in_batch + fx.Index(_b * QRM_RPB)
                if const_expr(QRM_GUARD):
                    _when(
                        _qrow < fx.Index(BLOCK_M),
                        lambda: _stage(
                            q_ap,
                            _fetch_q,
                            _qrow,
                            q_col_base,
                            QRM_BASE,
                            QRM_STRIDE,
                            QTR_BASE,
                            QTR_STRIDE,
                        ),
                    )
                else:
                    _stage(
                        q_ap,
                        _fetch_q,
                        _qrow,
                        q_col_base,
                        QRM_BASE,
                        QRM_STRIDE,
                        QTR_BASE,
                        QTR_STRIDE,
                    )
            for _b in range_constexpr(DORM_BATCHES):
                _dorow = do_row_in_batch + fx.Index(_b * DORM_RPB)
                if const_expr(DORM_GUARD):
                    _when(
                        _dorow < fx.Index(BLOCK_M),
                        lambda: _stage(
                            do_ap,
                            _fetch_do,
                            _dorow,
                            do_col_base,
                            DORM_BASE,
                            DORM_STRIDE,
                            DOTR_BASE,
                            DOTR_STRIDE,
                        ),
                    )
                else:
                    _stage(
                        do_ap,
                        _fetch_do,
                        _dorow,
                        do_col_base,
                        DORM_BASE,
                        DORM_STRIDE,
                        DOTR_BASE,
                        DOTR_STRIDE,
                    )

            if const_expr(BIAS):
                # (batch, q head, varlen q token origin) base of the bias plane; the group loop moves the head.
                _bias_base = (
                    _q_batch_v * fx.Index(stride_b_batch)
                    + head_q * fx.Index(stride_b_head)
                    + _q_row_off_v * fx.Index(stride_b_seq_q)
                )

            # LSE (premultiplied by log2 e; natural for a bias object) and Delta of this Q block, once per iteration for every wave.
            _rowq_base, _rowq_pitch = fmha.lse_row_addressing(
                varlen_bits, _q_batch_v, head_q, num_head_q, lse_tokens, _q_row_off_v
            )
            for _rb in range_constexpr(_ROWQ_BATCHES):
                _r = fx.Int32(tid) + fx.Int32(_rb * BLOCK_SIZE)
                if _r < fx.Int32(BLOCK_M):
                    _rr = _q_row_start_i32 + _r
                    _ok = _rr < seqlen_q_i32
                    _o = (
                        _rowq_base
                        + fx.Index(fx.Index(_rr) if _ok else fx.Index(0)) * _rowq_pitch
                    )
                    if const_expr(BIAS):
                        fmha.lds_f32_store(
                            _lds_byte_base,
                            ROWQ_BYTE0,
                            _r,
                            fx.Float32(load_global_f32(l_ptr, _o)),
                        )
                    else:
                        fmha.lds_f32_store(
                            _lds_byte_base,
                            ROWQ_BYTE0,
                            _r,
                            fastmath.mul(
                                fx.Float32(load_global_f32(l_ptr, _o)),
                                fx.Float32(_LOG2E),
                            ),
                        )
                    fmha.lds_f32_store(
                        _lds_byte_base,
                        ROWQ_BYTE0,
                        _r + fx.Int32(BLOCK_M),
                        load_global_f32(delta_ptr, _o),
                    )
            gpu.barrier()

            if const_expr(DUMP):
                # The executed position of this iteration is `_it`; the slot is its label (0, 1, last when n_iter >= 3), -1 if none.
                _dslot = (_it_i32 == fx.Int32(0)).select(
                    fx.Int32(0),
                    (_it_i32 == fx.Int32(1)).select(
                        fx.Int32(1),
                        (_it_i32 == _total_i32 - fx.Int32(1)).select(
                            fx.Int32(2), fx.Int32(-1)
                        ),
                    ),
                )
                _dump_on = _dump_wg & (_dslot >= fx.Int32(0))
                _dump_base = fx.Int32(_DUMP_HEADER_F32) + _dslot * fx.Int32(
                    DUMP_SLOT_F32
                )
                if _dump_on & (_tid_i32 == fx.Int32(8) + _dslot):
                    _pointer_store(
                        fx.as_ir_value(_bi_i32),
                        buffer_ops.get_element_ptr(
                            dump_ptr, fx.Int64(fx.Int32(8) + _dslot), elem_type=i32_ty
                        ),
                    )
                _dump_s_tile = fx.Int32(group_of_wave) * fx.Int32(
                    N_STR * _DUMP_TILE_F32
                ) + fx.Int32(lane) * fx.Int32(8)
                _dump_s_writer = _dump_on & (fx.Int32(_slot) == fx.Int32(0))

            for qsub in range_constexpr(Q_SUBTILES):
                _qs = qsub * WMMA_M

                # ==== GEMM1 (and dP): partial or full S [q][kv] = Q . K^T ====
                if const_expr(SPLIT_HEAD_DIM):
                    # One GEMM per wave: role 0 makes S, role 1 makes dP, over its shard's columns.
                    _mine = fx.as_ir_value(c_zero_v8f32)
                    for ks in range_constexpr(D_STEPS_SHARD):
                        if const_expr(CONTRACTION_SHARDS > 1):
                            _kc = fx.Index(
                                (_shard_k0 + fx.Int32(ks)) * fx.Int32(WMMA_K)
                            )
                        else:
                            _kc = fx.Index(ks * WMMA_K)
                        _a_op = _lds_vec(
                            lay.lds_rowmajor_operand(
                                QRM_BASE,
                                QRM_STRIDE,
                                fx.Index(_qs) + lane16,
                                _kc,
                                klane,
                                WMMA_V8,
                            )
                            + _rm_base_off,
                            OPW,
                        )
                        _mine = wmma_acc(_a_op, packs[ks], _mine)
                    _buf = (qsub % LY.pds_bufs) * PDS_BUF_F32
                    for _si in range_constexpr(8):
                        fmha.lds_f32_store(
                            _lds_byte_base,
                            PDS_BYTE0,
                            _pds_mine + fx.Int32(_buf + _si * WARP_SIZE),
                            fx.Float32(Vec(_mine)[_si]),
                        )
                    gpu.barrier()
                    if const_expr(CONTRACTION_SHARDS == 1):
                        # Relay: write mine, read the partner's; both waves then hold S and dP.
                        _other = [
                            fmha.lds_f32_load(
                                _lds_byte_base,
                                PDS_BYTE0,
                                _pds_theirs + fx.Int32(_buf + _si * WARP_SIZE),
                            )
                            for _si in range_constexpr(8)
                        ]
                        _s_vals = []
                        _dp_vals = []
                        for _si in range_constexpr(8):
                            _mv = fx.Float32(Vec(_mine)[_si])
                            _ov = fx.Float32(_other[_si])
                            _s_vals.append(_role0.select(_mv, _ov))
                            _dp_vals.append(_role0.select(_ov, _mv))
                    else:
                        # Reduce: slot = 2 * shard + role, so the even slots sum to S and the odd ones to dP.
                        _s_vals = []
                        _dp_vals = []
                        for _si in range_constexpr(8):
                            _s_sum = None
                            _dp_sum = None
                            for _c_sh in range_constexpr(CONTRACTION_SHARDS):
                                _sv_c = fmha.lds_f32_load(
                                    _lds_byte_base,
                                    PDS_BYTE0,
                                    _pds_team
                                    + fx.Int32(
                                        _buf
                                        + (2 * _c_sh) * PDS_SLOT_F32
                                        + _si * WARP_SIZE
                                    ),
                                )
                                _dv_c = fmha.lds_f32_load(
                                    _lds_byte_base,
                                    PDS_BYTE0,
                                    _pds_team
                                    + fx.Int32(
                                        _buf
                                        + (2 * _c_sh + 1) * PDS_SLOT_F32
                                        + _si * WARP_SIZE
                                    ),
                                )
                                _s_sum = (
                                    _sv_c if _c_sh == 0 else fastmath.add(_s_sum, _sv_c)
                                )
                                _dp_sum = (
                                    _dv_c
                                    if _c_sh == 0
                                    else fastmath.add(_dp_sum, _dv_c)
                                )
                            _s_vals.append(_s_sum)
                            _dp_vals.append(_dp_sum)
                else:
                    s_acc = fx.as_ir_value(c_zero_v8f32)
                    for ks in range_constexpr(D_STEPS):
                        _qa = _lds_vec(
                            lay.lds_rowmajor_operand(
                                QRM_BASE,
                                QRM_STRIDE,
                                fx.Index(_qs) + lane16,
                                fx.Index(ks * WMMA_K),
                                klane,
                                WMMA_V8,
                            ),
                            OPW,
                        )
                        s_acc = wmma_acc(_qa, k_packs[ks], s_acc)
                    dp_acc = fx.as_ir_value(c_zero_v8f32)
                    for ks in range_constexpr(DV_STEPS):
                        _da = _lds_vec(
                            lay.lds_rowmajor_operand(
                                DORM_BASE,
                                DORM_STRIDE,
                                fx.Index(_qs) + lane16,
                                fx.Index(ks * WMMA_K),
                                klane,
                                WMMA_V8,
                            ),
                            OPW,
                        )
                        dp_acc = wmma_acc(_da, v_packs[ks], dp_acc)
                    _s_vals = [
                        fx.Float32(Vec(s_acc)[_si]) for _si in range_constexpr(8)
                    ]
                    _dp_vals = [
                        fx.Float32(Vec(dp_acc)[_si]) for _si in range_constexpr(8)
                    ]

                # ==== P and dS: element r of a lane is q row `acc_row(r)` of this sub-tile at kv column `lane16` ====
                _kv_dead = kv_row_abs_i32 >= seqlen_k_i32
                if const_expr(BIAS):
                    # All eight loads first, then the converts.
                    _bias_raw = [
                        load_bias(
                            _bias_base,
                            _q_row_start_i32
                            + fx.Int32(_qs)
                            + fx.Int32(lay.acc_row(_si, klane, WMMA_V8)),
                            (
                                _q_row_start_i32
                                + fx.Int32(_qs)
                                + fx.Int32(lay.acc_row(_si, klane, WMMA_V8))
                            )
                            < seqlen_q_i32,
                        )
                        for _si in range_constexpr(8)
                    ]
                    _bias_nat = [
                        fx.Float32(elem_dtype(_b).to(fx.Float32)) for _b in _bias_raw
                    ]
                _p_vals = []
                _ds_vals = []
                _pdrop_vals = []
                _dpd_vals = []
                _s_dump = []
                for _si in range_constexpr(8):
                    _lrow = fx.Int32(_qs) + fx.Int32(lay.acc_row(_si, klane, WMMA_V8))
                    _q_row_i32 = _q_row_start_i32 + _lrow
                    _q_ok = _q_row_i32 < seqlen_q_i32
                    _lse2 = fx.Float32(
                        fmha.lds_f32_load(_lds_byte_base, ROWQ_BYTE0, _lrow)
                    )
                    _di = fx.Float32(
                        fmha.lds_f32_load(
                            _lds_byte_base, ROWQ_BYTE0, _lrow + fx.Int32(BLOCK_M)
                        )
                    )
                    # p = exp2(sm_scale log2e S - lse log2e), or exp2((sm_scale S + bias - lse) log2e) for a bias object (`_lse2` is then the natural lse);
                    # a row no key reaches carries lse = +inf, which makes p exactly 0.
                    if const_expr(BIAS):
                        _e = fastmath_e.mul(
                            fastmath_e.sub(
                                fastmath_e.add(
                                    fastmath_e.mul(fx.Float32(_s_vals[_si]), sm_scale),
                                    _bias_nat[_si],
                                ),
                                _lse2,
                            ),
                            fx.Float32(_LOG2E),
                        )
                    else:
                        _e = fastmath.sub(
                            fastmath.mul(fx.Float32(_s_vals[_si]), sm_log2e), _lse2
                        )
                    _p = fx.Float32(rocdl.exp2(f32_ty, fx.as_ir_value(_e)))
                    _dead = _kv_dead | (_q_ok == fx.Boolean(False))
                    if const_expr(CAUSAL):
                        _rel = _kv_wl - _q_row_i32
                        _dead = _dead | (_rel < fx.Int32(0)) | (_rel >= _span_w)
                    _p = _dead.select(fx.Float32(0.0), _p)
                    _p_vals.append(_p)
                    if const_expr(DROPOUT):
                        _poff = PHILOX.grid_offset(
                            _ph_base, _ph_stride, _q_row_i32, kv_row_abs_i32
                        )
                        _rv = PHILOX.u32(_ph_seed, _poff)
                        _pick = fx.Int32(_rv[0])
                        for _j in range_constexpr(PHILOX_RN - 1):
                            _pick = (_ph_slot == fx.Int32(_j + 1)).select(
                                fx.Int32(_rv[_j + 1]), _pick
                            )
                        _keep = keep_mask([_pick], idropout_p)[0]
                        # dV takes the dropped, rescaled P; dS keeps the undropped P and a dropped, rescaled dP.
                        _pdrop_vals.append(
                            _keep.select(
                                fastmath.mul(_p, dropout_scale), fx.Float32(0.0)
                            )
                        )
                        _dpv = _keep.select(
                            fastmath.mul(fx.Float32(_dp_vals[_si]), dropout_scale),
                            fx.Float32(0.0),
                        )
                    else:
                        _pdrop_vals.append(_p)
                        _dpv = fx.Float32(_dp_vals[_si])
                    _dpd_vals.append(_dpv)
                    _ds = fastmath.mul(_p, fastmath.sub(_dpv, _di))
                    if const_expr(CAUSAL):
                        _ds = _dead.select(fx.Float32(0.0), _ds)
                    _ds_vals.append(_ds)
                    if const_expr(CAUSAL):
                        _s_dump.append(
                            _dead.select(
                                fx.Float32(float("-inf")),
                                fastmath.mul(fx.Float32(_s_vals[_si]), sm_scale),
                            )
                        )
                    if const_expr(BIAS):
                        _s_dump.append(
                            _dead.select(
                                fx.Float32(float("-inf")),
                                fastmath_e.add(
                                    fastmath_e.mul(fx.Float32(_s_vals[_si]), sm_scale),
                                    _bias_nat[_si],
                                ),
                            )
                        )

                if const_expr(DUMP):
                    # S-type tiles of this iteration's sub-tile: written by the slot-0 wave of each team. S = scale * S_acc.
                    if _dump_s_writer:
                        _dtile = (
                            _dump_base + _dump_s_tile + fx.Int32(qsub * _DUMP_TILE_F32)
                        )
                        if const_expr(CAUSAL or BIAS):
                            _dump_store8(_dtile + fx.Int32(DUMP_OFF_S), _s_dump)
                        else:
                            _dump_store8(
                                _dtile + fx.Int32(DUMP_OFF_S),
                                [
                                    fastmath.mul(fx.Float32(_s_vals[_i]), sm_scale)
                                    for _i in range_constexpr(8)
                                ],
                            )
                        _dump_store8(_dtile + fx.Int32(DUMP_OFF_P), _p_vals)
                        _dump_store8(_dtile + fx.Int32(DUMP_OFF_DP), _dpd_vals)
                        _dump_store8(_dtile + fx.Int32(DUMP_OFF_DS), _ds_vals)

                _q0 = fx.Index(_qs)

                if const_expr(SPLIT_HEAD_DIM):
                    # The wave's own role: dS (dK) or P (dV) as the B operand, Q^T or dO^T as the A operand, its shard's columns.
                    _b_pack = _pack(
                        lay.acc_to_operand(
                            [
                                _role0.select(_ds_vals[_r], _pdrop_vals[_r])
                                for _r in range_constexpr(8)
                            ],
                            _exchange,
                            klane_is_zero,
                            WMMA_V8,
                        )
                    )
                    _rm_b = QRM_BASE + _rm_base_off
                    if const_expr(KEEP_TR_TILES):
                        _tr_b = QTR_BASE + _tr_base_off
                    else:
                        _tr_b = 0
                    for dc in range_constexpr(D_STEPS_SHARD):
                        if const_expr(CONTRACTION_SHARDS > 1):
                            _gd = fx.Index(
                                (fx.Int32(dc) + _shard_k0) * fx.Int32(WMMA_N)
                            )
                        else:
                            _gd = fx.Index(dc * WMMA_N)
                        _a = _tr_operand(
                            _rm_b, QRM_STRIDE, _tr_b, QTR_STRIDE, _gd + lane16, _q0
                        )
                        accs[dc] = wmma_acc(_a, _b_pack, accs[dc])
                else:
                    _p_pack = _pack(
                        lay.acc_to_operand(
                            _pdrop_vals, _exchange, klane_is_zero, WMMA_V8
                        )
                    )
                    _ds_pack = _pack(
                        lay.acc_to_operand(_ds_vals, _exchange, klane_is_zero, WMMA_V8)
                    )
                    # ==== dV^T[d][kv] += dO^T . P ====
                    for dc in range_constexpr(DV_STEPS):
                        _a = _tr_operand(
                            DORM_BASE,
                            DORM_STRIDE,
                            DOTR_BASE,
                            DOTR_STRIDE,
                            fx.Index(dc * WMMA_N) + lane16,
                            _q0,
                        )
                        accs[D_STEPS + dc] = wmma_acc(_a, _p_pack, accs[D_STEPS + dc])
                    # ==== dK^T[d][kv] += Q^T . dS ====
                    for dc in range_constexpr(D_STEPS):
                        _a = _tr_operand(
                            QRM_BASE,
                            QRM_STRIDE,
                            QTR_BASE,
                            QTR_STRIDE,
                            fx.Index(dc * WMMA_N) + lane16,
                            _q0,
                        )
                        accs[dc] = wmma_acc(_a, _ds_pack, accs[dc])

            if const_expr(DUMP):
                # Accumulators after this iteration's two accumulating GEMMs: the partial sums over positions 0 ... `_it`.
                _dump_acc_tile = fx.Int32(group_of_wave) * fx.Int32(
                    N_DK * _DUMP_TILE_F32
                ) + fx.Int32(lane) * fx.Int32(8)
                if const_expr(SPLIT_HEAD_DIM):
                    _acc_off = _role0.select(
                        fx.Int32(DUMP_OFF_DKACC), fx.Int32(DUMP_OFF_DVACC)
                    )
                if _dump_on:
                    if const_expr(SPLIT_HEAD_DIM):
                        for dc in range_constexpr(D_STEPS_SHARD):
                            if const_expr(CONTRACTION_SHARDS > 1):
                                _dd = _shard_k0 + fx.Int32(dc)
                            else:
                                _dd = fx.Int32(dc)
                            _dump_store8(
                                _dump_base
                                + _acc_off
                                + _dump_acc_tile
                                + _dd * fx.Int32(_DUMP_TILE_F32),
                                [Vec(accs[dc])[_i] for _i in range_constexpr(8)],
                            )
                    else:
                        for dc in range_constexpr(DV_STEPS):
                            _dump_store8(
                                _dump_base
                                + fx.Int32(DUMP_OFF_DVACC)
                                + _dump_acc_tile
                                + fx.Int32(dc * _DUMP_TILE_F32),
                                [
                                    Vec(accs[D_STEPS + dc])[_i]
                                    for _i in range_constexpr(8)
                                ],
                            )
                        for dc in range_constexpr(D_STEPS):
                            _dump_store8(
                                _dump_base
                                + fx.Int32(DUMP_OFF_DKACC)
                                + _dump_acc_tile
                                + fx.Int32(dc * _DUMP_TILE_F32),
                                [Vec(accs[dc])[_i] for _i in range_constexpr(8)],
                            )

            # Every wave must finish reading this Q/dO window before the next iteration overwrites it.
            gpu.barrier()
            loop_results = yield accs

        if const_expr(N_ACC == 1):
            # A loop with one carried value returns the value itself, not a list (the split's C = hd / 16 shard holds one accumulator).
            loop_results = [loop_results]
        # ---- Epilogue ----
        # dK^T / dV^T have rows = head dim: after one xor-16 trade a lane holds 8 contiguous head-dim values of its kv row, one v8 store.
        _row_live = kv_row_abs_i32 < seqlen_k_i32
        if const_expr(SPLIT_HEAD_DIM):
            out_base = out_tbase(_k_start_addr)
            _scale = _role0.select(fx.Float32(sm_scale), fx.Float32(1.0))

            def write_out(row, col, val):
                store_global_v8f16(out_ptr, out_base, out_toff(row, col), val)

            def write_out_elem(row, col, val):
                store_global_elem(out_ptr, out_base, out_toff(row, col), val)

        else:
            dk_tbase, dk_toff, _ = fmha.make_addr_pair(
                dk_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k
            )
            dv_tbase, dv_toff, _ = fmha.make_addr_pair(
                dv_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k
            )
            dk_base = dk_tbase(_k_start_addr)
            dv_base = dv_tbase(_k_start_addr)

            def write_dk(row, col, val):
                store_global_v8f16(dk_ptr, dk_base, dk_toff(row, col), val)

            def write_dv(row, col, val):
                store_global_v8f16(dv_ptr, dv_base, dv_toff(row, col), val)

            def write_dk_elem(row, col, val):
                store_global_elem(dk_ptr, dk_base, dk_toff(row, col), val)

            def write_dv_elem(row, col, val):
                store_global_elem(dv_ptr, dv_base, dv_toff(row, col), val)

        if _row_live:
            if const_expr(SPLIT_HEAD_DIM):
                for dc in range_constexpr(D_STEPS_SHARD):
                    _rows = lay.acc_to_rows8(
                        [
                            fx.Float32(Vec(loop_results[dc])[_i])
                            for _i in range_constexpr(8)
                        ],
                        _exchange,
                        klane_is_zero,
                        WMMA_V8,
                    )
                    _t = Vec.from_elements(
                        [
                            fx.Float32(fastmath.mul(v, _scale)).to(elem_dtype)
                            for v in _rows
                        ],
                        elem_dtype,
                    ).ir_value()
                    if const_expr(CONTRACTION_SHARDS > 1):
                        _gc = fx.Index(
                            (fx.Int32(dc) + _shard_k0) * fx.Int32(WMMA_N)
                        ) + lay.rows8_origin(klane)
                    else:
                        _gc = fx.Index(dc * WMMA_N) + lay.rows8_origin(klane)
                    store_chunk(
                        my_cols, write_out, write_out_elem, kv_row_in_tile, _gc, _t
                    )
            else:
                for dc in range_constexpr(D_STEPS):
                    _rows = lay.acc_to_rows8(
                        [
                            fx.Float32(Vec(loop_results[dc])[_i])
                            for _i in range_constexpr(8)
                        ],
                        _exchange,
                        klane_is_zero,
                        WMMA_V8,
                    )
                    _t = Vec.from_elements(
                        [
                            fx.Float32(fastmath.mul(v, sm_scale)).to(elem_dtype)
                            for v in _rows
                        ],
                        elem_dtype,
                    ).ir_value()
                    store_chunk(
                        qk_cols,
                        write_dk,
                        write_dk_elem,
                        kv_row_in_tile,
                        fx.Index(dc * WMMA_N) + lay.rows8_origin(klane),
                        _t,
                    )
                for dc in range_constexpr(DV_STEPS):
                    _rows = lay.acc_to_rows8(
                        [
                            fx.Float32(Vec(loop_results[D_STEPS + dc])[_i])
                            for _i in range_constexpr(8)
                        ],
                        _exchange,
                        klane_is_zero,
                        WMMA_V8,
                    )
                    _t = Vec.from_elements(
                        [fx.Float32(v).to(elem_dtype) for v in _rows], elem_dtype
                    ).ir_value()
                    store_chunk(
                        vo_cols,
                        write_dv,
                        write_dv_elem,
                        kv_row_in_tile,
                        fx.Index(dc * WMMA_N) + lay.rows8_origin(klane),
                        _t,
                    )

    # The kernel body is a plain function rewritten by hand, so the production kernel and the dump kernel (one more argument) share
    # it; the two kernels below only spell their parameter lists.
    _body = ASTRewriter.transform(_body)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
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
        _body(
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
            None,
        )

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def bwd_dkdv_dump_kernel(
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
        dump: fx.Pointer,
    ):
        _body(
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
            dump,
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
        flat_wg_attr = ir.StringAttr.get(f"{BLOCK_SIZE},{BLOCK_SIZE}")
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["rocdl.FLAT_WORK_GROUP_SIZE"] = flat_wg_attr
        passthrough_entries = []
        if const_expr(SCHED_STRATEGY):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("amdgpu-sched-strategy"),
                        ir.StringAttr.get(SCHED_STRATEGY),
                    ]
                )
            )
        passthrough_entries.append(
            ir.ArrayAttr.get(
                [
                    ir.StringAttr.get("denormal-fp-math-f32"),
                    ir.StringAttr.get("preserve-sign,preserve-sign"),
                ]
            )
        )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)
        launcher.launch(
            grid=(num_kv_tiles, num_head_k_idx, nseq_idx),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_fmha_bwd_dkdv_gfx950(
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
        num_kv_tiles = (fx.Index(max_seqlen_k) + (BLOCK_N - 1)) // BLOCK_N
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

    @flyc.jit
    def launch_fmha_bwd_dkdv_gfx11(
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
        dump: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        nseq_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        num_kv_tiles = (fx.Index(max_seqlen_k) + (BLOCK_N - 1)) // BLOCK_N
        _run(
            bwd_dkdv_dump_kernel(
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
                dump,
            ),
            stream,
            num_kv_tiles,
            fx.Index(num_head_k),
            nseq_idx,
        )

    _jit = launch_fmha_bwd_dkdv_gfx11 if DUMP else launch_fmha_bwd_dkdv_gfx950

    def _launch(*args, **kwargs):
        return flyc.compile(_jit, *args, **kwargs)

    return _launch
