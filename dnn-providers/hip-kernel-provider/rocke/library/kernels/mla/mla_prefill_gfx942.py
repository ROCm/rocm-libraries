# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 (CDNA3) MLA prefill kernels.

Two kernels live here:

  * :func:`build_mla_prefill_fwd` -- the product kernel. Full online-softmax
    forward with bottom-right causal masking, paged latent KV, and packed
    varlen block numbering.
  * :func:`build_mla_prefill_score_probe` -- the milestone-2 bring-up kernel
    that emits one raw ``16x16`` QK score tile with no softmax, no PV and no
    mask, graded against
    :func:`builders.mla.ref_mla_attn.ref_mla_prefill_scores`. It still expands
    ``K_nope`` explicitly, which is precisely what makes it a useful
    independent check on the absorbed path below.

**W_UK absorption.** The latent cache stores ``c_kv[token, r_kv]`` shared by
all heads; a head's keys and values are the up-projections
``K_nope[k] = C[k] @ W_UK[head, :, :d_nope]`` and
``V[k] = C[k] @ W_UK[head, :, d_nope:]``. Materialising those per key tile
costs ``r_kv x (d_nope + d_v)`` MACs per key -- an order of magnitude more
math than the attention itself. Both halves fold out of the loop instead::

    S[q,k] = Q_nope[q].K_nope[k] + Q_rope[q].K_rope[k]
           = (Q_nope[q] @ W_UK[head,:,:d_nope]^T) . C[k] + Q_rope[q].K_rope[k]
    O[q]   = sum_k P[q,k] * V[k]
           = (sum_k P[q,k] * C[k]) @ W_UK[head,:,d_nope:]

so the head projection is applied **once per workgroup** -- to ``Q`` in the
prologue and to the latent output accumulator in the epilogue -- and the key
loop runs entirely in the 512-wide latent space. Both rewrites are exact: the
online-softmax rescale and the ``1/l`` normalisation are linear, so they
commute with the ``W_UV`` projection.

Data flow for one workgroup, which owns one ``(q-block, head)`` pair::

    q_nope  <- q[token, head, :d_nope]        registers    absorb A operand
    for window in range(Bq / 16):                            -- Q absorb
        qa_lds[:, rope cols] <- q[token, head, d_nope:]    Q_rope, if here
        for r_slice in window:
            wq_lds  <- W_UK[head, r_slice, :d_nope]       [r_tile, d_nope]
            qa_lds[:, r_slice] <- q_nope @ wq_lds^T
        qa      <- qa_lds                     registers    the window's waves
    for k_tile in range(n_k_tiles):                          -- latent loop
        kv_lds[:, :r_kv] <- c_kv[page]        [Bk, r_KV]   paged latent
        kv_lds[:, r_kv:] <- k_rope[page]      [Bk, 64]     head-shared RoPE
        S       <- scale * (qa @ kv_lds^T)                 one K=576 GEMM
        P       <- online_softmax(mask(S))
        acc     += P @ kv_lds[:, :r_kv]                    latent-space PV
    for wave_part in range(num_warps):                       -- V absorb
        accl_lds <- acc[:, wave_part]         [Bq, r_KV / num_warps]
        for r_slice in wave_part:
            wt_lds <- transpose(W_UK[head, r_slice, d_nope:])  [d_v, r_tile]
            out    += accl_lds[:, r_slice] @ wt_lds
    out      <- out * (1/l)                   [Bq, d_v]

gfx942 constraints that shape the emission:

  * **Narrow MFMA only.** CDNA3 has no wide-K ``mfma_f32_16x16x32`` bf16 atom,
    so every GEMM uses ``mfma_f32_16x16x16_bf16`` with a K-step of 16.
  * **No transpose read.** ``ds_read_*_tr_*`` is gfx950-only, so a B operand
    that is not already stored as ``[n][k]`` must be transposed by hand. The
    PV GEMM wants ``C`` as ``[r][key]`` while ``kv_lds`` holds it as
    ``[key][r]`` for the score GEMM; its B operand is gathered from
    ``kv_lds`` one bf16 per key (a read-path transpose), which beat keeping a
    second, store-transposed copy. The ``W_UK`` V-half is transposed on the
    store path in the epilogue. The Q-absorb B operand needs **no** transpose:
    ``W_UK[head, r, :d_nope]`` is already stored as ``[r][nope]``, which is
    exactly ``[n][k]`` for that GEMM.
  * **LDS round-trips between GEMMs are mandatory.** The MFMA C decode lands
    a result at ``[m = c_row(lane, reg)][n = n_tile*16 + lane%16]``, the
    transpose of what an A operand read wants, so a register-level handoff
    from one GEMM to the next is impossible. Hence ``qa_lds`` and ``accl_lds``.

**Query input contract.** Both kernels take the query **already projected and
RoPE-rotated** (``[total_q, H_q, d_nope + d_rope]``): ``q[..., :d_nope]`` is
``Q_nope`` after ``W_UQ``, and ``q[..., d_nope:]`` is ``Q_rope`` after the
query-side RoPE. Neither the ``W_UQ`` projection nor the query RoPE is in-tree;
the caller must apply both. A pre-kernel that does so is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import BF16, F32, KernelDef
from rocke.helpers.attention import safe_inv_l

from kernels.common._read_ahead import read_ahead
from kernels.common.attention_arch import attention_lds_capacity_bytes
from kernels.common.attention_dense_spec import INT32_LIMIT
from kernels.mla._mla_body import (
    C_PER_LANE,
    C_STAGE_W,
    MFMA_K,
    MFMA_M,
    MFMA_N,
    V8_PAD,
    WAVE_SIZE,
    WT_PAD,
    KernelArg,
    acc_iter_args,
    kernel_signature,
    key_mask,
    lane_ids,
    mfma_16x16x16,
    on_wave,
    online_softmax_row,
    open_kernel,
    pack_p,
    pick4,
    rescale_if_any,
    run_k_loops,
    scale_frag,
    shift,
    softmax_consts,
    softmax_iter_args,
    split_softmax_state,
    tail_args,
    wave_tile_row,
    zero_tiles,
)
from kernels.mla._mla_prefill_gfx942_body import (
    AgprAcc,
    KvRegStager,
    KvSharedStager,
    KvTileDma,
    PageTable,
    ScorePartials,
    absorb_query,
    absorb_query_t,
    causal_k_tiles,
    causal_keys,
    clamp_tile,
    copy_rows_to_lds,
    epilogue_lane_row,
    expand_latent,
    full_k_tiles,
    gather_keys_t,
    in_window,
    lds_gemm_tile,
    load_q_frags,
    load_q_nope_rope,
    park_q_rope,
    project_wuv_heads,
    query_rows,
    read_k_rope_swizzled,
    read_kv_rows,
    read_pv_group,
    read_q_rope,
    reversed_q_block,
    row_of,
    seq_len_k,
    split_k_scores,
    stage_q_cols,
    store_acc_t,
    store_c_frag,
    store_latent_out,
    store_lse,
    store_out_frag,
    store_probe_scores,
    take_window_frags,
    tile_ahead,
    tile_pairs,
    varlen_tile,
    w_head_base,
    wave_head,
    wave_k_col,
    wave_q_start,
    window_rank,
)

__all__ = [  # noqa: RUF022  -- ordered spec -> gate -> launch geometry -> build
    "MlaPrefillSpec",
    "supports_mla_prefill",
    "require_mla_prefill",
    "check_mla_prefill_extents",
    "MLA_BINARY_SEARCH_ITERS",
    "MLA_MAX_NUM_SEQS",
    "mla_prefill_block",
    "mla_prefill_score_probe_grid",
    "mla_prefill_score_probe_signature",
    "build_mla_prefill_score_probe",
    "mla_prefill_fwd_grid",
    "mla_prefill_fwd_signature",
    "build_mla_prefill_fwd",
]

# k-steps per scheduler-ordered block in the score GEMM.
#
# That GEMM is LDS-read-bound, not MFMA-bound: at BQ == BK == MFMA_M == MFMA_N
# there is one M tile and one K tile, so each of its ``SCORE_K_ITERS`` steps is
# two ``ds_read``s feeding a single MFMA -- and every MFMA accumulates into the
# *same* register, so the MFMA chain is serial. Covering that chain needs the
# reads to run well ahead of it.
#
# Emitting a (DS_READ x 2G, MFMA x G) pair every G steps asks the post-RA
# scheduler for exactly that: a block of reads hoisted above the MFMAs they
# feed, G steps deep. G must divide the per-wave step count evenly -- a
# remainder leaves the tail steps unhinted and measured worse than any dividing
# value. The score K reduction is split across waves, so that count is
# ``SCORE_K_ITERS / num_warps`` (9 at the default geometry), and G = 9 hoists
# the whole wave's reads in one block. It is a scheduling hint only, so a wrong
# value costs speed and never correctness.
#
# This is mutually exclusive with ``iglp_opt``, which owns the whole-loop
# schedule. Both of its canned patterns were measured here and both regressed.
SCORE_SGB_GROUP = 9

# AGPR budget of the one-wave-per-head layout, as for the one-head kernel: no
# AGPRs. A wave holds its head's whole latent accumulator, 128 VGPRs at
# block_q == 16, and left free the backend parks it in AGPRs and pays a
# v_accvgpr_read / v_accvgpr_write pair per register for the per-tile rescale.
# Without them the kernel fits 256 VGPRs, two waves per SIMD. At block_q == 32
# the accumulator alone is 256 VGPRs and the kernel spills either way.
HPW_AGPR_ALLOC = (0, 0)

# Software pipelining of the one-wave-per-head k-loop. The loop sits at the
# 256-VGPR line that keeps two workgroups per CU, so the backend scheduler
# cannot find registers to run LDS reads ahead on its own: left alone it waits
# on every read just before the MFMA that consumes it. The loop is therefore
# emitted already pipelined, with ``sched_barrier`` fences that stop the
# scheduler from sinking the reads back down to their uses.
#
# Score k-steps whose ``kv_lds`` / ``qr_lds`` reads are issued ahead of the
# current k-step's MFMA.
HPW_SCORE_PREFETCH = 8
# k-steps between fences. Fencing every step keeps each 8 B read separate;
# a coarser fence lets adjacent k-steps merge into one ``ds_read2_b64``.
HPW_SCORE_FENCE_EVERY = 4
# PV read groups (four 8 B reads feeding four MFMAs) issued ahead. The first
# group is issued before the softmax, so its latency hides behind the
# softmax's VALU and cross-lane reduction. Two groups ahead spills.
HPW_PV_PREFETCH = 1
# Query-absorb r-tiles whose W_UK loads are issued ahead of the current
# r-tile's MFMAs, so the prologue does not expose one global round trip per
# r-tile.
HPW_ABSORB_PREFETCH = 4

# Lazy softmax rescale, in log2 units. The running max only moves -- and the
# latent accumulator is only rescaled -- when a row's new max exceeds the max
# in use by more than this. Otherwise the stale max is kept and probabilities
# may reach 2**HPW_LAZY_TAU; the final 1/l normalisation and the LSE use the
# same max, so the result is unchanged. bf16 P and the f32 accumulator and row
# sum have an 8-bit exponent: with 2**17 keys the sum stays below 2**81 and the
# accumulator below 2**81 * max|C|, far from the f32 limit, while a small
# threshold makes the rescale fire whenever the running max creeps up over a
# long key range.
HPW_LAZY_TAU = 64.0

# agpr_acc layout: PV read groups issued ahead (the DMA'd key tile frees the
# registers the staged layout needs for its loads) and key-tile DMA
# instructions issued at each score fence point, spread between MFMAs rather
# than in one burst after the barrier.
AGPR_PV_PREFETCH = 2
AGPR_DMA_PER_FENCE = 4

# Consecutive workgroups of the head-major order handed to one XCD at a time
# when ``MlaPrefillSpec.num_xcds`` is set. Small enough that an XCD's resident
# set still spans several q-block ranges -- which share key tiles across head
# groups in L2 -- and large enough that it holds only a few heads' weights.
XCD_CHUNK = 16

# v columns of W_UV staged per epilogue round in the one-wave-per-head layout.
# Each round holds a 64-row latent block for every head: 32 columns keep that
# at 17408 B, under the loop phase's pool.
EPI_VQ = 32

# Trip count of the in-kernel ``seq_idx`` binary search. It is fixed rather
# than sized per request, so a request's batch size never splits the kernel
# cache. Each iteration halves the ``[left, right)`` interval, which starts as
# ``[0, num_seqs)`` and holds ``num_seqs + 1`` candidate answers; after ``n``
# iterations ``right - left == num_seqs >> n``, so the search converges iff
# ``num_seqs < 2**n``. A larger batch would stop early and map q-blocks to the
# wrong sequence without any error, so dispatch caps ``num_seqs`` at
# MLA_MAX_NUM_SEQS.
MLA_BINARY_SEARCH_ITERS = 12
MLA_MAX_NUM_SEQS = 2**MLA_BINARY_SEARCH_ITERS - 1

# Offsets are i32 ``mul nsw`` / ``add nsw`` chains: signed overflow is UB, so
# every extent the kernel indexes has to stay below INT32_LIMIT (2**31; in
# elements for the global loads and stores, in bytes for the paged buffer
# loads).
# ``num_records`` of the paged c_kv / k_rope buffer resources.
KV_RSRC_BYTES = 0x7FFF0000


@dataclass(frozen=True)
class MlaPrefillSpec:
    """Immutable description of one gfx942 MLA prefill configuration.

    The defaults are the tuned single-head layout (``block_q`` and
    ``r_kv_tile`` were measured; see their comments), which dispatch keeps as
    its fallback. A caller normally sets only ``num_heads`` (128 for a
    DeepSeek-V3 shaped model, 64 for a Kimi-K2 shaped one); dispatch sets the
    one-wave-per-head layout fields (``heads_per_wg``, ``q_tiles_per_wg``,
    ``num_warps``, ``block_q``, ``agpr_acc``, ``num_xcds``) per request.
    """

    num_heads: int
    d_nope: int = 128
    d_rope: int = 64
    d_v: int = 128
    r_kv: int = 512
    # Two M tiles: every key tile staged into LDS feeds two query tiles' score
    # and PV GEMMs. The prologue and epilogue stage one M tile at a time, so
    # the LDS pool does not grow with it.
    block_q: int = 32
    block_k: int = 16
    # 32, not the bring-up 64: it is the widest slice that keeps the prologue,
    # the LDS peak, under 32 KB, so two workgroups fit per CU -- which measured
    # faster than 64 on every shape tried.
    r_kv_tile: int = 32
    num_warps: int = 4
    page_block_size: int = 16
    dtype: str = "bf16"
    # Fixed at MLA_BINARY_SEARCH_ITERS, which serves up to MLA_MAX_NUM_SEQS
    # sequences; admission refuses any other value (it is not in the name).
    binary_search_iters: int = MLA_BINARY_SEARCH_ITERS
    # Heads packed into one workgroup, one wave per head. At 1 (the default) a
    # workgroup owns one head and its waves split that head's work. Above 1 a
    # workgroup owns ``heads_per_wg`` consecutive heads: every key tile is staged
    # into LDS once and read by all of them, and each wave runs its own head's
    # score, softmax and latent PV end to end -- no split-K partials and no
    # cross-wave combine. The query absorb and the W_UV projection then stay
    # inside the wave as well. Requires
    # ``num_warps == heads_per_wg * q_tiles_per_wg``.
    heads_per_wg: int = 1
    # Query row tiles per workgroup in the one-wave-per-head layout: the
    # workgroup holds heads_per_wg heads x q_tiles_per_wg tiles of 16 queries,
    # one wave per (head, tile), so block_q == 16 * q_tiles_per_wg and
    # num_warps == heads_per_wg * q_tiles_per_wg. Waves sharing a head share its
    # W_UK / W_UV fetches, which cuts the per-row weight traffic.
    q_tiles_per_wg: int = 1
    # XCDs the device deals workgroups across (6 on a 228-CU gfx942, 8 on a
    # 304-CU one); 0
    # keeps launch order. One-wave-per-head layout only. In launch order each
    # XCD's resident workgroups span every head group, so its L2 holds all of
    # their W_UK / W_UV; with the XCD count the kernel reorders workgroups so
    # an XCD walks a few heads at a time. A wrong count costs speed only.
    num_xcds: int = 0
    # Latent accumulator in AGPRs (one-wave-per-head layout). It lets a wave
    # own 32 query rows -- two M tiles, a 256-register accumulator -- at one
    # workgroup per CU; see ``_mla_prefill_gfx942_body.AgprAcc``. Needs the
    # llvm22 or llvm23 flavor.
    agpr_acc: bool = False
    # Unfused head projections. False (the default) is the fused kernel: it
    # takes the projected query and ``W_UK`` and applies both head projections
    # itself. True takes the absorbed query ``Qa = [Q_nope @ W_UK_nopeᵀ |
    # Q_rope]`` (``[total_q, H, r_kv + d_rope]``) and writes the normalized
    # latent output (``[total_q, H, r_kv]``) plus the LSE; the caller runs both
    # projections as batched GEMMs. One-wave-per-head layout only
    # (``heads_per_wg > 1`` or ``q_tiles_per_wg > 1``). See
    # :func:`mla_prefill_fwd_signature`.
    absorbed_query: bool = False

    def __post_init__(self) -> None:
        """Reject specs that are impossible on *any* arch.

        Arch- and atom-specific admission (gfx942, the 16x16x16 MFMA shape,
        staging divisibility across the workgroup) stays in
        :func:`supports_mla_prefill`, which reports rather than raises so a
        dispatcher can fall through to another family. What is checked here
        cannot be satisfied by picking a different arch, so it is an error at
        construction instead: a caller that skips :func:`require_mla_prefill`
        would otherwise see it as a ``ZeroDivisionError`` deep inside the
        builder.
        """
        if self.dtype != "bf16":
            raise ValueError(
                f"MLA prefill supports bf16 only, got dtype={self.dtype!r}"
            )
        # Exact types, never coerced: "false" is truthy and True == 1, so a
        # loose value would silently build a different kernel.
        for name in ("agpr_acc", "absorbed_query"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(
                    f"{name} must be a bool, got {type(value).__name__} {value!r}"
                )
        for name in (
            "num_heads",
            "d_nope",
            "d_rope",
            "d_v",
            "r_kv",
            "block_q",
            "block_k",
            "r_kv_tile",
            "num_warps",
            "page_block_size",
            "binary_search_iters",
            "heads_per_wg",
            "q_tiles_per_wg",
            "num_xcds",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(
                    f"{name} must be an int, got {type(value).__name__} {value!r}"
                )
        positive = (
            "num_heads",
            "d_nope",
            "d_rope",
            "d_v",
            "r_kv",
            "r_kv_tile",
            "block_q",
            "block_k",
            "page_block_size",
            "binary_search_iters",
        )
        for name in positive:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError(f"num_warps must be one of 1/2/4/8, got {self.num_warps}")
        if self.heads_per_wg not in (1, 2, 4, 8):
            raise ValueError(
                f"heads_per_wg must be one of 1/2/4/8, got {self.heads_per_wg}"
            )
        if self.num_xcds < 0:
            raise ValueError(f"num_xcds must be >= 0, got {self.num_xcds}")
        if self.q_tiles_per_wg not in (1, 2, 4, 8):
            raise ValueError(
                f"q_tiles_per_wg must be one of 1/2/4/8, got {self.q_tiles_per_wg}"
            )
        if self.num_heads % self.heads_per_wg != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be a multiple of "
                f"heads_per_wg={self.heads_per_wg}"
            )
        if self.r_kv % self.r_kv_tile != 0:
            raise ValueError(
                f"r_kv={self.r_kv} must be a multiple of r_kv_tile={self.r_kv_tile}"
            )
        if self.page_block_size != self.block_k:
            # One k-tile must land inside exactly one page, otherwise the tile
            # straddles two block_table entries and the single-page gather in
            # the builder is wrong.
            raise ValueError(
                "page_block_size must equal block_k so a k-tile maps to one page, "
                f"got page_block_size={self.page_block_size}, block_k={self.block_k}"
            )

    @property
    def head_dim_qk(self) -> int:
        return self.d_nope + self.d_rope

    @property
    def w_uk_cols(self) -> int:
        """Column count of one ``W_UK`` row: ``K_nope`` half then ``V`` half."""
        return self.d_nope + self.d_v

    @property
    def threads(self) -> int:
        return WAVE_SIZE * self.num_warps

    def kernel_name(self) -> str:
        return self._name("mla_prefill_score_probe_gfx942")

    def fwd_kernel_name(self) -> str:
        return self._name("mla_prefill_fwd_gfx942")

    def _name(self, stem: str) -> str:
        return (
            f"{stem}"
            f"_h{self.num_heads}"
            f"_n{self.d_nope}r{self.d_rope}v{self.d_v}"
            f"_rkv{self.r_kv}"
            f"_q{self.block_q}k{self.block_k}"
            f"_rt{self.r_kv_tile}"
            f"_w{self.num_warps}"
            + (f"_hpw{self.heads_per_wg}" if self.heads_per_wg != 1 else "")
            + (f"_qpw{self.q_tiles_per_wg}" if self.q_tiles_per_wg != 1 else "")
            + (f"_x{self.num_xcds}" if self.num_xcds else "")
            + ("_aa" if self.agpr_acc else "")
            + f"_{self.dtype}"
            + ("_aq" if self.absorbed_query else "")
        )


def _fwd_lds_bytes(spec: MlaPrefillSpec) -> int:
    """Predict the LDS pool :func:`build_mla_prefill_fwd` will ask for.

    The lowerer pools the six buffers by liveness, so the pool is *not* their
    arithmetic sum -- see the LDS comment block in the builder for the achieved
    layout. The three phases never share a live buffer: the prologue pair
    (``wq_lds``, ``qa_lds``) is dead before the k-loop starts (the score A
    operand is read into registers first), and by the epilogue everything
    earlier is dead. Each phase therefore packs from the base on its own and
    the pool is the widest phase.

    ``qa_lds`` holds one wave-aligned window of ``1 / M_TILES`` of the absorbed
    columns for every M tile, and ``accl_lds`` one wave's ``1 / num_warps`` of
    the latent columns for every M tile.

    The packer reuses a dead slot only from its start offset. Each later phase
    allocates its small buffer first (``s_part``, ``accl_lds``) so it takes the
    ``wq_lds`` slot at the base, and the large one lands on the ``qa_lds``
    slot. When the small buffer outgrows ``wq_lds`` the large one is pushed
    past the prologue. Erring high is the point: this gates admission, so it
    must never under-predict. It is a model of observed behaviour rather than a
    guarantee, so check the emitted pool whenever a new geometry is introduced.
    """
    elem = 2  # bf16
    qa_cols = spec.r_kv + spec.d_rope
    if _one_wave_per_head(spec):
        # One wave per head: the loop holds the shared kv_lds plus every
        # wave's parked Q_rope, and the epilogue one 64-row latent block of
        # W_UV for every head, which pools onto them.
        kv_lds = elem * spec.block_k * (qa_cols + WT_PAD)
        if spec.agpr_acc:
            # The DMA'd key tile is double-buffered, with the RoPE columns in
            # their own unpadded buffer: two latent slots plus two RoPE slots.
            kv_lds = 2 * elem * spec.block_k * (spec.r_kv + WT_PAD + spec.d_rope)
        rows = spec.block_q // spec.q_tiles_per_wg
        qr_lds = elem * spec.num_warps * rows * (spec.d_rope + WT_PAD)
        # With absorbed_query there is no epilogue buffer: the latent goes
        # straight from registers to global.
        wt_all = (
            0
            if spec.absorbed_query
            else elem * spec.heads_per_wg * EPI_VQ * spec.q_tiles_per_wg * (64 + WT_PAD)
        )
        return -(-max(kv_lds + qr_lds, wt_all) // 16) * 16
    m_tiles = max(1, spec.block_q // MFMA_M)
    wq_lds = elem * spec.r_kv_tile * (spec.d_nope + V8_PAD)
    qa_lds = elem * spec.block_q * (qa_cols // m_tiles + WT_PAD)
    kv_lds = elem * spec.block_k * (qa_cols + WT_PAD)
    s_part = 4 * spec.num_warps * spec.block_q * spec.block_k
    accl_lds = elem * spec.block_q * (spec.r_kv // spec.num_warps + WT_PAD)
    wt_lds = elem * spec.d_v * (spec.r_kv_tile + WT_PAD)

    prologue = wq_lds + qa_lds

    def _phase(small: int, large: int) -> int:
        return (wq_lds if small <= wq_lds else max(small, prologue)) + large

    loop = _phase(s_part, kv_lds)
    epilogue = _phase(accl_lds, wt_lds)
    # The lowerer rounds the pool up to 16 B.
    return -(-max(prologue, loop, epilogue) // 16) * 16


def supports_mla_prefill(spec: MlaPrefillSpec, *, arch: str) -> Tuple[bool, str]:
    """Admission check for the gfx942 MLA prefill family.

    Returns ``(ok, reason)``; ``reason`` is empty when ``ok``. Every rejection
    names the offending field and the value that would be acceptable, so a
    caller never has to read this function to understand the refusal.

    Only arch- and atom-specific admission lives here, so a dispatcher can fall
    through to another family on a ``False``. Specs that are impossible on any
    arch are rejected earlier, by :meth:`MlaPrefillSpec.__post_init__`, which
    means this function may assume positivity and the bf16/page/warp-count
    invariants already hold.
    """
    if not arch.startswith("gfx942"):
        return False, f"MLA prefill is gfx942-only for now, got arch={arch!r}"
    # Dispatch admits up to MLA_MAX_NUM_SEQS sequences without knowing which
    # spec will serve them, so every admitted spec must search that far. Exactly
    # that far: the iteration count is not part of the kernel name, so a second
    # value would compile under the default spec's name and collide in any
    # name-keyed cache.
    if spec.binary_search_iters != MLA_BINARY_SEARCH_ITERS:
        return False, (
            f"binary_search_iters={spec.binary_search_iters}, but exactly "
            f"{MLA_BINARY_SEARCH_ITERS} is supported: fewer cannot resolve "
            f"num_seqs up to {MLA_MAX_NUM_SEQS}, and the count is not in the "
            "kernel name, so any other value would collide with the default's"
        )
    # Register blocking: the kernel body is written over M_TILES = block_q /
    # MFMA_M and K_TILES = block_k / MFMA_N, so any multiple of the atom is
    # expressible. The cost of a larger block_q is the latent accumulator,
    # which holds (block_q / MFMA_M) * (r_kv / MFMA_N) / num_warps VGPRs per
    # lane -- 32 at block_q=16, 64 at block_q=32 with the default geometry.
    if spec.block_q % MFMA_M != 0:
        return False, f"block_q must be a multiple of {MFMA_M}, got {spec.block_q}"
    if spec.block_k % MFMA_N != 0:
        return False, f"block_k must be a multiple of {MFMA_N}, got {spec.block_k}"
    for name in ("d_nope", "d_rope", "d_v", "r_kv"):
        value = getattr(spec, name)
        if value % MFMA_N != 0:
            return False, f"{name} must be a multiple of {MFMA_N}, got {value}"
    if spec.r_kv_tile % MFMA_K != 0:
        return False, f"r_kv_tile must be a multiple of {MFMA_K}, got {spec.r_kv_tile}"
    # MFMA_K == MFMA_N == 16 here, so the checks above already cover every
    # K-dimension requirement; there is nothing extra to demand for the K axis.
    if _one_wave_per_head(spec):
        return _supports_heads_per_wg(spec, arch=arch)
    if spec.absorbed_query:
        return False, (
            "absorbed_query is implemented for the one-wave-per-head layout "
            f"only, got heads_per_wg={spec.heads_per_wg}"
        )
    if spec.num_xcds or spec.agpr_acc:
        return False, (
            "num_xcds and agpr_acc are implemented for the one-wave-per-head "
            "layout only, "
            f"got heads_per_wg={spec.heads_per_wg}"
        )
    #
    # Each of the three GEMMs whose n-axis is split across waves needs that
    # split to be exact -- a wave owns [w * per_wave, (w+1) * per_wave) with no
    # remainder lane.
    # The query absorb is exempt when its slice is narrower than the
    # workgroup: it then runs one n-tile on each of the first r_kv_tile / 16
    # waves and gates the rest off.
    warp_splits = {
        "r_kv (PV latent n-tiles)": spec.r_kv,
        "d_v (epilogue n-tiles)": spec.d_v,
    }
    absorb_n_tiles = spec.r_kv_tile // MFMA_N
    if absorb_n_tiles >= spec.num_warps:
        warp_splits["r_kv_tile (query-absorb n-tiles)"] = spec.r_kv_tile
    for name, value in warp_splits.items():
        n_tiles = value // MFMA_N
        if n_tiles % spec.num_warps != 0:
            return False, (
                f"{name}: {value}/{MFMA_N} = {n_tiles} n-tiles must divide "
                f"evenly across num_warps={spec.num_warps}"
            )
    # The score GEMM splits its K reduction, not an n-axis: each wave owns an
    # equal run of the (r_kv + d_rope) / MFMA_K k-steps.
    score_k_steps = (spec.r_kv + spec.d_rope) // MFMA_K
    if score_k_steps % spec.num_warps != 0:
        return False, (
            f"score K: {spec.r_kv + spec.d_rope}/{MFMA_K} = {score_k_steps} k-steps "
            f"must divide evenly across num_warps={spec.num_warps}"
        )
    # The prologue stages the absorbed query in M_TILES column windows, each
    # covering the score k-steps of a whole number of waves and a whole number
    # of absorb r-slices; the epilogue stages the latent one wave's PV tiles at
    # a time, so each wave's run of r-slices must be whole.
    m_tiles = spec.block_q // MFMA_M
    if spec.num_warps % m_tiles != 0:
        return False, (
            f"block_q={spec.block_q} stages the absorbed query in {m_tiles} "
            f"wave-aligned windows, which num_warps={spec.num_warps} must divide"
        )
    qa_win = (spec.r_kv + spec.d_rope) // m_tiles
    if m_tiles > 1 and (
        qa_win % spec.r_kv_tile != 0 or qa_win * (m_tiles - 1) > spec.r_kv
    ):
        return False, (
            f"the absorbed-query window of {qa_win} columns must be a multiple of "
            f"r_kv_tile={spec.r_kv_tile} and leave d_rope in the last window"
        )
    if (spec.r_kv // spec.r_kv_tile) % spec.num_warps != 0:
        return False, (
            f"r_kv/r_kv_tile = {spec.r_kv // spec.r_kv_tile} epilogue slices must "
            f"divide evenly across num_warps={spec.num_warps}"
        )
    # Every global->LDS staging loop below hands each thread a fixed number of
    # fixed-width vector chunks with no remainder handling, so the element
    # count must divide by threads * width -- not merely by threads.
    threads = spec.threads
    stages = {
        # Q_nope is read straight into MFMA fragments, so it has no staging.
        "Q_rope tile": (spec.block_q * spec.d_rope, 4),
        "c_kv tile": (spec.block_k * spec.r_kv, C_STAGE_W),
        "k_rope tile": (spec.block_k * spec.d_rope, 4),
        "W_UK absorb slice": (spec.r_kv_tile * spec.d_nope, 8),
        "W_UV epilogue slice": (spec.r_kv_tile * spec.d_v, 8),
    }
    for name, (count, width) in stages.items():
        if count % (threads * width) != 0:
            return False, (
                f"{name} has {count} elements, which does not divide evenly "
                f"across {threads} threads at {width} elements per chunk"
            )
    # The prologue and epilogue buffers hold a column window of every M tile,
    # so block_q reaches LDS mostly through s_part; kv_lds scales with block_k.
    # Both still have to fit the layout's per-CU share of LDS.
    return _fits_lds(spec, arch=arch)


def _fwd_workgroups_per_cu(spec: MlaPrefillSpec) -> int:
    """Workgroups per CU the layout is built to run at.

    The kernel keeps a wave under 256 VGPRs, two waves per SIMD and eight per
    CU, so a four-wave workgroup is meant to run two per CU. An eight-wave
    workgroup already fills the CU's wave slots, and ``agpr_acc`` spends the
    whole 512-register file on one wave per SIMD, so both run one per CU.
    """
    return 1 if spec.agpr_acc or spec.num_warps >= 8 else 2


def _fits_lds(spec: MlaPrefillSpec, *, arch: str) -> tuple[bool, str]:
    """LDS admission against the occupancy the layout intends.

    The capacity is the arch catalog's per-workgroup LDS. Checking only that
    limit would admit a two-per-CU layout whose pool is over half of it, which
    then silently runs one per CU.
    """
    lds = _fwd_lds_bytes(spec)
    wgs = _fwd_workgroups_per_cu(spec)
    budget = attention_lds_capacity_bytes(arch) // wgs
    if lds > budget:
        return False, (
            f"the forward kernel needs about {lds} B of LDS at "
            f"heads_per_wg={spec.heads_per_wg}, q_tiles_per_wg="
            f"{spec.q_tiles_per_wg}, block_q={spec.block_q}, "
            f"block_k={spec.block_k}, r_kv_tile={spec.r_kv_tile}, which exceeds "
            f"the {budget} B that {wgs} workgroup(s) per CU allow"
        )
    return True, ""


def _one_wave_per_head(spec: MlaPrefillSpec) -> bool:
    """True for the one-wave-per-head layout (:func:`_build_mla_prefill_fwd_heads`)."""
    return spec.heads_per_wg > 1 or spec.q_tiles_per_wg > 1 or spec.agpr_acc


def _supports_heads_per_wg(spec: MlaPrefillSpec, *, arch: str) -> tuple[bool, str]:
    """Admission for the one-wave-per-head layout (:func:`_one_wave_per_head`).

    No GEMM axis is split across waves -- each wave owns one head end to end --
    so the only cross-wave work is the cooperative key-tile staging and the
    cooperative W_UV staging in the epilogue.
    """
    qpw = spec.q_tiles_per_wg
    if spec.num_warps != spec.heads_per_wg * qpw:
        return False, (
            f"heads_per_wg={spec.heads_per_wg} x q_tiles_per_wg={qpw} runs one "
            f"wave per (head, query tile), so num_warps must be their product, "
            f"got num_warps={spec.num_warps}"
        )
    # A wave holds its head's whole latent accumulator: 128 VGPRs at 16 rows.
    # At 32 rows it is 256 registers, which only fits held in AGPRs.
    rows = 32 if spec.agpr_acc else MFMA_M
    if spec.block_q != rows * qpw:
        return False, (
            f"the one-wave-per-head layout needs block_q={rows * qpw} "
            f"({rows} rows per wave x q_tiles_per_wg={qpw}), got "
            f"block_q={spec.block_q}"
        )
    if spec.agpr_acc:
        from rocke.core.lower_llvm import _is_modern_flavor, _resolve_llvm_flavor

        flavor = _resolve_llvm_flavor()
        if not _is_modern_flavor(flavor):
            return False, (
                "agpr_acc needs the llvm22 or llvm23 flavor (llvm20 does not "
                f"keep the backend out of the AGPR file), got {flavor}"
            )
        # The key-tile DMA moves 256 B per wave instruction: a latent row must
        # be a whole number of them, and a RoPE row exactly half of one (two
        # rows per instruction, swizzled over 16 8 B units).
        if (spec.r_kv * 2) % 256 != 0:
            return False, (
                f"agpr_acc DMAs latent rows in 256 B chunks: r_kv={spec.r_kv}, "
                "expected a multiple of 128"
            )
        if spec.d_rope * 2 != 128:
            return False, (
                f"agpr_acc DMAs two 128 B RoPE rows per chunk: d_rope={spec.d_rope}, "
                "expected 64"
            )
    # The PV gathers four latent tiles per read and the epilogue stages W_UV
    # in 64-row latent blocks and EPI_VQ-wide v windows.
    if spec.r_kv % 64 != 0:
        return False, f"heads_per_wg > 1 needs r_kv a multiple of 64, got {spec.r_kv}"
    if spec.d_v % (EPI_VQ * qpw) != 0:
        return False, (
            f"the one-wave-per-head layout needs d_v a multiple of "
            f"{EPI_VQ * qpw}, got {spec.d_v}"
        )
    threads = spec.threads
    kr_w = 4 if (spec.block_k * spec.d_rope) % (threads * 4) == 0 else 2
    stages = {
        "c_kv tile": (spec.block_k * spec.r_kv, C_STAGE_W),
        "k_rope tile": (spec.block_k * spec.d_rope, kr_w),
        "W_UV epilogue block": (spec.heads_per_wg * 64 * EPI_VQ * qpw, 32),
    }
    for name, (count, width) in stages.items():
        if count % (threads * width) != 0:
            return False, (
                f"{name} has {count} elements, which does not divide evenly "
                f"across {threads} threads at {width} elements per chunk"
            )
    # The key-tile staging shifts each thread's chunk by whole rows, so a row's
    # chunks must divide the thread count.
    for name, cols, width in (
        ("c_kv", spec.r_kv, C_STAGE_W),
        ("k_rope", spec.d_rope, kr_w),
    ):
        if threads % (cols // width) != 0:
            return False, (
                f"a {name} row is {cols // width} chunks of {width}, which "
                f"must divide the {threads} threads"
            )
    return _fits_lds(spec, arch=arch)


def require_mla_prefill(spec: MlaPrefillSpec, *, arch: str) -> None:
    """Raise :class:`NotImplementedError` if ``spec``/``arch`` is unsupported.

    Called at the top of the builder so an unsupported request fails with a
    clean Python error before any IR is emitted, rather than as a backend
    crash much later.
    """
    ok, reason = supports_mla_prefill(spec, arch=arch)
    if not ok:
        raise NotImplementedError(reason)


def check_mla_prefill_extents(
    spec: MlaPrefillSpec, *, total_q: int, num_kv_blocks: int = 0
) -> tuple[bool, str]:
    """Per-launch 32-bit addressing admission for :func:`build_mla_prefill_fwd`.

    The problem size is a runtime argument, so this has to run per request,
    not per compile. Every offset below is an i32 ``mul nsw`` chain, and signed
    overflow is UB, so an extent at or past 2**31 is refused:

    * query, in elements: ``(token * H + head) * q_cols``, where ``q_cols`` is
      ``d_nope + d_rope`` fused, or ``r_kv + d_rope`` with ``absorbed_query``;
    * output, in elements: ``token * H * d_v + head * d_v`` fused, or the
      latent ``(token * H + head) * r_kv`` with ``absorbed_query`` (the caller
      then writes the ``d_v`` output itself);
    * paged ``c_kv``, in bytes: the buffer-load ``soffset`` is
      ``page * page_block_size * r_kv * 2``, an i32 product, and the buffer
      resource is built with ``num_records`` = KV_RSRC_BYTES (0x7FFF0000, just
      under 2**31). The pool must fit the resource range, which also keeps the
      product clear of overflow. ``k_rope``'s page stride is smaller and never
      binds. Skipped when ``num_kv_blocks`` is 0 (pool size unknown); the
      dispatcher refuses 0, so only a direct caller can skip it.

    The LSE (``token * H + head``) is below every one of these.
    """
    if spec.absorbed_query:
        q_cols, out_cols, out_name = spec.r_kv + spec.d_rope, spec.r_kv, "latent output"
    else:
        q_cols, out_cols, out_name = spec.head_dim_qk, spec.d_v, "output"
    rows = int(total_q) * spec.num_heads
    for name, elems in (("query", rows * q_cols), (out_name, rows * out_cols)):
        if elems >= INT32_LIMIT:
            return False, (
                f"{name} extent is total_q={total_q} x num_heads={spec.num_heads}"
                f" x {elems // rows} = {elems} elements, at or past the 32-bit "
                f"offset limit ({INT32_LIMIT})"
            )
    kv_bytes = int(num_kv_blocks) * spec.page_block_size * spec.r_kv * 2
    if kv_bytes > KV_RSRC_BYTES:
        return False, (
            f"paged c_kv extent is num_kv_blocks={num_kv_blocks} x "
            f"{spec.page_block_size * spec.r_kv * 2} B = {kv_bytes} B, past the "
            f"{KV_RSRC_BYTES:#x} B buffer-resource range"
        )
    return True, ""


def fwd_kernel_args(spec) -> tuple[KernelArg, ...]:
    """The forward kernel's arguments, in declaration order.

    ``max_k`` is unused by the forward kernel but stays declared so it shares
    one argument pack with the score probe. With ``absorbed_query`` the query
    is the absorbed ``qa_ptr`` and there is no ``w_uk_ptr``.
    """
    io = "bf16" if spec.dtype == "bf16" else "f16"
    args = [
        KernelArg("out", "out_ptr", io, "out"),
        KernelArg("lse", "lse_ptr", "f32", "out"),
        KernelArg("q", "qa_ptr" if spec.absorbed_query else "q_ptr", io, "in"),
        KernelArg("c_kv", "c_kv_ptr", io, "in"),
        KernelArg("k_rope", "k_rope_ptr", io, "in"),
    ]
    if not spec.absorbed_query:
        args.append(KernelArg("w_uk", "w_uk_ptr", io, "in"))
    return (*args, *tail_args())


def probe_kernel_args(spec) -> tuple[KernelArg, ...]:
    """The score probe's arguments, in declaration order."""
    io = "bf16" if spec.dtype == "bf16" else "f16"
    return (
        KernelArg("scores", "scores_ptr", "f32", "out"),
        KernelArg("q", "q_ptr", io, "in"),
        KernelArg("c_kv", "c_kv_ptr", io, "in"),
        KernelArg("k_rope", "k_rope_ptr", io, "in"),
        KernelArg("w_uk", "w_uk_ptr", io, "in"),
        *tail_args(),
    )


def mla_prefill_block(spec: MlaPrefillSpec) -> Tuple[int, int, int]:
    """Workgroup shape: one flat dimension of ``64 * num_warps`` threads."""
    return (spec.threads, 1, 1)


def mla_prefill_score_probe_grid(
    spec: MlaPrefillSpec, *, num_q_blocks: int, num_k_tiles: int
) -> Tuple[int, int, int]:
    """Grid for the score probe.

    ``num_q_blocks`` is the packed-varlen block count in the AITER numbering
    :func:`rocke.helpers.attention.binary_search_seq_idx` inverts:
    ``total_q // Bq + num_seqs``, the same count as
    :func:`mla_prefill_fwd_grid` -- **not** ``sum(ceil(S_q / Bq))``, which
    under-launches whenever a ``cu_q`` entry is a multiple of ``Bq``. ``num_k_tiles``
    is ``ceil(max(S_k) / Bk)``; workgroups past a given sequence's own key
    length exit in the prologue.
    """
    if num_q_blocks <= 0 or num_k_tiles <= 0:
        raise ValueError(
            f"grid dims must be positive, got num_q_blocks={num_q_blocks}, "
            f"num_k_tiles={num_k_tiles}"
        )
    return (num_q_blocks, spec.num_heads, num_k_tiles)


def mla_prefill_score_probe_signature(spec: MlaPrefillSpec):
    """Launch signature, in declaration order.

    ``scale`` is the **raw** fp32 scale (``1/sqrt(d_nope + d_rope)``), not
    ``scale * log2(e)``: the probe reproduces
    :func:`builders.mla.ref_mla_attn.ref_mla_prefill_scores`, which is in the
    natural domain because no exponential has been applied yet. The log2-domain
    ABI question belongs to the milestone that introduces softmax.
    """
    return kernel_signature(probe_kernel_args(spec))


def build_mla_prefill_score_probe(
    spec: MlaPrefillSpec, *, arch: str = "gfx942"
) -> KernelDef:
    """Emit the milestone-2 score-tile probe.

    Writes ``scores[q_token, head, k_abs] = scale * dot(q[token, head],
    k[k_abs, head])`` for every (query, key) pair inside this workgroup's tile
    that is in range. No causal mask is applied -- the oracle is called with
    ``causal=False`` to match.
    """
    require_mla_prefill(spec, arch=arch)
    # Unlike the forward kernel, this probe is a single-tile debug aid: it was
    # never generalized over M_TILES/K_TILES, and it still expands the latent
    # in-kernel, so it also needs the d_nope n-tiles to split across waves.
    # supports_mla_prefill() admits multi-tile specs for the forward kernel, so
    # reject them here rather than emitting a silently-wrong probe.
    if spec.block_q != MFMA_M or spec.block_k != MFMA_N:
        raise NotImplementedError(
            "the MLA score probe is single-tile only: block_q must be "
            f"{MFMA_M} and block_k must be {MFMA_N}, got "
            f"block_q={spec.block_q}, block_k={spec.block_k}"
        )
    if (spec.d_nope // MFMA_N) % spec.num_warps != 0:
        raise NotImplementedError(
            f"the MLA score probe expands the latent in-kernel, so "
            f"d_nope/{MFMA_N} = {spec.d_nope // MFMA_N} n-tiles must divide "
            f"evenly across num_warps={spec.num_warps}"
        )
    # Its staging loops hand each thread whole chunks with no remainder, so a
    # tile that does not divide would leave part of the operand unstaged.
    for name, count, width in (
        ("Q tile", spec.block_q * spec.head_dim_qk, 4),
        ("c_kv tile", spec.block_k * spec.r_kv, 8),
        ("k_rope tile", spec.block_k * spec.d_rope, 4),
        ("W_UK slice", spec.r_kv_tile * spec.d_nope, 8),
    ):
        if count % (spec.threads * width) != 0:
            raise NotImplementedError(
                f"the MLA score probe stages its {name} ({count} elements) in "
                f"{width}-wide chunks, which must divide across "
                f"num_warps={spec.num_warps} x 64 threads"
            )

    dtype = BF16
    BQ = spec.block_q
    BK = spec.block_k
    D_NOPE = spec.d_nope
    D_ROPE = spec.d_rope
    HDQK = spec.head_dim_qk
    R_KV = spec.r_kv
    R_TILE = spec.r_kv_tile
    W_COLS = spec.w_uk_cols
    PAGE = spec.page_block_size
    THREADS = spec.threads
    H = spec.num_heads

    N_SLICES = R_KV // R_TILE
    N_TILES_PER_WAVE = (D_NOPE // MFMA_N) // spec.num_warps
    SCORE_K_ITERS = HDQK // MFMA_K

    b, args = open_kernel(spec.kernel_name(), THREADS, probe_kernel_args(spec))
    q_block_global_idx = b.block_id_x()
    head = b.block_id_y()
    k_tile = b.block_id_z()
    lanes = lane_ids(b)

    # ---- ragged bounds -------------------------------------------------
    # Both early exits are uniform across the workgroup (they depend only on
    # block ids), so no barrier below is reached by a partial workgroup.
    tile = varlen_tile(
        b, args, q_block_global_idx, block_q=BQ, iters=spec.binary_search_iters
    )
    s_k = seq_len_k(b, args, tile.seq_idx)
    kb_start = b.mul(k_tile, b.const_i32(BK))
    with b.scf_if(b.cmp_ge(kb_start, s_k)):
        b.ret()

    # ---- LDS -----------------------------------------------------------
    # Q_lds / KT_lds share one 192-wide layout: [:, :128] is the nope half and
    # [:, 128:] the rope half, so the score GEMM is a single uniform K loop.
    q_lds = b.smem_alloc(dtype, [BQ, HDQK], name_hint="q_lds")
    kt_lds = b.smem_alloc(dtype, [BK, HDQK], name_hint="kt_lds")
    c_lds = b.smem_alloc(dtype, [BK, R_KV], name_hint="c_lds")
    wt_lds = b.smem_alloc(dtype, [D_NOPE, R_TILE], name_hint="wt_lds")

    # ---- stage the projected query tile --------------------------------
    qrows = query_rows(b, tile, head, H)
    stage_q_cols(
        b,
        args.q,
        q_lds,
        qrows,
        tid=lanes.tid,
        threads=THREADS,
        rows=BQ,
        cols=HDQK,
        row_cols=HDQK,
        src_col=0,
        dst_col=0,
        dtype=dtype,
    )

    # ---- stage the paged latent tile and its head-shared RoPE half ------
    # page_block_size == block_k, so a k-tile is exactly one page and the
    # in-page row offset is the in-tile row offset. Rows past S_k inside the
    # final page hold whatever the allocator left there. They are finite bf16,
    # they only pollute their own key column, and that column is dropped by
    # the store predicate -- no cross-contamination.
    page = PageTable(b, args.block_table, tile.seq_idx, args.bt_stride).load(k_tile)
    copy_rows_to_lds(
        b,
        args.c_kv,
        c_lds,
        tid=lanes.tid,
        threads=THREADS,
        base=b.mul(page, b.const_i32(PAGE * R_KV)),
        rows=BK,
        cols=R_KV,
        width=8,
        dtype=dtype,
    )
    copy_rows_to_lds(
        b,
        args.k_rope,
        kt_lds,
        tid=lanes.tid,
        threads=THREADS,
        base=b.mul(page, b.const_i32(PAGE * D_ROPE)),
        rows=BK,
        cols=D_ROPE,
        width=4,
        dtype=dtype,
        dst_col=D_NOPE,
    )

    # ---- per-head latent K expansion -----------------------------------
    # K_nope[s, n] = sum_r c_kv[s, r] * W_UK[head, r, n], streamed in r-slices
    # so the 256 KB per-head W_UK never has to fit in LDS. Accumulators are
    # fp32 and live across every slice; wave w owns n-tiles
    # [w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE). The weight is transposed
    # on the store path: gfx942 has no ds_read_tr to do it on the read side,
    # and a strided read-side gather would cost 8x more ds_reads.
    acc = zero_tiles(b, 1, N_TILES_PER_WAVE)
    (acc,) = expand_latent(
        b,
        w_uk=args.w_uk,
        c_lds=c_lds,
        wt_lds=wt_lds,
        tid=lanes.tid,
        wave=lanes.wave,
        lane_row=lanes.lane_row,
        lane_kq=lanes.lane_kq,
        w_head_base=w_head_base(b, head, R_KV, W_COLS),
        col_offset=None,
        dtype=dtype,
        threads=THREADS,
        slices=range(N_SLICES),
        c_col_base=0,
        r_tile=R_TILE,
        d_out=D_NOPE,
        w_cols=W_COLS,
        exp_k_iters=R_TILE // MFMA_K,
        n_tiles_per_wave=N_TILES_PER_WAVE,
        swizzle=False,
        acc=acc,
    )

    # Write the expanded K_nope into the nope half of KT_lds. The C decode
    # gives row = kv token (the A-operand's m) and col = d_nope index (the
    # B-operand's n), which is exactly the [kv_token][d] layout the score
    # GEMM's B operand reads.
    b.sync()
    for t in range(N_TILES_PER_WAVE):
        col = wave_tile_row(b, lanes.wave, N_TILES_PER_WAVE, t, lanes.lane_row)
        store_c_frag(b, kt_lds, acc[t], lanes.lane, col=col)
    b.sync()

    # ---- score GEMM ----------------------------------------------------
    # One 16x16 output tile, so one wave does the whole thing. The nope and
    # rope contributions are the same loop because both operands are laid out
    # 192 wide -- the head-shared RoPE half is simply the tail 64 columns.
    with on_wave(b, lanes.wave, 0):
        s_acc = lds_gemm_tile(
            b, q_lds, kt_lds, lanes, k_iters=SCORE_K_ITERS, dtype=dtype
        )
        store_probe_scores(
            b,
            args.scores,
            s_acc,
            qrows,
            lanes,
            kb_start=kb_start,
            s_k=s_k,
            scale=args.scale,
            row_stride=args.max_k,
        )

    return b.kernel


def mla_prefill_fwd_grid(
    spec: MlaPrefillSpec, *, num_q_blocks: int
) -> Tuple[int, int, int]:
    """Grid for the full forward kernel.

    ``num_q_blocks`` is the packed-varlen block count in the AITER numbering the
    in-kernel binary search inverts: sequence ``i`` owns blocks starting at
    ``cu_q[i] // Bq + i``, so the count is ``total_q // Bq + num_seqs`` -- **not**
    ``sum(ceil(S_q / Bq))``, which under-launches whenever a ``cu_q`` entry is a
    multiple of ``Bq`` and leaves the tail rows of a sequence unwritten. Blocks
    past a sequence's own rows early-return on ``qb_start >= S_q``.

    There is no key dimension: unlike the score probe, one workgroup owns a query
    tile's **entire** key range and walks it with an in-kernel loop, because the
    online softmax state ``(m, l, acc)`` cannot be split across workgroups.
    """
    if num_q_blocks <= 0:
        raise ValueError(f"grid dims must be positive, got num_q_blocks={num_q_blocks}")
    return (num_q_blocks, spec.num_heads // spec.heads_per_wg, 1)


def mla_prefill_fwd_signature(spec: MlaPrefillSpec):
    """Launch signature, in declaration order.

    ``scale`` is the **raw** fp32 scale (``1/sqrt(d_nope + d_rope)``), matching
    the probe and the reference oracle. The kernel folds ``log2(e)`` in itself,
    right where the ``exp2`` lives, so the log2 domain never escapes into the
    ABI.

    ``max_k`` is unused by this kernel (``out``/``lse`` are indexed by packed
    token, not by key) but stays declared so the forward kernel and the probe
    share one 13-argument pack.

    With ``spec.absorbed_query`` the pack has 12 arguments: there is no
    ``w_uk_ptr``, ``qa_ptr`` replaces ``q_ptr`` and holds the absorbed query
    ``[total_q, H, r_kv + d_rope]``, and ``out_ptr`` receives the normalized
    latent output ``[total_q, H, r_kv]`` -- ``W_UV`` is applied by the caller.
    """
    return kernel_signature(fwd_kernel_args(spec))


def build_mla_prefill_fwd(spec: MlaPrefillSpec, *, arch: str = "gfx942") -> KernelDef:
    """Emit the chunked MLA prefill forward kernel, with ``W_UK`` absorbed.

    One workgroup owns one ``(query tile, head)`` pair and loops over the key
    tiles its causal window reaches, carrying ``(m, l, acc_latent)`` in ``scf.for`` iteration
    arguments. The per-head up-projection never enters that loop: it is folded
    into the query on the way in and into the output on the way out, so the
    k-loop only ever touches the compressed latent.

    Prologue, once per workgroup:

    1. read ``Q_nope`` into registers as the absorb's A operand, and stage
       ``Q_rope`` into the ``qa_lds`` columns of ``[r_kv, r_kv + d_rope)``;
    2. absorb -- ``Qa[q, r] = sum_n Q_nope[q, n] * W_UK[head, r, n]`` -- into
       ``qa_lds`` columns ``[0, r_kv)``, staging ``W_UK`` r-slices through
       ``wq_lds``, one wave-aligned column window at a time. Each wave keeps
       its score K-steps of ``Qa`` in registers. ``S[q, k] = sum_r C[k, r] *
       Qa[q, r]`` then reproduces the expanded score exactly.

    Per key tile:

    3. stage ``c_kv[page]`` into ``kv_lds`` columns ``[0, r_kv)`` and
       ``k_rope[page]`` into columns ``[r_kv, r_kv + d_rope)``;
    4. score ``S = scale * (Qa @ [C | K_rope]ᵀ)`` -- one ``K = r_kv + d_rope``
       reduction -- then mask and online-softmax update;
    5. accumulate ``acc_latent[q, r] += sum_k P[q, k] * C[k, r]``.

    Epilogue, once per workgroup:

    6. ``out[q, v] = (sum_r acc_latent[q, r] * W_UV[r, v]) * (1/l[q])``, where
       ``W_UV = w_uk[head, :, d_nope:]``. The rescale and ``1/l`` are linear and
       commute with the projection, so applying ``1/l`` to the ``[q][v]`` result
       is exact.

    The MFMA B operand is read ``[n][k]``: for the latent PV GEMM ``n`` is the
    latent dim and ``k`` is the key, the transpose of how ``kv_lds`` holds
    ``C``. gfx942 has no ``ds_read_tr``, so the PV gathers its B operand from
    ``kv_lds`` one bf16 per key. That read-path transpose measured faster than
    the store-path alternative -- a second, transposed copy of the tile written
    element-wise -- which also cost 20 KB of LDS.

    The k-loop works in the transposed frame: it computes ``Sᵀ`` and
    ``acc_latentᵀ``, swapping the A and B operands of the same fragments. In
    that frame a lane owns one query, so the softmax row reduce is mostly
    in-lane (two cross-lane stages instead of four per row), P comes out of the
    softmax already laid out as the PV GEMM's B operand and never round-trips
    through LDS, and the rescale by ``alpha`` is lane-local. The epilogue's
    ``W_UV`` GEMM is transposed the same way, so each lane normalizes its own
    query and writes four consecutive output columns at once.

    The score GEMM's K reduction is split across waves and the partials are
    summed through LDS in a fixed order, so every wave ends up holding the same
    full score tile. Every wave then redundantly computes the softmax state.
    That is deliberate: ``scf.if`` carries no results, so loop-carried state
    cannot be produced inside a wave-gated region, and a barrier inside one
    would hang. The state is bit-identical in every wave, so only the latent
    accumulator tiles are split across waves and only the final stores are
    predicated.
    """
    require_mla_prefill(spec, arch=arch)
    if _one_wave_per_head(spec):
        return _build_mla_prefill_fwd_heads(spec)

    dtype = BF16
    BQ = spec.block_q
    BK = spec.block_k
    D_NOPE = spec.d_nope
    D_ROPE = spec.d_rope
    D_V = spec.d_v
    HDQK = spec.head_dim_qk
    R_KV = spec.r_kv
    R_TILE = spec.r_kv_tile
    W_COLS = spec.w_uk_cols
    PAGE = spec.page_block_size
    THREADS = spec.threads
    H = spec.num_heads

    # Tiling of the (query, key) score tile across MFMA atoms.
    M_TILES = BQ // MFMA_M
    K_TILES = BK // MFMA_N

    # The absorbed query lives in [r_kv | d_rope]; one reduction covers both.
    QA_COLS = R_KV + D_ROPE
    SCORE_K_ITERS = QA_COLS // MFMA_K
    NUM_WARPS = spec.num_warps
    SCORE_K_PER_WAVE = SCORE_K_ITERS // NUM_WARPS

    # Latent PV accumulator: N_R_PER_WAVE * C_PER_LANE accumulator VGPRs per lane.
    N_R_TILES = R_KV // MFMA_N
    N_R_PER_WAVE = N_R_TILES // spec.num_warps

    # Epilogue W_UV projection, r_kv -> d_v.
    N_V_TILES = D_V // MFMA_N
    N_V_PER_WAVE = N_V_TILES // spec.num_warps
    N_SLICES = R_KV // R_TILE
    EXP_K_ITERS = R_TILE // MFMA_K

    # Prologue Q-absorb, d_nope -> r_kv, one r-slice at a time.
    ABSORB_N_TILES = R_TILE // MFMA_N
    if ABSORB_N_TILES >= spec.num_warps:
        ABSORB_N_PER_WAVE = ABSORB_N_TILES // spec.num_warps
        ABSORB_ACTIVE_WAVES = None
    else:
        # A slice narrower than the workgroup: one n-tile on each of the first
        # ABSORB_N_TILES waves. This is what lets r_kv_tile drop to 32, which
        # brings the prologue -- the LDS peak -- under the two-workgroups-per-CU
        # line.
        ABSORB_N_PER_WAVE = 1
        ABSORB_ACTIVE_WAVES = ABSORB_N_TILES
    ABSORB_K_ITERS = D_NOPE // MFMA_K

    # No AGPRs. Left free, the backend parks the PV accumulators in AGPRs, and
    # the online-softmax rescale of those accumulators then costs a
    # v_accvgpr_read / v_accvgpr_write pair per register on every key tile.
    # gfx942 MFMAs read and write arch VGPRs directly, and the whole register
    # budget still fits the 256 that two waves per SIMD allow.
    b, args = open_kernel(
        spec.fwd_kernel_name(), THREADS, fwd_kernel_args(spec), agpr_alloc=(0, 0)
    )

    # ---- prologue: identify (sequence, query tile, head) ---------------------
    q_block_global_idx, head = reversed_q_block(b, args, block_q=BQ)
    lanes = lane_ids(b)
    wave, lane_row, lane_kq = lanes.wave, lanes.lane_row, lanes.lane_kq
    tile = varlen_tile(
        b, args, q_block_global_idx, block_q=BQ, iters=spec.binary_search_iters
    )
    qb_start = tile.qb_start
    keys = causal_keys(b, args, tile)
    n_k_tiles = causal_k_tiles(b, qb_start, tile, keys, block_q=BQ, block_k=BK)

    # ---- LDS ---------------------------------------------------------------
    # No allocation is exclusive, so the lowerer pools them by liveness. The
    # three phases below never overlap. qa_lds holds one column window and
    # accl_lds one wave's latent columns, each for every M tile:
    #
    #   prologue  wq_lds   + qa_lds                   (BQ=32, BK=16:  27392 B)
    #   loop      s_part   + kv_lds                   (BQ=32, BK=16:  27264 B)
    #   epilogue  accl_lds + wt_lds                   (BQ=32, BK=16:  17920 B)
    #
    # The prologue is the peak, and the pool emitted is exactly that, 27392 B
    # -- under 32768, so two workgroups fit per CU. qa_lds is dead before the
    # loop (each wave reads its score A operand into registers first), so every
    # phase packs from the base. Within a phase the small buffer is allocated
    # first: the packer reuses a dead slot only from its base, so the small
    # buffer takes the wq_lds slot and the large one the qa_lds slot. Achieved
    # layout:
    #
    #   0      wq_lds / s_part / accl_lds
    #   8704   qa_lds / kv_lds / wt_lds    -> 27392 B total
    #
    # **Each buffer is allocated at the point its phase begins, not all up
    # front.** The lowerer seeds a live interval at the ``tile.smem_alloc`` op
    # itself, not at first use, so hoisting every allocation to the top of the
    # kernel would start all six intervals before anything runs and defeat
    # pooling entirely -- the pool becomes the arithmetic sum (71808 B here,
    # well over the 64 KB limit). Keep each ``smem_alloc`` next to the code that
    # first touches it.
    #
    # Pooling means a later buffer may land on an earlier one's bytes, so every
    # phase boundary needs a barrier before its first write: the k-loop body
    # opens with one (fencing the prologue's qa_lds readers against the kv_lds
    # writes) and the epilogue opens with one of its own.
    #
    # Two different pads appear, and the difference is not cosmetic:
    #
    # ``WT_PAD`` columns are never written or read. They exist only to move the
    # row stride off a multiple of the 32-bank LDS width: at the natural
    # [D_V, R_TILE] shape the stride is R_TILE * 2 == 128 B == exactly 32
    # dwords, so every row starts on bank 0. Both accesses walk the row index
    # -- the staging store steps ``n0`` by 8 rows and the MFMA B-operand read
    # steps ``b_row`` by one per lane -- so without the pad each collapses onto
    # a handful of banks.
    #
    # ``V8_PAD`` is an *alignment* requirement, not a tuning knob. ``wq_lds`` is
    # filled with 8-wide bf16 vector stores, and ``smem_store_vN`` declares
    # ``align = n * elem_bytes`` == 16 B. A row stride that is not a multiple of
    # 8 bf16 elements would put odd rows at 8 mod 16 and make that declaration a
    # lie. V8_PAD == 8 keeps the stride 16 B-aligned and lands the buffer off
    # bank 0 (68 dwords, 4 mod 32).
    #
    # ``kv_lds`` used to take ``V8_PAD`` for the same reason and no longer does.
    # Off bank 0 is necessary but not sufficient: a stride of 4 mod 32 has
    # period 8 over the row index, so the score GEMM's 16-lane B read collided
    # 2-way on every issue. It now stages at ``C_STAGE_W`` == 4 and carries
    # ``WT_PAD``, putting the stride at 2 mod 32 -- period 16, conflict-free.
    # See the ``C_STAGE_W`` comment for why the pad could not move on its own.
    #
    # Pads are a layout choice only; every index expression keeps its logical
    # range, so the kernel computes the same values either way.
    #
    # qa_lds carries WT_PAD too: its row stride is then 18 mod 32 dwords at
    # BQ == 32 (2 mod 32 at BQ == 16), period 16 over the row index either
    # way, so the score-fragment read's 16 rows land on distinct banks.
    #
    # The absorb's Q_nope A operand comes straight from global into registers,
    # so there is no q_lds, and every W_UK slice is staged into wq_lds once and
    # fed to all M tiles. qa_lds holds every M tile but only one window of the
    # absorbed columns at a time: a full BQ == 32 qa_lds would be 37120 B on
    # its own. The windows are wave-aligned -- each covers the score K-steps
    # of NUM_WARPS / M_TILES waves -- so after each window the waves that own
    # it take their score fragments into registers.
    # supports_mla_prefill guarantees the windows divide evenly.
    QA_PARTS = M_TILES
    QA_WIN = QA_COLS // QA_PARTS
    WAVES_PER_QA = NUM_WARPS // QA_PARTS
    wq_lds = b.smem_alloc(
        dtype, [R_TILE, D_NOPE + V8_PAD], name_hint="wq_lds"
    )  # 8704 B
    qa_lds = b.smem_alloc(dtype, [BQ, QA_WIN + WT_PAD], name_hint="qa_lds")  # 18688 B

    # ---- load Q once (loop-invariant) --------------------------------------
    # Q_nope is read as the absorb's A fragment (row = lane % 16, four
    # consecutive columns at lane_kq). Q_rope rides along in the absorbed
    # query, at the qa_lds columns of [R_KV, R_KV + D_ROPE), so the score GEMM
    # is one K = R_KV + D_ROPE reduction.
    qrows = query_rows(b, tile, head, H)
    q_frags = [
        load_q_frags(
            b,
            args.q,
            qrows.lane_q_offset(qb_start, lane_row, mt, HDQK, lane_kq),
            n=ABSORB_K_ITERS,
            dtype=dtype,
        )
        for mt in range(M_TILES)
    ]
    w_base = w_head_base(b, head, R_KV, W_COLS)

    # Each wave's score A operand is its own SCORE_K_PER_WAVE-step slice of the
    # absorbed query, and it is loop-invariant: it is read into registers once
    # (one bf16x4 per k-step and M tile) instead of from qa_lds every key tile.
    # qa_lds is then dead before the loop, so the loop's buffers can reuse its
    # bytes. The column inside the window is the wave's rank in its window.
    wave_in_win = window_rank(b, wave, WAVES_PER_QA, QA_PARTS)
    score_col = wave_k_col(b, wave, SCORE_K_PER_WAVE, lane_kq)
    win_col = (
        score_col
        if QA_PARTS == 1
        else wave_k_col(b, wave_in_win, SCORE_K_PER_WAVE, lane_kq)
    )
    qa_frags = [[None] * M_TILES for _ in range(SCORE_K_PER_WAVE)]
    rope_win = R_KV // QA_WIN
    for p in range(QA_PARTS):
        lo = p * QA_WIN
        hi = min(lo + QA_WIN, R_KV)
        if p > 0:
            # WAR: the previous window's fragment reads of qa_lds must retire
            # before this window's writes.
            b.sync_lds_only()
        if p == rope_win:
            stage_q_cols(
                b,
                args.q,
                qa_lds,
                qrows,
                tid=lanes.tid,
                threads=THREADS,
                rows=BQ,
                cols=D_ROPE,
                row_cols=HDQK,
                src_col=D_NOPE,
                dst_col=R_KV - lo,
                dtype=dtype,
            )

        # ---- absorb W_UK into this window of Q, every M tile at once --------
        absorb_query(
            b,
            w_uk=args.w_uk,
            q_frags=q_frags,
            wq_lds=wq_lds,
            qa_lds=qa_lds,
            tid=lanes.tid,
            lane=lanes.lane,
            wave=wave,
            lane_row=lane_row,
            lane_kq=lane_kq,
            w_head_base=w_base,
            dtype=dtype,
            threads=THREADS,
            slices=range(lo // R_TILE, hi // R_TILE),
            qa_col_base=lo,
            r_tile=R_TILE,
            d_nope=D_NOPE,
            w_cols=W_COLS,
            absorb_k_iters=ABSORB_K_ITERS,
            n_tiles_per_wave=ABSORB_N_PER_WAVE,
            active_waves=ABSORB_ACTIVE_WAVES,
            m_tiles=M_TILES,
        )

        # Publish the window. Every wave reads it -- scf.if carries no
        # results -- and keeps the value only if the window is its own.
        b.sync()
        take_window_frags(
            b,
            qa_lds,
            qa_frags,
            lane_row,
            win_col,
            in_win=in_window(b, wave, WAVES_PER_QA, QA_PARTS, p),
            first=p == 0,
            dtype=dtype,
        )

    consts = softmax_consts(b, args.scale)
    iter_args = softmax_iter_args(consts, M_TILES)
    iter_args += acc_iter_args(b, M_TILES, N_R_PER_WAVE)

    # Loop-phase buffers: allocated here so they pool onto wq_lds / qa_lds,
    # whose last readers are the prologue above and are fenced by the loop's
    # opening barrier.
    N_SCORE_TILES = M_TILES * K_TILES
    # Per-wave score partials, one f32x4 C fragment per lane per score tile,
    # lane-major: every wave stores and loads the same layout, so the exchange
    # needs no index remap. Allocated before kv_lds on purpose: the packer
    # reuses a dead slot only from its base, so the small buffer takes the base
    # slot and kv_lds lands at the start of the dead qa_lds slot, below the
    # prologue peak. In the other order the small buffer finds every dead slot
    # overlapping kv_lds and opens a new one above the pool.
    s_part = b.smem_alloc(
        F32, [NUM_WARPS * N_SCORE_TILES, WAVE_SIZE * C_PER_LANE], name_hint="s_part"
    )  # 8192 B
    kv_lds = b.smem_alloc(dtype, [BK, QA_COLS + WT_PAD], name_hint="kv_lds")  # 18560 B
    partials = ScorePartials(
        b, s_part, lanes, num_warps=NUM_WARPS, n_tiles=N_SCORE_TILES
    )

    # ---- k-loop staging, split so the loads can run ahead ------------------
    # Splitting the stage into a load half and a store half lets tile ``i+1``'s
    # global loads issue before tile ``i``'s compute and retire under it,
    # instead of sitting exposed between the two barriers. One stage is in
    # flight: ``c_chunks`` vectors of ``C_STAGE_W`` plus ``kr_chunks`` of 4 --
    # 18 VGPR at BK=16, against 160 of 256 in use.
    #
    # The equivalent pipeline in ``attention_tiled_2d`` hides the same latency
    # with a two-slot LDS double buffer, so its staged registers never cross an
    # iteration boundary. When this was built the pool was near the 64 KB limit
    # and a second slot did not fit, so the stage is carried in ``iter_args``
    # instead.
    #
    # The tile loads are buffer loads. Each thread's byte offset inside a page
    # is loop-invariant and computed once; the page base is workgroup-uniform
    # and rides in the SGPR soffset. Each k-tile then costs no VALU address
    # math -- a 64-bit global address per load otherwise. The rsrc range check
    # applies to voffset alone, which stays inside one page.
    stager = KvRegStager(
        b,
        args,
        kv_lds,
        lanes,
        threads=THREADS,
        block_k=BK,
        r_kv=R_KV,
        d_rope=D_ROPE,
        page_size=PAGE,
        c_width=C_STAGE_W,
        num_records=KV_RSRC_BYTES,
        dtype=dtype,
    )
    pages = PageTable(b, args.block_table, tile.seq_idx, args.bt_stride)

    # Prime the pipeline with tile 0.
    has_tiles = b.cmp_gt(n_k_tiles, b.const_i32(0))
    prologue = stager.loads(pages.load(b.const_i32(0), guard=has_tiles))
    n_stage = len(prologue)
    iter_args += [(f"stg{j}", v) for j, v in enumerate(prologue)]
    # The page the first iteration's run-ahead stages (tile 1, clamped to the
    # last tile), looked up a tile early -- see the run-ahead in the body.
    last_tile = b.sub(n_k_tiles, b.const_i32(1))
    iter_args += [("pgn", pages.load(clamp_tile(b, 1, last_tile), guard=has_tiles))]

    def _k_tile_body(k_tile, state, masked: bool):
        """One key tile. ``masked=False`` drops the per-element causal and
        ragged masks, for tiles every query row of this tile can fully see."""
        ms, ls, accs = split_softmax_state(state, M_TILES, N_R_PER_WAVE)
        staged = list(state[-(n_stage + 1) : -1])
        pg_next = state[-1]
        kb_start = b.mul(k_tile, b.const_i32(BK))

        # WAR: the previous iteration's readers of kv_lds must retire
        # before this iteration overwrites them.
        b.sync()

        # ---- drain this tile's stage into LDS ------------------------------
        # The data is already in registers: it was loaded either by the
        # prologue (tile 0) or by the previous iteration, under that
        # iteration's compute.
        stager.store(staged)
        b.sync()

        # ---- run ahead: issue the next tile's global loads -----------------
        # These retire under the score GEMM below rather than sitting exposed
        # between the two barriers. Their page was looked up one iteration
        # earlier, so they issue without waiting on the block table: a lookup
        # issued here would need a vmcnt(0) before its readfirstlane, exposing
        # one load latency per tile and holding the KV loads back until after
        # the score GEMM. The lookup for the tile after next is issued now and
        # retires under this whole iteration. Indices are clamped rather than
        # predicated: past the end they re-read the last tile's page, which is
        # known to be mapped, and the value is never consumed.
        staged_next = stager.loads(pg_next)
        pg_after = pages.load(tile_ahead(b, k_tile, 2, n_k_tiles))

        # ---- S = scale * (Qa @ [C | K_rope]ᵀ); C decodes as (query, key) ---
        # The K reduction is split across waves: wave w sums k-steps
        # [w * SCORE_K_PER_WAVE, (w + 1) * SCORE_K_PER_WAVE) and the partials
        # are combined through s_part below.
        s_acc = split_k_scores(
            b,
            kv_lds,
            qa_frags,
            lane_row,
            score_col,
            k_steps=SCORE_K_PER_WAVE,
            m_tiles=M_TILES,
            k_tiles=K_TILES,
            sgb_group=SCORE_SGB_GROUP,
            dtype=dtype,
        )
        # Every wave sums the partials in the same order, so the full S -- and
        # with it the softmax state below -- stays bit-identical across waves.
        s_acc = partials.combine(s_acc)

        # ---- online softmax, one query per lane ----------------------------
        mask = (
            key_mask(b, kb_start, lane_kq, keys.s_k, keys.context_off, k_tiles=K_TILES)
            if masked
            else None
        )
        new_ms = []
        new_ls = []
        p_frags = []
        for mt in range(M_TILES):
            row = online_softmax_row(
                b,
                s_acc[mt],
                ms[mt],
                ls[mt],
                consts,
                mask=mask,
                q_abs=row_of(b, qb_start, lane_row, mt) if masked else None,
            )
            new_ms.append(row.m)
            new_ls.append(row.l)
            # accᵀ holds this lane's query in every register, so the rescale is
            # one lane-local alpha for all of them.
            accs[mt] = [scale_frag(b, a, row.alpha) for a in accs[mt]]
            p_frags.append(pack_p(b, row.p, K_TILES, rne=False))

        # ---- accᵀ += Cᵀ · Pᵀ ------------------------------------------------
        for kt in range(K_TILES):
            a_col = shift(b, kt * MFMA_N, lane_kq)
            for t in range(N_R_PER_WAVE):
                b_row = wave_tile_row(b, wave, N_R_PER_WAVE, t, lane_row)
                c_t = gather_keys_t(b, kv_lds, a_col, b_row, dtype)
                for mt in range(M_TILES):
                    accs[mt][t] = mfma_16x16x16(
                        b, dtype, c_t, p_frags[mt][kt], accs[mt][t]
                    )

        flat_acc = [a for row in accs for a in row]
        return [*new_ms, *new_ls, *flat_acc, *staged_next, pg_after]

    n_full = full_k_tiles(b, qb_start, keys, n_k_tiles, block_k=BK)
    res = run_k_loops(
        b,
        iter_args,
        n_full=n_full,
        tail=(n_full, n_k_tiles),
        step=1,
        body=_k_tile_body,
    )

    # ---- epilogue: normalize, then project the latent out with W_UV -------
    ms, ls, accs = split_softmax_state(res, M_TILES, N_R_PER_WAVE)
    inv = [safe_inv_l(b, ls[mt]) for mt in range(M_TILES)]

    # Epilogue buffers: allocated after the loop so they pool onto the loop's
    # slots. That aliasing is why the barrier below is mandatory -- the loop's
    # last readers have to retire before accl_lds is written.
    #
    # accl_lds holds every M tile but only one wave's latent columns at a time:
    # a full BQ == 32 accl_lds would be 33024 B on its own. The latent goes
    # through it one wave's PV tiles at a time, and the W_UV projection walks
    # only that part's r-slices -- so each weight slice is staged once and
    # feeds every M tile. supports_mla_prefill guarantees the parts divide.
    #
    # accl_lds first: it fits the dead wq_lds slot at the base, and wt_lds then
    # takes the dead qa_lds slot.
    R_PART = N_R_PER_WAVE * MFMA_N
    SLICES_PER_PART = N_SLICES // NUM_WARPS
    accl_lds = b.smem_alloc(
        dtype, [BQ, R_PART + WT_PAD], name_hint="accl_lds"
    )  # 8448 B
    wt_lds = b.smem_alloc(dtype, [D_V, R_TILE + WT_PAD], name_hint="wt_lds")  # 9216 B

    out_acc = None
    for p in range(NUM_WARPS):
        # WAR: the loop's last kv_lds readers (p == 0), or the previous part's
        # accl_lds / wt_lds readers, must retire before the stores.
        b.sync()
        # Wave p owns this part.
        store_acc_t(b, accl_lds, accs, lanes, owner=p)

        # out_raw[q, v] = sum_r acc_latent[q, r] * W_UV[r, v],
        # W_UV = w_uk[.., D_NOPE:]. 1/l is linear and commutes with this
        # projection, so it is applied after. ``expand_latent`` opens each
        # slice with its own barrier, which fences both the accl_lds stores
        # above and the WAR on wt_lds.
        out_acc = expand_latent(
            b,
            w_uk=args.w_uk,
            c_lds=accl_lds,
            wt_lds=wt_lds,
            tid=lanes.tid,
            wave=wave,
            lane_row=lane_row,
            lane_kq=lane_kq,
            w_head_base=w_base,
            col_offset=D_NOPE,
            dtype=dtype,
            threads=THREADS,
            slices=range(p * SLICES_PER_PART, (p + 1) * SLICES_PER_PART),
            c_col_base=p * R_PART,
            r_tile=R_TILE,
            d_out=D_V,
            w_cols=W_COLS,
            exp_k_iters=EXP_K_ITERS,
            n_tiles_per_wave=N_V_PER_WAVE,
            m_tiles=M_TILES,
            out_transposed=True,
            acc=out_acc,
        )

    # outᵀ: lane holds query ``lane % 16`` and four consecutive ``v``, so it
    # normalizes by its own 1/l and writes them as one 8 B store.
    for t in range(N_V_PER_WAVE):
        col = wave_tile_row(b, wave, N_V_PER_WAVE, t, lane_kq)
        for mt in range(M_TILES):
            store_out_frag(
                b,
                args.out,
                out_acc[mt][t],
                inv[mt],
                qrows,
                q_local=row_of(b, qb_start, lane_row, mt),
                col=col,
                d_v=D_V,
            )

    # One writer per query: wave 0's first 16-lane group holds every query of
    # each M tile once.
    store_lse(
        b,
        args.lse,
        ms,
        ls,
        qrows,
        q_start=qb_start,
        lane_row=lane_row,
        lane=lanes.lane,
        wave=wave,
    )
    return b.kernel


def _build_mla_prefill_fwd_heads(spec: MlaPrefillSpec) -> KernelDef:
    """:func:`build_mla_prefill_fwd` with ``heads_per_wg`` heads per workgroup.

    The workgroup owns one query tile of ``heads_per_wg`` consecutive heads and
    wave ``w`` owns head ``heads_per_wg * blockIdx.y + w`` outright. In the
    absorbed formulation the latent key tile is shared by every head, so each
    staged ``kv_lds`` tile feeds all of the workgroup's heads -- the head-in-M
    packing of the AITER MLA kernels, with M split across waves one head each.

    Owning a whole head lets every inter-GEMM handoff stay in registers:

    * The query absorb runs transposed, ``Qaᵀ = W_UK · Q_nopeᵀ``, with both
      operands read straight from global memory. Its C fragment (``m = r``,
      ``n = query``) is exactly the score GEMM's B operand layout, so the
      absorbed query never touches LDS.
    * The score GEMM is the wave's own full ``K = r_kv + d_rope`` reduction, so
      there are no split-K partials and no cross-wave combine.
    * The latent accumulator ``accᵀ`` (``m = r``, ``n = query``) is likewise
      the W_UV projection's B operand, so it never round-trips through LDS
      either. Only the W_UV slice itself is staged, transposed, because MFMA
      wants ``r`` contiguous and global memory has ``v`` contiguous.
    """
    dtype = BF16
    # One wave owns BQ query rows (16, or 32 with agpr_acc) of one head; a
    # workgroup holds heads_per_wg heads x q_tiles_per_wg such row tiles.
    QPW = spec.q_tiles_per_wg
    BQ_WG = spec.block_q
    BQ = BQ_WG // QPW
    AGPR_ACC = spec.agpr_acc
    EVQ = EPI_VQ * QPW
    BK = spec.block_k
    D_NOPE = spec.d_nope
    D_ROPE = spec.d_rope
    D_V = spec.d_v
    R_KV = spec.r_kv
    W_COLS = spec.w_uk_cols
    PAGE = spec.page_block_size
    THREADS = spec.threads
    H = spec.num_heads
    HPW = spec.heads_per_wg

    M_TILES = BQ // MFMA_M
    K_TILES = BK // MFMA_N
    QA_COLS = R_KV + D_ROPE
    SCORE_K_ITERS = QA_COLS // MFMA_K
    ROPE_K_ITERS = D_ROPE // MFMA_K
    N_R_TILES = R_KV // MFMA_N
    N_V_TILES = D_V // MFMA_N
    # Unfused: the query arrives absorbed (qa_ptr, QA_COLS wide), there is no
    # W_UK, and out_ptr receives the normalized latent output (R_KV wide).
    UNFUSED = spec.absorbed_query

    # With agpr_acc the index tables are declared noalias, so the backend
    # fetches them with scalar loads: a vector load would share the in-order
    # vmcnt with the key-tile DMA and make the page lookup wait for it.
    b, args = open_kernel(
        spec.fwd_kernel_name(),
        THREADS,
        fwd_kernel_args(spec),
        agpr_alloc=HPW_AGPR_ALLOC,
        index_noalias=spec.agpr_acc,
    )

    # ---- prologue: identify (sequence, query tile, head group) --------------
    # Same reversed, heaviest-first block walk as the one-head kernel, made
    # chiplet-aware when the spec carries the XCD count.
    q_block_global_idx, head_grp = reversed_q_block(
        b,
        args,
        block_q=BQ_WG,
        num_groups=H // HPW,
        num_xcds=spec.num_xcds,
        xcd_chunk=XCD_CHUNK,
    )
    lanes = lane_ids(b)
    lane, wave, lane_row, lane_kq = (
        lanes.lane,
        lanes.wave,
        lanes.lane_row,
        lanes.lane_kq,
    )
    # The wave's head. Wave-uniform, so it goes to an SGPR and every per-head
    # address below is scalar math.
    head_local, head = wave_head(b, head_grp, wave, heads_per_wg=HPW, q_tiles=QPW)
    tile = varlen_tile(
        b, args, q_block_global_idx, block_q=BQ_WG, iters=spec.binary_search_iters
    )
    # The wave's own rows. A wave whose rows all lie past S_q still runs the
    # loop -- the staging and barriers are workgroup-wide -- on rows clamped
    # onto the last token, and its stores are masked off.
    qb_start = wave_q_start(b, tile, wave, q_tiles=QPW, rows=BQ)
    keys = causal_keys(b, args, tile)
    n_k_tiles = causal_k_tiles(b, tile.qb_start, tile, keys, block_q=BQ_WG, block_k=BK)

    # ---- absorbed query, entirely in registers -------------------------------
    # Q is read as an MFMA B fragment (n = query = lane % 16, four columns per
    # lane). Rows past S_q are clamped onto the last valid token.
    #
    # The absorb's d_nope reduction is read 16 B per lane: k-step pair p takes
    # columns 32 p + 8 g .. + 8 in lane group g = lane // 16, and k-step 2 p + h
    # uses half h of them. Q_nope and W_UK use the same permutation of d_nope,
    # so the reduction is unchanged -- only the order the MFMA visits the
    # columns in -- and each operand costs half the loads of an 8 B read.
    # Q_rope is the score GEMM's operand, whose k layout is fixed by kv_lds, so
    # it keeps the plain four-columns-at-lane_kq read.
    qrows = query_rows(b, tile, head, H)
    q_frags = []
    rope_frags = []
    # Unfused: Qa is read in exactly the score B layout -- four columns at
    # 16 t + lane_kq for query lane % 16 -- so there is no absorb at all.
    qa_frags = [[None] * M_TILES for _ in range(N_R_TILES)]
    for mt in range(M_TILES if UNFUSED else 0):
        q_base = qrows.lane_q_offset(qb_start, lane_row, mt, QA_COLS, lane_kq)
        for t, frag in enumerate(
            load_q_frags(b, args.q, q_base, n=N_R_TILES, dtype=dtype)
        ):
            qa_frags[t][mt] = frag
        rope_frags.append(
            load_q_frags(b, args.q, q_base, n=ROPE_K_ITERS, dtype=dtype, col0=R_KV)
        )
    for mt in range(0 if UNFUSED else M_TILES):
        q_nope, q_rope = load_q_nope_rope(
            b,
            args.q,
            qrows,
            qb_start,
            lanes,
            mt,
            d_nope=D_NOPE,
            d_rope=D_ROPE,
            dtype=dtype,
        )
        q_frags.append(q_nope)
        rope_frags.append(q_rope)

    # Qaᵀ[r, q] = sum_n W_UK[head, r, n] * Q_nope[q, n]. The A operand
    # (m = r = lane % 16) is a plain 16 B read of a W_UK row, split as above.
    # The C fragment holds r = 16 t + lane_kq + reg for query lane % 16 -- the
    # score GEMM's B layout for k-step t -- so it is used as is. Q_rope rides
    # in the last D_ROPE / 16 k-steps of the same reduction. The W_UK loads
    # run HPW_ABSORB_PREFETCH r-tiles ahead of the MFMAs.
    if not UNFUSED:
        absorb_query_t(
            b,
            args.w_uk,
            q_frags,
            qa_frags,
            lanes,
            head,
            r_kv=R_KV,
            d_nope=D_NOPE,
            w_cols=W_COLS,
            prefetch=HPW_ABSORB_PREFETCH,
            dtype=dtype,
        )
    # Q_rope is parked in LDS and re-read every key tile rather than held in
    # registers: the eight VGPRs it would pin are what keeps the kernel inside
    # 256. Each lane reads back exactly what it wrote, so no barrier is
    # involved. Allocated here, before kv_lds, and live through the loop.
    qr_lds = b.smem_alloc(
        dtype, [spec.num_warps * BQ, D_ROPE + WT_PAD], name_hint="qr_lds"
    )
    qr_row = park_q_rope(b, qr_lds, rope_frags, lanes, rows=BQ)

    consts = softmax_consts(b, args.scale)
    iter_args = softmax_iter_args(consts, M_TILES)
    if AGPR_ACC:
        acc_file = AgprAcc(b, M_TILES * N_R_TILES)
    else:
        iter_args += acc_iter_args(b, M_TILES, N_R_TILES)

    lane_row4 = b.mul(lane_row, b.const_i32(4))
    # The key tile, shared by every head.
    # With agpr_acc the workgroup has the CU's LDS to itself. The key tile is
    # then DMA'd straight into LDS (buffer_load ... lds) and double-buffered:
    # tile t lives in slot t % 2, the next tile's DMA goes into the other slot
    # right after one barrier, and the loop is unrolled by two so each copy's
    # slot -- and its address immediates -- are static. Per-allocation LDS
    # alias scopes tell the backend that reads of one slot need not wait for
    # the DMA in flight into the other.
    KV_ASYNC = AGPR_ACC
    PV_PF = AGPR_PV_PREFETCH if AGPR_ACC else HPW_PV_PREFETCH
    if KV_ASYNC:
        b.kernel.attrs["lds_alias_scopes"] = True
        # Latent rows [BK][R_KV + pad] and RoPE rows in their own unpadded
        # [BK * D_ROPE] buffer, two slots each.
        kv_lds = b.smem_alloc(dtype, [BK, R_KV + WT_PAD], name_hint="kvc_lds")
        kr_slots = [
            b.smem_alloc(dtype, [BK * D_ROPE], name_hint=f"kr_lds{i}") for i in range(2)
        ]
        kv_slots = [
            kv_lds,
            b.smem_alloc(dtype, [BK, R_KV + WT_PAD], name_hint="kv_lds1"),
        ]
    else:
        kv_lds = b.smem_alloc(dtype, [BK, QA_COLS + WT_PAD], name_hint="kv_lds")
        kr_slots = [None]
        kv_slots = [kv_lds]

    # k_rope is a quarter of the c_kv tile; at eight waves a 4-wide chunk per
    # thread would over-cover it, so the chunk narrows to 2.
    KR_W = 4 if (BK * D_ROPE) % (THREADS * 4) == 0 else 2
    stager = KvSharedStager(
        b,
        args,
        kv_lds,
        lanes,
        threads=THREADS,
        block_k=BK,
        r_kv=R_KV,
        d_rope=D_ROPE,
        page_size=PAGE,
        c_width=C_STAGE_W,
        kr_width=KR_W,
        num_records=KV_RSRC_BYTES,
        dtype=dtype,
    )
    pages = PageTable(b, args.block_table, tile.seq_idx, args.bt_stride)
    if KV_ASYNC:
        dma = KvTileDma(
            b,
            stager,
            lanes,
            kv_slots,
            kr_slots,
            num_warps=spec.num_warps,
            block_k=BK,
            r_kv=R_KV,
            d_rope=D_ROPE,
            page_size=PAGE,
            row_pad=WT_PAD,
        )

    has_tiles = b.cmp_gt(n_k_tiles, b.const_i32(0))
    last_tile = b.sub(n_k_tiles, b.const_i32(1))
    if KV_ASYNC:
        # Tile 0 is DMA'd into slot 0; the loop carries tile 1's page.
        for op in dma.ops(pages.load(b.const_i32(0), guard=has_tiles), 0):
            op()
        n_stage = 0
    else:
        prologue = stager.loads(pages.load(b.const_i32(0), guard=has_tiles))
        n_stage = len(prologue)
        iter_args += [(f"stg{j}", v) for j, v in enumerate(prologue)]
    iter_args += [("pgn", pages.load(clamp_tile(b, 1, last_tile), guard=has_tiles))]

    def _k_tile_body(k_tile, state, *, masked: bool, slot: int = 0):
        ms, ls, accs = split_softmax_state(state, M_TILES, 0 if AGPR_ACC else N_R_TILES)
        staged = list(state[-(n_stage + 1) : -1])
        pg_next = state[-1]
        kb_start = b.mul(k_tile, b.const_i32(BK))

        dma_ops = []
        if KV_ASYNC:
            # One barrier: the barrier's vmcnt(0) lands this tile's DMA, and
            # every reader of the other slot is done, so the next tile's DMA
            # can go there now.
            b.sync()
            dma_ops = dma.ops(pg_next, 1 - slot)
        else:
            b.sync()  # WAR: every head's readers of the previous tile
            stager.store(staged)
            b.sync()
        cur_kv, cur_kr = kv_slots[slot], kr_slots[slot]
        pg_after = pages.load(tile_ahead(b, k_tile, 2, n_k_tiles))

        # ---- Sᵀ = [C | K_rope] · Qaᵀ, the wave's whole K reduction ------------
        # Software-pipelined (see HPW_SCORE_PREFETCH): k-step kk + D's LDS
        # reads are emitted before k-step kk's MFMA, and the fences keep the
        # scheduler from sinking them back. With the DMA'd tile, the next
        # tile's DMA is spread over the fence points, AGPR_DMA_PER_FENCE
        # instructions at a time.
        s_acc = zero_tiles(b, M_TILES, K_TILES)

        def _score_operands(kk):
            col = shift(b, kk * MFMA_K, lane_kq)
            if kk < N_R_TILES:
                q_vs = qa_frags[kk]
            else:
                q_vs = read_q_rope(
                    b, qr_lds, qr_row, lane_kq, kk - N_R_TILES, M_TILES, dtype
                )
            if KV_ASYNC and kk >= N_R_TILES:
                b_vs = read_k_rope_swizzled(
                    b,
                    cur_kr,
                    lanes,
                    kk - N_R_TILES,
                    k_tiles=K_TILES,
                    d_rope=D_ROPE,
                    dtype=dtype,
                )
            else:
                b_vs = read_kv_rows(
                    b, cur_kv, lane_row, col, k_tiles=K_TILES, dtype=dtype
                )
            return q_vs, b_vs

        take = read_ahead(SCORE_K_ITERS, HPW_SCORE_PREFETCH, _score_operands)
        for kk in range(SCORE_K_ITERS):
            q_vs, b_vs = take(kk)
            for kt in range(K_TILES):
                for mt in range(M_TILES):
                    s_acc[mt][kt] = mfma_16x16x16(
                        b, dtype, b_vs[kt], q_vs[mt], s_acc[mt][kt]
                    )
            if (kk + 1) % HPW_SCORE_FENCE_EVERY == 0:
                for _ in range(AGPR_DMA_PER_FENCE if KV_ASYNC else 0):
                    if dma_ops:
                        dma_ops.pop(0)()
                b.sched_barrier(0)

        # The next tile's staging loads go out only now: issued before the
        # score, their registers would be live across it and crowd out the
        # score read-ahead.
        while dma_ops:
            dma_ops.pop(0)()
        staged_next = [] if KV_ASYNC else stager.loads(pg_next)

        # ---- PV read groups ---------------------------------------------------
        # The A operand wants four consecutive keys of one latent column, and
        # kv_lds holds keys as rows. The latent rows of an MFMA tile are free
        # to permute (the epilogue reads W_UV through the same permutation), so
        # row m of tile 4 w + e is r = 64 w + 4 m + e: one 8 B read of a key
        # row then hands a lane its element of four tiles, and four reads (one
        # per key of the lane's k-group) feed four MFMAs -- instead of sixteen
        # scalar reads. The first PV_PF groups (HPW_PV_PREFETCH, or
        # AGPR_PV_PREFETCH with agpr_acc) are read here, ahead
        # of the softmax, which hides their latency behind it.
        pv_groups = [(kt, w) for kt in range(K_TILES) for w in range(N_R_TILES // 4)]
        take_pv = read_ahead(
            len(pv_groups),
            PV_PF,
            lambda gi: read_pv_group(
                b, cur_kv, lane_kq, lane_row4, *pv_groups[gi], dtype=dtype
            ),
        )

        # ---- online softmax, one query per lane (as in the one-head kernel) ---
        mask = (
            key_mask(b, kb_start, lane_kq, keys.s_k, keys.context_off, k_tiles=K_TILES)
            if masked
            else None
        )
        new_ms = []
        new_ls = []
        p_frags = []
        for mt in range(M_TILES):
            # Lazy rescale (HPW_LAZY_TAU): keep the max in use unless the new
            # one is far enough above it.
            row = online_softmax_row(
                b,
                s_acc[mt],
                ms[mt],
                ls[mt],
                consts,
                mask=mask,
                q_abs=row_of(b, qb_start, lane_row, mt) if masked else None,
                lazy_tau=HPW_LAZY_TAU,
            )
            new_ms.append(row.m)
            new_ls.append(row.l)
            # The accumulator rescale runs only when some lane of the wave moved
            # its max.
            if AGPR_ACC:
                acc_file.scale_if(
                    row.any_need, range(mt * N_R_TILES, (mt + 1) * N_R_TILES), row.alpha
                )
            else:
                accs[mt] = rescale_if_any(
                    b, accs[mt], row.alpha, row.any_need, tag=f"{int(masked)}{slot}{mt}"
                )
            p_frags.append(pack_p(b, row.p, K_TILES, rne=True))

        # ---- accᵀ += Cᵀ · Pᵀ over every latent tile -------------------------
        for gi, (kt, w) in enumerate(pv_groups):
            xs = take_pv(gi)
            for e in range(4):
                c_t = pick4(b, xs, e, dtype)
                for mt in range(M_TILES):
                    if AGPR_ACC:
                        acc_file.mfma(mt * N_R_TILES + 4 * w + e, c_t, p_frags[mt][kt])
                    else:
                        accs[mt][4 * w + e] = mfma_16x16x16(
                            b, dtype, c_t, p_frags[mt][kt], accs[mt][4 * w + e]
                        )
            b.sched_barrier(0)

        flat_acc = [a for row in accs for a in row]
        return [*new_ms, *new_ls, *flat_acc, *staged_next, pg_after]

    n_full = full_k_tiles(b, tile.qb_start, keys, n_k_tiles, block_k=BK)
    if KV_ASYNC:
        # Pairs of tiles, slot 0 then slot 1. The unmasked run is rounded down
        # to even; the masked tail is rounded up to even, and its padding tile
        # (index n_k_tiles, past every key) masks to nothing.
        n_full, n_pad = tile_pairs(b, n_full, n_k_tiles)
        step, tail = 2, (n_full, n_pad)
    else:
        step, tail = 1, (n_full, n_k_tiles)

    def _drive(k_tile, state, masked):
        if not KV_ASYNC:
            return _k_tile_body(k_tile, state, masked=masked)
        state = _k_tile_body(k_tile, state, masked=masked, slot=0)
        return _k_tile_body(b.add(k_tile, b.const_i32(1)), state, masked=masked, slot=1)

    res = run_k_loops(b, iter_args, n_full=n_full, tail=tail, step=step, body=_drive)

    # ---- epilogue: project the latent out with W_UV, every head at once ------
    # The epilogue's query rows, rederived from the lane id with agpr_acc.
    lane_row_ep = epilogue_lane_row(b, lanes, rederive=AGPR_ACC)
    ms, ls, accs = split_softmax_state(res, M_TILES, 0 if AGPR_ACC else N_R_TILES)
    if AGPR_ACC:
        acc_file.fence()
        accs = acc_file.view(M_TILES, N_R_TILES)
    inv = [safe_inv_l(b, ls[mt]) for mt in range(M_TILES)]

    if UNFUSED:
        # The normalized latent goes straight from registers to global.
        store_latent_out(
            b,
            args.out,
            accs,
            inv,
            qrows,
            lanes,
            q_start=qb_start,
            lane_row=lane_row_ep,
            r_kv=R_KV,
        )
    else:
        # outᵀ = W_UVᵀ · accᵀ. accᵀ tile t is already the B fragment for its
        # latent k-step; only the A side needs LDS, because MFMA reads W_UVᵀ
        # with r contiguous and global memory has v contiguous. Allocated after
        # the loop so it pools onto the loop's buffers; each round's opening
        # barrier fences their last readers.
        wt_lds = b.smem_alloc(dtype, [HPW * EVQ, 64 + WT_PAD], name_hint="wt_lds")
        out_acc = project_wuv_heads(
            b,
            args.w_uk,
            wt_lds,
            accs,
            lanes,
            head_grp,
            head_local,
            threads=THREADS,
            heads_per_wg=HPW,
            vq=EVQ,
            m_tiles=M_TILES,
            r_kv=R_KV,
            d_nope=D_NOPE,
            d_v=D_V,
            w_cols=W_COLS,
            dtype=dtype,
        )
        for vt in range(N_V_TILES):
            col = shift(b, vt * MFMA_N, lane_kq)
            for mt in range(M_TILES):
                store_out_frag(
                    b,
                    args.out,
                    out_acc[mt][vt],
                    inv[mt],
                    qrows,
                    q_local=row_of(b, qb_start, lane_row_ep, mt),
                    col=col,
                    d_v=D_V,
                )

    # One writer per (query, head): the wave's first 16-lane group.
    store_lse(
        b, args.lse, ms, ls, qrows, q_start=qb_start, lane_row=lane_row_ep, lane=lane
    )
    return b.kernel
