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

    q_lds   <- q[token, head, :]              [Bq, 192]    projected query
    qa_lds[:, r_kv:] <- q_lds[:, d_nope:]     [Bq, 64]     Q_rope, copied
    for r_slice in range(r_KV / r_KV_tile):                  -- Q absorb
        wq_lds  <- W_UK[head, r_slice, :d_nope]           [r_tile, d_nope]
        qa_lds[:, r_slice] <- q_lds[:, :d_nope] @ wq_lds^T
    for k_tile in range(n_k_tiles):                          -- latent loop
        kv_lds[:, :r_kv] <- c_kv[page]        [Bk, r_KV]   paged latent
        kv_lds[:, r_kv:] <- k_rope[page]      [Bk, 64]     head-shared RoPE
        S       <- scale * (qa_lds @ kv_lds^T)             one K=576 GEMM
        P       <- online_softmax(mask(S))
        acc     += P @ kv_lds[:, :r_kv]                    latent-space PV
    accl_lds <- acc                           [Bq, r_KV]
    for r_slice in range(r_KV / r_KV_tile):                  -- V absorb
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

Both kernels take the query **already projected and rotated**
(``[total_q, H_q, d_nope + d_rope]``). Per the design doc the ``W_UQ`` GEMM
and the query-side RoPE are steps 2-3 of a separate pre-kernel, so they are
not this kernel's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import BF16, F32, I32, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers.atoms import MfmaAtom, make_c_warp_dstr_encoding
from rocke.helpers.attention import (
    binary_search_seq_idx as _binary_search_seq_idx,
    mfma_16x16x16_for_dtype as _mfma_16x16x16,
    safe_inv_l as _safe_inv_l,
)
from rocke.helpers.distribution import make_static_tile_distribution
from rocke.helpers.spec import SignatureBuilder

__all__ = [  # noqa: RUF022  -- ordered spec -> gate -> launch geometry -> build
    "MlaPrefillSpec",
    "supports_mla_prefill",
    "require_mla_prefill",
    "mla_prefill_block",
    "mla_prefill_score_probe_grid",
    "mla_prefill_score_probe_signature",
    "build_mla_prefill_score_probe",
    "mla_prefill_fwd_grid",
    "mla_prefill_fwd_signature",
    "build_mla_prefill_fwd",
]

MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
WAVE_SIZE = 64

# C registers each lane holds for one 16x16 MFMA tile.
C_PER_LANE = 4

# Unused trailing columns on an LDS buffer, in bf16 elements. Purely an LDS-layout
# knob: it shifts the row stride off the 32-dword bank width so that a strided
# walk down the row index spreads across banks instead of serialising on one.
# Must stay even -- ``smem_load_vN(..., n=4)`` reads 8 B and an odd element pad
# would misalign every row. 0 restores the natural (bank-conflicting) layout.
WT_PAD = 4

# Same idea as ``WT_PAD``, for buffers that are *written* with 8-wide (16 B) bf16
# vector stores. ``smem_store_vN`` declares ``align = n * elem_bytes``, so an
# 8-wide bf16 store promises 16 B alignment; a pad of 4 would leave odd rows only
# 8 B-aligned and the promise would be a lie. 8 keeps every row start at 16 B.
V8_PAD = 8

# Width, in bf16 elements, of one lane's slice when staging ``c_kv`` into
# ``kv_lds``. This is 4, not 8, and the narrower store is what lets ``kv_lds``
# carry ``WT_PAD`` instead of ``V8_PAD`` -- which is the whole point.
#
# ``kv_lds`` is the B operand of the score GEMM, and that read is the hottest
# LDS access in the kernel (``SCORE_K_ITERS * K_TILES`` of them per k-tile).
# Lane ``l`` of a 16-lane group reads row ``a = l % 16``, so the bank it lands
# on is ``(stride_dwords * a + const) % 32``. With ``V8_PAD`` the stride is 292
# dwords == 4 mod 32, whose period over ``a`` is only 8: lanes ``a`` and
# ``a + 8`` collide on every read, 2-way, for the whole loop. ``WT_PAD`` puts
# the stride at 290 == 2 mod 32, period 16, so all sixteen lanes land on
# distinct banks and the read goes conflict-free.
#
# The pad cannot simply be lowered on its own: a 4-element pad leaves odd rows
# at 8 mod 16 B, and an 8-wide store's ``align = 16`` declaration would become
# a lie -- a silent miscompile, not an assert. So the store width has to come
# down with it. An XOR swizzle was considered instead and is structurally
# blocked: a column bit XORed at bit ``k`` displaces the bank by ``2**(k-1)``
# dwords, and only a displacement of 2 mod 4 breaks a stride of 4 mod 32, which
# pins ``k == 2`` -- exactly the column bit an 8-wide store must preserve.
# Either route therefore requires the narrower store, and given that, the pad
# is the simpler of the two and costs no swizzle arithmetic.
#
# Narrowing is not a concession on either side of the copy. The global load
# stays fully coalesced (lanes read 8 B at an 8 B stride == contiguous), and
# the ``kv_lds`` *store* improves too: at width 8 its column term is ``4a``,
# the same 2-way pattern as the read, while at width 4 it is ``2a`` and goes
# conflict-free as well.
C_STAGE_W = 4

# LDS available to one workgroup on gfx942, in bytes.
LDS_BYTES_PER_WORKGROUP = 64 * 1024

# AMDGPU ``sched.group.barrier`` instruction-class mask bits.
_SGB_MFMA = 0x008  # MFMA / WMMA
_SGB_DS_READ = 0x100  # ds_read (LDS load)

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

_C16_DIST = make_static_tile_distribution(
    make_c_warp_dstr_encoding(MfmaAtom.bf16_16x16x16())
)


def _query_reduce(b: IRBuilder, scalar: Value, *, combine: str) -> Value:
    """Reduce ``scalar`` across the four 16-lane groups of a wave.

    In the transposed score fragment (Sᵀ, ``m = key``, ``n = query``) lane ``l``
    holds query ``l % 16`` and keys ``4 * (l // 16) + r``. After the in-lane
    fold over ``r``, a query's partials sit in lanes ``l``, ``l ^ 16``,
    ``l ^ 32`` and ``l ^ 48``: two XOR stages leave the result in all four. XOR
    16 stays inside a 32-lane half and runs as ``ds_swizzle``; XOR 32 crosses
    halves, which ``ds_swizzle`` cannot, and runs as ``ds_bpermute``.
    """
    fold = b.fadd if combine == "sum" else b.fmax
    v = fold(scalar, b.warp_shuffle_xor(scalar, 16))
    return fold(v, b.warp_shuffle_xor(v, 32))


def _mfma_16x16_c_row(b: IRBuilder, lane, reg: int):
    """MFMA-local output row for a ``16x16`` C element ``reg`` (0..3).

    ``lane`` must be the *wave*-local lane id, not the workgroup thread id:
    every wave decodes its own accumulator independently.
    """
    if not (0 <= reg < 4):
        raise ValueError(f"mfma_16x16 reg must be 0..3, got {reg}")
    m_blk = b.div(lane, b.const_i32(16))
    n = b.mod(lane, b.const_i32(16))
    row, _col = _C16_DIST.calculate_x(
        b, ys=[b.const_i32(0), b.const_i32(reg)], ps=[[m_blk, n]]
    )
    return row


def _shift(b: IRBuilder, offset: int, value: Value) -> Value:
    """``value + offset``, skipping the add when ``offset`` is 0.

    The IR builder is side-effecting: ``b.const_i32(0)`` emits an op whether or
    not its handle is used. Unrolled tile loops that degenerate to a single
    iteration would otherwise leave a trail of dead constants in the IR.
    """
    return value if offset == 0 else b.add(b.const_i32(offset), value)


def _wt_swizzle(b: IRBuilder, row: Value, col: Value, r_tile: int) -> Value:
    """Permute ``wt_lds`` 4-column groups by ``(row >> 4) & mask``.

    ``wt_lds`` is the epilogue's transposed ``W_UV`` slice, ``[d_v, r_tile]``,
    written one bf16 at a time and read as the MFMA B operand. Within a 16-lane group
    the store's column ``rl`` is uniform and its row is ``lane * 8 + e``, so at
    the ``WT_PAD`` stride (2 mod 32 dwords) the row term ``lane * 16`` reaches
    only two banks. XORing the column group with row bits 4 and up -- lane bits
    1-3, disjoint from the stride's lane bit 0 -- spreads it across 16.

    On the read side ``row >> 4`` is the MFMA n-tile, constant across the 16
    lanes of one read, so the twist only relabels which groups a tile reads and
    the read keeps its conflict-free bank pattern. Groups of four keep the
    ``n=4`` read contiguous and 8 B-aligned. The mask is the widest power of two
    (at most 8) dividing the group count, so the XOR never leaves the row.
    """
    groups = r_tile // 4
    mask = 1
    while mask < 8 and groups % (mask * 2) == 0:
        mask *= 2
    twist = b.land(b.lshr(row, b.const_i32(4)), b.const_i32(mask - 1))
    grp = b.xor(b.lshr(col, b.const_i32(2)), twist)
    return b.lor(b.shl(grp, b.const_i32(2)), b.land(col, b.const_i32(3)))


@dataclass(frozen=True)
class MlaPrefillSpec:
    """Immutable description of one gfx942 MLA prefill configuration.

    The defaults are the design doc's bring-up values, not tuned values. The
    only field a caller normally sets is ``num_heads`` (128 for a DeepSeek-V3
    shaped model, 64 for a Kimi-K2 shaped one).
    """

    num_heads: int
    d_nope: int = 128
    d_rope: int = 64
    d_v: int = 128
    r_kv: int = 512
    block_q: int = 16
    block_k: int = 16
    # 32, not the bring-up 64: it is the widest slice that keeps the prologue,
    # the LDS peak, under 32 KB, so two workgroups fit per CU -- which measured
    # faster than 64 on every shape tried.
    r_kv_tile: int = 32
    num_warps: int = 4
    page_block_size: int = 16
    dtype: str = "bf16"
    binary_search_iters: int = 8

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
            f"_{self.dtype}"
        )


def _fwd_lds_bytes(spec: MlaPrefillSpec) -> int:
    """Predict the LDS pool :func:`build_mla_prefill_fwd` will ask for.

    The lowerer pools the seven buffers by liveness, so the pool is *not* their
    arithmetic sum -- see the LDS comment block in the builder for the achieved
    layout. The three phases never share a live buffer: the prologue trio
    (``q_lds``, ``wq_lds``, ``qa_lds``) is dead before the k-loop starts (the
    score A operand is read into registers first), and by the epilogue
    everything earlier is dead. Each phase therefore packs from the base on its
    own and the pool is the widest phase.

    The packer reuses a dead slot only from its start offset. Each later phase
    allocates its small buffer first (``s_part``, ``wt_lds``) so it takes the
    base slot, and the large one lands at the start of the next dead slot.
    Erring high is the point: this gates admission, so it must never
    under-predict. It is a model of observed behaviour rather than a guarantee,
    so check the emitted pool whenever a new geometry is introduced.
    """
    elem = 2  # bf16
    qa_cols = spec.r_kv + spec.d_rope
    q_lds = elem * spec.block_q * (spec.d_nope + WT_PAD)
    wq_lds = elem * spec.r_kv_tile * (spec.d_nope + V8_PAD)
    qa_lds = elem * spec.block_q * (qa_cols + WT_PAD)
    kv_lds = elem * spec.block_k * (qa_cols + WT_PAD)
    s_part = 4 * spec.num_warps * spec.block_q * spec.block_k
    accl_lds = elem * spec.block_q * (spec.r_kv + WT_PAD)
    wt_lds = elem * spec.d_v * (spec.r_kv_tile + WT_PAD)

    prologue = q_lds + wq_lds + qa_lds
    # s_part takes the base slot and kv_lds the wq_lds slot; wt_lds takes the
    # base slot and accl_lds the qa_lds slot (see the allocation sites).
    loop = max(q_lds, s_part) + kv_lds
    epilogue = max(wt_lds, q_lds + wq_lds) + accl_lds
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
    # Every global->LDS staging loop below hands each thread a fixed number of
    # fixed-width vector chunks with no remainder handling, so the element
    # count must divide by threads * width -- not merely by threads.
    threads = spec.threads
    stages = {
        "Q_nope tile": (spec.block_q * spec.d_nope, 4),
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
    # Register blocking is cheap in VGPRs and expensive in LDS: qa_lds scales
    # with block_q and kv_lds with block_k, so the 64 KB budget -- not the
    # accumulator -- is what caps the tile here.
    lds = _fwd_lds_bytes(spec)
    if lds > LDS_BYTES_PER_WORKGROUP:
        return False, (
            f"the forward kernel needs about {lds} B of LDS at "
            f"block_q={spec.block_q}, block_k={spec.block_k}, r_kv={spec.r_kv}, "
            f"which exceeds the {LDS_BYTES_PER_WORKGROUP} B per-workgroup budget"
        )
    return True, ""


def require_mla_prefill(spec: MlaPrefillSpec, *, arch: str) -> None:
    """Raise :class:`NotImplementedError` if ``spec``/``arch`` is unsupported.

    Called at the top of the builder so an unsupported request fails with a
    clean Python error before any IR is emitted, rather than as a backend
    crash much later.
    """
    ok, reason = supports_mla_prefill(spec, arch=arch)
    if not ok:
        raise NotImplementedError(reason)


def mla_prefill_block(spec: MlaPrefillSpec) -> Tuple[int, int, int]:
    """Workgroup shape: one flat dimension of ``64 * num_warps`` threads."""
    return (spec.threads, 1, 1)


def mla_prefill_score_probe_grid(
    spec: MlaPrefillSpec, *, num_q_blocks: int, num_k_tiles: int
) -> Tuple[int, int, int]:
    """Grid for the score probe.

    ``num_q_blocks`` is the packed-varlen total, i.e. ``sum(ceil(S_q / Bq))``
    over the batch -- the same quantity
    :func:`rocke.helpers.attention.binary_search_seq_idx` inverts. ``num_k_tiles``
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
    io_dtype = "bf16" if spec.dtype == "bf16" else "f16"
    return (
        SignatureBuilder()
        .ptr("scores_ptr", "f32")
        .ptr("q_ptr", io_dtype)
        .ptr("c_kv_ptr", io_dtype)
        .ptr("k_rope_ptr", io_dtype)
        .ptr("w_uk_ptr", io_dtype)
        .ptr("cu_seqlens_q_ptr", "i32")
        .ptr("cu_seqlens_k_ptr", "i32")
        .ptr("block_table_ptr", "i32")
        .scalar("scale", "f32")
        .scalar("num_seqs", "i32")
        .scalar("block_table_stride", "i32")
        .scalar("max_k", "i32")
        .build()
    )


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
    N_TILES = D_NOPE // MFMA_N  # n-tiles of the K-expansion GEMM
    N_TILES_PER_WAVE = N_TILES // spec.num_warps
    EXP_K_ITERS = R_TILE // MFMA_K  # k-iters per r-slice
    SCORE_K_ITERS = HDQK // MFMA_K

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    scores = b.param(
        "scores_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=16
    )
    q_ptr = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    c_kv = b.param(
        "c_kv_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k_rope = b.param(
        "k_rope_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    w_uk = b.param(
        "w_uk_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    cu_q = b.param("cu_seqlens_q_ptr", PtrType(I32, "global"), readonly=True, align=4)
    cu_k = b.param("cu_seqlens_k_ptr", PtrType(I32, "global"), readonly=True, align=4)
    block_table = b.param(
        "block_table_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale_p = b.param("scale", F32)
    num_seqs_p = b.param("num_seqs", I32)
    bt_stride_p = b.param("block_table_stride", I32)
    max_k_p = b.param("max_k", I32)

    q_block_global_idx = b.block_id_x()
    head = b.block_id_y()
    k_tile = b.block_id_z()

    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(WAVE_SIZE))
    wave = b.div(tid, b.const_i32(WAVE_SIZE))
    lane_row = b.mod(lane, b.const_i32(16))  # MFMA A-row / B-row selector
    lane_kq = b.mul(b.div(lane, b.const_i32(16)), b.const_i32(4))  # K sub-offset

    # ---- ragged bounds -------------------------------------------------
    seq_idx = _binary_search_seq_idx(
        b,
        cu_q,
        q_block_global_idx,
        num_seqs_p,
        block_q=BQ,
        iterations=spec.binary_search_iters,
    )
    cu_q_start = b.global_load_i32(cu_q, seq_idx)
    cu_q_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    s_q = b.sub(cu_q_stop, cu_q_start)

    # Same inversion as the dense prefill kernel: the binary search's loop
    # invariant is `cu_q[i] // BLOCK_Q + i <= target`, so the first q-block of
    # this sequence sits at `cu_q_start // BQ + seq_idx`.
    q_block_start_idx = b.add(b.div(cu_q_start, b.const_i32(BQ)), seq_idx)
    q_block_local_idx = b.sub(q_block_global_idx, q_block_start_idx)
    qb_start = b.mul(q_block_local_idx, b.const_i32(BQ))
    with b.scf_if(b.cmp_ge(qb_start, s_q)):
        b.ret()

    cu_k_start = b.global_load_i32(cu_k, seq_idx)
    cu_k_stop = b.global_load_i32(cu_k, b.add(seq_idx, b.const_i32(1)))
    s_k = b.sub(cu_k_stop, cu_k_start)
    kb_start = b.mul(k_tile, b.const_i32(BK))
    with b.scf_if(b.cmp_ge(kb_start, s_k)):
        b.ret()

    # Both early exits are uniform across the workgroup (they depend only on
    # block ids), so no barrier below is reached by a partial workgroup.

    # ---- LDS -----------------------------------------------------------
    # Q_lds / KT_lds share one 192-wide layout: [:, :128] is the nope half and
    # [:, 128:] the rope half, so the score GEMM is a single uniform K loop.
    q_lds = b.smem_alloc(dtype, [BQ, HDQK], name_hint="q_lds")
    kt_lds = b.smem_alloc(dtype, [BK, HDQK], name_hint="kt_lds")
    c_lds = b.smem_alloc(dtype, [BK, R_KV], name_hint="c_lds")
    wt_lds = b.smem_alloc(dtype, [D_NOPE, R_TILE], name_hint="wt_lds")

    # ---- stage the projected query tile --------------------------------
    # Rows past S_q are clamped onto the last valid token rather than masked.
    # They still produce scores, but every such score is dropped by the store
    # predicate below, and clamping keeps the load in-bounds without needing a
    # masked vector load (which the builder does not offer).
    q_last_row = b.sub(s_q, b.const_i32(1))
    q_chunks = (BQ * HDQK) // (THREADS * 4)
    q_chunks_per_row = HDQK // 4
    for j in range(q_chunks):
        c = b.add(tid, b.const_i32(j * THREADS))
        m = b.div(c, b.const_i32(q_chunks_per_row))
        d0 = b.mul(b.mod(c, b.const_i32(q_chunks_per_row)), b.const_i32(4))
        q_local = b.add(qb_start, m)
        q_local = b.select(b.cmp_lt(q_local, s_q), q_local, q_last_row)
        token = b.add(cu_q_start, q_local)
        idx = b.add(
            b.mul(b.add(b.mul(token, b.const_i32(H)), head), b.const_i32(HDQK)), d0
        )
        b.smem_store_vN(q_lds, [m, d0], b.global_load_vN(q_ptr, idx, dtype, 4), 4)

    # ---- stage the paged latent tile and its head-shared RoPE half ------
    # page_block_size == block_k, so a k-tile is exactly one page and the
    # in-page row offset is the in-tile row offset.
    page = b.global_load_i32(block_table, b.add(b.mul(seq_idx, bt_stride_p), k_tile))
    c_base = b.mul(page, b.const_i32(PAGE * R_KV))
    c_chunks = (BK * R_KV) // (THREADS * 8)
    c_chunks_per_row = R_KV // 8
    for j in range(c_chunks):
        c = b.add(tid, b.const_i32(j * THREADS))
        m = b.div(c, b.const_i32(c_chunks_per_row))
        r0 = b.mul(b.mod(c, b.const_i32(c_chunks_per_row)), b.const_i32(8))
        idx = b.add(b.add(c_base, b.mul(m, b.const_i32(R_KV))), r0)
        b.smem_store_vN(c_lds, [m, r0], b.global_load_vN(c_kv, idx, dtype, 8), 8)

    # Rows past S_k inside the final page hold whatever the allocator left
    # there. They are finite bf16, they only pollute their own key column, and
    # that column is dropped by the store predicate -- no cross-contamination.
    kr_base = b.mul(page, b.const_i32(PAGE * D_ROPE))
    kr_chunks = (BK * D_ROPE) // (THREADS * 4)
    kr_chunks_per_row = D_ROPE // 4
    for j in range(kr_chunks):
        c = b.add(tid, b.const_i32(j * THREADS))
        m = b.div(c, b.const_i32(kr_chunks_per_row))
        d0 = b.mul(b.mod(c, b.const_i32(kr_chunks_per_row)), b.const_i32(4))
        idx = b.add(b.add(kr_base, b.mul(m, b.const_i32(D_ROPE))), d0)
        b.smem_store_vN(
            kt_lds,
            [m, b.add(d0, b.const_i32(D_NOPE))],
            b.global_load_vN(k_rope, idx, dtype, 4),
            4,
        )

    # ---- per-head latent K expansion -----------------------------------
    # K_nope[s, n] = sum_r c_kv[s, r] * W_UK[head, r, n], streamed in r-slices
    # so the 256 KB per-head W_UK never has to fit in LDS. Accumulators are
    # fp32 and live across every slice; wave w owns n-tiles
    # [w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE).
    acc = [b.zero_vec_f32(4) for _ in range(N_TILES_PER_WAVE)]
    w_head_base = b.mul(b.mul(head, b.const_i32(R_KV)), b.const_i32(W_COLS))
    wt_chunks = (R_TILE * D_NOPE) // (THREADS * 8)
    wt_chunks_per_row = D_NOPE // 8

    for s in range(N_SLICES):
        r_base = s * R_TILE
        # Single-buffered LDS: drain readers of the previous slice before
        # overwriting it. b.sync() emits `s_waitcnt vmcnt(0) lgkmcnt(0)`
        # ahead of the barrier, so the lgkmcnt drain is already covered.
        b.sync()
        # Store-path transpose: read along the contiguous n axis, scatter to
        # wt_lds[n][r_local]. gfx942 has no ds_read_tr to do this on the read
        # side, and a strided read-side gather would cost 8x more ds_reads.
        for j in range(wt_chunks):
            c = b.add(tid, b.const_i32(j * THREADS))
            rl = b.div(c, b.const_i32(wt_chunks_per_row))
            n0 = b.mul(b.mod(c, b.const_i32(wt_chunks_per_row)), b.const_i32(8))
            idx = b.add(
                b.add(
                    w_head_base,
                    b.mul(b.add(rl, b.const_i32(r_base)), b.const_i32(W_COLS)),
                ),
                n0,
            )
            v8 = b.global_load_vN(w_uk, idx, dtype, 8)
            for e in range(8):
                b.smem_store_vN(
                    wt_lds, [b.add(n0, b.const_i32(e)), rl], b.vec_extract(v8, e), 1
                )
        b.sync()

        for kk in range(EXP_K_ITERS):
            a_col = b.add(b.const_i32(r_base + kk * MFMA_K), lane_kq)
            a_v = b.smem_load_vN(c_lds, lane_row, a_col, dtype=dtype, n=4)
            b_col = b.add(b.const_i32(kk * MFMA_K), lane_kq)
            for t in range(N_TILES_PER_WAVE):
                n_tile = b.add(
                    b.mul(wave, b.const_i32(N_TILES_PER_WAVE)), b.const_i32(t)
                )
                b_row = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row)
                b_v = b.smem_load_vN(wt_lds, b_row, b_col, dtype=dtype, n=4)
                acc[t] = _mfma_16x16x16(b, dtype, a_v, b_v, acc[t])

    # Write the expanded K_nope into the nope half of KT_lds. The C decode
    # gives row = kv token (the A-operand's m) and col = d_nope index (the
    # B-operand's n), which is exactly the [kv_token][d] layout the score
    # GEMM's B operand reads.
    b.sync()
    for t in range(N_TILES_PER_WAVE):
        n_tile = b.add(b.mul(wave, b.const_i32(N_TILES_PER_WAVE)), b.const_i32(t))
        col = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row)
        packed = b.vec_trunc_f32_to_bf16(acc[t])
        for reg in range(4):
            row = _mfma_16x16_c_row(b, lane, reg)
            b.smem_store_vN(kt_lds, [row, col], b.vec_extract(packed, reg), 1)
    b.sync()

    # ---- score GEMM ----------------------------------------------------
    # One 16x16 output tile, so one wave does the whole thing. The nope and
    # rope contributions are the same loop because both operands are laid out
    # 192 wide -- the head-shared RoPE half is simply the tail 64 columns.
    with b.scf_if(b.cmp_eq(wave, b.const_i32(0))):
        s_acc = b.zero_vec_f32(4)
        for kk in range(SCORE_K_ITERS):
            col = b.add(b.const_i32(kk * MFMA_K), lane_kq)
            a_v = b.smem_load_vN(q_lds, lane_row, col, dtype=dtype, n=4)
            b_v = b.smem_load_vN(kt_lds, lane_row, col, dtype=dtype, n=4)
            s_acc = _mfma_16x16x16(b, dtype, a_v, b_v, s_acc)

        k_abs = b.add(kb_start, lane_row)
        for reg in range(4):
            row = _mfma_16x16_c_row(b, lane, reg)
            q_local = b.add(qb_start, row)
            value = b.fmul(b.vec_extract(s_acc, reg), scale_p)
            # Two-term predicate as nested ifs: the builder exposes no boolean
            # AND over i1.
            with b.scf_if(b.cmp_lt(q_local, s_q)):
                with b.scf_if(b.cmp_lt(k_abs, s_k)):
                    token = b.add(cu_q_start, q_local)
                    idx = b.add(
                        b.mul(
                            b.add(b.mul(token, b.const_i32(H)), head),
                            max_k_p,
                        ),
                        k_abs,
                    )
                    b.global_store(scores, idx, value, align=4)

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
    return (num_q_blocks, spec.num_heads, 1)


def mla_prefill_fwd_signature(spec: MlaPrefillSpec):
    """Launch signature, in declaration order.

    ``scale`` is the **raw** fp32 scale (``1/sqrt(d_nope + d_rope)``), matching
    the probe and the reference oracle. The kernel folds ``log2(e)`` in itself,
    right where the ``exp2`` lives, so the log2 domain never escapes into the
    ABI.

    ``max_k`` is unused by this kernel (``out``/``lse`` are indexed by packed
    token, not by key) but stays declared so the forward kernel and the probe
    share one 13-argument pack.
    """
    io_dtype = "bf16" if spec.dtype == "bf16" else "f16"
    return (
        SignatureBuilder()
        .ptr("out_ptr", io_dtype)
        .ptr("lse_ptr", "f32")
        .ptr("q_ptr", io_dtype)
        .ptr("c_kv_ptr", io_dtype)
        .ptr("k_rope_ptr", io_dtype)
        .ptr("w_uk_ptr", io_dtype)
        .ptr("cu_seqlens_q_ptr", "i32")
        .ptr("cu_seqlens_k_ptr", "i32")
        .ptr("block_table_ptr", "i32")
        .scalar("scale", "f32")
        .scalar("num_seqs", "i32")
        .scalar("block_table_stride", "i32")
        .scalar("max_k", "i32")
        .build()
    )


def _expand_latent(
    b: IRBuilder,
    *,
    w_uk,
    c_lds,
    wt_lds,
    tid,
    wave,
    lane_row,
    lane_kq,
    w_head_base,
    col_offset: int,
    dtype,
    threads: int,
    n_slices: int,
    r_tile: int,
    d_out: int,
    w_cols: int,
    exp_k_iters: int,
    n_tiles_per_wave: int,
    m_tiles: int = 1,
    out_transposed: bool = False,
):
    """``C_lds @ W_UK[head, :, col_offset : col_offset + d_out]``, in registers.

    Walks the latent dimension in ``r_tile`` slices, staging each slice of the
    weight through ``wt_lds`` transposed on the store path (MFMA reads the B
    operand as ``[n][k]``, and gfx942 has no ``ds_read_tr``). The caller owns the
    C-layout writeback, and the barrier that opens each slice lives here so a
    caller that shares the ``wt_lds`` slot with something else needs no extra
    fencing of its own.

    ``m_tiles`` register-blocks the M axis: the weight slice in ``wt_lds`` is
    read once and fed to every M tile, so a taller ``C_lds`` costs extra MFMAs
    and accumulators but no extra LDS traffic on the B side.

    Returns ``acc[m_tile][n_tile]``, one f32x4 per (M tile, n-tile owned by the
    calling wave).
    """
    acc = [[b.zero_vec_f32(4) for _ in range(n_tiles_per_wave)] for _ in range(m_tiles)]
    wt_chunks = (r_tile * d_out) // (threads * 8)
    wt_chunks_per_row = d_out // 8
    for s in range(n_slices):
        r_base = s * r_tile
        b.sync()  # WAR: previous slice's (or previous GEMM's) readers must retire
        for j in range(wt_chunks):
            c = b.add(tid, b.const_i32(j * threads))
            rl = b.div(c, b.const_i32(wt_chunks_per_row))
            n0 = b.mul(b.mod(c, b.const_i32(wt_chunks_per_row)), b.const_i32(8))
            idx = b.add(
                b.add(
                    w_head_base,
                    b.mul(b.add(rl, b.const_i32(r_base)), b.const_i32(w_cols)),
                ),
                b.add(n0, b.const_i32(col_offset)),
            )
            v8 = b.global_load_vN(w_uk, idx, dtype, 8)
            for e in range(8):
                row = b.add(n0, b.const_i32(e))
                b.smem_store_vN(
                    wt_lds,
                    [row, _wt_swizzle(b, row, rl, r_tile)],
                    b.vec_extract(v8, e),
                    1,
                )
        b.sync()
        for kk in range(exp_k_iters):
            a_col = b.add(b.const_i32(r_base + kk * MFMA_K), lane_kq)
            a_vs = [
                b.smem_load_vN(
                    c_lds, _shift(b, mt * MFMA_M, lane_row), a_col, dtype=dtype, n=4
                )
                for mt in range(m_tiles)
            ]
            b_col = b.add(b.const_i32(kk * MFMA_K), lane_kq)
            for t in range(n_tiles_per_wave):
                n_tile = b.add(
                    b.mul(wave, b.const_i32(n_tiles_per_wave)), b.const_i32(t)
                )
                b_row = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row)
                b_v = b.smem_load_vN(
                    wt_lds,
                    b_row,
                    _wt_swizzle(b, b_row, b_col, r_tile),
                    dtype=dtype,
                    n=4,
                )
                for mt in range(m_tiles):
                    if out_transposed:
                        # outᵀ = W_UVᵀ · accᵀ: the same two fragments, swapped,
                        # so the C fragment is (m = v, n = query).
                        acc[mt][t] = _mfma_16x16x16(b, dtype, b_v, a_vs[mt], acc[mt][t])
                    else:
                        acc[mt][t] = _mfma_16x16x16(b, dtype, a_vs[mt], b_v, acc[mt][t])
    return acc


def _absorb_query(
    b: IRBuilder,
    *,
    w_uk,
    q_lds,
    wq_lds,
    qa_lds,
    tid,
    lane,
    wave,
    lane_row,
    lane_kq,
    w_head_base,
    dtype,
    threads: int,
    n_slices: int,
    r_tile: int,
    d_nope: int,
    w_cols: int,
    absorb_k_iters: int,
    n_tiles_per_wave: int,
    m_tiles: int,
    active_waves: int | None = None,
):
    """``Qa[q, r] = sum_n Q_nope[q, n] * W_UK[head, r, n]``, written to ``qa_lds``.

    The W_UK absorption of the query side. Because
    ``S[q,k] = Q_nope[q] . K_nope[k] = Q_nope[q] . (C[k] @ W_UK_nope)``
    equals ``Qa[q] . C[k]``, folding ``W_UK`` into ``Q`` once per workgroup lets
    the whole key loop run in the 512-wide latent space.

    Unlike :func:`_expand_latent` this needs **no** store-path transpose: MFMA
    reads the B operand as ``[n][k]``, here ``[r][nope]``, which is exactly how
    ``w_uk[head, r, :d_nope]`` is already laid out. The weight slice therefore
    goes to LDS with contiguous 8-wide stores.

    Walks ``r`` in ``r_tile`` slices. Each slice's accumulator is complete after
    the full ``K = d_nope`` reduction, so it is written straight through to
    ``qa_lds`` and the registers are reused by the next slice.
    """
    wq_chunks = (r_tile * d_nope) // (threads * 8)
    wq_chunks_per_row = d_nope // 8
    for s in range(n_slices):
        r_base = s * r_tile
        b.sync()  # WAR on wq_lds; also fences the q_lds staging on s == 0
        for j in range(wq_chunks):
            c = b.add(tid, b.const_i32(j * threads))
            rl = b.div(c, b.const_i32(wq_chunks_per_row))
            n0 = b.mul(b.mod(c, b.const_i32(wq_chunks_per_row)), b.const_i32(8))
            idx = b.add(
                b.add(
                    w_head_base,
                    b.mul(b.add(rl, b.const_i32(r_base)), b.const_i32(w_cols)),
                ),
                n0,
            )
            b.smem_store_vN(wq_lds, [rl, n0], b.global_load_vN(w_uk, idx, dtype, 8), 8)
        b.sync()

        # With fewer n-tiles than waves (``active_waves``), only the first
        # ``active_waves`` waves compute; the rest skip to the next slice's
        # barrier. The gated region holds no barrier and its only outputs are
        # qa_lds stores, so carrying no results out of the ``scf.if`` is fine.
        def _slice_compute(r_base: int):
            acc = [
                [b.zero_vec_f32(4) for _ in range(n_tiles_per_wave)]
                for _ in range(m_tiles)
            ]
            for kk in range(absorb_k_iters):
                col = b.add(b.const_i32(kk * MFMA_K), lane_kq)
                a_vs = [
                    b.smem_load_vN(
                        q_lds, _shift(b, mt * MFMA_M, lane_row), col, dtype=dtype, n=4
                    )
                    for mt in range(m_tiles)
                ]
                for t in range(n_tiles_per_wave):
                    n_tile = b.add(
                        b.mul(wave, b.const_i32(n_tiles_per_wave)), b.const_i32(t)
                    )
                    b_row = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row)
                    b_v = b.smem_load_vN(wq_lds, b_row, col, dtype=dtype, n=4)
                    for mt in range(m_tiles):
                        acc[mt][t] = _mfma_16x16x16(b, dtype, a_vs[mt], b_v, acc[mt][t])
            # C decodes as [m = q][n = r - r_base]; scatter it into the latent query.
            for t in range(n_tiles_per_wave):
                n_tile = b.add(
                    b.mul(wave, b.const_i32(n_tiles_per_wave)), b.const_i32(t)
                )
                col = _shift(
                    b, r_base, b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row)
                )
                for mt in range(m_tiles):
                    packed = b.vec_trunc_f32_to_bf16(acc[mt][t])
                    for reg in range(C_PER_LANE):
                        row = _shift(b, mt * MFMA_M, _mfma_16x16_c_row(b, lane, reg))
                        b.smem_store_vN(
                            qa_lds, [row, col], b.vec_extract(packed, reg), 1
                        )

        if active_waves is None:
            _slice_compute(r_base)
        else:
            with b.scf_if(b.cmp_lt(wave, b.const_i32(active_waves))):
                _slice_compute(r_base)


def build_mla_prefill_fwd(spec: MlaPrefillSpec, *, arch: str = "gfx942") -> KernelDef:
    """Emit the chunked MLA prefill forward kernel, with ``W_UK`` absorbed.

    One workgroup owns one ``(query tile, head)`` pair and loops over the key
    tiles its causal window reaches, carrying ``(m, l, acc_latent)`` in ``scf.for`` iteration
    arguments. The per-head up-projection never enters that loop: it is folded
    into the query on the way in and into the output on the way out, so the
    k-loop only ever touches the compressed latent.

    Prologue, once per workgroup:

    1. stage ``Q_nope`` into ``q_lds`` and ``Q_rope`` into ``qa_lds`` columns
       ``[r_kv, r_kv + d_rope)``;
    2. absorb -- ``Qa[q, r] = sum_n Q_nope[q, n] * W_UK[head, r, n]`` -- into
       ``qa_lds`` columns ``[0, r_kv)``, staging ``W_UK`` r-slices through
       ``wq_lds``. ``S[q, k] = sum_r C[k, r] * Qa[q, r]`` then reproduces the
       expanded score exactly.

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

    LOG2E = math.log2(math.e)
    LN2 = math.log(2.0)

    b = IRBuilder(spec.fwd_kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    # No AGPRs. Left free, the backend parks the PV accumulators in AGPRs, and
    # the online-softmax rescale of those accumulators then costs a
    # v_accvgpr_read / v_accvgpr_write pair per register on every key tile.
    # gfx942 MFMAs read and write arch VGPRs directly, and the whole register
    # budget still fits the 256 that two waves per SIMD allow.
    b.kernel.attrs["agpr_alloc"] = (0, 0)

    out = b.param(
        "out_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    lse = b.param(
        "lse_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=16
    )
    q_ptr = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    c_kv = b.param(
        "c_kv_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k_rope = b.param(
        "k_rope_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    w_uk = b.param(
        "w_uk_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    cu_q = b.param("cu_seqlens_q_ptr", PtrType(I32, "global"), readonly=True, align=4)
    cu_k = b.param("cu_seqlens_k_ptr", PtrType(I32, "global"), readonly=True, align=4)
    block_table = b.param(
        "block_table_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale_p = b.param("scale", F32)
    num_seqs_p = b.param("num_seqs", I32)
    bt_stride_p = b.param("block_table_stride", I32)
    b.param("max_k", I32)  # unused here; keeps one 13-argument pack with the probe

    # ---- prologue: identify (sequence, query tile, head) ---------------------
    # Block ids are walked in reverse. With the causal loop bound a query tile's
    # work grows with its position in the sequence, and workgroups launch in
    # block-id order, so reversing puts the heaviest tiles first and leaves the
    # light ones to fill the tail. The grid's x extent is the AITER count
    # ``total_q // BQ + num_seqs`` (``mla_prefill_fwd_grid``), and
    # ``total_q == cu_seqlens_q[num_seqs]``.
    total_q = b.global_load_i32(cu_q, num_seqs_p)
    num_q_blocks = b.add(b.div(total_q, b.const_i32(BQ)), num_seqs_p)
    q_block_global_idx = b.sub(b.sub(num_q_blocks, b.const_i32(1)), b.block_id_x())
    head = b.block_id_y()
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(WAVE_SIZE))
    wave = b.div(tid, b.const_i32(WAVE_SIZE))
    lane_row = b.mod(lane, b.const_i32(16))
    lane_kq = b.mul(b.div(lane, b.const_i32(16)), b.const_i32(4))

    seq_idx = _binary_search_seq_idx(
        b,
        cu_q,
        q_block_global_idx,
        num_seqs_p,
        block_q=BQ,
        iterations=spec.binary_search_iters,
    )
    cu_q_start = b.global_load_i32(cu_q, seq_idx)
    cu_q_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    s_q = b.sub(cu_q_stop, cu_q_start)

    q_block_start_idx = b.add(b.div(cu_q_start, b.const_i32(BQ)), seq_idx)
    q_block_local_idx = b.sub(q_block_global_idx, q_block_start_idx)
    qb_start = b.mul(q_block_local_idx, b.const_i32(BQ))

    # Uniform across the workgroup (block ids only), so no barrier below is
    # reached by a partial workgroup.
    with b.scf_if(b.cmp_ge(qb_start, s_q)):
        b.ret()

    cu_k_start = b.global_load_i32(cu_k, seq_idx)
    cu_k_stop = b.global_load_i32(cu_k, b.add(seq_idx, b.const_i32(1)))
    s_k = b.sub(cu_k_stop, cu_k_start)
    # Bottom-right causal alignment: query row ``i`` of this sequence attends to
    # keys ``0 .. i + context_off``. ``context_off`` is negative when S_k < S_q,
    # which is exactly what produces the fully masked leading rows the driver
    # checks for exact zeros / exact -inf.
    context_off = b.sub(s_k, s_q)

    # Causal loop bound: no row of this tile sees a key at or past
    # ``q_end + context_off``, where ``q_end`` is the tile's last real row + 1,
    # so the tiles beyond it are fully masked and are skipped rather than
    # computed and discarded. ``q_end <= s_q`` keeps the bound within ``s_k``;
    # the clamp at 0 covers tiles whose rows are all fully masked, which then run
    # zero iterations and finish with ``l == 0`` exactly as masking would leave
    # them. The per-element masks below still handle the ragged edge tiles.
    q_end = b.select(
        b.cmp_lt(b.add(qb_start, b.const_i32(BQ)), s_q),
        b.add(qb_start, b.const_i32(BQ)),
        s_q,
    )
    k_end = b.add(q_end, context_off)
    k_end = b.select(b.cmp_gt(k_end, b.const_i32(0)), k_end, b.const_i32(0))
    n_k_tiles = b.div(b.add(k_end, b.const_i32(BK - 1)), b.const_i32(BK))

    # ---- LDS ---------------------------------------------------------------
    # No allocation is exclusive, so the lowerer pools them by liveness. The
    # three phases below never overlap, and the loop phase is the peak:
    #
    #   prologue  q_lds  + wq_lds + qa_lds            (BQ=BK=16:  31488 B)
    #   loop      s_part + kv_lds                     (BQ=BK=16:  22784 B)
    #   epilogue  wt_lds + accl_lds                   (BQ=BK=16:  29440 B)
    #
    # The prologue is the peak, and the pool emitted is exactly that, 31488 B --
    # under 32768, so two workgroups fit per CU. qa_lds is dead before the loop
    # (each wave reads its score A operand into registers first), so every
    # phase packs from the base. Within a phase the small buffer is allocated
    # first: the packer reuses a dead slot only from its base, so the small
    # buffer takes the base slot and the large one the next dead slot. Achieved
    # layout:
    #
    #   0      q_lds / s_part / wt_lds
    #   4224   wq_lds / kv_lds
    #   12928  qa_lds / accl_lds           -> 31488 B total
    #
    # **Each buffer is allocated at the point its phase begins, not all up
    # front.** The lowerer seeds a live interval at the ``tile.smem_alloc`` op
    # itself, not at first use, so hoisting every allocation to the top of the
    # kernel would start all seven intervals before anything runs and defeat
    # pooling entirely -- the pool becomes the arithmetic sum (79872 B here,
    # well over the 64 KB limit). Keep each ``smem_alloc`` next to the code that
    # first touches it.
    #
    # Pooling means a later buffer may land on an earlier one's bytes, so every
    # phase boundary needs a barrier before its first write: the k-loop body
    # opens with one (fencing the prologue's q_lds/wq_lds readers against the
    # kv_lds writes) and the epilogue opens with one of its own.
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
    # WT_PAD for the same reason as kv_lds: unpadded, the 384 B row stride is a
    # multiple of the 32-bank width, so the absorb's 16-lane A read lands every
    # row on bank 0. Padded, the stride is 2 mod 32 dwords -- conflict-free.
    q_lds = b.smem_alloc(dtype, [BQ, D_NOPE + WT_PAD], name_hint="q_lds")  # 4224 B
    wq_lds = b.smem_alloc(
        dtype, [R_TILE, D_NOPE + V8_PAD], name_hint="wq_lds"
    )  # 8704 B
    qa_lds = b.smem_alloc(dtype, [BQ, QA_COLS + WT_PAD], name_hint="qa_lds")  # 18560 B

    # ---- stage Q once (loop-invariant) -------------------------------------
    # Q_nope goes to q_lds, the absorb's A operand. Q_rope rides along in the
    # absorbed query at qa_lds columns [R_KV, R_KV + D_ROPE), so the score GEMM
    # is one K = R_KV + D_ROPE reduction instead of two; it is staged there
    # straight from global. The absorb's first slice barrier publishes q_lds,
    # and the barrier after the absorb publishes qa_lds.
    q_last_row = b.sub(s_q, b.const_i32(1))

    def _stage_q(dst, *, width: int, src_col: int, dst_col: int):
        chunks = (BQ * width) // (THREADS * 4)
        per_row = width // 4
        for j in range(chunks):
            c = b.add(tid, b.const_i32(j * THREADS))
            m = b.div(c, b.const_i32(per_row))
            d0 = b.mul(b.mod(c, b.const_i32(per_row)), b.const_i32(4))
            q_local = b.add(qb_start, m)
            q_local = b.select(b.cmp_lt(q_local, s_q), q_local, q_last_row)
            token = b.add(cu_q_start, q_local)
            idx = b.add(
                b.mul(b.add(b.mul(token, b.const_i32(H)), head), b.const_i32(HDQK)),
                _shift(b, src_col, d0),
            )
            v = b.global_load_vN(q_ptr, idx, dtype, 4)
            b.smem_store_vN(dst, [m, _shift(b, dst_col, d0)], v, 4)

    _stage_q(q_lds, width=D_NOPE, src_col=0, dst_col=0)
    _stage_q(qa_lds, width=D_ROPE, src_col=D_NOPE, dst_col=R_KV)

    w_head_base = b.mul(b.mul(head, b.const_i32(R_KV)), b.const_i32(W_COLS))

    # ---- absorb W_UK into Q, once per workgroup ---------------------------
    _absorb_query(
        b,
        w_uk=w_uk,
        q_lds=q_lds,
        wq_lds=wq_lds,
        qa_lds=qa_lds,
        tid=tid,
        lane=lane,
        wave=wave,
        lane_row=lane_row,
        lane_kq=lane_kq,
        w_head_base=w_head_base,
        dtype=dtype,
        threads=THREADS,
        n_slices=N_SLICES,
        r_tile=R_TILE,
        d_nope=D_NOPE,
        w_cols=W_COLS,
        absorb_k_iters=ABSORB_K_ITERS,
        n_tiles_per_wave=ABSORB_N_PER_WAVE,
        active_waves=ABSORB_ACTIVE_WAVES,
        m_tiles=M_TILES,
    )

    # Each wave's score A operand is its own SCORE_K_PER_WAVE-step slice of the
    # absorbed query, and it is loop-invariant: read it into registers once
    # (one bf16x4 per k-step and M tile) instead of from qa_lds every key tile.
    # qa_lds is then dead before the loop, so the loop's buffers can reuse its
    # bytes. The barrier publishes the absorb's qa_lds stores; the loop's
    # opening barrier fences these reads against the loop's first writes.
    b.sync()
    score_col = b.add(b.mul(wave, b.const_i32(SCORE_K_PER_WAVE * MFMA_K)), lane_kq)
    qa_frags = [
        [
            b.smem_load_vN(
                qa_lds,
                _shift(b, mt * MFMA_M, lane_row),
                _shift(b, kk * MFMA_K, score_col),
                dtype=dtype,
                n=4,
            )
            for mt in range(M_TILES)
        ]
        for kk in range(SCORE_K_PER_WAVE)
    ]

    # Fold log2(e) here, next to the exp2 that consumes it: the ABI stays raw.
    scale_log2 = b.fmul(scale_p, b.const_f32(LOG2E))
    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    # The loop works in the transposed frame: it computes Sᵀ (key x query) and
    # accᵀ (latent x query), so each lane owns one query of each M tile --
    # ``lane % 16`` -- and its softmax state is one (m, l) pair per M tile.
    iter_args = [(f"m{mt}", neg_inf) for mt in range(M_TILES)]
    iter_args += [(f"l{mt}", zero_f) for mt in range(M_TILES)]
    iter_args += [
        (f"acc{mt}_{t}", b.zero_vec_f32(4))
        for mt in range(M_TILES)
        for t in range(N_R_PER_WAVE)
    ]
    n_ml = M_TILES

    # Loop-phase buffers: allocated here so they pool onto q_lds / wq_lds, whose
    # last readers are the absorb above and are fenced by the loop's opening
    # barrier.
    N_SCORE_TILES = M_TILES * K_TILES
    # Per-wave score partials, one f32x4 C fragment per lane per score tile,
    # lane-major: every wave stores and loads the same layout, so the exchange
    # needs no index remap. Allocated before kv_lds on purpose: the packer
    # reuses a dead slot only from its base, so the small buffer takes the base
    # slot and kv_lds lands at the start of the dead wq_lds slot, below the
    # prologue peak. In the other order the small buffer finds every dead slot
    # overlapping kv_lds and opens a new one above the pool.
    s_part = b.smem_alloc(
        F32, [NUM_WARPS * N_SCORE_TILES, WAVE_SIZE * C_PER_LANE], name_hint="s_part"
    )  # 4096 B
    kv_lds = b.smem_alloc(dtype, [BK, QA_COLS + WT_PAD], name_hint="kv_lds")  # 18560 B
    part_row = b.mul(wave, b.const_i32(N_SCORE_TILES))
    lane_part = b.mul(lane, b.const_i32(C_PER_LANE))

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
    c_chunks = (BK * R_KV) // (THREADS * C_STAGE_W)
    c_chunks_per_row = R_KV // C_STAGE_W
    kr_chunks = (BK * D_ROPE) // (THREADS * 4)
    kr_chunks_per_row = D_ROPE // 4

    # The tile loads are buffer loads. Each thread's byte offset inside a page
    # is loop-invariant and computed once here; the page base is
    # workgroup-uniform and rides in the SGPR soffset. Each k-tile then costs
    # no VALU address math -- a 64-bit global address per load otherwise. The
    # rsrc range check applies to voffset alone, which stays inside one page.
    big_bytes = b.const_i32(0x7FFF0000)
    c_rsrc = b.buffer_rsrc(c_kv, big_bytes)
    kr_rsrc = b.buffer_rsrc(k_rope, big_bytes)
    c_voffs = []
    for j in range(c_chunks):
        c = b.add(tid, b.const_i32(j * THREADS))
        m = b.div(c, b.const_i32(c_chunks_per_row))
        r0 = b.mul(b.mod(c, b.const_i32(c_chunks_per_row)), b.const_i32(C_STAGE_W))
        c_voffs.append(b.mul(b.add(b.mul(m, b.const_i32(R_KV)), r0), b.const_i32(2)))
    kr_voffs = []
    for j in range(kr_chunks):
        c = b.add(tid, b.const_i32(j * THREADS))
        m = b.div(c, b.const_i32(kr_chunks_per_row))
        d0 = b.mul(b.mod(c, b.const_i32(kr_chunks_per_row)), b.const_i32(4))
        kr_voffs.append(b.mul(b.add(b.mul(m, b.const_i32(D_ROPE)), d0), b.const_i32(2)))

    def _stage_loads(tile, *, guard=None):
        """Issue one tile's global loads into registers. Touches no LDS."""
        page = b.global_load_i32(block_table, b.add(b.mul(seq_idx, bt_stride_p), tile))
        if guard is not None:
            # Zero-trip loop: the block-table slot may be uninitialised. Page 0
            # is always mapped, so the load stays in bounds; a zero-trip loop
            # never consumes the value.
            page = b.select(guard, page, b.const_i32(0))
        page = b.readfirstlane(page)
        c_soff = b.mul(page, b.const_i32(PAGE * R_KV * 2))
        kr_soff = b.mul(page, b.const_i32(PAGE * D_ROPE * 2))
        staged = [
            b.buffer_load_vN(c_rsrc, voff, c_soff, dtype, C_STAGE_W) for voff in c_voffs
        ]
        staged += [
            b.buffer_load_vN(kr_rsrc, voff, kr_soff, dtype, 4) for voff in kr_voffs
        ]
        return staged

    def _stage_store(staged):
        """Drain a staged tile into ``kv_lds``, ``[key][r]``.

        That is the natural layout for the score GEMM's B operand. The latent
        PV reads the same buffer transposed, one bf16 per key.
        """
        for j in range(c_chunks):
            c = b.add(tid, b.const_i32(j * THREADS))
            m = b.div(c, b.const_i32(c_chunks_per_row))
            r0 = b.mul(b.mod(c, b.const_i32(c_chunks_per_row)), b.const_i32(C_STAGE_W))
            b.smem_store_vN(kv_lds, [m, r0], staged[j], C_STAGE_W)
        for j in range(kr_chunks):
            c = b.add(tid, b.const_i32(j * THREADS))
            m = b.div(c, b.const_i32(kr_chunks_per_row))
            d0 = b.mul(b.mod(c, b.const_i32(kr_chunks_per_row)), b.const_i32(4))
            b.smem_store_vN(
                kv_lds, [m, b.add(d0, b.const_i32(R_KV))], staged[c_chunks + j], 4
            )

    # Prime the pipeline with tile 0.
    prologue = _stage_loads(b.const_i32(0), guard=b.cmp_gt(n_k_tiles, b.const_i32(0)))
    n_stage = len(prologue)
    iter_args += [(f"stg{j}", v) for j, v in enumerate(prologue)]

    def _k_tile_body(k_tile, state, *, masked: bool):
        """One key tile. ``masked=False`` drops the per-element causal and
        ragged masks, for tiles every query row of this tile can fully see."""
        ms = list(state[:M_TILES])
        ls = list(state[M_TILES : 2 * M_TILES])
        accs = [
            list(
                state[2 * n_ml + mt * N_R_PER_WAVE : 2 * n_ml + (mt + 1) * N_R_PER_WAVE]
            )
            for mt in range(M_TILES)
        ]
        staged = list(state[-n_stage:])
        kb_start = b.mul(k_tile, b.const_i32(BK))

        # WAR: the previous iteration's readers of kv_lds must retire
        # before this iteration overwrites them.
        b.sync()

        # ---- drain this tile's stage into LDS, in both orientations --------
        # The data is already in registers: it was loaded either by the
        # prologue (tile 0) or by the previous iteration, under that
        # iteration's compute.
        _stage_store(staged)
        b.sync()

        # ---- run ahead: issue the next tile's global loads -----------------
        # These retire under the score GEMM below rather than sitting exposed
        # between the two barriers. The index is clamped rather than
        # predicated: on the final iteration it re-reads the current tile,
        # a page already known to be mapped, and the value is never consumed.
        nxt = b.select(
            b.cmp_lt(b.add(k_tile, b.const_i32(1)), n_k_tiles),
            b.add(k_tile, b.const_i32(1)),
            k_tile,
        )
        staged_next = _stage_loads(nxt)

        # ---- S = scale * (Qa @ [C | K_rope]ᵀ); C decodes as (query, key) ---
        # The K reduction is split across waves: wave w sums k-steps
        # [w * SCORE_K_PER_WAVE, (w + 1) * SCORE_K_PER_WAVE) and the partials
        # are combined through s_part below.
        s_acc = [[b.zero_vec_f32(4) for _ in range(K_TILES)] for _ in range(M_TILES)]
        for kk in range(SCORE_K_PER_WAVE):
            col = _shift(b, kk * MFMA_K, score_col)
            a_vs = qa_frags[kk]
            for kt in range(K_TILES):
                b_v = b.smem_load_vN(
                    kv_lds, _shift(b, kt * MFMA_N, lane_row), col, dtype=dtype, n=4
                )
                # Sᵀ: kv_lds is the A operand (m = key), the query fragment the
                # B operand (n = query) -- the same two fragments, swapped.
                for mt in range(M_TILES):
                    s_acc[mt][kt] = _mfma_16x16x16(
                        b, dtype, b_v, a_vs[mt], s_acc[mt][kt]
                    )
            # Hoist the next ``SCORE_SGB_GROUP`` steps' LDS reads above the
            # MFMAs they feed. Skipped when the group does not divide the trip
            # count -- a partial trailing block measured worse than no hint.
            if (
                SCORE_K_PER_WAVE % SCORE_SGB_GROUP == 0
                and (kk + 1) % SCORE_SGB_GROUP == 0
            ):
                b.sched_group_barrier(_SGB_DS_READ, SCORE_SGB_GROUP * K_TILES, 0)
                b.sched_group_barrier(_SGB_MFMA, SCORE_SGB_GROUP * M_TILES * K_TILES, 0)

        # Combine the partials. Every wave sums them in the same order, so the
        # full S -- and with it the softmax state below -- stays bit-identical
        # across waves. The partial rows' previous readers retired before this
        # iteration's opening barrier, so only the RAW side needs fencing.
        for mt in range(M_TILES):
            for kt in range(K_TILES):
                row = _shift(b, mt * K_TILES + kt, part_row)
                b.smem_store_vN(s_part, [row, lane_part], s_acc[mt][kt], C_PER_LANE)
        b.sync_lds_only()
        for mt in range(M_TILES):
            for kt in range(K_TILES):
                total = None
                for w in range(NUM_WARPS):
                    part = b.smem_load_vN(
                        s_part,
                        b.const_i32(w * N_SCORE_TILES + mt * K_TILES + kt),
                        lane_part,
                        dtype=F32,
                        n=C_PER_LANE,
                    )
                    total = part if total is None else b.fadd(total, part)
                s_acc[mt][kt] = total

        # ---- online softmax, one query per lane ----------------------------
        # In the Sᵀ fragment a lane holds query ``lane % 16`` of its M tile and
        # keys ``kt*16 + 4*(lane/16) + r``. A query's row reduce is therefore
        # in-lane over its 4*K_TILES keys plus two cross-lane stages over the
        # four 16-lane groups.
        #
        # Two bounds, and both land on ``p`` rather than only on ``s``: the
        # ragged key bound (the last tile overhangs S_k, where page-granular
        # staging supplies real but meaningless bytes) and the bottom-right
        # causal bound. Masking ``s`` alone is not enough -- with the -1e30
        # sentinel a fully masked row gets m_new == -1e30, so every element
        # would yield exp2(s - m_new) == 1 and ``l`` would come out BK, not 0.
        # l == 0 is what makes ``safe_inv_l`` store exact zeros and the LSE
        # collapse to a true -inf.
        if masked:
            k_abs = [
                [
                    b.add(kb_start, _shift(b, kt * MFMA_N + r, lane_kq))
                    for r in range(C_PER_LANE)
                ]
                for kt in range(K_TILES)
            ]
            in_k = [[b.cmp_lt(k, s_k) for k in row] for row in k_abs]
        new_ms = []
        new_ls = []
        p_frags = [[None] * K_TILES for _ in range(M_TILES)]
        for mt in range(M_TILES):
            keep = []
            s_v = []
            if masked:
                q_abs = b.add(qb_start, _shift(b, mt * MFMA_M, lane_row))
                causal_hi = b.add(q_abs, context_off)
            for kt in range(K_TILES):
                for r in range(C_PER_LANE):
                    s_scaled = b.fmul(b.vec_extract(s_acc[mt][kt], r), scale_log2)
                    if masked:
                        keep.append(
                            b.land(in_k[kt][r], b.cmp_le(k_abs[kt][r], causal_hi))
                        )
                        s_v.append(b.select(keep[-1], s_scaled, neg_inf))
                    else:
                        s_v.append(s_scaled)
            local_max = s_v[0]
            for v in s_v[1:]:
                local_max = b.fmax(local_max, v)
            m_new = b.fmax(ms[mt], _query_reduce(b, local_max, combine="max"))
            # exp2_fast: one v_exp_f32, no range-reduction guard. Both
            # arguments here are <= 0 (m_new is a max over m and s), so there
            # is no overflow, and an underflow to 0 is the right answer for a
            # softmax weight.
            alpha = b.exp2_fast(b.fsub(ms[mt], m_new))
            if masked:
                p_v = [
                    b.select(k, b.exp2_fast(b.fsub(v, m_new)), zero_f)
                    for k, v in zip(keep, s_v)
                ]
            else:
                p_v = [b.exp2_fast(b.fsub(v, m_new)) for v in s_v]
            local_sum = p_v[0]
            for v in p_v[1:]:
                local_sum = b.fadd(local_sum, v)
            new_ms.append(m_new)
            new_ls.append(
                b.fadd(
                    b.fmul(ls[mt], alpha), _query_reduce(b, local_sum, combine="sum")
                )
            )
            # accᵀ holds this lane's query in every register, so the rescale is
            # one lane-local alpha for all of them.
            for t in range(N_R_PER_WAVE):
                for r in range(C_PER_LANE):
                    old = b.vec_extract(accs[mt][t], r)
                    accs[mt][t] = b.vec_insert(accs[mt][t], b.fmul(old, alpha), r)
            # The Sᵀ fragment is already the PV GEMM's B operand layout
            # (k = key, n = query), so P never leaves registers.
            for kt in range(K_TILES):
                p_frags[mt][kt] = b.vec_trunc_f32_to_bf16(
                    b.vec_pack(p_v[kt * C_PER_LANE : (kt + 1) * C_PER_LANE], F32)
                )

        # ---- accᵀ += Cᵀ · Pᵀ ------------------------------------------------
        for kt in range(K_TILES):
            a_col = _shift(b, kt * MFMA_N, lane_kq)
            for t in range(N_R_PER_WAVE):
                n_tile = b.add(b.mul(wave, b.const_i32(N_R_PER_WAVE)), b.const_i32(t))
                b_row = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row)
                # A[m][k] = Cᵀ[r][key], gathered from kv_lds four keys at a time:
                # the transpose happens on the read path, one bf16 per key.
                c_t = b.vec_pack(
                    [
                        b.vec_extract(
                            b.smem_load_vN(
                                kv_lds, _shift(b, i, a_col), b_row, dtype=dtype, n=1
                            ),
                            0,
                        )
                        for i in range(4)
                    ],
                    dtype,
                )
                for mt in range(M_TILES):
                    accs[mt][t] = _mfma_16x16x16(
                        b, dtype, c_t, p_frags[mt][kt], accs[mt][t]
                    )

        flat_acc = [accs[mt][t] for mt in range(M_TILES) for t in range(N_R_PER_WAVE)]
        b.scf_yield(*new_ms, *new_ls, *flat_acc, *staged_next)

    # Only the last tile or two of a query tile can be partially masked: a key
    # tile is fully visible when its last key is below s_k and at or below the
    # causal bound of the tile's first query row. Those run first without any
    # per-element masking; the rest run masked. The run-ahead stage and the
    # softmax state flow from the first loop into the second through the loop
    # results.
    n_full = b.div(b.add(qb_start, b.add(context_off, b.const_i32(1))), b.const_i32(BK))
    k_whole = b.div(s_k, b.const_i32(BK))
    n_full = b.select(b.cmp_lt(n_full, k_whole), n_full, k_whole)
    n_full = b.select(b.cmp_gt(n_full, b.const_i32(0)), n_full, b.const_i32(0))
    n_full = b.select(b.cmp_lt(n_full, n_k_tiles), n_full, n_k_tiles)

    main_loop = b.scf_for_iter(
        b.const_i32(0), n_full, b.const_i32(1), iter_args=iter_args, iv_name="k_main"
    )
    with main_loop as (k_tile, state):
        _k_tile_body(k_tile, state, masked=False)
    tail_loop = b.scf_for_iter(
        n_full,
        n_k_tiles,
        b.const_i32(1),
        iter_args=[
            (f"{name}_t", v) for (name, _), v in zip(iter_args, main_loop.results)
        ],
        iv_name="k_tail",
    )
    with tail_loop as (k_tile, state):
        _k_tile_body(k_tile, state, masked=True)
    k_loop = tail_loop

    # ---- epilogue: normalize, then project the latent out with W_UV -------
    res = k_loop.results
    ms = list(res[:M_TILES])
    ls = list(res[M_TILES : 2 * M_TILES])
    accs = [
        list(res[2 * n_ml + mt * N_R_PER_WAVE : 2 * n_ml + (mt + 1) * N_R_PER_WAVE])
        for mt in range(M_TILES)
    ]
    inv = [_safe_inv_l(b, ls[mt]) for mt in range(M_TILES)]

    # Epilogue buffers: allocated after the loop so they pool onto the loop's
    # slots. That aliasing is why the barrier below is mandatory -- the loop's
    # last readers have to retire before accl_lds is written.
    # wt_lds first, for the same packer reason as s_part/kv_lds: it takes the
    # base slot and accl_lds the dead qa_lds slot.
    wt_lds = b.smem_alloc(dtype, [D_V, R_TILE + WT_PAD], name_hint="wt_lds")  # 9216 B
    accl_lds = b.smem_alloc(dtype, [BQ, R_KV + WT_PAD], name_hint="accl_lds")  # 16512 B

    b.sync()
    # accᵀ puts one query and four consecutive latent columns in each lane, so
    # each fragment goes back to accl_lds ([query][r]) as a single 8 B store.
    for t in range(N_R_PER_WAVE):
        n_tile = b.add(b.mul(wave, b.const_i32(N_R_PER_WAVE)), b.const_i32(t))
        col = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_kq)
        for mt in range(M_TILES):
            row = _shift(b, mt * MFMA_M, lane_row)
            b.smem_store_vN(
                accl_lds, [row, col], b.vec_trunc_f32_to_bf16(accs[mt][t]), C_PER_LANE
            )

    # out_raw[q, v] = sum_r acc_latent[q, r] * W_UV[r, v], W_UV = w_uk[.., D_NOPE:].
    # 1/l is linear and commutes with this projection, so it is applied after.
    # ``_expand_latent`` opens each slice with its own barrier, which fences both
    # the accl_lds stores above and the WAR on wt_lds.
    out_acc = _expand_latent(
        b,
        w_uk=w_uk,
        c_lds=accl_lds,
        wt_lds=wt_lds,
        tid=tid,
        wave=wave,
        lane_row=lane_row,
        lane_kq=lane_kq,
        w_head_base=w_head_base,
        col_offset=D_NOPE,
        dtype=dtype,
        threads=THREADS,
        n_slices=N_SLICES,
        r_tile=R_TILE,
        d_out=D_V,
        w_cols=W_COLS,
        exp_k_iters=EXP_K_ITERS,
        n_tiles_per_wave=N_V_PER_WAVE,
        m_tiles=M_TILES,
        out_transposed=True,
    )

    # outᵀ: lane holds query ``lane % 16`` and four consecutive ``v``, so it
    # normalizes by its own 1/l and writes them as one 8 B store.
    for t in range(N_V_PER_WAVE):
        n_tile = b.add(b.mul(wave, b.const_i32(N_V_PER_WAVE)), b.const_i32(t))
        col = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_kq)
        for mt in range(M_TILES):
            q_local = b.add(qb_start, _shift(b, mt * MFMA_M, lane_row))
            vals = b.vec_pack(
                [
                    b.fmul(b.vec_extract(out_acc[mt][t], r), inv[mt])
                    for r in range(C_PER_LANE)
                ],
                F32,
            )
            with b.scf_if(b.cmp_lt(q_local, s_q)):
                token = b.add(cu_q_start, q_local)
                idx = b.add(b.mul(token, b.const_i32(H * D_V)), col)
                idx = b.add(idx, b.mul(head, b.const_i32(D_V)))
                b.global_store_vN(
                    out, idx, b.vec_trunc_f32_to_bf16(vals), C_PER_LANE, align=8
                )

    # One writer per query: wave 0's first 16-lane group holds every query of
    # each M tile once.
    with b.scf_if(b.cmp_eq(wave, b.const_i32(0))):
        with b.scf_if(b.cmp_lt(lane, b.const_i32(16))):
            for mt in range(M_TILES):
                q_local = b.add(qb_start, _shift(b, mt * MFMA_M, lane_row))
                value = b.fmul(b.fadd(ms[mt], b.log2(ls[mt])), b.const_f32(LN2))
                with b.scf_if(b.cmp_lt(q_local, s_q)):
                    token = b.add(cu_q_start, q_local)
                    idx = b.add(b.mul(token, b.const_i32(H)), head)
                    b.global_store(lse, idx, value, align=4)

    return b.kernel
