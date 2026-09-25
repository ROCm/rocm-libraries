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
        ct_lds           <- c_kv[page]^T      [r_KV, Bk]   same loads, scattered
        kv_lds[:, r_kv:] <- k_rope[page]      [Bk, 64]     head-shared RoPE
        S       <- scale * (qa_lds @ kv_lds^T)             one K=576 GEMM
        P       <- online_softmax(mask(S))
        acc     += P @ ct_lds^T                            latent-space PV
    accl_lds <- acc                           [Bq, r_KV]
    for r_slice in range(r_KV / r_KV_tile):                  -- V absorb
        wt_lds <- transpose(W_UK[head, r_slice, d_nope:])  [d_v, r_tile]
        out    += accl_lds[:, r_slice] @ wt_lds
    out      <- out * (1/l)                   [Bq, d_v]

gfx942 constraints that shape the emission:

  * **Narrow MFMA only.** CDNA3 has no wide-K ``mfma_f32_16x16x32`` bf16 atom,
    so every GEMM uses ``mfma_f32_16x16x16_bf16`` with a K-step of 16.
  * **No transpose read.** ``ds_read_*_tr_*`` is gfx950-only, so any B operand
    that is not already stored as ``[n][k]`` must be transposed on the
    **store** path: read from global along its contiguous axis with vector
    loads and scattered into LDS one element at a time. That is why ``ct_lds``
    exists alongside ``kv_lds`` -- the score GEMM wants ``C`` as ``[key][r]``
    and the PV GEMM wants it as ``[r][key]``, from the same global loads. The
    ``W_UK`` V-half gets the same treatment in the epilogue. The Q-absorb
    B operand needs **no** transpose: ``W_UK[head, r, :d_nope]`` is already
    stored as ``[r][nope]``, which is exactly ``[n][k]`` for that GEMM.
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
from rocke.helpers.distribution import (
    TileDistributionEncoding,
    block_tile_reduce_sync,
    make_static_distributed_tensor,
    make_static_tile_distribution,
)
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
# feed, G steps deep. G must divide ``SCORE_K_ITERS`` evenly -- a remainder
# leaves the tail steps unhinted and measured worse than any dividing value.
# G = 12 (three blocks of 36) was the best of the divisors swept; it is a
# scheduling hint only, so a wrong value costs speed and never correctness.
#
# This is mutually exclusive with ``iglp_opt``, which owns the whole-loop
# schedule. Both of its canned patterns were measured here and both regressed.
SCORE_SGB_GROUP = 12

_C16_DIST = make_static_tile_distribution(
    make_c_warp_dstr_encoding(MfmaAtom.bf16_16x16x16())
)

# Cross-lane row reduce matching the 16x16x16 MFMA C layout: within a 16-lane
# group every lane holds a different column of the same row, so reducing a row
# is an XOR butterfly over masks 1/2/4/8 (pure register, no LDS, no barrier).
# ``rocke.helpers.mfma_attention`` has the same encoding but keeps it private,
# and ``library`` must not import private ``platform`` symbols, so it is
# re-declared here from the public ``helpers.distribution`` primitives.
_ROW_REDUCE_ENC = TileDistributionEncoding(
    Rs=(16,),
    Hs=((1,),),
    Ps2RHs_major=((0,),),  # the single (lane) P feeds R (major 0)
    Ps2RHs_minor=((0,),),
    Ys2RHs_major=(1,),  # one keep-row Y on the M H-dim (length 1)
    Ys2RHs_minor=(0,),
)
_ROW_REDUCE_DIST = make_static_tile_distribution(_ROW_REDUCE_ENC)


def _row_reduce(b: IRBuilder, scalar: Value, *, combine: str) -> Value:
    """Reduce ``scalar`` across the 16 lanes sharing an MFMA C row."""
    dt = make_static_distributed_tensor(_ROW_REDUCE_DIST, F32)
    dt.storage[0] = scalar
    block_tile_reduce_sync(b, dt, combine=combine)
    return dt.storage[0]


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


def _ct_swizzle(b: IRBuilder, row: Value, col: Value) -> Value:
    """Permute ``ct_lds`` 4-column groups by ``(row >> 4) & 3``.

    ``ct_lds`` is the transposed latent staging buffer, written one bf16 at a
    time and read as the MFMA B operand. The write is what conflicts. Within a
    wave the store's column is wave-uniform and its row is ``lane * 8 + e``, so
    with a row stride of ``s`` dwords every lane lands on bank ``lane * 8 * s
    (mod 32)``. ``gcd(8s, 32) >= 16`` for every even ``s``, so no pad can spread
    the store past two banks -- the ``BK + WT_PAD`` pad already took it from one
    bank to two and that is the end of what padding can do. Breaking further
    needs the *column* to move with the row, which is this.

    The permutation is an XOR on the column-group index, so it is an involution
    and the same expression serves the store and the load. It is applied to
    groups of four columns rather than single columns because the B-operand read
    is ``smem_load_vN(..., n=4)`` at ``lane_kq in {0, 4, 8, 12}``: four
    contiguous columns off a 4-aligned base. Permuting whole groups keeps those
    four contiguous and keeps the base a multiple of four, so the vector read
    stays a single ``ds_read_b64`` and its declared 8 B alignment still holds.

    The shift is 4, not 3, and the distinction is the whole point. ``row >> 3``
    is the store's lane index exactly (``r0 = lane * 8`` and ``e < 8`` never
    carries), but its low bit is the *same* bit the row stride already spreads
    on: at ``s = 10`` the stride term is ``lane * 80 = lane * 16 (mod 32)``,
    which keys off ``lane & 1``. Twisting on a bit the stride has already used
    re-collides the halves and buys one extra bank, not three. ``row >> 4``
    takes lane bits 1-2 instead, disjoint from the stride's bit, so the two
    spreadings compose: 2 banks from the pad x 4 group offsets = 8 distinct
    banks per wave.

    The 2-bit mask is safe for every supported ``block_k``: MFMA needs a
    multiple of 16, so there are always at least four column groups to permute.
    """
    twist = b.land(b.lshr(row, b.const_i32(4)), b.const_i32(3))
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
    r_kv_tile: int = 64
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

    The lowerer pools the eight buffers by liveness, so the pool is *not* their
    arithmetic sum -- see the LDS comment block in the builder for the achieved
    layout. The packer is greedy in allocation order, which for this builder
    plays out as: the prologue trio (``q_lds``, ``wq_lds``, ``qa_lds``) is laid
    down first; ``q_lds`` + ``wq_lds`` then die and leave a hole at the base,
    into which the packer folds *one* loop buffer -- ``kv_lds``, the first one
    allocated -- after which everything else stacks above ``qa_lds``, which stays
    live across the loop. The epilogue pair starts from the base again, since by
    then nothing else is live, so it only matters when it is the widest phase.

    Exact against the emitted pool at the default geometry, at ``block_q=32``,
    and at ``r_kv_tile=128``; over by ``p_lds`` at ``block_k=32``, where the
    packer finds one more hole than this credits it with. Erring high is the
    point -- this gates admission, so it must never under-predict. It is still a
    model of observed behaviour rather than a guarantee, so check the emitted
    pool whenever a new geometry is introduced.
    """
    elem = 2  # bf16
    qa_cols = spec.r_kv + spec.d_rope
    q_lds = elem * spec.block_q * spec.head_dim_qk
    wq_lds = elem * spec.r_kv_tile * (spec.d_nope + V8_PAD)
    qa_lds = elem * spec.block_q * (qa_cols + WT_PAD)
    loop = (
        elem * spec.block_k * (qa_cols + WT_PAD),  # kv_lds
        elem * spec.r_kv * (spec.block_k + WT_PAD),  # ct_lds
        elem * spec.block_q * spec.block_k,  # p_lds
    )
    epilogue = elem * spec.block_q * (spec.r_kv + WT_PAD) + elem * spec.d_v * (
        spec.r_kv_tile + WT_PAD
    )

    hole = q_lds + wq_lds
    top = hole + qa_lds
    folded = loop[0] <= hole
    for size in loop[1:] if folded else loop:
        top += size
    # By the epilogue every earlier buffer is dead, so the pair starts again from
    # the base and only matters when it is wider than everything before it.
    return max(top, epilogue)


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
    warp_splits = {
        "r_kv (PV latent n-tiles)": spec.r_kv,
        "r_kv_tile (query-absorb n-tiles)": spec.r_kv_tile,
        "d_v (epilogue n-tiles)": spec.d_v,
    }
    for name, value in warp_splits.items():
        n_tiles = value // MFMA_N
        if n_tiles % spec.num_warps != 0:
            return False, (
                f"{name}: {value}/{MFMA_N} = {n_tiles} n-tiles must divide "
                f"evenly across num_warps={spec.num_warps}"
            )
    # Every global->LDS staging loop below hands each thread a fixed number of
    # fixed-width vector chunks with no remainder handling, so the element
    # count must divide by threads * width -- not merely by threads.
    threads = spec.threads
    stages = {
        "q tile": (spec.block_q * spec.head_dim_qk, 4),
        "Q_rope copy": (spec.block_q * spec.d_rope, 4),
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
    # with block_q and ct_lds with block_k, so the 64 KB budget -- not the
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
                b.smem_store_vN(
                    wt_lds, [b.add(n0, b.const_i32(e)), rl], b.vec_extract(v8, e), 1
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
                b_v = b.smem_load_vN(wt_lds, b_row, b_col, dtype=dtype, n=4)
                for mt in range(m_tiles):
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
        acc = [
            [b.zero_vec_f32(4) for _ in range(n_tiles_per_wave)] for _ in range(m_tiles)
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
            n_tile = b.add(b.mul(wave, b.const_i32(n_tiles_per_wave)), b.const_i32(t))
            col = _shift(b, r_base, b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row))
            for mt in range(m_tiles):
                packed = b.vec_trunc_f32_to_bf16(acc[mt][t])
                for reg in range(C_PER_LANE):
                    row = _shift(b, mt * MFMA_M, _mfma_16x16_c_row(b, lane, reg))
                    b.smem_store_vN(qa_lds, [row, col], b.vec_extract(packed, reg), 1)


def build_mla_prefill_fwd(spec: MlaPrefillSpec, *, arch: str = "gfx942") -> KernelDef:
    """Emit the chunked MLA prefill forward kernel, with ``W_UK`` absorbed.

    One workgroup owns one ``(query tile, head)`` pair and loops over the whole
    key range, carrying ``(m, l, acc_latent)`` in ``scf.for`` iteration
    arguments. The per-head up-projection never enters that loop: it is folded
    into the query on the way in and into the output on the way out, so the
    k-loop only ever touches the compressed latent.

    Prologue, once per workgroup:

    1. stage ``Q`` into ``q_lds`` and copy its rope half to ``qa_lds`` columns
       ``[r_kv, r_kv + d_rope)``;
    2. absorb -- ``Qa[q, r] = sum_n Q_nope[q, n] * W_UK[head, r, n]`` -- into
       ``qa_lds`` columns ``[0, r_kv)``, staging ``W_UK`` r-slices through
       ``wq_lds``. ``S[q, k] = sum_r C[k, r] * Qa[q, r]`` then reproduces the
       expanded score exactly.

    Per key tile:

    3. stage ``c_kv[page]`` into ``kv_lds`` columns ``[0, r_kv)`` **and**, from
       the same global loads, transposed into ``ct_lds``; stage ``k_rope[page]``
       into ``kv_lds`` columns ``[r_kv, r_kv + d_rope)``;
    4. score ``S = scale * (Qa @ [C | K_rope]ᵀ)`` -- one ``K = r_kv + d_rope``
       reduction -- then mask and online-softmax update;
    5. accumulate ``acc_latent[q, r] += sum_k P[q, k] * C[k, r]``.

    Epilogue, once per workgroup:

    6. ``out[q, v] = (sum_r acc_latent[q, r] * W_UV[r, v]) * (1/l[q])``, where
       ``W_UV = w_uk[head, :, d_nope:]``. The rescale and ``1/l`` are linear and
       commute with the projection, so applying ``1/l`` to the ``[q][v]`` result
       is exact.

    ``ct_lds`` holds **Cᵀ** (``[r_kv, Bk]``) because the MFMA B operand is read
    ``[n][k]``: for the latent PV GEMM ``n`` is the latent dim and ``k`` is the
    key. gfx942 has no ``ds_read_tr``, so the transpose happens on the store
    path -- the c_kv loads are scattered into ``ct_lds`` element-wise while the
    same vectors go into ``kv_lds`` untransposed for the score GEMM. Computing
    ``Sᵀ`` instead would not avoid this: the softmax row-reduce would then run
    over ``q``, which is the wrong axis.

    Every wave redundantly computes the score tile and the softmax state. That
    is deliberate: ``scf.if`` carries no results, so loop-carried state cannot be
    produced inside a wave-gated region, and a barrier inside one would hang.
    The state is bit-identical in every wave, so only the latent accumulator
    tiles are split across waves and only the final stores are predicated.
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
    ABSORB_N_PER_WAVE = ABSORB_N_TILES // spec.num_warps
    ABSORB_K_ITERS = D_NOPE // MFMA_K

    LOG2E = math.log2(math.e)
    LN2 = math.log(2.0)

    b = IRBuilder(spec.fwd_kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

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
    q_block_global_idx = b.block_id_x()
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
    n_k_tiles = b.div(b.add(s_k, b.const_i32(BK - 1)), b.const_i32(BK))
    # Bottom-right causal alignment: query row ``i`` of this sequence attends to
    # keys ``0 .. i + context_off``. ``context_off`` is negative when S_k < S_q,
    # which is exactly what produces the fully masked leading rows the driver
    # checks for exact zeros / exact -inf.
    context_off = b.sub(s_k, s_q)

    # ---- LDS ---------------------------------------------------------------
    # No allocation is exclusive, so the lowerer pools them by liveness. The
    # three phases below never overlap, and the loop phase is the peak:
    #
    #   prologue  q_lds  + wq_lds + qa_lds            (BQ=BK=16:  42112 B)
    #   loop      qa_lds + kv_lds + ct_lds + p_lds    (BQ=BK=16:  58112 B)
    #   epilogue  accl_lds + wt_lds                   (BQ=BK=16:  33920 B)
    #
    # The pool actually emitted is 63104 B, ~5.0 KB above that loop-phase ideal.
    # The packer is greedy first-fit in allocation order, so qa_lds is placed
    # above wq_lds (at 23552) while still live across the loop; kv_lds then
    # reuses the q_lds/wq_lds bytes from 0 but only reaches 18560, leaving
    # [18560, 23552) stranded for the whole loop. Shrinking kv_lds therefore
    # does not shrink the pool -- it only widens that stranded hole. Achieved
    # layout:
    #
    #   0      q_lds / kv_lds / accl_lds   (three phases share the base)
    #   6144   wq_lds
    #   23552  qa_lds / wt_lds
    #   42112  ct_lds
    #   62592  p_lds                        -> 63104 B total
    #
    # **Each buffer is allocated at the point its phase begins, not all up
    # front.** The lowerer seeds a live interval at the ``tile.smem_alloc`` op
    # itself, not at first use, so hoisting every allocation to the top of the
    # kernel would start all eight intervals before anything runs and defeat
    # pooling entirely -- the pool becomes the arithmetic sum (111616 B here,
    # well over the 64 KB limit). Keep each ``smem_alloc`` next to the code that
    # first touches it.
    #
    # Pooling means a later buffer may land on an earlier one's bytes, so every
    # phase boundary needs a barrier before its first write: the k-loop body
    # opens with one (fencing the prologue's q_lds/wq_lds readers against the
    # kv_lds/ct_lds writes) and the epilogue opens with one of its own.
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
    # ``ct_lds`` takes ``WT_PAD`` for the same reason, and the arithmetic is
    # worth spelling out because the two accesses want different things. Write
    # ``s`` for the row stride in dwords; unpadded ``s == BK/2 == 8``, padded
    # ``s == (BK + 4)/2 == 10``.
    #
    #   scatter store (one ds_write_b16 per lane, row ``lane*8 + e``, column
    #   ``m`` constant across the wave):
    #       addr_dw = (lane*8 + e)*s + m/2
    #       s =  8 -> lane*64 == 0 (mod 32): every lane on ONE bank.
    #       s = 10 -> lane*80 == lane*16 (mod 32): two banks.
    #
    #   MFMA B read (ds_read_b64 at [n_tile*16 + lane%16, kt*16 + (lane/16)*4]):
    #       addr_dw = (lane%16)*s + (lane/16)*2 + const
    #       s =  8 -> (lane%16)*8 mod 32 takes 4 values: 4-way.
    #       s = 10 -> (lane%16)*10 mod 32 takes all 16: conflict-free.
    #
    # So the pad is a 4x win on the read and only 2x on the store. The store
    # cannot do better with any legal pad: its lane stride is 8 rows, the bank
    # period is 32/gcd(8s, 32), and gcd(8s, 32) >= 16 for every even ``s``.
    # ``s`` odd would need a pad of 2 mod 4, which puts odd rows at 4 mod 8 and
    # makes the align-8 declaration on the n=4 read a lie. Driving the store
    # further needs an XOR swizzle on the column, not a wider pad.
    #
    # Pads are a layout choice only; every index expression keeps its logical
    # range, so the kernel computes the same values either way.
    q_lds = b.smem_alloc(dtype, [BQ, HDQK], name_hint="q_lds")  # 6144 B
    wq_lds = b.smem_alloc(
        dtype, [R_TILE, D_NOPE + V8_PAD], name_hint="wq_lds"
    )  # 17408 B
    qa_lds = b.smem_alloc(dtype, [BQ, QA_COLS + WT_PAD], name_hint="qa_lds")  # 18560 B

    # ---- stage Q once (loop-invariant) -------------------------------------
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

    # The Q_rope copy below reads ``q_lds``, and ``_absorb_query`` reads it as an
    # MFMA A operand -- both need the staging stores visible workgroup-wide.
    b.sync()

    # Q_rope rides along in the absorbed query at columns [R_KV, R_KV + D_ROPE),
    # so the score GEMM is one K = R_KV + D_ROPE reduction instead of two.
    qr_chunks = (BQ * D_ROPE) // (THREADS * 4)
    qr_chunks_per_row = D_ROPE // 4
    for j in range(qr_chunks):
        c = b.add(tid, b.const_i32(j * THREADS))
        m = b.div(c, b.const_i32(qr_chunks_per_row))
        d0 = b.mul(b.mod(c, b.const_i32(qr_chunks_per_row)), b.const_i32(4))
        v = b.smem_load_vN(q_lds, m, b.add(d0, b.const_i32(D_NOPE)), dtype=dtype, n=4)
        b.smem_store_vN(qa_lds, [m, b.add(d0, b.const_i32(R_KV))], v, 4)

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
        m_tiles=M_TILES,
    )
    # No trailing barrier: the k-loop opens with one, which fences qa_lds too.

    # Fold log2(e) here, next to the exp2 that consumes it: the ABI stays raw.
    scale_log2 = b.fmul(scale_p, b.const_f32(LOG2E))
    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    iter_args = [
        (f"m{mt}_{r}", neg_inf) for mt in range(M_TILES) for r in range(C_PER_LANE)
    ]
    iter_args += [
        (f"l{mt}_{r}", zero_f) for mt in range(M_TILES) for r in range(C_PER_LANE)
    ]
    iter_args += [
        (f"acc{mt}_{t}", b.zero_vec_f32(4))
        for mt in range(M_TILES)
        for t in range(N_R_PER_WAVE)
    ]
    n_ml = M_TILES * C_PER_LANE

    # Loop-phase buffers: allocated here so they pool onto q_lds / wq_lds, whose
    # last readers are the absorb above and are fenced by the loop's opening
    # barrier.
    kv_lds = b.smem_alloc(dtype, [BK, QA_COLS + WT_PAD], name_hint="kv_lds")  # 18560 B
    ct_lds = b.smem_alloc(dtype, [R_KV, BK + WT_PAD], name_hint="ct_lds")  # 20480 B
    p_lds = b.smem_alloc(dtype, [BQ, BK], name_hint="p_lds")  # 512 B

    # ---- k-loop staging, split so the loads can run ahead ------------------
    # Splitting the stage into a load half and a store half lets tile ``i+1``'s
    # global loads issue before tile ``i``'s compute and retire under it,
    # instead of sitting exposed between the two barriers. One stage is in
    # flight: ``c_chunks`` vectors of ``C_STAGE_W`` plus ``kr_chunks`` of 4 --
    # 18 VGPR at BK=16, against 160 of 256 in use.
    #
    # The equivalent pipeline in ``attention_tiled_2d`` hides the same latency
    # with a two-slot LDS double buffer, so its staged registers never cross an
    # iteration boundary. That is not available here: the pool is already at
    # 63104 of 65536 B and a second kv_lds/ct_lds slot needs ~38 KB more. The
    # stage is therefore carried in ``iter_args``. Registers are the only
    # resource this kernel still has spare, which is what makes the swap work.
    c_chunks = (BK * R_KV) // (THREADS * C_STAGE_W)
    c_chunks_per_row = R_KV // C_STAGE_W
    kr_chunks = (BK * D_ROPE) // (THREADS * 4)
    kr_chunks_per_row = D_ROPE // 4

    def _stage_loads(tile, *, guard=None):
        """Issue one tile's global loads into registers. Touches no LDS."""
        page = b.global_load_i32(block_table, b.add(b.mul(seq_idx, bt_stride_p), tile))
        if guard is not None:
            # Zero-trip loop: the block-table slot may be uninitialised. Page 0
            # is always mapped, so the load stays in bounds; a zero-trip loop
            # never consumes the value.
            page = b.select(guard, page, b.const_i32(0))
        c_base = b.mul(page, b.const_i32(PAGE * R_KV))
        kr_base = b.mul(page, b.const_i32(PAGE * D_ROPE))
        staged = []
        for j in range(c_chunks):
            c = b.add(tid, b.const_i32(j * THREADS))
            m = b.div(c, b.const_i32(c_chunks_per_row))
            r0 = b.mul(b.mod(c, b.const_i32(c_chunks_per_row)), b.const_i32(C_STAGE_W))
            idx = b.add(b.add(c_base, b.mul(m, b.const_i32(R_KV))), r0)
            staged.append(b.global_load_vN(c_kv, idx, dtype, C_STAGE_W))
        for j in range(kr_chunks):
            c = b.add(tid, b.const_i32(j * THREADS))
            m = b.div(c, b.const_i32(kr_chunks_per_row))
            d0 = b.mul(b.mod(c, b.const_i32(kr_chunks_per_row)), b.const_i32(4))
            idx = b.add(b.add(kr_base, b.mul(m, b.const_i32(D_ROPE))), d0)
            staged.append(b.global_load_vN(k_rope, idx, dtype, 4))
        return staged

    def _stage_store(staged):
        """Drain a staged tile into both LDS orientations.

        ``kv_lds`` is the B operand of the score GEMM, read as [key][r] -- the
        natural layout. ``ct_lds`` is the B operand of the latent PV, read as
        [r][key] -- the transpose. gfx942 has no ``ds_read_tr``, so the
        transpose happens on the store path; both copies come off the same
        staged register.
        """
        for j in range(c_chunks):
            c = b.add(tid, b.const_i32(j * THREADS))
            m = b.div(c, b.const_i32(c_chunks_per_row))
            r0 = b.mul(b.mod(c, b.const_i32(c_chunks_per_row)), b.const_i32(C_STAGE_W))
            v8 = staged[j]
            b.smem_store_vN(kv_lds, [m, r0], v8, C_STAGE_W)
            for e in range(C_STAGE_W):
                ct_row = _shift(b, e, r0)
                b.smem_store_vN(
                    ct_lds,
                    [ct_row, _ct_swizzle(b, ct_row, m)],
                    b.vec_extract(v8, e),
                    1,
                )
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

    k_loop = b.scf_for_iter(
        b.const_i32(0), n_k_tiles, b.const_i32(1), iter_args=iter_args, iv_name="k_tile"
    )
    with k_loop as (k_tile, state):
        ms = [
            list(state[mt * C_PER_LANE : (mt + 1) * C_PER_LANE])
            for mt in range(M_TILES)
        ]
        ls = [
            list(state[n_ml + mt * C_PER_LANE : n_ml + (mt + 1) * C_PER_LANE])
            for mt in range(M_TILES)
        ]
        accs = [
            list(
                state[2 * n_ml + mt * N_R_PER_WAVE : 2 * n_ml + (mt + 1) * N_R_PER_WAVE]
            )
            for mt in range(M_TILES)
        ]
        staged = list(state[-n_stage:])
        kb_start = b.mul(k_tile, b.const_i32(BK))

        # WAR: the previous iteration's readers of kv_lds / ct_lds must retire
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
        s_acc = [[b.zero_vec_f32(4) for _ in range(K_TILES)] for _ in range(M_TILES)]
        for kk in range(SCORE_K_ITERS):
            col = b.add(b.const_i32(kk * MFMA_K), lane_kq)
            a_vs = [
                b.smem_load_vN(
                    qa_lds, _shift(b, mt * MFMA_M, lane_row), col, dtype=dtype, n=4
                )
                for mt in range(M_TILES)
            ]
            for kt in range(K_TILES):
                b_v = b.smem_load_vN(
                    kv_lds, _shift(b, kt * MFMA_N, lane_row), col, dtype=dtype, n=4
                )
                for mt in range(M_TILES):
                    s_acc[mt][kt] = _mfma_16x16x16(
                        b, dtype, a_vs[mt], b_v, s_acc[mt][kt]
                    )
            # Hoist the next ``SCORE_SGB_GROUP`` steps' LDS reads above the
            # MFMAs they feed. Skipped when the group does not divide the trip
            # count -- a partial trailing block measured worse than no hint.
            if SCORE_K_ITERS % SCORE_SGB_GROUP == 0 and (kk + 1) % SCORE_SGB_GROUP == 0:
                b.sched_group_barrier(
                    _SGB_DS_READ, SCORE_SGB_GROUP * (M_TILES + K_TILES), 0
                )
                b.sched_group_barrier(_SGB_MFMA, SCORE_SGB_GROUP * M_TILES * K_TILES, 0)

        # ---- online softmax, one row slot at a time ------------------------
        # Two bounds, and both land on ``p`` rather than only on ``s``: the
        # ragged key bound (the last tile overhangs S_k, where page-granular
        # staging supplies real but meaningless bytes) and the bottom-right
        # causal bound. Masking ``s`` alone is not enough -- with the -1e30
        # sentinel a fully masked row gets m_new == -1e30, so every element
        # would yield exp2(s - m_new) == 1 and ``l`` would come out BK, not 0.
        # l == 0 is what makes ``safe_inv_l`` store exact zeros and the LSE
        # collapse to a true -inf.
        k_abs = [
            b.add(kb_start, _shift(b, kt * MFMA_N, lane_row)) for kt in range(K_TILES)
        ]
        in_k = [b.cmp_lt(k_abs[kt], s_k) for kt in range(K_TILES)]
        new_ms = []
        new_ls = []
        p_vecs = [[b.zero_vec_f32(4) for _ in range(K_TILES)] for _ in range(M_TILES)]
        for mt in range(M_TILES):
            for r in range(C_PER_LANE):
                q_abs = b.add(
                    qb_start, _shift(b, mt * MFMA_M, _mfma_16x16_c_row(b, lane, r))
                )
                causal_hi = b.add(q_abs, context_off)
                keep = []
                s_r = []
                for kt in range(K_TILES):
                    keep.append(b.land(in_k[kt], b.cmp_le(k_abs[kt], causal_hi)))
                    s_r.append(
                        b.select(
                            keep[kt],
                            b.fmul(b.vec_extract(s_acc[mt][kt], r), scale_log2),
                            neg_inf,
                        )
                    )
                local_max = s_r[0]
                for kt in range(1, K_TILES):
                    local_max = b.fmax(local_max, s_r[kt])
                row_max = _row_reduce(b, local_max, combine="max")
                m_new = b.fmax(ms[mt][r], row_max)
                alpha = b.exp2(b.fsub(ms[mt][r], m_new))
                local_sum = None
                for kt in range(K_TILES):
                    p_r = b.select(keep[kt], b.exp2(b.fsub(s_r[kt], m_new)), zero_f)
                    p_vecs[mt][kt] = b.vec_insert(p_vecs[mt][kt], p_r, r)
                    local_sum = p_r if local_sum is None else b.fadd(local_sum, p_r)
                row_sum = _row_reduce(b, local_sum, combine="sum")
                new_ms.append(m_new)
                new_ls.append(b.fadd(b.fmul(ls[mt][r], alpha), row_sum))
                for t in range(N_R_PER_WAVE):
                    old = b.vec_extract(accs[mt][t], r)
                    accs[mt][t] = b.vec_insert(accs[mt][t], b.fmul(old, alpha), r)

        # ---- publish P and accumulate acc_latent += P @ C ------------------
        # Every wave writes the same values; the barrier pair is what matters.
        b.sync()
        for mt in range(M_TILES):
            for kt in range(K_TILES):
                packed = b.vec_trunc_f32_to_bf16(p_vecs[mt][kt])
                col = _shift(b, kt * MFMA_N, lane_row)
                for r in range(C_PER_LANE):
                    row = _shift(b, mt * MFMA_M, _mfma_16x16_c_row(b, lane, r))
                    b.smem_store_vN(p_lds, [row, col], b.vec_extract(packed, r), 1)
        b.sync()

        for kt in range(K_TILES):
            a_col = _shift(b, kt * MFMA_N, lane_kq)
            a_vs = [
                b.smem_load_vN(
                    p_lds, _shift(b, mt * MFMA_M, lane_row), a_col, dtype=dtype, n=4
                )
                for mt in range(M_TILES)
            ]
            for t in range(N_R_PER_WAVE):
                n_tile = b.add(b.mul(wave, b.const_i32(N_R_PER_WAVE)), b.const_i32(t))
                b_row = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row)
                b_v = b.smem_load_vN(
                    ct_lds, b_row, _ct_swizzle(b, b_row, a_col), dtype=dtype, n=4
                )
                for mt in range(M_TILES):
                    accs[mt][t] = _mfma_16x16x16(b, dtype, a_vs[mt], b_v, accs[mt][t])

        flat_acc = [accs[mt][t] for mt in range(M_TILES) for t in range(N_R_PER_WAVE)]
        b.scf_yield(*new_ms, *new_ls, *flat_acc, *staged_next)

    # ---- epilogue: normalize, then project the latent out with W_UV -------
    res = k_loop.results
    ms = [list(res[mt * C_PER_LANE : (mt + 1) * C_PER_LANE]) for mt in range(M_TILES)]
    ls = [
        list(res[n_ml + mt * C_PER_LANE : n_ml + (mt + 1) * C_PER_LANE])
        for mt in range(M_TILES)
    ]
    accs = [
        list(res[2 * n_ml + mt * N_R_PER_WAVE : 2 * n_ml + (mt + 1) * N_R_PER_WAVE])
        for mt in range(M_TILES)
    ]
    inv = [
        [_safe_inv_l(b, ls[mt][r]) for r in range(C_PER_LANE)] for mt in range(M_TILES)
    ]

    # Epilogue buffers: allocated after the loop so they pool onto the loop's
    # slots. That aliasing is why the barrier below is mandatory -- the loop's
    # last readers have to retire before accl_lds is written.
    accl_lds = b.smem_alloc(dtype, [BQ, R_KV + WT_PAD], name_hint="accl_lds")  # 16512 B
    wt_lds = b.smem_alloc(dtype, [D_V, R_TILE + WT_PAD], name_hint="wt_lds")  # 17408 B

    b.sync()
    for t in range(N_R_PER_WAVE):
        n_tile = b.add(b.mul(wave, b.const_i32(N_R_PER_WAVE)), b.const_i32(t))
        col = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row)
        for mt in range(M_TILES):
            packed = b.vec_trunc_f32_to_bf16(accs[mt][t])
            for r in range(C_PER_LANE):
                row = _shift(b, mt * MFMA_M, _mfma_16x16_c_row(b, lane, r))
                b.smem_store_vN(accl_lds, [row, col], b.vec_extract(packed, r), 1)

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
    )

    for t in range(N_V_PER_WAVE):
        n_tile = b.add(b.mul(wave, b.const_i32(N_V_PER_WAVE)), b.const_i32(t))
        col = b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_row)
        for mt in range(M_TILES):
            for r in range(C_PER_LANE):
                row = _shift(b, mt * MFMA_M, _mfma_16x16_c_row(b, lane, r))
                q_local = b.add(qb_start, row)
                value = b.fmul(b.vec_extract(out_acc[mt][t], r), inv[mt][r])
                with b.scf_if(b.cmp_lt(q_local, s_q)):
                    token = b.add(cu_q_start, q_local)
                    idx = b.add(b.mul(token, b.const_i32(H * D_V)), col)
                    idx = b.add(idx, b.mul(head, b.const_i32(D_V)))
                    b.global_store(out, idx, b.trunc_f32_to_bf16(value), align=2)

    # One writer per row: lane%16 == 0 picks 4 lanes with distinct m_blk, and
    # each contributes 4 register rows -- exactly the 16 rows of a tile.
    with b.scf_if(b.cmp_eq(wave, b.const_i32(0))):
        with b.scf_if(b.cmp_eq(lane_row, b.const_i32(0))):
            for mt in range(M_TILES):
                for r in range(C_PER_LANE):
                    row = _shift(b, mt * MFMA_M, _mfma_16x16_c_row(b, lane, r))
                    q_local = b.add(qb_start, row)
                    value = b.fmul(
                        b.fadd(ms[mt][r], b.log2(ls[mt][r])), b.const_f32(LN2)
                    )
                    with b.scf_if(b.cmp_lt(q_local, s_q)):
                        token = b.add(cu_q_start, q_local)
                        idx = b.add(b.mul(token, b.const_i32(H)), head)
                        b.global_store(lse, idx, value, align=4)

    return b.kernel
