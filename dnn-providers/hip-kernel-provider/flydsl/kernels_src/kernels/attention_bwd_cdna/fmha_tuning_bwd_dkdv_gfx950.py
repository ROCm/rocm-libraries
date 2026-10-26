# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Tuning policy for the gfx950 dK/dV backward kernel: which knobs, for which shape.

The backward twin of `fmha_tuning_gfx950.py`, and split from the kernel for the
same reason: a number here moves when a sweep says so, and nothing here can
make a build *wrong*, only slow.

--- The role swap, stated once ---------------------------------------------

**dK/dV is the forward loop transposed.** K and V stay resident in registers
and Q and dO stream past through LDS, where the forward keeps Q resident and
streams K/V. The dualwave traits are written against the forward's roles, and
this module reuses them rather than forking them, so three of their names mean
the other thing here:

| traits field | forward | dK/dV |
|---|---|---|
| `BLOCK_M` | Q rows per workgroup | **KV rows per workgroup** |
| `BLOCK_N` | KV rows per streamed tile | **Q rows per streamed tile** |
| `ROWS_PER_WAVE` | Q rows per wave | **KV rows per wave** (`MFMA_ROWS`: 32, or 16 in the 16-row family) |
| `VO_SHARDS` | waves splitting O's D axis | **waves splitting dK/dV's D axis** |

`BwdDkDvTraits` re-exposes them under `BLOCK_KV`, `BLOCK_Q` and `DKV_SHARDS`,
and the kernel body uses only those. Everything derived from `BLOCK_N` --
`SMEM_N_RPT`, the DMA issue count, the LDS line layout -- is about *a tile of
64 rows staged through LDS* and is correct unchanged, because the staging does
not care whose rows they are. Reusing `VO_SHARDS` rather than adding a field is
deliberate: `make_traits` already derives and *validates* the shard split
(`D_CHUNKS_PER_STAGE_SHARD`, the even-chunk rule the LDS offset decomposition
needs), and a second field would be a second derivation to keep in step.

--- What this module covers ------------------------------------------------

Dense and varlen, causal and windowed, MHA and GQA, with or without dropout
and bias, in bf16 and f16, over the full ladder `(32 … 512)` and the 8xD input
contract with padded heads. `resolve` refuses what has no meaning or what the
kernel cannot compute (a window without causal, bias with causal, a head count
that does not divide, a geometry the body cannot serve) rather than building
it.

--- The two walls, and which one binds -------------------------------------

Per staged slot at `BLOCK_Q` 64: `68 * head_dim` elements, i.e. `136 * d`
bytes. Two tensors and `NUM_STREAM_BUFFERS` buffers, against a 163840 B cap.
**LDS never binds below 512 and binds only the second buffer at 512**, which is
why this kernel needs no `D_STAGES` at all -- see `_with_buffers`.

Registers do bind. A wave holds `d/4` VGPRs of resident K, `d/4` of V, and
**two** accumulators of `d/2` each: `1.5 * d` before a single transient. The
levers, in the order they were found, are `(num_waves, waves_per_eu)` -- which is
free and decided head_dim 128 -- and then `DKV_SHARDS`, which divides the
accumulator pair by the shard count at the cost of every shard recomputing S
and dP. `_GEOMETRY` is the table that resolves all three, and the measurements
behind it are recorded there.
"""

from dataclasses import dataclass, fields, replace

from kernels.attention_bwd_cdna.fmha_traits_gfx950 import (
    ParityDualwaveTraits,
    make_traits,
)
from kernels.attention_bwd_cdna.fmha_tuning_gfx950 import GRID_AXIS_HEAD_FASTEST

__all__ = [
    "LADDER",
    "PACK_SPLIT_LIMIT",
    "SLAB_BYTES_LIMIT",
    "BwdDkDvInputMetadata",
    "BwdDkDvKnobs",
    "BwdDkDvTraits",
    "bwd_dkdv_knobs",
    "check_launch_limits",
    "rung_below",
    "tile_width_for",
]

# The forward's ladder, entire. A head_dim between two rungs is served by the
# next rung up with the real extent passed at runtime and the surplus D columns
# masked, which is what `padded_head` records.
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)

# Q rows one streamed tile carries. 64 is what one pass of the 32-row family's
# transposed operand spans (four 16-row k-substeps); the 16-row family's spans
# 32, so it can take either, and at the wide rungs it wants 32 -- half the
# transposed reads live at once, and the halved LDS buys the second stream
# buffer back. Measured at head_dim 512: 280 TFLOP/s at 64 with 36 spills, 416
# at 32 with none. Per-rung in `_GEOMETRY`.

# The addressable LDS cap. Measured elsewhere, not inferred: the compiler
# reports "local memory (N) exceeds limit (163840)" and says nothing about
# which knob to move, which is why `_with_buffers` decides this here.
LDS_CAP_BYTES = 163840

# Per-wave register demand, in VGPRs, as a function of the tile width and the
# shard count. Not a hardware limit -- **it is demand, and measured demand
# and grant differed by 1.8x** -- but it is what orders the search.
#
#     resident K + V          d/4 + d/4  = d/2      (loop-invariant)
#     dK acc + dV acc         d/2 + d/2  = d        (loop-invariant, / shards)
#
# so `d/2 + d/shards` before a single transient: 384 at head_dim 256 unsharded,
# and 384 again at 512 with four shards.
#
# The transients (one Q pack, the two score accumulators, P and dS, one
# transposed chunk) add roughly 100 more and do not scale with the tile width
# the same way, because the d-contraction reads are interleaved with their
# MFMAs rather than gathered first.


# `(num_waves, waves_per_eu, dkv_shards, mfma_rows, block_q)` by tile width.
# `BLOCK_KV` follows as `mfma_rows * num_waves / shards` and is *not* stored
# beside them; see `_with_geometry`.
#
# **Four levers, found in this order, and each one reframed the last.**
#
# 1. **The AGPR cliff**. At 8 waves -- 2 per SIMD -- a wave may address
#    256 registers *in total*, so the allocator cannot reach the AGPR file at
#    all and spills to scratch instead. At head_dim 128, 8 waves gave 0 AGPRs
#    and 118 spills at 444 TFLOP/s; 2 waves gave 108 AGPRs, no spills, 788. And
#    at 4 waves the only thing between 403 and 721 was the occupancy *hint*:
#    sweep `(num_waves, waves_per_eu)` as a pair, never the wave count alone. A
#    build at 0 AGPRs with a nonzero spill count is not short of registers, it
#    is forbidden from using half of them.
#
# 2. **`BLOCK_KV` is a bandwidth lever**. Every workgroup streams the whole
#    of Q and dO for its head, so the read traffic is `seqlen / BLOCK_KV`
#    copies of that slab. Raising the wave count while still at one wave per
#    SIMD is free traffic relief -- head_dim 160 went 408 -> 690 TF and 224 went
#    486 -> 723, with the register allocation unchanged. It also reframes
#    `dkv_shards`, which *divides* `BLOCK_KV` and so pays this back at the same
#    time as it buys accumulator space.
#
# 3. **16 rows per wave**. The loop invariant is `0.75 * d` rather than
#    `1.5 * d`, which fits at head_dim 512 unsharded -- removing the duplicated
#    S/dP *and* doubling `BLOCK_KV` at once. It must be `16x16x32`:
#    `16x16x16` is `Write4PassMAI` for half the FLOPs, i.e. half rate, and a
#    family built on it measured 280 TF at head_dim 64 against 713.
#
# 4. **`BLOCK_Q`**. The 16-row transposed operand spans 32 q rows rather
#    than 64, so it can take either -- and at the wide rungs 32 halves the live
#    transposed reads and buys the second stream buffer back. head_dim 512:
#    280 TF with 36 spills at `BLOCK_Q` 64, 416 with none at 32.
#
# Measured at `B=2 H=8 S=4096` bf16 non-causal, nominal FLOPs, with every
# choice under 10% settled by interleaved single-GPU A/B rather than by the
# sweep -- the two disagreed at head_dim 96, where a sweep at another shape
# preferred 32 rows by 7% and the A/B at the reporting shape preferred 16 by
# 16%:
#
#   head_dim  rows  waves  wpe  block_q  BLOCK_KV  AGPR  spills   TFLOP/s
#      32      32     4     2      64      128        0      0       504
#      64      32     4     1      64      128        0      0       735  (**)
#      96      16     4     1      64       64        0      0       733
#     128      16     4     1      64       64        0      0       775
#     160      32     4     1      64      128      100      0       735  (*)
#     192      16     4     1      32       64        0      0       849
#     224      32     4     1      64      128      230      0       799
#     256      16     4     1      32       64        0      0       743
#     384      16     4     1      32       64       96      0       423
#     512      16     4     1      32       64      214      0       429
#
# **No rung shards and every rung is double-buffered**, and every one
# compiles with zero scratch. Builds peak at under three seconds, far inside
# the eight-minute build budget.
#
# **(**) head_dim 64's non-causal, bias-free build does not use this row's
# `wpe`.** It takes `waves_per_eu=2` from `_FEATURE_OVERRIDES` instead, on this
# geometry and no other change; the 735 above was measured before the logsumexp
# layout became a runtime branch, and with it the rung wants 276 registers
# rather than 234 and does not reach two waves per SIMD on its own. Causal,
# bias and every other head dim still read `wpe` from here. See that entry for
# the numbers and for why this 2 is grantable where 160's is not.
#
# **(*) head_dim 160 takes 1, because 2 is never granted.** The
# backend said so on every build of this rung -- "failed to meet occupancy
# target given by 'amdgpu-waves-per-eu': desired occupancy was 2, final
# occupancy is 1" -- and the reason is LDS, not registers: the workgroup takes
# 87040 B, so two of them want 174080 B against `LDS_CAP_BYTES` and the second
# never lands however few registers the wave holds. That makes this rung
# structurally different from head_dim 32, the other rung asking for 2, which
# takes 17408 B and does get it.
#
# An unmet hint is not automatically an inert one -- it is still a register
# budget for the scheduler, which is note 1's whole point -- so the question
# was settled on the artifacts rather than assumed. Both tables were built at
# 1 and at 2 and the 24 head_dim 160 hsacos compared: **byte-identical, 24 of
# 24**, same sizes, same VGPR/AGPR counts, same zero scratch. At this rung the
# budget is slack -- occupancy 1 already allows the full register file, so
# asking for 2 constrains nothing the allocator was going to exceed -- and the
# hint's entire observable effect is whether the backend prints the warning.
#
# So the choice of 1 is about the build log, not performance, and it is stated
# that way deliberately: an interleaved A/B did appear to show the dense arm
# 5.5% faster at 1, reproducibly, across two runs. It was position in the
# round, not the knob. Each variant ran second in its pair, right after an arm
# that streams a 512 MB bias tensor through L2; reversing the order moved the
# same 5.5% to the other arm. Given byte-identical binaries no A/B here can
# measure anything else, which is what makes the reversal the check worth
# running: alternate which arm runs first.
#
# Both tables carry 1 for the one reason that survives: a request the hardware
# provably cannot satisfy, and provably does not change the code, is a false
# signal in the build log 24 times per build.
#
# 96, 160 and 224 are the granule-32 rungs, and two of the three keep 32 rows
# for a structural reason: at granule 32 a staged tile has `SMEM_N_RPT = 4`
# lines, so the wave count cannot exceed 4 and the 16-row family's `BLOCK_KV`
# is capped at 64 against the 32-row family's 128. `BLOCK_Q` 128 would restore
# `SMEM_N_RPT = 8` there and is the obvious thing to try next.
# **Tight registers** (`_TIGHT_REGISTERS`): whether the **32-row** body is written for minimum live registers rather than
# maximum instruction-level parallelism; see `_with_register_pressure`. The
# 16-row body has no such knob -- a q group's live set there is two 16x16
# accumulators, a quarter of what the 32-row body holds, so there is nothing to
# trade. Measured at `B=4 H=8 S=4096` on the 32-row family:
#
#   head_dim   loose            tight
#      32      472 TF, 0 sp     511 TF, 0 sp
#      64      712 TF, 0 sp     703 TF, 0 sp
#     128      744 TF, 0 sp     606 TF, 0 sp
#     224      723 TF, 0 sp     645 TF, 0 sp
#     384      197 TF, 368 sp   283 TF,  31 sp
#
# head_dim 32 is the odd one at the narrow end and is not a register story:
# with `D_CHUNKS == 1` there is barely any independent work for the loose arm
# to overlap, so all it does is lengthen live ranges. The wide rungs that
# wanted the tight arm are all on the 16-row family now, so only 32 reaches it.
_TIGHT_REGISTERS = {
    32: True,
    64: False,
    96: False,
    128: False,
    160: False,
    192: False,
    224: False,
    256: False,
    384: True,
    512: True,
}

# **A feature's register cost lands on whichever rung was already at the cap**,
# and head_dim 224 at 32 rows is the one -- 486 VGPR of 512 with no spills in a
# plain dense build. Causal's mask adds about 19 live registers and tips it. At
# the policy's geometry every other rung spills zero in all four feature
# combinations, so this is one rung's exception rather than a second table:
#
#   head_dim 224                 VGPR  spills   TFLOP/s
#     dense,         32 loose     486     0       798
#     causal,        32 loose     512    53       777    <- the mask
#     causal,        32 tight     512     0      1199    <- override
#     causal+varlen, 32 tight     512    20       858    <- override
#     dense+varlen,  32 loose     486     0       720
#     dense+varlen,  32 tight       -     -       723    <- override
#
# **The last two rows are the ones that can ship**, since the varlen decode
# stopped being a build axis and there is no dense binary to fall back to; the
# tight one is what does, under `(224, False, False)` below. Non-causal at this
# rung measures 473 after the row address rewrite, which is what earns the
# override -- 1.5x rather than the
# 720-vs-798 the loose row shows here.
#
# Causal+varlen is still the rung that pays: 20 spills and 858 against the 1199
# a causal-only build reached. Every other head dim spills zero either way. If
# that becomes unacceptable it is an override to re-measure (the 16-row family
# is the candidate), not a flag to restore.
#
# **Varlen itself is nearly free here, and what made it look otherwise was the
# row-tensor read.** When every varlen build took the sixteen-scalar path, 224
# went to 232 spills and 257 TFLOP/s -- which for a while looked like a second
# override wanting the 16-row family. Making the layout a build axis fixed the
# symptom; selecting the arm once outside the tile loop fixed it properly, and
# without a build axis that could disagree with the runtime bits. See
# `BwdDkDvInputMetadata.lse_layout_th`.
#
# **Bias needs four more entries, and the reason is not register pressure at
# three of them.** A bias read is one scalar `buffer_load` per accumulator
# element -- a lane holds one KV column and 16 q rows, `stride_b_seq_q` apart,
# so there is no run to vectorise. The 32-row family holds *32* such elements
# per tile to the 16-row family's 16, and issues them in one place, which is
# why the two families pay so differently at the policy's geometry:
#
#   bias, B=2 H=8 S=4096            dense    bias   ratio
#     head_dim  96 / 128 / 192      (16-row)         1.33 / 1.26 / 1.17x
#     head_dim 256 / 384 / 512      (16-row)         2.13 / 1.78 / 1.79x
#     head_dim  32 /  64 / 160      (32-row)         4.05 / 5.04 / 2.77x
#     head_dim 224                  (32-row)         6.09x
#
# Only 224 spills (512 VGPR, 229 of them). The other three are the issue cost
# of 32 dependent scalar loads landing in a tile whose MFMA work is small, and
# the fix is a different geometry rather than more registers:
#
#   head_dim 64, bias      policy (32 loose)  149 TF
#                          32 tight           520 TF   <- 3.5x, and no spills
#                                                         either way
#   head_dim 160, bias     policy (32 loose)  277 TF
#                          16-row bq64        487 TF   <- 1.76x
#   head_dim 224, bias     policy (32 loose)  136 TF   (229 spills)
#                          32 tight           227 TF   <- 1.67x
#   head_dim 32, bias      policy (32 loose)  125 TF
#                          16-row bq64        182 TF   <- 1.46x
#
# So the "5x cost of bias at head_dim 64" was a geometry mismatch, not the
# feature: with the right arm it is 1.44x, which is what the bytes-per-flop
# law predicts (`BLOCK_KV / 4d` extra streamed bytes, 0.5 at that rung).
# 256/384/512 are already best at the policy's geometry and get no entry.
#
# **GQA gets no entry, and that is a measurement rather than an omission.**
# head_dim 224 at group 4 spills too (512 VGPR, 160 spills) and the 16-row
# family is 1.29x faster there -- but `32 tight`, which removes the spills, is
# only 1.02x, so the registers are not what binds. The 16-row family wins
# because its `BLOCK_KV` is half as wide and it therefore dispatches twice the
# workgroups, and under GQA the grid has already shrunk by the group size. That
# makes the right geometry a function of the group size, the batch *and* the
# sequence length rather than of the head dim, which is not something this
# table can express.
# **No varlen axis in the key**: the decode is unconditional, so one entry
# serves dense and varlen alike.
#
# `(160, False, True)` takes `waves_per_eu=1`, like `_GEOMETRY[160]` and for the
# same reason: the rung's 87040 B of LDS makes a second workgroup impossible, so
# a 2 would be a request the backend refuses on every build, and the hsaco is
# byte-identical either way. See the `(*)` note on the
# geometry table. head_dim 32 keeps its 2 -- it takes 17408 B, the hint is
# granted, and nothing here applies to it.
_FEATURE_OVERRIDES = {
    # (head_dim, causal, bias): (waves, waves_per_eu, shards, rows, block_q, tight)
    (224, True, False): (4, 1, 1, 32, 64, True),
    # Bias excludes causal by construction, so these can never collide with the
    # one above.
    (32, False, True): (4, 2, 1, 16, 64, False),
    (64, False, True): (4, 1, 1, 32, 64, True),
    (160, False, True): (4, 1, 1, 16, 64, False),
    (224, False, True): (4, 1, 1, 32, 64, True),
    # **Three rungs the runtime logsumexp layout moved**, measured at
    # `B=2 H=8 S=4096` packed varlen. `BwdDkDvSoftmaxHelper.load_row_values` now
    # emits both row-read arms and picks between them once outside the tile
    # loop, and the 32-row family -- thirty-two accumulator elements per lane to
    # the 16-row family's four -- is the one that gives way.
    #
    #   head_dim  64 causal        policy  917   16-row      1017
    #   head_dim  64 non-causal    wpe 1   635   wpe 2        734
    #   head_dim 224 non-causal    policy  473   32 tight     723
    (64, True, False): (4, 1, 1, 16, 64, False),
    (224, False, False): (4, 1, 1, 32, 64, True),
    # **This one asks for the occupancy, not a geometry** -- the five other
    # fields are `_GEOMETRY[64]`'s. A dense build reaches two waves per SIMD on
    # its own at 234 VGPRs; ours wants 276 and, told `waves_per_eu=1`, LLVM has
    # a 512-register budget and no reason to stop. It buys the second wave for
    # 24 scratch slots: the spills cost 2%, the wave is worth 15%. This is the
    # only rung where that trade has to be made rather than won outright.
    #
    # Unlike `(160, ...)` and `_GEOMETRY[160]` this 2 *is* grantable -- the rung
    # is nowhere near the LDS cliff that makes a second workgroup impossible at
    # 160 -- and the build log is the check: an `amdgpu-waves-per-eu` warning on
    # a head_dim 64 dK/dV build means this entry asked for something the backend
    # refused, and it should go rather than sit there unmet.
    (64, False, False): (4, 2, 1, 32, 64, False),
}


def _geometry_for(block_dmodel, meta):
    """`(waves, waves_per_eu, shards, rows, block_q, tight)` for a build."""
    key = (block_dmodel, bool(meta.causal), bool(meta.bias))
    if key in _FEATURE_OVERRIDES:
        return _FEATURE_OVERRIDES[key]
    return _GEOMETRY[block_dmodel] + (_TIGHT_REGISTERS[block_dmodel],)


_GEOMETRY = {
    # head_dim: (waves, waves_per_eu, shards, mfma_rows, block_q)
    32: (4, 2, 1, 32, 64),
    64: (4, 1, 1, 32, 64),
    96: (4, 1, 1, 16, 64),
    128: (4, 1, 1, 16, 64),
    160: (4, 1, 1, 32, 64),
    192: (4, 1, 1, 16, 32),
    224: (4, 1, 1, 32, 64),
    256: (4, 1, 1, 16, 32),
    384: (4, 1, 1, 16, 32),
    512: (4, 1, 1, 16, 32),
}


def _granule_for(block_dmodel):
    """The D-axis staging granule for a tile width.

    One DMA issue moves 512 bf16 per wave, so the granule fixes how many tokens
    that covers and therefore how many LDS lines a tile occupies. 64 is family
    A's; a width off the 64 grid needs 32, which is the forward's family S and
    is served by the same read helpers this
    kernel uses. The floor is the PV MFMA's 32-column output, so 16 is not
    available at any width.
    """
    return 64 if block_dmodel % 64 == 0 else 32


def rung_below(block_dmodel):
    """The widest rung strictly narrower than `block_dmodel`, or 0.

    `tile_width_for` rounds *up*, so a build it chose serves exactly the
    half-open range `(rung_below(R), R]`. That lower bound is what lets the
    kernel skip masking the D columns it knows are real -- see `HDIM_QK_FLOOR`
    in the kernel. A property of the ladder rather than of any build, so adding
    a rung silently tightens every wider build's floor, which is correct.
    """
    below = [r for r in LADDER if r < block_dmodel]
    return max(below) if below else 0


def tile_width_for(head_dim):
    """The compiled tile width serving `head_dim`, or raise saying why not."""
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    for rung in LADDER:
        if head_dim <= rung:
            return rung
    raise ValueError(f"head_dim {head_dim} exceeds the widest tile ({max(LADDER)})")


@dataclass(frozen=True)
class BwdDkDvTraits(ParityDualwaveTraits):
    """The forward's traits, read with the backward's role names.

    A subclass with no new *derived* state: every field is the parent's, and
    the properties below are aliases that make the kernel body readable rather
    than new numbers. That is deliberate -- the moment the LDS layout here
    stops being the forward's, the transpose read (which is the forward's V
    read, unmodified) stops being validated.
    """

    # How many stream buffers the body cycles. 2 prefetches one tile ahead: the
    # DMA for tile `j+2` is issued once every wave has finished reading tile
    # `j`, so it overlaps tile `j+1`'s compute. 1 is the same body with the
    # prefetch distance collapsed to zero, which `_with_buffers` falls back to
    # when two buffers do not fit LDS. The body issues two tiles per loop
    # iteration, so the count is exactly 1 or 2.
    NUM_STREAM_BUFFERS: int = 2

    # KV rows one wave owns, and therefore which MFMA family the body is: 32
    # for `v_mfma_f32_32x32x16` (`fmha_bwd_dkdv_gfx950`) or 16 for
    # `v_mfma_f32_16x16x32` (`fmha_bwd_dkdv_m16_gfx950`). The family is a trait
    # rather than a literal, so every site that depends on it reads it from
    # here.
    MFMA_ROWS: int = 32

    @property
    def BLOCK_KV(self):
        """KV rows one workgroup owns, resident in registers across the Q loop."""
        return self.BLOCK_M

    @property
    def BLOCK_Q(self):
        """Q rows one streamed tile carries through LDS."""
        return self.BLOCK_N

    @property
    def DKV_SHARDS(self):
        """Waves splitting the D axis of the dK and dV accumulators.

        `VO_SHARDS` under its forward name, and the same bargain: D is an
        *output* axis for both accumulators, so shards write disjoint columns
        and **never have to agree on anything** -- no cross-wave reduction, no
        extra barrier, no summation order to get wrong. The price is that every
        shard recomputes the whole of S and dP, which is half the arithmetic.
        """
        return self.VO_SHARDS

    @property
    def KV_BLOCKS_PER_WG(self):
        """Distinct 32-row KV blocks in one workgroup: `num_waves / shards`."""
        return self.Q_TILES

    @property
    def D_CHUNKS_PER_SHARD(self):
        """Output chunks one shard owns. Even, which the LDS offset needs."""
        return self.D_CHUNKS_PER_STAGE_SHARD

    @property
    def STREAM_LINE_STRIDE(self):
        """LDS elements per staged line.

        **Both streamed tiles use the forward's *V* line stride, not its K
        one**, and that is a correctness requirement rather than a padding
        preference. Q and dO are each read two ways -- row-major for the
        d-contracted GEMMs and column-major through `ds_read_b64_tr_b16` for
        the q-contracted ones -- and the transpose read path is only validated
        against `SMEM_V_LINE_STRIDE`. The row-major read does not care which
        stride it is given, so making both tiles V-shaped is what lets one
        staged tile serve both readers.
        """
        return self.SMEM_V_LINE_STRIDE

    @property
    def STREAM_TILE_ELEMS(self):
        """LDS elements one staged tile occupies."""
        return self.SMEM_V_TILE_ELEMS

    @property
    def LDS_STREAM_TOTAL_SIZE(self):
        """LDS elements for both tensors across all buffers."""
        return 2 * self.NUM_STREAM_BUFFERS * self.STREAM_TILE_ELEMS


def _as_bwd_traits(base, **extra):
    """Re-wrap a `ParityDualwaveTraits` as `BwdDkDvTraits`, field for field."""
    return BwdDkDvTraits(
        **{f.name: getattr(base, f.name) for f in fields(base)}, **extra
    )


@dataclass(frozen=True)
class BwdDkDvInputMetadata:
    """What to compute. Set by the caller; never by policy, and never by arch."""

    num_heads: int
    head_dim: int
    # `"bf16"` or `"f16"`. Everything the tuner decides is dtype-independent
    # -- both are two bytes, so the LDS budget, the staging granule, the tile
    # geometry and the register pressure are identical, and the tables are
    # shared rather than duplicated. Measured, not assumed.
    #
    # **The two differ in range, not in size, and only three things in this
    # kernel care.** `P = exp2(qk*scale + bias*log2e - lse*log2e)` is bounded
    # by 1 by construction (`lse` is the log-sum-exp of the row, so no score
    # exceeds it), which f16 holds exactly as well as bf16. `LSE` and `delta`
    # are f32 row tensors whatever this says, so the padding-row argument --
    # a zero row gives `P = exp2(0) = 1` and contributes zero -- is f32
    # arithmetic and carries unchanged. **`dS` is the one exposure**: it is
    # packed to this dtype for GEMM4 and is unbounded, so an f16 build
    # overflows at 65504 where bf16 would not. That is inherent to f16 and is
    # the same exposure the forward and every other f16 attention kernel has.
    dtype_str: str = "bf16"
    num_kv_heads: int | None = None
    head_dim_v: int | None = None
    sm_scale: float | None = None

    # `window` is a *left* bound on top of the causal right one, so it
    # requires `causal`; see `_checked_scope`.
    causal: bool = False
    window: bool = False
    # **There is no `varlen` field.** The kernel decodes `VarlenBits`
    # unconditionally and dense is `bits == 0`, so one build serves all six
    # configurations *and* the rectangular one; there is nothing left to decide
    # here.
    #
    # Whether the logsumexp and delta tensors use Transformer Engine's
    # `(T, H)` layout rather than AOTriton's `(H, T)`. `_HT` makes the row
    # pitch 1, so a lane's four accumulator rows are adjacent and one `dwordx4`
    # fetches them; `_TH` makes it `num_heads` and the same four rows need four
    # scalar loads.
    #
    # **This is a tuning record, not a build axis, and it does not reach
    # emitted code.** The kernel decodes the layout from `VarlenBits` at
    # runtime (`lse_row_addressing`) and picks one of two row-read arms once,
    # outside the tile loop, so one build serves either layout. The choice sits
    # outside the loop because a branch *inside* the row read is a scheduling
    # barrier the loads cannot be hoisted across. The field is kept here and in
    # the cache key so a build records which layout it was tuned against;
    # nothing reads it at trace time.
    lse_layout_th: bool = False
    # Whether this build regenerates the forward's philox mask. The rate,
    # the seed and the counter are runtime arguments; only the decision to
    # compile the PRNG at all is here, because a build without dropout must
    # emit none of it.
    dropout: bool = False
    # AOTriton's `BIAS_TYPE`: a `(b, h, sq, sk)` matrix added to the scores
    # before the softmax. A *read* here and nothing more -- `dB` belongs to the
    # dQ kernel, which is where `dS` is materialised per `(q, k)` element.
    #
    # A build axis for the reason the forward gives: the loop is latency-bound
    # and a bias that cost anything when unused would be paid by every caller
    # who does not want one. It also makes `bias=None` bitwise identical to a
    # build compiled without bias, since there is nothing to be identical to
    # otherwise. **Mutually exclusive with `causal`/`window`**, which
    # `make_traits` enforces; see `build_traits`.
    bias: bool = False


@dataclass(frozen=True)
class BwdDkDvKnobs:
    """How to compute it. `None` means "policy decides".

    `None` rather than a literal default for the same reason as the forward's
    knobs: it is the only way `resolve` can tell "the caller wants 1" from "the
    caller did not say".
    """

    block_dmodel: int | None = None
    block_dmodel_v: int | None = None
    padded_head: bool | None = None
    hdim_qk_floor: int | None = None

    # Pinned as a set of four; see `_with_geometry`.
    num_waves: int | None = None
    block_kv: int | None = None
    block_q: int | None = None
    head_dim_granule: int | None = None

    # Per-width policy a sweep wants to vary on its own, so not part of the
    # four-field set above.
    dkv_shards: int | None = None
    mfma_rows: int | None = None
    num_stream_buffers: int | None = None
    waves_per_eu: int | None = None
    tight_registers: bool | None = None

    # Perf knob, `None` (the shipped code path) or `1`. Not policy-resolved: a set value is what the build does.
    #   gqa_pack       fold the GQA group into the q-tile row axis: flat row r -> (head r // Sq, q r % Sq), one pass, no group loop
    #                  (16-row family; composes with causal, window and dropout; see `_checked_candidates`).
    gqa_pack: int | None = None

    daz: bool | None = None
    strides_constexpr: bool | None = None

    # Which quantity lands on which grid axis; the enum is documented in
    # `fmha_tuning_gfx950`. Set by `resolve`, never by a caller. Upper case
    # because the field name is the wire key the C++ side reads back. **Note
    # the head here is the *kv* head**, which is what the GQA fold made it.
    GRID_AXIS_ORDER: int | None = None

    # There is deliberately no `traits` field: a knob class is plain build
    # options, every value a scalar, so the set can be recorded beside the
    # compiled hsaco in a flat `k=v` wire format. `build_traits` derives them.

    def merge(self, other):
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {
            f.name: getattr(other, f.name)
            for f in fields(other)
            if getattr(other, f.name) is not None
        }
        return replace(self, **set_fields)

    def resolve(self, meta: BwdDkDvInputMetadata) -> "BwdDkDvKnobs":
        """The complete build configuration for `meta`, traits included.

        Idempotent: every derived field is recomputed from `meta` and the
        pinned fields rather than read back.
        """
        return (
            _FALLBACK.merge(self)
            ._checked_scope(meta)
            ._with_widths(meta)
            ._with_geometry(meta)
            ._with_buffers()
            ._with_register_pressure(meta)
            ._checked_against_traits(meta)
        )

    def _checked_scope(self, meta):
        """Refuse what has no meaning, at the decision rather than at an address.

        Every one of these would otherwise produce a plausible, finite, wrong
        gradient -- a dropped causal mask returns dense attention's, which is
        the right shape. What is refused is one arithmetic precondition (the head
        count) and one semantic exclusion (a window without causal), not a scope
        boundary.
        """
        if meta.window and not meta.causal:
            # A window without the causal machinery has no masked region to
            # apply itself to, and silently dropping it would return dense
            # attention -- the right shape, finite, and wrong. An unbounded
            # `window_right` expresses a pure left-band, so nothing is lost.
            # `make_traits` says the same; this says it before the ladder runs.
            raise ValueError(
                "window=True requires causal=True; it is a left bound on top of the causal one"
            )
        if meta.dtype_str not in ("bf16", "f16"):
            raise NotImplementedError(
                f"this kernel builds bf16 and f16, got {meta.dtype_str!r}"
            )
        num_kv_heads = (
            meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        )
        if meta.num_heads % num_kv_heads:
            # A group that does not divide evenly has no meaning, rather than
            # no implementation.
            raise ValueError(
                f"num_heads ({meta.num_heads}) must be a multiple of num_kv_heads ({num_kv_heads}): "
                "each KV head owns a whole group of query heads and dK/dV sum over it"
            )
        return self

    def _with_widths(self, meta):
        """Decide `block_dmodel`, `block_dmodel_v`, `padded_head` and the mask floor."""
        block_dmodel = self.block_dmodel
        derived = block_dmodel is None
        if derived:
            block_dmodel = tile_width_for(meta.head_dim)
        elif block_dmodel not in LADDER:
            raise ValueError(
                f"block_dmodel must be one of the built rungs {LADDER}, got {block_dmodel}"
            )
        if meta.head_dim > block_dmodel:
            raise ValueError(
                f"head_dim {meta.head_dim} does not fit the pinned block_dmodel {block_dmodel}"
            )

        head_dim_v = meta.head_dim if meta.head_dim_v is None else meta.head_dim_v
        if head_dim_v > block_dmodel:
            raise ValueError(
                f"head_dim_v {head_dim_v} does not fit block_dmodel {block_dmodel}"
            )
        block_dmodel_v = (
            block_dmodel if self.block_dmodel_v is None else self.block_dmodel_v
        )

        padded_head = self.padded_head
        if padded_head is None:
            padded_head = (meta.head_dim != block_dmodel) or (
                head_dim_v != block_dmodel_v
            )
        elif not padded_head and (
            meta.head_dim != block_dmodel or head_dim_v != block_dmodel_v
        ):
            raise ValueError(
                f"padded_head=False serves exactly a {block_dmodel}x{block_dmodel_v} tile; got head_dim "
                f"{meta.head_dim} and head_dim_v {head_dim_v}, whose surplus D columns the kernel would read as data"
            )

        # Only a *derived* tile carries the ladder's guarantee. A caller that
        # pins `block_dmodel` may pin 256 for head_dim 64, so there is no floor
        # to claim and the kernel masks every column.
        hdim_qk_floor = self.hdim_qk_floor
        if hdim_qk_floor is None:
            hdim_qk_floor = rung_below(block_dmodel) if derived else 0
        if not 0 <= hdim_qk_floor < block_dmodel:
            raise ValueError(
                f"hdim_qk_floor {hdim_qk_floor} must be in [0, block_dmodel={block_dmodel})"
            )
        if meta.head_dim <= hdim_qk_floor:
            raise ValueError(
                f"head_dim {meta.head_dim} is at or below this build's hdim_qk_floor {hdim_qk_floor}; "
                f"the {block_dmodel}-wide tile only serves ({hdim_qk_floor}, {block_dmodel}]"
            )
        if head_dim_v <= hdim_qk_floor:
            # One floor masks both the Q reads and the dO reads, so the dO columns between `head_dim_v` and the
            # floor would be read unmasked.
            raise ValueError(
                f"head_dim_v {head_dim_v} is at or below this build's hdim_qk_floor {hdim_qk_floor}; "
                f"the {block_dmodel}-wide tile only serves ({hdim_qk_floor}, {block_dmodel}] for both extents"
            )
        return replace(
            self,
            block_dmodel=block_dmodel,
            block_dmodel_v=block_dmodel_v,
            padded_head=bool(padded_head),
            hdim_qk_floor=int(hdim_qk_floor),
        )

    def _with_geometry(self, meta):
        """Decide waves, `BLOCK_KV`, the granule and the shard count from the tile width.

        `_GEOMETRY` is the whole policy and its table is where the measurement
        lives. **`BLOCK_KV` is derived rather than pinned beside the wave
        count** -- a wave owns exactly `mfma_rows` KV rows (the MFMA's M extent)
        and shards split the waves, not the rows, so
        `BLOCK_KV = mfma_rows * num_waves / shards` and the two cannot disagree;
        the forward shipped twelve configurations where that invariant was only
        prose. The 32-row body reduces over exactly 64 q rows per streamed tile,
        so `mfma_rows` 32 takes `block_q` 64 whatever the table says for the rung.

        Reads `self.block_dmodel`, so it runs after `_with_widths`.
        """
        shards = self.dkv_shards
        if shards is not None and shards <= 0:
            raise ValueError(f"dkv_shards must be positive, got {shards}")
        pinned = (self.num_waves, self.block_kv, self.block_q, self.head_dim_granule)
        if all(x is not None for x in pinned):
            if shards is None or self.mfma_rows is None:
                raise ValueError(
                    "a pinned geometry must pin dkv_shards and mfma_rows too; they decide BLOCK_KV"
                )
            return self
        if any(x is not None for x in pinned):
            raise ValueError(
                f"pin num_waves, block_kv, block_q and head_dim_granule together or not at all, got {pinned}"
            )
        num_waves, waves_per_eu, table_shards, table_rows, table_bq, _tight = (
            _geometry_for(self.block_dmodel, meta)
        )
        if shards is None:
            shards = table_shards
        rows = self.mfma_rows if self.mfma_rows is not None else table_rows
        return replace(
            self,
            num_waves=num_waves,
            waves_per_eu=(
                self.waves_per_eu if self.waves_per_eu is not None else waves_per_eu
            ),
            dkv_shards=shards,
            mfma_rows=rows,
            block_kv=rows * (num_waves // shards),
            block_q=64 if rows == 32 else table_bq,
            head_dim_granule=_granule_for(self.block_dmodel),
        )

    def _with_buffers(self):
        """Two stream buffers if LDS allows, one if it does not.

        **The second buffer is the only thing LDS ever costs this kernel**, and
        it is worth stating why there is no `D_STAGES` here at all. A staged
        slot is `68 * head_dim` elements, so two tensors double-buffered are
        `544 * head_dim` bytes: 139264 at head_dim 256 and 278528 at 512
        against a 163840 cap. Single-buffered they are `272 * head_dim`, which
        is 139264 at 512 -- so head_dim 512 fits *with a whole tile of each
        tensor resident*, and the D axis never has to be split in time.

        What is lost at one buffer is the prefetch distance: the DMA for the
        next tile is issued at the end of this one and waited on immediately.
        The body is otherwise identical, which is why this is a number rather
        than a second code path.
        """
        if self.num_stream_buffers is not None:
            return self
        per_slot = self._stream_slot_bytes()
        return replace(
            self, num_stream_buffers=2 if 4 * per_slot <= LDS_CAP_BYTES else 1
        )

    def _stream_slot_bytes(self):
        """Bytes one staged tile occupies, without building the traits.

        Duplicates two lines of `make_traits`' derivation, and does so
        knowingly: `_with_buffers` has to decide the buffer count *before* the
        traits exist, and the alternative is building throwaway traits to ask
        them. Guarded -- `build_traits` asserts the two agree.
        """
        granule = self.head_dim_granule
        smem_n_rpt = self.block_q // (512 // granule)
        line = 512 + 32
        return smem_n_rpt * (self.block_dmodel // granule) * line * 2

    def _with_register_pressure(self, meta):
        """Trade instruction-level parallelism for live registers, or not.

        One knob for two linked choices in the tile body -- whether a staged
        half is read pack-by-pack into its MFMA or all at once, and whether the
        two halves go through the softmax together. Both hold roughly the same
        thing live and both cost the same thing: the scheduler's freedom to
        overlap an MFMA burst with the loads and VALU of independent work.
        `_TIGHT_REGISTERS` records where they cross.
        """
        if self.tight_registers is not None:
            return self
        return replace(self, tight_registers=_geometry_for(self.block_dmodel, meta)[5])

    def _checked_against_traits(self, meta):
        """`resolve`'s last step: prove the traits are buildable, return `self`.

        The traits object is built and discarded. Only its verdict is kept --
        every check in `build_traits` names the knob to move, so a bad
        configuration fails here rather than at a kernel address. This class
        does not inherit `Gfx950Knobs`, so it carries its own copy of a method
        the forward and dQ share.
        """
        traits = self.build_traits(meta)
        self._checked_candidates(meta, traits)
        return replace(self, GRID_AXIS_ORDER=GRID_AXIS_HEAD_FASTEST)

    def _checked_candidates(self, meta, traits):
        if self.gqa_pack not in (None, 1):
            raise ValueError(
                f"gqa_pack is spelled gqa_pack=1 or left unset, got {self.gqa_pack!r}"
            )
        if self.gqa_pack:
            if self.mfma_rows != 16:
                raise ValueError(
                    f"gqa_pack is written for the 16-row family, got mfma_rows {self.mfma_rows}"
                )
            if traits.D_STAGES != 1:
                raise ValueError("gqa_pack needs D_STAGES == 1 (one D stage per DMA)")

    def build_traits(self, meta):
        """The traits this configuration implies. **The one knob->traits map.**

        Public, and called by the builder rather than read from a `traits` field,
        which is what keeps the knob object plain data -- see the note beside the
        field list.

        `make_traits` is the forward's, called with `block_m=block_kv`,
        `block_n=block_q` and `vo_shards=dkv_shards`. Those three slots change
        meaning; nothing else does. Using its `vo_shards` rather than a field
        of our own is what gets the shard validation -- the even-chunk rule the
        LDS offset decomposition depends on -- for free.
        """
        if self.mfma_rows not in (16, 32):
            raise ValueError(f"mfma_rows must be 16 or 32, got {self.mfma_rows}")
        if self.mfma_rows == 32 and self.block_q != 64:
            # `BwdDkDvGemmHelper.contract_q` and the 32-row tile body reduce over exactly 64 q rows (two 32-row halves) per
            # streamed tile, so a 32-row q tile would re-read the next tile's first rows. BLOCK_Q 32 exists only in the 16-row
            # family, whose q-group count derives from BLOCK_Q.
            raise ValueError(f"mfma_rows 32 needs block_q 64, got {self.block_q}")
        if self.dkv_shards is None or self.dkv_shards <= 0:
            raise ValueError(f"dkv_shards must be positive, got {self.dkv_shards}")
        if self.num_stream_buffers not in (1, 2):
            # The tile loop issues two tiles per iteration and prefetches one or two tiles ahead; another count would
            # consume tiles out of order.
            raise ValueError(
                f"num_stream_buffers must be 1 or 2, got {self.num_stream_buffers}"
            )
        if self.waves_per_eu is not None and self.waves_per_eu < 1:
            raise ValueError(f"waves_per_eu must be positive, got {self.waves_per_eu}")
        num_kv_heads = (
            meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        )
        base = make_traits(
            num_heads=meta.num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.block_dmodel,
            num_waves=self.num_waves,
            block_m=self.block_kv,
            block_n=self.block_q,
            granule=self.head_dim_granule,
            vo_shards=self.dkv_shards,
            causal=meta.causal,
            window=meta.window,
            dropout=meta.dropout,
            # `make_traits` is also where `bias and (causal or window)` is
            # refused, and taking that from there rather than restating it is
            # the point: the rule is semantic (a bias *is* an attention mask,
            # so pairing it with a positional one asks which wins where they
            # disagree) and it must read identically on both sides of the
            # gradient. The forward cannot build the pair either, so a backward
            # that accepted it would have no forward to produce its LSE.
            bias=meta.bias,
            # `make_traits` derives `CROSS_SEQLEN` from `causal`, so this
            # kernel gets it under causal whether or not it needs it. It does
            # not: the forward needs it because a Q block with no live key must
            # have `O` *written* as zero; dK/dV accumulate from zero and store
            # whatever they accumulated, so a KV block with no live query stores
            # zeros by construction. Nothing here reads the flag.
            dtype_str=meta.dtype_str,
            waves_per_eu=self.waves_per_eu,
            daz=self.daz,
        )
        traits = _as_bwd_traits(
            base, NUM_STREAM_BUFFERS=self.num_stream_buffers, MFMA_ROWS=self.mfma_rows
        )
        if self.mfma_rows == 16 and self.dkv_shards != 1:
            # The 16-row family exists so that sharding is unnecessary, and it
            # keeps `a16_chunk_offset` a compile-time immediate by not having a
            # runtime shard origin. Rejected rather than left as a precondition.
            raise ValueError(
                f"mfma_rows 16 with dkv_shards {self.dkv_shards}: the 16-row family does not shard. Its "
                "loop invariant is 0.75*d, which fits unsharded at every rung; sharding would also make "
                "the transpose read's chunk offset a runtime value. Pass dkv_shards=1."
            )
        if self.mfma_rows == 16 and traits.SMEM_N_RPT < 4:
            # The 16-row transposed read of `fmha_mfma16_gfx950.a16_read_base` addresses four staged lines per q group
            # (token = 8g + a, a in 0..3), so a tile with fewer lines would read the wrong tokens.
            raise ValueError(
                f"mfma_rows 16 needs SMEM_N_RPT >= 4 staged lines per tile, got {traits.SMEM_N_RPT} at block_q "
                f"{self.block_q} and head_dim_granule {self.head_dim_granule}"
            )

        # **The per-wave row ceiling, enforced rather than commented.** A wave holds
        # exactly the MFMA's M extent (`mfma_rows`) in KV rows: more would address
        # rows the accumulator does not have, fewer would run a full-height
        # MFMA and store rows the workgroup does not own. `make_traits` rejects
        # the first; this rejects the second, and together they make the
        # invariant checkable instead of stated. The forward shipped twelve
        # silently-wrong configurations for want of exactly this.
        if traits.ROWS_PER_WAVE != self.mfma_rows:
            raise ValueError(
                f"block_kv {self.block_kv} over {self.num_waves} waves at {self.dkv_shards} shards gives "
                f"{traits.ROWS_PER_WAVE} KV rows per wave, and this build's family serves exactly "
                f"{self.mfma_rows}. It is the MFMA's N extent, and the accumulator, the LSE/delta row "
                f"map, the operand packing and the store are all written against it. "
                f"Pass block_kv={self.mfma_rows * (self.num_waves // self.dkv_shards)}."
            )
        if traits.STREAM_TILE_ELEMS * traits.BF16_BYTES != self._stream_slot_bytes():
            raise AssertionError(
                f"`_stream_slot_bytes` says {self._stream_slot_bytes()} B per slot but the traits derive "
                f"{traits.STREAM_TILE_ELEMS * traits.BF16_BYTES}; the buffer-count decision was made "
                "against a stale copy of the LDS derivation"
            )
        lds_bytes = traits.LDS_STREAM_TOTAL_SIZE * traits.BF16_BYTES
        if lds_bytes > LDS_CAP_BYTES:
            raise ValueError(
                f"Q + dO staging needs {lds_bytes} B of LDS, over the {LDS_CAP_BYTES} B cap, for "
                f"block_dmodel {self.block_dmodel} at block_q {self.block_q} with "
                f"{self.num_stream_buffers} buffers. Drop to one buffer, or lower block_q."
            )
        return traits


# Defaults the policy has no shape-dependent opinion about. `daz` is the
# forward's, so a default build runs under the same denormal regime it was
# measured on. `waves_per_eu` is deliberately absent: it is per-width and
# `_with_geometry` supplies it.
_FALLBACK = BwdDkDvKnobs(
    daz=True,
    strides_constexpr=False,
)


def bwd_dkdv_knobs(arch="gfx950", **overrides):
    """Knobs for `arch`, with `overrides` pinned. Mirrors `fmha_knobs`."""
    if arch != "gfx950":
        raise ValueError(f"this tuning module is gfx950-only, got {arch!r}")
    return BwdDkDvKnobs(**overrides)


# A buffer descriptor's `num_records` is a 32-bit byte count, and every per-lane offset (and the store-suppression sentinel, which sits at
# the end of the slab) is an `Int32` element offset scaled to bytes in 32 bits, so a slab that reaches this many bytes wraps.
SLAB_BYTES_LIMIT = 1 << 32
# `m16.pack_split` splits a flat packed row `x` into `(head, q)` with `(x * magic) >> 31`, `magic = 0x7FFFFFFF // Sq + 1`, which is exact only
# while `x * Sq < 2^31`.
PACK_SPLIT_LIMIT = 1 << 31


def check_launch_limits(
    *, elem_bytes, block_q, gqa_pack, group, seqlen_q, seqlen_k, strides, bias=None
):
    """Refuse a launch whose addressing exceeds the kernel's 32-bit limits (the kernel itself carries no guard, and an AOT launcher never calls `_args`).

    `strides` maps each of `Q`, `K`, `V`, `DO`, `DK`, `DV` to `(stride_head, stride_seq)` in elements; `bias` is `(stride_head, stride_seq_q)` or
    `None`; `seqlen_q` / `seqlen_k` are the longest sequence (the padded length for a ragged batch); `group` is `num_head_q // num_head_k`.

    - **Slab bytes.** Each `(batch, head)` slab is bounded at `rows * stride_seq` elements (its suppression sentinel; the descriptor itself ends at
      `(rows - 1) * stride_seq + ceil8(D)`, below it), and `gqa_pack` extends the Q and dO slabs by `group - 1` head steps. Reaching
      `SLAB_BYTES_LIMIT` bytes wraps the offsets, and a "suppressed" padded-row or padded-column store lands in bounds and overwrites real data.
    - **Bias.** The slab is `(seqlen_q - 1) * stride_seq_q + seqlen_k` elements (plus the group extension under `gqa_pack`), bounded the same way.
    - **`gqa_pack`.** `pack_split` is exact for every packed row, pad to the tile count included, only while `Sq * (Sq * group + 2 * block_q) < 2^31`.
    """
    extra_heads = group - 1 if gqa_pack else 0
    rows = {
        "Q": seqlen_q,
        "DO": seqlen_q,
        "K": seqlen_k,
        "V": seqlen_k,
        "DK": seqlen_k,
        "DV": seqlen_k,
    }
    for name, (stride_head, stride_seq) in strides.items():
        elems = rows[name] * stride_seq + (
            extra_heads * stride_head if name in ("Q", "DO") else 0
        )
        if elems * elem_bytes >= SLAB_BYTES_LIMIT:
            raise ValueError(
                f"{name}'s per-head slab of {rows[name]} rows at stride_seq {stride_seq}"
                f"{f' plus {extra_heads} head steps of {stride_head}' if name in ('Q', 'DO') and extra_heads else ''} is "
                f"{elems * elem_bytes} B, at or over the kernel's 32-bit limit of {SLAB_BYTES_LIMIT} B; the offsets wrap and a "
                "suppressed store would land on real data"
            )
    if bias is not None:
        stride_head, stride_seq = bias
        elems = max(seqlen_q - 1, 0) * stride_seq + seqlen_k + extra_heads * stride_head
        if elems * elem_bytes >= SLAB_BYTES_LIMIT:
            raise ValueError(
                f"the bias slab of {seqlen_q} rows at stride {stride_seq} by {seqlen_k} columns is {elems * elem_bytes} B, at or over "
                f"the kernel's 32-bit limit of {SLAB_BYTES_LIMIT} B"
            )
    if gqa_pack and seqlen_q * (seqlen_q * group + 2 * block_q) >= PACK_SPLIT_LIMIT:
        raise ValueError(
            f"gqa_pack at seqlen_q {seqlen_q}, group {group}, block_q {block_q}: the packed-row split (x * magic) >> 31 is exact "
            f"only while seqlen_q * (seqlen_q * group + 2 * block_q) < 2^31; beyond it a flat row maps to the wrong head. "
            "Use a build without gqa_pack."
        )
