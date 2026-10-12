# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx1201 attention kernel: which knobs, for which shape.

Split out of `flash_attn_func_gfx1201_aiw.py` so that the kernel file is about
correctness and this one is about speed. They change for different reasons and
on different evidence -- a number here moves when a sweep says so, and nothing
here can make the kernel *wrong*, only slow.

**Why a third module rather than folding this into the interface**, which is
where `sdpa-readability.md` asked for it: the kernel needs these values too. It
resolves `k_prefetch_dist=None` and friends into defaults, and the geometry
helpers below are called with values -- `VO_WIDTH`, `BLOCK_N` -- that only
exist part-way through the builder. Moving the policy into the interface, which
already imports the kernel, would be either a circular import or a second copy
of that derivation. A module importing neither is importable by both.

Two kinds of thing live here, and the distinction is worth keeping when
deciding whether a change needs a benchmark or a correctness argument:

- **Policy** -- `default_prefetch_dist`, `qk_shards`, `default_block_m`,
  `default_block_n`, and the tables they read. Measured *choices*. Any of them
  could be changed without making a build incorrect.
- **Geometry** -- `q_tiles_per_block`, `vo_chunks`, `resolve_shards`. These
  compute what is *legal* given a choice, out of LDS capacity and divisibility
  constraints. Changing one can make a build invalid, and they are here only
  because the policy functions call them.
"""

from dataclasses import dataclass, fields, replace

# ---------------------------------------------------------------------------
# Tuning policy
#
# These tables come from measured sweeps, not from a formula; see the comments
# on each. They are the *default* knob settings for a given head_dim -- every
# one can be overridden per build.
# ---------------------------------------------------------------------------

# The binding-prefetch schedule wins from head_dim 48 up; 16 and 32 still prefer
# distance 0 (35.4/60.7 with prefetch against 37.3/61.2 without). Measured
# B=1 H=8 N=4096 f16 non-causal.
_PREFETCH_MIN_HEAD_DIM = 48

_TARGET_WAVES = 8
_ROWS_PER_Q_TILE = 16

# Measured (shards, q_tiles) per head_dim. There is no clean formula: more
# waves helps while registers and LDS allow, and hurts the moment it pushes
# either over. Both effects are only visible after compiling, so these come
# from a sweep (B=1 H=8 N=4096 f16 non-causal, TFLOPS at 8 vs 16 waves):
#
#   hdim   8 waves  16 waves           chosen         why not more
#    48      79.5     76.3             8 waves
#    64      84.3     90.1            16 waves
#    80      91.4     95.7            16 waves
#    96      98.4    100.3            16 waves
#   128      97.5    102.0            16 waves
#   160      92.9     94.2            16 waves
#   192      99.5     90.6             8 waves        16 spills 8 registers
#   224      62.2     79.7 (2 shards) 16 waves        1 shard spills at any count
#   256      81.9      rej             8 waves        reduction buffer over LDS
#   384      53.9     63.8 (12 waves) 12 waves        16 waves rejected
#   512      53.3      rej             8 waves        reduction buffer over LDS
#
# The 16-wave rejections are LDS: the cross-shard reduction buffer scales with
# NUM_WAVES, and past 8 waves it no longer fits inside the V window it aliases.
#
# head_dim 256 re-swept after the softmax correction and LSE moved the budget.
# The old entry (2 shards, 4 q_tiles -> BLOCK_M 64, 8 waves) came from a sweep
# whose note reads "16 waves rejected: reduction buffer over LDS" -- true only
# *with* sharding, since that buffer exists only when QK_SHARDS > 1. Unsharded
# 16 waves was therefore never tried, and it is much better:
#
#   (shards, q_tiles)   BLOCK_M  waves | non-causal  causal
#   (2, 4)  <- old         64       8  |    74.1      71.8
#   (1, 16) <- new        256      16  |    92.6      74.9
#
# i.e. 1.25x non-causal and 1.05x causal, despite spilling 3 registers where
# the old config spilled none (241 -> 256 VGPRs). BLOCK_M 64 -> 256 quarters
# the workgroup count and the K/V traffic with it, which outweighs the spills
# by a wide margin -- a reminder that spill count is not a proxy for speed.
#
# 384 and 512 were swept the same way and are already at their optimum
# (384: 61.4/60.9 at (3,4), best alternative 54.2; 512: 52.2/44.6 at (4,2),
# best alternative 36.7). Only 256 was mistuned.
#
# Re-swept in full after gSWA and varlen, both of which moved register
# pressure. Nine of eleven entries were confirmed unchanged. Two were not, and
# **both were mistuned the same way head_dim 256 had been**: a configuration
# rejected on spills or LDS was never retried after the surrounding budget
# changed, so the table kept a choice whose reason had expired.
#
#   head_dim  was      now      non-causal  causal   (interleaved, 5 reps)
#   224       (2, 8)   (1, 16)     1.269     1.230
#   384       (3, 4)   (2, 4)      1.089     1.062
#
# head_dim 224's old note reads "1 shard spills at any count" -- true, and
# irrelevant: unsharded 16 waves is 27% faster despite the spills. That is the
# third time this file has recorded that spill count is not a proxy for speed,
# and the second time the lesson was written down and then not applied to a
# neighbouring entry.
#
# head_dim 160 screened as a win for (1, 8) and is *not* one: interleaved it
# is 0.948 non-causal and 1.000 causal. Kept at (1, 16). Recorded because the
# screen and the confirmation disagreed by more than the effect being measured
# -- a single undrifted measurement cannot resolve a 1% difference on this
# board, and two of the three candidates it produced were real.
#
# head_dim 128 re-swept after the P2 region split, which duplicates the loop
# body and pushed its VGPR count 149 -> 212. 16 q_tiles (BLOCK_M 256, 16 waves)
# no longer fits; 8 is 92.5/92.0 against 84.4/81.2 TFLOPS non-causal/causal.
# 48/80/96/160 were swept at the same time and are unchanged; 192 is already
# at its optimum.
_SHARDS_BY_HEAD_DIM = {224: 1, 256: 1, 384: 2}
_Q_TILES_BY_HEAD_DIM = {
    48: 8,
    64: 16,
    80: 16,
    96: 16,
    128: 8,
    160: 16,
    192: 8,
    224: 16,
    256: 16,
    384: 4,
    512: 2,
}

# BLOCK_M for the distance-0 schedule. Per-wave register use is dominated by
# two head_dim-proportional terms -- o_accs = VO_WIDTH/2 VGPRs and
# q_b_packs = head_dim/4 -- neither of which depends on BLOCK_M or BLOCK_N, so
# tile size is a weak lever on spilling. It is not a null one, though, because
# it changes the cooperative-load geometry. Measured spills / TFLOPS at
# B=1 H=8 N=4096 f16 non-causal:
#
#   head_dim  BM=128        BM=64        BM=32
#   160       0sp / 80.8    - / 74.4     - / 48.7
#   192      24sp / 67.2    - / 59.0     - / 37.9
#   224     101sp / 33.5   64sp / 50.9   38sp / compile-fail
#   256      36sp / 67.2   20sp / 46.9   53sp / 19.9
#
# BLOCK_M=128 wins everywhere except head_dim=224, whose awkward
# THREADS_PER_ROW_LOAD=14 spills 101 registers and loses ~40%.
_DIST0_BLOCK_M_BY_HEAD_DIM = {224: 64}

# Small head_dim is softmax-bound, not saturation-bound: the per-(row, KV tile)
# softmax cost does not scale with head_dim, so at head_dim 16 a wave does only
# 4 WMMA against 17 v_exp_f32 plus 2 barriers. A wider KV tile amortises the
# per-tile part of that -- the correction exp, the m/l update, the O rescale and
# the barriers -- across more KV columns. Measured B=1 H=8 N=4096 f16
# non-causal:
#
#   head_dim   BN=32   BN=64   BN=128
#   16          37.4    44.6    48.2
#   32          61.4    72.5    70.7
#
# Causal is excluded: its mask is unrolled into 16 explicitly named scalars, so
# it requires NUM_S_VALS == 16, i.e. BLOCK_N == 32. Widening it would mean
# rewriting that unroll (planned for the interval-decomposition work).
_DIST0_BLOCK_N_BY_HEAD_DIM_NONCAUSAL = {16: 128, 32: 64}


def default_prefetch_dist(head_dim):
    """K/V prefetch distance for this head_dim."""
    return 1 if head_dim >= _PREFETCH_MIN_HEAD_DIM else 0


def qk_shards(head_dim):
    """Waves cooperating on one Q row sub-tile at this head_dim."""
    return _SHARDS_BY_HEAD_DIM.get(head_dim, max(1, head_dim // 128))


def q_tiles_per_block(head_dim, shards=None):
    """Q row sub-tiles per workgroup: TARGET_WAVES traded against the shard count.

    The V transpose tiling does not have to divide evenly across the waves --
    tail tiles are guarded at the LDS store -- so this is otherwise free.
    """
    shards = qk_shards(head_dim) if shards is None else shards
    return _Q_TILES_BY_HEAD_DIM.get(head_dim, max(1, _TARGET_WAVES // shards))


def vo_chunks(vo_width, block_n, shards, pad=4):
    """V staging passes needed to keep the *padded* K+V tile inside 64 KiB.

    Sharding V/O across waves means every wave's slice is live at once, so the
    full width of V^T would have to be resident -- 69888 B at 512 columns, over
    the cap. Staging V in `nc` passes makes only vo_width/nc columns resident,
    which restores the padding and with it conflict-free LDS. Costs one extra
    barrier pair per extra pass. Returns 1 whenever one pass fits.
    """
    for nc in (1, 2, 4, 8):
        if vo_width % nc:
            continue
        cols = vo_width // nc
        if cols % (shards * 16):  # each wave needs whole 16-col chunks
            continue
        if block_n * (vo_width + pad) * 2 + cols * (block_n + pad) * 2 <= 65536:
            return nc
    return 1


def resolve_shards(head_dim, vo_width, block_n, want=None):
    """Largest valid shard count no greater than the policy's preference.

    The policy table keys off head_dim alone, but the shard count also has to
    divide the *V/O window* into whole 16-column chunks. Those two constraints
    only diverge when the window is narrower than head_dim: head_dim 384 wants
    3 shards, which splits a 128-wide window into 42-column slices and is
    rejected downstream. Walk down from the preference to the first count that
    satisfies both, rather than failing the build.
    """
    want = qk_shards(head_dim) if want is None else want
    for s in range(want, 0, -1):
        if head_dim % s or (head_dim // s) % 16:
            continue
        cols = vo_width // vo_chunks(vo_width, block_n, s)
        if cols % s or (cols // s) % 16:
            continue
        return s
    return 1


def default_block_m(head_dim, prefetch_dist=None):
    """BLOCK_M for this head_dim under the default schedule."""
    dist = default_prefetch_dist(head_dim) if prefetch_dist is None else prefetch_dist
    if dist == 0:
        return _DIST0_BLOCK_M_BY_HEAD_DIM.get(head_dim, 128)
    return _ROWS_PER_Q_TILE * q_tiles_per_block(head_dim)


def default_block_n(head_dim, causal, prefetch_dist=None):
    """BLOCK_N for this head_dim under the default schedule."""
    dist = default_prefetch_dist(head_dim) if prefetch_dist is None else prefetch_dist
    if dist == 0 and not causal:
        return _DIST0_BLOCK_N_BY_HEAD_DIM_NONCAUSAL.get(head_dim, 32)
    return 32


# ---------------------------------------------------------------------------
# Interface-side policy: how the public options map onto the builder's knobs.
#
# Moved here from `flash_attn_func_gfx1201_interface.py` so that all tuning
# lives in one file. These differ from the block above in *who* they answer
# for: the functions above pick a default when the caller said nothing, these
# turn what the caller did say into a knob dict.
# ---------------------------------------------------------------------------

# The binding-prefetch kernel wins from head_dim 48 up; 16 and 32 still prefer
# the baseline (bp 35.4/60.7 against 37.3/61.2). Measured B=1 H=8 N=4096 f16
# non-causal. Below the threshold the tiles are small enough that bp's extra
# register-carried prefetch buys nothing.
_BP_MIN_HEAD_DIM = 48


# Q row sub-tiles per wave. At 2, each wave owns two of them, so one K or V
# operand feeds two WMMAs. BLOCK_M is *unchanged* -- the kernel divides
# Q_TILES_PER_BLOCK by this knob, see its comment -- so the grid is identical
# and what halves is the wave count per workgroup. The trade is therefore
# operand reuse per wave against waves available to hide latency with, which is
# why the answer depends on the shape and not only on the width.
#
# Ratio of row_subtiles=2 over 1, f16, B=1 H=8 across N in {512, 1024, 2048,
# 4096, 8192} plus (B=4, N=512) and (B=2, N=1024), best of 5 alternating reps:
#
#   head_dim  causal    worst   best  |  head_dim  causal    worst   best
#   --------  ------    -----  -----  |  --------  ------    -----  -----
#   16        yes        0.96   1.29  |  64        either     0.78   1.20
#   16        no         0.90   1.10  |  80        yes        0.98   1.07
#   32        yes        0.92   1.33  |  80        no         0.99   1.05
#   32        no         0.72   1.24  |  96        either     0.79   1.00
#   48        either     0.94   1.06  |
#
# head_dim 80 is the only width that wins at *every* shape measured, so it is
# the only one that takes two row sub-tiles by default. The rest are not left on
# the table by oversight. Their gains are real but jagged in N -- head_dim 32
# causal is 0.92 at N=2048 and 1.33 at N=4096 -- and that is not a threshold to
# key on: a policy fitted to that curve at one (B, H) would regress on every
# shape it was not fitted to. `variant="m32"` remains for callers who have
# measured their own shape. The full table is in `sdpa_lore_gfx1201.md`.
_ROW_SUBTILES_2_HEAD_DIMS = frozenset({80})

# Two row sub-tiles double the per-wave o_accs + q_b_packs + s_accs. Past this
# width that crosses the 256-VGPR cap and spills (-27% measured at head_dim 128).
_ROW_SUBTILES_2_MAX_HEAD_DIM = 80


# How the kernel forms the clamped KV address: hoist `row * stride_seq` out of
# the KV loop, or recompute it per access. See `kv_off` in the builder for the
# two forms.
#
# It is a pure hoist-versus-rematerialise trade, and it exists as a knob only
# because the answer is not monotone in the width. Hoisting removes a 64-bit
# multiply per load per iteration -- at BLOCK_DMODEL 192 the loop body drops
# from 648 instructions to 585 -- but it keeps one 64-bit value live per
# cooperative load for the whole loop.
#
# Full ladder, hoisted over recomputed, f16, B=1 H=8, N in {1024, 4096}, both
# masking modes, best of 5 alternating reps:
#
#   BLOCK_DMODEL   worst   best  |  BLOCK_DMODEL   worst   best
#   ------------   -----  -----  |  ------------   -----  -----
#   16              0.99   1.03  |  160 non-causal  1.03   1.05
#   32              1.02   1.04  |  160 causal      0.85   0.85
#   48 at N=1024    1.02   1.03  |  192             1.01   1.41
#   48 at N=4096    0.95   0.98  |  224             0.89   0.93
#   64              1.00   1.05  |  256             0.89   0.91
#   80              1.03   1.05  |  384             0.59   0.67
#   96              1.01   1.04  |  512             1.14   1.24
#   128             1.03   1.05  |
#
# So: hoist up to 192 and again at 512, with two exceptions below.
#
# 48 is the one width that splits on N rather than on the build: it gains 2-3%
# at N=1024 and loses 3-4% at N=4096, reproduced over three ladder runs. N is a
# runtime extent, so no knob can separate them, and the loss lands where the
# kernel spends sixteen times as long. It recomputes.
#
# 512 looks anomalous and is not. Scratch per lane, recomputed then hoisted:
#
#   BLOCK_DMODEL   192      256      384      512
#   scratch (B)    164->44  104->160 164->328 140->124
#   ratio          1.35     0.89     0.59     1.19
#
# The sign of the scratch change predicts the sign of the ratio at every width
# on the ladder, 512 included. Fewer instructions in the loop is what the hoist
# buys; whether that survives the extra live values is decided by the register
# allocator, one width at a time. Re-measure rather than extending this set by
# analogy -- and dump the ISA against the *recomputed* build, not against a
# pre-64-bit one, which spills differently for unrelated reasons.
_KV_ADDR_HOIST_HEAD_DIMS = frozenset({16, 32, 64, 80, 96, 128, 160, 192, 512})

# BLOCK_DMODEL 160 is the one width where the two masking modes disagree:
# +2 to +4% non-causal, -15% causal. Causal builds take a different BLOCK_N
# (`default_block_n` keys on it), so they are a different register problem
# wearing the same width, and this is the same kind of exception that function
# already makes. No other width on the ladder needs one.
_KV_ADDR_HOIST_CAUSAL_EXCLUDED = frozenset({160})


def _kv_addr_hoist(head_dim: int, causal: bool) -> bool:
    """Whether the KV address hoists `row * stride_seq` out of the loop."""
    if causal and head_dim in _KV_ADDR_HOIST_CAUSAL_EXCLUDED:
        return False
    return head_dim in _KV_ADDR_HOIST_HEAD_DIMS


def _row_subtiles(head_dim: int, variant: str) -> int:
    """Q row sub-tiles per wave: the tuning policy, or 2 where the caller forces it."""
    if variant == "m32":
        if head_dim > _ROW_SUBTILES_2_MAX_HEAD_DIM:
            raise ValueError(
                f"variant='m32' (row_subtiles=2) requires head_dim <= "
                f"{_ROW_SUBTILES_2_MAX_HEAD_DIM}, got {head_dim}"
            )
        return 2
    return 2 if head_dim in _ROW_SUBTILES_2_HEAD_DIMS else 1


def _use_bp(head_dim: int, use_binding_prefetch: bool, variant: str) -> bool:
    """Whether K is staged one tile ahead.

    Two row sub-tiles require it, so that case answers here rather than being
    patched up afterwards. Keeping the coupling in one place matters beyond the
    knob dict: this result is also an `lru_cache` key for `_get_kernel` and the
    prefetch distance the caller feeds to `aiw_block_m`, and the three
    disagreeing is the kind of thing that stays latent until some head_dim makes
    `block_m` depend on the distance.
    """
    if _row_subtiles(head_dim, variant) == 2:
        return True
    return use_binding_prefetch or head_dim >= _BP_MIN_HEAD_DIM


def _aiw_knobs(head_dim: int, use_bp: bool, variant: str) -> dict:
    """Map the public options onto aiw's knob space.

    ``use_bp`` is the *resolved* prefetch decision from `_use_bp`, not the
    caller's raw ``use_binding_prefetch`` flag.
    """
    dist = 1 if use_bp else default_prefetch_dist(head_dim)
    return {
        "k_prefetch_dist": dist,
        "v_lds_layout": "transposed" if dist else "row",
        "row_subtiles": _row_subtiles(head_dim, variant),
    }


# ---------------------------------------------------------------------------
# Which tile widths exist at all.
#
# Also tuning: the ladder decides how many kernels get compiled and how much
# a head_dim between rungs over-computes. `_MAX_HEAD_DIM` is derived from it
# rather than written twice -- they were both 512 in two files, which is one
# edit away from disagreeing.
# ---------------------------------------------------------------------------

# The compiled tile widths. `head_dim` is rounded up to the smallest of these
# that covers it; the real extent then rides along as a runtime argument and the
# kernel masks the difference. Mirrors AOTriton's `block_dmodel_values()`, which
# is the same list minus 384.
_BLOCK_DMODEL_LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512)


def _round_to_ladder(head_dim: int) -> int:
    """Smallest compiled tile width covering `head_dim`."""
    for w in _BLOCK_DMODEL_LADDER:
        if w >= head_dim:
            return w
    raise ValueError(
        f"head_dim {head_dim} exceeds the largest compiled tile "
        f"({_BLOCK_DMODEL_LADDER[-1]})"
    )


MAX_HEAD_DIM = _BLOCK_DMODEL_LADDER[-1]


# ---------------------------------------------------------------------------
# The two halves of a build request.
#
# Split on *who decides*, which is the distinction the 26-parameter signature
# they replace could not express: a caller states a problem, the tuning policy
# answers with a schedule. Keeping them apart is what makes "the tuning module
# is the only producer of a schedule" a checkable property rather than a
# convention -- a single blob would let a caller set `block_m` while claiming
# to have asked for a default, and nothing would notice.
#
# Both frozen: a schedule is used as part of a build-cache key, and a mutable
# key is a bug waiting for a second caller.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FmhaInputMetadata:
    """What to compute. Set by the caller; never by policy."""

    num_heads: int
    head_dim: int
    causal: bool = True
    dtype_str: str = "bf16"
    head_dim_v: int | None = None
    sm_scale: float | None = None
    causal_type: int | None = None
    bias: bool = False
    # Modification 3 (hipDNN vendoring): an attention sink, one logit per query head.
    sink: bool = False
    dropout: bool = False
    philox_width: int | None = None


@dataclass(frozen=True)
class FmhaKnobs:
    """How to compute it. Every field `None` means "policy decides".

    `None` rather than a literal default on purpose: it is the only way
    `resolve_knobs` can tell "the caller wants 1" from "the caller did not
    say", and that difference is the whole point of the overrides argument.
    """

    # Compile-time widths, baked into the binary. BLOCK_DMODEL is the maximum
    # head_dim the resulting hsaco can serve; the *real* extent travels as a
    # runtime argument and may be smaller, which is what `padded_head` records.
    # They are knobs and not input metadata because only the ladder decides
    # them, and the ladder is this module's.
    block_dmodel: int | None = None
    block_dmodel_v: int | None = None
    d_offset: int | None = None

    block_m: int | None = None
    block_n: int | None = None
    k_prefetch_dist: int | None = None
    v_prefetch_dist: int | None = None
    v_lds_layout: str | None = None
    row_subtiles: int | None = None
    waves_per_eu: int | None = None
    kv_addr_hoist: bool | None = None

    # The two knobs `resolve_knobs` cannot fill. Both are derived from
    # NUM_WAVES, which the builder computes from the shard count and the tile
    # geometry, so filling them here would mean duplicating that derivation.
    # `None` means "derive it"; every other field is guaranteed resolved.
    flat_work_group_size: int | None = None
    shards: int | None = None
    sched_strategy: str | None = None

    # Three floating-point knobs, which looks like two too many until you see
    # that they act at three different levels:
    #
    #   fp_mode        the explicit `arith.FastMathFlags` set put on the
    #                  *softmax* operations -- "fast", "noninf" or "safe".
    #                  This is the one that has been measured; "fast" includes
    #                  `ninf`, which silently deleted the KV tail mask once.
    #   fast_fp_math   the *ambient* fastmath for traced operations that do not
    #                  carry their own flags (`effective_fastmath_hint`).
    #   unsafe_fp_math a ROCm *backend* option, `unsafe-math`, applied to the
    #                  whole compilation (`compiler/backends/rocm.py`).
    #
    # So they are per-op, per-op-default and whole-compilation respectively.
    # Only `fp_mode` is ever varied in practice; the other two have been True
    # for every build ever made here, which means their effect is measured only
    # in combination and collapsing them would need its own experiment.
    fp_mode: str | None = None
    denormals_are_zero: bool | None = None
    unsafe_fp_math: bool | None = None
    fast_fp_math: bool | None = None
    strides_constexpr: bool | None = None

    # Derived, not chosen: true when the caller's head_dim is not itself a
    # compiled tile width, so the kernel computes ceil-to-tile columns and
    # masks the difference. It lives here rather than with the caller's inputs
    # because deciding it needs the ladder -- the set of every width that
    # exists -- which is knowledge only this module has.
    padded_head: bool | None = None
    lpt_tile_order: bool | None = None
    unsafe_no_kv_clamp: bool | None = None
    path_tag: str | None = None

    def merge(self, other: "FmhaKnobs | None") -> "FmhaKnobs":
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {
            f.name: getattr(other, f.name)
            for f in fields(other)
            if getattr(other, f.name) is not None
        }
        return replace(self, **set_fields)


# Defaults for the knobs the policy has no opinion about -- they are the same
# for every shape, so they are literals here rather than functions.
_KNOBS_FALLBACK = FmhaKnobs(
    v_prefetch_dist=1,
    waves_per_eu=2,
    fp_mode="noninf",
    denormals_are_zero=True,
    unsafe_fp_math=True,
    fast_fp_math=True,
    strides_constexpr=False,
    padded_head=False,
    d_offset=0,
    lpt_tile_order=True,
    unsafe_no_kv_clamp=False,
    path_tag="auto",
)


def resolve_knobs(
    meta: FmhaInputMetadata, overrides: "FmhaKnobs | None" = None
) -> FmhaKnobs:
    """The complete measured configuration for `meta`.

    The ordering below is the reason this function exists. `k_prefetch_dist`
    feeds `block_n`, and `row_subtiles` feeds `k_prefetch_dist`, so a caller
    assembling a schedule by hand has to know the dependency order to get the
    same answer -- and until now the interface was the thing that knew it.

    `overrides` is applied *first*, so a pinned knob participates in deriving
    the ones downstream of it rather than being stamped on afterwards. Pinning
    `row_subtiles=2` therefore also forces the prefetch distance it requires,
    which is what a caller asking for two row sub-tiles means.
    """
    s = _KNOBS_FALLBACK.merge(overrides)
    # The compiled widths come first: everything below is keyed on the width
    # baked into the binary, not on the caller's real extent. They coincide for
    # a direct builder call, whose head_dim must already be a tile width, and
    # differ whenever `plan` has rounded one up.
    if s.block_dmodel is None:
        s = replace(s, block_dmodel=meta.head_dim)
    if s.block_dmodel_v is None:
        s = replace(s, block_dmodel_v=meta.head_dim_v)
    hd = s.block_dmodel

    if s.row_subtiles is None:
        s = replace(s, row_subtiles=2 if hd in _ROW_SUBTILES_2_HEAD_DIMS else 1)
    if s.row_subtiles == 2 and hd > _ROW_SUBTILES_2_MAX_HEAD_DIM:
        raise ValueError(
            f"row_subtiles=2 requires head_dim <= {_ROW_SUBTILES_2_MAX_HEAD_DIM}, "
            f"got {hd}"
        )
    if s.kv_addr_hoist is None:
        s = replace(s, kv_addr_hoist=_kv_addr_hoist(hd, meta.causal))
    if s.k_prefetch_dist is None:
        # Two row sub-tiles require the prefetched, transposed V layout.
        s = replace(
            s,
            k_prefetch_dist=(
                1
                if s.row_subtiles == 2 or hd >= _BP_MIN_HEAD_DIM
                else default_prefetch_dist(hd)
            ),
        )
    if s.v_lds_layout is None:
        s = replace(s, v_lds_layout="transposed" if s.k_prefetch_dist else "row")
    if s.block_n is None:
        s = replace(s, block_n=default_block_n(hd, meta.causal, s.k_prefetch_dist))
    if s.block_dmodel_v is None:
        # Defaults to the full width: a build that does not slice the V/output
        # side computes all of it.
        s = replace(s, block_dmodel_v=s.block_dmodel)
    if s.block_m is None:
        # Always resolved, so there is exactly one BLOCK_M in the system. It
        # used to be filled only on the distance-0 path, with the host
        # separately computing its own copy for sequence padding and the grid
        # -- two values that agreed only because `default_block_m` happens not
        # to depend on the prefetch distance at any current head_dim.
        # BLOCK_M is invariant to row_subtiles by construction; see the
        # Q_TILES_PER_BLOCK comment in the kernel.
        s = replace(s, block_m=default_block_m(hd, s.k_prefetch_dist))
    return s


@dataclass(frozen=True)
class FmhaPlan:
    """Everything a host needs for one build, from one call.

    The two are not independent -- the knobs follow from the metadata by way
    of the ladder -- which is why they are produced together and why a caller
    should never assemble them separately.
    """

    meta: FmhaInputMetadata
    knobs: FmhaKnobs


def plan(request: FmhaInputMetadata, overrides: FmhaKnobs | None = None) -> FmhaPlan:
    """The one entry point into tuning: the caller's inputs in, a full plan out.

    `request` is returned as `FmhaPlan.meta` **unchanged** -- it is what the
    caller has, and nothing here may rewrite it. The rounding up to a compiled
    tile lands in `FmhaPlan.knobs.block_dmodel`, which is the width baked into
    the binary and therefore the maximum head_dim it can serve;
    `knobs.padded_head` records whether the two differ. No
    caller needs the ladder, the maximum, or the block-size tables -- every one
    of those used to be imported by the interface, which meant the ordering
    between them lived at the call site.

    `padded_head` is a knob rather than metadata because deciding it requires
    the ladder, which is this module's knowledge and not the caller's. Any
    value passed in `overrides` is overwritten: the rounding decides it.

    Raises for a head_dim past the largest compiled tile.
    """
    head_dim = request.head_dim
    if head_dim < 1 or head_dim > MAX_HEAD_DIM:
        raise ValueError(
            f"kernel requires 1 <= head_dim <= {MAX_HEAD_DIM}, got {head_dim}"
        )
    block_dmodel = _round_to_ladder(head_dim)
    knobs = replace(
        resolve_knobs(
            request,
            (overrides or FmhaKnobs()).merge(FmhaKnobs(block_dmodel=block_dmodel)),
        ),
        padded_head=block_dmodel != head_dim,
    )
    return FmhaPlan(request, knobs)
