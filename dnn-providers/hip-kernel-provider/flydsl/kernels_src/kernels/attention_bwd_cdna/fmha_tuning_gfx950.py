# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Tuning policy for the gfx950 parity attention kernel: which knobs, for which shape.

Split from the kernel for the usual reason: the kernel file is about
correctness and this one is about speed. A number here moves when a sweep says
so, and nothing here can make a build *wrong*, only slow.

--- One object, one call (R1) ---------------------------------------------

    knobs = fmha_knobs(arch, **overrides)   # factory -> arch-specific subclass
    cfg   = knobs.resolve(meta)             # fully resolved, ready to build

**Knobs and traits were always the same thing at two points on one axis.**
Knobs are the *partially resolved* form -- `None` means "policy decides" --
and the dualwave traits are the *fully resolved* one. Keeping them as separate
types with a converter between them meant every feature knob was declared
twice and threaded through a function call, which is why `cross_seqlen` used to
travel as a keyword argument *beside* the knob object instead of inside it.
(It has since stopped being a knob at all -- see `Gfx950Knobs`.)

So `resolve` is a method, and it owns the whole derivation: the ladder, the
padded-head decision, the wave geometry, and the traits the kernel is built
from. `FmhaInputMetadata` stays arch-neutral -- it is *what to compute*, and
no arch has an opinion about it. `FmhaKnobs` is *how*, and is subclassed per
arch.

Two kinds of thing still live here, and the distinction decides whether a
change needs a benchmark or a correctness argument:

- **Policy** -- `LADDER`, the fallbacks, `_with_wave_geometry`. Measured
  choices; any of them could change without making a build incorrect.
- **Geometry** -- `tile_width_for`, `staging_shape`, and the divisibility rules
  `resolve` enforces. These compute what is *legal*, and changing one can make
  a build invalid.

**The ladder is the design, and the granule is a knob within it.** A head_dim
between two rungs is served by compiling the next rung up and passing the real
extent as a runtime argument, which is what `padded_head` records. What decides
the rungs is the *staging granule* -- how many D elements one DMA issue covers
-- and that was long assumed to be 64 because the production kernel is built
that way. It is not a constant: `_with_wave_geometry` picks it per family, and
`staging_shape` states the rule it has to satisfy. Only `PV_MFMA_N` is a real
floor, and it is an instruction limit rather than a staging one.
"""

from dataclasses import dataclass, fields, replace

from kernels.attention_bwd_cdna.fmha_traits_gfx950 import make_traits

__all__ = [
    "LADDER",
    "LADDER_PLANNED",
    "FmhaInputMetadata",
    "FmhaKnobs",
    "Gfx950Knobs",
    "fmha_knobs",
    "tile_width_for",
]

# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

# Compiled tile widths, all measured at `B=4 H=8 S=4096` bf16 non-causal on
# real FLOPs (not the padded tile):
#
#   tile  waves  BLOCK_M  gran  stages  shards   AGPR  spills    LDS   TFLOP/s
#     32    4      128     32     1       1        -     0     17 KB     618
#     64    8      256     64     1       1        0     0     33 KB     889
#    128    8      256     64     1       1        0     0     67 KB    1117
#    160    4      128     32     1       1        -     0     83 KB     917
#    192    4      128     64     1       1      174     0    100 KB     936
#    224    4      128     32     1       1        -     0    116 KB     939
#    256    4      128     64     1       1      192     0    133 KB     940
#    384    4      128     64     2       1      112     0    100 KB     803
#    512    4       64     64     2       2       91     0    133 KB     479
#
# Three families. **A** (granule 64, 8 waves) serves 64 and 128; **B/S**
# (4 waves) serve the rest up to 256, B (granule 64) at 192 and 256 and S
# (granule 32) at every width that is not a multiple of 64; **W** stages the D axis (`D_STAGES`) and shards it
# (`VO_SHARDS`) for 384 and 512; the forward body that does this is not in
# this tree.
#
# The break at 384 is not a tuning preference, it is LDS: two KV tiles in
# flight need `2 * BLOCK_N * head_dim * ~8.3 B`, which is 199.5 KB at 384 and
# 266 KB at 512 against a 163840 B cap.
#
# 192 and 256 used to spill in the hundreds and produce non-deterministic NaN.
# Both had the same cause, and it was not register pressure: `_ds_read_tr16_b64_imm`
# emitted the LDS transpose read as inline asm, which `SIInsertWaitcnts` cannot
# see, so no `s_waitcnt lgkmcnt` was inserted before uses. Below 128 nothing read
# the destination before the kernel's own cluster-boundary wait; at 192 the
# allocator starts using AGPRs and places `v_accvgpr_write` copies ahead of that
# wait. The ROCDL op, which the compiler does track, does not have the problem.
#
# So the register file was not the binding constraint at 192/256 after all.
# It becomes one at 384 and 512, where a wave's Q (128 VGPR) and O (256 VGPR,
# the whole AGPR file) do not leave room for the K/V working set -- which is
# what `VO_SHARDS` answers. See `_with_wave_geometry` for the wave-count
# measurement, which is worth 1.6x on its own.
#
# The 32 and 96 rungs are built as granule-32 (family S) tiles. `LADDER_PLANNED`
# is consulted only after `LADDER` misses, and holds the widths that are
# designed but not yet built (none today).
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)
LADDER_PLANNED = ()

# The D-axis staging granule: how many bf16 elements of one token a single DMA
# issue covers. **Not a constant, and not 64 by necessity** -- see
# `_with_wave_geometry`. A wave moves 512 elements per issue (64 lanes x 8), so
# the granule and BLOCK_N together decide how many lines a KV tile occupies and
# how many issues each wave makes:
#
#     tokens per issue = 512 / granule
#     lines            = BLOCK_N / tokens_per_issue
#     issues per wave  = lines / NUM_WAVES        (must be a positive integer)
#
# Granule 64 with BLOCK_N 64 is one point in that space, not the only one.
DEFAULT_HEAD_DIM_GRANULE = 64

# ---------------------------------------------------------------------------
# The grid axis order, as a knob
# ---------------------------------------------------------------------------
#
# **A consumer that computes the launch grid itself needs this, and cannot see
# it.** AOTriton dispatches the hsaco directly and computes the grid in C++; it
# never runs our `@flyc.jit` launcher, so every decision the launcher makes
# about which quantity lands on which grid axis has to travel in the knob set.
# The alternative is an `if (arch == ...)` on their side, which silently rots
# the next time this is re-measured.
#
# The mapping *is* the interface, so it is written here rather than inferred:
#
#   0  HEAD_FASTEST  grid = (head, tile_blocks, batch_or_seq)
#   1  TILE_FASTEST  grid = (tile_blocks, head, batch_or_seq)
#
# "head" is the q head for the forward and dQ, the **kv** head for dK/dV, which
# is what the GQA fold made it. All three gfx950 launchers are HEAD_FASTEST;
# gfx1201's dK/dV is TILE_FASTEST, chosen there for causal load balance.
#
# gfx950 measured the other order and rejected it: KV-fastest was 12-15% slower
# at every rung, because MI355X's eight XCDs make this an L2-locality lever
# rather than a duration-spreading one. That measurement is why the value is a
# recorded knob and not a constant someone can quietly re-derive.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1

# The PV MFMA is `v_mfma_f32_32x32x16`, whose output is 32 D columns wide, so
# `D_CHUNKS = head_dim / 32` cannot go below 1. **This is what makes a granule
# of 16 impossible**, and it is an instruction limit rather than a staging one:
# the staging is perfectly regular at granule 16 (BLOCK_N 256 over 8 waves is
# one issue per wave), but the output accumulator cannot be narrower than 32.
# Serving head_dim 16 natively would need `v_mfma_f32_16x16x16`, whose
# accumulator is v4f32 rather than v16f32 -- a different register layout
# through every helper, i.e. its own family.
PV_MFMA_N = 32

# Hardware shape of one DMA issue, used by `staging_shape`.
WARP_SIZE = 64
DMA_BYTES = 16
BF16_BYTES = 2


def rung_below(block_dmodel):
    """The widest rung strictly narrower than `block_dmodel`, or 0.

    `tile_width_for` rounds *up* to the first rung that fits, so a build it
    chose serves exactly the half-open range `(rung_below(R), R]`. That lower
    bound is what lets the kernel skip masking the D columns it knows are real
    (the `hdim_qk_floor` knob).

    It is a property of the ladder rather than of any one build, so it lives
    next to `LADDER`: adding a rung silently tightens every wider build's
    floor, and that is the correct behaviour.
    """
    below = [r for r in LADDER if r < block_dmodel]
    return max(below) if below else 0


def tile_width_for(head_dim):
    """The compiled tile width serving `head_dim`, or raise saying why not.

    Rounds *up* to the next rung. head_dim 40 compiles as a 64-wide build with
    `hdim_qk=40`; head_dim 129 is served by the 160 rung.
    """
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    for rung in LADDER:
        if head_dim <= rung:
            return rung
    for rung in LADDER_PLANNED:
        if head_dim <= rung:
            raise NotImplementedError(
                f"head_dim {head_dim} needs the {rung}-wide tile, which is designed but not built. "
                f"Built rungs: {LADDER}."
            )
    widest = max(LADDER + LADDER_PLANNED)
    raise ValueError(f"head_dim {head_dim} exceeds the widest tile ({widest})")


# ---------------------------------------------------------------------------
# What to compute
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FmhaInputMetadata:
    """What to compute. Set by the caller; never by policy, and never by arch."""

    num_heads: int
    head_dim: int
    causal: bool = True
    dtype_str: str = "bf16"
    num_kv_heads: int | None = None
    head_dim_v: int | None = None
    sm_scale: float | None = None

    # P3. Whether this build reads the runtime `window_left`/`window_right`
    # pair and applies a left bound as well as the causal right one --
    # AOTriton's `CAUSAL_TYPE == 3`. The bounds are runtime arguments, so they
    # are not here; only the decision to compile for them is. Requires
    # `causal`, since a window is a left bound *on top of* the causal one.
    window: bool = False

    # P5. Whether this build reads a (B, H, Sq, Sk) bias matrix. Mutually
    # exclusive with `causal`; see `make_traits`.
    bias: bool = False

    # P6. Whether this build applies a philox dropout mask to P. The rate and
    # the seed are runtime arguments; only the decision to compile for them is
    # here.
    dropout: bool = False


# ---------------------------------------------------------------------------
# How to compute it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FmhaKnobs:
    """How to compute it, arch-neutral part. `None` means "policy decides".

    `None` rather than a literal default on purpose: it is the only way
    `resolve` can tell "the caller wants 1" from "the caller did not say", and
    that difference is the whole point of the overrides.

    Subclass per arch and implement `resolve`. Nothing may be added here that
    only one arch understands -- that is what the subclass is for.
    """

    # Compile-time widths. `block_dmodel` is the tile the hsaco serves; the
    # *real* extent travels as a runtime argument, which `padded_head` records.
    block_dmodel: int | None = None
    block_dmodel_v: int | None = None
    padded_head: bool | None = None

    # A compile-time *lower* bound on the runtime `hdim_qk`, exclusive. The
    # build promises nothing about hdim at or below it, and the dispatcher
    # rejects a call that violates it, so the kernel may treat D columns below
    # it as real data and skip masking them.
    #
    # Derived, not a taste knob, but pinnable to 0 to force the old
    # mask-everything behaviour if a floor is ever suspected of hiding a bug.
    hdim_qk_floor: int | None = None

    # There is no `hdim_mode`. A `"runtime_qk_steps"` mode used to be declared
    # here -- shorten the QK reduction to `ceil(hdim_qk/16)` MFMA K-steps at
    # runtime, against `"zero_fill"`'s tile-shaped count -- and it was **never
    # implemented**: the flag threaded through three files into
    # `ctx.RUNTIME_QK_STEPS`, which nothing read. The two modes produced
    # byte-identical ISA.
    #
    # Deleted rather than implemented, because measurement says it was aimed at
    # the wrong term. At head_dim 32 in the 64 tile the padded build has the
    # same 96 MFMAs as an unpadded 64 -- the 2x waste is real -- but it also has
    # 565 `v_cndmask` against 145 and 3584 instructions against 2562. The
    # masking, not the MFMA count, is what makes head_dim 32 *slower in
    # wall-clock* than head_dim 64 (185 us against 155). Shortening QK would cut
    # 8 of the 16 MFMAs per tile and leave that untouched.
    #
    # What removes that cost is a tile that does not pad: head_dim 32 runs in
    # the 32 tile (`tile_width_for(32)`, `LADDER`), which stages at granule 32
    # (family S, `_with_wave_geometry`).

    # Whether the strides fold to literals. False is the parity ABI; True
    # exists so the fast path can be shown to emit unchanged code.
    strides_constexpr: bool | None = None

    return_lse: bool | None = None

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

    def resolve(self, meta: FmhaInputMetadata):
        raise NotImplementedError(f"{type(self).__name__} does not implement resolve()")

    # -- resolution steps ------------------------------------------------
    #
    # Each `_with_*` takes knobs and returns knobs with some fields decided,
    # so `resolve` is a pipeline rather than a place where scattered tuples
    # get reassembled. Three things follow, and each removes a class of bug:
    #
    # - **The field names appear once.** Returning `(block_dmodel,
    #   block_dmodel_v, padded_head)` and `replace`-ing them at the call site
    #   spelled them all twice, in an order nothing checked.
    # - **The step order is the data dependency.** `_with_wave_geometry` reads
    #   `self.block_dmodel`, so it *must* run after `_with_widths`; it no
    #   longer takes it as an argument and cannot be handed a stale one.
    # - **`dataclasses.replace` preserves the subclass**, so a base-class step
    #   returns `Gfx950Knobs` and the pipeline can mix inherited and
    #   arch-specific steps freely.

    def _with_widths(self, meta):
        """Decide `block_dmodel`, `block_dmodel_v` and `padded_head`.

        Every arch has a ladder and a padded-head rule; only the ladder's
        contents differ, and those are `tile_width_for`'s. Kept on the base so
        a second arch cannot accidentally decide `padded_head` by another rule.
        """
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

        # Only a *derived* tile carries the ladder's guarantee. A caller that
        # pins `block_dmodel` may pin 256 for head_dim 64, so there is no floor
        # to claim and the kernel masks everything, as before.
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

        return replace(
            self,
            block_dmodel=block_dmodel,
            block_dmodel_v=block_dmodel_v,
            padded_head=bool(padded_head),
            hdim_qk_floor=int(hdim_qk_floor),
        )


@dataclass(frozen=True)
class Gfx950Knobs(FmhaKnobs):
    """The gfx950 dual-wave schedule.

    `paged` and `num_kv_splits` are ordinary fields here, not arguments threaded
    past the object. That is the point of R1: before it, `cross_seqlen` was a
    keyword-only parameter on the builder, a `kwargs.pop` in the front end, and
    an argument to a converter -- three places to keep in step for one boolean.

    **There is no `varlen` field, and no `cross_seqlen` one.** The kernel
    decodes `varlen_bits` unconditionally, so a ragged batch is a property of
    the call rather than of the build; `cross_seqlen` went with it, because
    with the lengths arriving at runtime no build can know whether
    `seqlen_q == seqlen_k`. `make_traits` emits both as constants.
    """

    # Dual-wave schedule.
    waves_per_eu: int | None = None
    daz: bool | None = None
    lazy_rescale: bool | None = None
    setprio: bool | None = None
    stagger: bool | None = None
    lpt_tile_order: bool | None = None

    # Problem modes. Ordinary fields; see the class docstring.
    paged: bool | None = None
    kv_cache_layout: str | None = None
    num_kv_splits: int | None = None

    # Wave geometry. `None` means "the family for this tile width", which is
    # what `_with_wave_geometry` decides. Pinnable so a sweep can cross a
    # family boundary without editing the table.
    num_waves: int | None = None
    block_m: int | None = None
    block_n: int | None = None
    head_dim_granule: int | None = None

    # The two D-axis splits. Deliberately *not* part of the four-field geometry
    # pinning rule above: those four are one indivisible choice (a family),
    # while these are per-width policy that a sweep wants to vary on its own.
    # `None` means "the width decides", as everywhere else here.
    d_stages: int | None = None
    qk_shards: int | None = None
    vo_shards: int | None = None

    # The four V-layout constants whose formula family A cannot pin down; see
    # `make_traits`. Exposed so a sweep can settle them by measurement instead
    # of picking one. `None` keeps the default formula.
    v_half_wave: int | None = None
    v_n_group: int | None = None
    v_k_substep: int | None = None
    v_dc_in_pair: int | None = None

    # Which quantity lands on which grid axis; see `GRID_AXIS_HEAD_FASTEST`.
    # Set by `resolve`, never by a caller -- it is a property of the launcher,
    # not a choice. **Spelled in upper case against the convention here** because
    # the field name is the wire key: the C++ side reads it back as
    # `perf().get_int("GRID_AXIS_ORDER")`.
    GRID_AXIS_ORDER: int | None = None

    # There is deliberately no `traits` field. **A knob class is plain build
    # options, every value a scalar**, which is what gfx1201's `FmhaKnobs` is
    # and what makes the object serialisable: AOTriton records the knob set
    # beside each compiled hsaco in a flat `k=v` wire format, and a nested
    # dataclass cannot be rendered into it.
    #
    # The traits are still derived here, by `build_traits` below, and `resolve`
    # still calls it -- so a configuration that cannot produce valid traits is
    # rejected at tuning time rather than at a kernel address. What changed is
    # only that the result is handed to the builder on request instead of being
    # carried on the object.

    def resolve(self, meta: FmhaInputMetadata) -> "Gfx950Knobs":
        """The complete build configuration for `meta`.

        Subsumes what used to be `resolve_knobs` *and* `make_parity_traits`.
        Idempotent: resolving an already-resolved object re-derives the same
        answer, since every derived field is recomputed from `meta` and the
        pinned fields rather than read back.

        The last step builds the traits and **throws the result away**, keeping
        only its verdict. That is not waste: every check in `build_traits` --
        the LDS cap, the shard split, the rows-per-wave ceiling -- reports which
        knob to move, and reporting it here means a bad configuration fails at
        `resolve` rather than at a kernel address. The builder calls
        `build_traits` again for the object itself; it is a dataclass
        construction, not a compile.
        """
        return (
            _GFX950_FALLBACK.merge(self)
            ._checked_modes()
            ._with_widths(meta)
            ._with_wave_geometry()
            ._with_occupancy_target(meta)
            ._checked_against_traits(meta)
        )

    def _with_occupancy_target(self, meta):
        """Ask for two waves per EU only where two workgroups can exist.

        LOCAL DIVERGENCE FROM UPSTREAM, which asks for 2 unconditionally.
        Retire when upstream derives the target from the build's own LDS.

        `waves_per_eu` used to be a flat 2 in `_GFX950_FALLBACK`, among the
        defaults with no shape-dependent opinion. It needs one. A workgroup
        here is 256 threads = 4 waves = one wave on each of the CU's 4 SIMDs,
        so two waves per EU means two *co-resident workgroups*, and LDS decides
        whether that is even possible before the register allocator is
        consulted. Past head_dim 128 it is not close -- head_dim 160 stages
        85120 B, so the pair wants 170240 B against a 163840 B cap -- and the
        backend says so once per build:

            failed to meet occupancy target given by 'amdgpu-waves-per-eu'
            in 'flyc_attn_fwd': desired occupancy was 2, final occupancy is 1

        which was 120 of the 216 shipped `flyc_attn_fwd` builds.

        **Dropping a refused request costs nothing, which was checked rather
        than assumed.** An unmet hint is still a register budget for the
        scheduler, so the 120 refused builds were built both ways and compared:
        118 byte-identical, 2 differing. The 2 are noise -- rebuilding all 216
        at *unchanged* settings also yields 2 differing, this backend is not
        bit-reproducible, and one of the two kernels is the same one both times
        (head_dim 256 bf16 causal+dropout, which alternates between two forms
        at identical register counts and identical size). So the wpe-vs-noise
        signal at the refused rungs is 2 against a floor of 2: nothing.
        Measured, every rung ties within 0.3% at `S=4096` on both a dense and a
        causal+dropout arm.

        The corollary is worth stating, since it limits what this method can
        show: byte-identity is strong evidence of no change -- nondeterminism
        creates differences, it cannot manufacture 118 coincidental matches --
        but a handful of *differences* proves nothing on its own and has to be
        read against the 2/216 floor.

        **The rule is LDS rather than a head_dim list**, because the granule,
        `BLOCK_N` and `d_stages` all move the staging footprint and a list
        would silently stop matching when one of them does. The probe builds
        traits at 1 purely to read `LDS_KV_TOTAL_SIZE`; `waves_per_eu` is a
        scheduling hint and does not feed the staging shape, so the answer does
        not depend on the value used to ask.

        **Where the hint is granted it is doing real work and stays.** head_dim
        96 is the case, and it is the one result here well clear of that noise
        floor: 17 of its 24 builds change when the request is dropped, 13 of
        them raising VGPRs -- 214 to 218, 224 to 245, 229 to 249 -- all in the
        same direction, which is what nondeterminism does not do. At 51072 B
        two workgroups do fit, so the budget is real there, and it is what
        holds the allocator under the 256 that keeps both resident. head_dim
        32, 64 and 128 are byte-identical either way; they never approach the
        ceiling, so the budget never binds.
        """
        if self.waves_per_eu is not None:
            return self  # explicitly pinned -- a sweep's override outranks this
        traits = replace(self, waves_per_eu=1).build_traits(meta)
        lds_bytes = traits.LDS_KV_TOTAL_SIZE * traits.BF16_BYTES
        return replace(
            self, waves_per_eu=2 if 2 * lds_bytes <= self.LDS_CAP_BYTES else 1
        )

    def _checked_modes(self):
        """Reject mode combinations the kernel does not implement.

        First in the pipeline because none of it depends on a derived field --
        anything that fails here would fail whatever the ladder decided, so
        failing before the derivation keeps the message about the caller's
        input rather than about something computed from it.
        """
        if self.num_kv_splits > 1:
            # The kernel decodes `varlen_bits` unconditionally, so every build
            # is a ragged-batch build -- and SPLITK's active guard is
            # `split_nonempty`, which was derived over a rectangular batch and
            # says nothing about a sequence shorter than `max_seqlen`. Nothing
            # asks for the combination today (every AOT description pins 1), so
            # this rejects it rather than shipping a guard that has not been
            # re-derived.
            raise ValueError(
                f"num_kv_splits must be 1 on gfx950, got {self.num_kv_splits}: the split guard has not "
                "been derived for a ragged batch, and the varlen decode is no longer optional"
            )
        if self.kv_cache_layout not in ("linear", "vectorized"):
            raise ValueError(
                f"kv_cache_layout must be 'linear' or 'vectorized', got {self.kv_cache_layout!r}"
            )
        return self

    def _with_wave_geometry(self):
        """Decide the wave geometry and staging granule from the tile width.

        **The selection lives here, not in the traits constructor**, which is
        the concrete reason: the families differ only in
        these four numbers plus what they imply, and a constructor that
        hardcodes them cannot host more than one.

        Three families, and the granule is a *choice* in each rather than the
        constant it used to be:

        | family | tile width | waves | BLOCK_M | BLOCK_N | granule |
        |---|---|---|---|---|---|
        | S | not a multiple of 64 | 4 | 128 | 64 | 32 |
        | A | multiple of 64, <= 128 | 8 | 256 | 64 | 64 |
        | B | multiple of 64, 129..256 | 4 | 128 | 64 | 64 |

        **A** is measured saturated at head_dim 128 -- 248 of 256 VGPRs, zero
        spills -- so **B** halves the wave count to double the per-lane
        register file. **S** takes the widths whose tile is not a multiple of
        64 (32, 96, 160, 224): its D axis is staged in granule-32 DMA issues,
        and it runs the wide-tile wave count at every such width.

        Reads `self.block_dmodel`, so it runs after `_with_widths`. That
        ordering is the whole reason the width is no longer an argument: an
        argument can be stale, a field the previous step wrote cannot.
        """
        me = self._with_d_axis_splits()
        pinned = (me.num_waves, me.block_m, me.block_n, me.head_dim_granule)
        if all(x is not None for x in pinned):
            return me
        if any(x is not None for x in pinned):
            raise ValueError(
                f"pin num_waves, block_m, block_n and head_dim_granule together or not at all, got {pinned}"
            )
        if self.block_dmodel is None:
            raise ValueError(
                "_with_wave_geometry runs after _with_widths; block_dmodel is not resolved"
            )
        if me.block_dmodel % 64:
            return replace(
                me, num_waves=4, block_m=128, block_n=64, head_dim_granule=32
            )  # family S
        if me.block_dmodel <= 128:
            return replace(
                me, num_waves=8, block_m=256, block_n=64, head_dim_granule=64
            )  # family A
        if me.block_dmodel <= 256:
            return replace(
                me, num_waves=4, block_m=128, block_n=64, head_dim_granule=64
            )  # family B
        # Family W, the wide path. Four waves, like B and for the same reason
        # -- halving the wave count doubles the per-lane register file -- but
        # here the effect is larger than "more registers", and it is the single
        # biggest lever measured on this path:
        #
        #   head_dim   waves   shards   BLOCK_M   AGPR   spills   TFLOP/s
        #      384       4        1       128      150      0       579
        #      384       8        1       256        0    104       372
        #      512       4        2        64      183      0       366
        #      512       8        2       128        0     96       266
        #      512       8        4        64        0     30       252
        #
        # At 8 waves the allocator stops using AGPRs *entirely* and spills to
        # scratch instead; at 4 it puts the O accumulator where it belongs and
        # spills nothing. That is worth 1.4-1.6x, and no amount of extra
        # sharding recovers it -- 8 waves at 4 shards still lands at zero AGPRs.
        #
        # `ROWS_PER_WAVE` stays 32 (the MFMA's M extent), so BLOCK_M is
        # `Q_TILES * 32` and the shards eat the waves rather than the rows.
        return replace(
            me,
            num_waves=4,
            block_m=(4 // me.vo_shards) * 32,
            block_n=64,
            head_dim_granule=64,
        )

    # LDS is `BLOCK_N * head_dim * ~8.3 B` for a double-buffered K+V pair, and
    # the addressable cap is 163840 B. Measured, not inferred: the compiler
    # rejects head_dim 384 at 204288 B and 512 at 272384 B.
    LDS_CAP_BYTES = 163840

    def _with_d_axis_splits(self):
        """Decide `d_stages` and `qk_shards` from the tile width.

        Both stay 1 through head_dim 256, which is what keeps the four rungs
        that work today tracing to identical IR -- every construct these gate
        sits behind `const_expr(... > 1)`.

        The wider rungs need them for *different* reasons, and conflating the
        two is the mistake this split exists to prevent:

        - `d_stages` answers **LDS**. One pass of head_dim 384 needs 199.5 KB
          against a 160 KB cap, so the D axis has to be staged in time.
        - `qk_shards` answers **registers**. At head_dim 512 a wave's
          loop-invariant Q is 128 VGPRs and its O accumulator is 256 -- exactly
          the whole AGPR file -- so the D axis also has to be split across
          waves. 384 does not need this; 512 does.
        """
        d_stages, qk_shards, vo_shards = self.d_stages, self.qk_shards, self.vo_shards
        if d_stages is not None and qk_shards is not None and vo_shards is not None:
            return self
        if self.block_dmodel is None:
            raise ValueError(
                "_with_d_axis_splits runs after _with_widths; block_dmodel is not resolved"
            )
        if d_stages is None:
            # 2 for both wide rungs -- the least staging that fits LDS, and
            # measurement says least is also best. More stages shrink the live
            # K/V window, so the obvious guess is that more is safer. The
            # opposite happens, and sharply:
            #
            #   head_dim   stages   AGPR   spills   compile
            #      384        2      150      0        4 s
            #      384        3        0    113        4 s
            #      384        6        0    113        4 s
            #      512        2      256    286       61 s
            #      512        4        0      -     > 480 s (killed)
            #      512        8        0      -     > 480 s (killed)
            #
            # Past 2 the allocator stops using AGPRs *at all* and spills to
            # scratch instead, and at 512 that makes the register allocator
            # itself run away -- a build nobody will wait for. The AGPR file is
            # where the O accumulator has to live at these widths, so anything
            # that scares the allocator off it is a loss however small the
            # working set gets.
            d_stages = 2 if self.block_dmodel > 256 else 1
        if vo_shards is None:
            # Only 512 needs it. O is `ROWS_PER_WAVE * head_dim / 64` = 256
            # VGPRs there -- the entire AGPR file -- and no amount of staging
            # touches that, because staging bounds the K/V *window* and O is
            # loop-carried. 384's O is 192 and fits with zero spills, so it
            # pays the QK duplication for nothing.
            vo_shards = 2 if self.block_dmodel > 384 else 1
        if qk_shards is None:
            qk_shards = 1
        return replace(
            self, d_stages=d_stages, qk_shards=qk_shards, vo_shards=vo_shards
        )

    # Geometries whose *address helpers* are known correct, which is a stricter
    # set than the ones `make_traits` can describe.
    # BLOCK_M is in this tuple but does not affect KV addressing at all --
    # `SMEM_N_RPT` follows BLOCK_N and the granule. (8, 128, ...) is family W
    # at 2 shards and reuses family A's staging exactly.
    _SUPPORTED_GEOMETRIES = (
        (8, 256, 64, 64),
        (4, 128, 64, 64),
        (8, 128, 64, 64),
        (4, 64, 64, 64),
        (8, 64, 64, 64),
        (4, 128, 64, 32),  # family S -- granule 32, for widths off the 64 grid
    )

    def _check_helpers_support_geometry(self):
        """Refuse a geometry the kernel's addressing cannot actually serve.

        `fmha_traits_gfx950.make_traits` takes the geometry as parameters, so
        it will happily *describe* families S and B. The addressing has not
        caught up, and the gap is specific:

        - `_k_dma_m0_base` / `_v_dma_m0_base` place a tile line per wave per
          d-band (`wave * LINE + d * N_RPT * LINE`), which assumes
          `SMEM_N_RPT == NUM_WAVES` -- one issue per wave. Family B needs four,
          and waves 4..15's lines would simply never be written.
        - `init_dma_thread_offsets` splits a lane as `lane // VEC_KV` tokens by
          `lane % VEC_KV` D-buckets, which is the right split only when the
          granule is `VEC_KV * VEC_KV == 64`.
        - `_k_lds_read_base_per_lane` and `_swizzled_ks_offset` fold `%8`,
          `//8` and `//4` constants that are `SMEM_N_RPT` and
          `granule // K_STEP_QK` at family A's numbers.

        Failing here rather than at those sites keeps the diagnosis at the
        level of the decision. A geometry that builds and runs but addresses
        the wrong LDS produces plausible numbers, which is the failure mode
        this whole guard exists to avoid -- head_dim 192 produced exactly
        that.
        """
        if self.qk_shards > 1:
            raise NotImplementedError(
                f"qk_shards {self.qk_shards} is described by the traits but not yet implemented in the "
                "kernel body: it needs the wave index split into (q_tile, shard), Q/O restricted to the "
                "wave's D slice, and an explicit-partial S reduction through LDS."
            )
        geom = (self.num_waves, self.block_m, self.block_n, self.head_dim_granule)
        if geom not in self._SUPPORTED_GEOMETRIES:
            raise NotImplementedError(
                f"geometry (waves, BLOCK_M, BLOCK_N, granule) = {geom} is describable but not yet "
                f"addressable: the DMA and LDS-read helpers assume SMEM_N_RPT == NUM_WAVES and a "
                f"64-element granule. Supported: {self._SUPPORTED_GEOMETRIES}."
            )
        return self

    def staging_shape(self):
        """`(tokens_per_issue, lines, issues_per_wave)` for this geometry.

        The coherence check the family table has to satisfy, written once so a
        new family is validated rather than asserted. A wave moves 512 bf16
        elements per DMA issue (64 lanes x 8), so the granule fixes how many
        tokens that covers, and BLOCK_N fixes how many such lines a KV tile
        needs. `issues_per_wave` must be a positive integer or the tile does
        not divide across the waves.
        """
        per_issue = (WARP_SIZE * DMA_BYTES // BF16_BYTES) // self.head_dim_granule
        if self.block_n % per_issue:
            raise ValueError(
                f"BLOCK_N {self.block_n} is not a multiple of {per_issue} tokens per DMA issue"
            )
        lines = self.block_n // per_issue
        if lines % self.num_waves:
            raise ValueError(
                f"{lines} KV tile lines do not divide across {self.num_waves} waves"
            )
        if self.block_dmodel < PV_MFMA_N:
            raise ValueError(
                f"block_dmodel {self.block_dmodel} is narrower than the PV MFMA's {PV_MFMA_N}-column "
                f"output; head_dim below {PV_MFMA_N} cannot have a native tile with v_mfma_f32_32x32x16"
            )
        return per_issue, lines, lines // self.num_waves

    def _checked_against_traits(self, meta):
        """`resolve`'s last step: prove the traits are buildable, return `self`.

        See `resolve` for why the traits object is discarded here.
        """
        self.build_traits(meta)
        return replace(self, GRID_AXIS_ORDER=GRID_AXIS_HEAD_FASTEST)

    def build_traits(self, meta):
        """The dualwave traits this configuration implies. **The one mapping.**

        Knobs are the tuning module's vocabulary and traits are the kernel's;
        this method is the only place the two are related, which is why it is
        public rather than a step of `resolve`. The builder calls it instead of
        reading a field off the knobs, so the knob object stays plain data --
        see the note where the `traits` field used to be.

        `fmha_traits_gfx950.make_traits` takes the geometry as parameters where
        the production constructor hardcodes it, and is checked field-by-field
        against production at family A's numbers -- so family A goes through
        this path today and the bitwise-vs-production test covers it.
        """
        self.staging_shape()  # the geometry must divide a KV tile evenly
        self._check_helpers_support_geometry()
        num_kv_heads = (
            meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        )
        traits = make_traits(
            num_heads=meta.num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.block_dmodel,
            num_waves=self.num_waves,
            block_m=self.block_m,
            block_n=self.block_n,
            granule=self.head_dim_granule,
            d_stages=self.d_stages,
            qk_shards=self.qk_shards,
            vo_shards=self.vo_shards,
            v_half_wave=self.v_half_wave,
            v_n_group=self.v_n_group,
            v_k_substep=self.v_k_substep,
            v_dc_in_pair=self.v_dc_in_pair,
            causal=meta.causal,
            window=meta.window,
            bias=meta.bias,
            dropout=meta.dropout,
            lpt_tile_order=self.lpt_tile_order,
            dtype_str=meta.dtype_str,
            waves_per_eu=self.waves_per_eu,
            daz=self.daz,
            lazy_rescale=self.lazy_rescale,
            setprio=self.setprio,
            stagger=self.stagger,
            num_kv_splits=self.num_kv_splits,
            paged=self.paged,
            kv_cache_layout=self.kv_cache_layout,
            kv_vectorized=self.paged and self.kv_cache_layout == "vectorized",
            return_lse=self.return_lse,
        )
        lds_bytes = traits.LDS_KV_TOTAL_SIZE * traits.BF16_BYTES
        if lds_bytes > self.LDS_CAP_BYTES:
            raise ValueError(
                f"KV staging needs {lds_bytes} B of LDS, over the {self.LDS_CAP_BYTES} B cap, for "
                f"block_dmodel {self.block_dmodel} at BLOCK_N {self.block_n} with d_stages "
                f"{self.d_stages}. Raise d_stages (LDS scales as 1/d_stages) or lower BLOCK_N. "
                "Left to the compiler this surfaces as 'local memory (N) exceeds limit (163840)' "
                "with no indication of which knob to move."
            )
        return traits


# Defaults the policy has no shape-dependent opinion about. These match the
# production `build_flash_attn_dualwave_swp_module` signature exactly, so a
# default-knob parity build is the schedule the baseline was measured on.
#
# `waves_per_eu` was here and is not any more: it turned out to have a
# shape-dependent opinion after all, and it belongs with the derivations rather
# than the constants. Left `None` so `_with_occupancy_target` can see that
# nobody pinned it -- production's value survives as what that step derives at
# every rung where the hardware can honour it.
_GFX950_FALLBACK = Gfx950Knobs(
    daz=True,
    # **`False`, against upstream's `True`. A LOCAL DIVERGENCE, and the one
    # place in this file where the default gives up measured speed.**
    #
    # `lazy_rescale` skips the per-tile rescale of the O accumulator when the
    # row max has not moved by more than `DUALWAVE_SWP_RESCALE_THRESHOLD`
    # (8.0), decided by a wave-uniform ballot so it is one branch rather than
    # per-lane predication. What it gives up is the guarantee that
    # `P = exp2(S - m_row) <= 1`: `m_row` is now stale by up to the threshold,
    # so P can reach 2**8, and P is written in the input dtype before the PV
    # MFMA. It buys 5-17% at S=4096 -- head_dim 64 -13/-16%, 128 -10/-15%,
    # 256 -5/-6%, dense/causal -- and the wide body (head_dim 384/512) never
    # asks for it at all, rescaling eagerly through `rescale_o_serial`.
    #
    # It is off because **the default configuration buys accuracy with
    # performance and not the other way round.** Speed is to come from the
    # algorithm, from GPU resource management and from instruction scheduling
    # -- `d_stages`, `vo_shards`, `_with_occupancy_target` and the scheduling
    # barriers are all of that shape and all stay on. Precision is not a
    # currency the default spends.
    #
    # It costs precision at every rung it touches. Max |err| against an fp64
    # reference, B=1 H=4 S=289 dense, `sm_scale=1.0`:
    #
    #     head_dim   bf16 on -> off      f16 on -> off
    #        64     1.6e-2 -> 8.1e-3   2.2e-3 -> 1.1e-3, 1920 inf -> 0
    #       128     1.8e-2 -> 8.5e-3   2.3e-3 -> 1.9e-3, 15232 inf -> 0
    #       256     2.3e-2 -> 1.2e-2   2.0e-3 -> 1.5e-3, 62464 inf -> 0
    #
    # In **f16 it is not an accuracy question at all**: the answer comes back
    # non-finite, at head_dim 64/128/256 for any `sm_scale` large enough, on
    # the shipped 0.14.1 library as much as here. There is no trade to weigh
    # when one arm returns NaN.
    #
    # **That f16 defect is NOT understood, and two plausible mechanisms are
    # ruled out.** The LSE is *correct* in the same runs that return an
    # infinite O, so the row max and the running sum are both fine and only
    # the O accumulation is not. It is the rescale branch rather than the skip
    # branch: raising the threshold, which rescales *less*, monotonically
    # reduces the count (8.0 -> 1920, 12.0 -> 1664, 16.0 -> 1152 at head_dim
    # 64). That rules out the obvious story -- P overflowing f16's 65504 where
    # bf16 has f32's exponent range -- since more skipping would then mean
    # more overflow, not less; and clamping P to 65504 before `cast_p` changes
    # nothing, the same 1920 elements and the same error on the finite ones.
    # `_scale_v_p` and `scale_o`, the two things the branch does touch, both
    # multiply by `corr = exp2(m_row - m_new) <= 1`, which cannot manufacture
    # an infinity from a finite operand. What is left is the branch's plumbing
    # -- the `scf.if` yielding P as a vec32 through `_v_p_to_vec32`/
    # `_v_vec32_to_p` alongside an inline-asm-anchored scalar -- and that wants
    # an ISA read rather than another hypothesis.
    #
    # Turning it back on is a one-keyword `lazy_rescale=True` on the knobs, so
    # a sweep that wants the old schedule still has it, infinities included.
    lazy_rescale=False,
    setprio=True,
    stagger=True,
    lpt_tile_order=False,
    paged=False,
    kv_cache_layout="linear",
    num_kv_splits=1,
    return_lse=False,
    strides_constexpr=False,
)

_BY_ARCH = {"gfx950": Gfx950Knobs}


def fmha_knobs(arch: str, **overrides) -> FmhaKnobs:
    """The knob object for `arch`, with `overrides` pinned.

    The one entry point. A caller names an architecture and the fields it
    cares about; everything else stays `None` until `resolve` decides it.
    Keyed on the arch *prefix* so a full `gcnArchName` --
    "gfx950:sramecc+:xnack-" -- works without the caller stripping it.
    """
    base = arch.split(":")[0].lower() if arch else ""
    for prefix, cls in _BY_ARCH.items():
        if base.startswith(prefix):
            known = {f.name for f in fields(cls)}
            unknown = set(overrides) - known
            if unknown:
                raise TypeError(f"unknown {cls.__name__} field(s): {sorted(unknown)}")
            return cls(**overrides)
    raise ValueError(f"no FMHA knobs for arch {arch!r}; known: {sorted(_BY_ARCH)}")
