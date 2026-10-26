# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Traits and tuning policy for the gfx950 backward **dQ (and dB)** kernel.

Split from `fmha_bwd_dq_gfx950.py` for the reason `fmha_tuning_gfx950.py`
gives: the kernel file is about correctness, this one is about speed, and a
number here can only make a build slow, never wrong.

--- What the dQ kernel adds to the forward's traits --------------------------

Three things, and all of them are *build* axes rather than tuning ones:

- `STORE_DB` -- whether the kernel *can* write `dB = dS`. AOTriton's
  `bwd_kernel_dq` emits `DQ, DB` because `dS` is materialised per (q, k) element
  only there; our gfx1201 port dropped it. A build without it emits no store at
  all, which matters because the store is per element (see the kernel's
  `BwdDbStoreHelper` for why it cannot be a vector store).
  Whether the store *runs* is a separate, runtime question -- a caller who wants
  no dB passes an all-zero dB stride triple, which `db_store_enabled` in the
  kernel reads. So this axis buys the instructions back for a build that has no
  bias at all; it does not decide anything for a build that does.
- `HDIM_VO_FLOOR` -- the `hdim_vo` counterpart of `HDIM_QK_FLOOR`. This kernel
  reads *two* tiles through the K register path, one whose D axis is the qk
  extent and one whose is the vo extent, so a single floor cannot serve both.
- `LDS_KV_TOTAL_SIZE`, cut to one buffer. `make_bwd_dq_traits` says why; it is
  what makes head_dim 512 fit at all.

--- The geometry is dQ's own, and it is measured -----------------------------

`BwdDqKnobs._with_wave_geometry` replaces the forward's family table rather
than extending it, and the docstring there carries the full
`(num_waves, waves_per_eu)` sweep it came from. The headline is that
`waves_per_eu = 1` is worth up to 4x and the forward's default of 2 is wrong
for this kernel at every rung measured.

The MFMA shape is still not baked in anywhere: `ROWS_PER_WAVE`, `D_CHUNK` and
`K_STEP_QK` all come from `fmha_traits_gfx950`, and `BLOCK_M` is derived as
`num_waves * MFMA_M` rather than chosen -- which is the P7 invariant this phase
had to stop stating in prose.
"""

from dataclasses import dataclass, fields, replace

from kernels.attention_bwd_cdna.fmha_traits_gfx950 import (
    ParityDualwaveTraits,
    make_traits,
)
from kernels.attention_bwd_cdna.fmha_tuning_gfx950 import (
    LADDER,
    Gfx950Knobs,
    tile_width_for,
)

__all__ = [
    "BWD_DQ_LADDER",
    "MFMA_PASSES",
    "BwdDqKnobs",
    "BwdDqTraits",
    "bwd_dq_knobs",
    "make_bwd_dq_traits",
    "mfma_flops_per_pass",
    "register_demand",
]

# Passes per MFMA on gfx950, from the *scheduling model* rather than from an
# ISA doc -- `SISchedule.td`, `let SchedModel = SIDPGFX950FullSpeedModel`. The
# lore is explicit that this is where the number lives and that getting it
# wrong invalidates everything derived from it.
#
# The consequence is the one number that decides B3.5's shape question, and it
# needs no lane-map probe:
#
#     shape        flops      passes   flops/pass
#     16x16x16      8192        4         2048     <- half rate
#     16x16x32     16384        4         4096
#     32x32x16     32768        8         4096
#
# **`v_mfma_f32_16x16x16_bf16` is exactly half the peak rate of the shape this
# kernel uses today.** A 16-row family built on it cannot beat the 32-row
# family on any rung where the 32-row family fits, and its ceiling is 50% of
# the machine. `16x16x32` gives 16 rows per wave at *full* rate, and both are
# already dispatched by `fx.rocdl.MFMA(m, n, k, bf16)`
# (`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp`), so neither needs C++ work.
MFMA_PASSES = {(16, 16, 16): 4, (16, 16, 32): 4, (32, 32, 16): 8}


def mfma_flops_per_pass(m, n, k):
    """Throughput of one MFMA shape, in flops per XDL pass.

    The comparison that matters is *per pass*, not per instruction: a shape
    that does half the work in half the passes is the same rate, and a shape
    that does a quarter of the work in half the passes is half.
    """
    if (m, n, k) not in MFMA_PASSES:
        raise KeyError(
            f"no gfx950 pass count recorded for {m}x{n}x{k}; read it out of SISchedule.td"
        )
    return 2 * m * n * k / MFMA_PASSES[(m, n, k)]


def register_demand(traits, *, store_db=False, a_tiles_live=1):
    """Per-lane 32-bit register demand for the dQ body, by term.

    **Structural, not fitted.** Every term below is an exact count of what the
    algorithm holds; nothing here is tuned against a measurement. That is what
    makes it usable as a prediction for a geometry that has never been built,
    which is the whole point of computing it in B3.5.

    Let `R` = rows per wave (the MFMA's N extent, since the query row is the N
    axis of both GEMM1 and GEMM3), `d` = head_dim, `n` = BLOCK_N, `W` = 64
    lanes. bf16 values pack two per register.

        q, do    R*d / 2W    loop-invariant B operands, one each
        dq       R*d / W     the f32 accumulator, loop-carried
        score    R*n / W     S and dP, one each, f32
        ds       R*n / 2W    the packed bf16 B operand of GEMM3
        a_tile   n*d / 2W    one fully materialised A operand (K, V or K-T)

    **`a_tiles_live` is the one modelling choice**, and it is the only place
    the compiler's judgement enters: the body reads three A tiles per KV block
    and consumes each immediately, so the scheduler decides how many are alive
    at once. `1` reproduces the measured VGPR count to within 4-30 registers at
    every rung that does not spill (measured against the built kernels),
    so the scheduler is in fact sinking two of the three.

    What the model does **not** include is addressing, the DMA descriptors and
    the loop scalars. Measured residual is 4-12 registers at two waves and
    28-30 at four; it is left out rather than fitted, because a constant chosen
    to make today's numbers work is not a prediction.

    `store_db` adds the dB store's address arithmetic. Measured at +34 (four
    waves) and +70 (two waves) -- itself wave-count-dependent and therefore
    also not modelled, but recorded here because it is what tips head_dim 256
    from zero spills into 14.
    """
    r, d, n, w = traits.ROWS_PER_WAVE, traits.HEAD_DIM, traits.BLOCK_N, traits.WARP_SIZE
    terms = {
        "q": r * d // (2 * w),
        "do": r * d // (2 * w),
        "dq_acc": r * d // w,
        "score_s": r * n // w,
        "score_dp": r * n // w,
        "ds_packs": r * n // (2 * w),
        "a_tiles": a_tiles_live * (n * d // (2 * w)),
    }
    if store_db:
        # Not a term of its own: the dB store reads `ds_packs`, which is
        # already counted. What it adds is address arithmetic, and that is the
        # part the model declines to guess at.
        pass
    terms["total"] = sum(terms.values())
    # The unified register file at one wave per SIMD. Above this the allocator
    # spills, and B3 measured that it does so non-linearly -- head_dim 512 is
    # 320 registers over and spills 546.
    terms["over_512"] = max(0, terms["total"] - 512)
    return terms


# The rungs this kernel is built and tested for. B3 brings it to the forward's
# `LADDER` in full; `tile_width_for` still owns the rounding rule, so a
# head_dim of 100 rounds to 128 exactly as it does forward.
#
# Kept as its own name rather than an alias for `LADDER` so a rung can be
# withdrawn here without touching the forward -- which is what "a rung that has
# never been run is not a rung" needs to stay enforceable.
BWD_DQ_LADDER = LADDER


@dataclass(frozen=True)
class BwdDqTraits(ParityDualwaveTraits):
    """The forward's parity traits plus what the dQ kernel alone needs.

    A subclass for the same reason `ParityDualwaveTraits` is one: the parent is
    frozen with no defaulted fields, so added fields need defaults and those
    defaults are the "behave like the forward" values. Nothing that reads the
    parent's fields has to know these exist.
    """

    # Write `dB = dS`. Off by default; the store is per element and costs 32
    # `buffer_store`s per lane per KV tile, so a build that does not want it
    # must not pay for it.
    STORE_DB: bool = False

    # Where the bias tile comes from, for a bias build. `""` is the direct
    # path: 32 scalar 2-byte `buffer_load`s per lane per KV tile, issued after
    # GEMM1 and consumed at once. `"lds"` DMAs the `BLOCK_M x BLOCK_N` bias tile
    # into its own LDS region with the K/V staging
    # (`BwdDqKvGmemToLdsLoader.stage_bias`) and reads each lane's row back with
    # `ds_read_b64`; it reads the bias through the 8-element row contract (see
    # `BwdDqKnobs.bias_stage`).
    BIAS_STAGE: str = ""

    # The `hdim_vo` counterpart of `HDIM_QK_FLOOR`, for the V tile's own
    # padded-head mask. A field rather than a knob-only value because the two
    # extents are read by *different* helper instances in this kernel and
    # nothing else would keep them together.
    HDIM_VO_FLOOR: int = 0

    # --- B3.5: the MFMA shape, named on all three axes -----------------------
    #
    # `(M, N, K)` of the one MFMA this body issues. Contract section 4 required
    # the shape to be a trait rather than a literal; the parent carries only
    # `D_CHUNK`, `K_STEP_QK` and `MFMA_LANE_K`, which at 32x32x16 happen to
    # equal M, K and M*K/64 -- so today the shape is *inferable* from them and
    # therefore not stated.
    #
    # **Three coincidences are load-bearing at 32x32x16 and none survives 16
    # rows**, which is why the axes get separate names before anything moves:
    #
    #     MFMA_M == D_CHUNK          GEMM3's A operand M is the d chunk (32)
    #     MFMA_M == BLOCK_N / 2      GEMM1's A operand M is the KV token, and
    #                                a 64-token tile is exactly two M steps
    #     MFMA_N == ROWS_PER_WAVE    the query row is the N axis of GEMM1 and
    #                                GEMM3 both
    #
    # At 16x16x32 the first two part company from `D_CHUNK` and from
    # `BLOCK_N/2` independently, and `MFMA_K` stops equalling `K_STEP_QK`.
    # Defaults reproduce today exactly, so a build is unchanged.
    MFMA_M: int = 32
    MFMA_N: int = 32
    MFMA_K: int = 16

    @property
    def BIAS_LDS_BYTES(self):
        """LDS the staged bias tile adds beside the K/V regions; 0 unless `BIAS_STAGE == "lds"`."""
        return (
            self.BLOCK_M * self.BLOCK_N * self.BF16_BYTES
            if self.BIAS_STAGE == "lds"
            else 0
        )

    @property
    def ACC_ELEMS(self):
        """f32 accumulator elements one lane holds for one MFMA.

        16 at 32x32x16, 4 at either 16-row shape. Every `range_constexpr(16)`
        over a score half in the body is this number, and every one of them is
        a different quarter of the kernel, so it is derived once here.
        """
        return self.MFMA_M * self.MFMA_N // self.WARP_SIZE

    @property
    def SCORE_MSTEPS(self):
        """MFMA steps along the KV-token axis of one score tile.

        2 at BLOCK_N 64 with a 32-row M extent -- the `lo`/`hi` pair the whole
        softmax path is written around -- and 4 at a 16-row one. The pair is
        spelled out as `s_lo, s_hi` in `flash_attn_utils`, so a shape that
        makes this 4 needs that path generalised and not merely re-parameterised.
        """
        return self.BLOCK_N // self.MFMA_M

    @property
    def OPERAND_LANE_ELEMS(self):
        """bf16 values one lane holds of an A or B operand, for one MFMA.

        8 at 32x32x16 and at 16x16x32; **4** at 16x16x16. `MFMA_LANE_K` on the
        parent is this same number, and it is kept separate here because the
        parent's is tied to `K_STEP_QK`.
        """
        return self.MFMA_M * self.MFMA_K // self.WARP_SIZE


def make_bwd_dq_traits(
    *, store_db=False, hdim_vo_floor=0, mfma_shape=(32, 32, 16), bias_stage="", **kwargs
):
    """`fmha_traits_gfx950.make_traits`, widened, and cut to **one** LDS buffer.

    Delegating rather than transcribing is the whole point: every derived
    field -- the LDS line strides, the DMA split, the V transpose strides the
    third GEMM reads K through -- comes from the one constructor the forward's
    `assert_matches_production` pins against production. A second copy of any
    of them would be a second thing to keep in step, and the read and write
    sides of an LDS layout drifting apart is the specific failure this avoids.

    **The one field this overrides is `LDS_KV_TOTAL_SIZE`.** The forward sizes
    its allocation for `NUM_PREFETCH_K` KV tiles in flight; the dQ kernel keeps
    one, and after B3's single-staging of K it needs exactly one K-pitch region
    plus one V-pitch region -- which is `DUALWAVE_SWP_KV_PER_BUFFER`, already
    derived. Overriding the *total* and nothing else is what keeps the buffer
    bases, the `m0` tables and the tile sizes the shared derivation's.

    At head_dim 512, BLOCK_N 64: 66560 + 69632 = 136192 B against a 163840 B
    cap. The three-slot B2 layout needed 199 KB there, which is why the second
    staging of K was a B3 blocker rather than a B7 optimisation.
    """
    base = make_traits(**kwargs)
    m, n, k = mfma_shape
    if (m, n, k) not in MFMA_PASSES:
        raise ValueError(
            f"unknown MFMA shape {mfma_shape}; add its gfx950 pass count to MFMA_PASSES first"
        )
    if (m, n, k) not in ((32, 32, 16), (16, 16, 32)):
        # `16x16x16` is describable and half rate (see `MFMA_PASSES`), so it is
        # refused rather than merely unbuilt: a family on it could not beat the
        # 32-row one anywhere the 32-row one fits.
        raise NotImplementedError(
            f"the dQ body has two families, 32x32x16 and 16x16x32; {m}x{n}x{k} is neither. "
            "16x16x16 is half rate on gfx950 and is deliberately not built."
        )
    if (m, n, k) == (16, 16, 32) and base.HEAD_DIM % 64:
        # The 16-row `_kt_read_base` folds `tok_off(4 * group)` into
        # `group * granule`, which holds only when `SMEM_N_RPT` divides 4 --
        # true at granule 64 (`SMEM_N_RPT == 4` with `BLOCK_N` 32) and not at
        # granule 32, where it is 2. The rungs off the 64 grid are all
        # comfortably served by the 32-row family, so this is a real limit
        # rather than a deferral, and it is refused where it is decided.
        raise NotImplementedError(
            f"the 16-row family is built for head_dim tiles that are multiples of 64, not "
            f"{base.HEAD_DIM}: its transpose read assumes the granule-64 staging shape"
        )
    if (m, n, k) == (16, 16, 32) and base.BLOCK_N != 2 * m:
        # `BLOCK_N / MFMA_M` is the number of score M-steps, and the softmax
        # path in `flash_attn_utils` is an `(s_lo, s_hi)` *pair* rather than a
        # list. Two is not a tuning choice here, it is the shape of shared
        # production code -- and 32 is also what closes head_dim 512's last 40
        # registers, which is why the cheap port and the register fix agree.
        raise ValueError(
            f"the 16-row family needs BLOCK_N {2 * m}, got {base.BLOCK_N}: the score tile must stay two "
            "MFMA steps or the (s_lo, s_hi) pair in flash_attn_utils stops describing it"
        )
    if n != base.ROWS_PER_WAVE:
        raise ValueError(
            f"MFMA N extent {n} must equal ROWS_PER_WAVE {base.ROWS_PER_WAVE}: the query row is N"
        )
    carried = {
        f.name: getattr(base, f.name)
        for f in fields(base)
        if f.name != "LDS_KV_TOTAL_SIZE"
    }
    return BwdDqTraits(
        **carried,
        STORE_DB=bool(store_db),
        BIAS_STAGE=bias_stage,
        HDIM_VO_FLOOR=int(hdim_vo_floor),
        LDS_KV_TOTAL_SIZE=base.DUALWAVE_SWP_KV_PER_BUFFER,
        MFMA_M=m,
        MFMA_N=n,
        MFMA_K=k,
    )


@dataclass(frozen=True)
class BwdDqKnobs(Gfx950Knobs):
    """The forward's knob pipeline, with the dQ build axes added.

    `resolve` is inherited whole; only the last step (`build_traits`) is
    replaced, because the pipeline before it -- mode checks, the ladder, the
    wave geometry -- asks exactly the same questions for the backward.
    """

    store_db: bool | None = None

    # `None` | `"direct"`, bias builds only. Unset stages the bias tile through
    # LDS wherever that is built -- the 32-row family at `block_n` 64 with the
    # staged tile inside the LDS cap -- and loads it directly elsewhere;
    # `"direct"` forces the per-tile direct loads. The staged path reads the bias
    # in 16-byte chunks, which is a *contract on the caller's bias*: every row,
    # head and batch stride a multiple of 8 elements and the pointer 16-byte
    # aligned (what PyTorch's padded attn_bias already is), and
    # `stride_seq_q >= ceil8(seqlen_k)` so the last row's final chunk lies inside
    # the allocation; a row stride of 0 (a row-broadcast `.expand()`) is
    # accepted only when `seqlen_k % 8 == 0`
    # (`fmha_bwd_dq_gfx950.bias_chunk_contract_ok`); `_args` enforces it.
    bias_stage: str | None = None

    # 32 or 16. The MFMA's N extent, which is the query rows one wave owns --
    # `_with_wave_geometry` turns it into the shape and the KV tile together,
    # because at 16 rows `BLOCK_N` is not free.
    mfma_rows: int | None = None

    # The forward's list plus the two-wave points. **Not a relaxation of the
    # check** -- the check exists because `_k_dma_m0_base` assumes one DMA
    # issue per wave, and `_ParityKvStaging` is the generalisation that lifts
    # it (`ISSUES_PER_WAVE = SMEM_N_RPT // NUM_WAVES`, 4 at two waves and
    # granule 64). What makes these entries legitimate is that every one of
    # them was run and agrees with the default build to the error-ratio gate.
    #
    # Two waves is here because B1 measured it as the best point at head_dim
    # 128 (788 TF against 403 at four waves and `waves_per_eu=2`), and a
    # geometry the tuner cannot express is a lever that does not exist.
    _SUPPORTED_GEOMETRIES = Gfx950Knobs._SUPPORTED_GEOMETRIES + (
        (2, 64, 64, 64),
        (2, 64, 64, 32),
        # The 16-row family: BLOCK_M is `4 waves * 16 rows` and BLOCK_N is 32,
        # so a KV tile is four lines and each of the four waves issues one DMA.
        (4, 64, 32, 64),
    )

    def resolve(self, meta) -> "BwdDqKnobs":
        """The forward's pipeline, seeded from *this* kernel's fallback.

        Overridden only for the seed. `Gfx950Knobs.resolve` merges into
        `_GFX950_FALLBACK`, which is a `Gfx950Knobs` and knows nothing about
        `store_db` -- `dataclasses.replace` on it would raise rather than drop
        the field, but the failure would come from a stack frame two files away
        from the cause. The step sequence below is the parent's, verbatim.
        """
        self._checked_mfma_rows()
        return (
            _BWD_DQ_FALLBACK.merge(self)
            ._checked_modes()
            ._with_widths(meta)
            ._with_wave_geometry()
            ._checked_against_traits(meta)
        )

    def _checked_mfma_rows(self):
        """`mfma_rows` is `None` (the table decides), 16 or 32; anything else would be coerced to the 32-row family."""
        if self.mfma_rows not in (None, 16, 32) or isinstance(self.mfma_rows, bool):
            raise ValueError(
                f"mfma_rows must be None, 16 or 32, got {self.mfma_rows!r}"
            )

    def _with_d_axis_splits(self):
        """Both D-axis cuts stay at 1. **This is load-bearing, not tidiness.**

        The forward's version turns `d_stages` on at block_dmodel > 256 and
        `vo_shards` on at > 384, because its LDS does not hold two KV tiles
        that wide and its O accumulator is the whole AGPR file. This body has
        one tile in flight and 133 KB at head_dim 512, so neither applies --
        and neither is *implemented* here.

        The failure if this is left to the parent is not a build error. With
        `D_STAGES = 2`, `ParityGemmHelper.qk` silently becomes `qk_stage(...,
        stage=0)` and `pv` writes only `v_o[0 : D_CHUNKS/2]`: the kernel
        reduces over half the head dim, writes half the accumulator, and
        returns a finite wrong answer. Caught here by an LDS figure that was
        half what it should have been, which is the only reason it was caught
        at all -- so `build_traits` also refuses a pinned `d_stages > 1`
        rather than relying on this default holding.
        """
        return replace(
            self,
            d_stages=1 if self.d_stages is None else self.d_stages,
            qk_shards=1 if self.qk_shards is None else self.qk_shards,
            vo_shards=1 if self.vo_shards is None else self.vo_shards,
        )

    def _hdim_vo_floor(self, meta):
        """The V tile's counterpart of `hdim_qk_floor`, exclusive.

        The floor exists so the padded-head mask can skip the D columns the
        dispatcher guarantees are real -- it is what keeps a padded build from
        paying the 27-54% the forward measured for masking every K-step. The
        *qk* floor comes from the ladder rung; the *vo* one has to come from
        the same rung, because both extents live in the same compiled tile.

        It is dropped to 0 when `head_dim_v` is at or below it. That happens
        for an asymmetric call like `head_dim 128, head_dim_v 40`: the 128 tile
        promises `hdim_qk > 96` and promises nothing at all about `hdim_vo`, so
        the V mask has to cover every step. Slower, and the alternative is
        reducing over the caller's padding.
        """
        head_dim_v = meta.head_dim if meta.head_dim_v is None else meta.head_dim_v
        floor = int(self.hdim_qk_floor or 0)
        return floor if head_dim_v > floor else 0

    def _with_wave_geometry(self):
        """dQ's own geometry table. Measured, not inherited.

        The forward's families answer questions this body does not ask: family
        W exists to shard its O accumulator (`VO_SHARDS`) and stages the D axis
        at 384 because two KV tiles in flight do not fit LDS there. Neither
        holds after B3 -- one tile in flight, and head_dim 512 is 133 KB.

        Measured at `B=2 H=8 S=2048` bf16 non-causal, TFLOP/s on `6*S^2*d`,
        over the whole `(num_waves, waves_per_eu)` grid:

            head_dim   w2/e1  w2/e2  w4/e1  w4/e2  w8/e1  w8/e2
                32       335    334    423    420      -      -
                64       501    500    582    580    436    435
                96       599    597    678    681      -      -
               128       715    622    685    702    496    505
               160       665    313    645    429      -      -
               192       672    260    657    350    311    316
               224       682    667    677    171      -      -
               256       712    694    675    169    162    162
               384       339    336    335    338    338    336
               512       115    113    113    117    115    114

        (`-` is `SMEM_N_RPT % NUM_WAVES != 0`: a granule-32 tile has four KV
        lines, so eight waves cannot each own one.)

        Two things in that table are policy and one is a finding.

        **`waves_per_eu = 1` unconditionally.** It is never worse than 2 and it
        is worth up to 4x -- head_dim 256 at four waves is 675 against 169. The
        addendum's discriminator explains it exactly: the ISA dump for the slow
        build reports 256 VGPRs, **0 AGPRs and 191 spills**, against 460 VGPR +
        204 AGPR and zero spills for the fast one. It was not short of
        registers, it was forbidden from using half of them.

        **Two waves from 128 to 256, four elsewhere.** Four wins below 128 and
        two wins above it, both by margins the table can support. Between the
        neighbouring points the difference is often under 10%, which per the
        lore is not a result a sweep can settle -- those are left where they
        fall rather than tuned.

        **384 and 512 are flat across the entire grid**, which is the finding:
        they are not occupancy-limited and no wave count reaches them.
        """
        me = self._with_d_axis_splits()
        # Decided before the pin check, and separately from it: the wave count
        # is a *geometry* choice that has to move with BLOCK_M, while this is a
        # register-budget hint that is right at 1 whatever the geometry. A
        # sweep pinning the four geometry fields still gets it.
        if me.waves_per_eu is None:
            me = replace(me, waves_per_eu=1)
        pinned = (me.num_waves, me.block_m, me.block_n, me.head_dim_granule)
        if all(x is not None for x in pinned):
            return me
        if any(x is not None for x in pinned):
            raise ValueError(
                f"pin num_waves, block_m, block_n and head_dim_granule together or not at all, got {pinned}"
            )
        if me.block_dmodel is None:
            raise ValueError(
                "_with_wave_geometry runs after _with_widths; block_dmodel is not resolved"
            )
        # **384 and up.** Measured at `B=2 H=8 S=4096`, 16 rows against 32:
        # 512 is 4.23x (116 -> 490 TF, 546 spills -> 0), 384 is 1.69x (303 ->
        # 511), 256 is 1.04x, 192 is exactly level, and 128 and 64 are *worse*
        # (0.94x, 0.86x). So the families split where the register file does,
        # and the 32-row one keeps everything below 384 -- this is additive,
        # not a replacement.
        #
        # 256 at 1.04x is inside the band the lore says a sweep cannot settle,
        # so for a long time it stayed on the 32-row family purely by
        # incumbency. Correctness moved it: the 32-row build at 256 with
        # dropout lands on 507 VGPR + 251 AGPR, five registers short of the
        # file, and the allocator responds by spilling the 32 LDS-DMA address
        # operands into AGPRs and rematerialising nine of them through a
        # single VGPR that is overwritten between consecutive
        # `buffer_load ... offen lds` issues. That build races: with
        # `B=3 H=5 sq=2048 sk=8192` it poisons one wave's 32x32 accumulator
        # tile (dQ columns 32..63) in one or two (batch, head, q-block)
        # instances per launch, non-deterministically, and only there -- the
        # dropout=False build at 256 (460/204) and the dropout=True build at
        # 224 (456/200) both stay under the cliff and are clean. The 16-row
        # family at 256 is 232 VGPR, 0 AGPR, 0 spills and half the LDS, with
        # every DMA address in its own register.
        #
        # So 256 moves down to 16 rows, which costs nothing: re-measured at
        # `B=2 H=8 S=4096`, 16 rows is 564 TF against 550 without dropout and
        # 527 against 509 with it, matching the 1.04x the original sweep saw.
        rows = (
            me.mfma_rows
            if me.mfma_rows is not None
            else (16 if me.block_dmodel >= 256 else 32)
        )
        if rows == 16:
            # BLOCK_N 32 for the reason `build_traits` states, and four waves
            # because the wide rungs need one wave per SIMD to see all 512
            # registers -- which is the whole premise of the family.
            return replace(
                me,
                num_waves=4,
                block_m=4 * 16,
                block_n=32,
                head_dim_granule=64,
                mfma_rows=16,
            )
        waves = 2 if 128 <= me.block_dmodel <= 256 else 4
        return replace(
            me,
            mfma_rows=32,
            num_waves=waves,
            # `ROWS_PER_WAVE` is pinned at the MFMA's M extent, so BLOCK_M is
            # derived rather than chosen -- `make_traits` raises on any other
            # value, which is the P7 invariant this phase had to stop stating
            # in prose.
            block_m=waves * 32,
            block_n=64,
            head_dim_granule=32 if me.block_dmodel % 64 else 64,
        )

    def build_traits(self, meta):
        """The forward's `build_traits`, against `make_bwd_dq_traits`.

        Copied in structure rather than called through, because the parent
        hardcodes `make_traits` as its last expression and there is no seam
        below it. The divergences are named: the rung check, `store_db`, and
        the modes B2 does not implement.
        """
        self._checked_mfma_rows()
        if self.block_dmodel not in BWD_DQ_LADDER:
            raise NotImplementedError(
                f"the backward dQ kernel is built for head_dim tiles {BWD_DQ_LADDER}, not "
                f"{self.block_dmodel}. head_dim {meta.head_dim} rounds to {tile_width_for(meta.head_dim)}."
            )
        if meta.dtype_str not in ("f16", "bf16"):
            # Same assertion and the same two values as gfx1201's `bwd_dq`, so
            # the two ports refuse the same set. It is a *build* axis: the
            # operand width reaches the MFMA opcode, the `dS` pack and the
            # store, so an unrecognised string would otherwise pick a dtype by
            # falling through a comparison.
            raise NotImplementedError(
                f"the backward dQ kernel supports f16/bf16, got {meta.dtype_str!r}"
            )
        if (self.mfma_rows or 32) == 16 and self.block_dmodel % 64:
            # Before `make_traits`, which would otherwise raise first with a
            # message about the granule and leave the caller to work out that
            # the granule came from the family. `_kt_read_base` folds
            # `tok_off(4 * group)` into `group * granule`, which needs
            # `SMEM_N_RPT` to divide 4 -- true at granule 64, not at 32.
            raise NotImplementedError(
                f"the 16-row family is built for head_dim tiles that are multiples of 64, not "
                f"{self.block_dmodel}: its transpose read assumes the granule-64 staging shape, and "
                "the off-grid rungs are all served by the 32-row family anyway"
            )
        # Refused rather than ignored. Each of these would build and run and
        # return the right *shape*, which is the failure mode the whole
        # backward plan is written to avoid.
        for name, value in (("paged", self.paged),):
            if value:
                raise NotImplementedError(
                    f"{name}=True is not implemented by the backward dQ kernel"
                )
        if meta.window and not meta.causal:
            # `make_traits` says the same thing, and this repeats it only so the
            # message names the *kernel* rather than the traits constructor: a
            # window is a left bound on top of the causal right one, and
            # dropping it silently would return dense attention -- right shape,
            # finite, wrong.
            raise ValueError(
                "window=True requires causal=True; it is a left bound on top of the causal one"
            )
        if self.num_kv_splits != 1:
            raise NotImplementedError(
                "split-K is out of scope for every backward kernel"
            )
        for name, value in (
            ("d_stages", self.d_stages),
            ("qk_shards", self.qk_shards),
            ("vo_shards", self.vo_shards),
        ):
            if value != 1:
                # **Refused, because the body does not implement it and the
                # inherited helpers half-do.** `D_STAGES > 1` makes
                # `ParityGemmHelper.qk` reduce over one stage of the head dim
                # and `pv` write one stage of the accumulator, and this loop
                # never advances the stage -- so the answer is finite, the
                # right shape, and computed over a fraction of `d`. The
                # forward's default policy turns `d_stages` on at
                # block_dmodel > 256, which is exactly the rungs B3 added.
                raise NotImplementedError(
                    f"{name}={value} is described by the traits but not implemented by the dQ body: it "
                    "would reduce over part of the head dim and return a finite wrong answer. The wide "
                    "rungs do not need it -- head_dim 512 is 133 KB of LDS unstaged."
                )
        if self.block_dmodel_v != self.block_dmodel:
            # Not the same thing as an asymmetric *head_dim*, which B3 does
            # serve: this is an asymmetric compiled *tile*, which would give
            # GEMM3's K reads and the dQ store two different `D_CHUNKS`. B2
            # refused it as part of a wider refusal; keep the narrow one.
            raise NotImplementedError(
                f"block_dmodel_v {self.block_dmodel_v} != block_dmodel {self.block_dmodel}: dQ's output "
                "width is the *qk* extent, so a second tile width has nothing here to describe"
            )

        if self.bias_stage not in (None, "direct"):
            raise ValueError(
                f"bias_stage is 'direct' or left unset, got {self.bias_stage!r}"
            )
        if self.bias_stage is not None and not meta.bias:
            raise ValueError("bias_stage applies to bias builds only")
        self.staging_shape()
        self._check_helpers_support_geometry()
        num_kv_heads = (
            meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        )
        shape = (16, 16, 32) if (self.mfma_rows or 32) == 16 else (32, 32, 16)
        stage = (
            "lds"
            if (
                meta.bias
                and self.bias_stage is None
                and (self.mfma_rows or 32) == 32
                and self.block_n == 64
            )
            else ""
        )
        traits = make_bwd_dq_traits(
            mfma_shape=shape,
            store_db=bool(self.store_db),
            bias_stage=stage,
            hdim_vo_floor=self._hdim_vo_floor(meta),
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
            dtype_str=meta.dtype_str,
            waves_per_eu=self.waves_per_eu,
            daz=self.daz,
            lazy_rescale=self.lazy_rescale,
            setprio=self.setprio,
            stagger=self.stagger,
            bias=meta.bias,
            dropout=meta.dropout,
            lpt_tile_order=self.lpt_tile_order,
            num_kv_splits=1,
            # No `varlen` or `cross_seqlen`: `make_traits` emits both as
            # constants now. `CROSS_SEQLEN` reaches less here than in the
            # forward anyway -- see `BwdDqKernelContext.compute_active_guard`,
            # which drops the `causal_end_raw_i32 > 0` term because a Q block
            # with an empty causal region walks zero tiles and stores its zero
            # seed, so this kernel needs no zeroing path.
            paged=False,
            kv_cache_layout=self.kv_cache_layout,
            kv_vectorized=False,
            return_lse=False,
        )
        if (
            stage
            and traits.LDS_KV_TOTAL_SIZE * traits.BF16_BYTES + traits.BIAS_LDS_BYTES
            > self.LDS_CAP_BYTES
        ):
            traits = replace(
                traits, BIAS_STAGE=""
            )  # no room for the staged tile beside K/V: direct loads
        # One K-pitch region plus one V-pitch region, which is what
        # `make_bwd_dq_traits` cut `LDS_KV_TOTAL_SIZE` down to. Checked against
        # the same cap the forward uses, because the cap is the hardware's.
        lds_bytes = traits.LDS_KV_TOTAL_SIZE * traits.BF16_BYTES + traits.BIAS_LDS_BYTES
        if lds_bytes > self.LDS_CAP_BYTES:
            raise ValueError(
                f"KV staging needs {lds_bytes} B of LDS, over the {self.LDS_CAP_BYTES} B cap, for "
                f"block_dmodel {self.block_dmodel} at BLOCK_N {self.block_n}"
            )
        return traits


def bwd_dq_knobs(arch: str = "gfx950", **overrides) -> BwdDqKnobs:
    """The knob object for the backward dQ kernel on `arch`.

    Mirrors `fmha_tuning_gfx950.fmha_knobs`, including its arch-*prefix* match
    so a full `gcnArchName` -- "gfx950:sramecc+:xnack-" -- works unstripped.
    """
    base = arch.split(":")[0].lower() if arch else ""
    if not base.startswith("gfx950"):
        raise ValueError(f"the backward dQ kernel is gfx950-only, got arch {arch!r}")
    known = {f.name for f in fields(BwdDqKnobs)}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"unknown BwdDqKnobs field(s): {sorted(unknown)}")
    return BwdDqKnobs(**overrides)


# Defaults the policy has no shape-dependent opinion about. `lpt_tile_order`
# and the causal-only schedule knobs are inert in a dense build; they are
# spelled out anyway so a resolved object never carries a `None` past
# `resolve`, which is what `Gfx950Knobs._checked_modes` reads.
_BWD_DQ_FALLBACK = BwdDqKnobs(
    # **Deliberately absent**: `waves_per_eu`. The forward pins 2 here; this
    # kernel wants 1 and `_with_wave_geometry` says why, with the measurement.
    # Left `None` so that step decides it and a caller can still override.
    daz=True,
    lazy_rescale=True,
    setprio=True,
    stagger=True,
    lpt_tile_order=False,
    paged=False,
    kv_cache_layout="linear",
    num_kv_splits=1,
    return_lse=False,
    strides_constexpr=False,
    store_db=False,
)
