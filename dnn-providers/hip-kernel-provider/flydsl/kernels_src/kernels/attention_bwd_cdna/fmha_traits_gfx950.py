# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Parity-side dualwave traits, with the tile geometry as parameters.

`_make_dualwave_swp_traits` hardcodes `num_waves=8`, `block_m=256`,
`block_n=64` and derives the D-axis staging granule from a fixed 128-byte row.
Those four numbers are the entire difference between the tile families (see
`Gfx950Knobs._with_wave_geometry`), so a constructor that fixes them can serve
exactly one. This is that constructor with them opened up.

--- How this is kept honest -----------------------------------------------

Every derivation here is a transcription of the production one, and
`assert_matches_production` checks that claim the only way that means anything:
by building both for a range of head_dims and comparing **every field**. It
is not called from within the tree; run it by hand after touching a derivation.

**What that check cannot tell you** is whether a *parameterization* is right,
only whether it reproduces family A. Several constants have more than one
plausible formula that coincide at granule 64 (`K_LDS_TO_REG_N_STRIP_STRIDE`,
the `V_LDS_TO_REG_*` offsets, the `*_PER_BAND` counts). They are the first
suspects if a new family produces plausible but wrong numbers; each family
that ships is built and measured, and a geometry no family builds has not
been exercised.
"""

from dataclasses import dataclass, fields

from kernels.attention_bwd_cdna.gfx950_standalone import dualwave

__all__ = ["make_traits", "assert_matches_production", "ParityDualwaveTraits"]


@dataclass(frozen=True)
class ParityDualwaveTraits(dualwave.DualwaveSwpTraits):
    """Production's traits plus the two axes that let a tile exceed head_dim 256.

    A subclass rather than new fields on `DualwaveSwpTraits`, because that
    dataclass lives in `flash_attn_utils.py` and is shared by four production
    kernels. Subclassing is free here: the parent is `frozen=True` with no
    defaulted fields, so added fields need defaults and those defaults are
    exactly the "behave like today" values. `assert_matches_production`
    iterates the *parent's* fields, so it keeps pinning family A to production
    without knowing these exist.

    Both default to 1, and **1 must mean the kernel is unchanged** -- every
    construct they gate sits behind `const_expr(... > 1)` so a default build
    traces to identical IR. That is what makes the four working rungs safe.

    - `D_STAGES` -- how many passes the KV tile's D axis is staged through LDS
      in. LDS is `BLOCK_N * head_dim * ~8.3 B` and the cap is 163840, so
      head_dim 384 (204288 B) and 512 (272384 B) do not fit in one pass. It
      also bounds the K/V register window, since only one stage is live.
    - `VO_SHARDS` -- how many waves split the *output* D axis of one Q tile.
      Wave *s* accumulates only `O[:, slice_s]`, so O drops by the shard count.
      Every wave still computes the whole S, which is why this needs **no
      cross-wave reduction at all** -- the shards never have to agree on
      anything, they just write disjoint columns. The price is that QK is
      recomputed per shard.
    - `QK_SHARDS` -- additionally splits the *reduction* D axis, so wave *s*
      holds only `Q[:, slice_s]` and computes a partial S that must then be
      summed across shards through LDS. Strictly more powerful than
      `VO_SHARDS` (it is the only thing that shrinks Q) and strictly more
      machinery. Not implemented yet; `VO_SHARDS` alone is what makes head_dim
      512 fit, because O is 256 VGPRs against Q's 128.

    The two are separate fields rather than one number because they buy
    different things at different prices, and conflating them would hide that
    the cheap one is sufficient today.
    """

    D_STAGES: int = 1
    QK_SHARDS: int = 1
    VO_SHARDS: int = 1
    STAGE_DIM: int = (
        0  # head_dim // D_STAGES; `make_traits` always sets it, 0 only on a hand-built instance
    )
    K_STEPS_PER_STAGE: int = 0
    D_CHUNKS_PER_STAGE: int = 0
    D_CHUNKS_PER_STAGE_SHARD: int = 0
    Q_TILES: int = 0  # NUM_WAVES // VO_SHARDS

    # How a granule subdivides, for the three sites that hardcoded granule 64.
    # Each is 1 at the smallest legal granule and equals family A's literal at
    # 64, which is why the production code could get away with a constant:
    #
    #   SMEM_D_BUCKETS    lanes covering one token's granule   64/8  = 8
    #   K_STEPS_PER_BAND  16-wide QK steps inside a granule    64/16 = 4
    #   D_CHUNKS_PER_BAND 32-wide PV chunks inside a granule   64/32 = 2
    SMEM_D_BUCKETS: int = 0
    K_STEPS_PER_BAND: int = 0
    D_CHUNKS_PER_BAND: int = 0

    # Generalized sliding-window attention: AOTriton's `CAUSAL_TYPE == 3`.
    #
    # **A window is causal plus a left bound**, which is why this is a flag on
    # top of `CAUSAL` rather than a mode beside it. The existing causal path
    # already masks `col <= row + delta` with `delta = seqlen_kv - seqlen_q`,
    # and that is exactly `window_right` under bottom-right alignment -- so
    # `WINDOW` re-points `delta` at the resolved right bound and adds the
    # second comparison for the left one.
    #
    # The bounds themselves are *runtime* i32 arguments, not fields here: they
    # carry sentinels that must be resolved against each sequence's own
    # lengths, which under varlen differ per sequence. See `resolve_window`.
    WINDOW: bool = False

    # AOTriton's `BIAS_TYPE`: 0 = none, 1 = a (B, H, Sq, Sk) matrix added to
    # the scores before the softmax. A *build* axis, so a build without bias
    # emits nothing at all -- the KV loop is latency-bound and a bias that cost
    # anything when unused would be paid by every caller who does not want one.
    BIAS_TYPE: int = 0

    # AOTriton's `ENABLE_DROPOUT`. A build axis for the same reason as
    # `BIAS_TYPE`: a build without dropout emits no PRNG at all.
    ENABLE_DROPOUT: bool = False

    # Dispatch the causal Q blocks longest-first. A bijection over the same
    # index set, so the output is bit-identical and this is purely a
    # load-balance choice; see `ParityKernelContext.init_thread_mapping`.
    LPT_TILE_ORDER: bool = False


# Hardware constants. Not parameters: these are the wave, the DMA width and the
# MFMA shape, and a build that changed one would not be this algorithm.
WARP_SIZE = 64
DMA_BYTES = 16
BF16_BYTES = 2
VEC_KV = 8  # bf16 elements one lane moves per DMA issue (16 B)
MFMA_LANE_K = 8
K_STEP_QK = 16  # MFMA K extent
MFMA_M = 32  # MFMA M extent -- what pins ROWS_PER_WAVE, whatever BLOCK_M says
D_CHUNK = 32  # PV MFMA N extent -- the O accumulator's width
PV_K_STEP = 16
K_SUB_N = 32


def make_traits(
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    num_waves,
    block_m,
    block_n,
    granule,
    d_stages=1,
    qk_shards=1,
    vo_shards=1,
    v_half_wave=None,
    v_n_group=None,
    v_k_substep=None,
    v_dc_in_pair=None,
    causal=True,
    window=False,
    bias=False,
    dropout=False,
    lpt_tile_order=False,
    dtype_str="bf16",
    waves_per_eu=2,
    daz=True,
    lazy_rescale=True,
    setprio=True,
    stagger=True,
    debug_lazy_counts=False,
    num_kv_splits=1,
    paged=False,
    kv_cache_layout="linear",
    kv_vectorized=None,
    return_lse=False,
    xcd_swizzle=False,
):
    """Dualwave traits for an arbitrary (waves, BLOCK_M, BLOCK_N, granule)."""
    if window and not causal:
        # A window without the causal machinery has no masked region to apply
        # itself to, and silently dropping it would return dense attention --
        # the right shape, finite, and wrong. An unbounded `window_right`
        # expresses a pure left-band, so nothing is lost by requiring this.
        raise ValueError(
            "window=True requires causal=True; it is a left bound on top of the causal one"
        )
    if bias and (causal or window):
        # Undefined, not unimplemented. Causal is an attention mask with a
        # fixed pattern; a bias *is* an attention mask supplied directly, since
        # a large negative or -inf entry is how a caller spells "do not attend
        # here". Asking for both asks which wins where they disagree, and there
        # is no answer -- the same thing has been said twice in two
        # vocabularies with no rule for reconciling them. AOTriton disables the
        # combination and PyTorch's math backend raises on it; gfx1201 rejects
        # it in the same words.
        raise ValueError(
            "bias and causal/window masking are mutually exclusive: a bias already is an attention "
            "mask, so combining it with a positional one has no defined meaning. Fold the causal "
            "pattern into the bias tensor, or drop the bias"
        )
    if num_waves % vo_shards:
        raise ValueError(f"vo_shards {vo_shards} must divide num_waves {num_waves}")
    # With `vo_shards` waves sharing one Q tile, the workgroup covers
    # `num_waves // vo_shards` tiles, not `num_waves`. Rows per wave is *not*
    # what shrinks -- it is pinned at 32 by the MFMA's M extent -- so BLOCK_M
    # falls instead, and what each wave saves is D columns of O.
    q_tiles = num_waves // vo_shards
    if block_m % q_tiles:
        raise ValueError(f"BLOCK_M {block_m} does not divide across {q_tiles} Q tiles")
    if head_dim % granule:
        raise ValueError(
            f"head_dim {head_dim} is not a multiple of the granule {granule}"
        )
    if head_dim % D_CHUNK:
        raise ValueError(
            f"head_dim {head_dim} is not a multiple of the PV MFMA width {D_CHUNK}"
        )

    block_size = num_waves * WARP_SIZE
    rows_per_wave = block_m // q_tiles
    if block_n != 64 and rows_per_wave != 16:
        # `K_LDS_TO_REG_N_STRIP_STRIDE` below is the constant 256, which is the
        # layout offset of 32 tokens only at BLOCK_N 64. The 32-row bodies read
        # it for the K lo/hi pair, so any other BLOCK_N would address the wrong
        # K rows and return finite output with no error. The 16-row family
        # (`rows_per_wave == 16`) builds at BLOCK_N 32 and does not read it: its
        # LDS reads use `fmha_mfma16_gfx950`'s token addressing instead.
        raise ValueError(
            f"BLOCK_N {block_n} with {rows_per_wave} rows per wave: the 32-row K read assumes "
            f"BLOCK_N 64 (K_LDS_TO_REG_N_STRIP_STRIDE is the constant 256). Use BLOCK_N 64, or the "
            f"16-row family"
        )
    if rows_per_wave > MFMA_M:
        # **The comment above is an invariant, and this enforces it.** A wave
        # holds at most the MFMA's M extent in rows, so `BLOCK_M` is really
        # `q_tiles * MFMA_M`; a larger one builds a kernel whose helpers
        # address rows its accumulator does not have. It does not fail. A sweep over head dims
        # found twelve such points -- all `head_dim 512` at 8 waves with
        # `vo_shards > 1`, giving 64 or 128 rows per wave -- each returning
        # finite garbage at 0.15 to 0.28 relative error.
        #
        # Sharding is where this bites, because `vo_shards` divides `q_tiles`
        # while `block_m` stays whatever was pinned: a geometry that was
        # consistent unsharded silently stops being so.
        #
        # Fewer rows per wave is legal and merely wasteful -- the wave runs a
        # full 32-row MFMA and discards the rows it does not own, which the
        # same sweep measured as almost exactly proportional (8/16/32 rows ->
        # 281/563/1111 TFLOP/s at head_dim 128). So this is a ceiling, not an
        # equality, and the loss below it is left to the tuner.
        raise ValueError(
            f"BLOCK_M {block_m} over {q_tiles} Q tiles gives {rows_per_wave} rows per wave, but the "
            f"MFMA's M extent caps it at {MFMA_M}. BLOCK_M is derived: pass "
            f"{q_tiles * MFMA_M} for num_waves={num_waves}, vo_shards={vo_shards}"
        )

    k_steps_qk = head_dim // K_STEP_QK
    d_chunks = head_dim // D_CHUNK
    pv_k_steps = K_SUB_N // PV_K_STEP

    # The D axis is cut two independent ways, and they compose: `d_stages`
    # splits it *in time* (one LDS residency per pass) and `qk_shards` splits
    # it *across waves* (one Q/O slice per wave). Validated together here so an
    # illegal pair fails at the decision rather than at an address.
    if d_stages < 1 or head_dim % d_stages:
        raise ValueError(
            f"head_dim {head_dim} is not a multiple of d_stages {d_stages}"
        )
    stage_dim = head_dim // d_stages
    if stage_dim % granule:
        raise ValueError(
            f"stage extent {stage_dim} (head_dim/{d_stages}) is not a multiple of granule {granule}"
        )
    if k_steps_qk % d_stages or d_chunks % d_stages:
        raise ValueError(
            f"d_stages {d_stages} must divide both K_STEPS_QK {k_steps_qk} and D_CHUNKS {d_chunks}; "
            "a stage that splits an MFMA step has no meaning"
        )
    if vo_shards < 1 or d_chunks % (d_stages * vo_shards):
        raise ValueError(
            f"vo_shards {vo_shards} x d_stages {d_stages} must divide D_CHUNKS {d_chunks}; "
            "each (stage, shard) owns a whole number of 32-column PV output chunks"
        )
    d_chunks_per_stage_shard = d_chunks // (d_stages * vo_shards)
    if vo_shards > 1 and d_chunks_per_stage_shard % 2:
        raise ValueError(
            f"D_CHUNKS per (stage, shard) is {d_chunks_per_stage_shard}, which must be even once sharded: the "
            "LDS offset is folded into `urv_base`, and `_swizzled_v_dc_off` only decomposes that way "
            "when the shard starts on an even chunk"
        )
    if qk_shards < 1 or head_dim % qk_shards:
        raise ValueError(
            f"head_dim {head_dim} is not a multiple of qk_shards {qk_shards}"
        )
    if num_waves % qk_shards:
        raise ValueError(f"qk_shards {qk_shards} must divide num_waves {num_waves}")
    k_steps_per_stage = k_steps_qk // d_stages
    d_chunks_per_stage = d_chunks // d_stages

    gqa_group_size = num_heads // num_kv_heads
    default_stride_q_n = num_heads * head_dim
    default_stride_kv_n = num_kv_heads * head_dim

    # One DMA issue per wave moves `smem_linear_wave` elements; the granule
    # decides how that splits into (tokens, D).
    smem_linear_wave = WARP_SIZE * DMA_BYTES // BF16_BYTES
    smem_n_per_wave = smem_linear_wave // granule
    smem_n_rpt = block_n // smem_n_per_wave
    # `stage_dim`, not `head_dim`: this is the sole term through which
    # `d_stages` reaches LDS. Everything downstream (tile elems, buffer bases,
    # LDS_KV_TOTAL_SIZE) is derived from it, so one substitution sizes the
    # whole allocation to a single pass. At `d_stages == 1` it is `head_dim`
    # and every derived number is bit-for-bit what it was.
    smem_d_rpt = stage_dim // granule
    if smem_n_rpt == 0 or block_n % smem_n_per_wave:
        raise ValueError(
            f"BLOCK_N {block_n} is not a multiple of {smem_n_per_wave} tokens per DMA issue"
        )
    if smem_n_rpt % num_waves:
        raise ValueError(
            f"{smem_n_rpt} KV tile lines do not divide across {num_waves} waves"
        )

    smem_k_pad = DMA_BYTES // BF16_BYTES
    smem_v_pad = 64 // BF16_BYTES
    smem_k_line_stride = smem_linear_wave + smem_k_pad
    smem_v_line_stride = smem_linear_wave + smem_v_pad
    smem_k_tile_elems = smem_n_rpt * smem_d_rpt * smem_k_line_stride
    smem_v_tile_elems = smem_n_rpt * smem_d_rpt * smem_v_line_stride

    num_prefetch_k = 2
    kv_per_buffer = smem_k_tile_elems + smem_v_tile_elems
    lds_kv_total_size = num_prefetch_k * kv_per_buffer

    # K LDS->VGPR. `n_strip_stride` is the offset from a lane's lo pack to its
    # hi pack: half a wave of lanes further on, each holding VEC_KV elements.
    # That is 32 tokens on, which in the layout below is
    # `(32 // SMEM_N_RPT) * granule` elements -- 256 at BLOCK_N 64 for both
    # granules, and a different number at any other BLOCK_N. The constant is
    # therefore valid only there, which `make_traits` enforces above.
    k_lds_to_reg_n_strip_stride = (WARP_SIZE // 2) * VEC_KV
    k_lds_to_reg_kstep_inner_stride = K_STEP_QK
    k_lds_to_reg_kstep_outer_stride = smem_n_rpt * smem_k_line_stride

    # V LDS->VGPR. Three of these are the same thing -- "advance the KV token
    # by t" -- and the old code spelled each as a separate granule-64 identity.
    #
    # A line holds `512 // granule` token slots; slot s, line n is token
    # `s * SMEM_N_RPT + n`. So advancing t tokens moves `t // N_RPT` slots and
    # `t % N_RPT` lines:
    #
    #     tok_off(t) = (t // N_RPT) * granule + (t % N_RPT) * line
    #
    #                    t   granule 64 (n_rpt 8)   granule 32 (n_rpt 4)
    #   half_wave        4   0*64 + 4*line = 2176   1*32 + 0      =   32
    #   transpose_pair   8   1*64 + 0      =   64   2*32 + 0      =   64
    #   k_substep       16   2*64 + 0      =  128   4*32 + 0      =  128
    #
    # At granule 64 that reproduces `4 * line`, `granule` and `2 * granule`
    # exactly, which is why those three literals survived -- and why they were
    # each wrong in a different way at granule 32. The probe caught it: a lane
    # was handed tokens 4 apart where the PV MFMA's B operand needs 8, because
    # `granule` advances one *slot*, which is `N_RPT` tokens, not eight.
    #
    # `n_group` is a D offset rather than a token one, and `granule // 4`
    # coincided with `2 * VEC_KV` at 64. The probe says 16 is the one the MFMA
    # wants (lane 16 must receive D 16, not 8).
    def _tok_off(t):
        return (t // smem_n_rpt) * granule + (t % smem_n_rpt) * smem_v_line_stride

    v_lds_to_reg_half_wave_stride = _tok_off(4) if v_half_wave is None else v_half_wave
    v_lds_to_reg_lane_quad_stride = smem_v_line_stride
    v_lds_to_reg_n_group_stride = (2 * VEC_KV) if v_n_group is None else v_n_group
    v_lds_to_reg_lane_in_quad_stride = 4
    v_lds_to_reg_k_substep_stride = _tok_off(16) if v_k_substep is None else v_k_substep
    v_lds_to_reg_dchunk_pair_stride = smem_n_rpt * smem_v_line_stride
    v_lds_to_reg_dchunk_in_pair_stride = (
        D_CHUNK if v_dc_in_pair is None else v_dc_in_pair
    )
    v_lds_to_reg_transpose_pair_stride = _tok_off(8)

    kv_vec_size = DMA_BYTES // BF16_BYTES
    # Stored verbatim, `None` included -- the production constructor does the
    # same and the caller computes it. Defaulting it here would be more
    # defensive and would make this a *different* traits object, which the
    # field-by-field check would then have to be loosened to accept.

    return ParityDualwaveTraits(
        D_STAGES=d_stages,
        QK_SHARDS=qk_shards,
        VO_SHARDS=vo_shards,
        D_CHUNKS_PER_STAGE_SHARD=d_chunks_per_stage_shard,
        Q_TILES=q_tiles,
        SMEM_D_BUCKETS=granule // VEC_KV,
        K_STEPS_PER_BAND=granule // K_STEP_QK,
        D_CHUNKS_PER_BAND=granule // D_CHUNK,
        STAGE_DIM=stage_dim,
        K_STEPS_PER_STAGE=k_steps_per_stage,
        D_CHUNKS_PER_STAGE=d_chunks_per_stage,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_N_OUT=block_n,
        K_SUB_N=K_SUB_N,
        WARP_SIZE=WARP_SIZE,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim,
        K_STEP_QK=K_STEP_QK,
        K_STEPS_QK=k_steps_qk,
        D_CHUNK=D_CHUNK,
        D_CHUNKS=d_chunks,
        PV_K_STEP=PV_K_STEP,
        PV_K_STEPS=pv_k_steps,
        MFMA_LANE_K=MFMA_LANE_K,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        WINDOW=window,
        BIAS_TYPE=1 if bias else 0,
        ENABLE_DROPOUT=bool(dropout),
        LPT_TILE_ORDER=bool(lpt_tile_order),
        DTYPE_STR=dtype_str,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(lazy_rescale),
        DUALWAVE_SWP_SETPRIO=bool(setprio),
        DUALWAVE_SWP_DEBUG_LAZY_COUNTS=bool(debug_lazy_counts),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=num_kv_splits > 1,
        PAGED=bool(paged),
        # LOCAL DIVERGENCE FROM UPSTREAM: upstream keeps `varlen` and
        # `cross_seqlen` as build axes and merely defaults `varlen` on; the
        # axes are removed here, so `make_traits` takes no such arguments and
        # the tuning tables carry no such fields. Retire when upstream removes
        # the axes.
        #
        # **Constants, not choices.** This arch decodes `varlen_bits` at
        # runtime, so there is no dense build to distinguish and no `varlen=`
        # argument to pass; the two fields survive only because they belong to
        # upstream's `DualwaveSwpTraits` dataclass. `VARLEN=True` is the value
        # that makes the one upstream method still inherited from there --
        # `compute_active_guard` -- return the unconditional `q_start <
        # seqlen_q_v` this design needs.
        #
        # `CROSS_SEQLEN` follows `CAUSAL` for the same reason it stopped being a
        # knob: Q and K lengths arrive at runtime from independent arrays, so
        # no build knows whether they match, and where `seqlen_k < seqlen_q`
        # bottom-right causal leaves leading Q blocks with no live key -- which
        # must be *written* as zero, not skipped. Both of its effects are
        # branch-guarded, so a dense causal call pays scalar compares.
        VARLEN=True,
        CROSS_SEQLEN=bool(causal),
        KV_CACHE_LAYOUT=kv_cache_layout,
        KV_VECTORIZED=kv_vectorized,
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
        DMA_BYTES=DMA_BYTES,
        BF16_BYTES=BF16_BYTES,
        D_128B_SIZE=granule,
        VEC_KV=VEC_KV,
        SMEM_LINEAR_WAVE=smem_linear_wave,
        SMEM_N_PER_WAVE=smem_n_per_wave,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_PAD=smem_k_pad,
        SMEM_V_PAD=smem_v_pad,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_V_LINE_STRIDE=smem_v_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        SMEM_V_TILE_ELEMS=smem_v_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=(0, kv_per_buffer),
        DUALWAVE_SWP_V_BUF_BASE=(smem_k_tile_elems, smem_k_tile_elems + kv_per_buffer),
        K_LDS_TO_REG_N_STRIP_STRIDE=k_lds_to_reg_n_strip_stride,
        K_LDS_TO_REG_KSTEP_INNER_STRIDE=k_lds_to_reg_kstep_inner_stride,
        K_LDS_TO_REG_KSTEP_OUTER_STRIDE=k_lds_to_reg_kstep_outer_stride,
        V_LDS_TO_REG_HALF_WAVE_STRIDE=v_lds_to_reg_half_wave_stride,
        V_LDS_TO_REG_LANE_QUAD_STRIDE=v_lds_to_reg_lane_quad_stride,
        V_LDS_TO_REG_N_GROUP_STRIDE=v_lds_to_reg_n_group_stride,
        V_LDS_TO_REG_LANE_IN_QUAD_STRIDE=v_lds_to_reg_lane_in_quad_stride,
        V_LDS_TO_REG_K_SUBSTEP_STRIDE=v_lds_to_reg_k_substep_stride,
        V_LDS_TO_REG_DCHUNK_PAIR_STRIDE=v_lds_to_reg_dchunk_pair_stride,
        V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE=v_lds_to_reg_dchunk_in_pair_stride,
        V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE=v_lds_to_reg_transpose_pair_stride,
        PAGED_BT_LDS_SIZE=2048,
        DUALWAVE_SWP_RESCALE_THRESHOLD=8.0,
        KV_VEC_SIZE=kv_vec_size,
        VEC_V_ROW_STRIDE=smem_v_line_stride,
        SCHED_MFMA_MASK=0x008,
        SCHED_VALU_MASK=0x002,
        SCHED_EXP_MASK=0x400,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
        RETURN_LSE=bool(return_lse),
        XCD_SWIZZLE=bool(xcd_swizzle),
    )


def assert_matches_production(head_dims=(64, 128), **kwargs):
    """Every field must equal the production constructor's, at family A's geometry.

    The only check that makes "this is a transcription" mean anything. Run it
    across the built rungs and a spread of modes, because a single head_dim
    would not catch a term that happens to vanish there.
    """
    for head_dim in head_dims:
        mine = make_traits(
            head_dim=head_dim,
            num_waves=8,
            block_m=256,
            block_n=64,
            granule=64,
            **kwargs,
        )
        theirs = dualwave._make_dualwave_swp_traits(
            kwargs["num_heads"],
            kwargs["num_kv_heads"],
            head_dim,
            causal=kwargs.get("causal", True),
            dtype_str=kwargs.get("dtype_str", "bf16"),
            waves_per_eu=kwargs.get("waves_per_eu", 2),
            daz=kwargs.get("daz", True),
            dualwave_swp_lazy_rescale=kwargs.get("lazy_rescale", True),
            dualwave_swp_setprio=kwargs.get("setprio", True),
            dualwave_swp_debug_lazy_counts=kwargs.get("debug_lazy_counts", False),
            dualwave_swp_enable_stagger=kwargs.get("stagger", True),
            num_kv_splits=kwargs.get("num_kv_splits", 1),
            # Mirrors the two constants `make_traits` emits; see there.
            varlen=True,
            cross_seqlen=bool(kwargs.get("causal", True)),
            paged=kwargs.get("paged", False),
            kv_cache_layout=kwargs.get("kv_cache_layout", "linear"),
            kv_vectorized=kwargs.get("kv_vectorized"),
            return_lse=kwargs.get("return_lse", False),
        )
        diffs = [
            (f.name, getattr(mine, f.name), getattr(theirs, f.name))
            for f in fields(dualwave.DualwaveSwpTraits)
            if getattr(mine, f.name) != getattr(theirs, f.name)
        ]
        if diffs:
            raise AssertionError(
                f"head_dim {head_dim}: {len(diffs)} field(s) differ from production: {diffs}"
            )
