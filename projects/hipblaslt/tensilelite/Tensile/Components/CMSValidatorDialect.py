################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""
CMS validator architecture dialect.

A ``ValidatorDialect`` bundles the architecture-specific knobs the CMS
validator passes consult, so the same passes can target CDNA 4 (MFMA,
DTL=1, wave64) and RDNA 3.5 (WMMA, DTL=0, wave32) without per-pass
branching. ``resolve_dialect(kernel)`` picks the right dialect from
``(IsaVersion, WavefrontSize, DirectToLds)``; the passes in
``CMSValidator.py`` read their per-architecture constants from the
returned dialect rather than hard-coding them.

Two dialects are registered:

* ``CDNA4_DIALECT`` mirrors the CDNA 4 module-level constants in
  ``CMSValidator.py``. Until those module-level constants are removed,
  both copies must stay in sync.
* ``RDNA35_WMMA_DIALECT`` covers gfx1151 wave32 DTL=0. Numeric
  ``timing`` fields are calibrated against gfx1151 hardware
  microbenchmarks; hazard semantics follow the RDNA 3.5 ISA reference
  (sections 2.1, 3.4.5, 5.6, 6.3-6.8, 7.9.1, 10.8, 12.5, 16.1, 16.5).
  gfx1150/1152/1153 fall back to ``CDNA4_DIALECT`` until those steps
  are characterized (see ``_is_rdna35_kernel``).

Adding a new architecture profile means (a) declaring a new
``ValidatorDialect`` and (b) extending ``resolve_dialect`` to dispatch
to it. The validator passes themselves do not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple

if TYPE_CHECKING:
    from Tensile.Components.CMSValidator import Timeline


# ---------------------------------------------------------------------------
# Sub-dialects: each one owns a logically cohesive set of constants.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuadCycleModel:
    """Quad-cycle timing parameters for matrix instructions.

    Values are in CDNA 4 quad-cycle units (1 quad = 4 SCLK ticks). RDNA
    SCLK measurements are translated as ``quad = ceil(SCLK / 4)``.
    """
    # Minimum quad-cycles between a CVT pack finishing issue and an MFMA/
    # WMMA that consumes its result.
    cvt_before_mfma: int
    # Minimum quad-cycles between a 4x4 MFMA pack finishing issue and the
    # CVT1 pack that consumes its result.
    mfma_4x4_before_cvt1: int
    # Quad-cycles for a standard matrix instruction to finish after issue.
    standard_mfma_finish: int
    # Quad-cycles for a 4x4 MFMA pack to finish after issue.
    mfma_4x4_finish: int
    # Minimum gap (in quad-cycles) before type-switching out of a standard
    # matrix instruction without a 1-cycle bump.
    type_switch_threshold_from_standard: int
    # Same, but measured out of a 4x4 MFMA pack.
    type_switch_threshold_from_4x4: int


@dataclass(frozen=True)
class PackGraph:
    """TF32 pack-dependency-graph shape and size constants.

    Only consulted when ``kernel['UseF32XEmulation']`` or
    ``kernel['UseMFMAF32XEmulation']`` is set. ``RDNA35_WMMA_DIALECT``
    populates these with ``-1`` poison sentinels because the RDNA 3.5
    pack graph has not been hw-calibrated. ``resolve_dialect`` rejects
    TF32-emulation kernels on RDNA 3.5 with ``UnsupportedKernelError``
    so the poison values are unreachable on the production codegen
    path; an accidental read (e.g. via a future call site that bypasses
    ``resolve_dialect``) crashes immediately on the first range / modulo
    / index lookup rather than silently applying uncalibrated CDNA 4
    timings.
    """
    # Size of a regular TF32 pack group (4 CVT0 + 16 middle + 4 CVT1).
    group_size_tf32: int
    # Size of a 4x4-MFMA-TF32 pack group (4 CVT0 + 2 MFMAs + 4 CVT1).
    group_size_tf32_4x4: int
    # Half-open index ranges within a regular TF32 group.
    tf32_cvt0_end: int
    tf32_middle_16_start: int
    tf32_middle_16_end: int
    # Half-open index range for 4x4 MFMAs within a 4x4-MFMA-TF32 group.
    tf32_4x4_mfma_start: int
    tf32_4x4_mfma_end: int
    # VGPRs per conversion group in TF32 emulation.
    vgprs_per_conversion_group: int
    # Matrix instructions per (A-tile, B-tile) pair.
    mfmas_per_tile_tf32: int
    mfmas_per_tile_bf16: int


@dataclass(frozen=True)
class ScalarClusterModel:
    """SCC-usage clustering for ``verify_scc_overlap``.

    On CDNA 4 the pass enforces the hardware NOP rule between SCC
    producers and consumers. On RDNA 3.5 the same interval-template
    check is reused as a data-flow integrity check (no intervening SCC
    writer between an ``S_ADD_U32`` / ``S_ADDC_U32`` producer/consumer
    pair); the RDNA 3.5 ISA does not require ``S_NOP`` for SCC W->R
    correctness (ISA section 5.6).
    """
    # GRInc cluster shape when ``kernel['Use64bShadowLimit']`` is True.
    interval_sizes_shadow_limit: Tuple[int, ...]
    # GRInc cluster shape when ``kernel['Use64bShadowLimit']`` is False.
    interval_sizes_no_shadow_limit: Tuple[int, ...]
    # Whether the validator should consult GRA/GRB's ``m0`` sub-indices.
    # CDNA DTL=1 embeds an SCC writer inside GR; RDNA DTL=0 GRs are
    # plain VMEM loads with no SCC write, so this is False there.
    # Required (no default) so a third architecture profile cannot
    # silently inherit CDNA semantics by omission.
    check_gr_m0_updates_when_dtl: bool


# ---------------------------------------------------------------------------
# Dialect container.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatorDialect:
    """Architecture-specific validator configuration.

    Fields:
        name: human-readable identifier (e.g. "CDNA4" / "RDNA35-WMMA-DTL0").
        timing: quad-cycle timing model.
        pack_graph: TF32 pack group layout.
        scc_cluster: SCC-usage cluster shape for ``verify_scc_overlap``.
        timeline_factory: optional override for the ``Timeline``
            constructor. ``None`` means "use the default ``Timeline``
            class" (CDNA 4 behavior).
        matrix_inst_label: human-readable name of the matrix-instruction
            family ("MFMA" / "WMMA"). Diagnostics only; validator
            internals key on ``isinstance(..., MatrixInst)``.
        gr_must_follow_grinc_in_same_loop: when True, each GR in a loop
            iteration must be issued after the last GRInc in the same
            iteration (CDNA 4 DTL=1: GRInc writes ``m0`` and the
            following buffer-load reads it). When False, the PGR=2
            DTL=0 pattern is allowed: GRs in iteration N use the
            scalar buffer address prepared by iteration N-1's GRInc,
            so GRInc and GR may interleave freely within one iteration.
            Cross-loop SALU data-flow is enforced separately by the
            scheduler's ``vmfmaIndex`` ordering.
        lr0_consumer_half_offset / lr1_consumer_half_offset: LR consumer
            phase, expressed in units of ``num_vmfma // 2`` added to
            the per-LR base column-major tile index. Both CDNA 4 and
            the gfx1151 RDNA 3.5 schedules use the convention where
            LR0 feeds the second half of the current iteration and LR1
            feeds the first half of the next iteration. Carried in the
            dialect so the consumer-index computation in
            ``_transform_index_standard`` and
            ``_transform_index_with_force_unroll_sub_iter`` stays
            architecture-agnostic.
        sync_insert_last: when False (CDNA 4), SYNC (SWaitCnt /
            SBarrier) instructions are inserted into the timeline
            BEFORE any other instruction sharing the same
            ``vmfma_index``. When True (RDNA 3.5 WMMA), SYNC is
            inserted LAST in each vmfma bucket so ``apply_swaits``'s
            backward walk also considers same-vmfma LR / GR ops as
            "issued before" the SWaitCnt. The kernel writer still
            emits instructions in ``optSchedule.keys()`` order; this
            flag only changes the validator's internal timeline.
        gr_must_start_after_lr0s: CDNA 4 DTL=1 GRs write (DDR -> LDS)
            into the same LDS block LR0 reads from, so each GR must
            follow the last same-iteration LR0. RDNA 3.5 DTL=0 GRs
            are plain ``buffer_load`` to VGPRs with no LDS interaction,
            so the constraint is dropped.
        gr_finish_before_lr: CDNA 4 DTL=1 invariant that GR writes
            land in the LDS block the next-iter LR1 will read. RDNA
            3.5 DTL=0 routes the LR1 dependency through a separate
            LocalWrite stream with its own handshake, so the
            constraint is dropped.
        stream_length_strict_equality: when True (CDNA 4 DTL=1,
            wave64), ``VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS``
            requires ``len(authored) == len(idMap)`` for every
            stream. When False (RDNA 3.5 DTL=0, wave32), authored
            load/store slots represent bundles of N physically-issued
            ops where N is a structural function of
            ``(MIWaveTile, LocalReadVectorWidth, bpe)``, and the
            invariant becomes
            ``idmap_len % authored_len == 0 and authored_len > 0``.
    """
    name: str
    timing: QuadCycleModel
    pack_graph: PackGraph
    scc_cluster: ScalarClusterModel
    timeline_factory: Optional[Callable] = None
    matrix_inst_label: str = "MFMA"
    gr_must_follow_grinc_in_same_loop: bool = True
    lr0_consumer_half_offset: int = 1
    lr1_consumer_half_offset: int = 2
    sync_insert_last: bool = False
    gr_must_start_after_lr0s: bool = True
    gr_finish_before_lr: bool = True
    stream_length_strict_equality: bool = True


# ---------------------------------------------------------------------------
# CDNA 4 dialect: mirrors the module-level constants in CMSValidator.py.
# Until those module-level constants are removed, both copies must stay
# in sync. TODO: refactor CMSValidator.py to read through the dialect so
# this duplication can be retired.
# ---------------------------------------------------------------------------

def _cdna4_timeline_factory(instruction_names_to_add, code_path, schedule_info, kernel, dialect):
    """Construct a ``CDNA4DTLTimeline`` from a dialect-level factory hook.

    Imported lazily to break the CMSValidator -> CMSValidatorDialect cycle.
    """
    from Tensile.Components.CMSValidator import CDNA4DTLTimeline
    return CDNA4DTLTimeline(instruction_names_to_add, code_path, schedule_info, kernel, dialect)


def _rdna35_wmma_timeline_factory(instruction_names_to_add, code_path, schedule_info, kernel, dialect):
    """Construct an ``RDNA35WMMATimeline`` from a dialect-level factory hook."""
    from Tensile.Components.CMSValidator import RDNA35WMMATimeline
    return RDNA35WMMATimeline(instruction_names_to_add, code_path, schedule_info, kernel, dialect)


CDNA4_DIALECT = ValidatorDialect(
    name="CDNA4",
    timing=QuadCycleModel(
        cvt_before_mfma=2,                          # == CMSValidator.QUAD_CYCLES_CVT_BEFORE_MFMA
        mfma_4x4_before_cvt1=5,                     # == CMSValidator.QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1
        standard_mfma_finish=3,                     # == CMSValidator.QUAD_CYCLES_STANDARD_MFMA_FINISH
        mfma_4x4_finish=1,                          # == CMSValidator.QUAD_CYCLES_MFMA_4X4_FINISH
        type_switch_threshold_from_standard=5,      # == CMSValidator.MFMA_TYPE_SWITCH_THRESHOLD_FROM_STANDARD
        type_switch_threshold_from_4x4=3,           # == CMSValidator.MFMA_TYPE_SWITCH_THRESHOLD_FROM_4X4
    ),
    pack_graph=PackGraph(
        group_size_tf32=24,                         # == CMSValidator.PACK_GROUP_SIZE_TF32
        group_size_tf32_4x4=10,                     # == CMSValidator.PACK_GROUP_SIZE_TF32_4X4
        tf32_cvt0_end=4,                            # == CMSValidator.TF32_CVT0_END
        tf32_middle_16_start=4,                     # == CMSValidator.TF32_MIDDLE_16_START
        tf32_middle_16_end=20,                      # == CMSValidator.TF32_MIDDLE_16_END
        tf32_4x4_mfma_start=4,                      # == CMSValidator.TF32_4X4_MFMA_START
        tf32_4x4_mfma_end=6,                        # == CMSValidator.TF32_4X4_MFMA_END
        vgprs_per_conversion_group=8,               # == CMSValidator.VGPRS_PER_CONVERSION_GROUP
        mfmas_per_tile_tf32=3,                      # == CMSValidator.MFMAS_PER_TILE_TF32
        mfmas_per_tile_bf16=1,                      # == CMSValidator.MFMAS_PER_TILE_BF16
    ),
    scc_cluster=ScalarClusterModel(
        interval_sizes_shadow_limit=(3, 2, 2, 2),
        interval_sizes_no_shadow_limit=(3, 2, 1),
        check_gr_m0_updates_when_dtl=True,
    ),
    timeline_factory=_cdna4_timeline_factory,
    matrix_inst_label="MFMA",
)


# ---------------------------------------------------------------------------
# RDNA 3.5 WMMA dialect (gfx1151, wave32, DTL=0).
#
# Numeric timing fields are calibrated against gfx1151 hardware. Where
# hardware disagrees with the LLVM ``GFX11SpeedModel`` cycle counts,
# hardware wins.
#
# Under DTL=0, GR is a plain ``buffer_load`` to VGPRs with no LDS or
# ``m0`` interaction, so the three CDNA 4 GR-ordering hazards
# (``gr_must_follow_grinc_in_same_loop``, ``gr_must_start_after_lr0s``,
# ``gr_finish_before_lr``) do not exist on this dialect; suppressing
# them is required for the gfx1151 schedule library.
# ---------------------------------------------------------------------------

RDNA35_WMMA_DIALECT = ValidatorDialect(
    name="RDNA35-WMMA-DTL0",
    timing=QuadCycleModel(
        # CVT -> WMMA chain shows no measurable stall beyond the CVT's
        # natural 5-cycle writeback. ceil(5/4) = 2 quads.
        cvt_before_mfma=2,

        # No 4x4 WMMA variant on RDNA 3.5; ``resolve_dialect`` rejects
        # TF32 emulation before any pass reads this. Sentinel 0 makes
        # an accidental read visibly wrong.
        mfma_4x4_before_cvt1=0,

        # Back-to-back dependent v_wmma_f32_16x16x16_f16 measures
        # ~34 SCLK / op on a single wave32. ceil(34/4) = 9 quads;
        # the validator formula ``mfma_free_at = issue + 1 +
        # standard_mfma_finish`` then gives ``finish = 9 - 2 = 7``
        # (subtracting one quad each for the producer and consumer
        # issue slots). The gfx1151 schedules carry no MFMA -> MFMA
        # accumulator chain through the issue model, so the value is
        # currently advisory; if a future schedule does, this is the
        # number to revisit.
        standard_mfma_finish=7,

        # No 4x4 WMMA variant; sentinel 0 (see mfma_4x4_before_cvt1).
        mfma_4x4_finish=0,

        # f16 <-> bf16 alternation shows zero penalty on the serial-
        # dep latency path. ISA section 7.9.1 allows "may stall"
        # without a numeric cycle count; hardware settles the
        # ambiguity at 0.
        type_switch_threshold_from_standard=0,

        # No 4x4 WMMA variant; sentinel 0.
        type_switch_threshold_from_4x4=0,
    ),
    pack_graph=PackGraph(
        # All current gfx1151 CMS schedules are BF16/FP16 HHS and do
        # not exercise any TF32 pack-graph field. ``resolve_dialect``
        # rejects TF32 emulation on RDNA 3.5 so these values are never
        # read; sentinel -1 makes any accidental consumer (range
        # iteration, modulo, indexing) blow up immediately rather than
        # silently apply uncalibrated CDNA 4 timings.
        # TODO: replace with hw-calibrated values before enabling TF32
        # emulation on RDNA 3.5.
        group_size_tf32=-1,
        group_size_tf32_4x4=-1,
        tf32_cvt0_end=-1,
        tf32_middle_16_start=-1,
        tf32_middle_16_end=-1,
        tf32_4x4_mfma_start=-1,
        tf32_4x4_mfma_end=-1,
        vgprs_per_conversion_group=-1,
        mfmas_per_tile_tf32=-1,
        mfmas_per_tile_bf16=-1,
    ),
    scc_cluster=ScalarClusterModel(
        # The PGR=2 64-bit buffer-address increment lowers to a 9-op
        # SALU sequence on RDNA 3.5:
        #   cluster 0: s_cmp_eq_u32 + s_cselect_b32 + s_cselect_b32  (3 SCC ops)
        #   cluster 1: s_add_u32    + s_addc_u32                     (2 SCC ops)
        #   cluster 2: s_sub_u32    + s_subb_u32                     (2 SCC ops)
        #   cluster 3: s_cmp_eq_u32 + s_cselect_b32                  (2 SCC ops)
        # ISA section 3.4.5 defines SCC and section 16.1 fixes
        # per-instruction SCC semantics. The mnemonics + SCC semantics
        # are ISA-stable across GCN/CDNA/RDNA, so the cluster shapes
        # match CDNA 4. The pass interpretation differs (data-flow
        # integrity, not a hardware NOP rule); see the
        # ``ScalarClusterModel`` docstring.
        interval_sizes_shadow_limit=(3, 2, 2, 2),
        interval_sizes_no_shadow_limit=(3, 2, 1),
        # DTL=0: GRA / GRB are plain wave32 buffer_loads with no
        # embedded m0 pointer-update SCC writer, so
        # ``verify_scc_overlap`` skips the GR streams.
        check_gr_m0_updates_when_dtl=False,
    ),
    timeline_factory=_rdna35_wmma_timeline_factory,
    matrix_inst_label="WMMA",
    # PGR=2 DTL=0 schedules interleave GRInc_N and GR_N within the same
    # mainloop iteration: the GRA ops read scalar addresses prepared by
    # iteration N-1's GRInc, and iteration N's GRInc prepares addresses
    # for iteration N+1's GRA. There is no same-loop GRInc -> GR
    # ordering requirement on RDNA 3.5.
    gr_must_follow_grinc_in_same_loop=False,
    # gfx1151 schedules use the CDNA 4 LR convention (LR0 feeds the
    # second half of the current iteration; LR1 feeds the first half
    # of the next iteration), so the offsets stay at the defaults.
    lr0_consumer_half_offset=1,
    lr1_consumer_half_offset=2,
    # gfx1151 emission order at sub-iteration boundaries puts SWaitCnt
    # after the same-boundary LRs / GRs:
    #     ds_load_*                # same-boundary LRs
    #     buffer_load_*            # same-boundary GRs (if any)
    #     s_waitcnt lgkmcnt(N)     # "wait for dependent lr"
    #     s_barrier                # optional; elided when wave32-only
    #     v_wmma_f32_16x16x16_f16  # the consuming WMMA
    # SYNC is inserted LAST in each vmfma bucket so SWaitCnts attribute
    # guaranteed-by times to same-vmfma LRs / GRs that are genuinely
    # in-flight when the SWaitCnt fires.
    sync_insert_last=True,
    # DTL=0 GR is a plain ``buffer_load`` to VGPRs; LDS traffic goes
    # through a separate LocalWrite stream with its own wait / barrier
    # discipline. The two CDNA 4 GR / LDS ordering hazards therefore
    # do not apply.
    gr_must_start_after_lr0s=False,
    gr_finish_before_lr=False,
    # wave32 DTL=0: authored load / store slots represent bundles of N
    # physically-issued ``ds_load`` / ``ds_store`` / ``buffer_load`` ops
    # where N depends on tile geometry. Strict equality would fail
    # every gfx1151 schedule; divisibility is the correct invariant.
    stream_length_strict_equality=False,
)


# ---------------------------------------------------------------------------
# Kernel -> dialect dispatch.
# ---------------------------------------------------------------------------

def _is_rdna35_kernel(kernel: "object") -> bool:
    """Detect RDNA 3.5 WMMA kernels.

    Matches gfx1151 (RDNA 3.5.1) with wave32 and ``DirectToLds=0``. The
    predicate is intentionally narrower than the architecture family:
    gfx1150/1152/1153 share the (11, 5, *) tuple but the timing
    constants in ``RDNA35_WMMA_DIALECT`` were only calibrated on
    gfx1151. Widen this check once those steps are characterized.

    Returns ``False`` on any attribute error or missing field so a
    malformed kernel falls back to the CDNA 4 dialect (the safer
    default for the historical caller set).
    """
    try:
        isa = kernel["ISA"]
    except (KeyError, TypeError):
        return False
    try:
        major, minor, step = isa[0], isa[1], isa[2]
    except (TypeError, IndexError):
        return False
    if (major, minor, step) != (11, 5, 1):
        return False
    try:
        wf = kernel["WavefrontSize"]
    except (KeyError, TypeError):
        return False
    if wf != 32:
        return False
    try:
        dtl = kernel["DirectToLds"]
    except (KeyError, TypeError):
        return False
    if dtl:
        return False
    return True


class UnsupportedKernelError(ValueError):
    """Raised when ``resolve_dialect`` is asked to pick a dialect for a
    kernel configuration that the validator cannot soundly model.

    Preferred over returning a sentinel dialect: the failure aborts
    ``customMainLoopSchedule`` at codegen time with a diagnostic
    pointing at the exact kernel attribute that triggered the guard,
    instead of silently passing an uncalibrated validation.
    """


def resolve_dialect(kernel) -> ValidatorDialect:
    """Return the ``ValidatorDialect`` that applies to ``kernel``.

    Returns ``RDNA35_WMMA_DIALECT`` for gfx1151 wave32 DTL=0 kernels
    and ``CDNA4_DIALECT`` for everything else (including
    gfx1150/1152/1153 until those steps are characterized). The CDNA 4
    fallback is the safe default since every historical caller of the
    validator passed a CDNA 4 kernel.

    Raises ``UnsupportedKernelError`` when an RDNA 3.5 kernel requests
    TF32 software emulation (``UseF32XEmulation`` or
    ``UseMFMAF32XEmulation``). The TF32 ``pack_graph`` constants in
    ``RDNA35_WMMA_DIALECT`` are poisoned sentinels (-1) and have not
    been calibrated against RDNA 3.5 hardware; running
    ``add_pack_constraints`` with them would crash on the first range
    iteration / modulo / index, which is the desired behavior. The
    explicit guard here surfaces a cleaner diagnostic than the
    ``IndexError`` / ``ZeroDivisionError`` that would otherwise fire
    downstream.
    """
    if _is_rdna35_kernel(kernel):
        def _get(name):
            try:
                return kernel[name]
            except (KeyError, TypeError):
                return False
        if _get("UseF32XEmulation") or _get("UseMFMAF32XEmulation"):
            raise UnsupportedKernelError(
                "RDNA 3.5 (gfx1151) TF32 emulation pack-graph is not "
                "hw-calibrated. Kernel requests "
                f"UseF32XEmulation={_get('UseF32XEmulation')!r}, "
                f"UseMFMAF32XEmulation={_get('UseMFMAF32XEmulation')!r}. "
                "The RDNA35_WMMA_DIALECT.pack_graph constants are poisoned "
                "sentinels; running the pack passes against them would crash "
                "on the first index/modulo. Disable TF32 emulation on this "
                "kernel or calibrate the RDNA 3.5 pack graph before using "
                "this path."
            )
        return RDNA35_WMMA_DIALECT
    return CDNA4_DIALECT
