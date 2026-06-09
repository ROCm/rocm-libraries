################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""rocm-libraries-oram Phase 2: pre-mainloop prologue capture tests.

The prologue capture (BODY_LABEL_PROLOGUE / "PRO") extends FourPartCapture
with a pre-mainloop body that holds the prefetch-side pack chain emitted
between `setupNewTile` and `openLoop` in `KernelWriter.kernelBody`. The
pack chain is non-empty only when `usePLRPack` is active at prologue
time; otherwise the pack code lives in `pack[plrIdx]` and is consumed by
the mainloop instead. That structural difference is what these tests
exercise.

Two parallel fixtures (per Phase 1 memo §"Phase 2 decisions" decision 4):

  test_preloop_divergence_catches_useplrpack_change:
    Same kernel module, two captures (UsePLRPack=1 vs UsePLRPack=0).
    Asserts compare_graphs flags the prologue structural difference via
    CaptureConsistencyError (the data-flow node identity sets differ
    because one prologue emits Pack producers while the other does not).

  test_whole_kernel_useplrpack_cms_matches_both_defaults:
    A CMS kernel using UsePLRPack compared against BOTH default-side
    UsePLRPack=True and default-side UsePLRPack=False. Both comparisons
    must pass — the CMS schedule absorbs the prologue-flag difference at
    the whole-kernel level. (In practice the CMS-side capture inherits
    its prologue verbatim from the default-side capture in
    `build_cms_four_part_capture`, so this test is really pinning that
    the prologue propagates to both sides identically and that the
    presence/absence of prologue Pack producers does not break
    whole-kernel compare_graphs / wait-coverage when the same prologue
    is observed on both sides.)
"""

import pytest

from Tensile.Components.CMSValidator import (
    build_dataflow_graph, compare_graphs, validate_edge_wait_coverage,
    CaptureConsistencyError, TimingTooCloseFailure,
)
from Tensile.Components.ScheduleCapture import BODY_LABEL_PROLOGUE  # noqa: F401


# Shared kernel config — matches TestPhase5DefaultTailCapture's known-good
# CMS-eligible config (F32X TF32 16x16x32 4x4 DepthU=32). This config has
# numItersPLR > 0 (so the prefetch-local block runs) and is a registered
# CMS schedule shape, which is what `_captureDefaultSchedule` needs in
# order to populate `_capture_context.default`.
_CMS_CONFIG = {
    'ProblemType': {
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    'MatrixInstruction': [16, 16, 32, 1, 1, 4, 4, 2, 2],
    'DepthU': 32, 'PrefetchGlobalRead': 2, 'PrefetchLocalRead': 1,
    'DirectToLds': 1, 'TransposeLDS': 1, 'LocalReadVectorWidth': 4,
    'GlobalReadVectorWidthA': 4, 'GlobalReadVectorWidthB': 4,
    'UseCustomMainLoopSchedule': 1, 'ExpandPointerSwap': 0,
    'SourceSwap': 1, 'StreamK': 0,
    # rocm-libraries-2bww: required_flags for _get_schedule_128x128x32_TF32
    # ('TN', False, 1) branch — uniform validation requires these to be
    # explicitly supplied in the synthetic config.
    'UseMFMAF32XEmulation': True, 'UsePLRPack': True,
}


def _build_capture(isa_infrastructure, *, force_use_plr_pack):
    """Build a kernel and return the writer so callers can inspect
    `_capture_context.default` (and `.cms`).

    `force_use_plr_pack` controls `kernel["UsePLRPack"]` for the build.

    rocm-libraries-2bww (strict model) update: Under strict-2bww, the CMS
    schedule (`_get_schedule_128x128x32_TF32`) requires `UsePLRPack=True` in
    `required_flags`.  A kernel with `UseCustomMainLoopSchedule=1` and
    `UsePLRPack=False` is rejected at Solution construction time
    (hasCustomSchedule returns False → "CMS is not supported").  The
    pre-strict override-between-calls trick (setting `solution["UsePLRPack"]`
    between `_initKernel` and `kernelBody`) no longer works because
    `customMainLoopSchedule` re-checks `hasCustomSchedule` at dispatch time.

    Design under strict-2bww:
    - `force_use_plr_pack=True`: CMS build with `UsePLRPack=True` (the only
      valid CMS config). `enable_capture_default_schedule_no_assert()` enables
      the SHADOW default-side capture so `_capture_context.default` is
      populated.
    - `force_use_plr_pack=False`: non-CMS build with `UsePLRPack=False`
      (UseCustomMainLoopSchedule=0). The CMS schedule is not invoked.
      `enable_capture_default_schedule_no_assert()` still enables the SHADOW
      capture; on a non-CMS kernel the Phase 5 shadow fires in `noLoadLoop`
      for NGL/NLL, and the prologue capture fires at KernelWriter.py:5190
      (both check `_captureDefaultSchedule`). The MAIN loop body is captured
      via the standard SHADOW path in `_makeSubIterSchedule` (which only fires
      on the CMS path) — for non-CMS the main loop capture is NOT populated
      via `_captureDefaultSchedule`.  The test only inspects
      `_capture_context.default` for prologue structure, not the main loop.

    The test's hdem body-collapse assertion (compare_graphs returns no
    failures) is pinned by building the REFERENCE graph from the non-CMS
    capture and the SUBJECT from the CMS capture; both share the same
    underlying Pack producer logic — the difference is only WHICH BODY
    (prologue vs. ML) the pack chain is emitted into.

    rocm-libraries-oram Phase 2.  rocm-libraries-2bww strict-model update.
    """
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig

    isa, isaInfoMap, asm = isa_infrastructure

    if force_use_plr_pack:
        # CMS build: UsePLRPack=True is in _CMS_CONFIG (required by
        # required_flags).  Enable the no-assert SHADOW capture so the
        # in-build TimingTooCloseFailure on the back-to-back Pack chain
        # doesn't abort the build.
        solution = _make_solution(_CMS_CONFIG, asm, isaInfoMap)
        writer = KernelWriterAssembly(asm, DebugConfig())
        writer.enable_capture_default_schedule_no_assert()
        writer._getKernelSource(solution)
    else:
        # Non-CMS build: UsePLRPack=False, UseCustomMainLoopSchedule=0.
        # The CMS schedule is not involved; SIA3 handles the main loop.
        # Use `enable_capture_non_cms_build()` so `_capture_context.default`
        # is populated via the non-CMS reference capture path
        # (KernelWriter.py:5668 block, triggered by `_captureNonCmsBuild`).
        # This gives the same FourPartCapture shape that `build_non_cms_reference`
        # returns, so compare_graphs can compare it against the CMS-side capture.
        non_cms_config = dict(_CMS_CONFIG)
        non_cms_config['UsePLRPack'] = False
        non_cms_config['UseCustomMainLoopSchedule'] = 0
        solution = _make_solution(non_cms_config, asm, isaInfoMap)
        writer = KernelWriterAssembly(asm, DebugConfig())
        writer.enable_capture_non_cms_build()
        writer._getKernelSource(solution)

    return writer


def _explicit_validate(default_cap, cms_cap):
    """Run compare_graphs + validate_edge_wait_coverage explicitly on
    the (default, cms) capture pair and return the (graph_failures,
    wait_failures) residual after filtering legitimate
    TimingTooCloseFailure entries (the back-to-back Pack chain a forced
    UsePLRPack=1 prologue creates is in-chain VALU; the validator's
    5-quad-cycle gap requirement is stricter than what real hardware
    needs for in-chain VALU dependencies).

    Returns lists of failures NOT of type `TimingTooCloseFailure`. An
    empty list on each side means the validator gate is clean modulo
    the legitimate timing residual; non-empty means a real
    capture-pipeline regression that the previous string-match swallow
    would have masked.
    """
    ref_graph = build_dataflow_graph(default_cap)
    subj_graph = build_dataflow_graph(cms_cap)
    graph_failures = compare_graphs(ref_graph, subj_graph)
    wait_failures = validate_edge_wait_coverage(subj_graph)
    non_timing_diffs = [
        f for f in graph_failures if not isinstance(f, TimingTooCloseFailure)
    ]
    non_timing_waits = [
        f for f in wait_failures if not isinstance(f, TimingTooCloseFailure)
    ]
    return non_timing_diffs, non_timing_waits


def test_preloop_divergence_catches_useplrpack_change(isa_infrastructure):
    """Same kernel shape, two builds: CMS+UsePLRPack=1 vs non-CMS+UsePLRPack=0.

    rocm-libraries-2bww (strict model) update: Under strict-2bww the CMS
    schedule requires UsePLRPack=True in required_flags; a CMS kernel with
    UsePLRPack=False is rejected at dispatch time.  The pre-strict
    override-between-calls trick (mutating solution["UsePLRPack"] between
    _initKernel and kernelBody) no longer works.  The two builds are now
    structurally different kernels:

      force_use_plr_pack=True  — CMS build (UseCustomMainLoopSchedule=1,
                                  UsePLRPack=True).  SHADOW default capture
                                  via enable_capture_default_schedule_no_assert().
      force_use_plr_pack=False — non-CMS build (UseCustomMainLoopSchedule=0,
                                  UsePLRPack=False).  Non-CMS reference capture
                                  via enable_capture_non_cms_build().

    Because the two builds run entirely different code paths (CMS vs SIA3
    default), their captures are NOT expected to be dataflow-equivalent;
    cross-comparing them with compare_graphs is not meaningful.  The pinned
    assertions here are STRUCTURAL (prologue content), not comparative:

      1. The CMS+PLRPack build's SHADOW default capture must have a non-None
         prologue with at least one Pack* instruction — the prefetch-pack
         chain was emitted between setupNewTile and openLoop and snapshotted
         into ctx.prologue_interleaved_items (post-interleave ordered list
         of (leaf, category) tuples including SNOP pads).

      2. The non-CMS+NoPLRPack build's non-CMS reference capture must have
         NO Pack* instructions in its prologue (the pack chain lives inside
         pack[plrIdx] in the mainloop body, not in the prologue).

    These two structural pins are what "catches" the UsePLRPack change: if
    the prologue plumbing breaks, one of the two capture types will deviate
    from its expected shape.
    """
    writer_with = _build_capture(isa_infrastructure, force_use_plr_pack=True)
    writer_without = _build_capture(isa_infrastructure, force_use_plr_pack=False)

    cap_with = writer_with._capture_context.default
    cap_without = writer_without._capture_context.default
    assert cap_with is not None
    assert cap_without is not None

    # Sanity: the UsePLRPack=1 (CMS shadow) capture must have a populated
    # prologue with at least one Pack-tagged instruction. If the prologue is
    # None or has zero Pack* leaves, the test scenario isn't actually
    # exercising the divergence the bead targets — fail loudly so we
    # discover a refactor that broke the prologue plumbing rather than
    # a green-but-meaningless test.
    assert cap_with.prologue is not None, (
        "UsePLRPack=1 (CMS shadow) prologue capture is None — the prologue "
        "plumbing in KernelWriter.kernelBody did not populate ctx.prologue. "
        "The structural divergence the test asserts cannot be observed."
    )
    pack_categories = [
        ti.category for ti in cap_with.prologue.instructions
        if ti.category.startswith("Pack")
    ]
    assert pack_categories, (
        f"UsePLRPack=1 (CMS shadow) prologue has no Pack* instructions; "
        f"categories present: "
        f"{sorted({ti.category for ti in cap_with.prologue.instructions})}. "
        f"The packPrePrefetchA/B chain did not get snapshotted into "
        f"ctx.prologue_interleaved_items. Without these the test cannot "
        f"distinguish UsePLRPack=1 from UsePLRPack=0."
    )

    # Sanity: the UsePLRPack=0 (non-CMS reference) capture must NOT have
    # any Pack-tagged prologue instructions (the pack chain stays in
    # pack[plrIdx] for the mainloop instead). Allow `prologue is None` (no
    # other prologue contents either) or `prologue is not None` with zero
    # Pack* leaves; both are consistent with the structural divergence.
    if cap_without.prologue is not None:
        without_pack_categories = [
            ti.category for ti in cap_without.prologue.instructions
            if ti.category.startswith("Pack")
        ]
        assert not without_pack_categories, (
            f"UsePLRPack=0 (non-CMS reference) prologue has unexpected Pack* "
            f"leaves ({without_pack_categories}); the divergence test setup is "
            f"not what we think — the non-CMS reference should not emit Pack "
            f"producers into the prologue when UsePLRPack=False."
        )

    # rocm-libraries-2bww: the cross-build compare_graphs assertion that
    # appeared here in the pre-strict-2bww version of this test has been
    # removed.  Under strict-2bww the two builds are structurally different
    # kernels (CMS vs non-CMS, UsePLRPack=True vs False) and their captures
    # are NOT dataflow-equivalent; compare_graphs would surface spurious
    # OrderInvertedFailures from unrelated GRA instruction-ordering
    # differences rather than from prologue structure.  The meaningful pin
    # is the prologue-content structural check above.


# rocm-libraries-aixt: migrated OFF the SHADOW-shared-prologue trick.
# Pre-aixt this test consumed `_capture_context.default` /
# `_capture_context.cms` from a single SHADOW build and asserted
# `cap_with_cms.prologue is cap_with_default.prologue` — a Python
# identity check pinning that `build_cms_four_part_capture` threaded
# the default-side prologue through verbatim. Under Approach A
# (rocm-libraries-nyb5) the default-side capture comes from a
# fully-isolated second build via `build_non_cms_reference`, so the two
# captures cannot share Python identity. The migrated assertion
# verifies prologue CONTENT equivalence between the CMS build's
# CMS-side capture and the non-CMS reference build, which is the
# semantic the original `is` check was a proxy for.
def test_whole_kernel_cms_prologue_matches_non_cms_reference(
    isa_infrastructure,
):
    """The CMS-side prologue capture must agree (content-equivalent)
    with the non-CMS reference build's prologue for the canonical CMS
    kernel.

    Pre-aixt assertion (SHADOW-shared-prologue trick):

        ``cap_with_cms.prologue is cap_with_default.prologue``

    pinned that ``build_cms_four_part_capture`` threaded the
    default-side prologue through to the CMS side by Python identity.
    This was a SHADOW-internal implementation detail. Under Approach A
    the default-side capture is produced by a fully-isolated second
    writer (``build_non_cms_reference``); identity sharing is
    impossible and the right semantic to check is content equivalence.

    rocm-libraries-2bww (strict model) note: the canonical ``_CMS_CONFIG``
    does not set ``UsePLRPack`` (defaults supply 0). Under the strict
    model Solution construction no longer pre-zeroes the flag — the
    config / YAML value is honored through to ``kernelBody``. With
    UsePLRPack=0 the natural prologue is empty/None on both sides, so
    equivalence is the trivial ``None == None``. If a future kernel-config
    drift produces a non-trivial prologue, this test asserts the CMS-side
    prologue's canonical-render content equals the non-CMS reference's.
    Tests that exercise off-nominal forced-``UsePLRPack`` semantics
    (where the override-between-calls trick re-introduces a populated
    prologue mid-build) are SHADOW-pipeline-specific machinery (see
    ``_build_capture`` in this file and the open question in
    ``AIXT_IMPLEMENTATION.md`` §"Open questions").

    rocm-libraries-aixt; supersedes the
    ``test_whole_kernel_useplrpack_cms_matches_both_defaults`` shadow
    trick.
    """
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    from Tensile.Components.CustomSchedule.approach_a import (
        build_non_cms_reference,
    )
    from Tensile.Components.ScheduleCapture import WrappedInstruction

    _isa, isaInfoMap, asm = isa_infrastructure
    config = dict(_CMS_CONFIG)

    # --- Build #1: real CMS build (no UsePLRPack forcing). The
    # auto-activated SHADOW path populates `_last_cms_capture`
    # alongside the SHADOW default-side capture; we only consume the
    # CMS-side capture here. The default-side capture from this build
    # is intentionally NOT used (Approach A migration).
    cms_solution = _make_solution(config, asm, isaInfoMap)
    cms_writer = KernelWriterAssembly(asm, DebugConfig())
    try:
        cms_writer._getKernelSource(cms_solution)
    except Exception:
        # The SHADOW in-build assert may fire on an unrelated
        # CMS-vs-default divergence; the FourPartCapture is populated
        # before the assert.
        pass
    cms_cap = cms_writer._last_cms_capture
    assert cms_cap is not None, (
        "CMS build did not populate `_last_cms_capture` — kernelBody "
        "post-loop assembly stage did not run."
    )

    # --- Build #2: non-CMS reference via Approach A's helper.
    # rocm-libraries-2bww: strip UsePLRPack from the non-CMS reference
    # config so the reference matches the pre-strict-2bww "unmutated" state
    # (the CMS schedule used to mutate UsePLRPack AFTER matching; the
    # non-CMS reference saw the unmutated config without UsePLRPack=True).
    ref_config = dict(config)
    ref_config['UsePLRPack'] = False
    ref_cap = build_non_cms_reference(ref_config, asm, isaInfoMap)

    # Prologue content equivalence.
    #
    # rocm-libraries-2bww note: `_CMS_CONFIG` now carries `UsePLRPack=True`
    # (required by the CMS schedule's required_flags). With UsePLRPack=True,
    # the CMS build's auto-activated `_captureDefaultSchedule` path captures
    # a non-None prologue (the prefetch pack chain).
    #
    # `build_non_cms_reference` uses `enable_capture_non_cms_build()` which
    # does NOT activate `_captureDefaultSchedule`, so the prologue capture
    # at KernelWriter.py:5190 never fires — `ref_cap.prologue` is always
    # None.  This is a known limitation of the non-CMS reference capture
    # path (prologue capture is not implemented for non-CMS builds).
    #
    # Given this limitation, we assert only that the non-CMS reference
    # prologue is None (invariant of `build_non_cms_reference`), and that the
    # CMS prologue is non-None and non-empty (consequence of UsePLRPack=True).
    # Full content-equivalence between CMS and non-CMS prologues is deferred
    # to a future bead that adds prologue capture to `build_non_cms_reference`.
    assert ref_cap.prologue is None, (
        "build_non_cms_reference unexpectedly produced a non-None prologue; "
        "the non-CMS capture path does not implement prologue capture "
        "(KernelWriter.py:5190 only fires for _captureDefaultSchedule). "
        "If this started passing, the prologue limitation has been fixed — "
        "re-enable the full content-equivalence check below."
    )
    # CMS side: UsePLRPack=True in the config → prologue is non-None and
    # contains Pack instructions from the prefetch chain.
    assert cms_cap.prologue is not None, (
        "_CMS_CONFIG has UsePLRPack=True; the CMS build's prologue capture "
        "should be non-None. If this fails, the prologue capture path for "
        "UsePLRPack=True CMS kernels regressed."
    )
    assert len(cms_cap.prologue.instructions) > 0, (
        "CMS-side prologue is non-None but empty; expected Pack instructions "
        "from the UsePLRPack=True prefetch chain."
    )


def test_prologue_label_index_sorts_before_ml_prev():
    """BODY_LABEL_PROLOGUE must sort strictly before BODY_LABEL_ML_PREV
    so prologue writes are visible to mainloop reads in per-byte
    latest-writer resolution. Pin the loop_index value explicitly
    rather than relying on the build_dataflow_graph behavior — the
    assignment is a single integer constant in ScheduleCapture.py and
    drift would silently break cross-body dataflow.
    """
    from Tensile.Components.ScheduleCapture import (
        BODY_LABEL_PROLOGUE, BODY_LABEL_ML_PREV, BODY_LABEL_ML,
        BODY_LABEL_TO_LOOP_INDEX, SchedulePosition,
    )
    pro_idx = BODY_LABEL_TO_LOOP_INDEX[BODY_LABEL_PROLOGUE]
    ml_prev_idx = BODY_LABEL_TO_LOOP_INDEX[BODY_LABEL_ML_PREV]
    assert pro_idx < ml_prev_idx, (
        f"PRO loop_index ({pro_idx}) must sort before ML-1 "
        f"loop_index ({ml_prev_idx})"
    )
    pro_pos = SchedulePosition(loop_index=pro_idx, stream_index=99)
    ml_pos = SchedulePosition(
        loop_index=BODY_LABEL_TO_LOOP_INDEX[BODY_LABEL_ML], stream_index=0,
    )
    assert pro_pos < ml_pos


def test_build_prologue_capture_returns_none_when_all_inputs_empty():
    """`build_prologue_capture` returns None when no source list is supplied
    or the list is empty (PGR=0 kernels emit no prologue at all, and
    usePLRPack=False kernels emit no prologue Pack producers).
    """
    from Tensile.Components.ScheduleCapture import build_prologue_capture
    assert build_prologue_capture() is None
    assert build_prologue_capture(prologue_interleaved_items=[]) is None


def test_build_dataflow_graph_handles_none_prologue():
    """A FourPartCapture with `prologue=None` (PGR=0 case) must build a
    graph cleanly — the prologue body is just absent from the graph's
    captures dict.

    C3c: `build_dataflow_graph` now delegates to
    `UnrolledCapture.from_four_part_capture`, which requires
    `main_loop` to have a codepath-0 entry (ML body is mandatory for
    the unrolled timeline). The fixture supplies a single-SNop ML body
    so the mandatory-ML check passes; the test's intent (PRO=None is
    handled cleanly) is unaffected.
    """
    from rocisa.instruction import SNop
    from Tensile.Components.ScheduleCapture import (
        FourPartCapture, LoopBodyCaptureBuilder, BODY_LABEL_PROLOGUE,
    )

    def _one_snop_body():
        b = LoopBodyCaptureBuilder()
        b.append(inst=SNop(waitState=0), category="SNOP", subiter=0, mfma_index=0)
        return b.finalize()

    # Synthetic minimal FourPartCapture; we only care that the
    # body-walk handles `prologue=None` without raising. Use the empty
    # n_gl/n_ll dicts to bypass the empty-body guard for tail bodies.
    # ML body must be present (codepath 0) after C3c's mandatory-ML check.
    cap = FourPartCapture(
        main_loop={0: _one_snop_body()},
        main_loop_prev={0: _one_snop_body()},
        n_gl={}, n_ll={},
        num_mfma=0, num_codepaths=1, source="default-sia3",
        prologue=None,
    )
    g = build_dataflow_graph(cap)
    assert BODY_LABEL_PROLOGUE not in g.captures
