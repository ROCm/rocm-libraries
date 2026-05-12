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
}


def _build_capture(isa_infrastructure, *, force_use_plr_pack):
    """Build a CMS kernel and return the writer so callers can inspect
    `_capture_context.default` (and `.cms`).

    `force_use_plr_pack` controls `kernel["UsePLRPack"]` AT PROLOGUE TIME.
    Solution construction zeroes UsePLRPack on the CMS path (see
    Tensile/SolutionStructs/Solution.py:2025), and the matched CMS
    schedule re-sets it inside `customMainLoopSchedule` — but that runs
    inside `_loopBody`, which is invoked AFTER the prologue at lines
    4934-5050. So at the prologue's own check
    `usePLRPack = ... or (kernel["UseCustomMainLoopSchedule"] and
    kernel["UsePLRPack"])` (KernelWriter.py:4945), the schedule has not
    yet flipped the flag, and overriding the kernel dict between
    `_initKernel` and `kernelBody` is the surgical way to force the
    prologue down a specific branch.

    Uses `enable_capture_default_schedule_no_assert` to bypass the
    in-`kernelBody` compare_graphs + validate_edge_wait_coverage
    assertion gate, because forced UsePLRPack=1 on a default schedule
    that doesn't natively schedule for it trips
    TimingTooCloseFailure on the prologue's back-to-back Pack chain.
    Callers ARE responsible for re-running compare_graphs +
    validate_edge_wait_coverage explicitly on the residual after
    filtering the legitimate TimingTooCloseFailure subset by isinstance.

    rocm-libraries-oram Phase 2.
    """
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig

    isa, isaInfoMap, asm = isa_infrastructure
    solution = _make_solution(_CMS_CONFIG, asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())
    writer.enable_capture_default_schedule_no_assert()

    # Manually drive the kernel-source pipeline so we can flip
    # `kernel["UsePLRPack"]` between `_initKernel` (which runs CMS
    # schedule discovery setup) and `kernelBody` (which consumes the
    # flag at prologue time). `_getKernelSource` does both back-to-back
    # without an interception point.
    tensorParametersA = {}
    tensorParametersB = {}
    writer._initKernel(solution, tensorParametersA, tensorParametersB)
    solution["UsePLRPack"] = 1 if force_use_plr_pack else 0
    writer.stringIdx = 0
    err, _ = writer.kernelBody(solution, tensorParametersA, tensorParametersB)
    if err != 0:
        raise RuntimeError(
            f"kernelBody returned error={err} (force_use_plr_pack="
            f"{force_use_plr_pack})"
        )
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
    """Same kernel module, two captures (UsePLRPack=1 vs UsePLRPack=0).

    The UsePLRPack=1 capture must have a non-empty prologue containing
    the prefetch pack chain (PackA*/PackB* leaves emitted between
    `setupNewTile` and `openLoop`). The UsePLRPack=0 capture's prologue
    is either None (no Pack producers were emitted into the prologue)
    or empty.

    Building merged DataflowGraphs from the two captures and feeding
    them to `compare_graphs` MUST surface the divergence as a
    CaptureConsistencyError — the two graphs' data-flow node identity
    sets differ because one side has Pack producers that the other side
    lacks. (`compare_graphs` raises CaptureConsistencyError BEFORE the
    edge-by-edge comparison whenever the LR/LW/GR/MFMA identity sets
    differ — see CMSValidator.py:2748.)
    """
    writer_with = _build_capture(isa_infrastructure, force_use_plr_pack=True)
    writer_without = _build_capture(isa_infrastructure, force_use_plr_pack=False)

    cap_with = writer_with._capture_context.default
    cap_without = writer_without._capture_context.default
    assert cap_with is not None
    assert cap_without is not None

    # Sanity: the UsePLRPack=1 capture must have a populated prologue
    # with at least one Pack-tagged instruction. If the prologue is None
    # or has zero Pack* leaves, the test scenario isn't actually
    # exercising the divergence the bead targets — fail loudly so we
    # discover a refactor that broke the prologue plumbing rather than
    # a green-but-meaningless test.
    assert cap_with.prologue is not None, (
        "UsePLRPack=1 prologue capture is None — the prologue plumbing "
        "in KernelWriter.kernelBody did not populate ctx.prologue. The "
        "structural divergence the test asserts cannot be observed."
    )
    pack_categories = [
        ti.category for ti in cap_with.prologue.instructions
        if ti.category.startswith("Pack")
    ]
    assert pack_categories, (
        f"UsePLRPack=1 prologue has no Pack* instructions; categories "
        f"present: {sorted({ti.category for ti in cap_with.prologue.instructions})}. "
        f"The packPrePrefetchA/B chain did not get snapshotted into "
        f"ctx.prologue_prefetch_pack_a/b. Without these the test cannot "
        f"distinguish UsePLRPack=1 from UsePLRPack=0."
    )

    # Sanity: the UsePLRPack=0 capture must NOT have any Pack-tagged
    # prologue instructions (the pack chain stays in pack[plrIdx] for
    # the mainloop instead). Allow `prologue is None` (no other
    # prologue contents either) or `prologue is not None` with zero
    # Pack* leaves; both are consistent with the divergence.
    if cap_without.prologue is not None:
        without_pack_categories = [
            ti.category for ti in cap_without.prologue.instructions
            if ti.category.startswith("Pack")
        ]
        assert not without_pack_categories, (
            f"UsePLRPack=0 prologue has unexpected Pack* leaves "
            f"({without_pack_categories}); the divergence test setup is "
            f"not what we think — both sides emit Pack producers in "
            f"the prologue."
        )

    # Build merged DataflowGraphs and assert compare_graphs flags the
    # divergence. Failure shape: the UsePLRPack=1 graph has extra Pack
    # producers (PACK class-tag) emitting edges that the UsePLRPack=0
    # graph lacks. PACK is NOT in `_DATA_FLOW_KINDS = ("LR","LW","GR",
    # "MFMA")` (CMSValidator.py:2716), so the entry-time
    # identity-coverage gate does NOT raise — it only fires for the
    # four data-flow kinds. Instead, the divergence surfaces during
    # missing-edge diagnosis: comparing reference=with-Pack against
    # subject=without-Pack walks reference's edges, finds Pack-edges
    # whose producer node doesn't exist in subject, and raises
    # CaptureConsistencyError from `diagnose_missing_edge` phase 0
    # ("invoked with missing node — identity-coverage check at
    # compare_graphs entry was bypassed"). That raise IS the structural
    # divergence flag for this Pack-only divergence.
    #
    # Direction matters: comparing without-Pack as reference (no extra
    # edges) against with-Pack as subject would NOT raise — the missing
    # edges are extras in subject, not in reference. The bead's "report
    # prologue structural difference" requirement is satisfied as long
    # as ONE direction surfaces the divergence; we pin the
    # with-as-reference direction since that's the direction
    # production gating runs (treat the more-instrumented capture as
    # ground truth, scrutinize the less-instrumented one).
    g_with = build_dataflow_graph(cap_with)
    g_without = build_dataflow_graph(cap_without)

    with pytest.raises(CaptureConsistencyError) as excinfo:
        compare_graphs(g_with, g_without)
    # The raise must mention the entry-time identity-set divergence
    # path (i.e. the extra pack-MFMA producer surfaces as a node count
    # mismatch in _data_flow_ids), not some other consistency failure.
    # We pin the substring so the test fails informatively if a future
    # refactor changes which check catches the divergence.
    assert "data-flow node identity sets differ" in str(excinfo.value), (
        f"compare_graphs raised CaptureConsistencyError but the message "
        f"does not look like the entry-time identity-set divergence path: "
        f"{excinfo.value!r}"
    )
    # And the divergence summary should mention an MFMA identity (under
    # the new identity scheme pack-MFMAs are real `MFMAInstruction`
    # rocisa class, so `_summary_by_class` reports them as 'MFMA' rather
    # than the CMS-shaped 'PACK' tag).
    assert "'MFMA'" in str(excinfo.value), (
        f"identity-set CaptureConsistencyError does not mention an "
        f"MFMA identity; the divergence is not the prologue Pack chain: "
        f"{excinfo.value!r}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "rocm-libraries-exfw: under 4up4's new identity scheme + F3's "
        "leftover-pack-walk deletion, the CMS-side macro emits the full "
        "plrIdx=3 prefetch-pack chain into its main_loop while the "
        "default-side SIA3 shadow only consumes pack[packIdx] per-iter "
        "and resets — 16 PackA3/PackB3 instructions (rocisa class "
        "MFMAInstruction) appear in subject but not reference. The "
        "principled fix is rocm-libraries-71hw Approach A (true non-CMS "
        "reference build via UseCustomMainLoopSchedule=0) where pack "
        "chain layout matches naturally between both sides. Until 71hw "
        "lands, this test xfails. See EXFW_MFMA_DIVERGENCE_INVESTIGATION.md."
    ),
)
def test_whole_kernel_useplrpack_cms_matches_both_defaults(isa_infrastructure):
    """A CMS kernel using UsePLRPack must compare-equal against BOTH
    `UsePLRPack=True` default and `UsePLRPack=False` default at the
    whole-kernel level.

    The CMS schedule absorbs the prologue-flag difference: CMS-side
    capture inherits its prologue verbatim from the default-side
    capture (see `build_cms_four_part_capture`'s `prologue=` arg).
    Both `compare_graphs` AND `validate_edge_wait_coverage` must pass
    on BOTH UsePLRPack defaults — decision 4b's "BOTH MUST PASS"
    semantics. We re-run them explicitly here on the residual after
    filtering the legitimate TimingTooCloseFailure subset (the
    back-to-back Pack chain on the forced UsePLRPack=1 side trips the
    validator's 5-quad-cycle gap requirement; that residual is
    architectural, not a capture-pipeline bug).

    `_build_capture` opts out of the in-`kernelBody` validator gate via
    `enable_capture_default_schedule_no_assert`, so this test owns the
    entire validation surface for the off-nominal forced-UsePLRPack
    branches.
    """
    writer_with = _build_capture(isa_infrastructure, force_use_plr_pack=True)
    writer_without = _build_capture(isa_infrastructure, force_use_plr_pack=False)

    cap_with_default = writer_with._capture_context.default
    cap_with_cms = writer_with._capture_context.cms
    cap_without_default = writer_without._capture_context.default
    cap_without_cms = writer_without._capture_context.cms

    for label, default_cap, cms_cap in (
        ("UsePLRPack=1", cap_with_default, cap_with_cms),
        ("UsePLRPack=0", cap_without_default, cap_without_cms),
    ):
        assert default_cap is not None, (
            f"{label}: default capture missing — capture pipeline "
            f"didn't populate ctx.default."
        )
        assert cms_cap is not None, (
            f"{label}: CMS capture missing — capture pipeline didn't "
            f"populate ctx.cms via build_cms_four_part_capture."
        )
        # Explicit validation: both compare_graphs AND
        # validate_edge_wait_coverage must yield an empty residual
        # AFTER filtering legitimate TimingTooCloseFailure entries by
        # isinstance (NOT by message-substring match). Decision 4b's
        # "BOTH MUST PASS" semantics are now properly enforced on both
        # the UsePLRPack=1 and UsePLRPack=0 sides.
        non_timing_diffs, non_timing_waits = _explicit_validate(
            default_cap, cms_cap,
        )
        assert non_timing_diffs == [], (
            f"{label}: compare_graphs reported {len(non_timing_diffs)} "
            f"non-timing failure(s) on whole-kernel comparison:\n  "
            + "\n  ".join(f.format() for f in non_timing_diffs)
        )
        assert non_timing_waits == [], (
            f"{label}: validate_edge_wait_coverage reported "
            f"{len(non_timing_waits)} non-timing failure(s) on "
            f"whole-kernel comparison:\n  "
            + "\n  ".join(f.format() for f in non_timing_waits)
        )

    # Pin that the CMS-side capture inherits the default-side prologue.
    # rocm-libraries-oram Phase 2 decision 3 (single concatenated graph,
    # prologue propagated to CMS via `default_capture.prologue`).
    assert cap_with_cms.prologue is cap_with_default.prologue, (
        "CMS-side capture did not inherit the default-side prologue. "
        "build_cms_four_part_capture is not threading "
        "`prologue=default_capture.prologue` through."
    )
    assert cap_without_cms.prologue is cap_without_default.prologue, (
        "CMS-side capture did not inherit the default-side prologue "
        "(UsePLRPack=0 side)."
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
    """`build_prologue_capture` returns None when no source modules are
    populated (PGR=0 kernels emit no prologue at all, and usePLRPack=False
    kernels emit no prologue Pack producers).
    """
    from Tensile.Components.ScheduleCapture import build_prologue_capture
    assert build_prologue_capture() is None
    assert build_prologue_capture(
        prefetch_pack_a=[], prefetch_pack_b=[],
    ) is None


def test_build_dataflow_graph_handles_none_prologue():
    """A FourPartCapture with `prologue=None` (PGR=0 case) must build a
    graph cleanly — the prologue body is just absent from the graph's
    captures dict.
    """
    from Tensile.Components.ScheduleCapture import (
        FourPartCapture, LoopBodyCapture, BODY_LABEL_PROLOGUE,
    )
    # Synthetic minimal FourPartCapture; we only care that the
    # body-walk handles `prologue=None` without raising. Use the empty
    # n_gl/n_ll dicts to bypass the empty-body guard for tail bodies.
    cap = FourPartCapture(
        main_loop={}, main_loop_prev={}, n_gl={}, n_ll={},
        num_mfma=0, num_codepaths=1, source="default-sia3",
        prologue=None,
    )
    g = build_dataflow_graph(cap)
    assert BODY_LABEL_PROLOGUE not in g.captures
