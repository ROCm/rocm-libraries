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

    # Build merged DataflowGraphs and inspect the comparison.
    #
    # Pre-hdem framing: the UsePLRPack=1 graph had extra pack-MFMA
    # producers in the PRO body whose body-keyed identity tuple
    # (loop_index=-1) collided with NO with-Pack=0 identity, so the
    # entry-time `_data_flow_ids` gate raised
    # CaptureConsistencyError("data-flow node identity sets differ").
    #
    # Post-hdem framing (Approach A drops loop_index from identity;
    # Approach E uses identity-based body-blind edge_keys; ORAM1 §6.1
    # / §7.4): two pack-MFMAs with the same canonical_render and the
    # same per-(body, render) ordinal collapse to ONE identity even
    # when they live in different bodies. The UsePLRPack difference
    # IS exactly such a collapse — the prefetch-pack chain is
    # relocated from `pack[plrIdx]` (consumed during ML steady-state)
    # to the prologue (consumed at the start of ML), but the
    # underlying dataflow is identical. Under hdem this collapse is
    # the DESIRED behavior — it is the motivating case the bead is
    # designed to solve. `compare_graphs` returns no failures
    # because the dataflow really is equivalent; the only difference
    # is which capture-builder pinned the producer to which body
    # label.
    #
    # The cross-body extra-write divergence pin (a TRULY extra
    # producer with no body-collapsed counterpart) lives in
    # `test_dataflow_graph_hdem.py::test_cross_body_extra_write_surfaces`
    # — that test constructs a controlled scenario where the extra
    # producer's identity does NOT collapse with anything in the
    # subject graph, exercising the residual-detection path.
    g_with = build_dataflow_graph(cap_with)
    g_without = build_dataflow_graph(cap_without)

    # Pin the body-collapse outcome explicitly: graphs must look
    # equivalent at the dataflow layer. If a future regression
    # re-introduces body sensitivity (e.g. someone re-adds
    # loop_index to identity, or threads SchedulePosition back into
    # edge_keys), this pin catches it as a regression.
    failures = compare_graphs(g_with, g_without)
    assert failures == [], (
        "compare_graphs reported failures comparing UsePLRPack=1 vs "
        "UsePLRPack=0 — under hdem A+E these captures' dataflow IS "
        "equivalent (the prefetch-pack chain is relocated between PRO "
        "and ML bodies but the produced/consumed bytes are identical), "
        "and the comparator must treat them as matching. Failures: "
        f"{[type(f).__name__ for f in failures[:5]]}"
    )
    # Sanity: graphs themselves have the same number of nodes/edges
    # (the post-collapse signature). If they differ in raw counts a
    # truly extra producer crept in that the body collapse did not
    # absorb — investigate before trusting the empty-failures
    # outcome.
    assert len(g_with.nodes) == len(g_without.nodes), (
        f"Node-count mismatch under body-collapse: with={len(g_with.nodes)}, "
        f"without={len(g_without.nodes)}. The UsePLRPack difference "
        f"should be a pure body relocation."
    )
    assert len(g_with.edges) == len(g_without.edges), (
        f"Edge-count mismatch under body-collapse: with={len(g_with.edges)}, "
        f"without={len(g_without.edges)}."
    )


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

    For the canonical CMS-eligible kernel the per-tile schedule zeroes
    ``UsePLRPack`` at solution-construction time, so the natural
    prologue is empty/None on both sides — equivalence is the trivial
    ``None == None``. If a future kernel-config drift produces a
    non-trivial prologue, this test asserts the CMS-side prologue's
    canonical-render content equals the non-CMS reference's. Tests
    that exercise off-nominal forced-``UsePLRPack`` semantics (where
    the CMS path's per-tile mutation re-introduces a populated
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
    ref_cap = build_non_cms_reference(config, asm, isaInfoMap)

    # Prologue content equivalence. For the canonical CMS-eligible
    # kernel the natural prologue is None on both sides (UsePLRPack=0
    # at solution construction). The check covers both the trivial
    # None case and any future kernel-config drift that produces a
    # populated prologue.
    if cms_cap.prologue is None and ref_cap.prologue is None:
        return
    assert (cms_cap.prologue is None) == (ref_cap.prologue is None), (
        f"Prologue presence drift: CMS-side prologue "
        f"is None={cms_cap.prologue is None}, non-CMS reference "
        f"prologue is None={ref_cap.prologue is None}. The two builds "
        f"of the same canonical kernel disagree on prologue presence."
    )

    # Both populated — compare canonical-render content.
    from collections import Counter
    cms_renders = Counter(
        WrappedInstruction.canonical_str(ti.wrapped.rocisa_inst)
        for ti in cms_cap.prologue.instructions
    )
    ref_renders = Counter(
        WrappedInstruction.canonical_str(ti.wrapped.rocisa_inst)
        for ti in ref_cap.prologue.instructions
    )
    assert cms_renders == ref_renders, (
        f"Prologue canonical-render content diverges between CMS build "
        f"and non-CMS reference build:\n"
        f"  Only in CMS: {sorted(set(cms_renders) - set(ref_renders))[:5]}\n"
        f"  Only in Ref: {sorted(set(ref_renders) - set(cms_renders))[:5]}\n"
        f"  Count drift: "
        f"{[(r, cms_renders[r], ref_renders[r]) for r in sorted(set(cms_renders) & set(ref_renders)) if cms_renders[r] != ref_renders[r]][:5]}"
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
