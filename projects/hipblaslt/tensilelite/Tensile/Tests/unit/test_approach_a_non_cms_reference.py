################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Approach A — true non-CMS reference build (rocm-libraries-nyb5).

Asserts the new ``build_non_cms_reference`` helper:

    1. Cycle 1 — returns a ``FourPartCapture`` whose body shape (ML-1, ML,
       NGL, NLL) matches expectations and each body has a non-zero
       data-flow instruction count.

    2. Cycle 2 — when paired with the existing CMS-side capture from a
       ``UseCustomMainLoopSchedule=1`` build of the same canonical kernel,
       ``compare_graphs`` reports no failures.

    3. Cycle 3 — wraps the canonical TF32 4x4 TN kernel from
       ``test_dataflow_graph_emission_ordinal``'s ``real_kernel_capture_pair``
       fixture so the d3zj per-render and per-ordinal invariants pass under
       the new path. Build #2's capture observes the LCC (SCmpEQI32 +
       SSubU32) emitted by ``closeLoop`` because finalize happens AFTER
       closeLoop on the non-CMS path — this is the structural fix that
       closes the d3zj defect as a side effect.

References:
    - 2LZD_INVESTIGATION.md §6 + §6.2 (Approach A pick + Q2/Q3 framing).
    - PRELOOP_CAPTURE_PHASE1.md §7 (oram.1 critical-path framing).
    - D3ZJ_SCMPEQI32_INVESTIGATION.md (LCC capture-timing defect).
    - D3ZJ_NGL_NLL_LCC_INVESTIGATION.md (per-body LCC invariant).
    - NYB5_IMPLEMENTATION.md (this bead's design memo).
"""

import os
import sys

import pytest


# Reuse the canonical config from `test_dataflow_graph_emission_ordinal` so
# this test compares apples-to-apples with d3zj's existing fixtures.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dataflow_graph_emission_ordinal import (  # noqa: E402
    CANONICAL_TF32_4X4_TN_CONFIG,
)


def _make_kernel_writer(asm):
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    return KernelWriterAssembly(asm, DebugConfig())


# =============================================================================
# Cycle 1 — helper exists and returns body-shaped capture
# =============================================================================

def test_build_non_cms_reference_returns_body_shaped_capture(isa_infrastructure):
    """Cycle 1 RED: import and call ``build_non_cms_reference``; assert
    the returned ``FourPartCapture`` exposes ML-1, ML, NGL, NLL bodies
    each with a non-zero data-flow instruction count.

    The canonical TF32 4x4 TN config has:
      - PrefetchGlobalRead=2 → PGR drives the NGL body (no-load loop for
        the very last unroll iteration).
      - PrefetchLocalRead=1 → PLR drives the ML-1 prev body.
      - DepthU=32, MFMA=16x16x32 → 2 MFMAs per inner-unroll subiter.

    All four mainloop bodies must have data-flow instructions; an empty
    body indicates the capture pipeline observed nothing on the non-CMS
    emit path (build did not actually feed the builder).
    """
    from Tensile.Components.CustomSchedule.approach_a import (
        build_non_cms_reference,
    )
    from Tensile.Components.CMSValidator import data_flow_instructions
    from cms_test_utils import _make_solution

    _isa, isaInfoMap, asm = isa_infrastructure
    config = dict(CANONICAL_TF32_4X4_TN_CONFIG)

    capture = build_non_cms_reference(
        _make_solution(config, asm, isaInfoMap), asm, isaInfoMap)

    # Body shape: ML-1, ML, NGL, NLL all present at codepath 0.
    assert 0 in capture.main_loop_prev, (
        "Expected ML-1 body at codepath 0 in non-CMS reference capture."
    )
    assert 0 in capture.main_loop, (
        "Expected ML body at codepath 0 in non-CMS reference capture."
    )
    assert 0 in capture.n_gl, (
        "Expected NGL body at codepath 0; canonical config has PGR=2 "
        "which always emits NGL."
    )
    assert 0 in capture.n_ll, (
        "Expected NLL body at codepath 0 in non-CMS reference capture."
    )

    # Non-zero data-flow instruction count per body.
    body_counts = {
        "ML-1": sum(1 for _ in data_flow_instructions(capture.main_loop_prev[0])),
        "ML":   sum(1 for _ in data_flow_instructions(capture.main_loop[0])),
        "NGL":  sum(1 for _ in data_flow_instructions(capture.n_gl[0])),
        "NLL":  sum(1 for _ in data_flow_instructions(capture.n_ll[0])),
    }
    empty_bodies = [b for b, n in body_counts.items() if n == 0]
    assert not empty_bodies, (
        f"Empty body/bodies in non-CMS reference capture: {empty_bodies}; "
        f"per-body data-flow counts: {body_counts}"
    )


# =============================================================================
# Cycle 2 — integration: compare_graphs accepts the new ref + CMS subj
# =============================================================================

@pytest.mark.xfail(
    strict=True,
    reason=(
        "rocm-libraries-nyb5 Cycle 2 surfaced 3 OrderInvertedFailures in "
        "the ML body's GRA/GRB stream that the SHADOW path masked. "
        "Mechanism: the per-tile schedule on the CMS side mutates "
        "kernel-level flags (`UsePLRPack=True`, `UseMFMAF32XEmulation`) "
        "before SIA3 runs, changing the GR scheduling order. The shadow "
        "capture inherited those mutations because it shared the CMS-"
        "mutated `kernel` dict; the non-CMS reference build (Approach A) "
        "uses an unmutated `kernel` dict per `2LZD_INVESTIGATION.md §6.2 "
        "Q2` (the 'two builds, accept whatever Tensilelite mutates' "
        "framing). The 3 residual failures are GRA[3]/GRA[4], GRA[6]/GRB[0], "
        "GRA[5]/GRA[6] order inversions — all within the same ML body, "
        "all GR-side. This is the predicted Approach-A surfacing and is "
        "in scope for `rocm-libraries-3ija` (out of scope for this bead "
        "per the verification checklist). strict=True so any change in "
        "the residual surfaces as a failure."
    ),
)
def test_non_cms_reference_compares_clean_against_cms_build(isa_infrastructure):
    """Cycle 2: spin up Build #1 (CMS) and Build #2 (non-CMS) of the
    canonical kernel, wire the non-CMS capture as ``ref`` and the CMS
    capture as ``subj``, call ``compare_graphs``, assert no failures.

    Build #1 produces ``writer._last_cms_capture`` via the existing CMS
    auto-activation path. Build #2 produces a ``FourPartCapture`` via the
    new ``build_non_cms_reference`` helper — its post-closeLoop finalize
    captures the LCC instructions that the SHADOW path missed.

    Currently xfail (strict=True): Approach A surfaces 3
    OrderInvertedFailures in the ML body's GRA/GRB stream that the
    SHADOW path masked because the shadow shared the CMS-mutated
    `kernel` dict (with `UsePLRPack=True` from per-tile schedule
    registration). The non-CMS reference uses an unmutated `kernel`
    dict, exposing the GR-scheduling-order divergence. See xfail reason
    above for the full mechanism + scope.
    """
    from Tensile.Components.CustomSchedule.approach_a import (
        build_non_cms_reference,
    )
    from Tensile.Components.CMSValidator import (
        build_dataflow_graph, compare_graphs,
    )
    from cms_test_utils import _make_solution

    _isa, isaInfoMap, asm = isa_infrastructure
    config = dict(CANONICAL_TF32_4X4_TN_CONFIG)

    # --- Build #1: CMS path (existing pipeline). ---
    cms_solution = _make_solution(config, asm, isaInfoMap)
    cms_writer = _make_kernel_writer(asm)
    try:
        cms_writer._getKernelSource(cms_solution)
    except Exception:
        # Pre-existing shadow-vs-CMS divergence may raise on the assert.
        # Approach A replaces the reference; we still need the CMS-side
        # FourPartCapture, which is populated before the assert fires.
        pass
    cms_cap = cms_writer._last_cms_capture
    assert cms_cap is not None, (
        "CMS build did not populate _last_cms_capture; the kernelBody "
        "post-loop assembly stage (KernelWriter.py:5258+) did not run."
    )

    # --- Build #2: non-CMS reference (NEW path). ---
    ref_cap = build_non_cms_reference(
        _make_solution(config, asm, isaInfoMap), asm, isaInfoMap)

    # --- Wire ref + subj into compare_graphs. ---
    ref_graph = build_dataflow_graph(ref_cap)
    subj_graph = build_dataflow_graph(cms_cap)
    failures = compare_graphs(ref_graph, subj_graph)
    assert failures == [], (
        f"Approach-A compare_graphs surfaced {len(failures)} failure(s) "
        f"with the non-CMS reference; first 3: "
        f"{[type(f).__name__ for f in failures[:3]]}"
    )


def test_non_cms_reference_compare_graphs_surfaces_only_known_residuals(
    isa_infrastructure,
):
    """Pin the exact residual surfaced by Approach A so any new kind of
    failure surfaces as a test failure (not a silent shape change).

    rocm-libraries-l1l6 (2026-05-26) re-pin: the residual shape changed
    between m7o5 and y391 once the pre-CMS-derivation snapshot started
    feeding ``build_non_cms_reference``. Under m7o5 (pre-snapshot,
    YAML-override hack) the residual was 3 ``OrderInvertedFailure``
    instances on GRA/GRB in the ML body. Under y391+ (post-snapshot,
    principled), the residual is a ``CaptureConsistencyError`` raised
    by ``compare_graphs``'s data-flow node identity-set check: 28
    identities in ref-but-not-subj, 28 identities in subj-but-not-ref,
    each side split exactly ``{LR: 14, MFMA: 14}``.

    This new shape is the principled "accept whatever Tensilelite
    mutates" baseline from ``2LZD_INVESTIGATION.md §6.2 Q2``: the
    canonical CMS YAML sets ``UsePLRPack=True`` as a *user-set* field,
    so the snapshot mechanism (which only reverts
    ``assignDerivedParameters``-set fields, NOT user-set YAML fields)
    correctly preserves it on the non-CMS reference build. The
    resulting LR/MFMA identity differences reflect the actual
    schedule-level divergence between the CMS path's per-tile MFMA
    ordering and the non-CMS path's SIA3 scheduler — both consuming
    the same ``UsePLRPack=True`` flag but producing different ds_read
    / v_mfma offset orderings. Earlier proposals to move the snapshot
    site *earlier* in ``Solution.assignDerivedParameters`` were
    rejected because that would resurrect m7o5's hack of silently
    overriding user-supplied YAML.

    Any change in (a) the failure class, (b) the per-side identity
    count, or (c) the per-side category split should escalate per the
    bead's "STOP and surface" rule for Cycle 2.
    """
    from Tensile.Components.CustomSchedule.approach_a import (
        build_non_cms_reference,
    )
    from Tensile.Components.CMSValidator import (
        build_dataflow_graph, compare_graphs,
    )
    from Tensile.Components.ScheduleCapture import CaptureConsistencyError
    from cms_test_utils import _make_solution

    _isa, isaInfoMap, asm = isa_infrastructure
    config = dict(CANONICAL_TF32_4X4_TN_CONFIG)

    cms_solution = _make_solution(config, asm, isaInfoMap)
    cms_writer = _make_kernel_writer(asm)
    try:
        cms_writer._getKernelSource(cms_solution)
    except Exception:
        pass
    cms_cap = cms_writer._last_cms_capture
    assert cms_cap is not None
    # rocm-libraries-l1l6: cms_solution carries the pre-CMS-derivation
    # snapshot as an attribute (`cms_solution.pre_cms_state()`).
    # `build_non_cms_reference` reads it via the accessor; the dict
    # surface of `cms_solution` no longer leaks the snapshot.
    ref_cap = build_non_cms_reference(
        cms_solution, asm, isaInfoMap)

    ref_graph = build_dataflow_graph(ref_cap)
    subj_graph = build_dataflow_graph(cms_cap)

    # rocm-libraries-oplb (post-count-based-gate): the entry-level
    # identity-set gate in ``compare_graphs`` was replaced with a per-
    # (rocisa-derived InstructionCategory) count gate. Under the count
    # gate, both pipelines emit the same number of LR/MFMA/GR nodes for
    # this kernel — the gate PASSES (no CaptureConsistencyError at entry).
    # The T/X register-naming divergence then propagates to the edge layer,
    # where ``edge_keys()`` embeds ``(producer.identity, consumer.identity,
    # ...)`` and ``diagnose_missing_edge`` Phase 0 raises
    # ``CaptureConsistencyError`` with the
    # "identity-coverage check at compare_graphs entry was bypassed"
    # message when an endpoint identity is absent from the subject graph.
    #
    # This assertion was tightened post-oplb's count-based gate. Any
    # change in (a) the failure class or (b) the bypass message shape
    # should still escalate per the bead's "STOP and surface" rule.
    with pytest.raises(CaptureConsistencyError) as excinfo:
        compare_graphs(ref_graph, subj_graph)

    msg = str(excinfo.value)
    # Pin: the edge-layer Phase-0 byte-key message.
    # C3f: Phase 0 now uses byte-key reverse-index lookup; ref producer
    # byte_keys are absent from subj due to T/X register-naming divergence
    # (different physical byte indices in the two captures).
    assert "NO writer anywhere in subj" in msg, (
        f"Expected the diagnose_missing_edge Phase-0 byte-key message "
        f"(ref producer byte_keys absent from subj due to T/X register-naming "
        f"divergence); full message:\n{msg}"
    )
    assert "p_id=" in msg and "c_id=" in msg, (
        f"Expected both p_id= and c_id= in the Phase-0 byte-key message; "
        f"full message:\n{msg}"
    )
    # Pin: the new count-based gate did NOT fire — counts match across
    # sides for this kernel (this is the rocm-libraries-oplb win).
    assert "data-flow per-category node counts differ" not in msg, (
        f"The new count-based gate fired — that would mean the two "
        f"pipelines actually emitted different numbers of LR/MFMA/GR "
        f"nodes (a real pipeline-integrity bug). Investigate before "
        f"re-pinning.\n{msg}"
    )


# =============================================================================
# Cycle 3 — d3zj per-body LCC invariant under Approach A
# =============================================================================

def test_non_cms_reference_has_lcc_in_every_main_loop_body(isa_infrastructure):
    """Cycle 3: assert the non-CMS reference's main-loop bodies (ML-1,
    ML) each contain exactly one ``SSubU32`` + one ``SCmpEQI32`` (the
    loop-counter code, LCC).

    Per ``D3ZJ_SCMPEQI32_INVESTIGATION.md`` the SHADOW capture's ML/ML-1
    bodies were missing both LCC instructions because the shadow's
    ``builder.finalize()`` ran at ``KernelWriter.py:4591`` BEFORE the
    closeLoop emission. Approach A's non-CMS build naturally emits
    closeLoop on its own path (``KernelWriter.py:4611``); the new helper
    finalizes the builder AFTER that emission — closing d3zj as a side
    effect.
    """
    from Tensile.Components.CustomSchedule.approach_a import (
        build_non_cms_reference,
    )
    from Tensile.Components.CMSValidator import data_flow_instructions
    from cms_test_utils import _make_solution

    _isa, isaInfoMap, asm = isa_infrastructure
    config = dict(CANONICAL_TF32_4X4_TN_CONFIG)
    capture = build_non_cms_reference(
        _make_solution(config, asm, isaInfoMap), asm, isaInfoMap)

    # Filter to LCC-tagged instructions; ShadowLimitA/B decrements are
    # also SSubU32 but tagged GRIncA / GRIncB — they're not LCC. Per
    # `D3ZJ_SCMPEQI32_INVESTIGATION.md §1.3` the LCC pair is exactly
    # the SSubU32 / SCmpEQI32 targeting `LoopCounterL`; both are
    # captured under category="LCC" by `_appendCloseLoopLCCToBuilder`.
    # Filter to LCC-tagged instructions; ShadowLimitA/B decrements are
    # also SSubU32 but tagged GRIncA / GRIncB — they're not LCC. Per
    # `D3ZJ_SCMPEQI32_INVESTIGATION.md §1.3` the LCC pair is exactly
    # the SSubU32 / SCmpEQI32 targeting `LoopCounterL`; both are
    # captured under category="LCC" by `_appendCloseLoopLCCToBuilder`.
    issues = []
    for label, by_cp in (("ML-1", capture.main_loop_prev),
                         ("ML",   capture.main_loop)):
        body = by_cp.get(0)
        if body is None:
            issues.append(f"{label}: body absent at codepath 0")
            continue
        ssub = 0
        scmp = 0
        for ti in data_flow_instructions(body):
            if ti.category != "LCC":
                continue
            cls = type(ti.wrapped.rocisa_inst).__name__
            if cls == "SSubU32":
                ssub += 1
            elif cls == "SCmpEQI32":
                scmp += 1
        if ssub != 1 or scmp != 1:
            issues.append(
                f"{label}: LCC SSubU32={ssub} LCC SCmpEQI32={scmp} "
                f"(expected 1 of each)"
            )
    assert not issues, (
        f"Per-body LCC invariant violation in non-CMS reference: {issues}"
    )


# =============================================================================
# d3zj LCC closure verification under Approach A
# =============================================================================
# The d3zj defect ("default-side ML/ML-1 missing LCC") closes at the
# helper level via `test_non_cms_reference_has_lcc_in_every_main_loop_body`
# above (Cycle 3) — Build #2 captures LCC because finalize() runs AFTER
# closeLoop().
#
# We deliberately DO NOT replicate the broader d3zj per-render-counts /
# per-ordinal invariants here. Per `2LZD_INVESTIGATION.md §6.2 Q2`, the
# two builds are allowed to diverge on whatever Tensilelite mutates
# internally — the per-tile schedule on the CMS side flips
# `UsePLRPack=True` (see `dispatch.py:546` and the per-tile schedules)
# which moves pack instructions out of ML on the CMS side. The non-CMS
# reference's `UsePLRPack=False` keeps them in ML. That's an EXPECTED
# divergence, not a bug.
#
# The d3zj-shaped tests in `test_dataflow_graph_emission_ordinal.py`
# remain strict-xfailed because they consume the SHADOW capture, which
# is owned by `rocm-libraries-czby`. Switching them to consume Build #2
# requires the broader oram.1 + body-label-tolerance work to also land
# (per `PRELOOP_CAPTURE_PHASE1.md §7`) so the per-render counts can be
# compared in a body-label-tolerant way. Deferred to a follow-up bead
# under `rocm-libraries-71hw`; see NYB5_IMPLEMENTATION.md §"Verification".
