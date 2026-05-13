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
"""Per-emission ordinal identity regression tests (rocm-libraries-4up4 +
rocm-libraries-hdem).

Companion to `EMISSION_ORDINAL_DESIGN.md` §6.2 and
`ORAM1_PRINCIPLED_APPROACH_INVESTIGATION.md` §2 / §7. The identity tuple
is `(canonical_render, emission_ordinal)` post-hdem (Approach A dropped
`loop_index` from the leading slot to make identity body-blind).
The per-(body, canonical_render) ordinal counter is preserved at capture
time so two emissions in different bodies both start their own per-body
counter at 0; cross-body collapse is the desired behavior for
cross-body pipelining (the residual false-negative risk is caught by
Approach E's byte-key edge-layer matching). This file pins:

* The on0t collision pattern (two same-render emissions in the same body
  get distinct identities) — the test that lets on0t stay closed.
* The per-canonical-render emission-count and ordinal-assignment
  determinism guarantees on synthetic captures whose emission shape is
  controlled directly (no kernel build).

The §6.2 list of regression tests includes per-render emission-count and
ordinal-assignment assertions against the canonical
`_get_schedule_160x128x64_TF32` kernel, plus an `example.yaml`
end-to-end assertion. Those depend on the determinism invariant (memo
§2.6) holding for the real-kernel capture pipeline; investigation
during the 4up4 rollout surfaced a pre-existing default-side capture
double-count for pack-MFMAs in that kernel that the new identity scheme
correctly exposes — see the cover-letter memo accompanying the bead
for details. They are gated under a real-kernel fixture for the same
reason.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    BODY_LABEL_TO_LOOP_INDEX,
    assign_emission_ordinals,
)


# =============================================================================
# Helpers — build per-render emission counters and ordinal maps from a
# `LoopBodyCapture`.
# =============================================================================

def _per_render_counts(capture, instructions=None):
    """Return `{canonical_render: count}` for one LoopBodyCapture's nodes.

    `instructions` defaults to `capture.instructions` (raw); pass
    `data_flow_instructions(capture)` to consume the shared dataflow
    filter (rocm-libraries-d3zj). Synthetic captures that contain no
    scheduler-control instructions get the same answer either way.
    """
    from collections import Counter
    counts = Counter()
    iterable = capture.instructions if instructions is None else instructions
    for ti in iterable:
        render = WrappedInstruction.canonical_str(ti.wrapped.rocisa_inst)
        counts[render] += 1
    return dict(counts)


def _per_render_ordinal_map(capture, instructions=None):
    """Return `{(canonical_render, emission_ordinal): TaggedInstruction}` for a capture.

    `instructions` defaults to `capture.instructions` (raw); pass
    `data_flow_instructions(capture)` to consume the shared dataflow
    filter (rocm-libraries-d3zj).
    """
    out = {}
    iterable = capture.instructions if instructions is None else instructions
    for ti in iterable:
        render = WrappedInstruction.canonical_str(ti.wrapped.rocisa_inst)
        out[(render, ti.emission_ordinal)] = ti
    return out


# =============================================================================
# Test 3 (memo §6.2) — Z012 collision regression
# =============================================================================
# Mirrors the on0t SCC pattern: two physically distinct
# `s_cmp_eq_u32 LoopCounterL, StaggerUIter` emissions in the same body
# (one for the GRIncA lowering, one for GRIncB; per
# `EXAMPLE_YAML_DEFECT_INVESTIGATION.md §1`). Under the historical
# identity tuple `(class_tag, loop_index, canonical_render)`, both
# collapsed to the same identity (last-writer-wins). Under
# `(loop_index, canonical_render, emission_ordinal)`, they get distinct
# ordinals (0 and 1) and therefore distinct identities — closing the
# collision.


class TestZ012CollisionRegression:
    def test_two_same_render_emissions_get_distinct_identities(self):
        """Two SCmpEQU32 emissions of `s_cmp_eq_u32 LoopCounterL,
        StaggerUIter` in the same body must produce distinct identity
        tuples.  This is the regression pin that lets on0t stay
        closed: under the historical identity scheme they collapsed to
        a single node by last-writer-wins.
        """
        from rocisa.instruction import SCmpEQU32
        from rocisa.container import sgpr

        def _build_scmp(seq, category):
            inst = SCmpEQU32(src0=sgpr("LoopCounterL"),
                             src1=sgpr("StaggerUIter"))
            return TaggedInstruction(
                wrapped=WrappedInstruction(inst),
                category=category,
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=0, sequence=seq),
            )

        ti_a = _build_scmp(seq=0, category="GRIncA")
        ti_b = _build_scmp(seq=1, category="GRIncB")

        # Verify the renders ARE byte-identical (the precondition for the
        # historical collision).
        ra = WrappedInstruction.canonical_str(ti_a.wrapped.rocisa_inst)
        rb = WrappedInstruction.canonical_str(ti_b.wrapped.rocisa_inst)
        assert ra == rb, (
            f"Test precondition: both SCmpEQU32 emissions must produce "
            f"the same canonical render. got {ra!r} vs {rb!r}."
        )

        # Apply emission-ordinal assignment over the canonical sort order
        # (identical to LoopBodyCaptureBuilder.finalize and make_capture).
        assign_emission_ordinals([ti_a, ti_b])
        assert ti_a.emission_ordinal == 0
        assert ti_b.emission_ordinal == 1

        id_a = ti_a.identity_for(BODY_LABEL_ML)
        id_b = ti_b.identity_for(BODY_LABEL_ML)
        assert id_a != id_b, (
            f"Two same-render same-body emissions must get distinct "
            f"identities under the per-emission-ordinal scheme. "
            f"got {id_a!r} == {id_b!r}."
        )
        # And the only differing slot must be the ordinal — canonical_render
        # is identical by construction. Identity tuple shape under
        # rocm-libraries-hdem Approach A is `(canonical_render,
        # emission_ordinal)` — `loop_index` was dropped from the leading
        # slot to make identity body-blind (cross-body pipelining must
        # collapse identities for `compare_graphs` to match the same
        # logical dataflow that lands in different bodies on each side).
        assert len(id_a) == 2
        assert len(id_b) == 2
        assert id_a[0] == id_b[0]               # canonical_render
        assert id_a[1] != id_b[1]               # emission_ordinal

    def test_three_same_render_emissions_get_three_distinct_identities(self):
        """Generalize the on0t pattern: N physically distinct emissions
        of the same render-text yield N distinct identities."""
        from rocisa.instruction import SCmpEQU32
        from rocisa.container import sgpr

        tis = []
        for seq in range(3):
            inst = SCmpEQU32(src0=sgpr("LoopCounterL"),
                             src1=sgpr("StaggerUIter"))
            tis.append(TaggedInstruction(
                wrapped=WrappedInstruction(inst),
                category="GRIncA" if seq == 0 else "GRIncB",
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=0, sequence=seq),
            ))
        assign_emission_ordinals(tis)
        idents = [ti.identity_for(BODY_LABEL_ML) for ti in tis]
        assert len(set(idents)) == 3, (
            f"Three same-render emissions must yield three distinct "
            f"identities; got {idents!r}."
        )
        # Ordinals are the consecutive integers 0, 1, 2.
        assert sorted(ti.emission_ordinal for ti in tis) == [0, 1, 2]

    def test_distinct_renders_ordinals_independent(self):
        """Two emissions of DIFFERENT renders both start at ordinal 0
        — the per-render counter is keyed on canonical_render, not on
        the body-wide instruction count."""
        from rocisa.instruction import SCmpEQU32
        from rocisa.container import sgpr

        def _build(src0_name, seq):
            inst = SCmpEQU32(src0=sgpr(src0_name), src1=sgpr("StaggerUIter"))
            return TaggedInstruction(
                wrapped=WrappedInstruction(inst),
                category="GRIncA",
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=0, sequence=seq),
            )

        ti_a = _build("LoopCounterL", 0)
        ti_b = _build("LoopCounterK", 1)
        assign_emission_ordinals([ti_a, ti_b])
        assert ti_a.emission_ordinal == 0
        assert ti_b.emission_ordinal == 0


# =============================================================================
# Synthetic per-render determinism (mirrors the spirit of memo §6.2 #1 and
# #2 without depending on a real-kernel build).
# =============================================================================
# These tests build TWO synthetic captures whose emission shape matches
# by construction (they share the same kernel-writer-source modules in
# the abstract sense) and assert the per-render emission counts and
# per-ordinal logical-instruction match across both. They do NOT exercise
# the real-kernel determinism invariant (memo §2.6) — that requires a
# kernel build, gated behind `isa_infrastructure`.


class TestSyntheticPerRenderDeterminism:
    def _build_pair(self):
        """Construct two captures that emit the SAME ordered sequence of
        canonical renders (some repeated). Returns `(cap_a, cap_b)`."""
        from rocisa.instruction import SCmpEQU32
        from rocisa.container import sgpr

        def _build_capture():
            tis = []
            # Two SCmpEQU32 emissions with the same render (the on0t pattern).
            for seq in range(2):
                inst = SCmpEQU32(src0=sgpr("LoopCounterL"),
                                 src1=sgpr("StaggerUIter"))
                tis.append(TaggedInstruction(
                    wrapped=WrappedInstruction(inst),
                    category="GRIncA" if seq == 0 else "GRIncB",
                    slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                                 mfma_index=0, sequence=seq),
                ))
            # One emission of a DIFFERENT render.
            inst2 = SCmpEQU32(src0=sgpr("LoopCounterK"),
                              src1=sgpr("StaggerUIter"))
            tis.append(TaggedInstruction(
                wrapped=WrappedInstruction(inst2),
                category="GRIncA",
                slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                             mfma_index=0, sequence=2),
            ))
            assign_emission_ordinals(tis)
            from Tensile.Components.ScheduleCapture import LoopBodyCapture
            return LoopBodyCapture(instructions=tis)

        return _build_capture(), _build_capture()

    def test_per_render_counts_match_across_synthetic_pair(self):
        """Memo §6.2 #1 (synthetic form): for every canonical render in
        either capture, both captures emit the same count."""
        cap_a, cap_b = self._build_pair()
        counts_a = _per_render_counts(cap_a)
        counts_b = _per_render_counts(cap_b)
        all_renders = set(counts_a) | set(counts_b)
        for r in all_renders:
            assert counts_a.get(r, 0) == counts_b.get(r, 0), (
                f"Per-render count mismatch for {r!r}: "
                f"a={counts_a.get(r, 0)} vs b={counts_b.get(r, 0)}"
            )

    def test_per_ordinal_logical_instruction_matches_across_synthetic_pair(self):
        """Memo §6.2 #2 (synthetic form): for every (canonical_render,
        emission_ordinal) pair appearing in either capture, both
        captures resolve to a TaggedInstruction whose underlying rocisa
        class name matches.

        This is the per-ordinal-N logical-instruction determinism
        check: ordinal-N's render in capture A IS the same logical
        instruction as ordinal-N's render in capture B.
        """
        cap_a, cap_b = self._build_pair()
        map_a = _per_render_ordinal_map(cap_a)
        map_b = _per_render_ordinal_map(cap_b)
        all_keys = set(map_a) | set(map_b)
        for key in all_keys:
            ti_a = map_a.get(key)
            ti_b = map_b.get(key)
            assert ti_a is not None and ti_b is not None, (
                f"(render, ordinal)={key!r} present in only one capture: "
                f"a_present={ti_a is not None}, b_present={ti_b is not None}"
            )
            ca = type(ti_a.wrapped.rocisa_inst).__name__
            cb = type(ti_b.wrapped.rocisa_inst).__name__
            assert ca == cb, (
                f"Per-ordinal class-name mismatch at {key!r}: "
                f"a={ca!r}, b={cb!r}"
            )


# =============================================================================
# Real-kernel determinism (memo §6.2 #1 / #2 against
# `_get_schedule_160x128x64_TF32`)
# =============================================================================
# These tests build the canonical TF32 4x4 TN kernel through the full
# CMS scheduling pipeline and assert the per-render emission counts /
# ordinal-N logical-instruction-match invariants hold across the
# default-side and CMS-side captures.
#
# NOTE: as of the rocm-libraries-4up4 rollout, the determinism
# invariant (memo §2.6) DOES NOT hold for this kernel — the
# default-side capture produces extra TaggedInstructions for several
# pack-MFMA renders (12 extra instances per body in ML and ML-1; same
# python rocisa object captured twice with different category tags).
# The new identity scheme correctly surfaces this divergence as a
# `CaptureConsistencyError` from `compare_graphs`'s data-flow identity
# coverage check; the historical scheme hid it because
# `class_tag='PACK'` excluded pack-MFMAs from the data-flow filter
# `("LR", "LW", "GR", "MFMA")`.
#
# These tests were previously strict-xfailed under rocm-libraries-d3zj
# while they consumed the SHADOW capture. rocm-libraries-aixt re-routed
# them to consume Approach A's `build_non_cms_reference()` helper via
# `real_kernel_capture_pair_approach_a`, and the xfail markers were
# removed — the LCC is now present in the default-side capture because
# Build #2 finalizes AFTER closeLoop().

CANONICAL_TF32_4X4_TN_CONFIG = {
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


@pytest.fixture(scope="module")
def real_kernel_capture_pair(isa_infrastructure):
    """Build the canonical TF32 4x4 TN kernel and return
    `(default_cap, cms_cap)` `FourPartCapture` pair extracted from the
    SHADOW pipeline.

    Module-scoped (~3-5s build); all real-kernel tests in this module
    that legitimately need the SHADOW-side default capture share the
    build. The d3zj per-body LCC-invariant tests below have been
    migrated OFF this fixture onto
    ``real_kernel_capture_pair_approach_a`` (which sources the
    default-side capture from a true second non-CMS build) — see the
    aixt migration notes there.
    """
    import os
    import sys

    _isa, isaInfoMap, asm = isa_infrastructure
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig

    config = dict(CANONICAL_TF32_4X4_TN_CONFIG)
    solution = _make_solution(config, asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())
    # `_getKernelSource` runs the whole kernel-build pipeline including
    # the dataflow-graph comparison. With the new identity scheme, the
    # pre-existing default-side double-capture defect surfaces and the
    # comparison raises before the captures are returned. Catch it so
    # tests can inspect the captures themselves.
    try:
        writer._getKernelSource(solution)
    except Exception:
        pass
    assert writer._last_default_capture is not None
    assert writer._last_cms_capture is not None
    return writer._last_default_capture, writer._last_cms_capture


@pytest.fixture(scope="module")
def real_kernel_capture_pair_approach_a(isa_infrastructure):
    """Approach A two-build capture pair for the canonical TF32 4x4 TN
    kernel. Returns ``(default_cap, cms_cap)`` where:

    - ``default_cap`` comes from ``build_non_cms_reference`` (Approach A
      Build #2 — a fully isolated writer with
      ``UseCustomMainLoopSchedule=0`` forced and finalize running
      AFTER ``closeLoop`` so LCC is captured).
    - ``cms_cap`` comes from a real CMS build's ``_last_cms_capture``.

    rocm-libraries-aixt migration: the SHADOW capture's default-side
    ML/ML-1 bodies missed the LCC pair (``SSubU32``/``SCmpEQI32``)
    because ``LoopBodyCaptureBuilder.finalize()`` ran BEFORE
    ``closeLoop()`` emitted them. nyb5 closed this defect at the
    helper level by hoisting finalize to AFTER ``closeLoop`` and
    appending the LCC instructions via
    ``_appendCloseLoopLCCToBuilder``. Tests that assert the per-body
    LCC invariant now consume this fixture; the legacy fixture above
    is retained for tests that depend on the SHADOW default-side
    capture's particular shape (e.g.
    ``test_example_yaml_no_spurious_order_inverted_failures``, which
    pins zero ``OrderInvertedFailure``s — the SHADOW capture inherits
    the CMS-mutated ``kernel`` dict so the GR-stream order matches;
    Approach A's unmutated ``kernel`` dict surfaces 3
    GR-OrderInverted residuals tracked under
    ``rocm-libraries-3ija`` per
    ``test_non_cms_reference_compare_graphs_surfaces_only_known_residuals``).

    See ``D3ZJ_SCMPEQI32_INVESTIGATION.md``, ``NYB5_IMPLEMENTATION.md``
    §"d3zj closure (LCC capture)", and ``AIXT_IMPLEMENTATION.md``.

    Module-scoped (~6-10s for both builds); shared across the 3 d3zj
    LCC tests below.
    """
    import os
    import sys

    _isa, isaInfoMap, asm = isa_infrastructure
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    from Tensile.Components.CustomSchedule.approach_a import (
        build_non_cms_reference,
    )

    config = dict(CANONICAL_TF32_4X4_TN_CONFIG)

    # --- Build #1: CMS path — produces the CMS-side capture. We only
    # consume `_last_cms_capture` from this build; the SHADOW-extracted
    # `_last_default_capture` is intentionally discarded.
    cms_solution = _make_solution(config, asm, isaInfoMap)
    cms_writer = KernelWriterAssembly(asm, DebugConfig())
    try:
        cms_writer._getKernelSource(cms_solution)
    except Exception:
        # Pre-existing SHADOW-vs-CMS divergence may raise on the
        # in-build assert. The CMS-side FourPartCapture is populated
        # before the assert fires.
        pass
    cms_cap = cms_writer._last_cms_capture
    assert cms_cap is not None, (
        "CMS build did not populate _last_cms_capture; the kernelBody "
        "post-loop assembly stage did not run."
    )

    # --- Build #2: non-CMS reference via Approach A's helper.
    default_cap = build_non_cms_reference(config, asm, isaInfoMap)
    return default_cap, cms_cap


def _captures_per_body(four_part_capture):
    """Return `{body_label: LoopBodyCapture}` for the codepath-0 main bodies."""
    out = {}
    for label, by_cp in (
        (BODY_LABEL_ML_PREV, four_part_capture.main_loop_prev),
        (BODY_LABEL_ML, four_part_capture.main_loop),
        (BODY_LABEL_NGL, four_part_capture.n_gl),
        (BODY_LABEL_NLL, four_part_capture.n_ll),
    ):
        if 0 in by_cp:
            out[label] = by_cp[0]
    return out


def test_real_kernel_per_render_counts_match(
    real_kernel_capture_pair_approach_a,
):
    """Memo §6.2 #1: for every (body, canonical_render) in either real
    build's captures, both builds emit the same count.

    Iterates the shared `data_flow_instructions(body)` helper from
    `CMSValidator` so this test consumes the same scheduler-control
    exclusion abstraction as `build_dataflow_graph` Phase 1
    (rocm-libraries-d3zj). SWaitCnt / SBarrier / SNop / SSetPrior are
    not user-program semantics; comparing them across CMS and default
    captures would assert against scheduler choice.

    rocm-libraries-aixt: re-routed onto
    ``real_kernel_capture_pair_approach_a`` so the default-side capture
    comes from ``build_non_cms_reference`` (Build #2 finalize-after-
    closeLoop) rather than the SHADOW capture (which was missing the
    LCC pair in ML/ML-1). The xfail marker that pinned the SHADOW
    capture's missing-LCC defect was removed alongside the re-route.
    """
    from Tensile.Components.CMSValidator import data_flow_instructions

    default_cap, cms_cap = real_kernel_capture_pair_approach_a
    default_bodies = _captures_per_body(default_cap)
    cms_bodies = _captures_per_body(cms_cap)
    assert set(default_bodies) == set(cms_bodies), (
        f"Body-set mismatch between default and CMS captures: "
        f"default={sorted(default_bodies)} cms={sorted(cms_bodies)}"
    )
    mismatches = []
    for label in sorted(default_bodies):
        default_counts = _per_render_counts(
            default_bodies[label],
            instructions=data_flow_instructions(default_bodies[label]),
        )
        cms_counts = _per_render_counts(
            cms_bodies[label],
            instructions=data_flow_instructions(cms_bodies[label]),
        )
        all_renders = set(default_counts) | set(cms_counts)
        for r in all_renders:
            if default_counts.get(r, 0) != cms_counts.get(r, 0):
                mismatches.append((label, r,
                                   default_counts.get(r, 0),
                                   cms_counts.get(r, 0)))
    assert not mismatches, (
        f"{len(mismatches)} per-(body, render) count mismatches; "
        f"first 4: {mismatches[:4]}"
    )


def test_real_kernel_per_ordinal_logical_instruction_matches(
    real_kernel_capture_pair_approach_a,
):
    """Memo §6.2 #2: for every (body, canonical_render, ordinal) tuple
    appearing in either real build, both builds resolve to a
    TaggedInstruction whose underlying rocisa class name matches.

    Iterates the shared `data_flow_instructions(body)` helper from
    `CMSValidator` (rocm-libraries-d3zj); see the docstring on
    `test_real_kernel_per_render_counts_match` above for the rationale.

    rocm-libraries-aixt: re-routed onto
    ``real_kernel_capture_pair_approach_a`` (Approach A two-build).
    The xfail marker citing nyb5 as the principled fix was removed
    alongside the re-route — nyb5's helper is the fix.
    """
    from Tensile.Components.CMSValidator import data_flow_instructions

    default_cap, cms_cap = real_kernel_capture_pair_approach_a
    default_bodies = _captures_per_body(default_cap)
    cms_bodies = _captures_per_body(cms_cap)
    mismatches = []
    for label in sorted(set(default_bodies) | set(cms_bodies)):
        if label not in default_bodies or label not in cms_bodies:
            continue
        d_map = _per_render_ordinal_map(
            default_bodies[label],
            instructions=data_flow_instructions(default_bodies[label]),
        )
        c_map = _per_render_ordinal_map(
            cms_bodies[label],
            instructions=data_flow_instructions(cms_bodies[label]),
        )
        all_keys = set(d_map) | set(c_map)
        for key in all_keys:
            ti_d = d_map.get(key)
            ti_c = c_map.get(key)
            if ti_d is None or ti_c is None:
                mismatches.append((label, key,
                                   "absent" if ti_d is None else type(ti_d.wrapped.rocisa_inst).__name__,
                                   "absent" if ti_c is None else type(ti_c.wrapped.rocisa_inst).__name__))
                continue
            cd = type(ti_d.wrapped.rocisa_inst).__name__
            cc = type(ti_c.wrapped.rocisa_inst).__name__
            if cd != cc:
                mismatches.append((label, key, cd, cc))
    assert not mismatches, (
        f"{len(mismatches)} per-ordinal class-name mismatches; "
        f"first 4: {mismatches[:4]}"
    )


def test_example_yaml_no_spurious_order_inverted_failures(
    real_kernel_capture_pair,
):
    """Memo §6.2 #4: against `_get_schedule_160x128x64_TF32`, assert the
    8 spurious `OrderInvertedFailure`s from
    `EXAMPLE_YAML_DEFECT_INVESTIGATION.md §1` no longer fire under the
    new per-emission ordinal identity scheme."""
    from Tensile.Components.CMSValidator import (
        build_dataflow_graph, compare_graphs,
    )

    default_cap, cms_cap = real_kernel_capture_pair
    ref_graph = build_dataflow_graph(default_cap)
    subj_graph = build_dataflow_graph(cms_cap)
    failures = compare_graphs(ref_graph, subj_graph)
    order_inverted = [
        f for f in failures
        if type(f).__name__ == "OrderInvertedFailure"
    ]
    assert order_inverted == [], (
        f"Expected zero OrderInvertedFailures under the per-emission "
        f"ordinal identity scheme; got {len(order_inverted)}: "
        f"{[type(f).__name__ for f in order_inverted[:5]]}"
    )


# =============================================================================
# Per-body LCC invariant (rocm-libraries-d3zj part 2)
# =============================================================================
# The narrowed invariant from `D3ZJ_NGL_NLL_LCC_INVESTIGATION.md` Q6:
#
#   > Every body whose `useLoop=1` flag is set in the CMS MAINLOOP macro
#   > invocation has exactly one LCC pair (1 SSubU32 + 1 SCmpEQI32 on
#   > `sgprLoopCounterL`); bodies invoked with `useLoop=0` have zero LCC ops.
#
# The existing per-(body, render) and per-ordinal tests above only assert
# AGREEMENT between the default and CMS sides — they would silently pass
# if both sides emitted 0 LCC in ML by accident. This test pins the
# invariant EXPLICITLY per side so a "both wrong the same way" regression
# is caught.
#
# Mapping from body label to the `useLoop` value of its corresponding
# MAINLOOP invocation (per `KernelWriter.py:3134, 3137` and
# `D3ZJ_NGL_NLL_LCC_INVESTIGATION.md` Q4):
#
#   - BODY_LABEL_ML       (ML)    → useLoop=1
#                                   site 1 (kernel_cms.s:2196: `MAINLOOP 0`)
#                                   uses macro default useLoop=1 — the only
#                                   iterating body with a back-edge.
#   - BODY_LABEL_ML_PREV  (ML-1)  → useLoop=1
#                                   prev-codepath copy of the same iterating
#                                   loop body; same default useLoop=1.
#   - BODY_LABEL_NGL      (NGL)   → useLoop=0
#                                   `KernelWriter.py:3135`
#                                   (`MAINLOOP 0,0,1,1,0`) — NGLL tail.
#   - BODY_LABEL_NLL      (NLL)   → useLoop=0
#                                   `KernelWriter.py:3138`
#                                   (`MAINLOOP 0,0,0,0,0`) — non-NGLL tail.
#
# If `_emitNoLoadLoopBodyCMSMacro` ever gains a useLoop=1 NGL/NLL path,
# this map must be updated alongside that change — the inline citation
# above is the drift hook.
_BODY_LABEL_USE_LOOP = {
    BODY_LABEL_ML: 1,
    BODY_LABEL_ML_PREV: 1,
    BODY_LABEL_NGL: 0,
    BODY_LABEL_NLL: 0,
}


def _count_lcc_on_loop_counter_l(capture):
    """Count `(scmp_eq_i32, ssub_u32)` instances whose canonical render
    references `sgprLoopCounterL`.

    Uses the rocisa class name as the structural filter (matching
    `test_dataflow_graph_lcc.py`'s `lcc_nodes` walk) AND filters by the
    presence of `sgprLoopCounterL` in the canonical render so we don't
    accidentally pick up tail-loop reuse of the same instruction class on
    `sgprSizesSum` / etc. Per `D3ZJ_NGL_NLL_LCC_INVESTIGATION.md` Q3, NGL/
    NLL still touch `sgprLoopCounterL` for tail-edge masking (`s_and_b32`,
    `s_cmov_b32`, `s_cmp_eq_u32`) but those are different rocisa classes
    or different unsigned variants — the LCC pair is specifically
    `SCmpEQI32` + `SSubU32` on the L counter.
    """
    cmp_count = 0
    sub_count = 0
    for ti in capture.instructions:
        cls_name = type(ti.wrapped.rocisa_inst).__name__
        if cls_name not in ("SCmpEQI32", "SSubU32"):
            continue
        render = WrappedInstruction.canonical_str(ti.wrapped.rocisa_inst)
        if "sgprLoopCounterL" not in render:
            continue
        if cls_name == "SCmpEQI32":
            cmp_count += 1
        else:
            sub_count += 1
    return cmp_count, sub_count


def test_lcc_invariant_per_body_use_loop_predicate(
    real_kernel_capture_pair_approach_a,
):
    """Memo Q6 (D3ZJ_NGL_NLL_LCC_INVESTIGATION.md): assert the
    per-body LCC invariant on EACH side of the capture pair independently.

    For each side (default, CMS) and each body present in that side's
    capture, assert that the count of (SCmpEQI32, SSubU32) instructions
    whose canonical render references `sgprLoopCounterL` matches the
    body's `useLoop` flag from `_BODY_LABEL_USE_LOOP`:

      - useLoop=1 body → exactly (1, 1)
      - useLoop=0 body → exactly (0, 0)

    Unlike `test_real_kernel_per_render_counts_match` (which only checks
    that the two sides AGREE), this test would catch a regression where
    both sides emit 0 LCC in ML by accident — the invariant is enforced
    per side without cross-side reference.

    rocm-libraries-aixt: re-routed onto
    ``real_kernel_capture_pair_approach_a`` (Approach A two-build); the
    xfail marker citing nyb5 as the principled fix was removed
    alongside the re-route. The default side now comes from
    ``build_non_cms_reference``, whose post-closeLoop finalize captures
    LCC correctly.
    """
    default_cap, cms_cap = real_kernel_capture_pair_approach_a
    violations = []
    for side_name, four_part in (("default", default_cap), ("cms", cms_cap)):
        bodies = _captures_per_body(four_part)
        for label, body in bodies.items():
            expected_use_loop = _BODY_LABEL_USE_LOOP.get(label)
            if expected_use_loop is None:
                continue  # unknown body label — skip rather than guess
            cmp_count, sub_count = _count_lcc_on_loop_counter_l(body)
            expected_pair = (1, 1) if expected_use_loop == 1 else (0, 0)
            actual_pair = (cmp_count, sub_count)
            if actual_pair != expected_pair:
                violations.append((side_name, label, expected_use_loop,
                                   expected_pair, actual_pair))
    assert not violations, (
        f"{len(violations)} per-body LCC invariant violation(s) "
        f"(side, body, useLoop, expected (cmp,sub), actual (cmp,sub)): "
        f"{violations}"
    )
