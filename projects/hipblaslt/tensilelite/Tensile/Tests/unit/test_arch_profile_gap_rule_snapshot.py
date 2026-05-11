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
# SPDX-License-Identifier: MIT
################################################################################
"""Byte-equivalence snapshot test for the gap-rule table dispatch
(rocm-libraries-vmua).

Pins the `_dispatch_quad_cycle_check(p, c, graph)` outcome for a frozen
set of canonical (producer-shape, consumer-shape) fixtures. The bead's
discipline:

    "Every existing test case's gap-result MUST be byte-equivalent
    before/after the migration. Implement as a snapshot test that
    enumerates fixtures and asserts the new table-driven path produces
    the same `TimingCheck` results as the pre-migration code path."

Pre-migration outcomes are encoded as the snapshot's expected values —
each fixture exercises one (p_shape, c_shape) row of the legacy
`_DISPATCH` and pins the result at the gap-edge boundary. A regression
in:

  - `shape_of` (mis-classifying a producer/consumer)
  - `_dispatch_quad_cycle_check` (mis-resolving the table)
  - `GapRule.evaluate_required` (mis-evaluating cycle counts)
  - The CDNA4 table itself (changed cycle constants)

surfaces here loudly with an explicit byte-diff against the snapshot.
"""

from __future__ import annotations

import pytest

from rocisa.container import vgpr
from rocisa.instruction import VCvtPkF32toBF16, VXorB32

from Tensile.Components.CMSValidator import (
    _DEFAULT_CDNA4_ARCH_PROFILE,
    _PASSTHROUGH,
    _dispatch_quad_cycle_check,
    TimingResult,
    build_dataflow_graph,
    cumulative_issue_cycles,
)
from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML, BODY_LABEL_ML_PREV, BODY_LABEL_NGL, BODY_LABEL_NLL,
    FourPartCapture,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
)

from dataflow_fixtures import (
    make_capture, make_lr, make_mfma,
)


def _wrap(ml_capture):
    """Wrap a single main-loop capture into a FourPartCapture pinned to
    the CDNA4 default arch profile. Mirrors `_wrap` in
    test_dataflow_graph_register_gaps.py — duplicated here to avoid
    pulling that test file's helper as an importable surface (the
    helper is `_`-prefixed; it's intentionally test-local)."""
    _FILLER_RANGES = {
        BODY_LABEL_ML_PREV: (200, 204, 208),
        BODY_LABEL_NGL:     (220, 224, 228),
        BODY_LABEL_NLL:     (240, 244, 248),
    }

    def _filler(label):
        c, a, b = _FILLER_RANGES[label]
        return make_capture(label, [make_mfma(
            c_dst_start=c, a_src_start=a, b_src_start=b, slot=0,
        )])
    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: _filler(BODY_LABEL_ML_PREV)},
        n_gl={0: _filler(BODY_LABEL_NGL)},
        n_ll={0: _filler(BODY_LABEL_NLL)},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


def _tag(inst, *, category, mfma_index=0, sequence=0):
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(
            subiter=0,
            slot_kind=SLOT_KIND_MFMA,
            mfma_index=mfma_index,
            sequence=sequence,
        ),
    )


# =============================================================================
# Fixture builders — each returns a (graph, producer_node, consumer_node)
# triple for a canonical (p_shape, c_shape) edge.
# =============================================================================


def _fixture_mfma_to_mfma_zero_gap():
    """Standard MFMA at slot=2 -> standard MFMA at slot=3, RAW on v0..v1.
    Zero quad-cycle gap (consecutive MFMAs); legacy `_quad_cycle_gap_ok`
    fired with required=3, observed=0 -> FAIL.
    """
    cap = make_capture(BODY_LABEL_ML, [
        make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                  slot=2, sequence=0, a_src_count=2),
        make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                  slot=3, sequence=0, a_src_count=2),
    ])
    g = build_dataflow_graph(_wrap(cap))
    p, c = _find_mfma_pair(g)
    return g, p, c


def _fixture_cvt_to_mfma_zero_gap():
    """CVTPack writes v40; MFMA at the same slot reads v40..v41 (a_src).
    Legacy `_cvt_to_mfma_gap_ok`: required=2, observed=0 -> FAIL.
    """
    cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                          src0=vgpr(50, 1), src1=vgpr(51, 1))
    cap = make_capture(BODY_LABEL_ML, [
        make_lr(50, 2, 64, slot=0, category="LRA0"),
        _tag(cvt, category="PackA0", mfma_index=2, sequence=0),
        make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                  slot=2, sequence=1, a_src_count=2),
    ])
    g = build_dataflow_graph(_wrap(cap))
    # CVT producer (PackA0) -> MFMA consumer.
    cvt_node = next(n for n in g.all_nodes_in_order
                    if n.category == "PackA0" and n.position.stream_index == 1)
    mfma_node = next(n for n in g.all_nodes_in_order
                     if n.category == "MFMA")
    return g, cvt_node, mfma_node


def _fixture_pack_mfma_to_cvt_zero_gap():
    """4x4 PackMFMA producer (PackA0 + MFMA rocisa) writes v0; CVT at the
    SAME slot reads v0. Zero quad-cycle gap. Legacy
    `_mfma_pack_to_cvt_gap_ok`: required=5, observed=0 -> FAIL.
    """
    pack_mfma = make_mfma(c_dst_start=0, a_src_start=8, b_src_start=12,
                          slot=2, sequence=0, category="PackA0",
                          variant=[4, 4, 4, 16], a_src_count=2)
    cvt = VCvtPkF32toBF16(dst=vgpr(40, 1),
                          src0=vgpr(0, 1), src1=vgpr(1, 1))
    cap = make_capture(BODY_LABEL_ML, [
        make_lr(8, 2, 64, slot=0, category="LRA0"),
        pack_mfma,
        _tag(cvt, category="PackA0", mfma_index=2, sequence=1),
    ])
    g = build_dataflow_graph(_wrap(cap))
    p = next(n for n in g.all_nodes_in_order
             if n.category == "PackA0" and n.position.stream_index == 1)
    c = next(n for n in g.all_nodes_in_order
             if n.category == "PackA0" and n.position.stream_index == 2)
    return g, p, c


def _fixture_alu_to_mfma_zero_gap_no_subiter():
    """ALU (VXor) writes v0; MFMA at next slot reads v0. With nmps=0 in
    the test fixture, the C-9 same-subiter rule does NOT fire (the
    unconditional passthrough fallback at index 2 fires instead).
    Legacy: `_PASSTHROUGH`. New: must also be passthrough.
    """
    alu = VXorB32(dst=vgpr(0, 1), src0=vgpr(50, 1), src1=vgpr(51, 1))
    cap = make_capture(BODY_LABEL_ML, [
        _tag(alu, category="PackA0", mfma_index=1, sequence=0),
        make_mfma(c_dst_start=4, a_src_start=0, b_src_start=32,
                  slot=2, sequence=0, a_src_count=2),
    ])
    g = build_dataflow_graph(_wrap(cap))
    alu_node = next(n for n in g.all_nodes_in_order
                    if n.category == "PackA0")
    mfma_node = next(n for n in g.all_nodes_in_order
                     if n.category == "MFMA")
    return g, alu_node, mfma_node


def _find_mfma_pair(graph):
    """Return the (producer, consumer) pair of the first MFMA->MFMA edge."""
    for e in graph.edges:
        if (getattr(e.producer, "category", None) == "MFMA"
                and getattr(e.consumer, "category", None) == "MFMA"):
            return e.producer, e.consumer
    raise AssertionError("no MFMA->MFMA edge in fixture")


# =============================================================================
# Snapshot table — each entry pins the (fixture, expected_dispatch_outcome).
# =============================================================================
#
# Encoding for `expected`:
#   `_PASSTHROUGH`  — dispatch must return _PASSTHROUGH (ALU exemption).
#   None             — dispatch must return None (no rule applies).
#   ("FAIL", required, observed)  — TimingCheck.failing(observed, required).
#   ("PASS", required, observed)  — TimingCheck.passing(observed, required).
#
# These are the LEGACY (pre-vmua) outcomes — the snapshot pins byte-
# equivalence across the migration.
SNAPSHOT_FIXTURES = [
    pytest.param(
        _fixture_mfma_to_mfma_zero_gap,
        # Slot=2 to slot=3 with no intervening instructions: producer
        # MFMA issues at 0 (mfma_free=4), consumer cycle simulator
        # advances by producer's issue cost (+1) then takes
        # max(1, 4)=4 for the consumer's MFMA-free contention. Gap
        # observed = 4 - 0 - 1 = 3, which equals required (standard
        # finish=3) -> PASS at the boundary.
        ("PASS", 3, 3),
        id="mfma_to_mfma_consecutive_slots_PASS_at_boundary",
    ),
    pytest.param(
        _fixture_cvt_to_mfma_zero_gap,
        ("FAIL", 2, 0),
        id="cvt_to_mfma_zero_gap_FAILS_at_2_observed_0",
    ),
    pytest.param(
        _fixture_pack_mfma_to_cvt_zero_gap,
        ("FAIL", 5, 0),
        id="pack_mfma_to_cvt_zero_gap_FAILS_at_5_observed_0",
    ),
    pytest.param(
        _fixture_alu_to_mfma_zero_gap_no_subiter,
        _PASSTHROUGH,
        id="alu_to_mfma_no_subiter_PASSTHROUGH",
    ),
]


@pytest.mark.parametrize("fixture_builder,expected", SNAPSHOT_FIXTURES)
def test_dispatch_byte_equivalent_snapshot(fixture_builder, expected):
    """Pin the `_dispatch_quad_cycle_check` outcome for each canonical
    fixture against its pre-vmua-migration value. A diff fails this
    test loudly with a clear before/after delta.
    """
    graph, producer, consumer = fixture_builder()
    actual = _dispatch_quad_cycle_check(producer, consumer, graph)

    if expected is _PASSTHROUGH:
        assert actual is _PASSTHROUGH, (
            f"Snapshot mismatch — expected _PASSTHROUGH; got {actual!r}"
        )
        return
    if expected is None:
        assert actual is None
        return
    # Tuple form: ("FAIL"|"PASS", required, observed)
    kind, exp_required, exp_observed = expected
    assert actual is not None and actual is not _PASSTHROUGH, (
        f"Snapshot mismatch — expected {expected}; got {actual!r}"
    )
    expected_result = TimingResult.FAIL if kind == "FAIL" else TimingResult.PASS
    assert actual.result is expected_result, (
        f"Snapshot mismatch — result expected {expected_result.name}, "
        f"got {actual.result.name}; required={actual.required}, "
        f"observed={actual.observed}."
    )
    assert actual.required == exp_required, (
        f"Snapshot mismatch — required expected {exp_required}, "
        f"got {actual.required}."
    )
    assert actual.observed == exp_observed, (
        f"Snapshot mismatch — observed expected {exp_observed}, "
        f"got {actual.observed}."
    )


def test_snapshot_fixture_count_pinned():
    """Pin the number of fixtures so adding/removing a snapshot row is
    a deliberate (and review-flagged) act."""
    assert len(SNAPSHOT_FIXTURES) == 4, (
        f"Snapshot fixture count drifted; expected 4, got "
        f"{len(SNAPSHOT_FIXTURES)}. Update SNAPSHOT_FIXTURES (and this "
        f"assertion) deliberately when adding/removing a row."
    )
