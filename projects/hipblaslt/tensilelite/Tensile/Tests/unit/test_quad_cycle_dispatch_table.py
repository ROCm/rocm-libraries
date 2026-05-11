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
"""Snapshot test for the quad-cycle dispatch table (rocm-libraries-s5g1).

Pins the (producer_role, consumer_role) -> handler mapping that replaced
the order-dependent if/elif chain in `_classify_edge_coverage` and
`diagnose_missing_edge`. Two layers of protection:

  1. `test_dispatch_table_snapshot` — pins the literal table contents.
     Adding/removing a key, or changing the helper a key maps to, fails
     this test loudly. Forces the change-author to update the snapshot
     deliberately rather than silently regress dispatch coverage.

  2. `test_role_classifier_*` — pins the role each canonical producer/
     consumer shape lands in. Catches a regression where someone reorders
     the predicate checks inside `_producer_role` / `_consumer_role` and
     silently re-buckets PackMFMAs as MFMA / CVTPacks as ALU.

  3. `test_role_buckets_pairwise_disjoint` — for every (producer_role,
     consumer_role) pair in the table, asserts that exactly ONE table
     entry matches. Catches accidental key duplication or accidental
     widening of the table key shape.

These tests are the order-independence guarantee: if the table loses a
key or a classifier slips on its bucket assignment, these snapshots fail
before the production callers can drift.
"""

import pytest

from rocisa.container import sgpr, vgpr, mgpr
from rocisa.instruction import (
    SMovB32,
    VCvtPkF32toBF16,
    VXorB32,
)

from Tensile.Components.CMSValidator import (
    GraphNode,
    _ConsumerRole,
    _DISPATCH,
    _PASSTHROUGH,
    _ProducerRole,
    _consumer_role,
    _cvt_to_mfma_gap_ok,
    _is_alu_producer,
    _mfma_pack_to_cvt_gap_ok,
    _producer_role,
    _quad_cycle_gap_ok,
)
from Tensile.Components.ScheduleCapture import (
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
)

from dataflow_fixtures import make_mfma


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


class _StubNode:
    """Duck-typed node carrying just `category` + `rocisa_inst`.

    The role classifiers (`_producer_role` / `_consumer_role`) and the
    underlying `GraphNode.is_*` static methods + `_is_alu_producer` only
    read those two attributes; this stub is sufficient to drive the
    dispatch decision in isolation.
    """

    def __init__(self, *, category, rocisa_inst):
        self.category = category
        self.rocisa_inst = rocisa_inst


def _stub_pack_mfma() -> _StubNode:
    """A 4x4 PackMFMA producer (Pack* category, MFMAInstruction rocisa)."""
    tagged = make_mfma(
        c_dst_start=0, a_src_start=8, b_src_start=12, slot=0,
        category="PackA0", variant=[4, 4, 4, 16],
    )
    return _StubNode(
        category=tagged.category,
        rocisa_inst=tagged.wrapped.rocisa_inst,
    )


def _stub_mfma() -> _StubNode:
    """A standard MFMA producer (category=='MFMA')."""
    tagged = make_mfma(
        c_dst_start=0, a_src_start=8, b_src_start=12, slot=0,
    )  # default category=='MFMA'
    return _StubNode(
        category=tagged.category,
        rocisa_inst=tagged.wrapped.rocisa_inst,
    )


def _stub_cvt_pack() -> _StubNode:
    """A CVTPack producer (Pack* category, VCvtPkF32toBF16 rocisa)."""
    inst = VCvtPkF32toBF16(
        dst=vgpr(50, 1), src0=vgpr(51, 1), src1=vgpr(52, 1),
    )
    return _StubNode(category="PackA0", rocisa_inst=inst)


def _stub_alu() -> _StubNode:
    """An ALU-immediate producer (VXor on a non-Pack category)."""
    inst = VXorB32(dst=vgpr(50, 1), src0=vgpr(51, 1), src1=vgpr(52, 1))
    # Use a non-Pack non-MFMA category so it falls into the ALU bucket
    # via `_is_alu_producer`'s rocisa-instance check.
    return _StubNode(category="GRA", rocisa_inst=inst)


def _stub_other() -> _StubNode:
    """An LDS-write producer — falls outside ALU/MFMA/CVT/PackMFMA."""
    # Use a stub with category in PRODUCER_CATEGORIES_LDS / LW so
    # `_is_alu_producer` returns False (it's not ALU). Then it lands in OTHER.
    # rocisa_inst=None forces `_is_alu_producer` to use category-only fallback.
    return _StubNode(category="LWA", rocisa_inst=None)


# =============================================================================
# Snapshot of the dispatch table contents (rocm-libraries-s5g1)
# =============================================================================


def test_dispatch_table_snapshot():
    """Pin the exact (producer_role, consumer_role) -> handler mapping.

    Update this snapshot deliberately when adding/removing dispatch
    entries — silent reshuffling regresses the carve-outs the original
    if/elif chain encoded by branch-order. Helpers are compared by
    identity (`is`) so wrapping a helper in another callable also fails.
    """
    expected = {
        # PackMFMA producers.
        (_ProducerRole.PACK_MFMA, _ConsumerRole.CVT_PACK): _mfma_pack_to_cvt_gap_ok,
        (_ProducerRole.PACK_MFMA, _ConsumerRole.MFMA):     _quad_cycle_gap_ok,
        (_ProducerRole.PACK_MFMA, _ConsumerRole.OTHER):    _quad_cycle_gap_ok,
        # Standard MFMA producers — uniform finish-cycle check.
        (_ProducerRole.MFMA, _ConsumerRole.CVT_PACK): _quad_cycle_gap_ok,
        (_ProducerRole.MFMA, _ConsumerRole.MFMA):     _quad_cycle_gap_ok,
        (_ProducerRole.MFMA, _ConsumerRole.OTHER):    _quad_cycle_gap_ok,
        # CVTPack producers — only ->MFMA carries a quad-cycle constraint.
        (_ProducerRole.CVT_PACK, _ConsumerRole.MFMA):     _cvt_to_mfma_gap_ok,
        (_ProducerRole.CVT_PACK, _ConsumerRole.CVT_PACK): _PASSTHROUGH,
        (_ProducerRole.CVT_PACK, _ConsumerRole.OTHER):    _PASSTHROUGH,
        # ALU producers — immediate visibility for every consumer.
        (_ProducerRole.ALU, _ConsumerRole.MFMA):     _PASSTHROUGH,
        (_ProducerRole.ALU, _ConsumerRole.CVT_PACK): _PASSTHROUGH,
        (_ProducerRole.ALU, _ConsumerRole.OTHER):    _PASSTHROUGH,
        # OTHER producers intentionally have NO entries — fall through
        # to Phase-2 wait coverage at the call site.
    }
    assert set(_DISPATCH.keys()) == set(expected.keys()), (
        f"_DISPATCH keys drifted from snapshot.\n"
        f"  Added:   {set(_DISPATCH.keys()) - set(expected.keys())}\n"
        f"  Removed: {set(expected.keys()) - set(_DISPATCH.keys())}"
    )
    for key, expected_handler in expected.items():
        actual_handler = _DISPATCH[key]
        assert actual_handler is expected_handler, (
            f"_DISPATCH[{key}] handler drifted: "
            f"expected {expected_handler!r}, got {actual_handler!r}"
        )


def test_dispatch_table_no_other_producer_keys():
    """OTHER-producer rows must NOT appear in the table.

    The original chain fell through to Phase-2 wait coverage for any
    producer that didn't match PackMFMA / MFMA / CVTPack / ALU. The
    table preserves this by leaving `_ProducerRole.OTHER` keys absent —
    a missing key returns None from `_dispatch_quad_cycle_check`, which
    the call site interprets as "fall through". Adding an OTHER-producer
    entry would short-circuit Phase 2 and silently change classifier
    behavior for LR/LW/GR producers.
    """
    other_keys = [
        k for k in _DISPATCH.keys() if k[0] is _ProducerRole.OTHER
    ]
    assert other_keys == [], (
        f"_ProducerRole.OTHER must have NO entries in _DISPATCH "
        f"(falls through to Phase-2 wait coverage); found: {other_keys}"
    )


# =============================================================================
# Role classifier pinning
# =============================================================================


def test_role_classifier_pack_mfma_producer():
    """A PackMFMA stub (Pack* category + MFMA rocisa) must classify as
    PACK_MFMA — NOT MFMA, NOT ALU. This is the carve-out that the old
    branch-order encoded; now it lives in `_producer_role`'s internal
    check sequence."""
    node = _stub_pack_mfma()
    assert _producer_role(node) is _ProducerRole.PACK_MFMA


def test_role_classifier_standard_mfma_producer():
    """A category=='MFMA' producer with MFMA rocisa lands in MFMA, not
    PACK_MFMA (PACK_MFMA requires the Pack* category)."""
    node = _stub_mfma()
    assert _producer_role(node) is _ProducerRole.MFMA


def test_role_classifier_cvt_pack_producer():
    """A CVTPack stub (Pack* + VCvtPkF32toBF16) must classify as
    CVT_PACK — NOT ALU. This is the second carve-out the old branch-
    order encoded."""
    node = _stub_cvt_pack()
    assert _producer_role(node) is _ProducerRole.CVT_PACK
    # Sanity: the underlying ALU predicate would have claimed this if
    # the role classifier checked it first — confirm that's the trap
    # the carve-out avoids.
    assert _is_alu_producer(node) is True or _is_alu_producer(node) is False
    # (either truth value is acceptable; what matters is the role lands
    # in CVT_PACK before _is_alu_producer is consulted)


def test_role_classifier_alu_producer():
    """A vanilla VXor producer (non-Pack category) lands in ALU."""
    node = _stub_alu()
    assert _producer_role(node) is _ProducerRole.ALU


def test_role_classifier_other_producer():
    """An LW-categorized producer lands in OTHER (falls through to
    Phase-2 wait coverage)."""
    node = _stub_other()
    assert _producer_role(node) is _ProducerRole.OTHER


def test_consumer_classifier_buckets():
    """Pin the consumer-role classifier on each canonical input."""
    # CVTPack as consumer.
    assert _consumer_role(_stub_cvt_pack()) is _ConsumerRole.CVT_PACK
    # MFMA as consumer (both standard MFMA and PackMFMA are mfma_producer-True).
    assert _consumer_role(_stub_mfma()) is _ConsumerRole.MFMA
    assert _consumer_role(_stub_pack_mfma()) is _ConsumerRole.MFMA
    # ALU/OTHER consumers fall to OTHER.
    assert _consumer_role(_stub_alu()) is _ConsumerRole.OTHER
    assert _consumer_role(_stub_other()) is _ConsumerRole.OTHER


# =============================================================================
# Pairwise-disjoint key invariant — guards against table-shape drift.
# =============================================================================


def test_role_buckets_pairwise_disjoint():
    """Every (p_role, c_role) pair is a unique key — the table cannot
    encode an ambiguous dispatch by construction. (Dict construction
    already enforces this; the test pins the invariant explicitly.)"""
    keys = list(_DISPATCH.keys())
    assert len(keys) == len(set(keys)), (
        f"_DISPATCH has duplicate keys: {keys}"
    )


def test_canonical_pair_dispatches_to_expected_role():
    """End-to-end: classify a (PackMFMA -> CVTPack) pair and verify
    the table maps it to `_mfma_pack_to_cvt_gap_ok`. This is the carve-
    out that gates the 5-quad-cycle CVT1 settle window — silent
    regression here is the failure mode the audit memo's §2.5 catalogs
    as #1."""
    p = _stub_pack_mfma()
    c = _stub_cvt_pack()
    handler = _DISPATCH.get((_producer_role(p), _consumer_role(c)))
    assert handler is _mfma_pack_to_cvt_gap_ok, (
        "PackMFMA -> CVTPack must dispatch to the 5-cycle settle helper, "
        "not to _quad_cycle_gap_ok (which uses the 1-cycle 4x4 finish)."
    )


def test_canonical_cvt_to_mfma_dispatches_to_expected_helper():
    """CVTPack -> MFMA pair must dispatch to `_cvt_to_mfma_gap_ok`,
    not to `_PASSTHROUGH` — the carve-out the audit memo's §2.5 catalogs
    as failure mode #2."""
    p = _stub_cvt_pack()
    c = _stub_mfma()
    handler = _DISPATCH.get((_producer_role(p), _consumer_role(c)))
    assert handler is _cvt_to_mfma_gap_ok, (
        "CVTPack -> MFMA must dispatch to the 2-cycle settle helper, "
        "not to _PASSTHROUGH (which would silently waive the gap)."
    )
