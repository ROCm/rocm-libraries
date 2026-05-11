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
"""Snapshot tests for the gap-rule table dispatch (rocm-libraries-vmua).

Replaces the legacy `_DISPATCH` snapshot (which keyed on `(_ProducerRole,
_ConsumerRole)` and stored helper functions). After vmua, the dispatch is
data-driven via `_DEFAULT_CDNA4_ARCH_PROFILE.gap_rules` keyed on
`(InstructionShape, InstructionShape)` with `List[GapRule]` values.

Three snapshot layers protect against silent regression:

  1. `test_gap_rule_table_keys_snapshot` — pins the exact set of
     `(p_shape, c_shape)` keys present in the table. Adding/removing a
     key changes the dispatch surface and must be reflected here.

  2. `test_gap_rule_table_canonical_pair_dispatches` — end-to-end checks
     for each load-bearing carve-out the legacy `_DISPATCH` encoded:
     PackMFMA->CVT_PACK gets the 5-cycle settle window; CVT_PACK->MFMA
     gets the 2-cycle window; ALU->MFMA passes through (cross-subiter
     fixture); 4x4 PackMFMA producers route to the 1-cycle finish.

  3. `test_shape_classifier_*` — pins the `shape_of(node)` classifier
     output for each canonical producer/consumer shape. A regression
     where `shape_of` re-buckets a PackMFMA as MFMA_STANDARD or a
     CVTPack as ALU is caught here before the table dispatch can drift.
"""

import pytest

from rocisa.container import sgpr, vgpr, mgpr
from rocisa.instruction import (
    SMovB32,
    VCvtPkF32toBF16,
    VXorB32,
)

from Tensile.Components.CMSValidator import (
    GapRule,
    GraphNode,
    _DEFAULT_CDNA4_ARCH_PROFILE,
    _GAP_RULE_PASSTHROUGH,
    _is_alu_producer,
)
from Tensile.Components.InstructionShape import (
    InstructionShape,
    shape_of,
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
    """Duck-typed node with `category` + `rocisa_inst` (only attributes
    `shape_of` reads). Sufficient for shape classification in isolation.
    """

    def __init__(self, *, category, rocisa_inst):
        self.category = category
        self.rocisa_inst = rocisa_inst


def _stub_pack_mfma() -> _StubNode:
    """4x4 PackMFMA producer (Pack* category, MFMAInstruction rocisa)."""
    tagged = make_mfma(
        c_dst_start=0, a_src_start=8, b_src_start=12, slot=0,
        category="PackA0", variant=[4, 4, 4, 16],
    )
    return _StubNode(
        category=tagged.category,
        rocisa_inst=tagged.wrapped.rocisa_inst,
    )


def _stub_mfma() -> _StubNode:
    """Standard MFMA producer (category=='MFMA')."""
    tagged = make_mfma(
        c_dst_start=0, a_src_start=8, b_src_start=12, slot=0,
    )  # default category=='MFMA'
    return _StubNode(
        category=tagged.category,
        rocisa_inst=tagged.wrapped.rocisa_inst,
    )


def _stub_cvt_pack() -> _StubNode:
    """CVTPack producer (Pack* category, VCvtPkF32toBF16 rocisa)."""
    inst = VCvtPkF32toBF16(
        dst=vgpr(50, 1), src0=vgpr(51, 1), src1=vgpr(52, 1),
    )
    return _StubNode(category="PackA0", rocisa_inst=inst)


def _stub_alu() -> _StubNode:
    """ALU-immediate producer (VXor with category="GRA").

    Mirrors the legacy `_is_alu_producer` semantic: when a rocisa
    instance is present, its CLASS (not its emission category) determines
    whether it's ALU. VXorB32 is not registered in `InstructionCategory`,
    so `shape_of` returns ALU regardless of the surrounding category
    bucket. This pins the same DTL-m0-setter shape derivation that
    distinguishes a real GR (BufferLoad with category="GRA") from an ALU
    (SMov m0-setter with category="GRA").
    """
    inst = VXorB32(dst=vgpr(50, 1), src0=vgpr(51, 1), src1=vgpr(52, 1))
    return _StubNode(category="GRA", rocisa_inst=inst)


def _stub_lr() -> _StubNode:
    """LR (DS-load) producer — no rocisa instance, category='LRA0'."""
    return _StubNode(category="LRA0", rocisa_inst=None)


# =============================================================================
# Snapshot of the gap-rule table key set (rocm-libraries-vmua)
# =============================================================================


def test_gap_rule_table_keys_snapshot():
    """Pin the exact `(p_shape, c_shape)` key set in the CDNA4 gap-rule
    table. Adding or removing a key must update this snapshot deliberately
    — the gap-rule matrix is the single dispatch source after vmua, so a
    silent key drift directly changes which edges get gap-checked.
    """
    expected_keys = {
        # MFMA_STANDARD producer.
        (InstructionShape.MFMA_STANDARD, InstructionShape.MFMA_STANDARD),
        (InstructionShape.MFMA_STANDARD, InstructionShape.MFMA_4x4),
        (InstructionShape.MFMA_STANDARD, InstructionShape.CVT_PACK),
        (InstructionShape.MFMA_STANDARD, InstructionShape.MIDDLE_PACK),
        (InstructionShape.MFMA_STANDARD, InstructionShape.OTHER),
        (InstructionShape.MFMA_STANDARD, InstructionShape.ALU),
        (InstructionShape.MFMA_STANDARD, InstructionShape.LR),
        (InstructionShape.MFMA_STANDARD, InstructionShape.GR),
        # MFMA_4x4 producer.
        (InstructionShape.MFMA_4x4, InstructionShape.MFMA_STANDARD),
        (InstructionShape.MFMA_4x4, InstructionShape.MFMA_4x4),
        (InstructionShape.MFMA_4x4, InstructionShape.MIDDLE_PACK),
        (InstructionShape.MFMA_4x4, InstructionShape.OTHER),
        (InstructionShape.MFMA_4x4, InstructionShape.CVT_PACK),
        (InstructionShape.MFMA_4x4, InstructionShape.ALU),
        (InstructionShape.MFMA_4x4, InstructionShape.LR),
        (InstructionShape.MFMA_4x4, InstructionShape.GR),
        # CVT_PACK producer.
        (InstructionShape.CVT_PACK, InstructionShape.MFMA_STANDARD),
        (InstructionShape.CVT_PACK, InstructionShape.MFMA_4x4),
        (InstructionShape.CVT_PACK, InstructionShape.CVT_PACK),
        (InstructionShape.CVT_PACK, InstructionShape.MIDDLE_PACK),
        (InstructionShape.CVT_PACK, InstructionShape.OTHER),
        # ALU producer (passthrough for most consumers; same-subiter
        # carve-out for MFMA consumers).
        (InstructionShape.ALU, InstructionShape.MFMA_STANDARD),
        (InstructionShape.ALU, InstructionShape.MFMA_4x4),
        (InstructionShape.ALU, InstructionShape.ALU),
        (InstructionShape.ALU, InstructionShape.CVT_PACK),
        (InstructionShape.ALU, InstructionShape.MIDDLE_PACK),
        (InstructionShape.ALU, InstructionShape.LR),
        (InstructionShape.ALU, InstructionShape.LW),
        (InstructionShape.ALU, InstructionShape.GR),
        (InstructionShape.ALU, InstructionShape.OTHER),
        # MIDDLE_PACK producer — passthrough only (TF32-emul ALU; pair
        # interleaving handled by validate_middle_pack_pair_interleaving).
        (InstructionShape.MIDDLE_PACK, InstructionShape.MFMA_STANDARD),
        (InstructionShape.MIDDLE_PACK, InstructionShape.MFMA_4x4),
        (InstructionShape.MIDDLE_PACK, InstructionShape.CVT_PACK),
        (InstructionShape.MIDDLE_PACK, InstructionShape.MIDDLE_PACK),
        (InstructionShape.MIDDLE_PACK, InstructionShape.ALU),
        (InstructionShape.MIDDLE_PACK, InstructionShape.LR),
        (InstructionShape.MIDDLE_PACK, InstructionShape.LW),
        (InstructionShape.MIDDLE_PACK, InstructionShape.GR),
        (InstructionShape.MIDDLE_PACK, InstructionShape.OTHER),
    }
    actual_keys = set(_DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.keys())
    assert actual_keys == expected_keys, (
        f"_DEFAULT_CDNA4_ARCH_PROFILE.gap_rules keys drifted from snapshot.\n"
        f"  Added:   {actual_keys - expected_keys}\n"
        f"  Removed: {expected_keys - actual_keys}"
    )


def test_gap_rule_table_no_other_producer_keys():
    """`InstructionShape.OTHER` and the LR/LW/GR/SMEM/FLAT/VECTOR_STORE
    producer-side shapes must NOT appear as keys in the table. Their edges
    fall through to Phase-2 wait coverage (the legacy `_DISPATCH` left
    `_ProducerRole.OTHER` keys absent for the same reason). Adding such a
    key would short-circuit Phase 2 for LR/LW/GR producers and change
    wait-coverage classifier behavior silently.
    """
    forbidden_producers = {
        InstructionShape.OTHER, InstructionShape.LR, InstructionShape.LW,
        InstructionShape.GR, InstructionShape.SMEM, InstructionShape.FLAT,
        InstructionShape.VECTOR_STORE, InstructionShape.SWAIT,
        InstructionShape.SBARRIER, InstructionShape.SNOP,
        InstructionShape.SSETPRIO,
    }
    bad = [
        k for k in _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.keys()
        if k[0] in forbidden_producers
    ]
    assert bad == [], (
        f"Forbidden producer-side shapes appeared in gap_rules; these "
        f"shapes must fall through to Phase-2 wait coverage instead of "
        f"being claimed by a gap rule. Found: {bad}"
    )


# =============================================================================
# Shape classifier pinning (replaces the legacy role-classifier tests).
# =============================================================================


def test_shape_classifier_pack_mfma_producer():
    """A 4x4 PackMFMA stub (Pack* + MFMAInstruction) must classify as
    MFMA_4x4 — not MFMA_STANDARD, not ALU."""
    assert shape_of(_stub_pack_mfma()) is InstructionShape.MFMA_4x4


def test_shape_classifier_standard_mfma_producer():
    """category=='MFMA' producer with MFMA rocisa lands in MFMA_STANDARD."""
    assert shape_of(_stub_mfma()) is InstructionShape.MFMA_STANDARD


def test_shape_classifier_cvt_pack_producer():
    """CVTPack stub (Pack* + VCvtPkF32toBF16) lands in CVT_PACK, not ALU."""
    assert shape_of(_stub_cvt_pack()) is InstructionShape.CVT_PACK


def test_shape_classifier_alu_producer():
    """Vanilla VXor (non-Pack category) lands in ALU."""
    assert shape_of(_stub_alu()) is InstructionShape.ALU


def test_shape_classifier_lr_producer():
    """LR-categorized producer (no rocisa_inst) lands in LR."""
    assert shape_of(_stub_lr()) is InstructionShape.LR


# =============================================================================
# Canonical-pair dispatch — same carve-outs the legacy `_DISPATCH` test pinned.
# =============================================================================


def test_canonical_pack_mfma_to_cvt_pack_dispatches_to_5_cycle_settle():
    """4x4 PackMFMA -> CVTPack must hit the 5-quad-cycle settle rule
    (legacy `_mfma_pack_to_cvt_gap_ok` carve-out). Silent regression here
    (e.g. routing to the 1-cycle finish rule) is failure mode #1 in
    QUAD_CYCLE_DISPATCH_AUDIT.md §2.5.
    """
    rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.get(
        (InstructionShape.MFMA_4x4, InstructionShape.CVT_PACK)
    )
    assert rules is not None and len(rules) == 1, (
        "Expected exactly one rule under (MFMA_4x4, CVT_PACK)."
    )
    rule = rules[0]
    assert rule.required_quad_cycles == 5, (
        f"PackMFMA -> CVT_PACK must require 5 quad-cycles "
        f"(per CDNA4 §7.6); got {rule.required_quad_cycles}."
    )


def test_canonical_cvt_pack_to_mfma_dispatches_to_2_cycle_settle():
    """CVTPack -> MFMA must hit the 2-quad-cycle settle rule (legacy
    `_cvt_to_mfma_gap_ok` carve-out). Failure mode #2 in
    QUAD_CYCLE_DISPATCH_AUDIT.md §2.5."""
    for c_shape in (InstructionShape.MFMA_STANDARD, InstructionShape.MFMA_4x4):
        rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.get(
            (InstructionShape.CVT_PACK, c_shape)
        )
        assert rules is not None and len(rules) == 1, (
            f"Expected one rule under (CVT_PACK, {c_shape.name})."
        )
        assert rules[0].required_quad_cycles == 2


def test_canonical_cvt_pack_to_non_mfma_passes_through():
    """CVTPack -> CVT_PACK / MIDDLE_PACK / OTHER must be passthrough
    (legacy `_PASSTHROUGH` rows)."""
    for c_shape in (InstructionShape.CVT_PACK, InstructionShape.MIDDLE_PACK,
                    InstructionShape.OTHER):
        rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.get(
            (InstructionShape.CVT_PACK, c_shape)
        )
        assert rules is not None and len(rules) == 1
        assert rules[0].condition == _GAP_RULE_PASSTHROUGH, (
            f"CVTPack -> {c_shape.name} must be passthrough "
            f"(producer is ALU-immediate-visibility for non-MFMA consumers)."
        )


def test_canonical_alu_producer_passthrough_for_non_mfma_consumers():
    """ALU -> non-MFMA consumers must be passthrough (legacy ALU exemption).
    The two-rule list under (ALU, MFMA_*) is the C-9 carve-out and is
    pinned by `test_alu_to_mfma_two_rule_list_for_c9` below; the rest of
    the ALU rows must remain plain passthroughs."""
    for c_shape in (InstructionShape.CVT_PACK, InstructionShape.MIDDLE_PACK,
                    InstructionShape.LR, InstructionShape.LW,
                    InstructionShape.GR, InstructionShape.OTHER):
        rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.get(
            (InstructionShape.ALU, c_shape)
        )
        assert rules is not None and len(rules) == 1
        assert rules[0].condition == _GAP_RULE_PASSTHROUGH


def test_alu_to_mfma_three_rule_list_for_c9():
    """ALU -> MFMA_* must carry THREE rules in declaration order:
      [0] cross_subiter_alu_artifact passthrough (bwfr resolver carve-out).
      [1] same_subiter 2-quad-cycle gap (audit-memo §2.2 C-9).
      [2] unconditional passthrough fallback (preserves byte-equivalence
          for test fixtures with num_mfma_per_subiter=0; production
          graphs always have a non-zero nmps so this row is only
          reached in test fixtures).

    The list ORDER is load-bearing: cross-subiter must evaluate first so
    the C-9 same-subiter rule never fires on cross-subiter ALU edges,
    and the unconditional passthrough must come LAST so it doesn't
    short-circuit either of the conditional rules in production.
    """
    for c_shape in (InstructionShape.MFMA_STANDARD, InstructionShape.MFMA_4x4):
        rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.get(
            (InstructionShape.ALU, c_shape)
        )
        assert rules is not None
        assert len(rules) == 3, (
            f"(ALU, {c_shape.name}) must carry three rules "
            f"(cross_subiter passthrough, same_subiter 2-cycle gap, "
            f"unconditional passthrough fallback); got {len(rules)}."
        )
        assert rules[0].condition == "cross_subiter_alu_artifact", (
            "First rule must be the cross-subiter passthrough (bwfr); "
            "reordering would let the C-9 same-subiter rule fire on "
            "cross-subiter ALU edges."
        )
        assert rules[1].condition == "same_subiter"
        assert rules[1].required_quad_cycles == 2
        assert rules[2].condition == _GAP_RULE_PASSTHROUGH


def test_pairwise_disjoint_keys():
    """Dict construction enforces unique keys; this test pins the
    invariant explicitly so a future drift to a list-of-tuples shape
    can't introduce silent key duplication."""
    keys = list(_DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.keys())
    assert len(keys) == len(set(keys)), (
        f"_DEFAULT_CDNA4_ARCH_PROFILE.gap_rules has duplicate keys: {keys}"
    )
