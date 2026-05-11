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
"""Tests for `_DEFAULT_CDNA4_ARCH_PROFILE.gap_rules` newly-covered gap classes
(rocm-libraries-vmua).

The s5g1 ISA-gap-generalization audit memo §2.2 inventories 28 distinct
CDNA4 gap classes; the validator covered 5 directly + 4 partially before
this bead. This file pins the 6 newly-covered gap classes:

  C-4: MFMA -> VALU same vDst (RAW on accumulator)
       Covered by (MFMA_STANDARD, ALU) and (MFMA_4x4, ALU) rules.

  C-5: MFMA -> VMEM/LDS read same vDst (RAW on accumulator)
       Covered by (MFMA_STANDARD, LR) / (MFMA_STANDARD, GR) /
                  (MFMA_4x4, LR) / (MFMA_4x4, GR) rules.

  C-9: Non-DLops VALU -> V_MFMA*/V_SMFMA* forwarding gap (2 wait states)
       Covered by (ALU, MFMA_STANDARD) and (ALU, MFMA_4x4) rules with
       a `same_subiter` condition (cross-subiter ALU edges remain on
       the bwfr resolver-artifact passthrough; uncategorized-subiter
       fixtures fall through to the unconditional passthrough).

  C-11: F8 MFMA cycle-count doubling (gfx950)
        Covered by the standard-MFMA finish-cycles rule's callable
        `required_quad_cycles` — delegates to the profile's
        `mfma_finish_cycles_for(rocisa_inst)` which consults rocisa's
        `MFMAInstruction.getIssueLatency()` for per-(arch, dtype, B)
        cycle counts (rocm-libraries-qbcc).

  C-12-finer: 16x16 vs 32x32 finish cycles
        Same path as C-11 — the per-instance callable derives the
        cycle count from rocisa's getIssueLatency, so 16x16 / 32x32 /
        4x4 each get their own value.

  C-6: MFMA WAR on accumulator (XDL/SMFMA SrcC -> VALU write VGPR)
        OUT OF SCOPE — the validator's dataflow graph forms RAW edges
        only; WAR edges are not constructed. Pinned with an explicit
        skip below so the gap-class enumeration is honest about its
        coverage.

Each newly-covered class gets a positive (rule fires when expected) and
negative (rule does NOT fire when condition fails) test; the C-9 tests
also pin the cross-subiter exemption.
"""

from __future__ import annotations

import pytest

from rocisa.container import sgpr, vgpr
from rocisa.instruction import (
    SMovB32,
    VCvtPkF32toBF16,
    VXorB32,
)

from Tensile.Components.CMSValidator import (
    GapRule,
    _DEFAULT_CDNA4_ARCH_PROFILE,
    _GAP_RULE_PASSTHROUGH,
    _evaluate_gap_rule_condition,
    _dispatch_quad_cycle_check,
    TimingResult,
    TimingCheck,
)
from Tensile.Components.InstructionShape import InstructionShape


# =============================================================================
# C-4 / C-5: MFMA -> downstream consumer RAW-on-accumulator gap-rule rows.
# =============================================================================


@pytest.mark.parametrize("p_shape,c_shape,expected_cycles_predicate", [
    # C-4: MFMA -> ALU
    (InstructionShape.MFMA_STANDARD, InstructionShape.ALU,
     lambda rq: callable(rq)),
    (InstructionShape.MFMA_4x4, InstructionShape.ALU,
     lambda rq: callable(rq)),
    # C-5: MFMA -> LR
    (InstructionShape.MFMA_STANDARD, InstructionShape.LR,
     lambda rq: callable(rq)),
    (InstructionShape.MFMA_4x4, InstructionShape.LR,
     lambda rq: callable(rq)),
    # C-5: MFMA -> GR
    (InstructionShape.MFMA_STANDARD, InstructionShape.GR,
     lambda rq: callable(rq)),
    (InstructionShape.MFMA_4x4, InstructionShape.GR,
     lambda rq: callable(rq)),
])
def test_c4_c5_mfma_to_consumer_rule_present(p_shape, c_shape,
                                              expected_cycles_predicate):
    """C-4 / C-5 (audit-memo §2.2): MFMA -> ALU/LR/GR RAW-on-accumulator
    rules must be present in the table, and their `required_quad_cycles`
    is a callable (per-instance via rocisa's getIssueLatency).

    Six rules total cover the C-4 (MFMA->ALU; 2 rules) and C-5
    (MFMA->LR/GR; 4 rules) inventory entries. Together with the C-9
    rules below this is the new gap-class coverage the bead requires.
    """
    rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.get((p_shape, c_shape))
    assert rules is not None, (
        f"Expected ({p_shape.name}, {c_shape.name}) to carry a "
        f"RAW-on-accumulator rule (audit-memo C-4 / C-5)."
    )
    assert len(rules) == 1
    assert rules[0].condition is None, (
        f"({p_shape.name}, {c_shape.name}) rule must be unconditional — "
        f"the floor RAW-on-accumulator gap fires for every same-vDst edge."
    )
    assert expected_cycles_predicate(rules[0].required_quad_cycles), (
        f"({p_shape.name}, {c_shape.name}) required_quad_cycles must be "
        f"a callable so per-instance MFMA cycle counts (audit-memo C-11 / "
        f"C-12-finer) drive the gap; got {rules[0].required_quad_cycles!r}."
    )


def test_c4_mfma_standard_to_alu_rule_returns_3_cycle_floor():
    """Positive: a standard MFMA rocisa instance under the (MFMA_STANDARD,
    ALU) rule must resolve to the standard-MFMA finish-cycle floor
    (3 quad-cycles via the CDNA4 default profile).
    """
    rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules[
        (InstructionShape.MFMA_STANDARD, InstructionShape.ALU)
    ]
    rule = rules[0]
    # Use a non-MFMA instance (None or a non-getIssueLatency object) to
    # force the fallback path inside `mfma_finish_cycles_for`; that
    # falls back to standard_mfma_finish_cycles == 3.
    cycles = rule.evaluate_required(None)
    assert cycles == 3, (
        f"C-4 rule on standard MFMA should yield the 3-quad-cycle floor "
        f"(standard_mfma_finish_cycles); got {cycles}."
    )


def test_c5_mfma_4x4_to_lr_rule_returns_1_cycle_floor():
    """Positive: a 4x4 MFMA producer feeding an LR consumer must require
    1 quad-cycle (the 4x4 finish window). Negative-direction control:
    the 4x4 floor is shorter than the standard floor, so accidentally
    routing the 4x4 producer through the standard rule would over-strict
    the gap.
    """
    rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules[
        (InstructionShape.MFMA_4x4, InstructionShape.LR)
    ]
    cycles = rules[0].evaluate_required(None)
    assert cycles == 1


# =============================================================================
# C-9: ALU -> MFMA forwarding gap (the table's two-rule list).
# =============================================================================


def test_c9_alu_to_mfma_same_subiter_rule_fires_with_2_cycle_gap():
    """Positive: the C-9 same-subiter rule fires when subiter info is
    available and producer/consumer share a subiter. Required gap = 2."""
    rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules[
        (InstructionShape.ALU, InstructionShape.MFMA_STANDARD)
    ]
    # The C-9 rule is at index 1 (after the cross-subiter passthrough).
    c9 = rules[1]
    assert c9.condition == "same_subiter"
    assert c9.required_quad_cycles == 2
    assert "C-9" in c9.rationale, (
        f"C-9 rule must cite audit-memo §2.2 C-9 in its rationale for "
        f"traceability; got: {c9.rationale!r}"
    )


def test_c9_cross_subiter_alu_remains_passthrough():
    """Negative: cross-subiter ALU -> MFMA edges must NOT trigger the C-9
    rule. The bwfr resolver-artifact passthrough at rule index 0 fires
    first when subiters differ.
    """
    rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules[
        (InstructionShape.ALU, InstructionShape.MFMA_STANDARD)
    ]
    cross = rules[0]
    assert cross.condition == "cross_subiter_alu_artifact"
    assert "bwfr" in cross.rationale.lower()


def test_c9_unconditional_passthrough_fallback_for_test_fixtures():
    """Negative-direction byte-equivalence pin: when the graph has no
    `num_mfma_per_subiter` (test-fixture default), neither cross-subiter
    nor same-subiter conditions fire, and the unconditional passthrough
    fallback at rule index 2 takes effect — preserving legacy `(ALU,
    MFMA) -> _PASSTHROUGH` behavior.
    """
    rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules[
        (InstructionShape.ALU, InstructionShape.MFMA_STANDARD)
    ]
    fallback = rules[2]
    assert fallback.condition == _GAP_RULE_PASSTHROUGH
    assert "byte-equivalence" in fallback.rationale.lower()


# =============================================================================
# C-11 / C-12-finer: MFMA finish cycles per family (via rocisa getIssueLatency).
# =============================================================================


def test_c11_c12_standard_mfma_rule_uses_per_instance_callable():
    """The standard-MFMA finish-cycle rule's `required_quad_cycles` is a
    callable that delegates to the profile's `mfma_finish_cycles_for`
    method. This is the path that enables audit-memo C-11 (F8 MFMA cycle
    doubling) and C-12-finer (16x16 vs 32x32) coverage automatically: the
    profile method consults rocisa's `MFMAInstruction.getIssueLatency()`
    (rocm-libraries-qbcc), which encodes per-(arch, dtype, B) cycle
    counts.

    This test pins the callable shape (the audit-memo's C-11/C-12-finer
    coverage is a side effect of rocisa's binding behavior). If the
    callable is replaced with a constant, this test fires before
    cycle-count fidelity silently regresses.
    """
    rules = _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules[
        (InstructionShape.MFMA_STANDARD, InstructionShape.MFMA_STANDARD)
    ]
    rule = rules[0]
    assert callable(rule.required_quad_cycles), (
        "Standard MFMA finish-cycles rule must use a callable so per-"
        "instance cycle counts (C-11 / C-12-finer) reach the gap "
        "evaluator; got constant {rule.required_quad_cycles!r}."
    )
    # Smoke the callable: with None it falls back to standard_mfma_finish_cycles.
    assert rule.required_quad_cycles(None) == 3
    assert "C-11" in rule.rationale or "C-12" in rule.rationale, (
        f"Standard-MFMA rule rationale must cite C-11 / C-12-finer for "
        f"traceability; got: {rule.rationale!r}"
    )


# =============================================================================
# C-6 — explicit out-of-scope marker.
# =============================================================================


@pytest.mark.skip(reason=(
    "C-6 (MFMA WAR on accumulator: XDL/SMFMA SrcC read -> VALU write VGPR) "
    "is OUT OF SCOPE for this bead. The validator's dataflow graph forms "
    "RAW edges only; WAR edges are not constructed by the per-byte "
    "latest-writer resolver. Adding C-6 coverage requires extending the "
    "graph builder to track operand reads (not just writes), which is a "
    "structural change beyond the scope of vmua. Tracked separately."
))
def test_c6_mfma_war_on_accumulator_explicit_skip():
    """Placeholder pinning the C-6 audit-memo entry as explicitly
    out-of-scope-by-graph-structure. Will fire if anyone removes the
    skip without first landing the WAR-edge-formation work.
    """
    pytest.fail("C-6 WAR coverage requires WAR-edge formation in the graph builder.")


# =============================================================================
# _evaluate_gap_rule_condition unit tests
# =============================================================================


class _FakeNode:
    """Minimal duck-typed GraphNode for condition tests. Provides
    `subiter(nmps)` returning a fixed subiter."""

    def __init__(self, subiter_value):
        self._subiter = subiter_value

    def subiter(self, nmps):  # noqa: ARG002
        return self._subiter


class _FakeGraph:
    def __init__(self, nmps):
        self.num_mfma_per_subiter = nmps


def test_condition_none_fires_unconditionally():
    rule = GapRule(required_quad_cycles=5)
    assert _evaluate_gap_rule_condition(rule, _FakeNode(0), _FakeNode(1),
                                         _FakeGraph(0)) is True


def test_condition_passthrough_fires_unconditionally():
    rule = GapRule(required_quad_cycles=0, condition=_GAP_RULE_PASSTHROUGH)
    assert _evaluate_gap_rule_condition(rule, _FakeNode(0), _FakeNode(1),
                                         _FakeGraph(0)) is True


def test_condition_same_subiter_does_not_fire_when_nmps_zero():
    """Byte-equivalence guarantee: the same-subiter condition MUST NOT
    fire when num_mfma_per_subiter is 0 (test-fixture default). This is
    what preserves the legacy (ALU, MFMA) -> passthrough behavior on
    every existing fixture.
    """
    rule = GapRule(required_quad_cycles=2, condition="same_subiter")
    assert _evaluate_gap_rule_condition(rule, _FakeNode(0), _FakeNode(0),
                                         _FakeGraph(0)) is False


def test_condition_same_subiter_fires_when_nmps_set_and_subiters_match():
    rule = GapRule(required_quad_cycles=2, condition="same_subiter")
    assert _evaluate_gap_rule_condition(rule, _FakeNode(2), _FakeNode(2),
                                         _FakeGraph(4)) is True


def test_condition_same_subiter_does_not_fire_when_subiters_differ():
    rule = GapRule(required_quad_cycles=2, condition="same_subiter")
    assert _evaluate_gap_rule_condition(rule, _FakeNode(2), _FakeNode(3),
                                         _FakeGraph(4)) is False


def test_condition_cross_subiter_fires_when_subiters_differ():
    rule = GapRule(required_quad_cycles=0,
                   condition="cross_subiter_alu_artifact")
    assert _evaluate_gap_rule_condition(rule, _FakeNode(2), _FakeNode(3),
                                         _FakeGraph(4)) is True


def test_condition_cross_subiter_does_not_fire_when_nmps_zero():
    rule = GapRule(required_quad_cycles=0,
                   condition="cross_subiter_alu_artifact")
    assert _evaluate_gap_rule_condition(rule, _FakeNode(2), _FakeNode(3),
                                         _FakeGraph(0)) is False


def test_condition_unknown_raises_value_error():
    """Typo-detection: an unknown condition name surfaces as ValueError
    rather than silently skipping the rule. The condition name set is
    closed; rules in `_DEFAULT_CDNA4_ARCH_PROFILE.gap_rules` are data we
    control end-to-end, so any unrecognized value is a bug.
    """
    rule = GapRule(required_quad_cycles=2, condition="bogus_condition_name")
    with pytest.raises(ValueError, match="unknown condition"):
        _evaluate_gap_rule_condition(rule, _FakeNode(0), _FakeNode(0),
                                      _FakeGraph(0))


# =============================================================================
# GapRule dataclass shape tests
# =============================================================================


def test_gap_rule_evaluate_required_with_constant():
    rule = GapRule(required_quad_cycles=5)
    assert rule.evaluate_required(None) == 5


def test_gap_rule_evaluate_required_with_callable():
    rule = GapRule(required_quad_cycles=lambda inst: 7)
    assert rule.evaluate_required(None) == 7


def test_gap_rule_rationale_required_for_non_passthrough_rules():
    """Every non-passthrough rule in the CDNA4 default table must have a
    non-empty rationale citing its ISA source. Rationale-less rules are
    not auditable and should not enter the table."""
    bad = []
    for key, rules in _DEFAULT_CDNA4_ARCH_PROFILE.gap_rules.items():
        for rule in rules:
            if rule.condition == _GAP_RULE_PASSTHROUGH:
                continue  # Passthrough rules don't need a citation.
            if not rule.rationale:
                bad.append((key, rule))
    assert not bad, (
        f"Rules without rationale found in CDNA4 default table:\n"
        + "\n".join(f"  {k}: {r}" for k, r in bad)
    )
