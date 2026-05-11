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
"""Per-rule tests for `validate_vopd_pair_formation`.

Covers RDNA3.5 §7.6 R-4 through R-7 — the pair-formation hard rules.
The ISA is explicit ("These are hard rules — the instruction does not
function if these rules are broken"), so each rule is exercised with a
positive (compliant pair, validator silent) and negative (violating
pair, validator emits a `VopdPairFormationFailure` with the right rule
tag) test.

The validator works against `DataflowGraph.vopd_pairs`, a list of
`VopdPair` records. Today the kernel emitter does not produce VOPD;
`vopd_pairs` is empty and the pass is dormant. These tests fabricate
`VopdPair` instances directly and assemble them onto a stub graph
(`_make_graph(*pairs)`) — no kernel build, no rocisa fixtures, no
LoopBodyCapture machinery needed because the pass reads pair-shape
fields only.
"""

import pytest

from Tensile.Components.CMSValidator import (
    DataflowGraph,
    VopdPair,
    VopdPairFormationFailure,
    validate_vopd_pair_formation,
)


def _make_graph(*pairs: VopdPair) -> DataflowGraph:
    """Construct a minimal DataflowGraph carrying only VOPD pairs.

    The validator pass touches only `graph.vopd_pairs` so the other
    fields are irrelevant; pass empty stubs to satisfy the dataclass
    constructor.
    """
    return DataflowGraph(
        nodes={},
        edges=[],
        captures={},
        vopd_pairs=list(pairs),
    )


def _failures_for_rule(failures, rule):
    return [f for f in failures if f.rule == rule]


# =============================================================================
# Empty graph — pass is dormant.
# =============================================================================

class TestVopdValidatorDormant:
    def test_empty_graph_returns_no_failures(self):
        graph = _make_graph()
        assert validate_vopd_pair_formation(graph) == []


# =============================================================================
# R-4 — SRCX0/SRCY0 different banks AND VSRCX1/VSRCY1 different banks.
# =============================================================================

class TestRuleR4SrcBank:
    def test_R4_compliant_pair_passes(self):
        # src0_a=v4 (bank 0), src0_b=v5 (bank 1) — different banks.
        # src1_a=v8 (bank 0), src1_b=v9 (bank 1) — different banks.
        # vdst even/odd, no SRC2, independent.
        pair = VopdPair(
            instruction_a="v_mul_f32_x",
            instruction_b="v_add_f32_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=9,
            vdst_a=12, vdst_b=15,
        )
        assert _failures_for_rule(
            validate_vopd_pair_formation(_make_graph(pair)), "R-4"
        ) == []

    def test_R4_violating_src0_same_bank_emits_failure(self):
        # src0_a=v4 (bank 0), src0_b=v8 (bank 0) — SAME bank => R-4.
        pair = VopdPair(
            instruction_a="v_mul_f32_x",
            instruction_b="v_add_f32_y",
            src0_a=4, src1_a=8,
            src0_b=8, src1_b=9,
            vdst_a=12, vdst_b=15,
        )
        failures = validate_vopd_pair_formation(_make_graph(pair))
        r4 = _failures_for_rule(failures, "R-4")
        assert len(r4) >= 1
        f = r4[0]
        assert isinstance(f, VopdPairFormationFailure)
        assert f.rule == "R-4"
        assert f.instruction_a == "v_mul_f32_x"
        assert f.instruction_b == "v_add_f32_y"
        assert "SRCX0" in f.why and "SRCY0" in f.why


# =============================================================================
# R-5 — destination VGPR parity (one even, one odd).
# =============================================================================

class TestRuleR5DstParity:
    def test_R5_compliant_pair_passes(self):
        # vdst_a=12 (even), vdst_b=15 (odd) — opposite parity.
        pair = VopdPair(
            instruction_a="v_mul_f32_x",
            instruction_b="v_add_f32_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=9,
            vdst_a=12, vdst_b=15,
        )
        assert _failures_for_rule(
            validate_vopd_pair_formation(_make_graph(pair)), "R-5"
        ) == []

    def test_R5_same_parity_emits_failure(self):
        # vdst_a=12 (even), vdst_b=14 (even) — SAME parity => R-5.
        pair = VopdPair(
            instruction_a="v_mul_f32_x",
            instruction_b="v_add_f32_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=9,
            vdst_a=12, vdst_b=14,
        )
        failures = validate_vopd_pair_formation(_make_graph(pair))
        r5 = _failures_for_rule(failures, "R-5")
        assert len(r5) == 1
        f = r5[0]
        assert isinstance(f, VopdPairFormationFailure)
        assert f.rule == "R-5"
        assert "vdstX" in f.why and "vdstY" in f.why
        assert "parity" in f.why


# =============================================================================
# R-6 — SRC2 even/odd (only when both ops use SRC2).
# =============================================================================

class TestRuleR6Src2Parity:
    def test_R6_compliant_pair_passes(self):
        # Both ops use SRC2: src2_a=20 (even), src2_b=23 (odd).
        # Independent vdsts well outside any source operand set.
        pair = VopdPair(
            instruction_a="v_fmamk_f32_x",
            instruction_b="v_dot2acc_f32_f16_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=9,
            vdst_a=30, vdst_b=33,
            src2_a=20, src2_b=23,
        )
        assert _failures_for_rule(
            validate_vopd_pair_formation(_make_graph(pair)), "R-6"
        ) == []

    def test_R6_same_parity_emits_failure(self):
        # Both use SRC2; src2_a=20 (even), src2_b=22 (even) — SAME parity.
        pair = VopdPair(
            instruction_a="v_fmamk_f32_x",
            instruction_b="v_dot2acc_f32_f16_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=9,
            vdst_a=30, vdst_b=33,
            src2_a=20, src2_b=22,
        )
        failures = validate_vopd_pair_formation(_make_graph(pair))
        r6 = _failures_for_rule(failures, "R-6")
        assert len(r6) == 1
        f = r6[0]
        assert isinstance(f, VopdPairFormationFailure)
        assert f.rule == "R-6"
        assert "SRC2X" in f.why and "SRC2Y" in f.why

    def test_R6_silent_when_only_one_op_uses_src2(self):
        # Only X uses SRC2 (src2_a >= 0); Y has src2_b = -1.
        # R-6 must NOT fire — the rule is gated on both consuming SRC2.
        pair = VopdPair(
            instruction_a="v_fmac_f32_x",
            instruction_b="v_mov_b32_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=9,
            vdst_a=30, vdst_b=33,
            src2_a=20, src2_b=-1,
        )
        assert _failures_for_rule(
            validate_vopd_pair_formation(_make_graph(pair)), "R-6"
        ) == []


# =============================================================================
# R-7 — independence (no RAW between X and Y, no WAW).
# =============================================================================

class TestRuleR7Independence:
    def test_R7_compliant_pair_passes(self):
        # Two ops with disjoint source/destination sets.
        pair = VopdPair(
            instruction_a="v_mul_f32_x",
            instruction_b="v_add_f32_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=9,
            vdst_a=30, vdst_b=33,
        )
        assert _failures_for_rule(
            validate_vopd_pair_formation(_make_graph(pair)), "R-7"
        ) == []

    def test_R7_RAW_X_to_Y_emits_failure(self):
        # X writes v30; Y reads v30 via src1 — RAW X→Y dependency.
        pair = VopdPair(
            instruction_a="v_mul_f32_x",
            instruction_b="v_add_f32_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=30,   # Y reads X's destination
            vdst_a=30, vdst_b=33,
        )
        failures = validate_vopd_pair_formation(_make_graph(pair))
        r7 = _failures_for_rule(failures, "R-7")
        assert len(r7) == 1
        f = r7[0]
        assert isinstance(f, VopdPairFormationFailure)
        assert f.rule == "R-7"
        assert "RAW X" in f.why


# =============================================================================
# Multi-rule violation — single pair triggers more than one R-x failure.
# =============================================================================

class TestMultipleRuleViolations:
    def test_pair_violating_R5_and_R7_emits_both(self):
        # Same vdst (v12) on X and Y: R-5 (same parity, even+even) AND
        # R-7 (WAW). Confirms each rule is independently checked.
        pair = VopdPair(
            instruction_a="v_mul_f32_x",
            instruction_b="v_add_f32_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=9,
            vdst_a=12, vdst_b=12,
        )
        failures = validate_vopd_pair_formation(_make_graph(pair))
        rules = sorted({f.rule for f in failures})
        assert "R-5" in rules
        assert "R-7" in rules


# =============================================================================
# Failure message renders cleanly through Failure.format().
# =============================================================================

class TestFailureRendering:
    def test_failure_format_is_human_readable(self):
        pair = VopdPair(
            instruction_a="v_mul_f32_x",
            instruction_b="v_add_f32_y",
            src0_a=4, src1_a=8,
            src0_b=5, src1_b=9,
            vdst_a=12, vdst_b=14,  # R-5
        )
        failures = validate_vopd_pair_formation(_make_graph(pair))
        r5 = _failures_for_rule(failures, "R-5")
        text = r5[0].format()
        assert "VOPD pair-formation violation" in text
        assert "R-5" in text
        assert "RDNA3.5 §7.6" in text
        assert "v_mul_f32_x" in text
        assert "v_add_f32_y" in text
