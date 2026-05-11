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
"""Per-gap-class tests for `validate_s_delay_alu_coverage`.

Covers RDNA3.5 §16.5 S_DELAY_ALU `INSTID_*` named-gap-class taxonomy
(R-10..R-13 of the s5g1 audit memo). For each ISA-named class the
encoding asserts a producer-back distance; the validator's coverage
predicate is `gap_class.required_back_distance <= actual_back_distance`.
Each non-degenerate class is exercised with a positive case (encoded
distance == actual; validator silent) and a negative case (encoded
distance > actual; validator emits an `SDelayAluCoverageFailure`).

The validator works against `DataflowGraph.s_delay_alu_instances`, a
list of `SDelayAluInstance` records. Today the kernel emitter does not
produce S_DELAY_ALU; `s_delay_alu_instances` is empty and the pass is
dormant. These tests fabricate `SDelayAluInstance` objects directly
and assemble them onto a stub graph (`_make_graph(*instances)`) — no
kernel build, no rocisa fixtures, no LoopBodyCapture machinery needed
because the pass reads instance-shape fields only.
"""

import pytest

from Tensile.Components.CMSValidator import (
    DataflowGraph,
    SDelayAluCoverageFailure,
    SDelayAluInstance,
    validate_s_delay_alu_coverage,
)
from Tensile.Components.InstructionCategory import RdnaSDelayAluClass


def _make_graph(*instances: SDelayAluInstance) -> DataflowGraph:
    """Construct a minimal DataflowGraph carrying only S_DELAY_ALU records.

    The validator pass touches only `graph.s_delay_alu_instances`, so the
    other fields are irrelevant; pass empty stubs to satisfy the dataclass
    constructor.
    """
    return DataflowGraph(
        nodes={},
        edges=[],
        captures={},
        s_delay_alu_instances=list(instances),
    )


# =============================================================================
# Empty graph — pass is dormant.
# =============================================================================

class TestSDelayAluValidatorDormant:
    def test_empty_graph_returns_no_failures(self):
        graph = _make_graph()
        assert validate_s_delay_alu_coverage(graph) == []

    def test_no_rdna35_instructions_returns_no_failures(self):
        # Mirrors the production path on a CDNA4 kernel: no S_DELAY_ALU
        # records emitted, no failures produced.
        graph = DataflowGraph(nodes={}, edges=[], captures={})
        assert validate_s_delay_alu_coverage(graph) == []


# =============================================================================
# NO_DEP — degenerate carrier; required distance 0; trivially passes.
# =============================================================================

class TestNoDepDegenerate:
    def test_NO_DEP_with_zero_actual_passes(self):
        inst = SDelayAluInstance(
            gap_class=RdnaSDelayAluClass.NO_DEP,
            producer_label="(none)",
            consumer_label="v_add_f32",
            actual_back_distance=0,
        )
        assert validate_s_delay_alu_coverage(_make_graph(inst)) == []

    def test_NO_DEP_with_nonzero_actual_passes(self):
        # NO_DEP's required distance is 0; any actual distance is >= 0,
        # so the predicate is trivially satisfied.
        inst = SDelayAluInstance(
            gap_class=RdnaSDelayAluClass.NO_DEP,
            producer_label="v_mul_f32",
            consumer_label="v_add_f32",
            actual_back_distance=5,
        )
        assert validate_s_delay_alu_coverage(_make_graph(inst)) == []


# =============================================================================
# VALU_DEP_N (N=1..4) — INSTID_VALU_DEP_N hazard.
# =============================================================================

@pytest.mark.parametrize("cls,n", [
    (RdnaSDelayAluClass.VALU_DEP_1, 1),
    (RdnaSDelayAluClass.VALU_DEP_2, 2),
    (RdnaSDelayAluClass.VALU_DEP_3, 3),
    (RdnaSDelayAluClass.VALU_DEP_4, 4),
])
class TestValuDep:
    def test_compliant_passes(self, cls, n):
        # encoded == actual: predicate holds.
        inst = SDelayAluInstance(
            gap_class=cls,
            producer_label="v_mul_f32",
            consumer_label="v_add_f32",
            actual_back_distance=n,
        )
        assert validate_s_delay_alu_coverage(_make_graph(inst)) == []

    def test_compliant_overclaim_passes(self, cls, n):
        # encoded < actual: encoding underclaims gap (conservative perf
        # cost, never correctness).
        inst = SDelayAluInstance(
            gap_class=cls,
            producer_label="v_mul_f32",
            consumer_label="v_add_f32",
            actual_back_distance=n + 1,
        )
        assert validate_s_delay_alu_coverage(_make_graph(inst)) == []

    def test_underclaim_emits_failure(self, cls, n):
        # encoded > actual (hard-fail). For VALU_DEP_1 the pathological case
        # is actual=0 (consumer immediately after producer with no slot);
        # for higher N it's actual=N-1.
        inst = SDelayAluInstance(
            gap_class=cls,
            producer_label="v_mul_f32",
            consumer_label="v_add_f32",
            actual_back_distance=n - 1,
        )
        failures = validate_s_delay_alu_coverage(_make_graph(inst))
        assert len(failures) == 1
        f = failures[0]
        assert isinstance(f, SDelayAluCoverageFailure)
        assert f.gap_class is cls
        assert f.encoded_back_distance == n
        assert f.actual_back_distance == n - 1
        assert f.gap_class.family == "VALU"


# =============================================================================
# TRANS32_DEP_N (N=1..3) — INSTID_TRANS32_DEP_N hazard.
# =============================================================================

@pytest.mark.parametrize("cls,n", [
    (RdnaSDelayAluClass.TRANS32_DEP_1, 1),
    (RdnaSDelayAluClass.TRANS32_DEP_2, 2),
    (RdnaSDelayAluClass.TRANS32_DEP_3, 3),
])
class TestTrans32Dep:
    def test_compliant_passes(self, cls, n):
        inst = SDelayAluInstance(
            gap_class=cls,
            producer_label="v_exp_f32",  # transcendental
            consumer_label="v_mul_f32",
            actual_back_distance=n,
        )
        assert validate_s_delay_alu_coverage(_make_graph(inst)) == []

    def test_underclaim_emits_failure(self, cls, n):
        inst = SDelayAluInstance(
            gap_class=cls,
            producer_label="v_exp_f32",
            consumer_label="v_mul_f32",
            actual_back_distance=n - 1,
        )
        failures = validate_s_delay_alu_coverage(_make_graph(inst))
        assert len(failures) == 1
        f = failures[0]
        assert isinstance(f, SDelayAluCoverageFailure)
        assert f.gap_class is cls
        assert f.encoded_back_distance == n
        assert f.actual_back_distance == n - 1
        assert f.gap_class.family == "TRANS32"


# =============================================================================
# SALU_CYCLE_N (N=1..3) — INSTID_SALU_CYCLE_N hazard.
# =============================================================================

@pytest.mark.parametrize("cls,n", [
    (RdnaSDelayAluClass.SALU_CYCLE_1, 1),
    (RdnaSDelayAluClass.SALU_CYCLE_2, 2),
    (RdnaSDelayAluClass.SALU_CYCLE_3, 3),
])
class TestSaluCycle:
    def test_compliant_passes(self, cls, n):
        inst = SDelayAluInstance(
            gap_class=cls,
            producer_label="s_add_u32",
            consumer_label="v_add_f32",
            actual_back_distance=n,
        )
        assert validate_s_delay_alu_coverage(_make_graph(inst)) == []

    def test_underclaim_emits_failure(self, cls, n):
        inst = SDelayAluInstance(
            gap_class=cls,
            producer_label="s_add_u32",
            consumer_label="v_add_f32",
            actual_back_distance=n - 1,
        )
        failures = validate_s_delay_alu_coverage(_make_graph(inst))
        assert len(failures) == 1
        f = failures[0]
        assert isinstance(f, SDelayAluCoverageFailure)
        assert f.gap_class is cls
        assert f.encoded_back_distance == n
        assert f.actual_back_distance == n - 1
        assert f.gap_class.family == "SALU"


# =============================================================================
# FMA_ACCUM_CYCLE_1 — INSTID_FMA_ACCUM_CYCLE_1 reserved.
# =============================================================================

class TestFmaAccumCycle1:
    def test_compliant_passes(self):
        inst = SDelayAluInstance(
            gap_class=RdnaSDelayAluClass.FMA_ACCUM_CYCLE_1,
            producer_label="v_fma_f32",
            consumer_label="v_fma_f32",
            actual_back_distance=1,
        )
        assert validate_s_delay_alu_coverage(_make_graph(inst)) == []

    def test_underclaim_emits_failure(self):
        inst = SDelayAluInstance(
            gap_class=RdnaSDelayAluClass.FMA_ACCUM_CYCLE_1,
            producer_label="v_fma_f32",
            consumer_label="v_fma_f32",
            actual_back_distance=0,
        )
        failures = validate_s_delay_alu_coverage(_make_graph(inst))
        assert len(failures) == 1
        f = failures[0]
        assert isinstance(f, SDelayAluCoverageFailure)
        assert f.gap_class is RdnaSDelayAluClass.FMA_ACCUM_CYCLE_1
        assert f.encoded_back_distance == 1
        assert f.actual_back_distance == 0
        assert f.gap_class.family == "FMA_ACCUM"


# =============================================================================
# Multiple instances — one failure per under-encoded record.
# =============================================================================

class TestMultipleInstances:
    def test_only_underclaiming_records_emit_failures(self):
        ok = SDelayAluInstance(
            gap_class=RdnaSDelayAluClass.VALU_DEP_2,
            producer_label="vP_ok", consumer_label="vC_ok",
            actual_back_distance=2,
        )
        bad = SDelayAluInstance(
            gap_class=RdnaSDelayAluClass.VALU_DEP_3,
            producer_label="vP_bad", consumer_label="vC_bad",
            actual_back_distance=1,
        )
        also_bad = SDelayAluInstance(
            gap_class=RdnaSDelayAluClass.SALU_CYCLE_2,
            producer_label="sP_bad", consumer_label="sC_bad",
            actual_back_distance=0,
        )
        failures = validate_s_delay_alu_coverage(_make_graph(ok, bad, also_bad))
        assert len(failures) == 2
        # Order is deterministic — instance iteration order.
        assert failures[0].producer_label == "vP_bad"
        assert failures[0].gap_class is RdnaSDelayAluClass.VALU_DEP_3
        assert failures[1].producer_label == "sP_bad"
        assert failures[1].gap_class is RdnaSDelayAluClass.SALU_CYCLE_2


# =============================================================================
# Failure message renders cleanly through Failure.format().
# =============================================================================

class TestFailureRendering:
    def test_failure_format_is_human_readable(self):
        inst = SDelayAluInstance(
            gap_class=RdnaSDelayAluClass.VALU_DEP_3,
            producer_label="v_mul_f32_a",
            consumer_label="v_add_f32_b",
            actual_back_distance=1,
        )
        failures = validate_s_delay_alu_coverage(_make_graph(inst))
        assert len(failures) == 1
        text = failures[0].format()
        assert "S_DELAY_ALU coverage violation" in text
        assert "RDNA3.5 §16.5" in text
        assert "VALU_DEP_3" in text
        assert "v_mul_f32_a" in text
        assert "v_add_f32_b" in text
        # Encoded distance + actual distance both surface in the message.
        assert "3" in text
        assert "1" in text


# =============================================================================
# Enum sanity — every §16.5 INSTID_* class is present and round-trips.
# =============================================================================

class TestEnumCoverage:
    """Lock the enum to the §16.5 closed source — a missing or surplus
    member is a regression."""

    def test_member_set_matches_isa_section_16_5(self):
        names = {m.name for m in RdnaSDelayAluClass}
        # Verbatim from RDNA3.5 §16.5 pp. 251-252.
        assert names == {
            "NO_DEP",
            "VALU_DEP_1", "VALU_DEP_2", "VALU_DEP_3", "VALU_DEP_4",
            "TRANS32_DEP_1", "TRANS32_DEP_2", "TRANS32_DEP_3",
            "FMA_ACCUM_CYCLE_1",
            "SALU_CYCLE_1", "SALU_CYCLE_2", "SALU_CYCLE_3",
        }

    def test_encoded_values_match_isa_section_16_5(self):
        # 4-bit encoded values per §16.5 ("INSTID0 = SIMM16[3:0]").
        expected = {
            "NO_DEP":            0x0,
            "VALU_DEP_1":        0x1,
            "VALU_DEP_2":        0x2,
            "VALU_DEP_3":        0x3,
            "VALU_DEP_4":        0x4,
            "TRANS32_DEP_1":     0x5,
            "TRANS32_DEP_2":     0x6,
            "TRANS32_DEP_3":     0x7,
            "FMA_ACCUM_CYCLE_1": 0x8,
            "SALU_CYCLE_1":      0x9,
            "SALU_CYCLE_2":      0xa,
            "SALU_CYCLE_3":      0xb,
        }
        for m in RdnaSDelayAluClass:
            assert m.value == expected[m.name]
