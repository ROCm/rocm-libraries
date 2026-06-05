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
"""Unit tests for UnrolledCapture / UnrolledIterRecord materialization.

Verifies:
- Record count and order for all body combinations (PRO / ML / NGL / NLL).
- Correct iter_index assignment on ML iter copies.
- unrolled_start monotonicity and total_instructions accounting.
- ML iter copies share the same list object and TaggedInstruction objects.
- Identity iter-blindness contract: same TaggedInstruction object across copies
  gives identical identity_for() tuple.
- ML_MAT_COUNT drives the ML copy count and is importable from ScheduleCapture.
- ValueError raised when main_loop has no codepath 0.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_PROLOGUE,
    BODY_LABEL_ML,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    ML_MAT_COUNT,
    FourPartCapture,
    LoopBodyCapture,
    LoopBodyCaptureBuilder,
    UnrolledCapture,
    UnrolledIterRecord,
    assign_emission_ordinals,
    WrappedInstruction,
)


# =============================================================================
# Fixture helpers
# =============================================================================


def _opaque_inst():
    """Return a fresh SNop — cheapest rocisa instruction with no dataflow."""
    from rocisa.instruction import SNop
    return SNop(waitState=0)


def _make_body(n: int) -> LoopBodyCapture:
    """Build a LoopBodyCapture containing `n` SNop-tagged instructions.

    Each instruction occupies its own (MFMA, mfma_index=i) slot so the slot
    lex sort is unambiguous. SNop is used because it passes finalize() guards
    (no SMEM / flat / store) and has no dataflow, making it safe for tests
    that don't need real register dataflow.
    """
    b = LoopBodyCaptureBuilder()
    for i in range(n):
        b.append(inst=_opaque_inst(), category="SNOP", subiter=0, mfma_index=i)
    return b.finalize()


def _make_fpc(*, pro_n=None, ml_n=3, ngl_n=None, nll_n=None) -> FourPartCapture:
    """Build a minimal FourPartCapture for UnrolledCapture tests.

    `pro_n`: number of instructions in PRO body; None means prologue=None.
    `ml_n`: number of instructions in ML body (always present, codepath 0).
    `ngl_n`: number of instructions in NGL body; None means n_gl={}.
    `nll_n`: number of instructions in NLL body; None means n_ll={}.
    """
    ml_body = _make_body(ml_n)
    return FourPartCapture(
        main_loop={0: ml_body},
        main_loop_prev={0: _make_body(ml_n)},
        n_gl={0: _make_body(ngl_n)} if ngl_n is not None else {},
        n_ll={0: _make_body(nll_n)} if nll_n is not None else {},
        num_mfma=0,
        num_codepaths=1,
        source="test-fixture",
        prologue=_make_body(pro_n) if pro_n is not None else None,
    )


# =============================================================================
# TestUnrolledCaptureBasicShape
# =============================================================================

class TestUnrolledCaptureBasicShape:
    """Record count and order for all body combinations."""

    def test_01_record_count_all_present(self):
        """PRO + ML + NGL + NLL: 1 + ML_MAT_COUNT + 1 + 1 records."""
        fpc = _make_fpc(pro_n=2, ml_n=3, ngl_n=2, nll_n=2)
        uc = UnrolledCapture.from_four_part_capture(fpc)
        assert len(uc.records) == 1 + ML_MAT_COUNT + 1 + 1

    def test_02_body_label_order(self):
        """Record labels follow PRO, ML, ML, NGL, NLL order."""
        fpc = _make_fpc(pro_n=1, ml_n=2, ngl_n=1, nll_n=1)
        uc = UnrolledCapture.from_four_part_capture(fpc)
        assert [r.body_label for r in uc.records] == [
            BODY_LABEL_PROLOGUE,
            BODY_LABEL_ML,
            BODY_LABEL_ML,
            BODY_LABEL_NGL,
            BODY_LABEL_NLL,
        ]

    def test_03_ml_iter_index_values(self):
        """ML records have iter_index 0 and 1; non-ML records have iter_index 0."""
        fpc = _make_fpc(pro_n=1, ml_n=2, ngl_n=1, nll_n=1)
        uc = UnrolledCapture.from_four_part_capture(fpc)
        ml_records = [r for r in uc.records if r.body_label == BODY_LABEL_ML]
        assert [r.iter_index for r in ml_records] == list(range(ML_MAT_COUNT))
        non_ml = [r for r in uc.records if r.body_label != BODY_LABEL_ML]
        assert all(r.iter_index == 0 for r in non_ml)

    def test_04_pro_absent_when_prologue_none(self):
        """When prologue=None, no PRO record appears and count is ML_MAT_COUNT+2."""
        fpc = _make_fpc(pro_n=None, ml_n=2, ngl_n=1, nll_n=1)
        uc = UnrolledCapture.from_four_part_capture(fpc)
        assert len(uc.records) == ML_MAT_COUNT + 1 + 1
        assert BODY_LABEL_PROLOGUE not in [r.body_label for r in uc.records]

    def test_05_ngl_absent_when_n_gl_empty(self):
        """When n_gl={}, no NGL record appears."""
        fpc = _make_fpc(pro_n=1, ml_n=2, ngl_n=None, nll_n=1)
        uc = UnrolledCapture.from_four_part_capture(fpc)
        assert BODY_LABEL_NGL not in [r.body_label for r in uc.records]

    def test_06_nll_absent_when_n_ll_empty(self):
        """When n_ll={}, no NLL record appears."""
        fpc = _make_fpc(pro_n=1, ml_n=2, ngl_n=1, nll_n=None)
        uc = UnrolledCapture.from_four_part_capture(fpc)
        assert BODY_LABEL_NLL not in [r.body_label for r in uc.records]


# =============================================================================
# TestUnrolledCapturePositionMonotonicity
# =============================================================================

class TestUnrolledCapturePositionMonotonicity:
    """unrolled_start values and total_instructions are consistent."""

    def _make_fpc_with_counts(self):
        """PRO=3, ML=4, NGL=2, NLL=5."""
        return _make_fpc(pro_n=3, ml_n=4, ngl_n=2, nll_n=5)

    def test_07_unrolled_start_values(self):
        """unrolled_start of each record follows cursor arithmetic."""
        fpc = self._make_fpc_with_counts()
        uc = UnrolledCapture.from_four_part_capture(fpc)
        # PRO=3, ML_iter0 starts at 3, ML_iter1 starts at 3+4=7,
        # NGL starts at 7+4=11, NLL starts at 11+2=13
        assert uc.records[0].unrolled_start == 0   # PRO
        assert uc.records[1].unrolled_start == 3   # ML_iter[0]
        assert uc.records[2].unrolled_start == 7   # ML_iter[1]
        assert uc.records[3].unrolled_start == 11  # NGL
        assert uc.records[4].unrolled_start == 13  # NLL

    def test_08_total_instructions(self):
        """total_instructions == sum of per-record instruction counts."""
        fpc = self._make_fpc_with_counts()
        uc = UnrolledCapture.from_four_part_capture(fpc)
        # PRO=3, ML_iter0=4, ML_iter1=4, NGL=2, NLL=5 => 3+4+4+2+5=18
        assert uc.total_instructions == 3 + 4 + 4 + 2 + 5

    def test_09_derived_positions_strictly_monotonic(self):
        """Derived per-instruction positions (unrolled_start + local_idx) are
        strictly increasing with no gaps across all records."""
        fpc = self._make_fpc_with_counts()
        uc = UnrolledCapture.from_four_part_capture(fpc)
        positions = []
        for r in uc.records:
            for local_idx in range(len(r.instructions)):
                positions.append(r.unrolled_start + local_idx)
        # Strictly increasing (consecutive integers starting at 0)
        assert positions == list(range(len(positions)))


# =============================================================================
# TestUnrolledCaptureMLSharing
# =============================================================================

class TestUnrolledCaptureMLSharing:
    """ML iter copies share the same list and TaggedInstruction objects."""

    def test_10_ml_iter_copies_share_instructions_list_identity(self):
        """The `instructions` list object is the SAME Python object for both ML copies."""
        fpc = _make_fpc(pro_n=1, ml_n=3, ngl_n=1, nll_n=1)
        uc = UnrolledCapture.from_four_part_capture(fpc)
        ml_records = [r for r in uc.records if r.body_label == BODY_LABEL_ML]
        assert len(ml_records) == ML_MAT_COUNT
        ml0, ml1 = ml_records[0], ml_records[1]
        # The list is shared — same Python object identity
        assert ml0.instructions is ml1.instructions

    def test_11_tagged_instruction_objects_are_same(self):
        """Each TaggedInstruction in ML_iter[0] is the same object as in ML_iter[1]."""
        fpc = _make_fpc(pro_n=1, ml_n=3, ngl_n=1, nll_n=1)
        uc = UnrolledCapture.from_four_part_capture(fpc)
        ml_records = [r for r in uc.records if r.body_label == BODY_LABEL_ML]
        ml0, ml1 = ml_records[0], ml_records[1]
        assert all(ti0 is ti1 for ti0, ti1 in zip(ml0.instructions, ml1.instructions))


# =============================================================================
# TestUnrolledCaptureIdentityIterBlindness
# =============================================================================

class TestUnrolledCaptureIdentityIterBlindness:
    """Identity is identical across ML iter copies; iter_index not in the tuple."""

    def _make_fpc_with_mfma(self):
        """FourPartCapture with a real MFMA in ML so emission_ordinal is assigned."""
        from rocisa.instruction import MFMAInstruction
        from rocisa.container import vgpr
        from rocisa.enum import InstType
        b = LoopBodyCaptureBuilder()
        acc = vgpr(0, 16)
        inst = MFMAInstruction(
            instType=InstType.INST_F32, accType=InstType.INST_F32,
            variant=[32, 32, 0, 1], mfma1k=False,
            acc=acc, a=vgpr(64, 2), b=vgpr(72, 2), acc2=acc,
        )
        b.append(inst=inst, category="MFMA", subiter=0, mfma_index=0)
        ml_body = b.finalize()
        return FourPartCapture(
            main_loop={0: ml_body},
            main_loop_prev={0: _make_body(1)},
            n_gl={},
            n_ll={},
            num_mfma=1,
            num_codepaths=1,
            source="test-fixture",
            prologue=None,
        )

    def test_12_identity_identical_across_ml_iter_copies(self):
        """ML_iter[0] and ML_iter[1] reference the same TaggedInstruction,
        so identity_for() returns the same tuple trivially."""
        fpc = self._make_fpc_with_mfma()
        uc = UnrolledCapture.from_four_part_capture(fpc)
        ml_records = [r for r in uc.records if r.body_label == BODY_LABEL_ML]
        ml0, ml1 = ml_records[0], ml_records[1]
        ti = ml0.instructions[0]
        ti2 = ml1.instructions[0]
        # Same object
        assert ti is ti2
        # Same identity
        assert ti.identity_for(BODY_LABEL_ML) == ti2.identity_for(BODY_LABEL_ML)

    def test_13_identity_is_3_tuple_without_iter_index(self):
        """identity_for() returns (canonical_render, source_module_id, emission_ordinal).

        Confirm the 3-tuple shape: no iter_index, no unrolled_position.
        The plan §2B CORRECTION notes the docstring says 2-tuple but the actual
        return at lines 549-551 of ScheduleCapture.py is a 3-tuple.
        """
        fpc = self._make_fpc_with_mfma()
        uc = UnrolledCapture.from_four_part_capture(fpc)
        ml_records = [r for r in uc.records if r.body_label == BODY_LABEL_ML]
        ti = ml_records[0].instructions[0]
        identity = ti.identity_for(BODY_LABEL_ML)
        # Must be a 3-tuple: (canonical_render: str, source_module_id: Optional[str],
        #                      emission_ordinal: int)
        assert isinstance(identity, tuple)
        assert len(identity) == 3
        canonical_render, source_module_id, emission_ordinal = identity
        assert isinstance(canonical_render, str) and len(canonical_render) > 0
        # source_module_id is None for fixtures that bypass source-module injection
        assert source_module_id is None
        assert isinstance(emission_ordinal, int) and emission_ordinal >= 0


# =============================================================================
# TestUnrolledCaptureMLMatCount
# =============================================================================

class TestUnrolledCaptureMLMatCount:
    """ML_MAT_COUNT constant drives ML copy count and is importable."""

    def test_14_ml_mat_count_drives_ml_copy_count(self):
        """Number of ML records equals ML_MAT_COUNT."""
        fpc = _make_fpc(pro_n=1, ml_n=2, ngl_n=1, nll_n=1)
        uc = UnrolledCapture.from_four_part_capture(fpc)
        assert sum(1 for r in uc.records if r.body_label == BODY_LABEL_ML) == ML_MAT_COUNT

    def test_15_ml_mat_count_importable_from_schedule_capture(self):
        """ML_MAT_COUNT is importable from ScheduleCapture and equals 2."""
        from Tensile.Components.ScheduleCapture import ML_MAT_COUNT as MC
        assert MC == 2


# =============================================================================
# TestUnrolledCaptureEdgeCases
# =============================================================================

class TestUnrolledCaptureEdgeCases:
    """Edge cases: missing ML codepath 0, empty ML body."""

    def test_16_raises_value_error_when_no_codepath_0(self):
        """ValueError raised if main_loop has no codepath 0 entry."""
        fpc = FourPartCapture(
            main_loop={1: _make_body(2)},  # codepath 1 only
            main_loop_prev={1: _make_body(2)},
            n_gl={},
            n_ll={},
            num_mfma=2,
            num_codepaths=1,
            source="test-fixture",
        )
        with pytest.raises(ValueError, match="codepath 0"):
            UnrolledCapture.from_four_part_capture(fpc)

    def test_17_empty_ml_body_produces_zero_instruction_records(self):
        """ML body with no instructions produces ML_MAT_COUNT records each with 0 instructions."""
        empty_ml = LoopBodyCapture(instructions=[])
        fpc = FourPartCapture(
            main_loop={0: empty_ml},
            main_loop_prev={0: empty_ml},
            n_gl={},
            n_ll={},
            num_mfma=0,
            num_codepaths=1,
            source="test-fixture",
        )
        uc = UnrolledCapture.from_four_part_capture(fpc)
        ml_records = [r for r in uc.records if r.body_label == BODY_LABEL_ML]
        assert len(ml_records) == ML_MAT_COUNT
        assert all(len(r.instructions) == 0 for r in ml_records)
        assert uc.total_instructions == 0
