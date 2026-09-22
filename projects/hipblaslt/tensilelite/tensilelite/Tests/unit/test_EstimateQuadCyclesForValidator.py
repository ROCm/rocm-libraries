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

from Tensile.Components.CMSValidator import estimate_quad_cycles_precomputed, MFMA, MFMAPack, Pack, precompute_issue_times, SchedulePosition, SNop, ValidatorInstruction


def _pos(vmfma_index: int, sub_index: int) -> SchedulePosition:
    return SchedulePosition(loop_index=0, vmfma_index=vmfma_index, sub_index=sub_index)


class TestEstimateQuadCyclesValidator:
    def validate(self, instruction: ValidatorInstruction, expected_quad_cycles: int, all_instructions: list[ValidatorInstruction]):
        """
        Helper method to validate quad-cycle estimation.

        Ensures:
        1. The estimate function returns the expected quad-cycles
        2. Only the starting instruction has estimated_quad_cycles_before_result_used set to expected value
        3. All other instructions have estimated_quad_cycles_before_result_used set to 0
        """
        issue_times = precompute_issue_times(all_instructions)
        i_instruction = all_instructions.index(instruction)
        i_needed_by = all_instructions.index(instruction.needed_by)

        result = estimate_quad_cycles_precomputed(i_instruction, i_needed_by, issue_times)
        instruction.estimated_quad_cycles_before_result_used = result

        for i, instr in enumerate(all_instructions):
            if hasattr(instr, 'estimated_quad_cycles_before_result_used'):
                if instr is instruction:
                    assert instr.estimated_quad_cycles_before_result_used == expected_quad_cycles, \
                        f"Instruction at index {i} should have estimated_quad_cycles_before_result_used={expected_quad_cycles}, got {instr.estimated_quad_cycles_before_result_used}"
                else:
                    assert instr.estimated_quad_cycles_before_result_used == 0, \
                        f"Instruction at index {i} should have estimated_quad_cycles_before_result_used=0, got {instr.estimated_quad_cycles_before_result_used}"

    def test_back_to_back_mfma(self):
        """
        Test that consecutive MFMA instructions correctly stall each other.

        Validates that when multiple MFMA instructions are issued back-to-back,
        the estimated quad-cycles account for the execution latency and stalling.
        """
        target_mfma = MFMA(name="MFMA", issued_at=_pos(3, 0))
        pack0 = Pack(name="Pack0", issue_index=0, issued_at=_pos(-1, 0), needed_by=target_mfma)
        all_instructions = [
            pack0,
            MFMA(name="MFMA", issued_at=_pos(0, 0)),
            MFMA(name="MFMA", issued_at=_pos(1, 0)),
            MFMA(name="MFMA", issued_at=_pos(2, 0)),
            target_mfma,
        ]
        # +4x3 for the 3 MFMAs to issue + finish
        self.validate(pack0, 12, all_instructions)


    def test_back_to_back_scalar(self):
        """
        Test that consecutive non-MFMA instructions are estimated correctly.
        """
        target_snop = SNop(name="SNop", issued_at=_pos(-1, 3), wait_state=0)
        pack0 = Pack(name="Pack0", issue_index=0, issued_at=_pos(-1, 0), needed_by=target_snop)
        all_instructions = [
            pack0,
            SNop(name="SNop", issued_at=_pos(-1, 1), wait_state=0),
            SNop(name="SNop", issued_at=_pos(-1, 2), wait_state=0),
            target_snop,
        ]
        # +1 for 1st SNop
        # +1 for 2nd SNop
        self.validate(pack0, 2, all_instructions)

    def test_parallel_mfma_snop(self):
        """
        Test that MFMA and non-MFMA instructions execute in parallel without stalling.

        Validates that when MFMA instructions are interleaved with non-MFMA operations,
        the cycle estimation reflects parallel execution on different execution units,
        with no unnecessary stalls between these independent instruction types.
        """
        target_mfma = MFMA(name="MFMA", issued_at=_pos(1, 0))
        pack0 = Pack(name="Pack0", issue_index=0, issued_at=_pos(-1, 0), needed_by=target_mfma)
        all_instructions = [
            pack0,
            MFMA(name="MFMA", issued_at=_pos(0, 0)),
            SNop(name="SNop", issued_at=_pos(0, 1), wait_state=0),
            SNop(name="SNop", issued_at=_pos(0, 2), wait_state=0),
            SNop(name="SNop", issued_at=_pos(0, 3), wait_state=0),
            target_mfma
        ]
        # +1 for MFMA to issue
        # +3 for MFMA to finish (3 SNops hidden in this latency)
        self.validate(pack0, 4, all_instructions)

    def test_mfma_before_start_non_mfma_instruction(self):
        """
        Test that an MFMA preceding the window does not stall a leading non-MFMA instruction.

        Validates that when the instruction window starts with a non-MFMA instruction,
        an MFMA immediately before the window boundary does not cause stalls for the
        first instruction, as they execute on different hardware units.
        """
        target_mfma = MFMA(name="MFMA", issued_at=_pos(1, 0))
        pack0 = Pack(name="Pack0", issue_index=0, issued_at=_pos(0, 1), needed_by=target_mfma)
        all_instructions = [
            MFMA(name="MFMA", issued_at=_pos(0, 0)),
            # Window starts with Pack
            pack0,
            SNop(name="SNop", issued_at=_pos(0, 2), wait_state=0),
            SNop(name="SNop", issued_at=_pos(0, 3), wait_state=0),
            SNop(name="SNop", issued_at=_pos(0, 4), wait_state=0),
            target_mfma
        ]
        self.validate(pack0, 3, all_instructions)

    def test_mfma_before_start_non_mfma_instruction_2(self):
        """
        Test that an out-of-window MFMA correctly stalls a subsequent in-window MFMA.

        Validates that when the window starts with a non-MFMA instruction (which is not
        stalled by a preceding out-of-window MFMA), but a later MFMA instruction appears
        within the window, that in-window MFMA is correctly stalled by the out-of-window
        MFMA instruction. This tests the propagation of stall effects across instruction
        types.
        """
        target_snop = SNop(name="SNop", issued_at=_pos(1, 2), wait_state=0)
        pack0 = Pack(name="Pack0", issue_index=0, issued_at=_pos(0, 1), needed_by=target_snop)
        all_instructions = [
            MFMA(name="MFMA", issued_at=_pos(0, 0)),
            # Window starts with Pack
            pack0,
            MFMA(name="MFMA", issued_at=_pos(1, 0)),  # +2 qs of stall
            SNop(name="SNop", issued_at=_pos(1, 1), wait_state=0),
            target_snop
        ]
        # +3 quad-cycles for MFMA before window to finish.
        # +1 quad-cycle for MFMA in window to issue
        # (last SNop issues in parallel with MFMA finish)
        self.validate(pack0, 4, all_instructions)

    def test_mfma_before_start_mfma_instruction(self):
        """
        Test that an MFMA preceding the window correctly stalls the first MFMA in the window.

        The effect of this is that the issuing of the MFMA is stalled,
        but the number of quad-cycles in the window is smaller than in the tests above.
        """
        target_snop = SNop(name="SNop", issued_at=_pos(2, 2), wait_state=0)
        pack0 = Pack(name="Pack0", issue_index=0, issued_at=_pos(0, 1), needed_by=target_snop)
        all_instructions = [
            MFMA(name="MFMA", issued_at=_pos(0, 0)),
            # Window starts with Pack
            pack0,
            MFMA(name="MFMA", issued_at=_pos(1, 0)),
            SNop(name="SNop", issued_at=_pos(1, 1), wait_state=0),
            MFMA(name="MFMA", issued_at=_pos(2, 0)),  # +2 qs of stall
            SNop(name="SNop", issued_at=_pos(2, 1), wait_state=0),
            target_snop
        ]
        # +1 for Pack to issue
        # +3 for first MFMA in window to finish (1 SNop issues in parallel)
        # +3 for second MFMA to finish (1 SNop issues in parallel)
        # +1 for final SNop to issue
        self.validate(pack0, 8, all_instructions)

    def test_4x4mfma_pack_parallel(self):
        """
        The 5th and 6th Pack for 4x4MFMA are MFMAs which take 1 cycle to finish.
        Ensure that we can issue things in parallel with them.
        """
        target_snop = SNop(name="SNop", issued_at=_pos(-1, 2), wait_state=0)
        pack0 = MFMAPack(name="PackA0", issue_index=5, issued_at=_pos(-1, 0), needed_by=target_snop)
        all_instructions = [
            pack0,
            SNop(name="SNop", issued_at=_pos(-1, 1), wait_state=1),
            target_snop,
        ]
        # +1 for Pack to finish (SNop issues in parallel)
        # +1 for SNop to finish
        self.validate(pack0, 2, all_instructions)

    def test_4x4mfma_to_mfma_extra_stall(self):
        """
        Extra 4 quad-cycle latency associated with switching MFMA type.
        """
        target_snop = SNop(name="SNop", issued_at=_pos(0, 1), wait_state=0)
        pack0 = MFMAPack(name="PackA0", issue_index=5, issued_at=_pos(-1, 0), needed_by=target_snop)
        all_instructions = [
            pack0,
            MFMA(name="MFMA", issued_at=_pos(0, 0)),
            target_snop,
        ]
        # +1 for Pack to finish (MFMA stalls)
        # +1 extra since switching MFMA type
        # +1 for MFMA to issue
        self.validate(pack0, 3, all_instructions)

    def test_4x4mfma_to_mfma_extra_stall_missing(self):
        """
        4x4 -> 16x16 has extra 4 quad-cycle latency only when issuing in the first 3 quad-cycles after the MFMA.
        """
        target_snop = SNop(name="SNop", issued_at=_pos(0, 1), wait_state=0)
        pack0 = MFMAPack(name="PackA0", issue_index=5, issued_at=_pos(-1, 0), needed_by=target_snop)
        all_instructions = [
            pack0,
            SNop(name="SNop", issued_at=_pos(-1, 1), wait_state=1),
            MFMA(name="MFMA", issued_at=_pos(0, 0)),
            target_snop,
        ]
        # +1 for Pack to finish (Snop issue hidden)
        # +1 for SNop finish
        # +1 for MFMA to issue
        self.validate(pack0, 3, all_instructions)

    def test_mfma_to_4x4mfma_extra_stall(self):
        """
        Extra 4 quad-cycle latency associated with switching MFMA type.
        """
        target_snop = SNop(name="SNop", issued_at=_pos(0, 2), wait_state=0)
        pack0 = Pack(name="Pack0", issue_index=0, issued_at=_pos(-1, 0), needed_by=target_snop)
        all_instructions = [
            pack0,
            MFMA(name="MFMA", issued_at=_pos(0, 0)),
            MFMAPack(name="PackA0", issue_index=5, issued_at=_pos(0, 1)),
            target_snop,
        ]
        # +1 for MFMA to issue
        # +3 for MFMA to finish (PackA0 stalls)
        # +1 extra since switching MFMA type
        # +1 for PackA0 to issue
        self.validate(pack0, 6, all_instructions)

    def test_mfma_to_4x4mfma_extra_stall_missing(self):
        """
        16x16 -> 4x4 has extra 4 quad-cycle latency only when issuing in the first 4 quad-cycles after the MFMA.
        """
        target_snop = SNop(name="SNop", issued_at=_pos(0, 3), wait_state=0)
        pack0 = Pack(name="Pack0", issue_index=0, issued_at=_pos(-1, 0), needed_by=target_snop)
        all_instructions = [
            pack0,
            MFMA(name="MFMA", issued_at=_pos(0, 0)),
            SNop(name="SNop", issued_at=_pos(0, 1), wait_state=3),
            MFMAPack(name="PackA0", issue_index=5, issued_at=_pos(0, 2)),
            target_snop,
        ]
        # +1 for MFMA to issue
        # +3 for MFMA to finish (SNop issue + 2 wait hidden)
        # +1 for SNop to finish (3rd of 3 wait_state)
        # +1 for PackA0 to issue
        self.validate(pack0, 6, all_instructions)
