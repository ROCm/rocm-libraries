################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
from typing import Any, Optional
from rocisa.instruction import SWaitCnt

from Tensile.Components.CMSValidator import (
    add_local_read_constraints,
    index_for_force_unroll_sub_iter, lr_needed_by_mfma,
)
from Tensile.Components.ScheduleCapture import WaitTooLateFailure
from cms_validation_base import CMSValidationTestBase
from Tensile.Common import IsaVersion


def _assert_wait_too_late(failure, *, producer_name, producer_idx,
                          consumer_idx, wait_idx):
    """Assert WaitTooLateFailure carries the expected LR/MFMA/wait positions."""
    assert failure.producer.name == producer_name, (
        f"producer.name: expected {producer_name!r}, got {failure.producer.name!r}"
    )
    assert failure.producer.issued_at.vmfma_index == producer_idx, (
        f"producer.issued_at.vmfma_index: expected {producer_idx}, "
        f"got {failure.producer.issued_at.vmfma_index}"
    )
    assert failure.consumer.name == "MFMA", (
        f"consumer.name: expected 'MFMA', got {failure.consumer.name!r}"
    )
    assert failure.consumer.issued_at.vmfma_index == consumer_idx, (
        f"consumer.issued_at.vmfma_index: expected {consumer_idx}, "
        f"got {failure.consumer.issued_at.vmfma_index}"
    )
    assert failure.wait_position.vmfma_index == wait_idx, (
        f"wait_position.vmfma_index: expected {wait_idx}, "
        f"got {failure.wait_position.vmfma_index}"
    )


class TestValidateLRsCompleteBeforeVMFMA(CMSValidationTestBase):
    validator_passes = [add_local_read_constraints]

    def test_simple_LR0(self):
        """
        Verify the simple case where both LRA0 and LRB0 are issued and finished before the halfway point of the main loop.
        """
        assert self.num_vmfma == 8
        
        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

        optSchedule["LRA0"] = [[1, 6]]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA0", producer_idx=6,
                              consumer_idx=5, wait_idx=3)

        optSchedule["LRA0"] = [[1, 2]]
        optSchedule["LRB0"] = [[3, 6]]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRB0", producer_idx=3,
                              consumer_idx=4, wait_idx=3)

    def test_simple_LR0_w_LR1(self):
        """
        Handle case where we start reading LRA1 before halfway point.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[3, 7]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[2]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_complex_LR0(self):
        """
        2nd LRB0 is not needed until iteration 6 & 7, can have SWaitCnt for it after the halfway point.
        """
        assert self.num_vmfma == 8
        
        optSchedule = {
            "SYNC": [[3, 5]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 2]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_simple_LR1(self):
        """
        Case where LR1 is finished before the end of the current iteration.
        """
        assert self.num_vmfma == 8
        
        optSchedule = {
            "SYNC": [[1, 7]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[4, 4]],
            "LRB1": [[4, 4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_pre_loop_SWaitCnt(self):
        """
        Case where LR1 is finished before start of next iteration.
        """
        assert self.num_vmfma == 8
        
        optSchedule = {
            "SYNC": [[-1, 3]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[5, 5]],
            "LRB1": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_pre_loop_LR(self):
        """
        Case where an LR is issued before the start of the loop (idx=-1).
        """
        assert self.num_vmfma == 8
        
        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[-1, -1]],
            "LRB0": [[-1, -1]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_simple_LR1_guaranteed_too_late(self):
        """
        Case where LR1 is guaranteed too late.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[1]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[4, 4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA1", producer_idx=4,
                              consumer_idx=0, wait_idx=1)

    def test_complex_LR1(self):
        """
        Case where LR1 finishes during the beginning of next iteration.
        """
        assert self.num_vmfma == 8
        
        optSchedule = {
            "SYNC": [[1, 3, 7]],
            "LRA0": [[2, 2]],
            "LRB0": [[2, 2]],
            "LRA1": [[4, 4]],
            "LRB1": [[4, 4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/2 LRB1"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All of LRA0 and LRB0"),
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="2/2 LRA1 and 1/2 LRB1"),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

        # Failing case: LRA1 finishes too late
        optSchedule["LRA1"] = [[4, 5]]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA1", producer_idx=5,
                              consumer_idx=1, wait_idx=1)

    def test_more_LRs(self):
        """
        Case where each LR reads less than 1 WaveTile worth of data.
        Even though there are only 2 tiles of A and 2 of B there are 4 LRAs and 4 LRBs, each loading 1/2 a tile of A and B respectively.
        """
        assert self.num_vmfma == 8
        
        optSchedule = {
            "SYNC": [[0, 1, 3, 4, 5, 7]],
            "LRA0": [[0, 0, 3, 3]],
            "LRB0": [[1, 1, 2, 4]],
            "LRA1": [[5, 5, 7, 7]],
            "LRB1": [[6, 6, 7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="4/4 LRA1 and 2/4 LRB1"),
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="4/4 LRB1"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/4 LRA0 and 3/4 LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="4/4 LRA0 and 3/4 LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="4/4 LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/4 LRA1 and 2/4 LRB1"),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_more_LRs_failure(self):
        """
        Case where each LR reads less than 1 WaveTile worth of data, but barriers set up wrong.
        Even though there are only 2 tiles of A and 2 of B there are 4 LRAs and 4 LRBs, each loading 1/2 a tile of A and B respectively.
        """
        assert self.num_vmfma == 8
        
        optSchedule = {
            "SYNC": [[3, 4]],
            "LRA0": [[1, 1, 1, 1]],
            "LRB0": [[0, 0, 0, 0]],
        }
        syncCode = [
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Incorrectly wait for only LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all loads"),
        ]
        
        # Failure case 1: Don't wait for any LRA0
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA0", producer_idx=1,
                              consumer_idx=4, wait_idx=4)

        # Failure case 2: Wait for only 1/4 LRA0 (need at least 2/4 LRA0) to do VMFMA 4.
        syncCode[0].dscnt = 3
        syncCode[0].comment = "Wait for LRB0 and 1/4 LRA0"
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA0", producer_idx=1,
                              consumer_idx=4, wait_idx=4)

        # Passing case: Correctly SWaitCnt for 2/4 LRA0 (i.e. 1/2 As) in time for VMFMA 4.
        syncCode[0].dscnt = 2
        syncCode[0].comment = "Wait for LRB0 and 2/4 LRA0"
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)
    
    def test_less_LRs(self):
        """
        Case where each LR reads more than 1 WaveTile worth of data.
        Even though there are 4 tiles of A and 4 of B there are only 2 LRAs and 2 LRBs, each loading 2 tiles of A and B respectively.
        """
        self.setup_method(kernel_updates={
            "MIWaveTileA": 4,
            "MIWaveTileB": 4
        })
        assert self.num_vmfma == 32
        
        optSchedule = {
            "SYNC": [[7, 15, 31]],
            "LRA0": [[12, 13]],
            "LRB0": [[13, 14]],
            "LRA1": [[16, 17]],
            "LRB1": [[18, 19]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/2 LRB1"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRA0 and LRB0"),
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="LRA1 and 1/2 LRB1"),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

        # Failure case
        optSchedule["SYNC"][0][0] = 8
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRB1", producer_idx=19,
                              consumer_idx=8, wait_idx=8)

    def test_handling_instruction_order(self):
        """
        The order in which instructions are added to optSchedule dictate the order in which instructions are added at each index.
        Ensure that the code correctly handles this ordering, specifically the relative order of LRs and the relative order of SWaitCnts.
        """
        assert self.num_vmfma == 8

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LR0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LR1s"),
        ]

        # 1. SwaitCnt before all LRs
        optSchedule = {
            "SYNC": [[3, 7]],
            "LRA0": [[3]],
            "LRB0": [[3]],
            "LRA1": [[7]],
            "LRB1": [[7]],
        }
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA0", producer_idx=3,
                              consumer_idx=4, wait_idx=7)

        # 2. SwaitCnt after LR0s but before LR1s
        # LR0s will now pass, but LR1s will fail
        optSchedule = {
            "LRA0": [[3]],
            "LRB0": [[3]],
            "SYNC": [[3, 7]],
            "LRA1": [[7]],
            "LRB1": [[7]],
        }
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA1", producer_idx=7,
                              consumer_idx=0, wait_idx=3)

        # 3. SwaitCnt after all LRs
        optSchedule = {
            "LRA0": [[3]],
            "LRB0": [[3]],
            "LRA1": [[7]],
            "LRB1": [[7]],
            "SYNC": [[3, 7]],
        }
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

class TestValidateLRsCompleteBeforeVMFMA_tf32(CMSValidationTestBase):
    """
    For TF32, each tile produces 3 MFMAs (hi*hi, hi*lo, lo*hi).
    With MIWaveTileA=2, MIWaveTileB=2, we have 4 tile pairs, producing 12 MFMAs total.
    
    MFMA ordering (column-major):
    - MFMAs 0,1,2: A3*B3 (hi*hi, hi*lo, lo*hi)
    - MFMAs 3,4,5: A0*B3
    - MFMAs 6,7,8: A3*B0
    - MFMAs 9,10,11: A0*B0
    
    LRB0 needed at 6
    LRA0 needed at 3
    LRB3 needed at 12 (0 of next iter)
    LRA3 needed at 12 (0 of next iter)
    """
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None):
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update({"UseF32XEmulation": True, "ISA": IsaVersion(9,5,0), "DepthU": 32, "ForceUnrollSubIter": True})
        super().setup_method(method, kernel_updates=kernel_updates)

    validator_passes = [add_local_read_constraints]

    def test_LR0s_pass(self):
        """
        Simple TF32 test case.
        """
        assert self.num_vmfma == 12
        
        optSchedule = {
            "SYNC": [[2, 5]],
            "LRA0": [[0, 0]],
            "LRB0": [[1, 1]],
        }
        syncCode = [
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="LRA0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRB0"),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)
    
    def test_LR0s_fail(self):
        """
        Same as passing case, but LRA0 guaranteed too late.
        """
        assert self.num_vmfma == 12
        
        optSchedule = {
            "SYNC": [[3, 5]],
            "LRA0": [[0, 0]],
            "LRB0": [[1, 1]],
        }
        syncCode = [
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="LRA0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRB0"),
        ]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA0", producer_idx=0,
                              consumer_idx=3, wait_idx=3)

    def test_LR0s_fail2(self):
        """
        Same as passing case, but LRB0 guaranteed too late.
        """
        assert self.num_vmfma == 12
        
        optSchedule = {
            "SYNC": [[2, 6]],
            "LRA0": [[0, 0]],
            "LRB0": [[1, 1]],
        }
        syncCode = [
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="LRA0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRB0"),
        ]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRB0", producer_idx=1,
                              consumer_idx=6, wait_idx=6)

    def test_LR3s_pass(self):
        """
        Passing case for LRA3s
        """
        assert self.num_vmfma == 12

        optSchedule = {
            "SYNC": [[2, 5, 7]],
            "LRA0": [[0, 0]],
            "LRB0": [[1, 1]],
            "LRA1": [[6, 6]],
            "LRB1": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="LRA0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRA3 and LRB3"),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

class TestValidateLRsCompleteBeforeVMFMA_MfmaReorder(CMSValidationTestBase):
    validator_passes = [add_local_read_constraints]

    def test_simple_bf16(self):
        """
        Simple test when the indices are fully reversed. 
        """
        assert self.num_vmfma == 8
        
        mfmaReorder = list(range(self.num_vmfma-1, -1, -1))
        optSchedule = { 
            # LR0s needed for first half
            "LRA0": [[-1, -1]],
            "LRB0": [[-1, -1]],
            # LR1s needed for second half
            "LRA1": [[0, 0]],
            "LRB1": [[0, 0]],
            "SYNC": [[-1, self.num_vmfma // 2 - 1]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LR0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LR1s"),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0, None, mfmaReorder=mfmaReorder)

    def test_complex_bf16(self):
        """
        Change from column-major to row-major.

        | 0 2 | -> | 0 1 |
        | 1 3 | -> | 2 3 |
        """
        assert self.num_vmfma == 8

        mfmaReorder = [0, 2, 1, 3, 4, 6, 5, 7]
        optSchedule = {
            "LRB0": [[3, 4]],
            "LRA0": [[3, 5]],
            "LRB1": [[5, 6]],
            "LRA1": [[5, 7]],
            "SYNC": [[0, 1, 3, 4, 5, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/2 LRB1"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/2 LRA1"),

            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="1/2 LRB0 and 1/2 LRA0."),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/2 LRB0"),
            SWaitCnt(dscnt=0+1, vlcnt=-1, vscnt=-1, comment="2/2 LRA0"),

            SWaitCnt(dscnt=2,vlcnt=-1, vscnt=-1, comment="1/2 LRB1 and 1/2 LRA1."),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0, None, nllZeroDscnt=True, mfmaReorder=mfmaReorder)

    def test_asymmetric_reorder_bf16(self):
        """
        Test with asymmetric mfmaReorder that exposes the inverse mapping bug.
        
        Uses 3x2 tile configuration (n_tiles_a=3, n_tiles_b=2) for 12 total MFMAs.
        The reordering groups by A tile within each half, keeping halves separate.
        This mirrors the pattern used in the 96x256x64 schedule.
        
        Default column-major MFMA layout:
           B0  B1                    B0  B1
        A0 [ 0   3 ]  1st half    A0 [ 6   9 ]  2nd half
        A1 [ 1   4 ]              A1 [ 7  10 ]
        A2 [ 2   5 ]              A2 [ 8  11 ]
        
        Reorder: group by A tile within each half (A0s first, then A1s, then A2s)
        mfmaReorder[new_pos] = original_pos
        - 1st half: new [0,1,2,3,4,5] get originals [0,3, 1,4, 2,5]
        - 2nd half: new [6,7,8,9,10,11] get originals [6,9, 7,10, 8,11]
        
        Inverse mapping (original -> new_pos) for 1st half:
        - original 0 -> new 0,  original 3 -> new 1
        - original 1 -> new 2,  original 4 -> new 3
        - original 2 -> new 4,  original 5 -> new 5
        
        For LRA1 loading A tile 2 (for next iteration, original indices 2, 5):
        - Correct: inverse[2]=4, inverse[5]=5, min=4
        - Buggy: mfma_reorder[2]=1, mfma_reorder[5]=5, min=1
        
        With buggy logic:
        - needed_by = 1
        
        With correct logic:
        - needed_by = 4
        """
        # Use 3x2 tile configuration
        self.setup_method(kernel_updates={"MIWaveTileA": 3, "MIWaveTileB": 2})
        assert self.num_vmfma == 12  # 2 * 3 * 2
        
        # Reorder: Reorder to row-major order.
        # Before reordering (default col-major order), MFMA tiles are multiplied in the following order:
        #   idx: 0    1    2    3    4    5    | 6    7    8    9    10   11
        #   mul: A0B0 A1B0 A2B0 A0B1 A1B1 A2B1 | A0B0 A1B0 A2B0 A0B1 A1B1 A2B1

        # After reordering (mfmaReorder), new order of multiplication:
        #     idx: 0    1    2    3    4    5    | 6    7    8    9    10   11
        # old idx: 0    3    1    4    2    5    | 6    9    7    10    8   11
        #     mul: A0B0 A0B1 A1B0 A1B1 A2B0 A2B1 | A0B0 A0B1 A1B0 A1B1 A2B0 A2B1
        mfmaReorder = [
            0, 3, 
            1, 4, 
            2, 5,
            
            6, 9,
            7, 10,
            8, 11
        ]
        
        syncTable = [
            -1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LR1s done"),
            
            1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LRB0 done"),

            5, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="All LRB0 and 1/3 LRA0s done"),
            7, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="2/3 LRA1s done"),
            9, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="3/3 LRA1s done"),
        ]

        optSchedule = {
            "SYNC": [syncTable[::2]],
            # LR0s load data for 2nd half - issue early
            "LRB0": [[0,0]],
            "LRA0": [[2, 3, 4]],
            
            "LRA1": [[11,11,11]],
            "LRB1": [[11,11]],
        }
        syncCode = syncTable[1::2]
        
        self.validate(optSchedule, syncCode, 1, None, None, 0, None, mfmaReorder=mfmaReorder)

    def test_tf32(self):
        """
        """
        self.setup_method(kernel_updates={"UseF32XEmulation": True, "ISA": IsaVersion(9,5,0), "DepthU": 32, "ForceUnrollSubIter": True})
        assert self.num_vmfma == 12
        
        # Row-major reordering: swap middle two groups
        # Original: [0,1,2, 3,4,5, 6,7,8, 9,10,11] (col-major: A3B3, A0B3, A3B0, A0B0)
        # Reorder:  [0,1,2, 6,7,8, 3,4,5, 9,10,11] (row-major: A3B3, A3B0, A0B3, A0B0)
        mfmaReorder = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]
        
        # After reordering, the physical MFMAs are:
        # 0-2: A0*B0, 3-5: A0*B1, 6-8: A1*B0, 9-11: A1*B1
        # LR0 loads for 2nd half (originally 6-11, which are tiles A0*B1 and A1*B1)
        # After reordering, these tiles are at physical indices 3-5 and 9-11
        optSchedule = {
            "LRA0": [[0, 0]],  # Load A0 and A1 early
            "LRB0": [[0, 0]],  # Load B0 and B1 early
            "LRA1": [[6, 6]],  # Load for next iteration
            "LRB1": [[7, 7]],  
            "SYNC": [[2, 10]],  # Wait for LR0s before first use, and LR1s before next iter
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LR0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LR1s"),
        ]
        
        self.validate(optSchedule, syncCode, 1, None, None, 0, None, mfmaReorder=mfmaReorder)

class TestValidateLRsCompleteBeforeVMFMA_ForceUnrollSubIter(CMSValidationTestBase):
    """
    When ForceUnrollSubIter is enabled, the MFMAs are issued in a different order than the default column-major order.
    Further, each loop iteration only contains MIWaveTileA * MIWaveTileB MFMAs, instead of 2 * MIWaveTileA * MIWaveTileB MFMAs.

    Default is column-major ordering of MFMAs:
    A  B ->
    | 0 4  8 12 |
    | 1 5  9 13 |
    | 2 6 10 14 |
    | 3 7 11 15 |

    When ForceUnrollSubIter is enabled, the MFMAs are issued in 4 quandrants in column-major order, within each quandrant the MFMAsare issued in row-major order:
    | 0 2  8 10 |
    | 1 3  9 11 |
    | 4 6 12 14 |
    | 5 7 13 15 |

    LR0s load the second half of tile data, and LR3s load the first half.
    Thus:
    - LRA0 must be finished before num_vmfma // 4 - 1
    - LRB0 must be finished before num_vmfma // 2 - 1
    - LRA3 must be finished before 0 (of next iteration)
        - MUST also start after 3 * num_vmfma // 4 (since it's reusing registers used by LRA0)
    - LRB3 must be finished before 0 (of next iteration)
        - MUST also start after num_vmfma // 2 (since it's reusing registers used by LRB0)

    NOTE:   These tests do not include the Pack commands which are REQUIRED for a valid kernel.
            Not including them here in order to test just the LR-VMFMA orderling logic.
    """
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None):
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update({"ForceUnrollSubIter": True, "MIWaveTileA": 4, "MIWaveTileB": 4, "DepthU": 32})
        super().setup_method(method, kernel_updates=kernel_updates)

    validator_passes = [add_local_read_constraints]


    def test_bf16_pass(self):
        assert self.num_vmfma == 16

        optSchedule = {
            "SYNC": [[3, 15]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA3": [[7, 7]],
            "LRB3": [[7, 7]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="For LR0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="For LR3s"),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_bf16_fail(self):
        """Same as above, but with the proper SWaitCnt for the LR3s."""
        assert self.num_vmfma == 16

        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA3": [[7, 7]],
            "LRB3": [[7, 7]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="For LR0s"),
        ]

        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA3", producer_idx=7,
                              consumer_idx=0, wait_idx=3)


class TestIndexForForceUnrollSubIter:
    def test_index_for_force_unroll_sub_iter(self):
        M, N = 4, 4
        
        """
        Mapping between column-major linear index and the vmfma index when mfmas are done in column-major order.
        | 0 4  8 12 |
        | 1 5  9 13 |
        | 2 6 10 14 |
        | 3 7 11 15 |
        """
        index_col_major = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        """"
        Mapping between column-major linear index and the vmfma index when mfmas are done in for_unroll_sub_iter order.
        | 0 2  8 10 |
        | 1 3  9 11 |
        | 4 6 12 14 |
        | 5 7 13 15 |
        """
        index_expected  = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]

        index_permuted = [index_for_force_unroll_sub_iter(i, M, N) for i in index_col_major]
        assert index_permuted == index_expected

class TestLRNeededByMFMA:
    def test_simple_bf16(self):
        """
        Simple test with default (column-major) MFMA ordering.
        """
        n_tiles_a, n_tiles_b = 4, 4
        n_local_reads_a, n_local_reads_b = 4, 4
        num_vmfma = 2 * n_tiles_a * n_tiles_b

        force_unroll_sub_iter = False

        mfma_reorder = []

        expected_results = {
            "LRA0": [16, 17, 18, 19],
            "LRB0": [16, 20, 24, 28],
            "LRA1": [32, 33, 34, 35],  # 1st half of next iteration
            "LRB1": [32, 36, 40, 44],  # 1st half of next iteration
        }

        for lr_name, expected_indices in expected_results.items():
            actual_indices = [lr_needed_by_mfma(lr_name, lr_idx, num_vmfma, mfma_reorder, n_tiles_a, n_tiles_b, n_local_reads_a, n_local_reads_b, force_unroll_sub_iter, use_f32x_emulation=False) for lr_idx in range(len(expected_indices))]
            assert actual_indices == expected_indices

    def test_single_sub_iter_bf16(self):
        """
        Non-force-unroll case where MFMAs only cover a single sub-iteration.
        """
        n_tiles_a, n_tiles_b = 8, 8
        n_local_reads_a, n_local_reads_b = 8, 8
        num_vmfma = n_tiles_a * n_tiles_b

        force_unroll_sub_iter = False
        mfma_reorder = []

        expected_results = {
            "LRA0": [32, 32, 33, 33, 34, 34, 35, 35],
            "LRB0": [32, 32, 40, 40, 48, 48, 56, 56],
            "LRA1": [64, 64, 65, 65, 66, 66, 67, 67],
            "LRB1": [64, 64, 72, 72, 80, 80, 88, 88],
        }

        for lr_name, expected_indices in expected_results.items():
            actual_indices = [
                lr_needed_by_mfma(
                    lr_name,
                    lr_idx,
                    num_vmfma,
                    mfma_reorder,
                    n_tiles_a,
                    n_tiles_b,
                    n_local_reads_a,
                    n_local_reads_b,
                    force_unroll_sub_iter,
                    use_f32x_emulation=False,
                )
                for lr_idx in range(len(expected_indices))
            ]
            assert actual_indices == expected_indices
    
    def test_simple_tf32(self):
        """
        TF32 without ForceUnrollSubIter - each tile produces 3 MFMAs.
        """
        n_tiles_a, n_tiles_b = 4, 4
        n_local_reads_a, n_local_reads_b = 4, 4
        num_vmfma = 2 * n_tiles_a * n_tiles_b * 3  # TF32: 3 MFMAs per tile

        force_unroll_sub_iter = False
        mfma_reorder = []

        expected_results = {
            "LRA0": [48, 51, 54, 57],
            "LRB0": [48, 60, 72, 84],
            "LRA1": [96, 99, 102, 105],
            "LRB1": [96, 108, 120, 132],
        }

        for lr_name, expected_indices in expected_results.items():
            actual_indices = [lr_needed_by_mfma(lr_name, lr_idx, num_vmfma, mfma_reorder,
                             n_tiles_a, n_tiles_b, n_local_reads_a, n_local_reads_b,
                             force_unroll_sub_iter, use_f32x_emulation=True)
                             for lr_idx in range(len(expected_indices))]
            assert actual_indices == expected_indices
    
    def test_mfma_reorder_bf16(self):
        """
        Simple test with MFMAs reordered into row-major order.
        """
        n_tiles_a, n_tiles_b = 4, 4
        n_local_reads_a, n_local_reads_b = 4, 4
        num_vmfma = 2 * n_tiles_a * n_tiles_b

        force_unroll_sub_iter = False
        
        mfma_reorder = [ 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15,
                        16, 20, 24, 28, 17, 21, 25, 29, 18, 22, 26, 30, 19, 23, 27, 31]

        expected_results = {
            "LRA0": [16, 20, 24, 28],
            "LRB0": [16, 17, 18, 19],
            "LRA1": [32, 36, 40, 44],  # 1st half of next iteration
            "LRB1": [32, 33, 34, 35],  # 1st half of next iteration
        }

        for lr_name, expected_indices in expected_results.items():
            actual_indices = [lr_needed_by_mfma(lr_name, lr_idx, num_vmfma, mfma_reorder, n_tiles_a, n_tiles_b, n_local_reads_a, n_local_reads_b, force_unroll_sub_iter, use_f32x_emulation=False) for lr_idx in range(len(expected_indices))]
            assert actual_indices == expected_indices

    def test_force_unroll_sub_iter_bf16(self):
        """
        Simple test case with force_unroll_sub_iter enabled. MFMA order:
        |  0  2 |  8 10 |
        |  1  3 |  9 11 |
        |-------|-------|
        |  4  6 | 12 14 |
        |  5  7 | 13 15 |
        """
        n_tiles_a, n_tiles_b = 4, 4
        n_local_reads_a, n_local_reads_b = 4, 4
        num_vmfma = n_tiles_a * n_tiles_b

        force_unroll_sub_iter = True
        
        mfma_reorder = []

        expected_results = {
            "LRA0": [ 4,  4,  5,  5],
            "LRB0": [ 8,  8, 10, 10],
            "LRA3": [16, 16, 17, 17],  # 1st half of next iteration
            "LRB3": [16, 16, 18, 18],  # 1st half of next iteration
        }

        for lr_name, expected_indices in expected_results.items():
            actual_indices = [lr_needed_by_mfma(lr_name, lr_idx, num_vmfma, mfma_reorder, n_tiles_a, n_tiles_b, n_local_reads_a, n_local_reads_b, force_unroll_sub_iter, use_f32x_emulation=False) for lr_idx in range(len(expected_indices))]
            assert actual_indices == expected_indices
    
    def test_mfma_reorder_and_force_unroll_sub_iter_bf16(self):
        """
        Test combining force_unroll_sub_iter with an additional mfma_reorder.
        
        The reordering process has two steps:
        1. force_unroll_sub_iter permutation is applied first, splitting the 4x4 tile
           into quadrants processed in column-major order, column-major within each quadrant:
           |  0  2 |  8 10 |
           |  1  3 |  9 11 |
           |-------|-------|
           |  4  6 | 12 14 |
           |  5  7 | 13 15 |
        
        2. Then mfma_reorder is applied on top to change to row-major quadrants 
           with row-major within each quadrant:
           |  0  1 |  4  5 |
           |  2  3 |  6  7 |
           |-------|-------|
           |  8  9 | 12 13 |
           | 10 11 | 14 15 |
        
        The mfma_reorder maps each force_unroll_sub_iter position to its new
        execution position in the row-major scheme.
        """
        n_tiles_a, n_tiles_b = 4, 4
        n_local_reads_a, n_local_reads_b = 4, 4
        num_vmfma = n_tiles_a * n_tiles_b

        force_unroll_sub_iter = True
        
        # Row-major quadrants, row-major within each quadrant
        # Maps force_unroll index -> execution position
        mfma_reorder = [0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15]

        expected_results = {
            # LRA0: second half of rows (rows 2-3)
            # lr_idx 0,1 -> row 2 -> +offset=2 -> force_unroll(2)=4 -> reorder[4]=8
            # lr_idx 2,3 -> row 3 -> +offset=3 -> force_unroll(3)=5 -> reorder[5]=10
            "LRA0": [8, 8, 10, 10],
            # LRB0: second half of cols (cols 2-3)
            # lr_idx 0,1 -> col 2 -> +offset=8 -> force_unroll(8)=8 -> reorder[8]=4
            # lr_idx 2,3 -> col 3 -> +offset=12 -> force_unroll(12)=10 -> reorder[10]=5
            "LRB0": [4, 4, 5, 5],
            # LRA3: first half of rows for next iteration
            # lr_idx 0,1 -> row 0 -> force_unroll(0)=0 -> reorder[0]=0 -> +16=16
            # lr_idx 2,3 -> row 1 -> force_unroll(1)=1 -> reorder[1]=2 -> +16=18
            "LRA3": [16, 16, 18, 18],
            # LRB3: first half of cols for next iteration
            # lr_idx 0,1 -> col 0 -> force_unroll(0)=0 -> reorder[0]=0 -> +16=16
            # lr_idx 2,3 -> col 1 -> col-major idx 4 -> force_unroll(4)=2 -> reorder[2]=1 -> +16=17
            "LRB3": [16, 16, 17, 17],
        }

        for lr_name, expected_indices in expected_results.items():
            actual_indices = [lr_needed_by_mfma(lr_name, lr_idx, num_vmfma, mfma_reorder, n_tiles_a, n_tiles_b, n_local_reads_a, n_local_reads_b, force_unroll_sub_iter, use_f32x_emulation=False) for lr_idx in range(len(expected_indices))]
            assert actual_indices == expected_indices

    def test_force_unroll_sub_iter_tf32(self):
        """
        Test TF32 (F32X emulation) combined with ForceUnrollSubIter.
        
        This test catches the bug where the *3 multiplication for F32X was applied
        BEFORE index_for_force_unroll_sub_iter() instead of AFTER. The reordering
        function operates on tile indices (0 to n_tiles_a * n_tiles_b - 1), so the
        *3 conversion to MFMA indices must happen after reordering.
        
        With ForceUnrollSubIter, each tile produces 3 MFMAs (hi*hi, hi*lo, lo*hi):
        |  0-2   6-8  | 24-26 30-32 |
        |  3-5   9-11 | 27-29 33-35 |
        |-------------|-------------|
        | 12-14 18-20 | 36-38 42-44 |
        | 15-17 21-23 | 39-41 45-47 |
        
        LR0s load the second half of tile data:
        - LRA0 loads rows 2-3 (bottom half), needed starting at MFMA 12
        - LRB0 loads cols 2-3 (right half), needed starting at MFMA 24
        
        LR3s load the first half for next iteration.
        """
        n_tiles_a, n_tiles_b = 4, 4
        n_local_reads_a, n_local_reads_b = 4, 4
        num_vmfma = n_tiles_a * n_tiles_b * 3  # TF32: 3 MFMAs per tile

        force_unroll_sub_iter = True
        mfma_reorder = []

        expected_results = {
            # LRA0: second half of rows (rows 2-3)
            # Tile calculation: lr_idx 0,1 -> tile row 2, lr_idx 2,3 -> tile row 3
            # After offset (+2) and force_unroll reordering, then *3 for TF32:
            # lr_idx 0,1: offset=2, force_unroll(2)=4, *3=12
            # lr_idx 2,3: offset=3, force_unroll(3)=5, *3=15
            "LRA0": [12, 12, 15, 15],
            # LRB0: second half of cols (cols 2-3)
            # lr_idx 0,1: linear=0, offset=8, force_unroll(8)=8, *3=24
            # lr_idx 2,3: linear=4, offset=12, force_unroll(12)=10, *3=30
            "LRB0": [24, 24, 30, 30],
            # LRA3: first half of rows for next iteration
            # lr_idx 0,1: linear=0, force_unroll(0)=0, *3=0, +48=48
            # lr_idx 2,3: linear=1, force_unroll(1)=1, *3=3, +48=51
            "LRA3": [48, 48, 51, 51],
            # LRB3: first half of cols for next iteration
            # lr_idx 0,1: linear=0, force_unroll(0)=0, *3=0, +48=48
            # lr_idx 2,3: linear=4, force_unroll(4)=2, *3=6, +48=54
            "LRB3": [48, 48, 54, 54],
        }

        for lr_name, expected_indices in expected_results.items():
            actual_indices = [lr_needed_by_mfma(lr_name, lr_idx, num_vmfma, mfma_reorder,
                             n_tiles_a, n_tiles_b, n_local_reads_a, n_local_reads_b,
                             force_unroll_sub_iter, use_f32x_emulation=True)
                             for lr_idx in range(len(expected_indices))]
            assert actual_indices == expected_indices, f"{lr_name}: expected {expected_indices}, got {actual_indices}"

    def test_mfma_reorder_requires_min_across_consumers_bf16(self):
        """
        Test that exposes the bug where MFMA reorder causes a later logical consumer
        to execute before the first logical consumer.
        
        LR data is consumed by multiple MFMAs (one per tile in the opposite dimension).
        Before the fix, the code assumed that the first mfma before the reorder would still be the first mfma after the reorder. 
        With complex reordering, a different consumer can execute first.
        
        This test uses a reorder that swaps columns in the second half:
        - Column 0 ↔ Column 3
        - Column 1 ↔ Column 2
        
        Second half layout (column-major) before reorder:
        | 16 20 24 28 |
        | 17 21 25 29 |
        | 18 22 26 30 |
        | 19 23 27 31 |
        
        After column swap reorder:
        | 28 24 20 16 |
        | 29 25 21 17 |
        | 30 26 22 18 |
        | 31 27 23 19 |
        
        For LRA0 (A tile 0 in second half):
        - Logical consumers: 16, 20, 24, 28 (entire row 0)
        - After reorder: 28, 24, 20, 16
        - Correct answer: min(28, 24, 20, 16) = 16
        - Bug (checking only first consumer): mfma_reorder[16] = 28
        """
        n_tiles_a, n_tiles_b = 4, 4
        n_local_reads_a, n_local_reads_b = 4, 4
        num_vmfma = 2 * n_tiles_a * n_tiles_b

        force_unroll_sub_iter = False
        
        # Column swap reorder for second half: col 0↔3, col 1↔2
        # First half unchanged, second half has columns swapped
        mfma_reorder = list(range(16)) + [28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19]

        expected_results = {
            # LRA0: A tiles in second half, consumed by MFMAs in each row
            # For A tile i, consumers are at columns 0,1,2,3 -> after reorder, column 3 executes first
            "LRA0": [16, 17, 18, 19],  # min across all B tiles after reorder
            # LRB0: B tiles in second half, consumed by MFMAs in each column
            # For B tile j, consumers are at rows 0,1,2,3 (same column) -> reorder within column
            "LRB0": [28, 24, 20, 16],  # min across all A tiles after reorder
            # LRA1/LRB1: first half of next iteration (unchanged by this reorder)
            "LRA1": [32, 33, 34, 35],
            "LRB1": [32, 36, 40, 44],
        }

        for lr_name, expected_indices in expected_results.items():
            actual_indices = [
                lr_needed_by_mfma(lr_name, lr_idx, num_vmfma, mfma_reorder, n_tiles_a, n_tiles_b,
                                  n_local_reads_a, n_local_reads_b, force_unroll_sub_iter, use_f32x_emulation=False)
                for lr_idx in range(len(expected_indices))
            ]
            assert actual_indices == expected_indices, f"{lr_name}: expected {expected_indices}, got {actual_indices}"

    def test_mfma_reorder_requires_min_across_consumers_force_unroll_bf16(self):
        """
        Test that exposes the bug with ForceUnrollSubIter enabled.
        
        ForceUnrollSubIter layout before reorder:
        |  0  2 |  8 10 |
        |  1  3 |  9 11 |
        |-------|-------|
        |  4  6 | 12 14 |
        |  5  7 | 13 15 |

        After reorder:
        |  3  1 |  11 9 |
        |  2  0 |  10 8 |
        |-------|-------|
        |  7  5 |  15 13 |
        |  6  4 |  14 12 |
        
        This test uses a reorder that reverses execution within each quadrant,
        so the "first" consumer in each quadrant executes last.
        
        For LRA0 (A tile 2 = second half of rows, lr_idx=0):
        - Consumers: (2,0), (2,1), (2,2), (2,3) in original coords
        - After ForceUnrollSubIter: 4, 6, 12, 14
        - After reorder (reverse in quadrant): 5, 7, 15, 13
        - Correct answer: min(5, 7, 15, 13) = 5
        - Bug (checking only first consumer): mfma_reorder[4] = 5
        
        Note: In this particular test, the bug doesn't manifest for LRA0 idx=0
        because the first consumer happens to still be the minimum after reorder.
        But it does manifest for other cases where the reorder moves a non-first
        consumer to an earlier execution slot.
        """
        n_tiles_a, n_tiles_b = 4, 4
        n_local_reads_a, n_local_reads_b = 4, 4
        num_vmfma = n_tiles_a * n_tiles_b

        force_unroll_sub_iter = True
        
        # Reorder that reverses within each quadrant
        # Quadrant 0: [0,1,2,3] -> [3,2,1,0]
        # Quadrant 1: [4,5,6,7] -> [7,6,5,4]
        # Quadrant 2: [8,9,10,11] -> [11,10,9,8]
        # Quadrant 3: [12,13,14,15] -> [15,14,13,12]
        mfma_reorder = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]

        expected_results = {
            # LRA0: A tiles 2-3 (second half)
            # A tile 2: col-major (2,0),(2,1),(2,2),(2,3) = 2,6,10,14 -> ForceUnroll 4,6,12,14 -> reorder 7,5,15,13 -> min=5
            # A tile 3: col-major (3,0),(3,1),(3,2),(3,3) = 3,7,11,15 -> ForceUnroll 5,7,13,15 -> reorder 6,4,14,12 -> min=4
            "LRA0": [5, 5, 4, 4],
            # LRB0: B tiles 2-3 (second half)
            # B tile 2: col-major (0,2),(1,2),(2,2),(3,2) = 8,9,10,11 -> ForceUnroll 8,9,12,13 -> reorder 11,10,15,14 -> min=10
            # B tile 3: col-major (0,3),(1,3),(2,3),(3,3) = 12,13,14,15 -> ForceUnroll 10,11,14,15 -> reorder 9,8,13,12 -> min=8
            "LRB0": [10, 10, 8, 8],
            # LRA3/LRB3: first half for next iteration (computed, then + num_vmfma)
            # A tile 0: col-major (0,0),(0,1),(0,2),(0,3) = 0,4,8,12 -> ForceUnroll 0,2,8,10 -> reorder 3,1,11,9 -> min=1 -> +16=17
            # A tile 1: col-major (1,0),(1,1),(1,2),(1,3) = 1,5,9,13 -> ForceUnroll 1,3,9,11 -> reorder 2,0,10,8 -> min=0 -> +16=16
            "LRA3": [17, 17, 16, 16],
            # B tile 0: col-major (0,0),(1,0),(2,0),(3,0) = 0,1,2,3 -> ForceUnroll 0,1,4,5 -> reorder 3,2,7,6 -> min=2 -> +16=18
            # B tile 1: col-major (0,1),(1,1),(2,1),(3,1) = 4,5,6,7 -> ForceUnroll 2,3,6,7 -> reorder 1,0,5,4 -> min=0 -> +16=16
            "LRB3": [18, 18, 16, 16],
        }

        for lr_name, expected_indices in expected_results.items():
            actual_indices = [
                lr_needed_by_mfma(lr_name, lr_idx, num_vmfma, mfma_reorder, n_tiles_a, n_tiles_b,
                                  n_local_reads_a, n_local_reads_b, force_unroll_sub_iter, use_f32x_emulation=False)
                for lr_idx in range(len(expected_indices))
            ]
            assert actual_indices == expected_indices, f"{lr_name}: expected {expected_indices}, got {actual_indices}"


class TestLRAfterMFMA(CMSValidationTestBase):
    """
    Negative regression tests: LR is positioned or guaranteed after the MFMA that
    consumes its data. Covers LR0 and LR1 in the default kernel; LR3 lives in
    TestLRAfterMFMA_LR3 below.

    Two shapes per LR type:
      A: LR placed at a high vmfma_index. The only SWaitCnt that can cover it
         comes from the next loop iteration (cross-iter wraparound) and falls
         after the consumer MFMA.
      B: LR placed at a low vmfma_index, but the SWaitCnt that covers it sits
         in the same iteration after the consumer MFMA.

    Both shapes fire LocalRead.validate() at CMSValidator.py:356 ("issued too late").
    """
    validator_passes = [add_local_read_constraints]

    def test_LR0_A_positionally_late(self):
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[0, 6]],
            "LRB0": [[0, 0]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA0", producer_idx=6,
                              consumer_idx=5, wait_idx=3)

    def test_LR0_B_waitcnt_after_consumer(self):
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[6]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA0", producer_idx=0,
                              consumer_idx=4, wait_idx=6)

    def test_LR0_baseline_passing(self):
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_LR1_A_positionally_late(self):
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[7, 7]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA1", producer_idx=7,
                              consumer_idx=0, wait_idx=3)

    def test_LR1_B_waitcnt_after_consumer(self):
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[4, 4]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA1", producer_idx=4,
                              consumer_idx=0, wait_idx=1)


class TestLRAfterMFMA_LR3(CMSValidationTestBase):
    """LR3 variant of TestLRAfterMFMA. Requires ForceUnrollSubIter kernel setup."""
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None):
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update({"ForceUnrollSubIter": True, "MIWaveTileA": 4, "MIWaveTileB": 4, "DepthU": 32})
        super().setup_method(method, kernel_updates=kernel_updates)

    validator_passes = [add_local_read_constraints]

    def test_LR3_A_positionally_late(self):
        assert self.num_vmfma == 16
        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA3": [[15, 15]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA3", producer_idx=15,
                              consumer_idx=0, wait_idx=3)

    def test_LR3_B_waitcnt_after_consumer(self):
        assert self.num_vmfma == 16
        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA3": [[7, 7]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")]
        f = self.validate(optSchedule, syncCode, 1, None, None, 0,
                          expected_failure=WaitTooLateFailure)
        _assert_wait_too_late(f, producer_name="LRA3", producer_idx=7,
                              consumer_idx=0, wait_idx=3)
