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
from rocisa.instruction import SWaitCnt

from Tensile.Components.CMSValidator import verify_lrs_finished_before_vmfma
from cms_validation_base import CMSValidationTestBase

class TestValidateLRsCompleteBeforeVMFMA(CMSValidationTestBase):
    def validation_function(self, sched, kernel_dict, codePathIdx):
        return verify_lrs_finished_before_vmfma(sched, kernel_dict, codePathIdx)

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
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "LRA0 at index 6 is not valid. Needed before index 5, but only guaranteed at index 3."
        )

        optSchedule["LRA0"] = [[1, 2]]
        optSchedule["LRB0"] = [[3, 6]]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "LRB0 at index 3 is not valid. Needed before index 4, but only guaranteed at index 3."
        )

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

    def test_simple_LR1_never_guaranteed(self):
        """
        Case where LR1 is finished before the end of loop.
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
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "LRA1 at index 4 is not valid. Needed before index 0, but only guaranteed at index 1."
        )

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
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "LRA1 at index 5 is not valid. Needed before index 1 (of next iteration), but only guaranteed at index 1."
        )

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
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "LRA0 at index 1 is not valid. Needed before index 4, but only guaranteed at index 4."
        )

        # Failure case 2: Wait for only 1/4 LRA0 (need at least 2/4 LRA0) to do VMFMA 4.
        syncCode[0].dscnt = 3
        syncCode[0].comment = "Wait for LRB0 and 1/4 LRA0"
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "LRA0 at index 1 is not valid. Needed before index 4, but only guaranteed at index 4."
        )

        # Passing case: Correctly SWaitCnt for 2/4 LRA0 (i.e. 1/2 As) in time for VMFMA 4.
        syncCode[0].dscnt = 2
        syncCode[0].comment = "Wait for LRB0 and 2/4 LRA0"
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)
    
    def test_less_LRs(self):
        """
        Case where each LR reads more than 1 WaveTile worth of data.
        Even though there are 4 tiles of A and 4 of B there are only 2 LRAs and 2 LRBs, each loading 2 tiles of A and B respectively.
        """
        from test_CustomSchedule import create_base_kernel
        self.kernel = create_base_kernel()
        self.kernel["MIWaveTileA"] = 4
        self.kernel["MIWaveTileB"] = 4
        self.num_vmfma = 2 * self.kernel["MIWaveTileA"] * self.kernel["MIWaveTileB"]
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
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "LRB1 at index 19 is not valid. Needed before index 8 (of next iteration), but only guaranteed at index 8."
        )

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
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "LRA0 at index 3 is not valid. Needed before index 4, but only guaranteed at index 7."
        )

        # 2. SwaitCnt after LR0s but before LR1s
        # LR0s will now pass, but LR1s will fail
        optSchedule = {
            "LRA0": [[3]],
            "LRB0": [[3]],
            "SYNC": [[3, 7]],
            "LRA1": [[7]],
            "LRB1": [[7]],
        }
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "LRA1 at index 7 is not valid. Needed before index 0, but only guaranteed at index 3."
        )

        # 3. SwaitCnt after all LRs
        optSchedule = {
            "LRA0": [[3]],
            "LRB0": [[3]],
            "LRA1": [[7]],
            "LRB1": [[7]],
            "SYNC": [[3, 7]],
        }
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

class TestValidateLRsCompleteBeforeVMFMA_MfmaReorder(CMSValidationTestBase):
    def validation_function(self, sched, kernel_dict, codePathIdx):
        return verify_lrs_finished_before_vmfma(sched, kernel_dict, codePathIdx)

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
