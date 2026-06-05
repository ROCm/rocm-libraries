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
from rocisa.instruction import SWaitCnt, SBarrier

from Tensile.Components.CMSValidator import add_gr_finish_before_lr_constraints
from cms_validation_base import CMSValidationTestBase

class TestValidateGRsCompleteBeforeLr1s(CMSValidationTestBase):
    validator_passes = [add_gr_finish_before_lr_constraints]

    def test_simple_case_success(self):
        """
        Simple case where the LR1s occur after the GRs with SWait and Sbarrier between them.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 2, 4, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0,0]],
            "GRB":  [[0,0]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_grs_not_swait(self):
        """
        Failure case where GRs are not guaranteed before LR1s due to incorrect SWaitCnt.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 2, 4, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0,0]],
            "GRB":  [[0,0]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(
            optSchedule, syncCode, 1, 2, 2, 0,
            "GRA @ idx=0 is not valid. There are no guarantees on when it will be done."
        )

    def test_no_sbarrier(self): 
        """
        Failure case where there is no SWait between GRs and LR1s.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 1, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0,0]],
            "GRB":  [[0,0]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(
            optSchedule, syncCode, 1, 2, 2, 0,
            "GRA @ idx=0 is not valid. There is no SBarrier acting on it."
        )

    def test_swait_after_sbarrier(self):
        """
        Failure case when the latest barrier acting on the GRs occurs BEFORE the SWaitCnt that guarantees those GRs.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 2, 3, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0,0]],
            "GRB":  [[0,0]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(
            optSchedule, syncCode, 1, 2, 2, 0,
            "GRA @ idx=0 is not valid. No SBarrier between SWait @ idx=3 and LRA1 @ idx=6. Order must be GRA -> SWait -> SBarrier -> LRA1."
        )

    def test_guaranteed_after_first_lr1(self):
        """
        Simple failure case where the last GR is guaranteed AFTER the first LR1 starts.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 4, 4, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0,0, 0,0]],
            "GRB":  [[0,0, 0,0]],
            "LRA1": [[3,6,6,6]],
            "LRB1": [[6,6,6,6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]

        self.validate(
            optSchedule, syncCode, 1, 4, 4, 0,
            "GRA @ idx=0 is not valid. It is guaranteed by the SWait @ idx=4 which is after the first corresponding LRA1 @ idx=3. Order must be GRA -> SWait -> SBarrier -> LRA1."
        )

    def test_less_than_1_full_iteration_to_complete(self):
        """
        Passing case where the GRs have and SWait and SBarrier on them before a full iteration has completed.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 1, 2, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[3,3]],
            "GRB":  [[3,3]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_some_grs_less_than_1_full_iteration_to_complete(self):
        """
        Passing case where some GRs have and SWait and SBarrier on them before a full iteration has completed.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 3, 3, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[1,1, 2,2, 3,3]],
            "GRB":  [[1,1, 2,2, 3,3]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 6, 6, 0, None)

    def test_swait_and_sbarrier_in_preloop(self):
        """
        Passing, but pathalogical, case where the SWait and SBarrier are in the preloop.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "SYNC": [[-1, -1, -1, self.num_vmfma-1]],
            "GRA":  [[3,3]],
            "GRB":  [[3,3]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_grs_in_preloop(self):
        """
        Passing, but pathalogical, case where the GRs are in the preloop.
        """
        assert self.num_vmfma == 8
        
        optSchedule = {
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "SYNC": [[-1, 0, 0, self.num_vmfma-1]],
            "GRA":  [[-1,-1]],
            "GRB":  [[-1,-1]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_last_gr_swait_sbarrier_in_same_index(self):
        """
        At index 1 we have [GRA, GRB, SWait, SBarrier].
        While we wouldn't schedule this, it is still valid.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "GRA":  [[1,1]],
            "GRB":  [[1,1]],
            "SYNC": [[0, 1, 1, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_swait_sbarrier_first_lr1_in_same_index(self):
        """
        At index 6 we have [SWait, SBarrier, LRA1, LRB1].
        While it's unlike this would actually be scheduled it's still valid and must pass validation.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 6, 6, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[1,1]],
            "GRB":  [[1,1]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_swap_global_read_order_success(self):
        """
        Passing case where the GlobalRead order is swapped.
        """
        assert self.num_vmfma == 8
        self.kernel["SwapGlobalReadOrder"] = True

        optSchedule = {
            "SYNC": [[0, 1, 1, 4, 4, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0,0]],
            "GRB":  [[3,3]],
            "LRA1": [[5]],
            "LRB1": [[2]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRBs (loading A)"),
            SBarrier(comment="For GRBs (loading A)"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRAs (loading B)"),
            SBarrier(comment="For GRAs (loading B)"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_swap_global_read_order_failure(self):
        """
        Failure case where the GlobalRead order is swapped.
        E.g. Someone forgets that the GRAs actually load B and the GRBs actually load A and scheudles them as normally.
        """
        assert self.num_vmfma == 8
        self.kernel["SwapGlobalReadOrder"] = True

        optSchedule = {
            "SYNC": [[0, 1, 1, 4, 4, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0,0]],
            "GRB":  [[3,3]],
            "LRA1": [[2]],
            "LRB1": [[5]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRAs (loading B)"),
            SBarrier(comment="For GRAs (loading B)"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRBs (loading A)"),
            SBarrier(comment="For GRBs (loading A)"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(
            optSchedule, syncCode, 1, 2, 2, 0,
            "GRB (Swapped, loading A) @ idx=3 is not valid. It is guaranteed by the SWait @ idx=4 which is after the first corresponding LRA1 @ idx=2. Order must be GRB (Swapped, loading A) -> SWait -> SBarrier -> LRA1."
        )

    def test_fail_with_odd_number_of_grs(self):
        """
        When using DirectToLds, we must have an even number of GRs.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 2, 4, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0]],
            "GRB":  [[0,0]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]

        self.kernel["DirectToLds"] = True
        expected_message = "Code path 0: GRA has an odd number of indices. Must be even if DirectToLds is True."
        try:
            self.validate(optSchedule, syncCode, 1, 2, 2, 0, expected_message)
        except AssertionError as e:
            assert str(e) == expected_message, f"Expected: {expected_message}, Got: {str(e)}"
