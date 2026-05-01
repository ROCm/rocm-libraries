################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from rocisa.instruction import SWaitCnt, SBarrier

from Tensile.Components.CMSValidator import (
    add_local_read_constraints, add_gr_finish_before_lr_constraints,
)
from Tensile.Components.ScheduleCapture import (
    MissingWaitFailure, WaitTooLateFailure,
)
from cms_validation_base import CMSValidationTestBase


class TestValidateNgl(CMSValidationTestBase):
    validator_passes = [add_gr_finish_before_lr_constraints]

    def make_simple_schedule_and_sync(self) -> tuple[dict[str, list[list[int]]], list[SWaitCnt | SBarrier]]:
        """
        Create a simple schedule that is valid in the mainloop, but is susceptible to race conditions in the NGL loop based on the value of shift.
        It is susceptible becasue only SWaitcnt for the Global Reads occurs AFTER the when the GRs in this iteration would be issued.

        Schedule contains 3 GRAs and 3 GRBs.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[0, 5, 5, self.num_vmfma-1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0,0, 1,1, 2,2]],
            "GRB":  [[0,0, 1,1, 2,2]],
            "LRA1": [[6]],
            "LRB1": [[6]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        return optSchedule, syncCode

    def test_simple_case_success(self):
        """
        Test the simple schedule using the exact amount of shift required to avoid race conditions.
        """
        shift_value = 3 + 3  # 3 GRAs and 3 GRBs
        optSchedule, syncCode = self.make_simple_schedule_and_sync()
        self.validate(optSchedule, syncCode, 1, shift_value, shift_value, 0)

    def test_simple_case_failure(self):
        """
        Test the simple schedule using a shift value that is too small to avoid race conditions.
        """
        shift_value = 3 + 3 - 1  # 3 GRAs and 3 GRBs - 1
        optSchedule, syncCode = self.make_simple_schedule_and_sync()
        self.validate(optSchedule, syncCode, 1, shift_value, shift_value, 0,
                                         expected_failure=MissingWaitFailure)

    def test_simple_case_success_too_high(self):
        """
        Test the simple schedule using a shift value larger than needed, ensure that we don't add in negative values.
        """
        shift_value = 3 + 3 + 1  # 3 GRAs and 3 GRBs + 1
        optSchedule, syncCode = self.make_simple_schedule_and_sync()
        self.validate(optSchedule, syncCode, 1, shift_value, shift_value, 0)

class TestValidateNll(CMSValidationTestBase):
    validator_passes = [add_local_read_constraints]

    def test_lr0_swait_depends_on_lr1(self):
        """
        Simple failure case where the SWaitCnt for LRB0 depends on the LRA1.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[3, 7]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[2]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0,
                                         expected_failure=WaitTooLateFailure)

    def test_lr0_swait_depends_on_lr1_realistic(self, useZeroDscnt: bool=False):
        """
        A more realistic version of `test_lr0_swait_depends_on_lr1`. GRs are now present. Optionally checks zero dscnt option for NLL
        """
        self.kernel["MIWaveTileA"] = 4
        self.kernel["MIWaveTileB"] = 4
        self.num_vmfma = 2 * self.kernel["MIWaveTileA"] * self.kernel["MIWaveTileB"]
        assert self.num_vmfma == 32

        syncTable = [
            1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Finish LRA0s"),
            1, SBarrier(comment=""),

            3, SWaitCnt(dscnt=-1, vlcnt=1, vscnt=-1, comment="Finish GRAs"),
            3, SBarrier(comment=""),

            15, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Finish 2/4 LRB0s"),

            # NOTE: This SWaitCnt is wrong for the NLL. It depends on the LRA1 being issued in order to guarantee that the LRB0s are finished.
            23, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Finish LRB0s"),

            28, SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Finish GRBs"),
            28, SBarrier(comment=""),

            31, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1s (but in NLL this leaves 2 LRB0s in-flight)"),
        ]
        optSchedule = {
            "SYNC": [syncTable[::2]],

            "LRA0": [[0]],
            "GRA":  [[2, 2]],

            "LRB0": [[3, 3, 3, 3]],

            "LRA1": [[22]],

            # Irrelevant, but need to schedule for correctness
            "GRB":  [[28, 28]],
            "LRB1": [[30, 30, 30, 30]],
        }
        # We need to set nglshift and nllshift for the vlcnt adjustments
        num_gr = 2  # 2 GRs total (1 GRA + 1 GRB, but we only count the actual reads not the increments)
        expected_failure = None if useZeroDscnt else WaitTooLateFailure
        self.validate(optSchedule, syncTable[1::2], 1, num_gr, num_gr, 0,
                      nllZeroDscnt=useZeroDscnt, expected_failure=expected_failure)

    def test_lr0_swait_depends_on_lr1_realistic_zero_dscnt(self):
        """
        Same as above, but uses the zero dscnt option for NLL, so the test should now pass.
        """
        self.test_lr0_swait_depends_on_lr1_realistic(useZeroDscnt=True)
