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
"""Negative regression tests for the validator's GR -> LR3 dependency path.

`set_gr_needed_by_from_lrs` (CMSValidator.py:1334-1371) has a fallback: when
LRA1/LRB1 are absent (ForceUnrollSubIter mode), it switches the GR target from
LRA1/LRB1 to LRA3/LRB3. The existing GR test files exercise only the LR1 path;
these tests cover the LR3 fallback by mirroring the LR1 negative tests with
ForceUnrollSubIter setup.
"""
from typing import Any, Optional

from rocisa.instruction import SWaitCnt, SBarrier

from Tensile.Components.CMSValidator import add_gr_finish_before_lr_constraints
from Tensile.Components.ScheduleCapture import (
    MissingWaitFailure, MissingBarrierFailure, WaitTooLateFailure,
)
from cms_validation_base import CMSValidationTestBase


class TestValidateGRsCompleteBeforeLr3s(CMSValidationTestBase):
    """Mirror of TestValidateGRsCompleteBeforeLr1s for the LR3 fallback path.

    The schedule omits LRA1/LRB1 entirely, forcing
    set_gr_needed_by_from_lrs (CMSValidator.py:1349-1350) into the LR3 branch.
    Each negative test mirrors a corresponding LR1 test verbatim with the
    target changed to LRA3.
    """
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None):
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update({"ForceUnrollSubIter": True, "MIWaveTileA": 4, "MIWaveTileB": 4, "DepthU": 32})
        super().setup_method(method, kernel_updates=kernel_updates)

    validator_passes = [add_gr_finish_before_lr_constraints]

    def test_LR3_simple_case_success(self):
        """Baseline: GRs precede an SWait + SBarrier, then LR3s in next iteration."""
        assert self.num_vmfma == 16

        optSchedule = {
            "SYNC": [[0, 2, 4, self.num_vmfma - 1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0, 0]],
            "GRB":  [[0, 0]],
            "LRA3": [[7, 7]],
            "LRB3": [[7, 7]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR3s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0)

    def test_LR3_grs_not_swait(self):
        """GR-wait has vlcnt=-1, so apply_swaits never sets GR.guaranteed_by.
        Fires GlobalRead._validate_needed_by line 550."""
        assert self.num_vmfma == 16

        optSchedule = {
            "SYNC": [[0, 2, 4, self.num_vmfma - 1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0, 0]],
            "GRB":  [[0, 0]],
            "LRA3": [[7, 7]],
            "LRB3": [[7, 7]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR3s"),
        ]
        self.validate(
            optSchedule, syncCode, 1, 2, 2, 0,
            expected_failure=MissingWaitFailure,
        )

    def test_LR3_no_sbarrier(self):
        """SBarrier omitted from syncCode, so apply_barriers leaves GR.barriered_at empty.
        Fires GlobalRead._validate_needed_by line 557."""
        assert self.num_vmfma == 16

        optSchedule = {
            "SYNC": [[0, 1, self.num_vmfma - 1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0, 0]],
            "GRB":  [[0, 0]],
            "LRA3": [[7, 7]],
            "LRB3": [[7, 7]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR3s"),
        ]
        self.validate(
            optSchedule, syncCode, 1, 2, 2, 0,
            expected_failure=MissingBarrierFailure,
        )

    def test_LR3_swait_after_sbarrier(self):
        """SBarrier sits before the GR-SWait, so no SBarrier falls between SWait and LR3.
        Fires GlobalRead._validate_needed_by line 565."""
        assert self.num_vmfma == 16

        optSchedule = {
            "SYNC": [[0, 2, 3, self.num_vmfma - 1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0, 0]],
            "GRB":  [[0, 0]],
            "LRA3": [[7, 7]],
            "LRB3": [[7, 7]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR3s"),
        ]
        self.validate(
            optSchedule, syncCode, 1, 2, 2, 0,
            expected_failure=MissingBarrierFailure,
        )

    def test_LR3_guaranteed_after_first_lr3(self):
        """GR-SWait at idx=4 lands AFTER the first LRA3 (placed at idx=3).
        Fires GlobalRead._validate_needed_by line 561."""
        assert self.num_vmfma == 16

        optSchedule = {
            "SYNC": [[0, 4, 4, self.num_vmfma - 1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0, 0, 0, 0]],
            "GRB":  [[0, 0, 0, 0]],
            "LRA3": [[3, 7, 7, 7]],
            "LRB3": [[7, 7, 7, 7]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR3s"),
        ]

        self.validate(
            optSchedule, syncCode, 1, 4, 4, 0,
            expected_failure=WaitTooLateFailure,
        )

    def test_LR3_swap_global_read_order_failure(self):
        """SwapGlobalReadOrder=True. GRBs load A and must precede LRA3, but the
        scheduler placed waits as if no swap. Exercises the swap-after-LR3-fallback
        ordering at CMSValidator.py:1349-1353."""
        assert self.num_vmfma == 16
        self.kernel["SwapGlobalReadOrder"] = True

        optSchedule = {
            "SYNC": [[0, 1, 1, 4, 4, self.num_vmfma - 1]],
            "LRA0": [[-1]],
            "LRB0": [[-1]],
            "GRA":  [[0, 0]],
            "GRB":  [[3, 3]],
            "LRA3": [[2, 2]],
            "LRB3": [[7, 7]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRAs (loading B)"),
            SBarrier(comment="For GRAs (loading B)"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRBs (loading A)"),
            SBarrier(comment="For GRBs (loading A)"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR3s"),
        ]
        self.validate(
            optSchedule, syncCode, 1, 2, 2, 0,
            expected_failure=WaitTooLateFailure,
        )
