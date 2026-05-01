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
"""Cross-cutting LR/Pack negative regression tests.

This file covers Category 4 from the validator regression-test plan: schedules
where BOTH the LR and the Pack are positioned after the MFMA that consumes
their data, AND the LR is strictly before the Pack and guaranteed before it.

In this shape:
  - Pack.must_start_after (RAW from LR) IS satisfied (LR.guaranteed_by < Pack.issued_at)
  - LR.guaranteed_by > consumer_mfma.issued_at (LR check fails)
  - Pack.issued_at > consumer_mfma.issued_at (Pack check would also fail)

`validate_timeline` walks instructions in linearized issue order, so the LR
(with the lower vmfma_index) validates first and its error fires. This
specifically tests that the LR check still catches the violation even when
the LR<->Pack ordering is locally consistent.

LR1/LR3 variants are not buildable here: their consumer MFMA lives in the next
loop iteration, and the schedule only addresses the current iteration. To
position Pack "after" a next-iter consumer would require placing Pack in the
next iter, which the single-iter schedule format does not express.
"""
from rocisa.instruction import SWaitCnt

from Tensile.Components.CMSValidator import (
    add_local_read_constraints, add_pack_constraints,
)
from Tensile.Components.ScheduleCapture import WaitTooLateFailure
from cms_validation_base import CMSValidationTestBase


class TestLRAndPackAfterMFMA(CMSValidationTestBase):
    """LR and Pack both after the consumer MFMA, with LR finishing before Pack."""
    validator_passes = [add_local_read_constraints, add_pack_constraints]

    def test_LR0_both_after_MFMA_LR_before_Pack(self):
        """LRA0 @ idx=5 (after consumer MFMA @ idx=4); SWaitCnt @ idx=6 covers it
        (LRA0.guaranteed_by=6); PackA0 @ idx=7 (strictly after LRA0.guaranteed_by).
        The LR<->Pack RAW dependency is locally satisfied, but LR.guaranteed_by
        exceeds its consumer MFMA's issued_at. LR.validate() fires first."""
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[2, 6]],
            "LRA0": [[5, 5, 5, 5, 5, 5, 5, 5]],
            "LRB0": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "PackA0": [[7, 7, 7, 7, 7, 7, 7, 7]],
            "PackB0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            "LRA1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "LRB1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "PackA1": [[-1, -1, -1, -1, -1, -1, -1, -1]],
            "PackB1": [[-1, -1, -1, -1, -1, -1, -1, -1]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRA0"),
        ]
        f = self.validate(optSchedule, syncCode, 1, 2, 2, 0,
                          expected_failure=WaitTooLateFailure)
        assert f.producer.name == "LRA0"
        assert f.producer.issued_at.vmfma_index == 5
        assert f.consumer.name == "MFMA"
        assert f.consumer.issued_at.vmfma_index == 4
        assert f.wait_position.vmfma_index == 6

    def test_LR0_baseline_passing(self):
        """Same skeleton with LRA0 and PackA0 at valid positions."""
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[2, 5]],
            "LRA0": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "LRB0": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "PackA0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            "PackB0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            "LRA1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "LRB1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "PackA1": [[-1, -1, -1, -1, -1, -1, -1, -1]],
            "PackB1": [[-1, -1, -1, -1, -1, -1, -1, -1]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRA0"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0)
