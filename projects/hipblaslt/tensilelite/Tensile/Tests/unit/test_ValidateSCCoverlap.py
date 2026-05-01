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

from Tensile.Components.CMSValidator import verify_scc_overlap
from Tensile.Components.ScheduleCapture import SCCConflictFailure
from cms_validation_base import CMSValidationTestBase

class TestValidateSCCOverlap(CMSValidationTestBase):
    needs_timeline = False

    def validation_function(self, sched, kernel_dict, codePathIdx, timeline=None):
        return verify_scc_overlap(sched, kernel_dict, codePathIdx)

    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None):
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update({"MIWaveTileA": 4, "MIWaveTileB": 4, "DirectToLds": 1})
        super().setup_method(method, kernel_updates=kernel_updates)

    def _expect_scc_conflict(self, optSchedule, syncCode, *,
                             conflicting_name, grinc_name,
                             conflicting_index, interval_start, interval_end):
        """Validate that the schedule fails with the expected SCC-conflict shape."""
        failure = self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=SCCConflictFailure,
        )
        self.assert_scc_conflict(
            failure,
            conflicting_name=conflicting_name,
            grinc_name=grinc_name,
            conflicting_index=conflicting_index,
            interval_start=interval_start,
            interval_end=interval_end,
        )

    
    def test_gr_simple(self):
        self.kernel["Use64bShadowLimit"] = 1
        assert self.num_vmfma == 32
        optSchedule = {
            "SYNC": [0],
            "GRIncA": [[0, 0, 1, 1, 2, 2, 3, 3, 4]],
            "GRIncB": [[5, 5, 6, 6, 7, 7, 8, 8, 9]],
            "GRA": [[10, 11]],
            "GRB": [[12, 13]],
            'LWSA': [[31]],
            'LWSB': [[31]]
        }
        syncCode = [
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment=""),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0)

        # GRA in GRIncA
        optSchedule["GRA"] = [[2, 11]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRA", grinc_name="GRIncA",
            conflicting_index=2, interval_start=2, interval_end=3,
        )

        # GRB in GRIncB
        optSchedule["GRA"] = [[10, 11]]
        optSchedule["GRB"] = [[6, 11]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRB", grinc_name="GRIncB",
            conflicting_index=6, interval_start=6, interval_end=7,
        )

    def test_gr_declaration_order(self):
        self.kernel["Use64bShadowLimit"] = 1
        assert self.num_vmfma == 32
        optSchedule = {
            "SYNC": [0],
            "GRA": [[16, 17]],
            "GRIncA": [[0, 0, 1,
                        2, 3,
                        4, 5,
                        6, 7]],
            "GRIncB": [[8, 8, 9,
                        10, 11,
                        12, 13,
                        14, 15]],
            "GRB": [[18, 19]],
            'LWSA': [[31]],
            'LWSB': [[31]]
        }
        syncCode = [
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment=""),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0)

        # GRA just before GRIncA
        optSchedule["GRA"] = [[0, 11]]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

        # GRB just before GRIncB
        optSchedule["GRB"] = [[8, 11]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRB", grinc_name="GRIncB",
            conflicting_index=8, interval_start=8, interval_end=9,
        )

        # GRA just after GRIncA
        optSchedule["GRB"] = [[12, 13]]
        optSchedule["GRA"] = [[1, 11]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRA", grinc_name="GRIncA",
            conflicting_index=1, interval_start=0, interval_end=1,
        )

        # GRB just after GRIncA
        optSchedule["GRA"] = [[10, 11]]
        optSchedule["GRB"] = [[1, 11]]
        self.validate(optSchedule, syncCode, 1, None, None, 0)


    def test_gr_interval(self):
        self.kernel["Use64bShadowLimit"] = 1
        assert self.num_vmfma == 32
        optSchedule = {
            "SYNC": [0],
            "GRIncA": [[0, 0, 1,
                        2, 3,
                        4, 5,
                        6, 7]],
            "GRIncB": [[8, 8, 9,
                        10, 11,
                        12, 13,
                        14, 15]],
            "GRA": [[16, 17]],
            "GRB": [[18, 19]],
            'LWSA': [[31]],
            'LWSB': [[31]]
        }
        syncCode = [
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment=""),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0)

        optSchedule["GRA"] = [[0, 11]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRA", grinc_name="GRIncA",
            conflicting_index=0, interval_start=0, interval_end=1,
        )

        optSchedule["GRA"] = [[2, 11]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRA", grinc_name="GRIncA",
            conflicting_index=2, interval_start=2, interval_end=3,
        )

        optSchedule["GRA"] = [[3, 11]]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

        optSchedule["GRB"] = [[6, 11]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRB", grinc_name="GRIncA",
            conflicting_index=6, interval_start=6, interval_end=7,
        )


    def test_gr_noshadow(self):
        self.kernel["Use64bShadowLimit"] = 0
        assert self.num_vmfma == 32
        optSchedule = {
            "SYNC": [0],
            "GRIncA": [[0, 0, 1,
                        2, 3,
                        4]],
            "GRA": [[16, 17]],
            "GRIncB": [[6, 7, 8,
                        9, 10,
                        10]],
            "GRB": [[18, 19]],
            'LWSA': [[31]],
            'LWSB': [[31]]
        }
        syncCode = [
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment=""),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0)

        # GRA inside GRInc interval
        optSchedule["GRA"] = [[0, 11]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRA", grinc_name="GRIncA",
            conflicting_index=0, interval_start=0, interval_end=1,
        )

        # GRA just after GRInc interval
        optSchedule["GRA"] = [[1, 11]]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

        # GRB after GRIncB-10
        optSchedule["GRB"] = [[10, 11]]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

        # GRA before GRIncB-10 -> middle of interval
        optSchedule["GRA"] = [[10, 11]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRA", grinc_name="GRIncB",
            conflicting_index=10, interval_start=9, interval_end=10,
        )

    def test_lws(self):
        self.kernel["Use64bShadowLimit"] = 0
        assert self.num_vmfma == 32
        optSchedule = {
            "SYNC": [0],
            'LWSA': [[31]],
            "GRIncA": [[0, 0, 1,
                        2, 3,
                        4]],
            "GRA": [[16, 17]],
            "GRIncB": [[6, 7, 8,
                        9, 10,
                        10]],
            "GRB": [[18, 19]],
            'LWSB': [[31]]
        }
        syncCode = [
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment=""),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0)

        optSchedule["LWSA"] = [[0]]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

        optSchedule["LWSA"] = [[1]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="LWSA", grinc_name="GRIncA",
            conflicting_index=1, interval_start=0, interval_end=1,
        )

        optSchedule["LWSA"] = [[2]]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

        optSchedule["LWSB"] = [[9]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="LWSB", grinc_name="GRIncB",
            conflicting_index=9, interval_start=9, interval_end=10,
        )

        
    def test_gr_inc_together(self):
        self.kernel["Use64bShadowLimit"] = 0
        assert self.num_vmfma == 32
        optSchedule = {
            "SYNC": [0],
            'LWSA': [[31]],
            "GRIncA": [[0, 0, 1,
                        3, 4,
                        4]],
            "GRA": [[16, 17]],
            "GRIncB": [[6, 7, 8,
                        9, 10,
                        10]],
            "GRB": [[18, 19]],
            'LWSB': [[31]]
        }
        syncCode = [
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment=""),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0)

        optSchedule["GRIncB"] = [[0, 0, 1,
                                  9, 10,
                                  10]]
        self._expect_scc_conflict(
            optSchedule, syncCode,
            conflicting_name="GRIncB", grinc_name="GRIncA",
            conflicting_index=0, interval_start=0, interval_end=1,
        )

        optSchedule["GRIncB"] = [[1, 1, 2,
                            9, 10,
                            10]]
        self.validate(optSchedule, syncCode, 1, None, None, 0)



    