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

from Tensile.Components.CMSValidator import verify_gr_inc_order
from cms_validation_base import CMSValidationTestBase

class TestGRIncOrder(CMSValidationTestBase):
    def validation_function(self, sched, kernel_dict, codePathIdx):
        return verify_gr_inc_order(sched, kernel_dict, codePathIdx)

    def setUp(self):
        super().setUp()
        self.kernel["MIWaveTileA"] = 4
        self.kernel["MIWaveTileB"] = 4

        self.num_vmfma = 2 * self.kernel["MIWaveTileA"] * self.kernel["MIWaveTileB"]

    
    def test_gr_no_swap(self):
        self.kernel["SwapGlobalReadOrder"] = 0
        assert self.num_vmfma == 32
        optSchedule = {
            "SYNC": [0],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 4, 5, 5]],
            "GRA": [[11, 11]],
            "GRB": [[12, 12]],
            "GRIncB": [[7, 7, 8, 8, 9, 9, 10, 10, 10]],
            'LWSA': [[31]],
            'LWSB': [[31]]
        }
        syncCode = [
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment=""),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

        # GRA before GRIncA
        optSchedule["GRA"] = [[0, 0]]
        self.validate(optSchedule, syncCode, 1, None, None, 0,
                                         "GRIncA finishes after GRA starts (5 vs 0)")

        # GRA before GRIncB
        optSchedule["GRA"] = [[11, 11]]
        optSchedule["GRB"] = [[6, 6]]
        self.validate(optSchedule, syncCode, 1, None, None, 0,
                                         "GRIncB finishes after GRB starts (10 vs 6)")

        # GRA Pass with same index
        optSchedule["GRA"] = [[5, 5]]
        optSchedule["GRB"] = [[12, 12]]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

        # GRB Fail with same index
        optSchedule["GRB"] = [[10, 10]]
        self.validate(optSchedule, syncCode, 1, None, None, 0,
                                         "GRIncB finishes after GRB starts (10 vs 10)")


    def test_gr_swap(self):
        self.kernel["SwapGlobalReadOrder"] = 1
        assert self.num_vmfma == 32
        optSchedule = {
            "SYNC": [0],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 4, 5, 5]],
            "GRA": [[11, 11]],
            "GRB": [[12, 12]],
            "GRIncA": [[7, 7, 8, 8, 9, 9, 10, 10, 10]],
            'LWSA': [[31]],
            'LWSB': [[31]]
        }
        syncCode = [
            SWaitCnt(dscnt=-1, vlcnt=-1, vscnt=-1, comment=""),
        ]

        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

        # GRA before GRIncB
        optSchedule["GRA"] = [[0, 0]]
        self.validate(optSchedule, syncCode, 1, None, None, 0,
                                         "GRIncB finishes after GRA starts (5 vs 0)")

        # GRB before GRIncA
        optSchedule["GRA"] = [[11, 11]]
        optSchedule["GRB"] = [[6, 6]]
        self.validate(optSchedule, syncCode, 1, None, None, 0,
                                         "GRIncA finishes after GRB starts (10 vs 6)")
