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
import unittest
from rocisa.instruction import SWaitCnt

from Tensile.Components.CMSValidator import verify_gr_inc_order
from test_CustomSchedule import create_base_kernel, ScheduleInfo

class TestGRIncOrder(unittest.TestCase):
    def setUp(self):
        self.kernel = create_base_kernel()
        self.kernel["MIWaveTileA"] = 4
        self.kernel["MIWaveTileB"] = 4
        
        self.kernel["DirectToLds"] = 1
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

        sched = ScheduleInfo(1, self.num_vmfma, optSchedule, syncCode, None, None)
        status, message = verify_gr_inc_order(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        # GRA before GRIncA
        optSchedule["GRA"] = [[0, 0]]
        status, message = verify_gr_inc_order(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        # GRA before GRIncB
        optSchedule["GRA"] = [[11, 11]]
        optSchedule["GRB"] = [[6, 6]]
        status, message = verify_gr_inc_order(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        # GRA Pass with same index
        optSchedule["GRA"] = [[5, 5]]
        optSchedule["GRB"] = [[12, 12]]
        status, message = verify_gr_inc_order(sched, {"kernel": self.kernel})
        assert  status, f"Schedule should have passed validation but did not. {message}"

        # GRB Fail with same index
        optSchedule["GRB"] = [[10, 10]]
        status, message = verify_gr_inc_order(sched, {"kernel": self.kernel})
        assert  not status, f"Schedule should have failed validation but did not. {message}"


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

        sched = ScheduleInfo(1, self.num_vmfma, optSchedule, syncCode, None, None)
        status, message = verify_gr_inc_order(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        # GRA before GRIncB
        optSchedule["GRA"] = [[0, 0]]
        status, message = verify_gr_inc_order(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        # GRB before GRIncA
        optSchedule["GRA"] = [[11, 11]]
        optSchedule["GRB"] = [[6, 6]]
        status, message = verify_gr_inc_order(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"
