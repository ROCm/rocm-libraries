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

from Tensile.Components.CustomSchedule import verify_scc_overlap
from test_CustomSchedule import create_base_kernel, ScheduleInfo

class TestVerifySCCOverlap(unittest.TestCase):
    def setUp(self):
        self.kernel = create_base_kernel()
        self.kernel["MIWaveTileA"] = 4
        self.kernel["MIWaveTileB"] = 4
        
        self.kernel["DirectToLds"] = 1
        self.num_vmfma = 2 * self.kernel["MIWaveTileA"] * self.kernel["MIWaveTileB"]

    
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

        sched = ScheduleInfo(1, self.num_vmfma, optSchedule, syncCode, None, None)
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        # GRA in GRIncA
        optSchedule["GRA"] = [[2, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        # GRB in GRIncB
        optSchedule["GRA"] = [[10, 11]]
        optSchedule["GRB"] = [[6, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

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

        sched = ScheduleInfo(1, self.num_vmfma, optSchedule, syncCode, None, None)
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        # GRA just before GRIncA
        optSchedule["GRA"] = [[0, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert  status, f"Schedule should have passed validation but did not. {message}"

        # GRB just before GRIncB
        optSchedule["GRB"] = [[8, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        # GRA just after GRIncA
        optSchedule["GRB"] = [[12, 13]]
        optSchedule["GRA"] = [[1, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        # GRB just after GRIncA
        optSchedule["GRA"] = [[10, 11]]
        optSchedule["GRB"] = [[1, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"


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

        sched = ScheduleInfo(1, self.num_vmfma, optSchedule, syncCode, None, None)
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        optSchedule["GRA"] = [[0, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert  not status, f"Schedule should have failed validation but did not. {message}"

        optSchedule["GRA"] = [[2, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        optSchedule["GRA"] = [[3, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert  status, f"Schedule should have passed validation but did not. {message}"

        optSchedule["GRB"] = [[6, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"


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

        sched = ScheduleInfo(1, self.num_vmfma, optSchedule, syncCode, None, None)
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        # GRA inside GRInc interval
        optSchedule["GRA"] = [[0, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        # GRA just after GRInc interval
        optSchedule["GRA"] = [[1, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        # GRB after GRIncB-10 
        optSchedule["GRB"] = [[10, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        # GRA before GRIncB-10 -> middle of interval
        optSchedule["GRA"] = [[10, 11]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

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

        sched = ScheduleInfo(1, self.num_vmfma, optSchedule, syncCode, None, None)
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        optSchedule["LWSA"] = [[0]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        optSchedule["LWSA"] = [[1]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        optSchedule["LWSA"] = [[2]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        optSchedule["LWSB"] = [[9]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        
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

        sched = ScheduleInfo(1, self.num_vmfma, optSchedule, syncCode, None, None)
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        optSchedule["GRIncB"] = [[0, 0, 1,
                                  9, 10,
                                  10]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert not status, f"Schedule should have failed validation but did not. {message}"

        optSchedule["GRIncB"] = [[1, 1, 2,
                            9, 10,
                            10]]
        status, message = verify_scc_overlap(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"



    