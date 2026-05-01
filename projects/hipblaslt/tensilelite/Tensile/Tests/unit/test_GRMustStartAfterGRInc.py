################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

from Tensile.Components.CMSValidator import add_gr_not_too_early_constraints
from Tensile.Components.ScheduleCapture import (
    OrderInvertedFailure, MissingBarrierFailure,
)
from rocisa.instruction import SBarrier, SWaitCnt
from cms_validation_base import CMSValidationTestBase


class TestGRMustStartAfterGRInc(CMSValidationTestBase):
    """
    Tests for the GRInc -> GR timeline-based constraint integrated into
    add_gr_not_too_early_constraints.

    num_vmfma = 8 for the default kernel (MIWaveTileA=2, MIWaveTileB=2,
    DepthU=64, MI[2]=32).  Valid vmfma indices are [-1, 7].

    GR schedules use paired entries for DirectToLds: even indices are
    m0-pointer-updates (ignored), odd indices are actual buffer_loads.
    """
    validator_passes = [add_gr_not_too_early_constraints]

    def test_basic_grinc_before_gr(self):
        """
        GRIncA finishes well before GRA.
        LR0 + barrier present. Pass.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 4]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 4]],
            "GRA": [[5, 5]],
            "GRB": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_gr_before_grinc(self):
        """
        GRA issued before last GRIncA finishes.
        Fail: "issued too early... GRIncA is guaranteed done."
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRA": [[4, 4]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 5]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 4]],
            "GRB": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        f = self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )
        self.assert_order_inverted(
            f, producer_name="GRIncA", producer_idx=5,
            consumer_name="GRA", consumer_idx=4,
        )

    def test_grinc_tighter_than_lr0_with_barrier(self):
        """
        Key test: Last GRIncA at idx 5 (after LR0 guaranteed at idx 3).
        SBarrier at idx 4 (between LR0 done and GR). GRA at idx 7. Pass:
        both constraints satisfied.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[3, 4]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 5]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 4]],
            "GRA": [[7, 7]],
            "GRB": [[7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_grinc_tighter_but_no_barrier(self):
        """
        Key test: Same as above but no SBarrier between LR0 guaranteed_by and GR.
        GRInc ordering passes, but LR0 barrier fails.
        Fail: "SBarrier missing..."
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[3]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 5]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 4]],
            "GRA": [[7, 7]],
            "GRB": [[7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=MissingBarrierFailure,
        )

    def test_swap_grinc_before_gr(self):
        """
        SwapGlobalReadOrder: GRA needs GRIncB (not GRIncA). Pass.
        """
        assert self.num_vmfma == 8
        self.kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 2, 5, 5]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 3]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 3]],
            "GRA": [[6, 6]],
            "GRB": [[3, 3]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_swap_wrong_pairing(self):
        """
        Swap: GRA starts before GRIncB finishes. Fail.
        GRA loads B so needs GRIncB. GRIncB last at 5, GRA at 4.
        """
        assert self.num_vmfma == 8
        self.kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 3]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 5]],
            "GRA": [[4, 4]],
            "GRB": [[7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        f = self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )
        self.assert_order_inverted(
            f, producer_name="GRIncB", producer_idx=5,
            consumer_name="GRA", consumer_idx=4,
        )

    def test_both_operands(self):
        """
        GRIncA/GRA and GRIncB/GRB both present and valid. Pass.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 4]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 4]],
            "GRA": [[5, 5]],
            "GRB": [[5, 5]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_b_before_grinc_b(self):
        """
        GRB starts before GRIncB finishes. Fail.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 4]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 6]],
            "GRA": [[5, 5]],
            "GRB": [[5, 5]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        f = self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )
        self.assert_order_inverted(
            f, producer_name="GRIncB", producer_idx=6,
            consumer_name="GRB", consumer_idx=5,
        )

    def test_no_grinc_in_schedule(self):
        """
        GRA present but no GRIncA. GRInc constraint inactive.
        Pass (only LR0 checked).
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRA": [[5, 5]],
            "GRB": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_grinc_and_gr_same_index_grinc_declared_first(self):
        """
        Same vmfma_index, GRInc declared before GR in dict.
        Pass (sub_index ordering: GRInc gets lower sub_index).
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 5]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 4]],
            "GRA": [[5, 5]],
            "GRB": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_grinc_and_gr_same_index_grinc_declared_after(self):
        """
        Same vmfma_index, GRInc declared after GR in dict.
        Fail (sub_index ordering: GR gets lower sub_index than GRInc).
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRA": [[5, 5]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 5]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 4]],
            "GRB": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        f = self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )
        self.assert_order_inverted(
            f, producer_name="GRIncA", producer_idx=5,
            consumer_name="GRA", consumer_idx=5,
        )

    def test_multiple_grinc_instructions(self):
        """
        GRIncA at [1,1,2,2,3,3,4,5,5]. GRA at 7. Pass.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 4, 5, 5]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 4, 5, 5]],
            "GRA": [[7, 7]],
            "GRB": [[7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_swap_both_operands_cross_pairing(self):
        """
        Swap: verify GRIncA->GRB and GRIncB->GRA cross-pairing.
        Both valid. Pass.
        """
        assert self.num_vmfma == 8
        self.kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 2, 5, 5]],
            "GRIncA": [[1, 1, 2, 2, 3, 3, 4]],
            "GRIncB": [[1, 1, 2, 2, 3, 3, 3]],
            "GRB": [[5, 5]],
            "GRA": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)
