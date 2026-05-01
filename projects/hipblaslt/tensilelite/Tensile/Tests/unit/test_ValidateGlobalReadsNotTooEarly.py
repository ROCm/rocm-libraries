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

from typing import Any, Optional
from Tensile.Components.CMSValidator import add_gr_not_too_early_constraints
from Tensile.Components.ScheduleCapture import (
    OrderInvertedFailure, MissingBarrierFailure,
)
from rocisa.instruction import SBarrier, SWaitCnt
from cms_validation_base import CMSValidationTestBase

class TestValidateGlobalReadsNotTooEarly(CMSValidationTestBase):
    """
    Tests for the Timeline-based verify_grs_not_too_early validation.

    num_vmfma = 8 for the default kernel (MIWaveTileA=2, MIWaveTileB=2, DepthU=64, MI[2]=32).
    Valid vmfma indices are [-1, 7].

    Since the Timeline class requires DirectToLds=True, GR schedules must contain
    paired entries: even indices are m0-pointer-updates (ignored), odd indices are actual
    buffer_loads. For example, GRA: [[3, 5]] means m0-update at 3, load at 5.
    """
    validator_passes = [add_gr_not_too_early_constraints]

    def test_basic(self):
        """
        LRA0 at 0, LRB0 at 1. SWaitCnt(dscnt=0) at 3, SBarrier at 4.
        GRA load at 5, GRB load at 6. All safe.
        """
        assert self.num_vmfma == 8
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4]],
            "GRA": [[5, 5]],
            "LRA0": [[0]],
            "GRB": [[6, 6]],
            "LRB0": [[1]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_lda_before_ldb_so_gra_safe(self):
        """
        LRA0 appears before LRB0 in the schedule dict.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 1, 5, 5]],
            "GRA": [[2, 2]],
            "LRA0": [[0]],
            "GRB": [[7, 7]],
            "LRB0": [[0]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_lda_before_ldb_so_grb_unsafe(self):
        """
        LRA0 appears before LRB0 in the schedule dict, so the waitcnt at index 1 completes LRA0 but not LRB0.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 1, 5, 5]],
            "GRA": [[7, 7]],
            "LRA0": [[0]],
            "GRB": [[2, 2]],
            "LRB0": [[0]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_different_simd_codes(self):
        """
        Multiple code paths. Path 0 has LRA0 at 0, path 1 has it at 2.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4]],
            "LRA0": [[0], [2]],
            "LRB0": [[1]],
            "GRA": [[5, 5]],
            "GRB": [[6, 6], [7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier()
            ]
        self.validate(optSchedule, syncCode, 2, None, None, 0)

    def test_negative_different_simd_codes(self):
        """
        Code path 1 has LRA0 at 4. SWaitCnt at 3 fires before LRA0 is issued so
        LRA0 is not guaranteed done. GRA load at 5 starts too early.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4]],
            "LRA0": [[0], [4]],  # Read for codepath 1 is too late.
            "LRB0": [[1]],
            "GRA": [[5, 5]],
            "GRB": [[6, 6], [7, 7]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]

        # Code path 0 should pass
        self.validate(optSchedule, syncCode, 2, None, None, 0)

        # Code path 1 should fail: LRA0 at 4, SWaitCnt at 3 doesn't cover it.
        # LRA0's guaranteed_by stays inf until the SWaitCnt(dscnt=0) in ML loop which makes done_idx
        # very high. GRA's issued_at < must_start_after.done_idx() -> too early.
        self.validate(
            optSchedule, syncCode, 2, None, None, 1,
            expected_failure=OrderInvertedFailure,
        )

    def test_negative_b_too_early(self):
        """
        SWaitCnt(dscnt=1) at 3 leaves 1 outstanding. LRA0 comes first,
        so LRB0 is still pending. GRB load at 6 starts too early.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4]],
            "GRA": [[5, 5]],
            "LRA0": [[0]],
            "GRB": [[6, 6]],
            "LRB0": [[1]],
        }
        syncCode = [SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_negative_barrier_before_swait(self):
        """
        SBarrier at 2 comes before SWaitCnt(dscnt=0) at 3.
        No barrier exists between LR0 done (at SWaitCnt=3) and GRA load at 5.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[2, 3]],
            "GRA": [[5, 5]],
            "LRA0": [[0]],
            "GRB": [[6, 6]],
            "LRB0": [[1]],
        }
        syncCode = [SBarrier(comment=""), SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=MissingBarrierFailure,
        )

    def test_interleave_separate_pairs(self):
        """
        Separate SWaitCnt+SBarrier pairs: dscnt=1 at 2 guarantees LRA0. GRA at 4 safe.
        dscnt=0 at 5 guarantees LRB0. GRB at 7 safe.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[2, 3, 5, 6]],
            "GRA": [[4, 4]],
            "LRA0": [[0]],
            "GRB": [[7, 7]],
            "LRB0": [[1]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_interleave_multiple_lr0s(self):
        """
        LRA0 at [0,2], LRB0 at [1,3]. SWaitCnt(dscnt=1) at 4 keeps 1 (LRB0(3)) outstanding.
        All LRA0 done. GRA load at 5 safe.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[4, 4, 6, 6]],
            "GRA": [[5, 5]],
            "LRA0": [[0, 2]],
            "GRB": [[7, 7]],
            "LRB0": [[1, 3]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_a_too_early(self):
        """
        SWaitCnt(dscnt=2) at 3: 4 LR0s total, keeps 2 outstanding including LRA0(2).
        LRA0(2) is only guaranteed by the SWaitCnt(dscnt=0) at 6.
        GRA load at 5 is too early.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4, 6, 6]],
            "GRA": [[5, 5]],
            "LRA0": [[0, 2]],
            "GRB": [[7, 7]],
            "LRB0": [[1, 3]],
        }
        syncCode = [
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_sync_and_gr_at_same_index(self):
        """
        We assume an ordering: s_waitcnt < s_barrier < buffer_load (within a vmfma_index)
        For A: lra0 at 0, waitcnt at 1, barrier at 1, gra at 1
        For B: lrb0 at 4, barrier at 5, grb at 5, waitcnt at 5
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 1, 5, 5]],
            "GRA": [[1, 1, 6, 6]],
            "LRA0": [[0]],
            "GRB": [[5, 5, 6, 6]],
            "LRB0": [[4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_redundant_waitcnt(self):
        """
        Multiple redundant SWaitCnts. Only the dscnt=0 at index 3 matters.
        The old validator sees GRB's first entry (m0-update at 4) and LRB0 at 4,
        so it finds no waitcnt in range [5,5) for B.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 2, 3, 4, 5, 5, 5]],
            "GRA": [[5, 5, 7, 7]],
            "LRA0": [[0]],
            "GRB": [[7, 7, 7, 7]],
            "LRB0": [[4]],
        }
        syncCode = [
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),  # <-- the useful waitcnt for A
            SBarrier(comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_swap_global_read_order(self):
        """
        SwapGlobalReadOrder: GRA loads B -> must start after LRB0,
        GRB loads A -> must start after LRA0.
        """
        assert self.num_vmfma == 8
        self.kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            "SYNC": [[1, 2, 5, 6]],
            "GRA": [[6, 7]],
            "LRA0": [[0]],
            "GRB": [[2, 3]],
            "LRB0": [[4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_swap_global_read_order(self):
        """
        SwapGlobalReadOrder: GRA must start after LRB0.
        LRB0 at 4, guaranteed by SWaitCnt(dscnt=0) at 5.
        GRA load at 3 is before LRB0 is guaranteed done (done at 5).
        """
        assert self.num_vmfma == 8
        self.kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            "SYNC": [[1, 2, 5, 6]],
            "GRA": [[3, 3]],
            "LRA0": [[0]],
            "GRB": [[7, 7]],
            "LRB0": [[4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_ab_tiebreaking_lra0_before_lrb0(self):
        """
        Check that we correctly handle order of LR0s ordered at the same index.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[2]],
            "LRB0": [[2]],
            "SYNC": [[3, 3]],
            "GRA": [[4,4]],  # The read for A is safe, LRA0 appears before LRB0.
            "GRB": [[7,7]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier()
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_ab_tiebreaking_lrb0_before_lra0(self):
        """
        Same as above, but with LRB0 before LRA0.
        dscnt=1 at index 3 clears the most recent (LRB0, since LRA0 appears after LRB0),
        so LRA0 is still outstanding. GRA at index 4 is issued too early.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRB0": [[2]],
            "LRA0": [[2]],
            "SYNC": [[3, 3]],
            "GRA": [[4,4]],  # The read for A is NOT safe, LRA0 appears after LRB0.
            "GRB": [[7,7]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_waitcnt_barrier_relative_order_barrier_too_late_for_a(self):
        """
        The barrier at index 6 is too late for A.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[5]],
            "LRB0": [[5]],
            # The first barrier after the waitcnt is at index 6. Too late.
            "SYNC": [[1, 5, 5, 5, 6]],
            "GRA": [[5,5]],
            "GRB": [[5,5]],
        }
        syncCode = [
            SBarrier(),
            SBarrier(),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0,
            expected_failure=MissingBarrierFailure,
        )

    def test_waitcnt_barrier_relative_order_barrier_too_late_for_b(self):
        """
        The barrier at index 6 is too late for B.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[1]],
            "LRB0": [[2]],
            # The first barrier after the required waitcnt is at index 6. Too late.
            "SYNC": [[5, 5, 5, 5, 6]],
            "GRA": [[7,7]],
            "GRB": [[5,5]],
        }
        syncCode = [
            SBarrier(),
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=MissingBarrierFailure,
        )

    def test_within_mfma_index_order_local_reads_before_syncs_before_global_read(self):
        """
        Within-index dict ordering: LR0 < SYNC < GR. At index 3, LR0 is issued,
        then SWaitCnt+SBarrier, then GR. Safe.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[5]],
            "LRB0": [[5]],
            "SYNC": [[5, 5]],
            "GRA": [[5,5]],
            "GRB": [[5,5]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_within_mfma_index_order_sync_after_global_read_for_b(self):
        """
        GRB appears before SYNC in dict. At index 5, GRB load fires before SWaitCnt/SBarrier.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[5]],
            "LRB0": [[5]],
            "GRB": [[5,5]],
            "SYNC": [[5,5]],
            "GRA": [[5,5]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_within_mfma_index_order_sync_before_local_read_for_a(self):
        """
        SYNC appears before LRA0 in dict. At index 5, SWaitCnt fires before LRA0.
        LRA0 is not covered by the wait.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRB0": [[5]],
            "SYNC": [[5,5]],
            "LRA0": [[5]],
            "GRA": [[5,5]],
            "GRB": [[5,5]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_within_mfma_index_order_no_sync_for_global_read_a(self):
        """
        The order in which instructions are added to optSchedule dictates
        the order in which instructions are added at each index.
        no sync for global read A.
        """
        optSchedule = {
            "SYNC": [[5, 5, 6, 6]],
            "LRB0": [[5]],
            "LRA0": [[5]],
            "GRA": [[5,5]],
            "GRB": [[7,7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_within_mfma_index_order_sync_after_gra_too_late(self):
        """
        The order in which instructions are added to optSchedule dictates
        the order in which instructions are added at each index.
        In this case, because "SYNC" appears after "GRA", the barrier at index 6 comes after
        the GRA: too late.
        """
        optSchedule = {
            "LRA0": [[5]],
            "LRB0": [[5]],
            "GRB": [[7,7]],
            "GRA": [[6,6]],
            "SYNC": [[5, 6]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=MissingBarrierFailure,
        )

    def test_on_the_edge(self):
        """
        LRA0 at [0,1,2,3], LRB0 at [2,3,4,5]. SWaitCnt(dscnt=2) at 4: keeps 2
        outstanding (LRB0(4), LRB0(5)). All LRA0 done. GRA at 5 safe.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0, 1, 2, 3]],
            "LRB0": [[2, 3, 4, 5]],
            "SYNC": [[4, 4, 6, 6]],
            "GRA": [[5, 5]],
            "GRB": [[7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_on_the_edge_tipped_by_a(self):
        """
        SWaitCnt(dscnt=3) at 4: keeps 3 outstanding, including LRA0(3).
        LRA0(3) is only guaranteed by the SWaitCnt(dscnt=0) at 6.
        GRA load at 5 starts before LRA0 guaranteed done.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0, 1, 2, 3]],
            "LRB0": [[2, 3, 4, 5]],
            "SYNC": [[4, 4, 6, 6]],
            "GRA": [[5, 5]],
            "GRB": [[7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_on_the_edge_tipped_by_a_saved_by_lr1(self):
        """
        LRA0 at [0,3], LRA1 at [3,4], LRB0 at [3,4]. SWaitCnt(dscnt=4) at 4: keeps 4
        outstanding (LRA1(3), LRA1(4), LRB0(3), LRB0(4)). All LRA0 done. GRA at 4 safe.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0, 3]],
            "LRA1": [[3, 4]],
            "LRB0": [[3, 4]],
            "SYNC": [[4,4, 6,6]],
            "GRA": [[4,4]],
            "GRB": [[7,7]],
        }
        syncCode = [
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_lr1_in_the_middle(self):
        """
        Check that incorrect counts caused by adding LR1s is caught.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [2 * [3]],
            "LRA1": [3 * [3]],
            "LRB1": [4 * [3]],
            "SYNC": [[3, 3]],
            "GRA": [[4,4]],
        }
        syncCode = [
            # 3 LRA1 and 4 LRB1 can be outstanding.
            SWaitCnt(dscnt=3 + 4 + 1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_multiple_grs_all_safe(self):
        """
        Multiple GRA loads at different indices, all after LRA0 guaranteed.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[2, 3]],
            "GRA": [[3, 4, 4, 5, 5, 6]],
            "GRB": [[6, 7]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_one_gr_too_early(self):
        """
        First GRA load at 2 is before SWaitCnt at 3. GR issued before LR0 guaranteed.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "GRA": [[1, 2, 5, 6, 6, 7]],
            "SYNC": [[3, 4]],
            "GRB": [[6, 7]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_negative_lrb0_not_guaranteed_by_waitcnt_at_same_index_as_lra0(self):
        """
        Exercises a bug in get_most_recent_local_reads where a local read (LRA0)
        issued at the same vmfma index as a SWaitCnt -- but AFTER it in within-index
        ordering -- is incorrectly counted as a "recent" read by the SWaitCnt.

        Setup (modelled after 256x160x64 TN code path 1):
          - LRB0 issued at vmfma [0, 1]  (2 reads, all before the sync)
          - LRA0 issued at vmfma [2]    (1 read, same index as the sync)
          - SYNC at vmfma 2: SWaitCnt(dscnt=1) then SBarrier
          - GRB starts at vmfma 3

        Dict key order: SYNC < LRB0 < LRA0 < GRB, so within vmfma 14 the
        execution order is:  SWaitCnt -> SBarrier -> (LRB0: nothing) -> LRA0 -> ...

        At the moment the SWaitCnt(dscnt=1) fires, only 2 LRB0 reads are outstanding
        (LRA0 at vmfma 2 hasn't been issued yet).  After dscnt=1, at most 1 remains,
        and that 1 is a LRB0.  GRB at vmfma 3 therefore starts while 1 LRB0 may
        still be in flight -- a data hazard.

        The validator used to incorrectly pass because get_most_recent_local_reads saw
        LRA0 at vmfma 3 in its history and attributes the 1 outstanding read to it.
        (reporting LRB0: 0 outstanding).
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[-1, 2, 2]],
            "LRB0": [[0, 1]],
            "LRA0": [[2]],
            "GRB":  [[3, 3]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="pre-loop wait"),
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 -- but leaves 1 outstanding"),
            SBarrier(comment=""),
        ]
        # Should FAIL: dscnt=1 leaves 1 LRB0 outstanding when GRB starts at 15.
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )


class TestValidateGlobalReadsNotTooEarlyDtlPlusLdsBuf(CMSValidationTestBase):
    """
    Tests for DtlPlusLdsBuf=True support in add_gr_not_too_early_constraints.

    With DtlPlusLdsBuf=True, GRs in iteration N write to a different LDS block than
    same-iteration LR0s, so there is no same-iteration dependency. Instead, GRs depend
    on the *previous* iteration's LR0s (cross-iteration dependency).
    """
    validator_passes = [add_gr_not_too_early_constraints]

    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None):
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update({"DtlPlusLdsBuf": True})
        super().setup_method(method, kernel_updates=kernel_updates)

    def test_basic(self):
        """
        Same schedule as the standard test_basic. Should pass.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "SYNC": [[3, 3]],
            "GRA": [[5, 5]],
            "GRB": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment="")
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_gr_before_same_iter_sync_safe(self):
        """
        GRA load at 2 is before the SWaitCnt at 3 in the same iteration.
        Without DtlPlusLdsBuf this would fail (same-iteration LRA0's guaranteed_by=3 > GRA=2).
        With DtlPlusLdsBuf, GRA depends on ML-1's LRA0, which is guaranteed by ML-1's
        SWaitCnt at (loop=0, vmfma=3). GRA at (loop=1, vmfma=2) > (loop=0, vmfma=3).
        ML-1's SBarrier at (loop=0, vmfma=4) provides the required barrier.
        
        Should pass.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "GRA": [[1, 2]],
            "SYNC": [[3, 4]],
            "GRB": [[6, 6]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_prev_iter_lr0_not_guaranteed(self):
        """
        ML-1's LRA0 is at index 5, AFTER ML-1's SWaitCnt(dscnt=0) at index 3.
        So ML-1's LRA0 is NOT guaranteed within ML-1. It's guaranteed by ML's SWaitCnt
        at (loop=1, vmfma=3). ML's GRA at (loop=1, vmfma=2) < (loop=1, vmfma=3) -> too early.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4]],
            "LRA0": [[5]],
            "LRB0": [[1]],
            "GRA": [[2, 2]],
            "GRB": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment="")
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_swap_global_read_order(self):
        """
        SwapGlobalReadOrder + DtlPlusLdsBuf: GRA depends on prev LRB0, GRB depends on prev LRA0.
        Both are guaranteed within ML-1. Should pass.
        """
        assert self.num_vmfma == 8
        self.kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            "SYNC": [[1,2, 5, 6]],
            "GRA": [[6, 7]],
            "LRA0": [[0]],
            "GRB": [[2, 3]],
            "LRB0": [[4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_swap_global_read_order(self):
        """
        SwapGlobalReadOrder + DtlPlusLdsBuf: GRA depends on prev LRB0 (swapped).
        ML-1's LRB0 is at index 5, AFTER ML-1's SWaitCnt at 3. So ML-1's LRB0
        is guaranteed by ML's SWaitCnt at (loop=1, vmfma=3).
        GRA at (loop=1, vmfma=2) < (loop=1, vmfma=3) -> too early.
        """
        assert self.num_vmfma == 8
        self.kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[5]],
            "SYNC": [[3, 4]],
            "GRA": [[2, 2]],
            "GRB": [[7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment="")
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )

    def test_multiple_lr0s(self):
        """
        Multiple LRA0s in previous iteration, all guaranteed by ML-1's SWaitCnt(dscnt=0).
        max(guaranteed_by) selects the LR0 with the highest guaranteed_by. Both are at (0,3).
        GRA at (1,5) > (0,3). Should pass.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0, 2]],
            "LRB0": [[1]],
            "SYNC": [[3, 4]],
            "GRA": [[5, 5]],
            "GRB": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment="")
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0)

    def test_negative_multiple_lr0s_one_not_guaranteed(self):
        """
        Two LRA0s in previous iteration: one at 0 (guaranteed by ML-1's SWaitCnt at 3)
        and one at 5 (NOT guaranteed within ML-1, guaranteed by ML's SWaitCnt at 3).
        max(guaranteed_by) = (loop=1, vmfma=3). GRA at (1,2) < (1,3) -> too early.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4]],
            "LRA0": [[0, 5]],
            "LRB0": [[1]],
            "GRA": [[2, 2]],
            "GRB": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment="")
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            expected_failure=OrderInvertedFailure,
        )
