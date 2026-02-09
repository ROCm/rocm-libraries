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

from Tensile.Components.CMSValidator import verify_global_reads_not_too_early, get_most_recent_local_reads
from rocisa.instruction import SBarrier, SWaitCnt
from cms_validation_base import CMSValidationTestBase

class TestHelperFunctions:
    def test_get_most_recent_basic(self):
        """
        Test of utility function used to determine how many of the most recent ds_read
        operations are for A, and how many are for B.
        """

        def get_most_recent_local_reads_without_lr1(indices, counts, A, B, aBeforeB):
            """
            Assume that LRA0 appears before LRB0 within mfma index slots, and
            don't include LRA1 or LRB1 in the analysis.
            """
            positions = {"LRA1": -1, "LRB1": -1, "LRA0": 1 - aBeforeB, "LRB0": aBeforeB}

            localReads = [
                ("LRA0", A),
                ("LRB0", B),
                ("LRA1", []),
                ("LRB1", []),
            ]
            localReads.sort(key=lambda x: positions[x[0]])

            history = {}
            for symbol, values in localReads:
                for v in values:
                    if v not in history:
                        history[v] = []
                    history[v].append(symbol)
            history = sorted(history.items(), key=lambda t: t[0])

            mr = get_most_recent_local_reads(indices, counts, history)
            filtered = []
            for l in mr:
                filtered.append({"A": l["LRA0"], "B": l["LRB0"]})
            return filtered

        A = [0, 1, 4, 6, 8, 9]
        B = [2, 6, 7]
        #         0 1 2 3 4 5 6 7 8 9
        #         ===================
        # A       x x     x   x   x x
        # B           x       x x
        # index     ^
        #
        # Looking backwards from (and including) index=1, find the
        # counts=2 most recent appearances of A or B. How many
        # are A and how many are B?
        indices = [1]
        counts = [2]
        aBeforeB = True
        bBeforeA = not aBeforeB
        result = get_most_recent_local_reads_without_lr1(
            indices, counts, A, B, aBeforeB
        )
        expected0 = [{"A": 2, "B": 0}]
        assert result == expected0

        # A similar example, but with a different index and count.
        #         0 1 2 3 4 5 6 7 8 9
        #         ===================
        # A       x x     x   x   x x
        # B           x       x x
        # index               ^
        indices = [6]
        counts = [4]
        result = get_most_recent_local_reads_without_lr1(
            indices, counts, A, B, aBeforeB
        )
        expected1 = [{"A": 2, "B": 2}]
        assert result == expected1

        # Multiple indices and counts are handled independently
        indices = [1, 6]
        counts = [2, 4]
        result = get_most_recent_local_reads_without_lr1(
            indices, counts, A, B, aBeforeB
        )
        assert result == [expected0[0], expected1[0]]

        A = [1, 1, 1]
        B = [1, 1, 2]
        #    0 1 2 4 5 ...
        #    =============
        # A  0 3 0 0 0 ...
        # B  0 2 1 0 0 ....
        indices = [5, 5, 5]
        counts = [6, 2, 0]

        # index=5, count=6 : A:3 B:3
        #
        # index=5, count=2 : the most recent going back from index=5 is a
        # B at index 2. We need 1 more (count=2), but at index 1 there are
        # As and Bs. We use that A happens before B within the index (
        # and so as we're going in reverse chronological order, we take B).
        #
        # index=5, count=0: A:0 B:0
        result = get_most_recent_local_reads_without_lr1(
            indices, counts, A, B, aBeforeB
        )
        assert result == [{"A": 3, "B": 3}, {"A": 0, "B": 2}, {"A": 0, "B": 0}]

        # Changed so that B is before A.
        result = get_most_recent_local_reads_without_lr1(
            indices, counts, A, B, bBeforeA
        )
        assert result == [{"A": 3, "B": 3}, {"A": 1, "B": 1}, {"A": 0, "B": 0}]

    def test_get_most_recent(self):
        """
        Testing that within vmfma index ordering is correctly handled.
        """

        indices = [10, 10, 10, 10]
        counts = [1, 2, 3, 4]

        history = [[7, ["LRA0", "LRA1", "LRB0", "LRB1"]]]
        foo = get_most_recent_local_reads(indices, counts, history)
        # counts = 1
        assert foo[0] == {"LRA0": 0, "LRB0": 0, "LRA1": 0, "LRB1": 1}
        # counts = 2
        assert foo[1] == {"LRA0": 0, "LRB0": 1, "LRA1": 0, "LRB1": 1}
        # counts = 3
        assert foo[2] == {"LRA0": 0, "LRB0": 1, "LRA1": 1, "LRB1": 1}
        # counts = 4
        assert foo[3] == {"LRA0": 1, "LRB0": 1, "LRA1": 1, "LRB1": 1}

        history = [[7, ["LRB1", "LRB0", "LRA1", "LRA0"]]]
        foo = get_most_recent_local_reads(indices, counts, history)
        assert foo[0] == {"LRA0": 1, "LRB0": 0, "LRA1": 0, "LRB1": 0}
        assert foo[1] == {"LRA0": 1, "LRB0": 0, "LRA1": 1, "LRB1": 0}
        assert foo[2] == {"LRA0": 1, "LRB0": 1, "LRA1": 1, "LRB1": 0}
        assert foo[3] == {"LRA0": 1, "LRB0": 1, "LRA1": 1, "LRB1": 1}

class TestValidateGlobalReadsNotTooEarly(CMSValidationTestBase):
    def validation_function(self, sched, kernel_dict, codePathIdx):
        return verify_global_reads_not_too_early(sched, kernel_dict, codePathIdx)

    def setUp(self):
        super().setUp()
        self.kernel["DirectToLds"] = False

    def test_basic(self):
        """
        LRA0 at 0, LRB0 at 1. SWaitCnt(dscnt=0) at 3, SBarrier at 4.
        GRA load at 5, GRB load at 6. All safe.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4]],
            "GRA": [[5]],
            "LRA0": [[0]],
            "GRB": [[6]],
            "LRB0": [[1]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_lda_before_ldb_so_gra_safe(self):
        """
        LRA0 appears before LRB0 in the schedule dict.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 1, 5, 5]],
            "GRA": [[2]],
            "LRA0": [[0]],
            "GRB": [[7]],
            "LRB0": [[0]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_negative_lda_before_ldb_so_grb_unsafe(self):
        """
        LRA0 appears before LRB0 in the schedule dict, so the waitcnt at index 1 completes LRA0 but not LRB0.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 1, 5, 5]],
            "GRA": [[7]],
            "LRA0": [[0]],
            "GRB": [[2]],
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
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:0. "
            "First global read for B issued at vmfma_index:2. "
            "1 waitcnt operation(s) in [1, 3) provide upper bounds on the number of outstanding LRB0 operations: [1] <-- none of these is 0."
        )

    def test_interleave_gr_and_lrs(self):
        """
        This is like the preceding test, but now the waitcnt 1 is for an unrelated ds_load
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 1, 5, 5]],
            "GRA": [[6]],
            "LRA0": [[0]],
            "GRB": [[2]],
            "LRB0": [[0]],
            "LRB1": [[0]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_different_simd_codes(self):
        """
        Multiple code paths. Path 0 has LRA0 at 0, path 1 has it at 2.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4]],
            "LRA0": [[0], [2]],
            "LRB0": [[1]],
            "GRA": [[5]],
            "GRB": [[6], [7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier()
            ]
        self.validate(optSchedule, syncCode, 2, None, None, 0, None)

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
            "GRA": [[5]],
            "GRB": [[6], [7]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]

        # Code path 0 should pass
        self.validate(optSchedule, syncCode, 2, None, None, 0, None)

        # Code path 1 should fail: LRA0 at 4, SWaitCnt at 3 doesn't cover it.
        # LRA0's guaranteed_by stays inf until the SWaitCnt(dscnt=0) in ML loop which makes done_idx
        # very high. GRA's issued_at < must_start_after.done_idx() -> too early.
        self.validate(
            optSchedule, syncCode, 2, None, None, 1,
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:4. "
            "First global read for A issued at vmfma_index:5. "
            "0 waitcnt operation(s) in [5, 6) provide upper bounds on the number of outstanding LRA0 operations: [] <-- none of these is 0."
        )

    def test_negative_b_too_early(self):
        """
        SWaitCnt(dscnt=1) at 3 leaves 1 outstanding. LRA0 comes first,
        so LRB0 is still pending. GRB load at 6 starts too early.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4]],
            "GRA": [[5]],
            "LRA0": [[0]],
            "GRB": [[6]],
            "LRB0": [[1]],
        }
        syncCode = [SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:1. "
            "First global read for B issued at vmfma_index:6. "
            "1 waitcnt operation(s) in [2, 7) provide upper bounds on the number of outstanding LRB0 operations: [1] <-- none of these is 0."
        )

    def test_negative_barrier_before_swait(self):
        """
        SBarrier at 2 comes before SWaitCnt(dscnt=0) at 3.
        No barrier exists between LR0 done (at SWaitCnt=3) and GRA load at 5.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[2, 3]],
            "GRA": [[5]],
            "LRA0": [[0]],
            "GRB": [[6]],
            "LRB0": [[1]],
        }
        syncCode = [SBarrier(comment=""), SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that a barrier (to sync waves) exists between completion of local reads for A and the first global read for A. "
            "Last local read of A issued at vmfma_index 0, first global read of A issued at vmfma_index 5, wave completion at vmfma_index 3. "
            "Expected a barrier in the range [3, 6)."
        )

    def test_interleave_separate_pairs(self):
        """
        Separate SWaitCnt+SBarrier pairs: dscnt=1 at 2 guarantees LRA0. GRA at 4 safe.
        dscnt=0 at 5 guarantees LRB0. GRB at 7 safe.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[2, 3, 5, 6]],
            "GRA": [[4]],
            "LRA0": [[0]],
            "GRB": [[7]],
            "LRB0": [[1]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_interleave_multiple_lr0s(self):
        """
        LRA0 at [0,2], LRB0 at [1,3]. SWaitCnt(dscnt=1) at 4 keeps 1 (LRB0(3)) outstanding.
        All LRA0 done. GRA load at 5 safe.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[4, 4, 6, 6]],
            "GRA": [[5]],
            "LRA0": [[0, 2]],
            "GRB": [[7]],
            "LRB0": [[1, 3]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_negative_a_too_early(self):
        """
        SWaitCnt(dscnt=2) at 3: 4 LR0s total, keeps 2 outstanding including LRA0(2).
        LRA0(2) is only guaranteed by the SWaitCnt(dscnt=0) at 6.
        GRA load at 5 is too early.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[3, 4, 6, 6]],
            "GRA": [[5]],
            "LRA0": [[0, 2]],
            "GRB": [[7]],
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
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:2. "
            "First global read for A issued at vmfma_index:5. "
            "1 waitcnt operation(s) in [3, 6) provide upper bounds on the number of outstanding LRA0 operations: [1] <-- none of these is 0."
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
            "GRA": [[1, 6]],
            "LRA0": [[0]],
            "GRB": [[5, 6]],
            "LRB0": [[4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_redundant_waitcnt(self):
        """
        Multiple redundant SWaitCnts. Only the dscnt=0 at index 3 matters.
        The old validator sees GRB's first entry (m0-update at 4) and LRB0 at 4,
        so it finds no waitcnt in range [5,5) for B.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 2, 3, 4, 5, 5, 5]],
            "GRA": [[5, 7]],
            "LRA0": [[0]],
            "GRB": [[7, 7]],
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
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_direct_to_lds_a(self):
        """
        When using direct to LDS, the instructions in GRA and GRB are of the form

        0: update the m0 register,
        1: issue buffer_load,
        2: update the m0 register,
        3: issue buffer_load

        etc. Therefore the instructions at the even indices can be ignored.
        In this test, the first GRA instruction, issued in mfma vmfma_index 3,
        is not a buffer_load, and so we don't need to ensure the LRA0 instructions
        are complete yet for it.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[5, 6]],
            "GRA": [[3, 7]],
            "LRA0": [[2]]
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]

        self.kernel["DirectToLdsA"] = True
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_direct_to_lds(self):
        """
        When using direct to LDS, the instructions in GRA and GRB are of the form

        0: update the m0 register,
        1: issue buffer_load,
        2: update the m0 register,
        3: issue buffer_load

        etc. Therefore the instructions at the even indices can be ignored.
        In this test, the first GRA instruction, issued in mfma vmfma_index 3,
        is not a buffer_load, and so we don't need to ensure the LRA0 instructions
        are complete yet for it.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[5, 6]],
            "GRA": [[3, 7]],
            "LRA0": [[2]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]

        self.kernel["DirectToLds"] = True
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_negative_direct_to_lds_b(self):
        optSchedule = {
            "SYNC": [[5, 6]],
            "GRB": [[3, 4]],
            "LRB0": [[2]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]

        self.kernel["DirectToLdsB"] = True
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:2. "
            "First global read for B issued at vmfma_index:4. "
            "0 waitcnt operation(s) in [3, 5) provide upper bounds on the number of outstanding LRB0 operations: [] <-- none of these is 0."
        )

    def test_swap_global_read_order(self):
        """
        SwapGlobalReadOrder: GRA loads B -> must start after LRB0,
        GRB loads A -> must start after LRA0.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 2, 5, 6]],
            "GRA": [[7]],
            "LRA0": [[0]],
            "GRB": [[3]],
            "LRB0": [[4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.kernel["SwapGlobalReadOrder"] = True
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_negative_swap_global_read_order(self):
        """
        SwapGlobalReadOrder: GRA must start after LRB0.
        LRB0 at 4, guaranteed by SWaitCnt(dscnt=0) at 5.
        GRA load at 3 is before LRB0 is guaranteed done (done at 5).
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "SYNC": [[1, 2, 5, 6]],
            "GRA": [[3]],
            "LRA0": [[0]],
            "GRB": [[7]],
            "LRB0": [[4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
        ]
        self.kernel["SwapGlobalReadOrder"] = True
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:4. "
            "First global read for B issued at vmfma_index:3. "
            "0 waitcnt operation(s) in [5, 4) provide upper bounds on the number of outstanding LRB0 operations: [] <-- none of these is 0."
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
            "GRA": [[4]],  # The read for A is safe, LRA0 appears before LRB0.
            "GRB": [[7]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier()
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:2. "
            "First global read for B issued at vmfma_index:7. "
            "1 waitcnt operation(s) in [2, 8) provide upper bounds on the number of outstanding LRB0 operations: [1] <-- none of these is 0."
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
            "GRA": [[4]],  # The read for A is NOT safe, LRA0 appears after LRB0.
            "GRB": [[7]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:2. "
            "First global read for A issued at vmfma_index:4. "
            "1 waitcnt operation(s) in [2, 5) provide upper bounds on the number of outstanding LRA0 operations: [1] <-- none of these is 0."
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
            "GRA": [[5]],
            "GRB": [[5]],
        }
        syncCode = [
            SBarrier(),
            SBarrier(),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that a barrier (to sync waves) exists between completion of local reads for A and the first global read for A. "
            "Last local read of A issued at vmfma_index 5, first global read of A issued at vmfma_index 5, wave completion at vmfma_index 5. "
            "Expected a barrier in the range [5, 6)."
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
            "GRA": [[7]],
            "GRB": [[5]],
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
            "Failed to verify that a barrier (to sync waves) exists between completion of local reads for B and the first global read for B. "
            "Last local read of B issued at vmfma_index 2, first global read of B issued at vmfma_index 5, wave completion at vmfma_index 5. "
            "Expected a barrier in the range [5, 6)."
        )

    def test_within_mfma_index_order_local_reads_before_syncs_before_global_read(self):
        """
        Within-index dict ordering: LR0 < SYNC < GR. At index 5, LR0 is issued,
        then SWaitCnt+SBarrier, then GR. Safe.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[5]],
            "LRB0": [[5]],
            "SYNC": [[5, 5]],
            "GRA": [[5]],
            "GRB": [[5]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_within_mfma_index_order_sync_after_global_read_for_b(self):
        """
        GRB appears before SYNC in dict. At index 5, GRB load fires before SWaitCnt/SBarrier.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[5]],
            "LRB0": [[5]],
            "GRB": [[5]],
            "SYNC": [[5,5]],
            "GRA": [[5]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:5. "
            "First global read for B issued at vmfma_index:5. "
            "0 waitcnt operation(s) in [5, 5) provide upper bounds on the number of outstanding LRB0 operations: [] <-- none of these is 0."
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
            "GRA": [[5]],
            "GRB": [[5]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:5. "
            "First global read for A issued at vmfma_index:5. "
            "0 waitcnt operation(s) in [6, 6) provide upper bounds on the number of outstanding LRA0 operations: [] <-- none of these is 0."
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
            "GRA": [[5]],
            "GRB": [[7]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:5. "
            "First global read for A issued at vmfma_index:5. "
            "0 waitcnt operation(s) in [6, 6) provide upper bounds on the number of outstanding LRA0 operations: [] <-- none of these is 0."
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
            "GRB": [[7]],
            "GRA": [[6]],
            "SYNC": [[5, 6]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that a barrier (to sync waves) exists between completion of local reads for A and the first global read for A. "
            "Last local read of A issued at vmfma_index 5, first global read of A issued at vmfma_index 6, wave completion at vmfma_index 5. "
            "Expected a barrier in the range [5, 6)."
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
            "GRA": [[5]],
            "GRB": [[7]],
        }
        syncCode = [
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

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
            "GRA": [[5]],
            "GRB": [[7]],
        }
        syncCode = [
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:3. "
            "First global read for A issued at vmfma_index:5. "
            "1 waitcnt operation(s) in [3, 6) provide upper bounds on the number of outstanding LRA0 operations: [1] <-- none of these is 0."
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
            "GRA": [[4]],
            "GRB": [[7]],
        }
        syncCode = [
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

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
            "GRA": [[4]],
        }
        syncCode = [
            # 3 LRA1 and 4 LRB1 can be outstanding.
            SWaitCnt(dscnt=3 + 4 + 1, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(),
        ]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:3. "
            "First global read for A issued at vmfma_index:4. "
            "1 waitcnt operation(s) in [3, 5) provide upper bounds on the number of outstanding LRA0 operations: [1] <-- none of these is 0."
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
            "GRA": [[4, 5, 6]],
            "GRB": [[7]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]
        self.validate(optSchedule, syncCode, 1, None, None, 0, None)

    def test_negative_one_gr_too_early(self):
        """
        First GRA load at 2 is before SWaitCnt at 3. GR issued before LR0 guaranteed.
        """
        assert self.num_vmfma == 8
        optSchedule = {
            "LRA0": [[0]],
            "LRB0": [[1]],
            "GRA": [[2, 6, 7]],
            "SYNC": [[3, 4]],
            "GRB": [[7]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier(comment="")]
        self.validate(
            optSchedule, syncCode, 1, None, None, 0,
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:0. "
            "First global read for A issued at vmfma_index:2. "
            "0 waitcnt operation(s) in [0, 2) provide upper bounds on the number of outstanding LRA0 operations: [] <-- none of these is 0."
        )
