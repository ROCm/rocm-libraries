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

import pytest

from Tensile.Components.CustomSchedule import (
    ScheduleInfo,
    verify_global_reads_not_too_early,
    get_most_recent_local_reads,
)

from rocisa.instruction import SBarrier, SWaitCnt


def getSyncsAndCodes(waitcnts: dict, barriers: list):
    """
    Utility for unit tests.

    waitcnts holds data on SWaitCnt instructions in the schedule. The
    key is the vmfma index where the instruction is found and the value
    is the dscnt of the instruction.

    Example:
    input:   waitcnts = {3:0, 5:1} barriers = [4]
    output:  syncs = [3, 4, 5] codes = [waitcnt(0), barrier, waitcnt(1)]
    """
    combined = []
    for k, v in waitcnts.items():
        combined.append((k, SWaitCnt(dscnt=v, vlcnt=-1, vscnt=-1, comment="")))
    for b in barriers:
        combined.append((b, SBarrier(comment="")))
    # sort by key to get the right order
    combined.sort(key=lambda x: x[0])
    syncs = [[]]
    codes = []
    for s, c in combined:
        syncs[0].append(s)
        codes.append(c)

    return syncs, codes


def scheduleFromSequence(seq: list):
    """
    Utility for unit tests. Constructs a schedule from a sequence description.

    Example input:
    [('lra0', 0), ('lrb0', 1), ('waitcnt', 1, 2), ('barrier', 3), ('gra', 5), ('grb', 10)]

    Each element in the list above is of the form (what, when). Interleaving the different
    types of instructions like this sometimes makes it easier to quickly see what the
    schedule is doing.

    Currently only supports single SIMD, multi-SIMD tests must use the default description.
    """

    counts = {
        "gra": [],
        "grb": [],
        "lra0": [],
        "lrb0": [],
        "lra1": [],
        "lrb1": [],
        "waitcnt": {},
        "barrier": [],
    }

    for item in seq:
        kind = item[0]
        if kind not in counts:
            raise ValueError(f"Unknown item type: {kind}")
        if kind == "waitcnt":
            counts[kind][item[2]] = item[1]
        else:
            counts[kind].append(item[1])

    syncs, syncCode = getSyncsAndCodes(counts["waitcnt"], counts["barrier"])

    # The order of the key:value pairs here matters! It ensures that operations for A happen before operations for B, within an index.
    optSchedule = {
        "SYNC": syncs,
        "GRA": [counts["gra"]],
        "LRA0": [counts["lra0"]],
        "LRA1": [counts["lra1"]],
        "GRB": [counts["grb"]],
        "LRB0": [counts["lrb0"]],
        "LRB1": [counts["lrb1"]],
    }
    numCodePaths = 1
    return ScheduleInfo(numCodePaths, None, optSchedule, syncCode, None, None, None)


class TestCustomScheduleGlobalReadValidation:

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

    def test_basic(self):
        schedule = scheduleFromSequence(
            [
                (
                    "lra0",
                    0,
                ),  # local read of A, at vmfma_index=0 (between mfma 0 and mfma 1)
                ("lrb0", 1),  # local read of B, at vmfma_index=1
                (
                    "waitcnt",
                    0,
                    5,
                ),  # at vmfma_index=5, wait for count=0 (all local reads of wave, done)
                ("barrier", 6),  # barrier, ensures all waves synced.
                ("gra", 10),
                ("grb", 11),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_basic_native_format(self):
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "SYNC": [[5, 6]],
                "LRA0": [[0]],
                "LRB0": [[1]],
                "GRA": [[10]],
                "GRB": [[11]],
            },
            [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_lda_before_ldb_so_gra_safe(self):
        """
        scheduleFromSequence is implemented with LRA0 before LDB0.
        """
        schedule = scheduleFromSequence(
            [
                ("lrb0", 0),
                ("lra0", 0),
                ("waitcnt", 1, 1),
                ("barrier", 1),
                ("gra", 2),
                ("waitcnt", 0, 9),
                ("barrier", 9),
                ("grb", 10),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_negative_lda_before_ldb_so_grb_unsafe(self):
        """
        scheduleFromSequence is implemented with LRA0 after LRB0.
        """
        schedule = scheduleFromSequence(
            [
                ("lrb0", 0),
                ("lra0", 0),
                ("waitcnt", 1, 1),
                ("barrier", 1),
                ("grb", 2),
                ("waitcnt", 0, 9),
                ("barrier", 9),
                ("gra", 10),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:0. "
            "First global read for B issued at vmfma_index:2. "
            "1 waitcnt operation(s) in [1, 3) provide upper bounds on the number of outstanding LRB0 operations: [1] <-- none of these is 0."
        )
        assert status is False

    def test_interleave_gr_and_lrs(self):
        """
        This is like the preceding test, but now the waitcnt 1 is for an unrelated ds_load
        """
        schedule = scheduleFromSequence(
            [
                ("lrb0", 0),
                ("lra0", 0),
                ("lrb1", 0),
                ("waitcnt", 1, 1),
                ("barrier", 1),
                ("grb", 2),
                ("waitcnt", 0, 9),
                ("barrier", 9),
                ("gra", 10),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_different_simd_codes(self):
        """
        A test of a case where the SIMDs do different sequences of operations.
        """
        schedule = ScheduleInfo(
            2,  # 2 code paths
            None,
            {
                "SYNC": [[5, 6]],
                "LRA0": [[0], [2]],
                "LRB0": [[1]],
                "GRA": [[10]],
                "GRB": [[11], [12]],
            },
            [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_negative_different_simd_codes(self):
        schedule = ScheduleInfo(
            2,  # number of code paths
            None,
            {
                "SYNC": [[5, 6]],
                "LRA0": [[0], [6]],  # local read on SIMD 1 is late (at vmfma_index 6).
                "LRB0": [[1]],
                "GRA": [[10]],
                "GRB": [[11], [12]],
            },
            [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:6. "
            "First global read for A issued at vmfma_index:10. "
            "0 waitcnt operation(s) in [7, 11) provide upper bounds on the number of outstanding LRA0 operations: [] <-- none of these is 0."
        )
        assert status is False

    def test_negative_b_too_early(self):
        schedule = scheduleFromSequence(
            [
                ("lra0", 0),
                ("lrb0", 1),
                ("waitcnt", 1, 5),
                ("barrier", 6),
                ("gra", 10),
                ("grb", 11),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:1. "
            "First global read for B issued at vmfma_index:11. "
            "1 waitcnt operation(s) in [2, 12) provide upper bounds on the number of outstanding LRB0 operations: [1] <-- none of these is 0."
        )
        assert status is False

    def test_negative_b_sync_required(self):
        schedule = scheduleFromSequence(
            [
                ("lra0", 0),
                ("lrb0", 1),
                ("barrier", 4),
                ("waitcnt", 1, 5),
                ("gra", 10),
                ("grb", 11),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that a barrier (to sync waves) exists between completion of local reads for A and the first global read for A. "
            "Last local read of A issued at vmfma_index 0, first global read of A issued at vmfma_index 10, wave completion at vmfma_index 5. "
            "Expected a barrier in the range [5, 11)."
        )
        assert status is False

    def test_interleave_1(self):
        schedule = scheduleFromSequence(
            [
                ("lra0", 0),
                ("lrb0", 1),
                ("waitcnt", 1, 3),
                ("barrier", 4),
                ("gra", 5),
                ("waitcnt", 0, 7),
                ("barrier", 8),
                ("grb", 10),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_interleave_2(self):
        schedule = scheduleFromSequence(
            [
                ("lra0", 0),
                ("lrb0", 1),
                ("lra0", 2),
                ("lrb0", 3),
                # only (lrb0, 3) is still outstanding:
                ("waitcnt", 1, 4),
                ("barrier", 5),
                ("gra", 6),
                ("waitcnt", 0, 7),
                ("barrier", 8),
                ("grb", 10),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_negative_a_too_early(self):
        schedule = scheduleFromSequence(
            [
                ("lra0", 0),
                ("lrb0", 1),
                ("lra0", 2),
                ("lrb0", 3),
                # (lra0, 2) and (lrb0, 3) are outstanding:
                ("waitcnt", 2, 4),
                ("barrier", 5),
                ("gra", 6),
                ("waitcnt", 0, 7),
                ("barrier", 8),
                ("grb", 10),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:2. "
            "First global read for A issued at vmfma_index:6. "
            "1 waitcnt operation(s) in [3, 7) provide upper bounds on the number of outstanding LRA0 operations: [1] <-- none of these is 0."
        )
        assert status is False

    def test_wait_then_barrier_then_load(self):
        """
        We assume an ordering: s_waitcnt < s_barrier < buffer_load (within a vmfma_index)
        """
        schedule = scheduleFromSequence(
            [
                # For a:
                ("lra0", 0),
                ("waitcnt", 0, 1),
                ("barrier", 1),
                ("gra", 1),
                ("gra", 100),
                # For b:
                ("lrb0", 10),
                ("barrier", 11),
                ("grb", 11),
                ("waitcnt", 0, 11),
                ("grb", 100),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_redundant_waitcnt(self):
        schedule = scheduleFromSequence(
            [
                ("lra0", 0),
                ("waitcnt", 4, 1),
                ("waitcnt", 2, 2),
                ("waitcnt", 0, 4),  # <-- the useful waitcnt for A.
                ("barrier", 5),
                ("gra", 5),
                ("gra", 100),
                ("lrb0", 10),
                ("barrier", 11),
                ("grb", 11),
                ("waitcnt", 0, 11),
                ("grb", 100),
            ]
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

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
        schedule = scheduleFromSequence(
            [("lra0", 2), ("waitcnt", 0, 5), ("barrier", 6), ("gra", 3), ("gra", 100)]
        )

        status, message = verify_global_reads_not_too_early(
            schedule, {"kernel": {"DirectToLdsA": True}}
        )
        assert message == ""
        assert status is True

        status, message = verify_global_reads_not_too_early(
            schedule, {"kernel": {"DirectToLds": True}}
        )
        assert message == ""
        assert status is True

    def test_negative_direct_to_lds_b(self):
        schedule = scheduleFromSequence(
            [("lrb0", 2), ("waitcnt", 0, 5), ("barrier", 6), ("grb", 3), ("grb", 4)]
        )

        status, message = verify_global_reads_not_too_early(
            schedule, {"kernel": {"DirectToLdsB": True}}
        )
        assert "First global read for B issued at vmfma_index:4" in message
        assert message == (
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:2. "
            "First global read for B issued at vmfma_index:4. "
            "0 waitcnt operation(s) in [3, 5) provide upper bounds on the number of outstanding LRB0 operations: [] <-- none of these is 0."
        )
        assert status is False

    def test_swap_global_read_order(self):
        schedule = scheduleFromSequence(
            [
                ("lra0", 0),
                ("waitcnt", 0, 1),
                ("barrier", 2),
                ("gra", 100),
                ("lrb0", 10),
                ("waitcnt", 0, 11),
                ("barrier", 12),
                ("grb", 3),
            ]
        )
        status, message = verify_global_reads_not_too_early(
            schedule, {"kernel": {"SwapGlobalReadOrder": True}}
        )
        assert message == ""
        assert status is True

    def test_negative_swap_global_read_order(self):
        schedule = scheduleFromSequence(
            [
                ("lra0", 0),
                ("waitcnt", 0, 1),
                ("barrier", 2),
                ("gra", 3),
                ("lrb0", 10),
                ("waitcnt", 0, 11),
                ("barrier", 12),
                ("grb", 100),
            ]
        )
        status, message = verify_global_reads_not_too_early(
            schedule, {"kernel": {"SwapGlobalReadOrder": True}}
        )
        assert message == (
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:10. "
            "First global read for B issued at vmfma_index:3. "
            "0 waitcnt operation(s) in [11, 4) provide upper bounds on the number of outstanding LRB0 operations: [] <-- none of these is 0."
        )
        assert status is False

    def test_ab_tiebreaking(self):
        schedule = ScheduleInfo(
            1,
            None,
            {
                "LRA0": [[5]],
                "LRB0": [[5]],
                "SYNC": [[5, 5, 6, 6, 7, 7]],
                # the read for A is safe: because LRA0 appears before LRB0 above, the sync
                # ensures LRA0 is done.
                "GRA": [[5]],
                # No s_waitcnt for GBR, expect failure because of this.
                "GRB": [[10]],
            },
            [
                SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
                SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
                SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
            ],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:5. "
            "First global read for B issued at vmfma_index:10. "
            "3 waitcnt operation(s) in [5, 11) provide upper bounds on the number of outstanding LRB0 operations: [1, 1, 1] <-- none of these is 0."
        )
        assert status is False

        schedule = ScheduleInfo(
            1,
            None,
            {
                "LRB0": [[5]],
                "LRA0": [[5]],
                "SYNC": [[5, 5, 6, 6, 7, 7]],
                # the read for A is not safe: because LRA0 appears after LRB0 above, the sync
                # ensures LRB0 is done.
                "GRA": [[5]],
                # No s_waitcnt for GBR, expect failure because of this.
                "GRB": [[10]],
            },
            [
                SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
                SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
                SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
            ],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:5. "
            "First global read for A issued at vmfma_index:5. "
            "1 waitcnt operation(s) in [5, 6) provide upper bounds on the number of outstanding LRA0 operations: [1] <-- none of these is 0."
        )
        assert status is False

    def test_waitcnt_barrier_relative_order(self):
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRA0": [[5]],
                "LRB0": [[5]],
                # The first barrier after the waitcnt is at index 6. Too late.
                "SYNC": [[1, 5, 5, 5, 6]],
                "GRA": [[5]],
                "GRB": [[5]],
            },
            [
                SBarrier(),
                SBarrier(),
                SBarrier(),
                SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
            ],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert (
            "Failed to verify that a barrier (to sync waves) exists between completion of local reads"
            in message
        )
        assert status is False

        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRA0": [[1]],
                "LRB0": [[2]],
                # The first barrier after the required waitcnt is at index 6. Too late.
                "SYNC": [[5, 5, 5, 5, 6]],
                "GRA": [[10]],
                "GRB": [[5]],
            },
            [
                SBarrier(),
                SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
                SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
            ],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert (
            "Failed to verify that a barrier (to sync waves) exists between completion of local reads for B"
            in message
        )
        assert status is False

    def test_within_mfma_index_order(self):
        """
        The order in which instructions are added to optSchedule dictates
        the order in which instructions are added at each index.
        """

        # local reads < syncs < global read, all within a single mfma index.
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRA0": [[5]],
                "LRB0": [[5]],
                "SYNC": [[5, 5]],
                "GRA": [[5]],
                "GRB": [[5]],
            },
            [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

        # sync after global read for B.
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRA0": [[5]],
                "LRB0": [[5]],
                "GRB": [[5]],
                "SYNC": [[5, 5]],
                "GRA": [[5]],
            },
            [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for B (LRB0) are complete before the first global read for B is issued. "
            "Last local read for B issued at vmfma_index:5. "
            "First global read for B issued at vmfma_index:5. "
            "0 waitcnt operation(s) in [5, 5) provide upper bounds on the number of outstanding LRB0 operations: [] <-- none of these is 0."
        )
        assert status is False

        # sync before local read for A.
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRB0": [[5]],
                "SYNC": [[5, 5]],
                "LRA0": [[5]],
                "GRA": [[5]],
                "GRB": [[5]],
            },
            [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            (
                "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
                "Last local read for A issued at vmfma_index:5. "
                "First global read for A issued at vmfma_index:5. "
                "0 waitcnt operation(s) in [6, 6) provide upper bounds on the number of outstanding LRA0 operations: [] <-- none of these is 0."
            )
        )
        assert status is False

        # no sync for global read A.
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "SYNC": [[5, 5, 90, 95]],
                "LRB0": [[5]],
                "LRA0": [[5]],
                "GRA": [[5]],
                "GRB": [[100]],
            },
            [
                SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
                SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
            ],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:5. "
            "First global read for A issued at vmfma_index:5. "
            "0 waitcnt operation(s) in [6, 6) provide upper bounds on the number of outstanding LRA0 operations: [] <-- none of these is 0."
        )
        assert status is False

        # In this case, because "SYNC" appears after "GRA", the barrier at index 10 comes after
        # the GRA: too late.
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRA0": [[5]],
                "LRB0": [[5]],
                "GRB": [[12]],
                "GRA": [[10]],
                "SYNC": [[5, 10]],
            },
            [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""), SBarrier()],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that a barrier (to sync waves) exists between completion of local reads for A and the first global read for A. "
            "Last local read of A issued at vmfma_index 5, first global read of A issued at vmfma_index 10, wave completion at vmfma_index 5. "
            "Expected a barrier in the range [5, 10)."
        )
        assert status is False

    def test_on_the_edge(self):
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRA0": [[0, 1, 2, 3]],
                "LRB0": [[2, 3, 4, 5, 6, 7]],
                "SYNC": [[4, 4, 8, 8]],
                "GRA": [[4]],
                "GRB": [[10]],
            },
            [
                SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
                SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
            ],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_on_the_edge_tipped_by_a(self):
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRA0": [[0, 1, 2, 3]],
                "LRB0": [[2, 3, 4, 5, 6, 7]],
                "SYNC": [[4, 4, 8, 8]],
                "GRA": [[4]],
                "GRB": [[10]],
            },
            [
                SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
                SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
            ],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:3. First global read for A issued at vmfma_index:4. "
            "1 waitcnt operation(s) in [3, 5) provide upper bounds on the number of outstanding LRA0 operations: [1] <-- none of these is 0."
        )
        assert status is False

    def test_on_the_edge_tipped_by_a_saved_by_lr1(self):
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRA0": [[0, 1, 2, 3]],
                "LRA1": [[3, 4]],
                "LRB0": [[2, 3, 4, 5, 6, 7]],
                "SYNC": [[4, 4, 8, 8]],
                "GRA": [[4]],
                "GRB": [[10]],
            },
            [
                SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
                SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
            ],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == ""
        assert status is True

    def test_lr1_in_the_middle(self):
        schedule = ScheduleInfo(
            1,  # 1 code path.
            None,
            {
                "LRA0": [2 * [3]],
                "LRA1": [3 * [3]],
                "LRB0": [[]],
                "LRB1": [4 * [3]],
                "SYNC": [[3, 3]],
                "GRA": [[4]],
                "GRB": [[]],
            },
            [
                # 3 LRA1 and 4 LRB1 can be outstanding.
                SWaitCnt(dscnt=3 + 4 + 1, vlcnt=-1, vscnt=-1, comment=""),
                SBarrier(),
            ],
            None,
            None,
            None,
        )
        status, message = verify_global_reads_not_too_early(schedule, {})
        assert message == (
            "Failed to verify that all local reads for A (LRA0) are complete before the first global read for A is issued. "
            "Last local read for A issued at vmfma_index:3. "
            "First global read for A issued at vmfma_index:4. "
            "1 waitcnt operation(s) in [3, 5) provide upper bounds on the number of outstanding LRA0 operations: [1] <-- none of these is 0."
        )
        assert status is False
