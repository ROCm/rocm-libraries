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

from Tensile.Common.DataType import DataType
from Tensile.Components.CMSValidator import verify_packs_start_and_end_at_correct_indices
from cms_validation_base import CMSValidationTestBase

class TestValidatePackBF16(CMSValidationTestBase):
    """
    Validate the Pack instructions present in BF16 kernels.
    Here, the pack commands map to v_perm.
    """
    def setUp(self, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        super().setUp(kernel_updates)
    
    def validation_function(self, sched, kernel_dict, codePathIdx):
        return verify_packs_start_and_end_at_correct_indices(sched, kernel_dict, codePathIdx)

    def test_passing(self):
        """
        Passing case where pack instructions are issued after the SWaitCnt guaranteeing LRs are complete.
        - 8 LRs per group (LRA0, LRB0)
        - 8 Packs per group (PackA0, PackB0) - testing a subset of the 32 expected
        """
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
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),

            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_passing_different_number_LRs(self):
        """
        Same as above, except that there are more LRA0s and fewer LRB0s (than MIWave).
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[2, 5]],
            
            "LRA0": [[0, 0, 0, 0]],
            "LRB0": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            "PackA0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            "PackB0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            
            "LRA1": [[4, 4, 4, 4]],
            "LRB1": [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]],
            "PackA1": [[-1, -1, -1, -1, -1, -1, -1, -1]],
            "PackB1": [[-1, -1, -1, -1, -1, -1, -1, -1]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),

            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_fail_too_early(self):
        """
        Failing case where pack instructions are issued before the SWaitCnt guaranteeing LRs are complete.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[3, self.num_vmfma-1]],
            "LRA0": [[0, 0, 0, 0, 1, 1, 1, 1]],
            "LRB0": [[0, 0, 0, 0, 1, 1, 1, 1]],
            "PackA0": [[2, 3, 3, 3, 3, 3, 3, 3]],
            "PackB0": [[3, 3, 3, 3, 3, 3, 3, 3]],

            "LRA1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "LRB1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "PackA1": [[-1, -1, -1, -1, -1, -1, -1, -1]],
            "PackB1": [[-1, -1, -1, -1, -1, -1, -1, -1]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        # PackA0 at index 2 is invalid because LRs are only guaranteed at index 3 (after SWaitCnt)
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=2 issued too early, must be issued after idx=3 (because of LRA0 issued @ idx=0).")


class TestValidatePackBF16PLRPack(CMSValidationTestBase):
    """
    Validate the Pack instructions present in BF16 kernels with UsePLRPack.
    Here, Pack commands map to v_perm.
    """
    def setUp(self, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates["UsePLRPack"] = True
        super().setUp(kernel_updates)
    
    def validation_function(self, sched, kernel_dict, codePathIdx):
        return verify_packs_start_and_end_at_correct_indices(sched, kernel_dict, codePathIdx)

    def test_passing_plr_pack(self):
        """
        Passing case where pack instructions are issued after the SWaitCnt guaranteeing LRs are complete.
        - 8 LRs per group (LRA0, LRB0)
        - 8 Packs per group (PackA0, PackB0) - testing a subset of the 32 expected
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[2, 5]],
            "LRA0": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "LRB0": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "PackA0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            "PackB0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            
            "LRA1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "LRB1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "PackA1": [[6, 6, 6, 6, 6, 6, 6, 6]],
            "PackB1": [[6, 6, 6, 6, 6, 6, 6, 6]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),

            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_passing_plr_pack_different_number_LRs(self):
        """
        Same as above, except that there are more LRA0s and fewer LRB0s (than MIWaveTileA).
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[2, 5]],
            "LRA0": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            "LRB0": [[0, 0, 0, 0]],
            "PackA0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            "PackB0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            
            "LRA1": [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]],
            "LRB1": [[4, 4, 4, 4]],
            "PackA1": [[6, 6, 6, 6, 6, 6, 6, 6]],
            "PackB1": [[6, 6, 6, 6, 6, 6, 6, 6]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),

            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_fail_too_early_plr_pack(self):
        """
        Failing case where pack instructions are issued before the SWaitCnt guaranteeing LRs are complete.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[3, 6]],
            "LRA0": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "LRB0": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "PackA0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            "PackB0": [[3, 3, 3, 3, 3, 3, 3, 3]],
 
            "LRA1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "LRB1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "PackA1": [[5, 6, 6, 6, 6, 6, 6, 6]],
            "PackB1": [[6, 6, 6, 6, 6, 6, 6, 6]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        # PackA1 at index 5 is invalid because LRs are only guaranteed at index 6 (after SWaitCnt)
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA1 @ idx=5 issued too early, must be issued after idx=6 (because of LRA1 issued @ idx=4).")
    
    def test_fail_too_early_more_lrs_plr_pack(self):
        """
        Same as above except there are more LRAs than MIWaveTileA.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[3, 6]],
            "LRA0": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            "LRB0": [[0, 0, 0, 0, 0, 0, 0, 0]],
            "PackA0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            "PackB0": [[3, 3, 3, 3, 3, 3, 3, 3]],
 
            "LRA1": [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]],
            "LRB1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "PackA1": [[5, 6, 6, 6, 6, 6, 6, 6]],
            "PackB1": [[6, 6, 6, 6, 6, 6, 6, 6]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        # PackA1 at index 5 is invalid because LRs are only guaranteed at index 6 (after SWaitCnt)
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA1 @ idx=5 issued too early, must be issued after idx=6 (because of LRA1 issued @ idx=4).")

    def test_fail_too_early_less_lrs_plr_pack(self):
        """
        Same as above except there are fewer LRAs than MIWaveTileA.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[3, 6]],
            "LRA0": [[0, 0, 0, 0]],
            "LRB0": [[0, 0, 0, 0, 1, 1, 1, 1]],
            "PackA0": [[3, 3, 3, 3, 3, 3, 3, 3]],
            "PackB0": [[3, 3, 3, 3, 3, 3, 3, 3]],
 
            "LRA1": [[4, 4, 4, 4]],
            "LRB1": [[4, 4, 4, 4, 4, 4, 4, 4]],
            "PackA1": [[5, 6, 6, 6, 6, 6, 6, 6]],
            "PackB1": [[6, 6, 6, 6, 6, 6, 6, 6]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        # PackA1 at index 5 is invalid because LRs are only guaranteed at index 6 (after SWaitCnt)
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA1 @ idx=5 issued too early, must be issued after idx=6 (because of LRA1 issued @ idx=4).")

# TODO: Tests currently do not validate that the LRs start in the correct quarters.
#       Will be added in a future PR since they require modification to the LRs.
class TestValidatePackTF32(CMSValidationTestBase):
    """
    Only tests with UsePLRPack since performance is better.
    """
    def setUp(self, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates["UsePLRPack"] = True
        kernel_updates["UseF32XEmulation"] = True
        kernel_updates["ForceUnrollSubIter"] = True
        kernel_updates["DepthU"] = 32
        super().setUp(kernel_updates)

        self.q1s = 0
        self.q1e = self.num_vmfma // 4 - 1

        self.q2s = self.q1e + 1
        self.q2e = self.num_vmfma // 2 - 1

        self.q3s = self.q2e + 1
        self.q3e = self.num_vmfma // 4 * 3 - 1

        self.q4s = self.q3e + 1
        self.q4e = self.num_vmfma - 1
    
    def validation_function(self, sched, kernel_dict, codePathIdx):
        return verify_packs_start_and_end_at_correct_indices(sched, kernel_dict, codePathIdx)
    
    def test_passing(self):
        """
        Simple passing case.
        """
        # 2 A tiles, 2 B tiles, 3 bf16 MFMAs per tile
        assert self.num_vmfma == 12

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1, self.q3s+1, self.q4s+1]],

            # Must finish before 2nd quarter (i < 3)
            "LRA0": [[self.q1s] * 2],
            "PackA0": [[self.q1e] * 24],

            # Must finish before 3rd quarter (i < 6)
            "LRB0": [[self.q2s] * 2],
            "PackB0": [[self.q2e] * 24],
            
            # Must start after 2nd quarter (i > 5)
            "LRB3": [[self.q3s] * 2],
            "PackB3": [[self.q3e] * 24],

            # Must start after 3rd quarter (i > 8)
            "LRA3": [[self.q4s] * 2],
            "PackA3": [[self.q4e] * 24],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB3s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3s")
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_failing_too_early(self):
        """
        Failing case where the PackB0 are issued before the LRB0s are complete.
        """
        # 2 A tiles, 2 B tiles, 3 bf16 MFMAs per tile
        assert self.num_vmfma == 12

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2e]],

            # Must finish before 2nd quarter (i < 3)
            "LRA0": [[self.q1s] * 2],
            "PackA0": [[self.q1e] * 24],

            # Must finish before 3rd quarter (i < 6)
            "LRB0": [[self.q2s] * 2],
            "PackB0": [[self.q2s] * 24],  # There are issued before the SWaitCnt for the LRB0s
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackB0 @ idx=3 issued too early, must be issued after idx=5 (because of LRB0 issued @ idx=3).")

    def test_failing_too_late(self):
        """
        Failing case where PackA0s are issued to late (after their result is needed by the v_mfmas).

        """
        # 2 A tiles, 2 B tiles, 3 bf16 MFMAs per tile
        assert self.num_vmfma == 12
        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],

            # Must finish before 2nd quarter (i < 3)
            "LRA0": [[self.q1s] * 2],
            "PackA0": [[self.q2e] * 24],  # Issued after the first mfma in the 2nd quarter

            # Must finish before 3rd quarter (i < 6)
            "LRB0": [[self.q2s] * 2],
            "PackB0": [[self.q2e] * 24],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=5 issued too late, must be issued before MFMA @ idx=3.")

class TestValidatePackTF32CrossPackInterleaving(CMSValidationTestBase):
    """
    Tests for TF32 validation with PackA0 and PackB0 interleaving.
    Seperate class since more MFMAs are needed to allow for enough room to interleave.
    """
    def setUp(self, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates["UsePLRPack"] = True
        kernel_updates["UseF32XEmulation"] = True
        kernel_updates["ForceUnrollSubIter"] = True
        kernel_updates["DepthU"] = 32
        kernel_updates["MIWaveTileA"] = 4
        kernel_updates["MIWaveTileB"] = 4
        super().setUp(kernel_updates)
    
    def validation_function(self, sched, kernel_dict, codePathIdx):
        return verify_packs_start_and_end_at_correct_indices(sched, kernel_dict, codePathIdx)
    
    def test_passing_interleaved(self):
        """
        Passing case where PackA0 and PackB0 middle-16 packs are interleaved correctly.
        """
        assert self.num_vmfma == 48

        packA0_schedule = \
            [0] * 4\
            + [ 0,0,
                2,2,
                4,4,
                6,6,
                8,8,
                10,10,
                10,10,
                10,10] \
            + [11] * 4 \
            + [11] * 24  # Other set of 24, no conflicts in here.

        packB0_schedule = \
            [0] * 4 \
            + [ 1,1,
                3,3,
                5,5,
                7,7,
                9,9,
                11,11,
                11,11,
                11,11] \
            + [11] * 4 \
            + [11] * 24  # Other set of 24, no conflicts in here.

        optSchedule = {
            "LRA0": [[-1] * 4],
            "LRB0": [[-1] * 4],
            "SYNC": [[-1]],
            "PackA0": [packA0_schedule],
            "PackB0": [packB0_schedule],
        }

        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s")]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_failing_interleaved(self):
        """
        Failing case where PackA0 and PackB0 middle-16 packs are interleaved incorrectly.
        The case fails because the 1st pair (1, 3) in PackB0's middle-16 has a pair of PackA0s middle-16s (2,2) between the producer Pack (1) and consumer Pack (3).
        """
        assert self.num_vmfma == 48

        packA0_schedule = \
            [0] * 4\
            + [ 0,0,
                2,2,
                4,4,
                6,6,
                8,8,
                10,10,
                10,10,
                10,10] \
            + [11] * 4 \
            + [11] * 24  # Other set of 24, no conflicts in here.

        packB0_schedule = \
            [0] * 4 \
            + [ 1,3,
                3,3,
                5,5,
                7,7,
                9,9,
                11,11,
                11,11,
                11,11] \
            + [11] * 4 \
            + [11] * 24  # Other set of 24, no conflicts in here.

        optSchedule = {
            "LRA0": [[-1] * 4],
            "LRB0": [[-1] * 4],
            "SYNC": [[-1]],
            "PackA0": [packA0_schedule],
            "PackB0": [packB0_schedule],
        }

        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s")]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackB0 @ idx=1 has wrong interleaving. Should have been followed by PackB0 @ idx=3 but was followed by PackA0 @ idx=2.")

class TestValidatePackTF32MultipleGroups(CMSValidationTestBase):
    """
    Tests for TF32 pair constraint validation with multiple groups.
    The middle-16 packs (indices 4-19 within each group of 24) share a temporary register
    and must be scheduled as pairs without any other middle-16 pack between them.
    """
    def setUp(self, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates["UsePLRPack"] = True
        kernel_updates["UseF32XEmulation"] = True
        kernel_updates["ForceUnrollSubIter"] = True
        kernel_updates["DepthU"] = 32

        # 4 A tiles to get 2 groups of 24 packs (48 total)
        kernel_updates["MIWaveTileA"] = 4
        kernel_updates["MIWaveTileB"] = 2
        super().setUp(kernel_updates)

        self.q1s = 0
        self.q1e = self.num_vmfma // 4 - 1

        self.q2s = self.q1e + 1
        self.q2e = self.num_vmfma // 2 - 1

        self.q3s = self.q2e + 1
        self.q3e = self.num_vmfma // 4 * 3 - 1

        self.q4s = self.q3e + 1
        self.q4e = self.num_vmfma - 1
    
    def validation_function(self, sched, kernel_dict, codePathIdx):
        return verify_packs_start_and_end_at_correct_indices(sched, kernel_dict, codePathIdx)
    
    def test_passing_two_groups_consecutive(self):
        """
        Valid case: 2 groups of 24 packs each, the groups are not interleaved and the middle 16 pairs are consecutive.
        """
        # 4 A tiles, 2 B tiles, 3 bf16 MFMAs per tile = 24 vmfmas
        assert self.num_vmfma == 24
               
        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],

            "LRA0": [[self.q1s] * 4],
            "PackA0": [[self.q1e] * 48],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [[self.q2e] * 24],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)
    
    def test_passing_two_groups_interleaved(self):
        """
        Valid case: 2 groups of 24 packs each, with the instructions interleaved.
        The packs are split into 3 parts: first 4 packs, middle 16, last 4.
        The three parts are interleaved between groups, but each part is scheduled as a consecutive block.
        """
        # 4 A tiles, 2 B tiles, 3 bf16 MFMAs per tile = 24 vmfmas
        assert self.num_vmfma == 24

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],
            
            "LRA0": [[self.q1s] * 4],
            # Do the first 4 packs from each group, then the middle 16 packs, then the last 4 packs.
            "PackA0": [
                [self.q1s+2] * 4 + [self.q1s+3] * 16 + [self.q1s+4] * 4 +
                [self.q1s+2] * 4 + [self.q1s+3] * 16 + [self.q1s+4] * 4  
            ],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [[self.q2e] * 24]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_passing_two_groups_fully_interleaved(self):
        """
        Passing case: 2 groups of 24 packs each, with the instructions interleaved.
        The packs are split into 3 parts: first 4 packs, middle 16, last 4.
        The the parts are fully interleaved between groups.
        """
        # 4 A tiles, 2 B tiles, 3 bf16 MFMAs per tile = 24 vmfmas
        assert self.num_vmfma == 24

        optSchedule = {
            "LRA0": [[-1, -1, -1, -1]],
            "SYNC": [[-1, self.q2s+1]],
            "PackA0": [
                [1,1,3,3] + [3,3, 3,3, 3,3, 3,3, 5,5, 5,5, 5,5, 5,5] + [5,5,5,5] +
                [0,0,2,2] + [2,2, 2,2, 2,2, 2,2, 4,4, 4,4, 4,4, 4,4] + [5,5,5,5]  
            ],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [[self.q2e] * 24]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_failing_two_groups_fully_interleaved(self):
        """
        Failing case: 2 groups of 24 packs each, with the instructions interleaved.
        The packs are split into 3 parts: first 4 packs, middle 16, last 4.
        The the parts are fully interleaved between groups.

        The case fails because the second instruction in the middle 16 of the second group is out of order.
        """
        # 4 A tiles, 2 B tiles, 3 bf16 MFMAs per tile = 24 vmfmas
        assert self.num_vmfma == 24

        optSchedule = {
            "LRA0": [[-1, -1, -1, -1]],
            "SYNC": [[-1, self.q2s+1]],
            "PackA0": [
                [1,1,3,3] + [3,3, 3,3, 3,3, 3,3, 5,5, 5,5, 5,5, 5,5] + [5,5,5,5] +
                [0,0,2,2] + [2,3, 2,2, 2,2, 2,2, 4,4, 4,4, 4,4, 4,4] + [5,5,5,5]    # Second instruction in the middle 16 is out of order.
            ],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [[self.q2e] * 24]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, 'PackA0 @ idx=2 has wrong interleaving. Should have been followed by PackA0 @ idx=3 but was followed by PackA0 @ idx=2.')
