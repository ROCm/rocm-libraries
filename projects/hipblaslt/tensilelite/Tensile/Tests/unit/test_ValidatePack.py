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
import pytest
from rocisa.instruction import SWaitCnt, SNop, SBarrier

from Tensile.Components.CMSValidator import (
    add_local_read_constraints, add_pack_constraints, isValid, ValidationContext,
)
from Tensile.Components.CustomSchedule import ScheduleInfo
from cms_validation_base import CMSValidationTestBase
from cms_test_utils import make_mock_id_map, make_mock_mfma_code

class TestValidatePackBF16(CMSValidationTestBase):
    """
    Validate the Pack instructions present in BF16 kernels.
    Here, the pack commands map to v_perm.
    """
    validator_passes = [add_local_read_constraints, add_pack_constraints]

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

class TestValidatePackBF16MFMAReorder(CMSValidationTestBase):
    """
    Validate the Pack instructions present in BF16 kernels.
    Here, the pack commands map to v_perm.

    Change from column-major to row-major.

    1st half
    | 0 2 | -> | 0 1 |
    | 1 3 | -> | 2 3 |
    2nd half
    | 4 6 | -> | 4 5 |
    | 5 7 | -> | 6 7 |
    """
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        super().setup_method(method, kernel_updates=kernel_updates)
        self.mfma_reorder = [0, 2, 1, 3, 4, 6, 5, 7]
    
    validator_passes = [add_local_read_constraints, add_pack_constraints]

    def test_passing(self):
        """
        Passing case where pack instructions are issued after the SWaitCnt guaranteeing LRs are complete.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[2]],
            
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "PackA0": [[3, 3, 5, 5]],
            "PackB0": [[3, 3, 4, 4]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None, mfmaReorder=self.mfma_reorder)

    def test_fail_too_early(self):
        """
        Failing case where pack instructions are issued before the SWaitCnt guaranteeing LRs are complete.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "LRA0": [[-1, -1]],
            "LRB0": [[-1, -1]],
            "SYNC": [[3]],
            "PackA0": [[2, 3, 5, 5]],
            "PackB0": [[3, 3, 4, 4]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=2 issued too early, must be issued after idx=3 (because of LRA0 issued @ idx=-1).", mfmaReorder=self.mfma_reorder)

    def test_failing_too_late(self):
        """
        Failing case where pack instructions are issued after the MFMA that needs them.
        
        With mfma_reorder = [0, 2, 1, 3, 4, 6, 5, 7]:
        - Logical index 4 -> execution position 4
        - Logical index 5 -> execution position 6
        - Logical index 6 -> execution position 5
        
        For PackA0 (2 tiles, 2 packs per tile):
        - Tile 0: needed at logical 4 -> execution position 4
        - Tile 1: needed at logical 5 -> execution position 6
        
        If we issue PackA0 at position 4 or later, it's too late for tile 0.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "LRA0": [[-1, -1]],
            "LRB0": [[-1, -1]],
            "SYNC": [[-1]],
            "PackA0": [[3, 4, 5, 5]],
            "PackB0": [[3, 3, 4, 4]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=4 issued too late, must be issued before MFMA @ idx=4.", mfmaReorder=self.mfma_reorder)

class TestValidatePackBF16PLRPack(CMSValidationTestBase):
    """
    Validate the Pack instructions present in BF16 kernels with UsePLRPack.
    Here, Pack commands map to v_perm.
    """
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update({"UsePLRPack": True})
        super().setup_method(method, kernel_updates=kernel_updates)
    
    validator_passes = [add_local_read_constraints, add_pack_constraints]

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
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update({"UsePLRPack": True, "UseF32XEmulation": True, "ForceUnrollSubIter": True, "UseDirect32XEmulation": True, "DepthU": 32})
        super().setup_method(method, kernel_updates=kernel_updates)

        self.q1s = 0
        self.q1e = self.num_vmfma // 4 - 1

        self.q2s = self.q1e + 1
        self.q2e = self.num_vmfma // 2 - 1

        self.q3s = self.q2e + 1
        self.q3e = self.num_vmfma // 4 * 3 - 1

        self.q4s = self.q3e + 1
        self.q4e = self.num_vmfma - 1
    
    validator_passes = [add_local_read_constraints, add_pack_constraints]
    
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

    def test_failing_too_late_B(self):
        """
        Failing case where PackB0s are issued too late (after their result is needed by the v_mfmas).

        """
        # 2 A tiles, 2 B tiles, 3 bf16 MFMAs per tile
        assert self.num_vmfma == 12
        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],

            # Must finish before 2nd quarter (i < 3)
            "LRA0": [[self.q1s] * 2],
            "PackA0": [[self.q1e] * 24],

            # Must finish before 3rd quarter (i < 6)
            "LRB0": [[self.q2s] * 2],
            "PackB0": [[self.q3e] * 24],  # Issued in 3rd quarter, too late
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackB0 @ idx=8 issued too late, must be issued before MFMA @ idx=6.")

    def test_failing_not_enough_time_CVT1_MFMA(self):
        """
        Failing case where there is not enough time between the last cvt pack command the first "real" mfma using the result.
        """
        assert self.num_vmfma == 12

        optSchedule = {
            "SYNC": [[self.q1s+2, self.q2s+2]],
            "SNOP": [[self.q2s]],  # 

            # Must finish before 2nd quarter (i < 3)
            "LRA0": [[self.q1s+1] * 2],
            "PackA0": [
                [self.q1e] * 22 +
                [self.q2s] *2 
            ],

            # Must finish before 3rd quarter (i < 6)
            "LRB0": [[self.q2s+1] * 2],
            "PackB0": [[self.q2e] * 24],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]

        snopCode = [
            SNop(waitState=1, comment="Needed to force the last 2 PackA0s to be too close to the MFMA."),
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=3 has too little gap between it and MFMA @ idx=4. Expected at least 2 quad-cycles but only 1 passed.", snopCode=snopCode)

class TestValidatePackTF32MFMAReorder(CMSValidationTestBase):
    """
    Only tests with UsePLRPack since performance is better.
    Use MFMA reorder to swap Q2 and Q3.
    
    With MIWaveTileA=4, MIWaveTileB=4, TF32 (3 MFMAs per tile), we have 48 MFMAs:
    - Q1 (indices 0-11): first quarter
    - Q2 (indices 12-23): second quarter
    - Q3 (indices 24-35): third quarter
    - Q4 (indices 36-47): fourth quarter
    
    MFMA reorder swaps Q2 and Q3 (indices 12-23 execute at 24-35 positions and vice versa).
    """
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update(
            {
                "UsePLRPack": True, "UseF32XEmulation": True, "UseDirect32XEmulation": True, "ForceUnrollSubIter": True, "DepthU": 32, 
                "MIWaveTileA": 4, "MIWaveTileB": 4  # Need >= 4 for n_tiles_quarter >= 1
            }
        )
        super().setup_method(method, kernel_updates=kernel_updates)

        self.q1s = 0
        self.q1e = self.num_vmfma // 4 - 1

        self.q2s = self.q1e + 1
        self.q2e = self.num_vmfma // 2 - 1

        self.q3s = self.q2e + 1
        self.q3e = self.num_vmfma // 4 * 3 - 1

        self.q4s = self.q3e + 1
        self.q4e = self.num_vmfma - 1

        # MFMA reorder: swap Q2 (indices 12-23) and Q3 (indices 24-35)
        self.mfma_reorder = list(range(12)) + list(range(24, 36)) + list(range(12, 24)) + list(range(36, 48))
    
    validator_passes = [add_local_read_constraints, add_pack_constraints]
    
    def test_passing(self):
        """
        Simple passing case.
        """
        assert self.num_vmfma == 48

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1, self.q3s+1, self.q4s+1]],

            "LRA0": [[self.q2s] * 4],
            "PackA0": [[self.q2e] * 48],

            "LRB0": [[self.q1s] * 4],
            "PackB0": [[self.q1e] * 48],
            
            "LRB3": [[self.q3s] * 4],
            "PackB3": [[self.q3e] * 48],

            "LRA3": [[self.q4s] * 4],
            "PackA3": [[self.q4e] * 48],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB3s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3s")
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None, mfmaReorder=self.mfma_reorder)

    def test_failing_too_early(self):
        """
        Failing case where PackB0 are issued too early (before LRB0s are complete).
        """
        assert self.num_vmfma == 48

        optSchedule = {
            "SYNC": [[self.q1e]],
            "LRA0": [[self.q1s] * 4],
            "LRB0": [[self.q1s] * 4],
            "PackB0": [[self.q1s+1] * 48],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackB0 @ idx=1 issued too early, must be issued after idx=11 (because of LRB0 issued @ idx=0).", mfmaReorder=self.mfma_reorder)

class TestValidatePackTF32CrossPackInterleaving(CMSValidationTestBase):
    """
    Tests for TF32 validation with PackA0 and PackB0 interleaving.
    Seperate class since more MFMAs are needed to allow for enough room to interleave.
    """
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update({"UsePLRPack": True, "UseF32XEmulation": True, "UseDirect32XEmulation": True, "ForceUnrollSubIter": True, "DepthU": 32, "MIWaveTileA": 4, "MIWaveTileB": 4})
        super().setup_method(method, kernel_updates=kernel_updates)
    
    validator_passes = [add_local_read_constraints, add_pack_constraints]
    
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
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update(
            {
                "UsePLRPack": True, "UseF32XEmulation": True, "UseDirect32XEmulation": True, "ForceUnrollSubIter": True, "DepthU": 32, 
                "MIWaveTileA": 4, "MIWaveTileB": 2  # 4 A tiles to get 2 groups of 24 packs (48 total)
            }
        )
        super().setup_method(method, kernel_updates=kernel_updates)

        self.q1s = 0
        self.q1e = self.num_vmfma // 4 - 1

        self.q2s = self.q1e + 1
        self.q2e = self.num_vmfma // 2 - 1

        self.q3s = self.q2e + 1
        self.q3e = self.num_vmfma // 4 * 3 - 1

        self.q4s = self.q3e + 1
        self.q4e = self.num_vmfma - 1
    
    validator_passes = [add_local_read_constraints, add_pack_constraints]
    
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

class TestValidatePackTF32MFMA4x4x4(CMSValidationTestBase):
    """
    Tests for TF32 validation with 4x4x4 MFMA.
    """
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update(
            {
                "UsePLRPack": True, "UseF32XEmulation": True, "UseMFMAF32XEmulation": True, "UseDirect32XEmulation": True, "ForceUnrollSubIter": True, "DepthU": 32, 
                "MIWaveTileA": 4, "MIWaveTileB": 4  # 4 A tiles, 4 B tiles, 3 MFMAs per tile = 48 vmfmas
            }
        )
        super().setup_method(method, kernel_updates=kernel_updates)

        self.q1s = 0
        self.q1e = self.num_vmfma // 4 - 1

        self.q2s = self.q1e + 1
        self.q2e = self.num_vmfma // 2 - 1

        self.q3s = self.q2e + 1
        self.q3e = self.num_vmfma // 4 * 3 - 1

        self.q4s = self.q3e + 1
        self.q4e = self.num_vmfma - 1
    
    validator_passes = [add_local_read_constraints, add_pack_constraints]
    
    def test_passing(self):
        """
        Simple passing case.
        """
        assert self.num_vmfma == 48

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1, self.q3s+1, self.q4s+1]],

            "LRA0": [[self.q1s] * 2],
            "PackA0": [
                [self.q1s+2] * 4 + 
                [self.q1s+2] * 2 +
                [self.q1s+4] * 4  # Needs +2 MFMAs to satisfy the 5 quad-cycle gap.
            ],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [
                [self.q2s+2] * 4 +
                [self.q2s+2] * 2 +
                [self.q2s+4] * 4  # Needs +2 MFMAs to satisfy the 5 quad-cycle gap.
            ],
            
            "LRB3": [[self.q3s] * 2],
            "PackB3": [
                [self.q3s+2] * 4 +
                [self.q3s+2] * 2 +
                [self.q3s+4] * 4  # Needs +2 MFMAs to satisfy the 5 quad-cycle gap.
            ],

            "LRA3": [[self.q4s] * 2],
            "PackA3": [
                [self.q4s+2] * 4 +
                [self.q4s+2] * 2 +
                [self.q4s+4] * 4  # Needs +2 MFMAs to satisfy the 5 quad-cycle gap.
            ],
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
        assert self.num_vmfma == 48

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2e]],

            "LRA0": [[self.q1s] * 2],
            "PackA0": [
                [self.q1s+2] * 4 +
                [self.q1s+2] * 2 +
                [self.q1s+4] * 4
            ],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [
                [self.q2s+1] +  # Issued too early.
                [self.q3s] * 3 +
                [self.q3s] * 2 +
                [self.q3s+2] * 4
            ],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackB0 @ idx=13 issued too early, must be issued after idx=23 (because of LRB0 issued @ idx=12).")

    def test_failing_too_late(self):
        """
        Failing case where PackA0s are issued too late (after their result is needed by the v_mfmas).
        """
        assert self.num_vmfma == 48

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],

            # Must finish before 2nd quarter
            "LRA0": [[self.q1s] * 2],
            "PackA0": [
                [self.q1s+1] * 4 +
                [self.q1s+1] * 2 +
                [self.q2s+2] * 4  # Issued after the 2nd mfma in the 2nd quarter
            ],

            # Must finish before 3rd quarter
            "LRB0": [[self.q2s] * 2],
            "PackB0": [
                [self.q2s+2] * 4 +
                [self.q2s+2] * 2 +
                [self.q2s+3] * 4
            ],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=14 issued too late, must be issued before MFMA @ idx=13.")

    def test_passing_snop(self):
        """
        Passing case that relies on s_nop instructions to guarantee there is enough time around the 4x4 MFMAs.
        This test verifies that SNop instructions are properly counted for quad-cycle spacing.
        The first 4 packs (CVT0) need 2 quad-cycles before their result is used by the 4x4 MFMA.
        
        For TF32 4x4 MFMA with MIWaveTileA=4, MIWaveTileB=4:
        - 48 total vmfmas (4*4*3)
        - q1: 0-11, q2: 12-23, q3: 24-35, q4: 36-47
        - PackA0 packs are needed for MFMAs in q2 starting at idx 12
        """
        assert self.num_vmfma == 48

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],
            "SNOP": [[self.q1s+2, self.q2s+2]],

            # Must finish before 2nd quarter
            "LRA0": [[self.q1s, self.q1s]],
            "PackA0": [
                [self.q1s+1] * 4 +
                [self.q1s+1] + 
                [self.q1s+1] + # Only 2 quad-cycles before Pack[6] without SNop.
                [self.q1s+2] * 4
            ],

            # Must finish before 3rd quarter
            "LRB0": [[self.q2s, self.q2s]],
            "PackB0": [
                [self.q2s+1] * 4 +
                [self.q2s+1] +
                [self.q2s+1] + # Only 2 quad-cycles before Pack[6] without SNop.
                [self.q2s+2] * 4
            ],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]

        snopCode = [
            SNop(waitState=1, comment="Needed to have 5 quad-cycle space between PackA0[5] and PackBo[6]."),
            SNop(waitState=1, comment="Needed to have 5 quad-cycle space between PackB0[5] and PackB0[6]."),
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None, snopCode=snopCode)

    def test_failing_not_enough_time_CVTO_MFMA(self):
        """
        Failing case where there is not enough time between the first pair of CVT0 and the first 4x4 MFMA.
        """
        assert self.num_vmfma == 48

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],
            
            "LRA0": [[self.q1s] * 2],
            "PackA0": [
                [self.q1s+1] * 2 +
                [self.q1s+2] * 2 +
                [self.q1s+1] +
                [self.q1s+3] +
                [self.q1s+4] * 4
            ],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [
                [self.q2s+1] * 2 +
                [self.q2s+2] * 2 +
                [self.q2s+1] +
                [self.q2s+3] +
                [self.q2s+4] * 4
            ],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=1 has too little gap between it and PackA0 @ idx=1. Expected at least 2 quad-cycles but only 1 passed.")
    
    def test_failing_not_enough_time_MFMA_CVT1(self):
        """
        Failing case where there is not enough time between the 4x4 MFMA (Pack 5) and the first CVT1 pack (Pack 6).
        Pack 5 (second 4x4 MFMA) needs 5 quad-cycles before Pack 6 (first CVT1) can use its result.
        With only 1 standard MFMA between them, there's only 3 quad-cycles, which is not enough.
        """
        assert self.num_vmfma == 48

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],

            "LRA0": [[self.q1s] * 2],
            "PackA0": [
                [self.q1s+2] * 4 +
                [self.q1s+2] * 2 +
                [self.q1s+3] * 4     # CVT1: indices 6-9 at vmfma 3 (only 1 MFMA between, not enough time!)
            ],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [
                [self.q2s+2] * 4 +
                [self.q2s+2] * 2 +
                [self.q2s+4] * 4  # CVT1 at vmfma q2s+4, enough time for B
            ],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=2 has too little gap between it and PackA0 @ idx=3. Expected at least 5 quad-cycles but only 3 passed.")

    def test_failing_not_enough_time_CVT1_MFMA(self):
        """
        Failing case where there is not enough time between the last CVT1 pack and the first "real" MFMA using the result.
        CVT1 packs (indices 6-9) need 2 quad-cycles before the MFMAs can use their results.
        """
        assert self.num_vmfma == 48

        optSchedule = {
            "SYNC": [[self.q1s+2, self.q2s+2]],
            "SNOP": [[self.q2s]],

            "LRA0": [[self.q1s+1] * 2],
            "PackA0": [
                [self.q1s+2] * 4 +
                [self.q1s+2] * 2 +
                [self.q1e] * 2 +
                [self.q2s] * 2       # CVT1 (8-9) at vmfma 12 (too close to MFMA at q2s+1)
            ],

            "LRB0": [[self.q2s+1] * 2],
            "PackB0": [[self.q2e] * 10],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s")
        ]

        snopCode = [
            SNop(waitState=1, comment="Needed to force the last 2 PackA0s to be too close to the MFMA."),
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=12 has too little gap between it and MFMA @ idx=13. Expected at least 2 quad-cycles but only 1 passed.", snopCode=snopCode)

class TestValidatePackTF32MFMA4x4x4MultipleTiles(CMSValidationTestBase):
    """
    Tests for TF32 4x4 MFMA validation with multiple tiles (multiple groups of 10 packs).
    This tests that the tile-based offset is correctly calculated for each group.
    
    With MIWaveTileA=4, there are 4 A tiles, each requiring 10 pack instructions (40 total PackA0).
    Each tile's packs should be needed by different MFMAs:
    - Tile 0 (packs 0-9): needed by MFMAs at base_offset + 0
    - Tile 1 (packs 10-19): needed by MFMAs at base_offset + 3
    - Tile 2 (packs 20-29): needed by MFMAs at base_offset + 6
    - Tile 3 (packs 30-39): needed by MFMAs at base_offset + 9
    """
    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates.update(
            {
                "UsePLRPack": True, "UseF32XEmulation": True, "UseMFMAF32XEmulation": True, "UseDirect32XEmulation": True, "ForceUnrollSubIter": True, "DepthU": 32, 
                "MIWaveTileA": 4, "MIWaveTileB": 2  # 4 A tiles, 2 B tiles, 3 MFMAs per tile = 24 vmfmas
            }
        )
        super().setup_method(method, kernel_updates=kernel_updates)

        # With 4 A tiles, 2 B tiles, 3 MFMAs per tile = 24 vmfmas
        self.q1s = 0
        self.q1e = self.num_vmfma // 4 - 1  # 5

        self.q2s = self.q1e + 1  # 6
        self.q2e = self.num_vmfma // 2 - 1  # 11

        self.q3s = self.q2e + 1  # 12
        self.q3e = self.num_vmfma // 4 * 3 - 1  # 17

        self.q4s = self.q3e + 1  # 18
        self.q4e = self.num_vmfma - 1  # 23
    
    validator_passes = [add_local_read_constraints, add_pack_constraints]
    
    def _make_valid_pack_group(self, base_idx: int) -> list[int]:
        """
        Creates a valid group of 10 packs with proper spacing for 4x4 MFMA TF32.
        Pack 5 (second 4x4 MFMA) needs 5 quad-cycles before Pack 6 (first CVT1).
        """
        return (
            [base_idx] * 4 +      # CVT0 (packs 0-3)
            [base_idx] * 2 +      # 4x4 MFMAs (packs 4-5)
            [base_idx + 2] * 4    # CVT1 (packs 6-9) - needs +2 MFMA indices for 5 quad-cycle gap
        )
    
    def test_passing_multiple_tiles_different_timings(self):
        """
        Passing case where 4 groups of 10 packs are scheduled at different times
        appropriate for their respective tiles.
        
        - Tile 0 packs (0-9): needed by MFMAs 6-8 (q2s, q2s+1, q2s+2)
        - Tile 1 packs (10-19): needed by MFMAs 9-11 (q2s+3, q2s+4, q2s+5)
        - Tile 2 packs (20-29): needed by MFMAs 6-8 (for B tile 1)
        - Tile 3 packs (30-39): needed by MFMAs 9-11 (for B tile 1)
        
        All packs are scheduled early in Q1 with proper spacing.
        """
        assert self.num_vmfma == 24

        # Each tile's packs have proper 5 quad-cycle gap for 4x4 MFMA
        packA0_schedule = (
            self._make_valid_pack_group(self.q1s+2) +  # Tile 0: base at 2, CVT1 at 4
            self._make_valid_pack_group(self.q1s+2) +  # Tile 1
            self._make_valid_pack_group(self.q1s+2) +  # Tile 2
            self._make_valid_pack_group(self.q1s+2)    # Tile 3
        )

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],

            "LRA0": [[self.q1s] * 4],  # 4 LRs for 4 tiles
            "PackA0": [packA0_schedule],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [self._make_valid_pack_group(self.q2s+2) + self._make_valid_pack_group(self.q2s+2)],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s"),
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_passing_tile1_packs_after_tile0_deadline(self):
        """
        Passing case that exercises the tile-based offset calculation.
        
        Tile 0 packs (0-9) are needed by MFMAs starting at q2s (index 6).
        Tile 1 packs (10-19) are needed by MFMAs starting at q2s+3 (index 9).
        
        We schedule tile 1's CVT0 packs at index 6, which would be INVALID if they
        were incorrectly assigned to tile 0's needed_by (index 6) since 6 >= 6, but
        is VALID because they should be assigned to tile 1's needed_by (index 9).
        
        This test specifically validates that the tile offset is correctly applied:
        - Without the fix, pack 10-13 (tile 1's CVT0) at index 6 would be checked
          against tile 0's needed_by (MFMA 6), failing with "too late" since 6 >= 6
        - With the fix, pack 10-13 is correctly checked against tile 1's needed_by
          (MFMA 9), which passes since 6 < 9
        
        CVT1 needed_by for tile 1 is MFMA at index 10 (base=6 + tile_offset=3 + pack_offset=1).
        We place CVT1 at index 8 (base 6 + 2) which is before MFMA 10.
        """
        assert self.num_vmfma == 24

        # Tile 0: schedule at index 2 (valid, well before needed_by at 6)
        # Tile 1: base at 6 (CVT0/4x4 at 6, CVT1 at 8)
        #   - CVT0 at 6 would fail if checked against tile 0's needed_by (MFMA 6)
        #   - CVT0 at 6 passes when correctly checked against tile 1's needed_by (MFMA 9)
        #   - CVT1 at 8 is before tile 1's CVT1 needed_by (MFMA 10)
        packA0_schedule = (
            self._make_valid_pack_group(self.q1s+2) +  # Tile 0: base at 2, CVT1 at 4, before needed_by at 6
            self._make_valid_pack_group(self.q2s) +    # Tile 1: base at 6, CVT1 at 8, before needed_by at 9/10
            self._make_valid_pack_group(self.q1s+2) +  # Tile 2
            self._make_valid_pack_group(self.q1s+2)    # Tile 3
        )

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],

            "LRA0": [[self.q1s] * 4],
            "PackA0": [packA0_schedule],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [self._make_valid_pack_group(self.q2s+2) + self._make_valid_pack_group(self.q2s+2)],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s"),
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_failing_tile1_packs_actually_too_late(self):
        """
        Failing case where tile 1's packs are scheduled too late for tile 1's actual deadline.
        
        Tile 1 packs (10-19) are needed by MFMAs starting at q2s+3 (index 9).
        We schedule tile 1's CVT0 packs at index 9, which is AT the needed_by index,
        meaning they're too late (must be strictly before).
        """
        assert self.num_vmfma == 24

        # Tile 1: CVT0 at index 9 - exactly at the needed_by, which is too late
        packA0_schedule = (
            self._make_valid_pack_group(self.q1s+2) +  # Tile 0: valid
            [self.q2s+3] * 4 + [self.q2s+3] * 2 + [self.q2s+5] * 4 +  # Tile 1: CVT0 at 9 (too late!)
            self._make_valid_pack_group(self.q1s+2) +  # Tile 2
            self._make_valid_pack_group(self.q1s+2)    # Tile 3
        )

        optSchedule = {
            "SYNC": [[self.q1s+1, self.q2s+1]],

            "LRA0": [[self.q1s] * 4],
            "PackA0": [packA0_schedule],

            "LRB0": [[self.q2s] * 2],
            "PackB0": [self._make_valid_pack_group(self.q2s+2) + self._make_valid_pack_group(self.q2s+2)],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0s"),
        ]

        # Tile 1's CVT0 at index 9 is too late - needed by MFMA at index 9
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=9 issued too late, must be issued before MFMA @ idx=9.")

    def test_broken_pack_schedule_with_mfma_reorder(self):
        """
        Test that the validator detects pack scheduling bugs (read-before-write hazards)
        with complex MFMA reordering.
        
        Broken PackA0 schedule (packs at idx 27-30, MFMA needs them at idx 26) should fail.
        This is a real-world 128x256x32 TF32 NT configuration with MFMA reordering.
        """
        # Minimal kernel config for 128x256x32 TF32 NT validation
        kernel = {
            "UseF32XEmulation": True, "UseDirect32XEmulation": True, "UseMFMAF32XEmulation": True,
            "ForceUnrollSubIter": True, "DirectToLds": True, "SwapGlobalReadOrder": False,
            "UsePLRPack": True, "MIWaveTileA": 4, "MIWaveTileB": 8, "Use64bShadowLimit": 1,
            "VectorWidthA": 1, "VectorWidthB": 1,
            "ProblemType": {"TLUA": True, "TLUB": True},
        }

        # Hardcoded schedule from _get_schedule_128x256x32_TF32 NT case
        # with broken PackA0: packs at indices 27-30, but MFMA at index 26 needs their results
        schedule_info = ScheduleInfo(
            numCodePaths=2,
            numMfma=96,
            optSchedule={

                # BROKEN PackA0: indices 27-30 instead of 24-26, causing read-before-write
                "PackA0": [[
                    20, 20, 21, 21, 
                    23, 23, 
                    28, 28, 28, 28,

                    27, 27, 27, 27,
                    28, 28, 
                    30, 30, 30, 30]],

                "SYNC": [[1, 1, 3, 49, 78]],
                "GRIncA": [[0]*9],
                "GRIncB": [[0]*9],
                "LRA0": [[0]*16],
                "LRB0": [[0]*32],
                "LRA3": [[77]*16],
                "LRB3": [[48]*32],
                "LCC": [[0]*9],
                "LWSA": [[0]],
                "LWSB": [[0]],
                "LWRA": [[0]],
                "LWRA": [[0]],
                "GRA": [[2]*8],
                "GRB": [[2]*8],
                "PackB0": [[42, 42, 42, 43, 48, 48, 49, 49, 49, 50, 43, 43, 44, 44, 48, 48, 50, 50, 50, 51, 45, 45, 46, 46, 48, 48, 51, 51, 52, 52, 46, 47, 47, 47, 48, 48, 52, 53, 53, 53]],
                "PackB3": [[67, 67, 68, 68, 75, 75, 76, 76, 77, 77, 69, 69, 70, 70, 75, 75, 78, 78, 86, 86, 71, 71, 72, 72, 75, 75, 87, 87, 88, 88, 73, 73, 74, 74, 75, 75, 89, 89, 90, 90]],
                "PackA3": [[89, 89, 90, 90, 92, 92, 93, 93, 93, 94, 90, 91, 91, 91, 92, 92, 94, 94, 95, 95]],
            },
            syncCode=[
                SWaitCnt(dscnt=0),
                SBarrier(comment=""),
                SWaitCnt(vlcnt=0),
                SWaitCnt(dscnt=0),
                SWaitCnt(dscnt=0),
            ],
            nglshift=12,
            nllshift=12,
            mfmaReorder=[
                 0,  1,  2,  3,
                 4,  5,  6,  7,
                 8,  9, 10, 11,
                12, 13, 14, 15,
                16, 17, 18, 19,
                20, 21, 22, 23,

                # Reordered to do AhBh, AhBl, AlBh in row-major order.
                # Gives more time to finish Al's
                24, 26, 30, 32,
                36, 38, 42, 44,
                25, 31, 37, 43,
                27, 29, 33, 35,
                39, 41, 45, 47,
                28, 34, 40, 46,

                # Reordered to do AhBh, AlBh, AhBl in column-major order.
                # Gives more time to finish Bl's
                48, 49, 51, 52,
                50, 53, 54, 55,
                57, 58, 56, 59,
                60, 61, 63, 64,
                62, 65, 66, 67,
                69, 70, 68, 71,

                72, 73, 74, 75,
                76, 77, 78, 79,
                80, 81, 82, 83,
                84, 85, 86, 87,
                88, 89, 90, 91,
                92, 93, 94, 95],
        )

        valid, message = isValid(schedule_info, ValidationContext(kernel=kernel, id_map=make_mock_id_map(schedule_info, kernel), mfma_code=make_mock_mfma_code(schedule_info.numMfma)))
        assert valid  # Schedule previously failed, now passes.

        # Check it fails where it's expected to.
        schedule_info.optSchedule["PackA0"] = [[
            20, 20, 21, 21,
            23, 23,
            28, 28, 28, 28,

            37, 37, 37, 37,
            40, 40,
            41, 41, 41, 41]]
        valid, message = isValid(schedule_info, ValidationContext(kernel=kernel, id_map=make_mock_id_map(schedule_info, kernel), mfma_code=make_mock_mfma_code(schedule_info.numMfma)))
        assert not valid
        assert message == "Code path 0: PackA0 @ idx=37 issued too late, must be issued before MFMA @ idx=36."


class TestValidatePackTF32MFMA4x4x4SwapPacks(CMSValidationTestBase):
    """
    Tests for TF32 4x4 MFMA validation with VectorWidth > 1 (swap packs).
    When VW > 1, VSwapB32 instructions appear at the beginning of the pack sequence
    to transpose registers after wider local reads. Count: 4 * (vw - 1) per side.

    With ForceUnrollSubIter, both LRA0 and LRB0 are needed by MFMAs starting at
    index 24 (q2s), so both must be issued and guaranteed before index 24. We place
    both LR groups and their SWaitCnt early in q1.

    Parametrized over (DepthU, SUB_ITER_SUFFIX, ForceUnrollSubIter):
      - (64, "1", False): DepthU > matrixInstK, standard sub-iteration naming
      - (32, "3", True):  DepthU == matrixInstK, ForceUnrollSubIter, A/B3 suffix
      - (32, "1", False): DepthU == matrixInstK, no ForceUnrollSubIter, A/B1 suffix
    """
    @pytest.fixture(autouse=True, params=[
        pytest.param((64, "1", False), id="DU64"),
        pytest.param((32, "3", True), id="DU32_ForceUnroll"),
        pytest.param((32, "1", False), id="DU32_NoForceUnroll"),
    ])
    def _config(self, request):
        self.DEPTH_U, self.SUB_ITER_SUFFIX, self.FORCE_UNROLL_SUB_ITER = request.param
        # Run setup_method here (after fixture sets config params)
        self.setup_method()

    def setup_method(self, method=None, *, kernel_updates: Optional[dict[str, Any]] = None) -> None:
        if not hasattr(self, 'DEPTH_U'):
            # Called before _config fixture — skip, _config will call us after setting params
            return
        kernel_updates = kernel_updates.copy() if kernel_updates else {}
        kernel_updates["UsePLRPack"] = True
        kernel_updates["UseF32XEmulation"] = True
        kernel_updates["UseMFMAF32XEmulation"] = True
        kernel_updates["UseDirect32XEmulation"] = True
        kernel_updates["ForceUnrollSubIter"] = self.FORCE_UNROLL_SUB_ITER
        kernel_updates["DepthU"] = self.DEPTH_U
        kernel_updates.setdefault("MIWaveTileA", 4)
        kernel_updates.setdefault("MIWaveTileB", 4)
        kernel_updates.setdefault("VectorWidthA", 1)
        kernel_updates.setdefault("VectorWidthB", 1)
        super().setup_method(method, kernel_updates=kernel_updates)

        self.q1s = 0
        self.q1e = self.num_vmfma // 4 - 1

        self.q2s = self.q1e + 1
        self.q2e = self.num_vmfma // 2 - 1

        self.q3s = self.q2e + 1
        self.q3e = self.num_vmfma // 4 * 3 - 1

        self.q4s = self.q3e + 1
        self.q4e = self.num_vmfma - 1

    validator_passes = [add_local_read_constraints, add_pack_constraints]

    def _make_valid_pack_group(self, base_idx: int) -> list[int]:
        """Creates a valid group of 10 packs with proper spacing for 4x4 MFMA TF32."""
        return (
            [base_idx] * 4 +      # CVT0 (packs 0-3)
            [base_idx] * 2 +      # 4x4 MFMAs (packs 4-5)
            [base_idx + 2] * 4    # CVT1 (packs 6-9) - needs +2 MFMA indices for 5 quad-cycle gap
        )

    def _make_base_schedule(self, packA0_schedule, packB0_schedule,
                            pack_alt_a=None, pack_alt_b=None,
                            n_lrs_a=2, n_lrs_b=2):
        """
        Build a schedule with LRs early in q1 (both guaranteed before q2s).
        """
        s = self.SUB_ITER_SUFFIX

        if pack_alt_a is None:
            pack_alt_a = self._make_valid_pack_group(self.q4s+2)
        if pack_alt_b is None:
            pack_alt_b = self._make_valid_pack_group(self.q3s+2)

        optSchedule = {
            # One SYNC at idx 1 guarantees both LRA0+LRB0 (needed before q2s).
            # Separate SYNCs for LRB{s} and LRA{s}.
            "SYNC": [[1, self.q3s+1, self.q4s+1]],

            "LRA0": [[0] * n_lrs_a],
            "PackA0": [packA0_schedule],

            "LRB0": [[0] * n_lrs_b],
            "PackB0": [packB0_schedule],

            f"LRB{s}": [[self.q3s] * n_lrs_b],
            f"PackB{s}": [pack_alt_b],

            f"LRA{s}": [[self.q4s] * n_lrs_a],
            f"PackA{s}": [pack_alt_a],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0+LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=f"Wait for LRB{s}s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=f"Wait for LRA{s}s"),
        ]

        return optSchedule, syncCode

    def _build_swap_schedule(self, vw, base_idx):
        """Build swap packs + regular pack groups for a given VectorWidth."""
        n_swaps = 4 * (vw - 1)
        return [base_idx] * n_swaps + self._make_valid_pack_group(base_idx) * vw

    @pytest.mark.parametrize("vw_a,vw_b,extra_kernel", [
        pytest.param(1, 1, {}, id="vw1_both"),
        pytest.param(2, 1, {}, id="vw2_a_only"),
        pytest.param(4, 1, {}, id="vw4_a_only"),
        pytest.param(2, 2, {"ProblemType": {"TLUB": True}}, id="vw2_both"),
    ])
    def test_passing_vw_combinations(self, vw_a, vw_b, extra_kernel):
        """Valid schedules with various VectorWidth combinations."""
        kernel = {"VectorWidthA": vw_a, "VectorWidthB": vw_b}
        kernel.update(extra_kernel)
        self.setup_method(kernel_updates=kernel)

        packA0 = self._build_swap_schedule(vw_a, 2)
        packB0 = self._build_swap_schedule(vw_b, 2)
        pack_alt_a = self._build_swap_schedule(vw_a, self.q4s + 2)
        pack_alt_b = self._build_swap_schedule(vw_b, self.q3s + 2)
        n_lrs_a = 8 if vw_a >= 2 else 2
        n_lrs_b = 8 if vw_b >= 2 else 2

        optSchedule, syncCode = self._make_base_schedule(
            packA0, packB0,
            pack_alt_a=pack_alt_a, pack_alt_b=pack_alt_b,
            n_lrs_a=n_lrs_a, n_lrs_b=n_lrs_b)
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_passing_multiple_groups_with_swaps(self):
        """VW_A=2, MIWaveTileA=4, MIWaveTileB=2. 2 groups of 10 regular packs + 4 swap packs on A, 2 groups on B."""
        self.setup_method(kernel_updates={"VectorWidthA": 2, "VectorWidthB": 1, "MIWaveTileA": 4, "MIWaveTileB": 2})

        # Both LRA0 and LRB0 at idx 0, guaranteed by SYNC at idx 1
        # 4 swap packs + 2 groups of 10 regular packs = 24 total PackA0
        # 8 LRs needed: dsReadConvTable has 8 entries for VW=2
        packA0_schedule = (
            [2] * 4 +
            self._make_valid_pack_group(2) +
            self._make_valid_pack_group(2)
        )

        optSchedule = {
            "SYNC": [[1]],

            "LRA0": [[0] * 8],
            "PackA0": [packA0_schedule],

            "LRB0": [[0] * 2],
            "PackB0": [self._make_valid_pack_group(2) + self._make_valid_pack_group(2)],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0+LRB0"),
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_fail_swap_before_lr_done(self):
        """SwapPacks issued before LR SWaitCnt. Expected: 'issued too early' error."""
        self.setup_method(kernel_updates={"VectorWidthA": 2, "VectorWidthB": 1})

        # Swap packs at idx 0, but SYNC is at idx 1, so they're before LR is guaranteed
        packA0 = [0] * 4 + self._make_valid_pack_group(2) * 2
        packB0 = self._make_valid_pack_group(2)
        pack_alt_a = [self.q4s+2] * 4 + self._make_valid_pack_group(self.q4s+2) * 2

        optSchedule, syncCode = self._make_base_schedule(packA0, packB0, pack_alt_a=pack_alt_a, n_lrs_a=8)
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=0 issued too early, must be issued after idx=1 (because of LRA0 issued @ idx=0).")

    def test_fail_regular_pack_before_swap_done(self):
        """First CVT0 pack issued before its swap dependency is done."""
        self.setup_method(kernel_updates={"VectorWidthA": 2, "VectorWidthB": 1})

        # Swap packs at idx 2, group 0's CVT0 at idx 1 (before swaps!)
        # Group 0's CVT0 pack 0 reads reg 1 which was swapped by swap 0, so it depends on swap 0.
        packA0 = (
            [2] * 4 +     # 4 swap packs at idx 2
            [1] * 4 +     # CVT0 group 0 at idx 1, BEFORE the swap packs!
            [2] * 2 +     # 4x4 MFMAs group 0
            [4] * 4 +     # CVT1 group 0
            self._make_valid_pack_group(2)  # group 1 valid
        )
        packB0 = self._make_valid_pack_group(2)
        pack_alt_a = [self.q4s+2] * 4 + self._make_valid_pack_group(self.q4s+2) * 2

        optSchedule, syncCode = self._make_base_schedule(packA0, packB0, pack_alt_a=pack_alt_a, n_lrs_a=8)
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, "PackA0 @ idx=1 issued too early, must be issued after idx=2 (because of PackA0 issued @ idx=2).")

    def test_swap_depends_on_specific_lrs_vw4(self):
        """VW=4: T0 swaps (0,1,2) depend only on T0 LRs (LR0-LR3). Placing them
        after a partial guarantee (dscnt=4, guaranteeing LR0-LR3) should pass."""
        self.setup_method(kernel_updates={"VectorWidthA": 4, "VectorWidthB": 1})
        s = self.SUB_ITER_SUFFIX

        # 8 A-side LRs all at idx 0, 2 B-side LRs at idx 0.
        # SYNC(dscnt=4) at idx 1: leaves 4 in flight (LRA0[4..7]), guarantees LRA0[0..3] + LRB0[0..1].
        # SYNC(dscnt=0) at idx 6: guarantees remaining LRA0[4..7].
        #
        # Under T/X interleave mapping (_logical_reg_to_lr_index):
        #   LR0-LR3 are T0 LRs (regs with idx%8 < 4)
        #   LR4-LR7 are X0 LRs (regs with idx%8 >= 4)
        #
        # T0 swaps (0,1,2,6,7,10) only depend on LR0-LR3 → can be placed early.
        # X0 swaps (3,4,5,8,9,11) depend on LR4-LR7 → must wait for full guarantee.
        swap_schedule = (
            [2] * 3 + # swaps 0-2 (T0: regs 1↔8, 2↔16, 3↔24) → ok after idx 1
            [7] * 3 + # swaps 3-5 (X0: regs 5↔12, 6↔20, 7↔28) → after idx 6
            [2] * 2 + # swaps 6-7 (T0: regs 10↔17, 11↔25) → ok after idx 1
            [7] * 2 + # swaps 8-9 (X0: regs 14↔21, 15↔29) → after idx 6
            [2] +     # swap 10 (T0: regs 19↔26) → ok after idx 1
            [7]       # swap 11 (X0: regs 23↔30) → after idx 6
        )
        packA0 = list(swap_schedule) + self._make_valid_pack_group(8) * 4
        packB0 = self._make_valid_pack_group(2)
        pack_alt_a = [self.q4s+2] * 12 + self._make_valid_pack_group(self.q4s+2) * 4

        optSchedule = {
            "SYNC": [[1, 6, self.q3s+1, self.q4s+1]],

            # LRB0 before LRA0 so dscnt=4 skips only LRA0[4..7] (most recent 4),
            # guaranteeing LRB0[0..1] + LRA0[0..3].
            "LRB0": [[0] * 2],
            "PackB0": [packB0],

            "LRA0": [[0] * 8],
            "PackA0": [packA0],

            f"LRB{s}": [[self.q3s] * 2],
            f"PackB{s}": [self._make_valid_pack_group(self.q3s+2)],

            f"LRA{s}": [[self.q4s] * 8],
            f"PackA{s}": [pack_alt_a],
        }

        syncCode = [
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Partial: guarantee LRA0[0..3]+LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Full: guarantee remaining LRA0[4..7]"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=f"Wait for LRB{s}s"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=f"Wait for LRA{s}s"),
        ]

        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

    def test_cvt0_depends_on_specific_swaps_vw4(self):
        """VW=4: group 2's CVT0 packs depend on swaps that touch regs 16-23,
        not on all swaps. Placing group 2's CVT0 before swaps that only affect
        other groups should pass."""
        self.setup_method(kernel_updates={"VectorWidthA": 4, "VectorWidthB": 1})

        # With VW=4, 4 groups, the swap register pairs are:
        #   swap 0: 1↔8    (groups 0,1)
        #   swap 1: 2↔16   (groups 0,2)  ← group 2 depends on this
        #   swap 2: 3↔24   (groups 0,3)
        #   swap 3: 5↔12   (groups 0,1)
        #   swap 4: 6↔20   (groups 0,2)  ← group 2 depends on this
        #   swap 5: 7↔28   (groups 0,3)
        #   swap 6: 10↔17  (groups 1,2)  ← group 2 depends on this
        #   swap 7: 11↔25  (groups 1,3)
        #   swap 8: 14↔21  (groups 1,2)  ← group 2 depends on this
        #   swap 9: 15↔29  (groups 1,3)
        #   swap 10: 19↔26 (groups 2,3)  ← group 2 depends on this
        #   swap 11: 23↔30 (groups 2,3)  ← group 2 depends on this
        #
        # Group 2 (regs 16-23) depends on swaps 1,4,6,8,10,11.
        # Group 2 does NOT depend on swaps 0,2,3,5,7,9.
        #
        # Strategy: place all swaps at idx 2, but put group 2's CVT0 packs at idx 3
        # (after all swaps). This should pass for group 2 since its dependencies are satisfied.
        # To make the test meaningful, place swaps 0,2,3,5,7,9 at idx 4 (AFTER group 2's CVT0).
        # Group 2's CVT0 at idx 3 should still pass because it doesn't depend on those late swaps.

        swap_schedule = (
            [4] +     # swap 0 (groups 0,1) → late
            [2] +     # swap 1 (groups 0,2) → early
            [4] +     # swap 2 (groups 0,3) → late
            [4] +     # swap 3 (groups 0,1) → late
            [2] +     # swap 4 (groups 0,2) → early
            [4] +     # swap 5 (groups 0,3) → late
            [2] +     # swap 6 (groups 1,2) → early
            [4] +     # swap 7 (groups 1,3) → late
            [2] +     # swap 8 (groups 1,2) → early
            [4] +     # swap 9 (groups 1,3) → late
            [2] +     # swap 10 (groups 2,3) → early
            [2]       # swap 11 (groups 2,3) → early
        )
        # Group 0 and 1 CVT0 packs at idx 5 (after ALL swaps including late ones at idx 4)
        # Group 2 CVT0 packs at idx 3 (after early swaps at idx 2 but before late swaps at idx 4)
        # Group 3 CVT0 packs at idx 5 (after ALL swaps)
        packA0 = (
            list(swap_schedule) +
            [5] * 4 + [5] * 2 + [7] * 4 +    # group 0: CVT0@5, MFMA@5, CVT1@7
            [5] * 4 + [5] * 2 + [7] * 4 +    # group 1: CVT0@5, MFMA@5, CVT1@7
            [3] * 4 + [3] * 2 + [5] * 4 +    # group 2: CVT0@3 (early!), MFMA@3, CVT1@5
            [5] * 4 + [5] * 2 + [7] * 4      # group 3: CVT0@5, MFMA@5, CVT1@7
        )
        packB0 = self._make_valid_pack_group(2)
        pack_alt_a = [self.q4s+2] * 12 + self._make_valid_pack_group(self.q4s+2) * 4

        optSchedule, syncCode = self._make_base_schedule(packA0, packB0, pack_alt_a=pack_alt_a, n_lrs_a=8)
        self.validate(optSchedule, syncCode, 1, 2, 2, 0, None)

