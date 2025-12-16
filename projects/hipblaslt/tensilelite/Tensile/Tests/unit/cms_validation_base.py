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
from abc import abstractmethod
from typing import Any
import unittest

from test_CustomSchedule import create_base_kernel, ScheduleInfo


class CMSValidationTestBase(unittest.TestCase):
    """
    Base class for CMS validation tests that provides common setup and helper methods.
    """
    @abstractmethod
    def validation_function(self, sched, kernel_dict, codePathIdx):
        """
        Method that must be implemented by subclasses.
        NOTE: Don't use abstractmethod. Pytest will fail if this method is abstract since it tries instantiating this class.
        Should call the appropriate validation function with the provided arguments.
        
        Args:
            sched: ScheduleInfo object to validate
            kernel_dict: Dictionary containing kernel configuration
            codePathIdx: Code path index to validate
            
        Returns:
            Tuple of (status: bool, message: str)
        """
        raise NotImplementedError("Subclasses must implement validation_function")
    
    def setUp(self, kernel_updates: dict[str, Any] | None = None):
        """Initialize kernel and compute number of VMFMAs."""
        self.kernel = create_base_kernel()
        if kernel_updates:
            self.kernel.update(kernel_updates)
        
        self.num_vmfma = self.kernel["MIWaveTileA"] * self.kernel["MIWaveTileB"]
        self.num_vmfma *= self.kernel["DepthU"] // self.kernel["MatrixInstruction"][2]
        if self.kernel.get("UseF32XEmulation", False):
            self.num_vmfma *= 3

    def validate(
        self,
        optSchedule,
        syncCode,
        numCodePaths: int,
        nglshift: int,
        nllshift: int,
        codePathIdx: int,
        expected_message: str | None,
        nllZeroDscnt: bool = False,
        mfmaReorder: list[int] = None,
        snopCode: list[Any] = None,
    ):
        """
        Creates a ScheduleInfo and validates it using the validation function from the subclass.
        
        Args:
            optSchedule: The schedule dictionary mapping instruction types to indices
            syncCode: List of sync instructions (SWaitCnt, SBarrier, etc.)
            numCodePaths: Number of code paths in the schedule
            nglshift: NGL shift value
            nllshift: NLL shift value
            codePathIdx: Code path index to validate
            expected_message: Expected error message (None if validation should pass, str if validation should fail)
            nllZeroDscnt: Whether to use zero dscnt for NLL loop (default: False)
            mfmaReorder: List of MFMA reorder indices
            snopCode: List of SNOP instructions
        """
        if mfmaReorder is None:
            mfmaReorder = []
        if snopCode is None:
            snopCode = []
        
        sched = ScheduleInfo(numCodePaths, self.num_vmfma, optSchedule, syncCode, nglshift, nllshift, nllZeroDscnt, mfmaReorder, snopCode)

        status, message = self.validation_function(sched, {"kernel": self.kernel}, codePathIdx)
        
        if expected_message is None:
            assert status, f"Schedule should have passed validation but did not. {message}"
        else:
            assert not status, f"Schedule should have failed but passed."
            assert message == expected_message, f"Expected: {expected_message}, Got: {message}"

