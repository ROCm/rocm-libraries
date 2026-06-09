################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import pytest
from pathlib import Path

from Tensile.TensileLogic.ValidWorkGroup import _validateWorkGroup


@pytest.mark.unit
class TestValidWorkGroup:
    """Tests for ValidWorkGroup validation function"""

    def test_validate_workgroup_valid_16x16x1(self):
        """Test validation with valid 16x16x1 workgroup"""
        solution = {
            "SolutionIndex": 0,
            "WorkGroup": [16, 16, 1],
            "Valid": True
        }
        filepath = Path("test.yaml")

        result = _validateWorkGroup(solution, filepath)
        assert result is True

    def test_validate_workgroup_valid_8x8x1(self):
        """Test validation with valid 8x8x1 workgroup"""
        solution = {
            "SolutionIndex": 0,
            "WorkGroup": [8, 8, 1],
            "Valid": True
        }
        filepath = Path("test.yaml")

        result = _validateWorkGroup(solution, filepath)
        assert result is True

    def test_validate_workgroup_valid_32x8x1(self):
        """Test validation with valid 32x8x1 workgroup"""
        solution = {
            "SolutionIndex": 0,
            "WorkGroup": [32, 8, 1],
            "Valid": True
        }
        filepath = Path("test.yaml")

        result = _validateWorkGroup(solution, filepath)
        assert result is True

    def test_validate_workgroup_invalid_solution_flag(self):
        """Test validation with solution Valid flag set to False"""
        solution = {
            "SolutionIndex": 1,
            "WorkGroup": [16, 16, 1],
            "Valid": False
        }
        filepath = Path("test.yaml")

        result = _validateWorkGroup(solution, filepath)
        assert result is False

    def test_validate_workgroup_missing_workgroup_key(self):
        """Test validation rejects solution missing WorkGroup key"""
        solution = {
            "SolutionIndex": 2,
            "Valid": True
        }
        filepath = Path("test.yaml")

        result = _validateWorkGroup(solution, filepath)
        assert result is False

    def test_validate_workgroup_invalid_dimensions(self):
        """Test validation rejects invalid workgroup dimensions"""
        # [0, 0, 0] is not in valid workgroups list
        solution = {
            "SolutionIndex": 3,
            "WorkGroup": [0, 0, 0],
            "Valid": True
        }
        filepath = Path("test.yaml")

        result = _validateWorkGroup(solution, filepath)
        assert result is False

    def test_validate_workgroup_invalid_negative(self):
        """Test validation rejects negative workgroup dimensions"""
        solution = {
            "SolutionIndex": 4,
            "WorkGroup": [-1, 16, 1],
            "Valid": True
        }
        filepath = Path("test.yaml")

        result = _validateWorkGroup(solution, filepath)
        assert result is False

    def test_validate_workgroup_invalid_arbitrary(self):
        """Test validation rejects arbitrary invalid workgroup"""
        # [999, 999, 999] is not a valid workgroup
        solution = {
            "SolutionIndex": 5,
            "WorkGroup": [999, 999, 999],
            "Valid": True
        }
        filepath = Path("test.yaml")

        result = _validateWorkGroup(solution, filepath)
        assert result is False
