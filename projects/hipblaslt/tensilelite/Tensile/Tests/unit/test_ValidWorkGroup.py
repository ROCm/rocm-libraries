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
from unittest.mock import patch

from Tensile.TensileLogic.ValidWorkGroup import _validateWorkGroup


@pytest.mark.unit
class TestValidWorkGroup:
    """Tests for ValidWorkGroup validation function"""

    def test_validate_workgroup_valid_solution(self):
        """Test validation with a valid solution"""
        solution = {
            "SolutionIndex": 0,
            "WorkGroup": [16, 16, 1],
            "ThreadTile": [4, 4],
            "Valid": True
        }
        filepath = Path("test.yaml")

        with patch('Tensile.TensileLogic.ValidWorkGroup.validateWorkGroup'):
            result = _validateWorkGroup(solution, filepath)
            assert result is True

    def test_validate_workgroup_invalid_solution(self):
        """Test validation with invalid solution (Valid=False)"""
        solution = {
            "SolutionIndex": 1,
            "WorkGroup": [0, 0, 0],  # Invalid
            "Valid": False
        }
        filepath = Path("test.yaml")

        result = _validateWorkGroup(solution, filepath)
        assert result is False

    def test_validate_workgroup_assertion_error(self):
        """Test validation when validateWorkGroup raises AssertionError"""
        solution = {
            "SolutionIndex": 2,
            "WorkGroup": [16, 16, 1],
            "Valid": True
        }
        filepath = Path("test.yaml")

        with patch('Tensile.TensileLogic.ValidWorkGroup.validateWorkGroup',
                   side_effect=AssertionError("Test error")):
            result = _validateWorkGroup(solution, filepath)
            assert result is False

    def test_validate_workgroup_with_different_dimensions(self):
        """Test with different workgroup dimensions"""
        test_cases = [
            ([8, 8, 1], True),
            ([32, 8, 1], True),
            ([16, 32, 1], True),
        ]

        for workgroup, expected_valid in test_cases:
            solution = {
                "SolutionIndex": 0,
                "WorkGroup": workgroup,
                "Valid": expected_valid
            }
            filepath = Path("test.yaml")

            with patch('Tensile.TensileLogic.ValidWorkGroup.validateWorkGroup'):
                result = _validateWorkGroup(solution, filepath)
                if expected_valid:
                    assert result is True
