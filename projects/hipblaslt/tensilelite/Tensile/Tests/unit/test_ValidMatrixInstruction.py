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
from unittest.mock import patch, Mock

from Tensile.TensileLogic.ValidMatrixInstruction import _validateMatrixInstruction


@pytest.mark.unit
class TestValidMatrixInstruction:
    """Tests for ValidMatrixInstruction validation function"""

    def test_validate_matrix_instruction_valid(self):
        """Test validation with valid matrix instruction"""
        solution = {
            "SolutionIndex": 0,
            "MatrixInstruction": [16, 16, 1, 4],
            "Valid": True
        }
        isaInfoMap = {}
        filepath = Path("test.yaml")

        with patch('Tensile.TensileLogic.ValidMatrixInstruction.validateMIParameters'):
            result = _validateMatrixInstruction(solution, isaInfoMap, filepath)
            assert result is True

    def test_validate_matrix_instruction_invalid(self):
        """Test validation with invalid solution (Valid=False)"""
        solution = {
            "SolutionIndex": 1,
            "MatrixInstruction": [0, 0, 0, 0],
            "Valid": False
        }
        isaInfoMap = {}
        filepath = Path("test.yaml")

        result = _validateMatrixInstruction(solution, isaInfoMap, filepath)
        assert result is False

    def test_validate_matrix_instruction_assertion_error(self):
        """Test validation when validateMIParameters raises AssertionError"""
        solution = {
            "SolutionIndex": 2,
            "MatrixInstruction": [16, 16, 1, 4],
            "Valid": True
        }
        isaInfoMap = {}
        filepath = Path("test.yaml")

        with patch('Tensile.TensileLogic.ValidMatrixInstruction.validateMIParameters',
                   side_effect=AssertionError("Invalid MI parameters")):
            result = _validateMatrixInstruction(solution, isaInfoMap, filepath)
            assert result is False

    def test_validate_different_matrix_instructions(self):
        """Test with different matrix instruction formats"""
        test_cases = [
            ([16, 16, 1, 4], True),
            ([32, 32, 1, 2], True),
            ([4, 4, 1, 16], True),
        ]

        isaInfoMap = {}
        for mi, expected_valid in test_cases:
            solution = {
                "SolutionIndex": 0,
                "MatrixInstruction": mi,
                "Valid": expected_valid
            }
            filepath = Path("test.yaml")

            with patch('Tensile.TensileLogic.ValidMatrixInstruction.validateMIParameters'):
                result = _validateMatrixInstruction(solution, isaInfoMap, filepath)
                if expected_valid:
                    assert result is True

    def test_validate_with_extended_format(self):
        """Test with extended matrix instruction format (9 parameters)"""
        solution = {
            "SolutionIndex": 0,
            "MatrixInstruction": [32, 32, 1, 2, 1, 4, 1, 2, 2],
            "Valid": True
        }
        isaInfoMap = {}
        filepath = Path("test.yaml")

        with patch('Tensile.TensileLogic.ValidMatrixInstruction.validateMIParameters'):
            result = _validateMatrixInstruction(solution, isaInfoMap, filepath)
            assert result is True
