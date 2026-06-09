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
from Tensile.Common.DataType import DataType

from Tensile.TensileLogic.ValidMatrixInstruction import _validateMatrixInstruction


@pytest.fixture
def base_valid_solution():
    """Create a base valid solution with all required MI fields"""
    return {
        "SolutionIndex": 0,
        "Valid": True,
        "ISA": (9, 4, 2),  # Use gfx942 which is well-supported
        "MatrixInstruction": [16, 16, 1, 4],
        "EnableMatrixInstruction": True,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 1,
        "MatrixInstB": 4,
        "MatrixInstBM": 1,
        "MIWaveTile": [4, 1],
        "MIWaveGroup": [2, 2],
        "MIInputPerThread": 4,
        "MIInputPerThreadA": 4,
        "MIInputPerThreadB": 4,
        "WorkGroup": [256, 4, 1],
        "WavefrontSize": 64,
        "ThreadTile": [1, 1],
        "ProblemType": {
            "DataType": DataType('s'),
            "MacDataTypeA": DataType('s'),
            "MacDataTypeB": DataType('s'),
            "Sparse": 0,
        }
    }


@pytest.mark.unit
class TestValidMatrixInstruction:
    """Tests for ValidMatrixInstruction validation function"""

    def test_validate_matrix_instruction_disabled(self, base_valid_solution):
        """Test validation with MI disabled"""
        base_valid_solution["MatrixInstruction"] = []
        base_valid_solution["EnableMatrixInstruction"] = False

        isaInfoMap = {}
        filepath = Path("test.yaml")

        result = _validateMatrixInstruction(base_valid_solution, isaInfoMap, filepath)
        assert result is True

    def test_validate_matrix_instruction_invalid_solution_flag(self, base_valid_solution):
        """Test validation with solution Valid flag set to False"""
        base_valid_solution["Valid"] = False
        # Also disable MI to avoid ISA lookup
        base_valid_solution["MatrixInstruction"] = []
        base_valid_solution["EnableMatrixInstruction"] = False

        isaInfoMap = {}
        filepath = Path("test.yaml")

        result = _validateMatrixInstruction(base_valid_solution, isaInfoMap, filepath)
        assert result is False

    def test_validate_matrix_instruction_invalid_mi_length(self, base_valid_solution):
        """Test validation rejects MI with wrong length"""
        base_valid_solution["MatrixInstruction"] = [16, 16, 1]  # Only 3 elements

        isaInfoMap = {}
        filepath = Path("test.yaml")

        result = _validateMatrixInstruction(base_valid_solution, isaInfoMap, filepath)
        assert result is False

    def test_validate_matrix_instruction_enabled_but_empty(self, base_valid_solution):
        """Test validation rejects empty MI when enabled"""
        base_valid_solution["MatrixInstruction"] = []
        base_valid_solution["EnableMatrixInstruction"] = True

        isaInfoMap = {}
        filepath = Path("test.yaml")

        result = _validateMatrixInstruction(base_valid_solution, isaInfoMap, filepath)
        assert result is False
