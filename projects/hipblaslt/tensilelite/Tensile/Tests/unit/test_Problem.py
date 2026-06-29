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
from unittest.mock import Mock, MagicMock, patch
from Tensile.Common.DataType import DataType
import sys
from importlib import import_module


# Import the Problem.py module directly (not through __init__.py which does from .Problem import *)
ProblemModule = import_module('Tensile.SolutionStructs.Problem')
from Tensile.SolutionStructs.Problem import (
    Problem,
    ProblemType,
    ProblemSizesMock,
    ProblemSizesMockDummy,
)


@pytest.mark.unit
class TestProblemClass:
    """Tests for Problem class"""

    def test_problem_init_with_sizes(self):
        """Test Problem initialization with sizes"""
        sizes = [128, 128, 64]
        prob = Problem(sizes=sizes)

        assert prob.sizes == tuple(sizes)
        assert prob.stridesA is None
        assert prob.stridesB is None
        assert prob.stridesC is None
        assert prob.stridesD is None

    def test_problem_init_with_strides(self):
        """Test Problem initialization with strides"""
        sizes = [128, 128, 64]
        stridesA = [1, 128]
        stridesB = [1, 128]
        stridesC = [1, 128]
        stridesD = [1, 128]

        prob = Problem(
            sizes=sizes,
            stridesA=stridesA,
            stridesB=stridesB,
            stridesC=stridesC,
            stridesD=stridesD
        )

        assert prob.sizes == tuple(sizes)
        assert prob.stridesA == tuple(stridesA)
        assert prob.stridesB == tuple(stridesB)
        assert prob.stridesC == tuple(stridesC)
        assert prob.stridesD == tuple(stridesD)

    def test_problem_init_with_count(self):
        """Test Problem initialization with count"""
        sizes = [64, 64, 32]
        count = 10

        prob = Problem(sizes=sizes, count=count)

        assert prob.count == count

    def test_problem_str(self):
        """Test Problem string representation"""
        sizes = [128, 128, 64]
        prob = Problem(sizes=sizes)

        result = str(prob)
        assert "sizes:" in result
        assert "128" in result

    def test_problem_str_with_strides(self):
        """Test Problem string representation with strides"""
        sizes = [64, 64, 32]
        stridesA = [1, 64]

        prob = Problem(sizes=sizes, stridesA=stridesA)

        result = str(prob)
        assert "sizes:" in result
        assert "stridesA:" in result


@pytest.mark.unit
class TestGetRealDataTypeHelpers:
    """Tests for getRealDataTypeA/B helper functions"""

    def test_get_real_data_type_a_regular(self):
        """Test getRealDataTypeA with regular data type"""
        from rocisa.enum import DataTypeEnum
        dt = DataType(DataTypeEnum.Float)

        result = ProblemModule.getRealDataTypeA(dt)
        assert result == dt

    def test_get_real_data_type_b_regular(self):
        """Test getRealDataTypeB with regular data type"""
        from rocisa.enum import DataTypeEnum
        dt = DataType(DataTypeEnum.Float)

        result = ProblemModule.getRealDataTypeB(dt)
        assert result == dt

    def test_get_real_data_type_a_float8(self):
        """Test getRealDataTypeA with Float8 type returns same type"""
        from rocisa.enum import DataTypeEnum
        dt = DataType(DataTypeEnum.Float8)

        result = ProblemModule.getRealDataTypeA(dt)
        # For standard types, should return same type
        assert result == dt

    def test_get_real_data_type_b_bfloat8(self):
        """Test getRealDataTypeB with BFloat8 type returns same type"""
        from rocisa.enum import DataTypeEnum
        dt = DataType(DataTypeEnum.BFloat8)

        result = ProblemModule.getRealDataTypeB(dt)
        # For standard types, should return same type
        assert result == dt

    def test_get_real_data_type_a_int8(self):
        """Test getRealDataTypeA with Int8"""
        from rocisa.enum import DataTypeEnum
        dt = DataType(DataTypeEnum.Int8)

        result = ProblemModule.getRealDataTypeA(dt)
        # Should handle Int8
        assert result == dt or result.value == DataTypeEnum.Int8

    def test_get_real_data_type_mixed_float8_bfloat8(self):
        """Test getRealDataType with mixed Float8BFloat8 type splits correctly"""
        from rocisa.enum import DataTypeEnum
        # Float8BFloat8 means A is Float8, B is BFloat8
        dt = DataType(DataTypeEnum.Float8BFloat8)

        result_a = ProblemModule.getRealDataTypeA(dt)
        result_b = ProblemModule.getRealDataTypeB(dt)

        # A should be Float8
        expected_a = DataType(DataTypeEnum.Float8)
        assert result_a == expected_a, f"Expected {expected_a} for A, got {result_a}"
        # B should be BFloat8
        expected_b = DataType(DataTypeEnum.BFloat8)
        assert result_b == expected_b, f"Expected {expected_b} for B, got {result_b}"

    def test_get_real_data_type_mixed_bfloat8_float8(self):
        """Test getRealDataType with mixed BFloat8Float8 type splits correctly"""
        from rocisa.enum import DataTypeEnum
        # BFloat8Float8 means A is BFloat8, B is Float8
        dt = DataType(DataTypeEnum.BFloat8Float8)

        result_a = ProblemModule.getRealDataTypeA(dt)
        result_b = ProblemModule.getRealDataTypeB(dt)

        # A should be BFloat8
        expected_a = DataType(DataTypeEnum.BFloat8)
        assert result_a == expected_a, f"Expected {expected_a} for A, got {result_a}"
        # B should be Float8
        expected_b = DataType(DataTypeEnum.Float8)
        assert result_b == expected_b, f"Expected {expected_b} for B, got {result_b}"

    def test_get_real_data_type_mixed_float8_bfloat8_fnuz(self):
        """Test getRealDataType with mixed Float8BFloat8_fnuz type splits correctly"""
        from rocisa.enum import DataTypeEnum
        # Float8BFloat8_fnuz means A is Float8_fnuz, B is BFloat8_fnuz
        dt = DataType(DataTypeEnum.Float8BFloat8_fnuz)

        result_a = ProblemModule.getRealDataTypeA(dt)
        result_b = ProblemModule.getRealDataTypeB(dt)

        # A should be Float8_fnuz
        expected_a = DataType(DataTypeEnum.Float8_fnuz)
        assert result_a == expected_a, f"Expected {expected_a} for A, got {result_a}"
        # B should be BFloat8_fnuz
        expected_b = DataType(DataTypeEnum.BFloat8_fnuz)
        assert result_b == expected_b, f"Expected {expected_b} for B, got {result_b}"

    def test_get_real_data_type_mixed_bfloat8_float8_fnuz(self):
        """Test getRealDataType with mixed BFloat8Float8_fnuz type splits correctly"""
        from rocisa.enum import DataTypeEnum
        # BFloat8Float8_fnuz means A is BFloat8_fnuz, B is Float8_fnuz
        dt = DataType(DataTypeEnum.BFloat8Float8_fnuz)

        result_a = ProblemModule.getRealDataTypeA(dt)
        result_b = ProblemModule.getRealDataTypeB(dt)

        # A should be BFloat8_fnuz
        expected_a = DataType(DataTypeEnum.BFloat8_fnuz)
        assert result_a == expected_a, f"Expected {expected_a} for A, got {result_a}"
        # B should be Float8_fnuz
        expected_b = DataType(DataTypeEnum.Float8_fnuz)
        assert result_b == expected_b, f"Expected {expected_b} for B, got {result_b}"


@pytest.mark.unit
class TestProblemSizesMock:
    """Tests for ProblemSizesMock adapter class"""

    def test_problem_sizes_mock_basic(self):
        """Test ProblemSizesMock initialization"""
        exact_logic = [
            ([128, 128, 64], "solution1"),
            ([256, 256, 128], "solution2")
        ]

        mock = ProblemSizesMock(exact_logic)

        assert len(mock.problems) == 2
        assert all(isinstance(p, Problem) for p in mock.problems)


@pytest.mark.unit
class TestProblemSizesMockDummy:
    """Tests for ProblemSizesMockDummy class"""

    def test_problem_sizes_mock_dummy(self):
        """Test ProblemSizesMockDummy initialization"""
        dummy = ProblemSizesMockDummy()

        assert len(dummy.problems) == 1
        assert isinstance(dummy.problems[0], Problem)
        assert dummy.problems[0].sizes == (128, 128, 1, 512)  # It's a tuple, not a list


@pytest.mark.unit
class TestGetBiasDataTypeListDefault:
    """Tests for getBiasDataTypeListDefault function"""

    def test_get_bias_data_type_list_default_single(self):
        """Test getBiasDataTypeListDefault with single precision"""
        problem_type = Mock()
        problem_type.__getitem__ = Mock(side_effect=lambda x: DataType('s') if 'DataType' in x or 'ComputeDataType' in x or 'DestDataType' in x else None)

        result = ProblemModule.getBiasDataTypeListDefault(problem_type)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_get_bias_data_type_list_default_filters_small(self):
        """Test that getBiasDataTypeListDefault filters out small types"""
        # Create a problem type with mix of small and large types
        # Int8 has numBytes=1, should be filtered out
        # Float has numBytes=4, should be included
        problem_type = Mock()
        problem_type.__getitem__ = Mock(side_effect=lambda x: DataType('I8') if x == "DataType" else DataType('s'))

        result = ProblemModule.getBiasDataTypeListDefault(problem_type)
        # Should filter out types with numBytes <= 1
        assert isinstance(result, list)
        # Verify no element in result has numBytes <= 1
        for dtype in result:
            assert dtype.numBytes() > 1, f"Type {dtype} with {dtype.numBytes()} bytes should be filtered out"


@pytest.mark.unit
class TestProblemTypeEnum:
    """Tests for problemTypeToEnum function"""

    def test_problem_type_to_enum_basic(self):
        """Test problemTypeToEnum converts data types"""
        problem_type = {
            "DataType": Mock(value=0),
            "MacDataTypeA": Mock(value=0),
            "MacDataTypeB": Mock(value=0),
            "DataTypeA": Mock(value=0),
            "DataTypeB": Mock(value=0),
            "DataTypeE": Mock(value=0),
            "DataTypeAmaxD": Mock(value=0),
            "DestDataType": Mock(value=0),
            "ComputeDataType": Mock(value=0),
            "BiasDataTypeList": [Mock(value=1), Mock(value=2)],
            "ActivationComputeDataType": Mock(value=0),
            "ActivationType": Mock(value=3),
            "F32XdlMathOp": Mock(value=0)
        }

        ProblemModule.problemTypeToEnum(problem_type)

        assert problem_type["DataType"] == 0
        assert problem_type["MacDataTypeA"] == 0
        assert problem_type["BiasDataTypeList"] == [1, 2]

    def test_problem_type_to_enum_with_metadata(self):
        """Test problemTypeToEnum with metadata"""
        problem_type = {
            "DataType": Mock(value=0),
            "MacDataTypeA": Mock(value=0),
            "MacDataTypeB": Mock(value=0),
            "DataTypeA": Mock(value=0),
            "DataTypeB": Mock(value=0),
            "DataTypeE": Mock(value=0),
            "DataTypeAmaxD": Mock(value=0),
            "DestDataType": Mock(value=0),
            "ComputeDataType": Mock(value=0),
            "BiasDataTypeList": [],
            "ActivationComputeDataType": Mock(value=0),
            "ActivationType": Mock(value=0),
            "F32XdlMathOp": Mock(value=0),
            "DataTypeMetadata": Mock(value=5)
        }

        ProblemModule.problemTypeToEnum(problem_type)

        assert problem_type["DataTypeMetadata"] == 5


@pytest.mark.unit
class TestProblemTypeAssignDerivedParameters:
    """Tests for ProblemType.assignDerivedParameters static method"""

    def test_assign_derived_parameters_basic(self):
        """Test assignDerivedParameters with basic state"""
        state = {
            "IndexAssignmentsA": [0, 2],
            "IndexAssignmentsB": [1, 2],
            "IndexAssignmentsMetadata": [0, 2],  # Required for Sparse handling
            "NumIndicesC": 2,
            "AllowNoFreeDims": False,
            "MXBlockA": 0,
            "MXBlockB": 0,
            "Sparse": 0
        }

        ProblemType.assignDerivedParameters(state, printIndexAssignmentInfo=False)

        assert "AssignedDerivedParameters" in state
        assert state["AssignedDerivedParameters"] == True
        assert "TotalIndices" in state
        assert "IndicesFree" in state
        assert "IndicesBatch" in state
        assert "IndicesSummation" in state

    def test_assign_derived_parameters_already_assigned(self):
        """Test assignDerivedParameters skips if already assigned"""
        state = {
            "AssignedDerivedParameters": True
        }

        # Should return early
        ProblemType.assignDerivedParameters(state, printIndexAssignmentInfo=False)

        # Should not add new keys
        assert "TotalIndices" not in state

    def test_assign_derived_parameters_with_batch(self):
        """Test assignDerivedParameters with batch indices"""
        state = {
            "IndexAssignmentsA": [0, 2, 3],
            "IndexAssignmentsB": [1, 2, 3],
            "IndexAssignmentsMetadata": [0, 2, 3],  # Required for Sparse handling
            "NumIndicesC": 3,
            "AllowNoFreeDims": False,
            "MXBlockA": 0,
            "MXBlockB": 0,
            "Sparse": 0
        }

        ProblemType.assignDerivedParameters(state, printIndexAssignmentInfo=False)

        assert len(state["IndicesBatch"]) > 0


@pytest.mark.unit
class TestConvertLeadingDims:
    """Tests for ExactList.convertLeadingDims static method"""

    def test_convert_leading_dims_basic(self):
        """Test convertLeadingDims with basic inputs"""
        problem_type = {
            "NumIndicesC": 2,
            "IndexAssignmentsLD": [3, 4, 5, 6],
            "IndexAssignmentsA": [0, 2],
            "IndexAssignmentsB": [1, 2]
        }
        problem_size = (128, 128, 64, 128, 128, 128, 128)

        result = ProblemModule.ExactList.convertLeadingDims(problem_type, problem_size)

        assert isinstance(result, tuple)
        assert len(result) == 7  # Same as input problem_size
        # First NumIndicesC elements should be unchanged
        assert result[0] == 128
        assert result[1] == 128
        # LD entries computed as max(problemSize[i], sizes derived from index assignments)
        # LDD, LDC should be >= 128, LDA, LDB >= 128
        assert result[3] >= 128  # LDD
        assert result[4] >= 128  # LDC
        assert result[5] >= 128  # LDA
        assert result[6] >= 128  # LDB

    def test_convert_leading_dims_with_strides(self):
        """Test convertLeadingDims with stride information"""
        problem_type = {
            "NumIndicesC": 2,
            "IndexAssignmentsLD": [3, 4, 5, 6],
            "IndexAssignmentsA": [0, 2],
            "IndexAssignmentsB": [1, 2]
        }
        problem_size = (64, 64, 32, 64, 64, 64, 64)
        stridesA = (1, 100)
        stridesB = (1, 100)
        stridesC = (1, 100)
        stridesD = (1, 100)

        result = ProblemModule.ExactList.convertLeadingDims(
            problem_type, problem_size,
            stridesA=stridesA, stridesB=stridesB,
            stridesC=stridesC, stridesD=stridesD
        )

        assert isinstance(result, tuple)
        # With custom strides, LD values should reflect max(problemSize, stride-derived)
        # stridesA/B/C/D have stride 100 in dimension 1, so LD should be at least 100
        assert result[3] == max(64, 100)  # LDD
        assert result[4] == max(64, 100)  # LDC
        assert result[5] == max(64, 100)  # LDA
        assert result[6] == max(64, 100)  # LDB
