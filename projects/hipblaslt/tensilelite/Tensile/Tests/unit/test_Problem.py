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
        # At least one type (ComputeDataType or DestDataType) should be included
        assert len(result) >= 1


@pytest.mark.unit
class TestProblemTypeValidGEMMTypes:
    """Tests for valid GEMM type lists"""

    def test_valid_gemm_types_exists(self):
        """Test that _validGEMMTypes exists and is populated"""
        assert hasattr(ProblemModule, '_validGEMMTypes')
        assert isinstance(ProblemModule._validGEMMTypes, list)
        assert len(ProblemModule._validGEMMTypes) > 0

    def test_valid_gemm_types_format(self):
        """Test _validGEMMTypes format"""
        # Each entry should be a 4-tuple (Ti, To, Tc, Tc)
        for entry in ProblemModule._validGEMMTypes:
            assert isinstance(entry, tuple)
            assert len(entry) == 4

    def test_hpa_types_exists(self):
        """Test that _HPATypes exists"""
        assert hasattr(ProblemModule, '_HPATypes')
        assert isinstance(ProblemModule._HPATypes, list)
        assert len(ProblemModule._HPATypes) > 0

    def test_hpa_types_format(self):
        """Test _HPATypes format"""
        for entry in ProblemModule._HPATypes:
            assert isinstance(entry, tuple)
            assert len(entry) == 4


@pytest.mark.unit
class TestDefaultProblemType:
    """Tests for _defaultProblemType dictionary"""

    def test_default_problem_type_exists(self):
        """Test that _defaultProblemType exists"""
        assert hasattr(ProblemModule, '_defaultProblemType')
        assert isinstance(ProblemModule._defaultProblemType, dict)

    def test_default_problem_type_has_required_keys(self):
        """Test _defaultProblemType has required keys"""
        required_keys = [
            "OperationType", "DataType", "UseBeta",
            "TransposeA", "TransposeB", "Batched",
            "IndexAssignmentsA", "IndexAssignmentsB",
            "NumIndicesC"
        ]

        for key in required_keys:
            assert key in ProblemModule._defaultProblemType

    def test_default_problem_type_operation_type(self):
        """Test default OperationType"""
        assert ProblemModule._defaultProblemType["OperationType"] == "GEMM"

    def test_default_problem_type_transpose(self):
        """Test default transpose settings"""
        assert ProblemModule._defaultProblemType["TransposeA"] == False
        assert ProblemModule._defaultProblemType["TransposeB"] == True

    def test_default_problem_type_index_assignments(self):
        """Test default index assignments"""
        assert ProblemModule._defaultProblemType["IndexAssignmentsA"] == [0, 2]
        assert ProblemModule._defaultProblemType["IndexAssignmentsB"] == [1, 2]

    def test_default_problem_type_num_indices(self):
        """Test default number of indices"""
        assert ProblemModule._defaultProblemType["NumIndicesC"] == 2
        assert ProblemModule._defaultProblemType["NumIndicesLD"] == 4


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
        assert len(result) > problem_type["NumIndicesC"]

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
