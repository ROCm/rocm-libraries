# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compute inputs advertised for rocRoller custom-kernel ProblemTypes.

Mixed MX kernels embed DataTypeA and DataTypeB and leave MacDataType unset.
TypesEqual must report those operand types as the compute inputs. A same-type
FP6 kernel has no per-operand types and must keep DataType for both inputs.
"""

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.Contractions import ProblemType as ContractionProblemType
from tensilelite.CustomKernels import readCustomKernelConfig
from tensilelite.SolutionStructs.Problem import ProblemType as SolutionProblemType

pytestmark = pytest.mark.unit

_MIXED_FP8_BF8 = (
    "RR_GEMM_TN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"
)
_SAME_FP6 = (
    "RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2"
)


def _contraction_from_embedded(kernel_name):
    embedded = readCustomKernelConfig(kernel_name)["ProblemType"]
    derived = SolutionProblemType(embedded, printIndexAssignmentInfo=False)
    return embedded, ContractionProblemType.FromOriginalState(derived.state)


def _types_equal(problem):
    predicates = [pred for pred in problem.predicates(includeType=True) if pred.tag == "TypesEqual"]
    assert len(predicates) == 1
    return predicates[0].value


def test_mixed_custom_kernel_compute_inputs_follow_operand_types():
    embedded, problem = _contraction_from_embedded(_MIXED_FP8_BF8)
    operand_a = DataType(embedded["DataTypeA"])
    operand_b = DataType(embedded["DataTypeB"])

    assert operand_a != operand_b
    assert problem.computeInputTypeA == operand_a
    assert problem.computeInputTypeB == operand_b
    assert _types_equal(problem)[4] == operand_a
    assert _types_equal(problem)[5] == operand_b


def test_same_type_fp6_custom_kernel_compute_inputs_unchanged():
    embedded, problem = _contraction_from_embedded(_SAME_FP6)
    data_type = DataType(embedded["DataType"])

    assert "DataTypeA" not in embedded
    assert "DataTypeB" not in embedded
    assert problem.computeInputTypeA == data_type
    assert problem.computeInputTypeB == data_type
    assert _types_equal(problem)[4] == data_type
    assert _types_equal(problem)[5] == data_type


def test_memory_compute_conversion_keeps_mac_compute_inputs():
    """DataType is the MAC type; DataTypeA is only the memory format."""
    derived = SolutionProblemType(
        {
            "OperationType": "GEMM",
            "DataType": "h",
            "DataTypeA": "F8N",
            "DataTypeB": "h",
            "DestDataType": "h",
            "ComputeDataType": "s",
            "HighPrecisionAccumulate": True,
            "TransposeA": True,
            "TransposeB": False,
            "UseBeta": True,
            "Batched": True,
        },
        printIndexAssignmentInfo=False,
    )
    problem = ContractionProblemType.FromOriginalState(derived.state)
    half = DataType("h")
    assert problem.computeInputTypeA == half
    assert problem.computeInputTypeB == half
