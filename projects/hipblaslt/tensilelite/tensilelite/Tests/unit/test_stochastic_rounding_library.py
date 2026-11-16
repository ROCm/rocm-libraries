# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The library problem type must keep the YAML StochasticRounding value.

Kernel generation reads the logic YAML directly. Verify StochasticRounding
value is preserved and propagated to the host.
"""

import pytest

from tensilelite.Common.Utilities import state
from tensilelite.Contractions import ProblemType

pytestmark = pytest.mark.unit


def generic_gemm(**overrides) -> ProblemType:
    """A single-batch GEMM dict that FromOriginalState accepts."""
    source = {
        "OperationType": "GEMM",
        "DataType": "H",
        "DestDataType": "H",
        "ComputeDataType": "S",
        "TransposeA": False,
        "TransposeB": True,
        "Batched": True,
        "UseBeta": True,
        "TotalIndices": 4,
        "NumIndicesC": 3,
        "IndicesFree": [0, 1],
        "IndicesBatch": [2],
        "IndicesSummation": [3],
        "IndexAssignmentsA": [0, 3, 2],
        "IndexAssignmentsB": [3, 1, 2],
        "ComplexConjugateA": False,
        "ComplexConjugateB": False,
        "ActivationComputeDataType": "S",
    }
    source.update(overrides)
    problem = ProblemType.FromOriginalState(source)
    # Solution.FromOriginalState sets this after ProblemType.FromOriginalState.
    problem.mxScaleFormat = 0
    return problem


def test_library_state_has_yaml_stochastic_rounding():
    """The library state has the YAML StochasticRounding value."""
    assert "stochasticRounding" in ProblemType.StateKeys

    enabled = state(generic_gemm(StochasticRounding=True))
    disabled = state(generic_gemm(StochasticRounding=False))
    omitted = state(generic_gemm())

    assert enabled["stochasticRounding"] is True
    assert disabled["stochasticRounding"] is False
    assert omitted["stochasticRounding"] is False


def test_problem_type_defaults_stochastic_rounding_off():
    """A problem type built without the YAML key does not append RNDSeed."""
    assert generic_gemm().stochasticRounding is False
