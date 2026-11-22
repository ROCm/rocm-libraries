# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A requested sigmoid+bias specialization must not become a GEMM.

Feature: reject the original contradictory request before activation normalization.
Pass: ProblemType raises an explicit error for sigmoid with any nonzero UseBias.
Break again: silently normalize sigmoid to none, or admit sigmoid with bias.
"""

import pytest

from tensilelite.SolutionStructs.Problem import ProblemType

pytestmark = pytest.mark.unit


def problem_type(activation, bias):
    return {
        "OperationType": "GEMM",
        "DataType": "s",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "TransposeA": False,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "Activation": True,
        "ActivationType": activation,
        "UseBias": bias,
        "BiasDataTypeList": ["s"] if bias else [],
    }


@pytest.mark.parametrize("bias", [1, 2, 3], ids=["M", "N", "MN"])
def test_sigmoid_with_bias_is_an_explicit_generation_error(bias):
    # Controls prove that bias alone and sigmoid without bias are parseable;
    # rejection must be of their combination, not the surrounding GEMM schema.
    assert ProblemType(problem_type("hipblaslt_all", bias), False)["UseBias"] == bias
    assert ProblemType(problem_type("sigmoid", 0), False)["UseBias"] == 0
    with pytest.raises((RuntimeError, ValueError, SystemExit)) as rejected:
        ProblemType(problem_type("sigmoid", bias), False)
    assert str(rejected.value).strip()
    if isinstance(rejected.value, SystemExit):
        assert rejected.value.code not in (None, 0)


@pytest.mark.parametrize("bias", [1, 2, 3], ids=["M", "N", "MN"])
@pytest.mark.parametrize("activation", ["sigmoid", "Sigmoid", "SIGMOID"],
                         ids=["lower", "title", "upper"])
def test_sigmoid_without_activation_key_with_bias_is_an_explicit_generation_error(bias, activation):
    # An explicit ActivationType request must be checked before omitted/default
    # Activation is normalized away; neither surrounding schema control is refused.
    bias_only = problem_type("hipblaslt_all", bias)
    no_bias = problem_type(activation, 0)
    request = problem_type(activation, bias)
    for config in (bias_only, request):
        del config["Activation"]
    assert ProblemType(bias_only, False)["UseBias"] == bias
    assert ProblemType(no_bias, False)["UseBias"] == 0
    with pytest.raises((RuntimeError, ValueError, SystemExit)) as rejected:
        ProblemType(request, False)
    assert str(rejected.value).strip()
    if isinstance(rejected.value, SystemExit):
        assert rejected.value.code not in (None, 0)


@pytest.mark.parametrize("bias", [1, 2, 3], ids=["M", "N", "MN"])
def test_sigmoid_with_bias_is_refused_when_activation_is_false(bias):
    # Explicit false and omitted Activation cannot disagree about refusal of
    # the same contradictory request; this does not change activation defaults.
    bias_only = problem_type("hipblaslt_all", bias)
    no_bias = problem_type("sigmoid", 0)
    request = problem_type("sigmoid", bias)
    for config in (bias_only, request):
        config["Activation"] = False
    assert ProblemType(bias_only, False)["UseBias"] == bias
    assert ProblemType(no_bias, False)["UseBias"] == 0
    with pytest.raises((RuntimeError, ValueError, SystemExit)) as rejected:
        ProblemType(request, False)
    assert str(rejected.value).strip()
    if isinstance(rejected.value, SystemExit):
        assert rejected.value.code not in (None, 0)

