# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Consumer-visible generation refusal when nominal bias facts are unobservable."""
import pytest

from tensilelite.SolutionStructs.Problem import ProblemType


def request():
    return {"DataType": 4, "MacDataTypeA": 7, "MacDataTypeB": 7,
            "ComputeDataType": 0, "DestDataType": 0, "HighPrecisionAccumulate": True,
            "UseBias": 1, "BiasDataTypeList": [0, 7]}


def test_generation_refuses_unobservable_nominal_bias_type():
    """Feature: refuse an omitted nominal Half type not recoverable by the host.

    Adding the nominal-type generation refusal passes; removing it fails again.
    """
    with pytest.raises(RuntimeError):
        ProblemType(request(), False)


@pytest.mark.parametrize("visible", ["whitelisted", "mac"], ids=["nominal-in-list", "nominal-is-mac"])
def test_generation_preserves_host_visible_nominal_bias_type(visible):
    """Preservation: do not reject lists the existing host predicates can enforce.

    Constrain the refusal to unobservable omitted types; an unconditional
    UseBias/wider-nominal-type refusal makes these accepted requests fail again.
    """
    config = request()
    if visible == "whitelisted":
        config["BiasDataTypeList"] = [0, 4]
    else:
        config["MacDataTypeA"] = config["MacDataTypeB"] = 4
    problem = ProblemType(config, False)
    assert problem["UseBias"] == 1
    assert problem["DataType"].isHalf()
    assert [t.value for t in problem["BiasDataTypeList"]] == config["BiasDataTypeList"]
