# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Unit tests for the PostLoopStoreInNll term in the kernel name.

The key is named only where the fused store can exist, so that adding PLSIN to
the library does not rename every kernel in it. Two things have to hold at once:
an underived solution must not carry its default into the name, and a derived
one must get the same name wherever that name is taken. The second is what
broke: the gate's datatype predicates answer "not in scope" on a state whose
ProblemType still holds raw values, so reading them in Naming gave one MXFP8
solution a kernel named PLSIN1 and a benchmark column without it.
"""

from copy import deepcopy

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Naming import getKernelNameMin

pytestmark = pytest.mark.unit

FP8 = DataType("float8")
FP4 = DataType("float4")
BF16 = DataType("bfloat16")


def _kernel(*, dtA=FP8, dtB=FP8, dtD=BF16, plsin=True, derived=True, raw=False):
    """The smallest nameable MX subtile kernel, in one of two representations.

    ``raw=True`` replaces the operand types with the integer values a solution
    dict holds before they are built into DataType objects, which is how the
    same solution reaches Naming at different points in a run.
    """
    kernel = deepcopy(defaultSolution)
    kernel["ProblemType"] = {
        "OperationIdentifier": "Cijk_Alik_Bljk",
        "DataType": dtA.value if raw else dtA,
        "DataTypeA": dtA.value if raw else dtA,
        "DataTypeB": dtB.value if raw else dtB,
        "DestDataType": dtD.value if raw else dtD,
        "ComputeDataType": 0,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TransposeA": True,
        "TransposeB": False,
        "GroupedGemm": False,
        "UseBeta": False,
        "UseBias": 0,
    }
    kernel["AssignedDerivedParameters"] = derived
    kernel["PostLoopStoreInNll"] = plsin
    kernel["UseSubtileImpl"] = True
    kernel["MacroTile0"] = 256
    kernel["MacroTile1"] = 64
    kernel["DepthU"] = 128
    kernel["MatrixInstM"] = 16
    kernel["MatrixInstN"] = 16
    kernel["MatrixInstB"] = 1
    kernel["MatrixInstruction"] = [16, 16, 128, 1]
    kernel["MIWaveTile"] = [8, 2]
    kernel["WorkGroup"] = [32, 8, 1]
    kernel["NumThreads"] = 256
    kernel["ISA"] = (9, 5, 0)
    return kernel


def test_name_does_not_depend_on_how_the_problem_type_is_held():
    """The regression: one solution, two names, and a benchmark that failed.

    A run names the same solution more than once and the ProblemType is not in
    the same shape each time. Any term in the naming decision that reads the
    operand types therefore has to answer the same in both shapes -- and the
    datatype predicates deliberately do not, because "not in scope" is the
    conservative answer for a scope gate. The decision is keyed on
    AssignedDerivedParameters instead, which survives the change of shape.
    """
    assert getKernelNameMin(_kernel(raw=False), False) \
        == getKernelNameMin(_kernel(raw=True), False)


@pytest.mark.parametrize("raw", [False, True], ids=["DataType objects", "raw values"])
def test_derived_fp8_records_the_fused_store(raw):
    assert "PLSIN1" in getKernelNameMin(_kernel(raw=raw), False)


@pytest.mark.parametrize("raw", [False, True], ids=["DataType objects", "raw values"])
def test_underived_default_stays_out_of_the_name(raw):
    """``Solution.__init__`` pre-fills the key True from defaultSolution.

    A solution rejected ahead of ``assignPostLoopStoreInNll`` keeps that True
    without having earned it, and naming it PLSIN1 would hash it alongside
    kernels that really do carry the store.
    """
    name = getKernelNameMin(_kernel(derived=False, raw=raw), False)
    assert "PLSIN" not in name


@pytest.mark.parametrize("raw", [False, True], ids=["DataType objects", "raw values"])
def test_fp8_without_the_store_is_named_as_before(raw):
    """The fp8 kernels that do not take the store must not be renamed.

    This is what keeps the change to the library a diff of 52 kernels rather
    than of every fp8 kernel in it.
    """
    name = getKernelNameMin(_kernel(plsin=False, raw=raw), False)
    assert "PLSIN" not in name


def test_fp4_naming_is_untouched():
    """fp4 reaches the name through isMxf4SubtilePath, which is unchanged.

    Both values of the key stay in the name there, which is develop's behaviour
    and what makes the fp4 half of the library byte-identical across this
    change.
    """
    for plsin, token in ((True, "PLSIN1"), (False, "PLSIN0")):
        name = getKernelNameMin(_kernel(dtA=FP4, dtB=FP4, plsin=plsin), False)
        assert token in name
