# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Unit tests for the WorkGroupPhaseStagger parameters (the workgroups whose WorkGroup0 has bit
WorkGroupPhaseStaggerBit set sleep WorkGroupPhaseStagger x 64 cycles before their first
instruction): 0, the default, must emit nothing and leave every kernel name as it was.
"""

from copy import deepcopy

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Naming import getKernelNameMin

pytestmark = pytest.mark.unit

try:
    from rocisa import rocIsa
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly as _KWA
    _KWA_AVAILABLE = True
except Exception:
    _KWA_AVAILABLE = False


def _minimal_kernel(**overrides):
    """The smallest kernel dict that exercises Naming._getName."""
    kernel = deepcopy(defaultSolution)
    kernel["ProblemType"] = {
        "OperationIdentifier": "Cijk_Alik_Bljk",
        "DataType": 0,
        "DestDataType": 0,
        "ComputeDataType": 0,
        "GroupedGemm": False,
        "UseBeta": False,
        "UseBias": 0,
    }
    kernel["MacroTile0"] = 128
    kernel["MacroTile1"] = 96
    kernel["DepthU"] = 128
    kernel["MatrixInstM"] = 16
    kernel["MatrixInstN"] = 16
    kernel["MatrixInstB"] = 1
    kernel["MatrixInstruction"] = [16, 16, 1, 1]
    kernel["MIWaveTile"] = [2, 3]
    kernel["WorkGroup"] = [64, 4, 1]
    kernel["ISA"] = (11, 5, 0)
    kernel.update(overrides)
    return kernel


def test_default_does_not_rename_kernels():
    """No delay: neither parameter enters the name, whatever the bit says."""
    base = getKernelNameMin(_minimal_kernel(), False)
    assert defaultSolution["WorkGroupPhaseStagger"] == 0
    assert getKernelNameMin(_minimal_kernel(WorkGroupPhaseStaggerBit=5), False) == base


def test_delay_and_bit_name_the_kernel():
    """A delay changes the code, so the kernel is named apart, and so is another bit."""
    base = getKernelNameMin(_minimal_kernel(), False)
    staggered = getKernelNameMin(_minimal_kernel(WorkGroupPhaseStagger=80), False)
    other_bit = getKernelNameMin(_minimal_kernel(WorkGroupPhaseStagger=80, WorkGroupPhaseStaggerBit=2), False)
    assert len({base, staggered, other_bit}) == 3


@pytest.fixture
def _gfx1150_isa():
    if not _KWA_AVAILABLE:
        pytest.skip("rocisa / KernelWriterAssembly not importable")
    rocIsa.getInstance().setKernel((11, 5, 0), 32)


def _emit(units, bit=3):
    return str(_KWA.workGroupPhaseStagger(None, {"WorkGroupPhaseStagger": units, "WorkGroupPhaseStaggerBit": bit}))


def test_no_delay_emits_nothing(_gfx1150_isa):
    assert _emit(0).strip() == ""


def test_delay_tests_the_bit_and_sleeps_in_s_sleep_steps(_gfx1150_isa):
    """200 x 64 cycles: s_sleep takes at most 127, so 127 + 73, skipped when the bit is clear."""
    lines = [l.split("//")[0].strip() for l in _emit(200, bit=3).splitlines() if l.strip()]
    assert lines == [
        "s_bitcmp1_b32 s[sgprWorkGroup0], 3",
        "s_cbranch_scc0 label_WorkGroupPhaseStaggerEnd",
        "s_sleep 127",
        "s_sleep 73",
        "label_WorkGroupPhaseStaggerEnd:",
    ]
