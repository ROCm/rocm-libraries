# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Unit tests for WorkGroupMappingXCC (WGMXCC) kernel naming.

WorkGroupMappingXCC must be included in the kernel name because it generates different
kernel code for -1 (chunking) and other values (regular XCC mapping). This test checks
that kernel names include WGMXCC so kernels are not rejected as duplicates.
"""

from copy import deepcopy

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Naming import getKernelNameMin, getKeyNoInternalArgs

pytestmark = pytest.mark.unit


def _minimal_kernel(*, work_group_mapping_xcc=-1):
    """Build the smallest kernel dict that exercises Naming._getName."""
    kernel = deepcopy(defaultSolution)
    kernel["ProblemType"] = {
        "OperationIdentifier": "Cijk_Ailk_Bljk",
        "DataType": 0,
        "DestDataType": 0,
        "ComputeDataType": 0,
        "GroupedGemm": False,
        "UseBeta": False,
        "UseBias": 0,
    }
    kernel["WorkGroupMappingXCC"] = work_group_mapping_xcc
    kernel["MacroTile0"] = 64
    kernel["MacroTile1"] = 32
    kernel["DepthU"] = 256
    kernel["MatrixInstM"] = 16
    kernel["MatrixInstN"] = 16
    kernel["MatrixInstB"] = 1
    kernel["MatrixInstruction"] = [16, 16, 1, 1]
    kernel["MIWaveTile"] = [2, 2]
    kernel["WorkGroup"] = [32, 4, 2]
    kernel["ISA"] = (9, 5, 0)
    return kernel


@pytest.mark.parametrize("work_group_mapping_xcc", [-1, 0, 2, 8])
def test_kernel_name_includes_wgmxcc(work_group_mapping_xcc):
  """Kernel names must include a WGMXCC tag for dedup and codegen lookup."""
  name = getKernelNameMin(_minimal_kernel(work_group_mapping_xcc=work_group_mapping_xcc), False)

  assert "WGMXCC" in name


def test_auto_wgmxcc_encodes_as_n1():
  """Runtime auto WGMXCC (-1) is abbreviated WGMXCCn1 in the kernel name."""
  name = getKernelNameMin(_minimal_kernel(work_group_mapping_xcc=-1), False)

  assert "WGMXCCn1" in name


def test_fixed_wgmxcc_encodes_as_one():
  """Any fixed WGMXCC value is normalized to WGMXCC1 in the kernel name."""
  name = getKernelNameMin(_minimal_kernel(work_group_mapping_xcc=8), False)

  assert "WGMXCC1" in name
  assert "WGMXCCn1" not in name


def test_auto_and_fixed_wgmxcc_produce_distinct_names():
  """Auto and fixed WGMXCC must not collapse to the same kernel name."""
  auto_name = getKernelNameMin(_minimal_kernel(work_group_mapping_xcc=-1), False)
  fixed_name = getKernelNameMin(_minimal_kernel(work_group_mapping_xcc=8), False)

  assert auto_name != fixed_name


def test_auto_and_fixed_wgmxcc_produce_distinct_dedup_keys():
  """getKeyNoInternalArgs must also distinguish auto from fixed WGMXCC."""
  auto_key = getKeyNoInternalArgs(_minimal_kernel(work_group_mapping_xcc=-1), False)
  fixed_key = getKeyNoInternalArgs(_minimal_kernel(work_group_mapping_xcc=8), False)

  assert "WGMXCC" in auto_key
  assert "WGMXCC" in fixed_key
  assert auto_key != fixed_key
