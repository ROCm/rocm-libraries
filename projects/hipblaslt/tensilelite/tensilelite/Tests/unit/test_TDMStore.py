################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Unit tests for TDMStore: solution validation and the kernel-name token."""

from copy import deepcopy

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Naming import getKernelNameMin
from tensilelite.SolutionStructs.Validators.TDMStore import validateTDMStore

pytestmark = pytest.mark.unit


def _state(**overrides):
  """A TDMStore solution inside the supported configurations, with `overrides` applied."""
  problemType = {
    "DestDataType": DataType("b"),
    "ComputeDataType": DataType("s"),
    "HighPrecisionAccumulate": True,
    "FusedGemmA2A": False,
    "UseE": False,
    "Gradient": False,
    "UseGateResidual": False,
    "GroupedGemm": False,
    "UseBias": 0,
    "UseScaleAlphaVec": 0,
    "UseScaleAB": "",
    "UseScaleCD": False,
    "ActivationType": "none",
    "OutputAmaxD": False,
  }
  state = {
    "TDMStore": 1,
    "ISA": (12, 5, 0),
    "BufferStore": True,
    "GlobalSplitU": 1,
    "TileProcessingStrategy": "None",
    "WorkAssignment": "StaticGrid",
    "PrefetchAcrossPersistent": 0,
    "ReuseAcrossPersistent": 0,
    "UseSubtileImpl": False,
    "StoreRemapVectorWidth": 0,
    "CompactLoopStore": False,
    "MacroTile1": 256,
    "NumThreads": 128,
    "WavefrontSize": 32,
    "ProblemType": problemType,
  }
  for key, value in overrides.items():
    (problemType if key in problemType else state)[key] = value
  return state


@pytest.mark.parametrize("dest", ["b", "h"])
def test_accepts_bf16_and_fp16(dest):
  assert validateTDMStore(_state(DestDataType=DataType(dest)), False)


def test_accepts_non_atomic_streamk3():
  assert validateTDMStore(_state(TileProcessingStrategy="StreamK", GlobalSplitU=0), False)


@pytest.mark.parametrize("overrides", [
  {"UseBias": 1},
  {"UseScaleAlphaVec": 1},
  {"UseScaleAB": "Scalar"},
  {"UseScaleAB": "Vector"},
  {"ActivationType": "relu"},
  {"ActivationType": "hipblaslt_all"},
  {"UseBias": 1, "UseScaleAlphaVec": 1, "ActivationType": "hipblaslt_all",
   "TileProcessingStrategy": "StreamK", "GlobalSplitU": 0},
])
def test_accepts_epilogue_vectors_and_activation(overrides):
  assert validateTDMStore(_state(**overrides), False)


def test_disabled_is_not_validated():
  assert validateTDMStore(_state(TDMStore=0, ISA=(9, 5, 0), GlobalSplitU=4), False)


@pytest.mark.parametrize("overrides", [
  {"ISA": (9, 5, 0)},
  {"DestDataType": DataType("s")},
  {"HighPrecisionAccumulate": False},
  {"BufferStore": False},
  {"GlobalSplitU": 2},
  {"GlobalSplitU": -1},
  {"TileProcessingStrategy": "DataParallel", "GlobalSplitU": 0},
  {"TileProcessingStrategy": "StreamK", "GlobalSplitU": 0, "WorkAssignment": "DynamicWorkQueue"},
  {"TileProcessingStrategy": "StreamK", "GlobalSplitU": 0, "StreamKAtomic": 1},
  {"TileProcessingStrategy": "StreamK", "GlobalSplitU": 0, "PrefetchAcrossPersistent": 1},
  {"TileProcessingStrategy": "StreamK", "GlobalSplitU": 0, "ReuseAcrossPersistent": 1},
  {"TileProcessingStrategy": "StreamK", "GlobalSplitU": 0, "ClusterDim": [2, 1]},
  {"UseSubtileImpl": True},
  {"StoreRemapVectorWidth": 4},
  {"CompactLoopStore": True},
  {"FusedGemmA2A": True},
  {"UseE": True},
  {"Gradient": True},
  {"UseGateResidual": True},
  {"UseScaleCD": True},
  {"OutputAmaxD": True},
  {"GroupedGemm": True},
  {"UseBias": 2},
  {"UseBias": 3},
  {"UseScaleAlphaVec": 2},
  {"NumThreads": 96},
])
def test_rejects_unsupported(overrides):
  state = _state(**overrides)
  assert not validateTDMStore(state, False)
  assert state["Valid"] is False


def _kernel(tdmStore):
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
  kernel["MacroTile0"] = 256
  kernel["MacroTile1"] = 256
  kernel["DepthU"] = 64
  kernel["MatrixInstM"] = 16
  kernel["MatrixInstN"] = 16
  kernel["MatrixInstB"] = 1
  kernel["MatrixInstruction"] = [16, 16, 32, 1]
  kernel["MIWaveTile"] = [8, 8]
  kernel["WorkGroup"] = [32, 4, 1]
  kernel["ISA"] = (12, 5, 0)
  kernel["TDMStore"] = tdmStore
  return kernel


def test_kernel_name_token_only_when_enabled():
  """TDMStore=0 keeps every existing kernel name; TDMStore=1 adds a TDMS1 token."""
  off = getKernelNameMin(_kernel(0), False)
  on = getKernelNameMin(_kernel(1), False)
  assert "TDMS1" not in off
  assert on.replace("_TDMS1", "", 1) == off
