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

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.RequiredParameters import getRequiredParametersMin
from Tensile.Common.ValidParameters import validParameters
from Tensile.Contractions import SizeMapping
from Tensile.SolutionStructs.Solution import _validateStreamKForceDPOnly, validateParameterTypes


def minimal_size_mapping_state():
    return {
        "_GlobalAccumulation": "PartialsBuffer",
        "_WorkspaceSizePerElemBias": 0,
        "_WorkspaceSizePerElemC": 4,
        "_staggerStrideShift": 0,
        "ActivationFused": True,
        "AdaptiveGemmGSUA": 0,
        "ClusterDim": [1, 1],
        "CUOccupancy": 1,
        "CustomKernelName": "",
        "DepthU": 64,
        "DirectToLdsA": False,
        "DirectToLdsB": False,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "EnableMatrixInstruction": True,
        "ExpertSchedulingMode": 0,
        "GlobalReadVectorWidthA": 1,
        "GlobalReadVectorWidthB": 1,
        "GlobalSplitU": 0,
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "KernelLanguage": "Assembly",
        "LocalSplitU": 1,
        "MIWaveGroup": [2, 2],
        "MIWaveTile": [2, 2],
        "MacroTile0": 128,
        "MacroTile1": 128,
        "MagicDivAlg": 1,
        "MathClocksUnrolledLoop": 0,
        "MatrixInstruction": [16, 16, 32, 1],
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalD": 0,
        "NumElementsPerBatchStore": 0,
        "NumElementsPerThread": 1,
        "NumLoadsCoalescedA": 1,
        "NumLoadsCoalescedB": 1,
        "NumThreads": 256,
        "PrefetchGlobalRead": 1,
        "SpaceFillingAlgo": [],
        "StaggerU": 0,
        "StaggerUMapping": 0,
        "StoreVectorWidth": 1,
        "StreamK": 3,
        "StreamKAtomic": 0,
        "ThreadTile": [1, 1],
        "ThreadTile0": 1,
        "ThreadTile1": 1,
        "UnrollLoopSwapGlobalReadOrder": 0,
        "UseCustomMainLoopSchedule": 0,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "WaveSeparateGlobalReadA": False,
        "WaveSeparateGlobalReadB": False,
        "WavefrontSize": 64,
        "WorkGroup": [16, 16, 1],
        "WorkGroupMapping": 1,
        "WorkGroupMappingXCC": 0,
        "WorkGroupMappingXCCGroup": 0,
    }


def test_streamk_force_dp_only_is_valid_tuning_parameter():
    assert validParameters["StreamKForceDPOnly"] == [0, 1]
    assert defaultSolution["StreamKForceDPOnly"] == 0
    assert "StreamKForceDPOnly" in getRequiredParametersMin()


def test_streamk_force_dp_only_type_is_checked_as_int():
    state = {"StreamKForceDPOnly": 1}
    validateParameterTypes(state)


def test_streamk_force_dp_only_defaults_in_size_mapping():
    state = minimal_size_mapping_state()
    size_mapping = SizeMapping.FromOriginalState(state)

    assert "streamKForceDPOnly" in SizeMapping.StateKeys
    assert size_mapping.streamKForceDPOnly == 0


def test_streamk_force_dp_only_round_trips_to_size_mapping():
    state = minimal_size_mapping_state()
    state["StreamKForceDPOnly"] = 1

    size_mapping = SizeMapping.FromOriginalState(state)

    assert size_mapping.streamKForceDPOnly == 1


def test_streamk_force_dp_only_requires_streamk3():
    state = {
        "StreamK": 2,
        "StreamKAtomic": 0,
        "StreamKForceDPOnly": 1,
        "Valid": True,
    }

    valid = _validateStreamKForceDPOnly(state, False)

    assert valid is False
    assert state["Valid"] is False


def test_streamk_force_dp_only_rejects_atomic_streamk():
    state = {
        "StreamK": 3,
        "StreamKAtomic": 1,
        "StreamKForceDPOnly": 1,
        "Valid": True,
    }

    valid = _validateStreamKForceDPOnly(state, False)

    assert valid is False
    assert state["Valid"] is False


def test_streamk_force_dp_only_accepts_streamk3_tree_path():
    state = {
        "StreamK": 3,
        "StreamKAtomic": 0,
        "StreamKForceDPOnly": 1,
        "Valid": True,
    }

    valid = _validateStreamKForceDPOnly(state, False)

    assert valid is True
    assert state["Valid"] is True
