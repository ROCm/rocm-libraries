# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate clustered StreamK capabilities, argument ABI, and kernel identity."""

import pytest

from Tensile.ExecutionPolicy import normalize_execution_policy
from Tensile.CustomKernels import validateCustomPersistentArgs


pytestmark = pytest.mark.unit


def test_abi2_is_explicit_and_custom_descriptors_are_rejected():
    state = normalize_execution_policy({"TileProcessingStrategy": "StreamK",
                                        "StreamKClusterMulticast": True})
    assert state["InternalSupportParams"]["PersistentLoopArgsVersion"] == 2
    with pytest.raises(ValueError, match="requires StreamKClusterMulticast"):
        normalize_execution_policy({"TileProcessingStrategy": "StreamK",
                                    "InternalSupportParams": {"PersistentLoopArgsVersion": 2}}, regenerate=False)
    with pytest.raises(ValueError, match="Custom kernels"):
        validateCustomPersistentArgs(state)
    for strategy, version in (("DataParallel", 1), ("StreamK", 0)):
        assert normalize_execution_policy({"TileProcessingStrategy": strategy})["InternalSupportParams"]["PersistentLoopArgsVersion"] == version


@pytest.mark.parametrize("shape", [(2, 1), (4, 1), (2, 2), (2, 4)])
def test_supported_shapes_match_host_capability(shape):
    from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
    from test_streamk_multicast import _mc_state, _isa_map
    state = _mc_state(TileProcessingStrategy="StreamK", StreamKClusterMulticast=True,
                      StreamKFixupTreeReduction=1, ClusterDim=list(shape))
    assert _validateStreamKMulticast(state, False, _isa_map())


@pytest.mark.parametrize("overrides", [
    {"ClusterDim": [8, 1]}, {"ClusterDim": [4, 2]}, {"ClusterDim": [4, 4]},
    {"ClusterDim": [1, 2]}, {"PrefetchAcrossPersistent": 1}, {"PrefetchGlobalRead": 0},
    {"StreamKFixupTreeReduction": 0}, {"StreamKAtomic": 1}, {"UseSubtileImpl": 1},
    {"StoreRemapVectorWidth": 4}, {"ReuseAcrossPersistent": 1}, {"SpaceFillingAlgo": [1]},
    {"TDMInst": 2}, {"ProblemType": {"Sparse": 1}}, {"DebugStreamK": 1},
    {"ProblemType": {"OutputAmaxD": True}},
    {"ProblemType": {"Gradient": True, "UseBias": True}},
])
def test_unsupported_cluster_capabilities_fail_explicitly(overrides):
    from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
    from test_streamk_multicast import _mc_state, _isa_map
    state = _mc_state(TileProcessingStrategy="StreamK", StreamKClusterMulticast=True,
                      StreamKFixupTreeReduction=1)
    state.update(overrides)
    assert not _validateStreamKMulticast(state, False, _isa_map())
