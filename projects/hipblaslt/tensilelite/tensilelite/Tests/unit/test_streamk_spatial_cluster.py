# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate clustered StreamK capabilities, argument ABI, and kernel identity."""

import pytest

from Tensile.ExecutionPolicy import normalize_execution_policy
from Tensile.CustomKernels import validateCustomPersistentArgs


pytestmark = pytest.mark.unit


def test_abi2_is_derived_from_cluster_geometry_and_custom_descriptors_are_rejected():
    state = normalize_execution_policy({"TileProcessingStrategy": "StreamK",
                                        "ClusterDim": [2, 1]})
    assert state["InternalSupportParams"]["PersistentLoopArgsVersion"] == 2
    assert "StreamKClusterMulticast" not in state
    with pytest.raises(ValueError, match="requires StreamK/StaticGrid"):
        normalize_execution_policy({"TileProcessingStrategy": "StreamK",
                                    "InternalSupportParams": {"PersistentLoopArgsVersion": 2}}, regenerate=False)
    with pytest.raises(ValueError, match="Custom kernels"):
        validateCustomPersistentArgs(state)
    for strategy, version in (("DataParallel", 1), ("StreamK", 0)):
        assert normalize_execution_policy({"TileProcessingStrategy": strategy})["InternalSupportParams"]["PersistentLoopArgsVersion"] == version


def test_regenerated_cluster_kernel_identity_distinguishes_the_prebuilt_payload():
    from Tensile.SolutionStructs.Naming import getKernelNameMin, getKeyNoInternalArgs, getSolutionNameFull
    from test_wgmxcc_kernel_name import _minimal_kernel

    source = _minimal_kernel()
    source.update(TileProcessingStrategy="StreamK", ClusterDim=[2, 1])
    source["InternalSupportParams"] = {"KernArgsVersion": 3, "PersistentLoopArgsVersion": 0}
    prebuilt = normalize_execution_policy(source, regenerate=False)
    regenerated = normalize_execution_policy(source)
    for naming in (getKernelNameMin, getSolutionNameFull, getKeyNoInternalArgs):
        old_name = naming(prebuilt, False)
        new_name = naming(regenerated, False)
        assert "_PLAV" not in old_name
        assert "_PLAV2_" in new_name
        assert "SKCM" not in old_name and "SKCM" not in new_name
        assert new_name.replace("_PLAV2", "") == old_name


@pytest.mark.parametrize("strategy,version,fragment", [("DataParallel", 1, "_PLAV1_"), ("StreamK", 0, "")])
def test_existing_unclustered_kernel_identity_retains_its_payload_marker(strategy, version, fragment):
    from Tensile.SolutionStructs.Naming import getKernelNameMin
    from test_wgmxcc_kernel_name import _minimal_kernel

    source = _minimal_kernel()
    source.update(TileProcessingStrategy=strategy, ClusterDim=[1, 1])
    source["InternalSupportParams"] = {"KernArgsVersion": 3, "PersistentLoopArgsVersion": version}
    before = getKernelNameMin(source, False)
    after = getKernelNameMin(normalize_execution_policy(source), False)
    assert before == after
    assert fragment in after if fragment else "_PLAV" not in after


@pytest.mark.parametrize("shape", [(2, 1), (4, 1), (2, 2), (2, 4)])
def test_supported_shapes_match_host_capability(shape):
    from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
    from test_streamk_multicast import _mc_state, _isa_map
    state = _mc_state(TileProcessingStrategy="StreamK",
                      StreamKFixupTreeReduction=1, ClusterDim=list(shape))
    assert _validateStreamKMulticast(state, False, _isa_map())


@pytest.mark.parametrize("overrides", [
    {"ClusterDim": [8, 1]}, {"ClusterDim": [4, 2]}, {"ClusterDim": [4, 4]},
    {"PrefetchAcrossPersistent": 1}, {"PrefetchGlobalRead": 0},
    {"StreamKFixupTreeReduction": 0}, {"StreamKAtomic": 1}, {"UseSubtileImpl": 1},
    {"StoreRemapVectorWidth": 4}, {"ReuseAcrossPersistent": 1}, {"SpaceFillingAlgo": [1]},
    {"TDMInst": 2}, {"ProblemType": {"Sparse": 1}}, {"DebugStreamK": 1},
    {"ProblemType": {"OutputAmaxD": True}},
    {"ProblemType": {"Gradient": True, "UseBias": True}},
])
def test_unsupported_cluster_capabilities_fail_explicitly(overrides):
    from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
    from test_streamk_multicast import _mc_state, _isa_map
    state = _mc_state(TileProcessingStrategy="StreamK",
                      StreamKFixupTreeReduction=1)
    state.update(overrides)
    assert not _validateStreamKMulticast(state, False, _isa_map())


def test_n_only_streamk_cluster_is_rejected_during_solution_derivation(tmp_path):
    from pathlib import Path
    import yaml
    from config_harness import derive_states

    fixture = (Path(__file__).parent / "characterization/_codegen/data/test_data/_designed/gfx1250"
               / "streamk_split_cluster_multicast.yaml")
    config = yaml.safe_load(fixture.read_text())
    for fork in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "ClusterDim" in fork:
            fork["ClusterDim"] = [[1, 2]]
    path = tmp_path / "n_only_cluster.yaml"
    path.write_text(yaml.safe_dump(config))
    assert derive_states(str(path), arch="gfx1250", limit_solutions=1) == []
