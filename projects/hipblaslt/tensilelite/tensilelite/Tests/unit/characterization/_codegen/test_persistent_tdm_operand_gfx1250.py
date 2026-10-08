# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compile persistent TDM paths that use constants instead of optional SGPRs."""

from pathlib import Path

import pytest
import yaml

from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config
from Tensile import __file__ as tensile_file

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("feature", ["store_remap", "single_wave"])
def test_persistent_tdm_constant_operands_assemble(tmp_path, feature):
    source = "Tests/common/streamk/gfx1250/data_parallel_cluster_multicast_boundaries.yaml"
    config = yaml.safe_load((Path(tensile_file).parent / source).read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    group = config["BenchmarkProblems"][0][1]
    values = {key: value[:1] for entry in group["ForkParameters"]
              for key, value in entry.items()}
    values.update(TileProcessingStrategy=["DataParallel"], WorkAssignment=["StaticGrid"],
                  ClusterDim=[[2, 2]], PrefetchAcrossPersistent=[0], StaggerU=[0],
                  GlobalSplitU=[0], ForceDisableShadowInit=[True])
    if feature == "store_remap":
        # Whole-tile stores have no runtime GSULog2BpeD register.
        values["StoreRemapVectorWidth"] = [4]
    else:
        # A single wave uses A/B increments directly; paired-wave descriptors
        # carry their own increments and would not catch this regression.
        values["MatrixInstruction"] = [[16, 16, 32, 1, 1, 1, 1, 1, 1]]
    # Unstaggered TDM increments can be constants. Assembly must not refer
    # to an SGPR that was removed when the increment was folded.
    group["ForkParameters"] = [{key: value} for key, value in values.items()]
    path = tmp_path / (feature + ".yaml")
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    assert_real_gfx1250_kernels(results)
    for base, assembly, _ in results:
        assert_assembles(assembly, base)
