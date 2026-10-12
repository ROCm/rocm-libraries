# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""FP32 GSU atomics must cover all CUs updating the shared workspace."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("arch", ["gfx1250", "gfx942"])
def test_singlebuffer_fp32_atomic_scope(tmp_path, arch):
    if arch == "gfx1250":
        source_config = (Path(__file__).resolve().parents[3]
                         / "common/gemm/gfx12/gsuasb_fp32_atomic_gfx1250.yaml")
        expected_count = 2  # SIA0 and SIA4.
    else:
        source_config = Path(__file__).parent / "data/test_data/_designed/gfx942/gsu.yaml"
        expected_count = 1
    config = yaml.safe_load(source_config.read_text())
    _, benchmark = config["BenchmarkProblems"][0]
    forks = {key: values if key == "ScheduleIterAlg" else values[:1]
             for fork in benchmark["ForkParameters"] for key, values in fork.items()}
    forks.update(GlobalSplitU=[2], GlobalSplitUAlgorithm=["SingleBuffer"],
                 StoreRemapVectorWidth=[0])
    benchmark["ForkParameters"] = [{key: values} for key, values in forks.items()]
    config_path = tmp_path / "atomic_scope.yaml"
    config_path.write_text(yaml.safe_dump(config))

    results = emit_kernels_from_config(config_path, arch=arch,
                                      expected_fork_count=expected_count)
    assert len(results) == expected_count
    for name, source, error in results:
        assert error == 0, name
        atomics = re.findall(r"^buffer_atomic_add_f32\b.*$", source, re.MULTILINE)
        assert atomics, name
        if arch == "gfx1250":
            assert all("scope:SCOPE_DEV" in instruction for instruction in atomics)
        else:
            # Legacy targets do not have gfx1250's CU-local default scope.
            assert all("scope:" not in instruction for instruction in atomics)
