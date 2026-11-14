# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import yaml

from config_harness import emit_kernels_from_config


def test_packed_int8_mac_codegen(tmp_path):
    config_path = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    config = yaml.safe_load(config_path.read_text())
    problem = config["BenchmarkProblems"][0][0]
    problem.update(DataType="4xi8", DestDataType="i", ComputeDataType="i",
                   HighPrecisionAccumulate=False)
    params = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for entry in params:
        if "LocalReadVectorWidth" in entry:
            entry["LocalReadVectorWidth"] = [1]
    path = tmp_path / "packed-int8.yaml"
    path.write_text(yaml.safe_dump(config))
    kernels = emit_kernels_from_config(path, limit=8, arch="gfx1030")
    assert kernels
    assert all(err == 0 and "v_dot4_i32_i8" in src for _, src, err in kernels)
