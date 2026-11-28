# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import shutil
import subprocess

import pytest

import yaml

from config_harness import emit_kernels_from_config


def test_packed_fp16_mac_codegen(tmp_path):
    config_path = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    config = yaml.safe_load(config_path.read_text())
    problem = config["BenchmarkProblems"][0][0]
    problem.update(ComputeDataType="h", HighPrecisionAccumulate=False)
    params = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for entry in params:
        if "LocalReadVectorWidth" in entry:
            entry["LocalReadVectorWidth"] = [1]
    path = tmp_path / "packed-fp16.yaml"
    path.write_text(yaml.safe_dump(config))
    kernels = emit_kernels_from_config(path, limit=8, arch="gfx1030", canonical=False)
    assert kernels
    assembler = shutil.which("amdclang++")
    if not assembler:
        pytest.skip("amdclang++ is unavailable")
    for index, (_, source, err) in enumerate(kernels):
        assert err == 0 and "v_pk_fma_f16" in source
        asm = tmp_path / f"kernel-{index}.s"
        asm.write_text(source)
        subprocess.run(
            [assembler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx1030", "-c", str(asm), "-o", str(asm.with_suffix(".o"))],
            check=True, capture_output=True)
