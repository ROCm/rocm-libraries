# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from config_harness import emit_kernels_from_config


@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (False, True),
                                                      (True, False), (True, True)])
def test_bf16_fallback_codegen(tmp_path, transpose_a, transpose_b):
    source = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    config = yaml.safe_load(source.read_text())
    config["BenchmarkProblems"][0][0].update(
        DataType="b", DestDataType="s", ComputeDataType="s",
        HighPrecisionAccumulate=True, TransposeA=transpose_a, TransposeB=transpose_b)
    for entry in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "LocalReadVectorWidth" in entry:
            entry["LocalReadVectorWidth"] = [1]
    path = tmp_path / "bf16-valu.yaml"
    path.write_text(yaml.safe_dump(config))
    kernels = emit_kernels_from_config(path, limit=8, arch="gfx1030", canonical=False)
    assert kernels
    assembler = shutil.which("amdclang++")
    if not assembler:
        pytest.skip("amdclang++ is unavailable")
    for index, (_, assembly, err) in enumerate(kernels):
        assert err == 0 and "v_fma_f32" in assembly and "v_lshlrev_b32" in assembly
        assert "ds_read_u16_d16_hi" in assembly
        asm = tmp_path / f"kernel-{index}.s"
        asm.write_text(assembly)
        subprocess.run(
            [assembler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx1030", "-c", str(asm), "-o", str(asm.with_suffix(".o"))],
            check=True, capture_output=True)
