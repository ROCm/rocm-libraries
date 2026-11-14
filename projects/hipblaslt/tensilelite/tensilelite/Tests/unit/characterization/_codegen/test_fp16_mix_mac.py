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
def test_fp16_mix_codegen(tmp_path, transpose_a, transpose_b):
    source = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    config = yaml.safe_load(source.read_text())
    config["BenchmarkProblems"][0][0].update(
        TransposeA=transpose_a, TransposeB=transpose_b)
    for entry in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "LocalReadVectorWidth" in entry:
            entry["LocalReadVectorWidth"] = [1]
    path = tmp_path / "fp16-mix.yaml"
    path.write_text(yaml.safe_dump(config))
    kernels = emit_kernels_from_config(path, limit=8, arch="gfx90a", canonical=False)
    assert kernels
    assembler = shutil.which("amdclang++")
    if not assembler:
        pytest.skip("amdclang++ is unavailable")
    for index, (_, assembly, err) in enumerate(kernels):
        assert err == 0 and ("v_fma_mix_f32" in assembly or "v_mad_mix_f32" in assembly)
        asm = tmp_path / f"kernel-{index}.s"
        asm.write_text(assembly)
        subprocess.run(
            [assembler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx90a", "-c", str(asm), "-o", str(asm.with_suffix(".o"))],
            check=True, capture_output=True)
