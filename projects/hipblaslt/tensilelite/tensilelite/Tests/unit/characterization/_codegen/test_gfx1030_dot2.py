# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import shutil
import subprocess

import pytest

from config_harness import emit_kernels_from_config


def test_gfx1030_fp16_dot2_codegen(tmp_path):
    config = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    kernels = emit_kernels_from_config(config, limit=8, arch="gfx1030", canonical=False)
    assert kernels
    assembler = shutil.which("amdclang++")
    if not assembler:
        pytest.skip("amdclang++ is unavailable")
    for index, (_, source, err) in enumerate(kernels):
        assert err == 0 and "v_dot2_f32_f16" in source
        asm = tmp_path / f"kernel-{index}.s"
        asm.write_text(source)
        subprocess.run(
            [assembler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx1030", "-c", str(asm), "-o", str(asm.with_suffix(".o"))],
            check=True, capture_output=True)
