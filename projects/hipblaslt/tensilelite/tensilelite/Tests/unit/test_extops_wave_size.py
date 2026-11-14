# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import re
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("generator", ["amax", "layer_norm", "softmax"])
@pytest.mark.parametrize("arch,wave_size", [("gfx1030", 32), ("gfx90a", 64)])
def test_generated_extop_wave_size(tmp_path, generator, arch, wave_size):
    compiler = shutil.which("amdclang++")
    if not compiler:
        pytest.skip("amdclang++ is unavailable")
    output = tmp_path / f"{generator}.s"
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-m", f"tensilelite._extops.{generator}_generator",
                    "--arch", arch, "--toolchain", compiler, "--output", str(output)],
                   env=env, check=True)
    source = output.read_text()
    assert re.search(rf"\.wavefront_size:\s+{wave_size}\b", source)
    subprocess.run([compiler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
                    f"-mcpu={arch}", "-c", str(output), "-o", str(output.with_suffix(".o"))],
                   check=True, capture_output=True)
