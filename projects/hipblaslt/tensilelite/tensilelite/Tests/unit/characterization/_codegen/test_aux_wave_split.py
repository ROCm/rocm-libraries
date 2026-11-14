# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
import shutil
import subprocess

import pytest

import yaml

from config_harness import emit_kernels_from_config


def test_aux_output_with_wave_split(tmp_path):
    config_path = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["BenchmarkProblems"][0][0].update(
        UseE=True, Activation=True, ActivationType="all")
    config["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"].append(
        {"ActivationArgs": [[{"Enum": "relu"}]]})
    params = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for entry in params:
        if "WorkGroup" in entry:
            entry["WorkGroup"] = [[8, 8, 2]]
    params.append({"WaveSplitK": [True]})
    path = tmp_path / "aux-split.yaml"
    path.write_text(yaml.safe_dump(config))
    kernels = emit_kernels_from_config(path, limit=8, arch="gfx942", canonical=False)
    assert kernels
    assembler = shutil.which("amdclang++")
    if not assembler:
        pytest.skip("amdclang++ is unavailable")
    mask = re.compile(
        r"v_and_b32\s+(v\d+),\s*1,\s*v\[vgprSerial\][^\n]*\n"
        r"v_cmpx_ne_u32\s+exec,\s*1,\s*\1[^\n]*\n"
        r"v_mov_b32\s+v\d+,\s*v\d+[^\n]*LDE clip if kidx != NumWaveSplitK-1"
    )
    for index, (_, source, err) in enumerate(kernels):
        assert err == 0 and mask.search(source)
        asm = tmp_path / f"kernel-{index}.s"
        asm.write_text(source)
        subprocess.run(
            [assembler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx942", "-c", str(asm), "-o", str(asm.with_suffix(".o"))],
            check=True, capture_output=True)
