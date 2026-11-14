# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import shutil
import subprocess

import pytest
import rocisa


def test_dot4_capability_matches_gfx1030_assembler(tmp_path):
    assembler = shutil.which("amdclang++")
    if not assembler:
        pytest.skip("amdclang++ is unavailable")
    source = tmp_path / "dot4.s"
    source.write_text(".text\nv_dot4_i32_i8 v47, v36, v34, v47\n")
    result = subprocess.run(
        [assembler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx1030", "-c", str(source), "-o", str(tmp_path / "dot4.o")],
        capture_output=True,
    )
    if result.returncode:
        pytest.skip("assembler does not support gfx1030 dot4")
    ri = rocisa.rocIsa.getInstance()
    ri.init((10, 3, 0), assembler)
    ri.setKernel((10, 3, 0), 32)
    assert ri.getAsmCaps()["v_dot4_i32_i8"]
