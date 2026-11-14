# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import shutil
import subprocess

import pytest

from rocisa.container import vgpr
from rocisa.instruction import VDot4I32I8


@pytest.mark.parametrize(
    ("arch", "instruction"),
    [
        ("gfx1030", VDot4I32I8(vgpr(0), vgpr(1), vgpr(2), vgpr(0))),
    ],
)
def test_valu_instruction_assembles(tmp_path, arch, instruction):
    assembler = shutil.which("amdclang++")
    if not assembler:
        pytest.skip("amdclang++ is unavailable")
    source = tmp_path / "instruction.s"
    source.write_text(".text\n" + str(instruction) + "\n")
    subprocess.run(
        [assembler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         f"-mcpu={arch}", "-c", str(source), "-o", str(tmp_path / "instruction.o")],
        check=True,
        capture_output=True,
    )
