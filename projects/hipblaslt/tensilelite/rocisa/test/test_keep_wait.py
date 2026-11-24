# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A composite wait passes keepWait to the instructions it expands into, since rocIsaPass replaces
the composite by them before StinkyTofu reads the flag."""

import os
import shutil

import pytest
import rocisa
from rocisa.instruction import SWaitCnt

_ISA = (12, 5, 0)


@pytest.fixture(scope="module", autouse=True)
def _isa_context():
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([os.path.join(rocm_path, "bin"), os.path.join(rocm_path, "lib", "llvm", "bin")])
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    rocisa.rocIsa.getInstance().init(_ISA, assembler, False)
    rocisa.rocIsa.getInstance().setKernel(_ISA, 32)


@pytest.mark.parametrize("keep", [False, True])
def test_expanded_waits_carry_keep_wait(keep):
    wait = SWaitCnt(vlcnt=0, dscnt=0, comment="staging stores done")
    wait.keepWait = keep
    instructions = wait.getInstructions()
    assert instructions
    assert [instruction.keepWait for instruction in instructions] == [keep] * len(instructions)
