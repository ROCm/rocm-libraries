# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Duplicate-assignment optimization preserves special-register restores."""

import os
import re
import shutil

import pytest
import rocisa
from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.code import KernelBody, Module
from rocisa.container import EXEC, EXECHI, EXECLO, VCC, sgpr
from rocisa.instruction import SMovB32


@pytest.fixture(scope="module", autouse=True)
def _isa_context():
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    search = os.pathsep.join([
        os.path.join(rocm, "bin"), os.path.join(rocm, "lib", "llvm", "bin")])
    assembler = shutil.which("amdclang++", path=search) or "amdclang++"
    rocisa.rocIsa.getInstance().init((12, 5, 0), assembler, False)
    rocisa.rocIsa.getInstance().setKernel((12, 5, 0), 32)


@pytest.mark.parametrize("special", ["vcc_lo", "vcc_hi", "exec", "exec_lo", "exec_hi"])
def test_special_register_restores_survive_duplicate_assignment(special):
    destinations = {
        "vcc_lo": VCC(), "vcc_hi": VCC(setHi=True),
        "exec": EXEC(), "exec_lo": EXECLO(), "exec_hi": EXECHI(),
    }
    destination = destinations[special]
    module = Module("restore_special_register")
    # Each restore is reached through s0's graph; its special destination is
    # not a RegisterContainer and must not be optimized as an ordinary SGPR.
    module.add(SMovB32(dst=sgpr(0), src=1))
    module.add(SMovB32(dst=destination, src=sgpr(0)))
    module.add(SMovB32(dst=sgpr(0), src=2))
    module.add(SMovB32(dst=destination, src=sgpr(0)))
    # A normal duplicate still disappears with the same pass options.
    module.add(SMovB32(dst=sgpr(1), src=7))
    module.add(SMovB32(dst=sgpr(1), src=7))
    kernel = KernelBody("restore_special_register")
    kernel.addBody(module)
    kernel.setGprs(1, 0, 2)
    options = rocIsaPassOption()
    options.removeDupFunc = False
    options.removeDupAssign = True
    options.getCycles = False
    rocIsaPass(kernel, options)

    instructions = [line.strip() for line in str(module).splitlines()]
    restores = [i for i, line in enumerate(instructions)
                if re.match(rf"s_mov_b32 {str(destination)}, s0(?:\s|$)", line)]
    assert len(restores) == 2, instructions
    assert re.match(r"s_mov_b32 s0, 1(?:\s|$)", instructions[restores[0] - 1])
    assert re.match(r"s_mov_b32 s0, 2(?:\s|$)", instructions[restores[1] - 1])
    assert sum(bool(re.match(r"s_mov_b32 s1, 7(?:\s|$)", line))
               for line in instructions) == 1, instructions
