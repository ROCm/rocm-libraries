################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Regression: MUBUF isOff=True vaddr must survive rocisa → StinkyTofu as 'off', not 'v[vgproff]'."""

import re

import pytest
import rocisa
from rocisa.code import Module, SignatureBase
from rocisa.container import sgpr, vgpr
from rocisa.instruction import BufferStoreB32

_ISA = (12, 5, 0)

# Skip entire module when the target backend isn't compiled into the registry.
pytestmark = pytest.mark.skipif(
    not rocisa.isSupportedByStinkyTofu(_ISA),
    reason=f"gfx{''.join(str(v) for v in _ISA)} not registered in StinkyTofu BackendRegistry",
)


@pytest.fixture(scope="module", autouse=True)
def _isa_context():
    import os

    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    rocisa.rocIsa.getInstance().init(_ISA, rocm_path + "/bin/amdclang++", False)
    rocisa.rocIsa.getInstance().setKernel(_ISA, 32)


@pytest.fixture(scope="module")
def _mubuf_off_asm() -> str:
    mod = Module("mubuf_off_vaddr")
    mod.add(
        BufferStoreB32(
            src=vgpr(12),
            vaddr=vgpr("off", isOff=True),  # isOff → MUBUF 'off' keyword, not a named VGPR
            saddr=sgpr(60, 4),
            soffset=sgpr(46),
        )
    )
    mod.setParent()  # resolves symbolic register names before conversion

    sig = SignatureBase(
        kernelName="mubuf_off_vaddr",
        kernArgsVersion=1,
        codeObjectVersion="4",
        groupSegmentSize=0,
        sgprWorkGroup=(1, 1, 0),
        vgprWorkItem=0,
        flatWorkGroupSize=64,
        preloadKernArgs=False,
    )

    stinky_module_options = {"OptLevel": 0}
    st = rocisa.toStinkyTofuModule(mod, _ISA, "mubuf_off_vaddr", signature=sig, options=stinky_module_options)
    st.runOptimizationPipeline()
    return st.emitAssembly()


def test_mubuf_off_vaddr_stinkytofu(_mubuf_off_asm):
    # Assembler rejects 'v[vgproff]'; 'off' must appear as the literal vaddr operand.
    assert re.search(r"buffer_store_b32 v12, off, s\[60:63\], s46", _mubuf_off_asm)
