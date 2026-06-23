# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regressions for rocisa::SBarrierSignalIsFirst -> StinkyTofu lowering (gfx1250)."""

import pytest
import rocisa
from rocisa.code import Module, SignatureBase
from rocisa.instruction import SBarrierSignalIsFirst, SEndpgm

_ISA = (12, 5, 0)

pytestmark = pytest.mark.skipif(
    not rocisa.isSupportedByStinkyTofu(_ISA),
    reason=f"gfx{''.join(str(v) for v in _ISA)} not registered in StinkyTofu BackendRegistry",
)


@pytest.fixture(scope="module", autouse=True)
def _isa_context():
    import os
    import shutil

    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    rocisa.rocIsa.getInstance().init(_ISA, assembler, False)
    rocisa.rocIsa.getInstance().setKernel(_ISA, 32)


def _emit(mod: Module, name: str) -> str:
    mod.setParent()
    sig = SignatureBase(
        kernelName=name,
        kernArgsVersion=1,
        codeObjectVersion="4",
        groupSegmentSize=0,
        sgprWorkGroup=(1, 1, 0),
        vgprWorkItem=0,
        flatWorkGroupSize=64,
        numSgprPreload=0,
    )
    st = rocisa.toStinkyTofuModule(mod, _ISA, name, signature=sig, options={"OptLevel": 0})
    st.runOptimizationPipeline()
    return st.emitAssembly()


def test_s_barrier_signal_isfirst_lowers_and_emits():
    mod = Module("isfirst_barrier")
    mod.add(SBarrierSignalIsFirst(False, "workgroup barrier signal (isfirst)"))
    mod.add(SEndpgm(comment="end"))
    asm = _emit(mod, "isfirst_barrier")
    assert "s_barrier_signal_isfirst" in asm
    assert "s_barrier_signal_isfirst -1" in asm
