################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Regressions for lane-select rocisa -> StinkyTofu lowering (gfx1250).

Covers the two instructions the auto-WGMXCC path (``WorkGroupMappingXCC: -1``)
emits from ``scalarUInt24DivideAndRemainderPair``:

* ``VWritelaneB32`` -> ``v_writelane_b32 vdst, ssrc0, ssrc1``
* ``VReadlaneB32``  -> ``v_readlane_b32 sdst, vsrc0, ssrc1``

Both existed on the rocisa side but had no gfx1250 hardware definition or
Rocisa conversion entry, so kernel generation aborted with
"No conversion entry for rocisa N6rocisa13VWritelaneB32E in arch gfx1250".
No gfx1250 config in Tensile/Tests exercises ``WorkGroupMappingXCC: -1``
(``common/streamk/sk_sgemm_quick.yaml`` covers it but is marked skip-gfx1250),
so the conversion is locked down here instead.

The boilerplate (ISA init fixture, SignatureBase setup, toStinkyTofuModule
invocation) mirrors ``test_streamk_fences.py``.
"""

import re

import pytest
import rocisa
from rocisa.code import Module, SignatureBase
from rocisa.container import sgpr, vgpr
from rocisa.instruction import VMovB32, VReadlaneB32, VWritelaneB32

_ISA = (12, 5, 0)

# Skip entire module when the target backend isn't compiled into the registry.
pytestmark = pytest.mark.skipif(
    not rocisa.isSupportedByStinkyTofu(_ISA),
    reason=f"gfx{''.join(str(v) for v in _ISA)} not registered in StinkyTofu BackendRegistry",
)


# rocIsa is a process-wide singleton, so selecting gfx1250 here leaks into any
# module that runs afterwards and does not set an ISA of its own
# (test_macro_inline expects the gfx9 `v_add_u32` spelling, not `v_add_nc_u32`).
# Restore the gfx9 kernel test_base.py establishes once this module is done.
_RESTORE_ISA = (9, 0, 10)
_RESTORE_WAVE = 64


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
    isa = rocisa.rocIsa.getInstance()
    isa.init(_ISA, assembler, False)
    isa.setKernel(_ISA, 32)

    yield

    isa.init(_RESTORE_ISA, assembler, False)
    isa.setKernel(_RESTORE_ISA, _RESTORE_WAVE)


def _emit(mod: Module, name: str) -> str:
    """Convert ``mod`` to a stinkytofu module and return the emitted assembly."""
    mod.setParent()  # resolves symbolic register names before conversion

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

    st = rocisa.toStinkyTofuModule(
        mod, _ISA, name, signature=sig, options={"OptLevel": 0}
    )
    st.runOptimizationPipeline()
    return st.emitAssembly()


# ---------------------------------------------------------------------------
# v_writelane_b32 / v_readlane_b32
# ---------------------------------------------------------------------------

def test_writelane_immediate_lane_emits_to_stinkytofu():
    mod = Module("writelane_imm_lane")
    mod.add(VWritelaneB32(dst=vgpr(0), src0=sgpr(0), src1=1))

    asm = _emit(mod, "writelane_imm_lane")

    assert re.search(r"v_writelane_b32\s+v0,\s*s0,\s*1", asm), (
        f"expected 'v_writelane_b32 v0, s0, 1' in emitted assembly, got:\n{asm}"
    )


def test_readlane_immediate_lane_emits_to_stinkytofu():
    mod = Module("readlane_imm_lane")
    mod.add(VReadlaneB32(dst=sgpr(0), src0=vgpr(0), src1=1))

    asm = _emit(mod, "readlane_imm_lane")

    assert re.search(r"v_readlane_b32\s+s0,\s*v0,\s*1", asm), (
        f"expected 'v_readlane_b32 s0, v0, 1' in emitted assembly, got:\n{asm}"
    )


def test_readlane_sgpr_lane_select_emits_to_stinkytofu():
    """The lane select also accepts an SGPR, not just an inline constant.

    ``KernelWriterAssembly`` passes ``src1=sgpr(...)``; the gfx1250 operand
    field is typed ``ssrc``, which accepts scalar registers but rejects VGPRs.
    """
    mod = Module("readlane_sgpr_lane")
    mod.add(VReadlaneB32(dst=sgpr(0), src0=vgpr(0), src1=sgpr(1)))

    asm = _emit(mod, "readlane_sgpr_lane")

    assert re.search(r"v_readlane_b32\s+s0,\s*v0,\s*s1", asm), (
        f"expected 'v_readlane_b32 s0, v0, s1' in emitted assembly, got:\n{asm}"
    )


# ---------------------------------------------------------------------------
# auto-WGMXCC pair-wise divide sequence
# ---------------------------------------------------------------------------

def test_writelane_readlane_roundtrip_sequence():
    """The shape ``scalarUInt24DivideAndRemainderPair`` builds.

    A broadcast ``v_mov_b32`` seeds every lane, ``v_writelane_b32`` overwrites
    lane 1 only, and ``v_readlane_b32`` reads both lanes back. ``v_writelane_b32``
    is a read-modify-write of its destination, so the seeding ``v_mov_b32`` must
    survive lowering for lane 0 to hold a defined value.
    """
    mod = Module("lane_roundtrip")
    mod.add(VMovB32(dst=vgpr(0), src=sgpr(0)))
    mod.add(VWritelaneB32(dst=vgpr(0), src0=sgpr(1), src1=1))
    mod.add(VReadlaneB32(dst=sgpr(2), src0=vgpr(0), src1=0))
    mod.add(VReadlaneB32(dst=sgpr(3), src0=vgpr(0), src1=1))

    asm = _emit(mod, "lane_roundtrip")

    for expected in (
        r"v_mov_b32(_e32)?\s+v0,\s*s0",
        r"v_writelane_b32\s+v0,\s*s1,\s*1",
        r"v_readlane_b32\s+s2,\s*v0,\s*0",
        r"v_readlane_b32\s+s3,\s*v0,\s*1",
    ):
        assert re.search(expected, asm), (
            f"expected /{expected}/ in emitted assembly, got:\n{asm}"
        )
