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

"""Regressions for MUBUF rocisa -> StinkyTofu lowering."""

import re

import pytest
import rocisa
from rocisa.code import Module, SignatureBase
from rocisa.container import MUBUFModifiers, sgpr, vgpr
from rocisa.enum import CacheScope
from rocisa.instruction import (
    BufferAtomicAddF32,
    BufferLoadB32,
    BufferLoadB64,
    BufferLoadB128,
    BufferStoreB32,
)

_ISA = (12, 5, 0)

# Skip entire module when the target backend isn't compiled into the registry.
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
    mod.add(
        BufferStoreB32(
            src=vgpr(13),
            vaddr=vgpr("off", isOff=True),
            saddr=sgpr(64, 4),
            soffset=0,
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
    assert re.search(r"buffer_store_b32 v12, off, s\[60:63\], s46", _mubuf_off_asm)
    assert re.search(r"buffer_store_b32 v13, off, s\[64:67\], null", _mubuf_off_asm)


@pytest.fixture(scope="module")
def _mubuf_scope_asm() -> str:
    mod = Module("mubuf_scope_modifiers")
    mod.add(
        BufferStoreB32(
            src=vgpr(12),
            vaddr=vgpr(32),
            saddr=sgpr(60, 4),
            soffset=sgpr(46),
            mubuf=MUBUFModifiers(offen=True, scope=CacheScope.SCOPE_DEV),
        )
    )
    mod.add(
        BufferLoadB32(
            dst=vgpr(13),
            vaddr=vgpr(33),
            saddr=sgpr(64, 4),
            soffset=sgpr(47),
            mubuf=MUBUFModifiers(offen=True, scope=CacheScope.SCOPE_DEV),
        )
    )
    mod.setParent()

    sig = SignatureBase(
        kernelName="mubuf_scope_modifiers",
        kernArgsVersion=1,
        codeObjectVersion="4",
        groupSegmentSize=0,
        sgprWorkGroup=(1, 1, 0),
        vgprWorkItem=0,
        flatWorkGroupSize=64,
        preloadKernArgs=False,
    )

    st = rocisa.toStinkyTofuModule(
        mod, _ISA, "mubuf_scope_modifiers", signature=sig, options={"OptLevel": 0}
    )
    st.runOptimizationPipeline()
    return st.emitAssembly()


def test_mubuf_scope_modifiers_stinkytofu(_mubuf_scope_asm):
    assert re.search(
        r"buffer_store_b32 v12, v32, s\[60:63\], s46 offen offset:0 scope:SCOPE_DEV",
        _mubuf_scope_asm,
    )
    assert re.search(
        r"buffer_load_b32 v13, v33, s\[64:67\], s47 offen offset:0 scope:SCOPE_DEV",
        _mubuf_scope_asm,
    )


def test_rocisa_load_null_soffset_adds_offen():
    inst = BufferLoadB32(
        dst=vgpr(13),
        vaddr=vgpr(33),
        saddr=sgpr(64, 4),
        soffset=0,
    )

    assert str(inst).strip() == "buffer_load_b32 v13, v33, s[64:67], null offen offset:0"


def test_rocisa_store_null_soffset_adds_offen():
    inst = BufferStoreB32(
        src=vgpr(12),
        vaddr=vgpr(32),
        saddr=sgpr(60, 4),
        soffset=0,
    )

    assert str(inst).strip() == "buffer_store_b32 v12, v32, s[60:63], null offen offset:0"


def test_rocisa_atomic_null_soffset_adds_offen():
    inst = BufferAtomicAddF32(
        src=vgpr(12),
        vaddr=vgpr(32),
        saddr=sgpr(60, 4),
        soffset=0,
    )

    assert str(inst).strip() == "buffer_atomic_add_f32 v12, v32, s[60:63], null offen offset:0"


def test_rocisa_off_vaddr_null_soffset():
    inst = BufferStoreB32(
        src=vgpr(12),
        vaddr=vgpr("off", isOff=True),
        saddr=sgpr(60, 4),
        soffset=0,
    )

    assert str(inst).strip() == "buffer_store_b32 v12, off, s[60:63], null"


@pytest.fixture(scope="module")
def _mubuf_zero_soffset_asm() -> str:
    mod = Module("mubuf_zero_soffset")
    mod.add(
        BufferLoadB128(
            dst=vgpr(0, 4),
            vaddr=vgpr(32),
            saddr=sgpr(60, 4),
            soffset=0,
            mubuf=MUBUFModifiers(offen=True),
            comment="G -> Reg 0_0_0_0",
        )
    )
    mod.add(
        BufferLoadB64(
            dst=vgpr(4, 2),
            vaddr=vgpr(33),
            saddr=sgpr(60, 4),
            soffset=0,
            comment="G -> Reg 0_0_1_0",
        )
    )
    mod.add(
        BufferLoadB32(
            dst=vgpr(6),
            vaddr=vgpr(34),
            saddr=sgpr(60, 4),
            soffset=0,
            mubuf=MUBUFModifiers(offen=False),
            comment="G -> Reg 0_0_2_0",
        )
    )
    mod.setParent()

    sig = SignatureBase(
        kernelName="mubuf_zero_soffset",
        kernArgsVersion=1,
        codeObjectVersion="4",
        groupSegmentSize=0,
        sgprWorkGroup=(1, 1, 0),
        vgprWorkItem=0,
        flatWorkGroupSize=64,
        preloadKernArgs=False,
    )

    st = rocisa.toStinkyTofuModule(
        mod, _ISA, "mubuf_zero_soffset", signature=sig, options={"OptLevel": 0}
    )
    st.runOptimizationPipeline()
    return st.emitAssembly()


def test_stinky_null_soffset_explicit_offen(_mubuf_zero_soffset_asm):
    assert re.search(
        r"buffer_load_b128 v\[0:3\], v32, s\[60:63\], null offen offset:0",
        _mubuf_zero_soffset_asm,
    )


def test_stinky_null_soffset_default_mubuf(_mubuf_zero_soffset_asm):
    assert re.search(
        r"buffer_load_b64 v\[4:5\], v33, s\[60:63\], null offen offset:0",
        _mubuf_zero_soffset_asm,
    )


def test_stinky_null_soffset_offen_false(_mubuf_zero_soffset_asm):
    assert re.search(
        r"buffer_load_b32 v6, v34, s\[60:63\], null offen offset:0",
        _mubuf_zero_soffset_asm,
    )
