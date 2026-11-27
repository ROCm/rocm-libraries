# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 cache scope on SMEM loads and on global load / store."""

import os
import shutil

import pytest
import rocisa
from rocisa.code import Module, SignatureBase
from rocisa.container import GLOBALModifiers, SMEMModifiers, sgpr, vgpr
from rocisa.enum import CacheScope
from rocisa.instruction import (
    GlobalLoadB32,
    GlobalStoreB32,
    GlobalStoreB128,
    SLoadB64,
)

_ISA = (12, 5, 0)

pytestmark = pytest.mark.skipif(
    not rocisa.isSupportedByStinkyTofu(_ISA),
    reason=f"gfx{''.join(str(v) for v in _ISA)} not registered in StinkyTofu BackendRegistry",
)


@pytest.fixture(scope="module", autouse=True)
def _isa_context():
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    rocisa.rocIsa.getInstance().init(_ISA, assembler, False)
    rocisa.rocIsa.getInstance().setKernel(_ISA, 32)


_CASES = {
    "s_load_b64": (
        lambda: SLoadB64(dst=sgpr(0, 2), base=sgpr(4, 2), soffset=0,
                         smem=SMEMModifiers(scope=CacheScope.SCOPE_SYS)),
        "s_load_b64 s[0:1], s[4:5], 0 scope:SCOPE_SYS",
    ),
    "global_load_b32": (
        lambda: GlobalLoadB32(dst=vgpr(1), vaddr=vgpr(0), saddr=sgpr(2, 2),
                              modifier=GLOBALModifiers(offset=4, scope=CacheScope.SCOPE_SYS)),
        "global_load_b32 v1, v0, s[2:3] offset:4 scope:SCOPE_SYS",
    ),
    "global_store_b32": (
        lambda: GlobalStoreB32(vaddr=vgpr(0), src=vgpr(1), saddr=sgpr(2, 2),
                               modifier=GLOBALModifiers(scope=CacheScope.SCOPE_SYS)),
        "global_store_b32 v0, v1, s[2:3] scope:SCOPE_SYS",
    ),
    "global_store_b128_off": (
        lambda: GlobalStoreB128(vaddr=vgpr(0, 2), src=vgpr(4, 4),
                                saddr=vgpr("off", 1, False, False, True),
                                modifier=GLOBALModifiers(offset=16, scope=CacheScope.SCOPE_DEV)),
        "global_store_b128 v[0:1], v[4:7], off offset:16 scope:SCOPE_DEV",
    ),
}

_case = pytest.mark.parametrize("build, text", _CASES.values(), ids=_CASES.keys())


@_case
def test_rocisa_text(build, text):
    assert str(build()).strip() == text


@_case
@pytest.mark.parametrize("viaLogicalIR", [True, False], ids=["logical", "direct"])
def test_stinkytofu_text(build, text, viaLogicalIR):
    mod = Module("k")
    mod.add(build())
    mod.setParent()
    sig = SignatureBase(
        kernelName="k",
        kernArgsVersion=1,
        codeObjectVersion="4",
        groupSegmentSize=0,
        sgprWorkGroup=(1, 1, 0),
        vgprWorkItem=0,
        flatWorkGroupSize=64,
        numSgprPreload=0,
    )
    st = rocisa.toStinkyTofuModule(
        mod, _ISA, "k", signature=sig,
        options={"OptLevel": 0, "LowerViaLogicalIR": viaLogicalIR},
    )
    st.runOptimizationPipeline()

    assert [ln.strip() for ln in st.emitAssembly().splitlines() if ln.strip().startswith(text.split()[0])] == [text]
