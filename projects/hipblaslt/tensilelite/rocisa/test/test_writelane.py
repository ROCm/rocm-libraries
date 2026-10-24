# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Partial lane writes retain the other lanes through native lowering."""

import os
import re
import shutil

import pytest
import rocisa
from rocisa.code import Module, SignatureBase
from rocisa.container import sgpr, vgpr
from rocisa.instruction import BufferStoreB128, VMovB32, VReadlaneB32, VWritelaneB32

_ISA = (12, 5, 0)
pytestmark = pytest.mark.skipif(
    not rocisa.isSupportedByStinkyTofu(_ISA), reason="gfx1250 backend unavailable")


@pytest.fixture(autouse=True)
def _isa_context():
    isa = rocisa.rocIsa.getInstance()
    previous = isa.getKernel()
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    search = os.pathsep.join([os.path.join(rocm, "bin"),
                              os.path.join(rocm, "lib", "llvm", "bin")])
    assembler = shutil.which("amdclang++", path=search) or "amdclang++"
    isa.init(_ISA, assembler, False)
    isa.setKernel(_ISA, 32)
    try:
        yield
    finally:
        isa.setKernel(tuple(previous.isa), previous.wavefrontSize)


@pytest.mark.parametrize("payload", [32, 673])
@pytest.mark.parametrize("opt_level", [0, 2])
def test_successive_lane_writes_preserve_all_coordinates(payload, opt_level, capfd):
    module = Module("packed_coordinates")
    # Every write preserves the three coordinates stored in other lanes. The
    # later store makes all four scalar results observable to native passes.
    for lane in range(4):
        module.add(VWritelaneB32(dst=vgpr(payload), src0=sgpr(lane), src1=lane))
    for lane in range(4):
        module.add(VReadlaneB32(dst=sgpr(8 + lane), src0=vgpr(payload), src1=lane))
        module.add(VMovB32(dst=vgpr(1 + lane), src=sgpr(8 + lane)))
    module.add(BufferStoreB128(src=vgpr(1, 4), vaddr=vgpr("off", isOff=True),
                               saddr=sgpr(20, 4), soffset=0))
    module.setParent()
    signature = SignatureBase(
        kernelName="packed_coordinates", kernArgsVersion=1, codeObjectVersion="4",
        groupSegmentSize=0, sgprWorkGroup=(1, 1, 0), vgprWorkItem=0,
        flatWorkGroupSize=32, numSgprPreload=0)
    native = rocisa.toStinkyTofuModule(
        module, _ISA, "packed_coordinates", signature=signature,
        options={"OptLevel": opt_level, "wavefrontSize": 32,
                 "TileA0": 4, "TileB0": 4, "TileM0": 16})
    native.runOptimizationPipeline()
    assembly = native.emitAssembly()
    diagnostics = capfd.readouterr()
    assert "read-write" not in diagnostics.err.lower(), diagnostics.err
    assert "StinkyIRVerifier" not in diagnostics.err, diagnostics.err
    lines = [line.split("//", 1)[0].strip() for line in assembly.splitlines()]
    writes = [(i, line) for i, line in enumerate(lines) if line.startswith("v_writelane_b32 ")]
    reads = [(i, line) for i, line in enumerate(lines) if line.startswith("v_readlane_b32 ")]
    assert len(writes) == len(reads) == 4, assembly
    assert max(i for i, _ in writes) < min(i for i, _ in reads), assembly
    # The old destination is a dependency only, never a fourth printed operand.
    assert all(len(line.split(",")) == 3 for _, line in writes + reads), assembly
    assert {int(line.rsplit(",", 1)[1]) for _, line in writes} == set(range(4)), assembly
    assert {int(line.rsplit(",", 1)[1]) for _, line in reads} == set(range(4)), assembly
    if payload > 255:
        assert "s_set_vgpr_msb" in assembly, assembly
        assert all(re.search(r"v\[673-512\]", line) for _, line in writes + reads), assembly
