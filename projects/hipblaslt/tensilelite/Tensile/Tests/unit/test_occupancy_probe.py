# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the CU-occupancy probe (SupportOccupancyProbe) codegen."""

import contextlib
import os
import re
import shutil
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly

import rocisa
from rocisa.code import Label, SignatureBase
from rocisa.instruction import (
    GlobalStoreB32,
    SCBranchSCC0,
    SCBranchSCC1,
    SCmpEQU64,
    SCmpGeU32,
    SEndpgm,
    SGetRegB32,
    SLoadB32,
    SLoadB64,
    VCmpXEqU32,
)
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType, SignatureValueKind as SVK

from Tensile.Common.GlobalParameters import globalParameters
from Tensile.Components.Signature import addOccupancyProbeArgs
from Tensile.ExecutionPolicy import normalize_execution_policy
from Tensile.SolutionStructs.Solution import _supportOccupancyProbe
from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit


@pytest.fixture
def emit_probe_flag():
    saved = globalParameters.get("EmitOccupancyProbe", False)
    yield lambda value: globalParameters.__setitem__("EmitOccupancyProbe", value)
    globalParameters["EmitOccupancyProbe"] = saved


def _state(streamk=5, isa=(9, 5, 0), grouped=False):
    state = normalize_execution_policy({"StreamK": streamk})
    state["ISA"] = isa
    state["ProblemType"] = {"GroupedGemm": grouped}
    return state


@pytest.mark.parametrize(
    "flag, kwargs, expected",
    [
        (False, {}, False),
        (True, {}, True),
        (True, {"isa": (9, 4, 2)}, True),
        (True, {"streamk": 3}, False),
        (True, {"streamk": 4}, False),
        (True, {"grouped": True}, False),
        (True, {"isa": (12, 5, 0)}, False),
    ],
)
def test_support_occupancy_probe_gate(emit_probe_flag, flag, kwargs, expected):
    emit_probe_flag(flag)
    assert _supportOccupancyProbe(_state(**kwargs)) is expected


class _Labels:
    def __init__(self):
        self._count = 0

    def getNameInc(self, name):
        self._count += 1
        return "%s_%u" % (name, self._count)


def _probe_writer(offset):
    kwa = KernelWriterAssembly.__new__(KernelWriterAssembly)
    kwa.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    kwa.sgprPool.add(0, 16, "unit")
    kwa.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    kwa.vgprPool.add(0, 8, "unit")
    kwa.vgprPool.checkOut(1, "Serial")
    kwa.labels = _Labels()
    kwa.db = {"AssertOnSgprOverflow": True}
    kwa.states = SimpleNamespace(probeKernArgOffset=offset, laneSGPRCount=2,
                                 regCaps={"MaxSgpr": 102})
    return kwa


def test_occupancy_probe_absent_without_offset():
    module = _probe_writer(-1).occupancyProbe({"WavefrontSize": 64})
    assert list(module.flatitems()) == []


def test_occupancy_probe_emission():
    with preserve_rocisa_kernel_state():
        ri = rocisa.rocIsa.getInstance()
        ri.init((9, 5, 0), shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++")
        ri.setKernel((9, 5, 0), 64)
        kwa = _probe_writer(200)
        module = kwa.occupancyProbe({"WavefrontSize": 64})
        items = list(module.flatitems())
        text = str(module)
        lines = [str(i) for i in items]

    loads = [(i, line) for i, line in zip(items, lines) if isinstance(i, (SLoadB64, SLoadB32))]
    assert [type(i) for i, _ in loads] == [SLoadB64, SLoadB64]
    assert "0xc8" in loads[0][1] and "0xd0" in loads[1][1]
    assert "hwreg(HW_REG_HW_ID,8,8)" in text
    assert "hwreg(HW_REG_XCC_ID,0,4)" in text
    assert sum(isinstance(i, SGetRegB32) for i in items) == 2

    # Null address skips everything; only thread 0 stores, with a plain store;
    # the raw-id test is set before the store and ends probe-only WGs after it.
    order = [type(i) for i in items
             if isinstance(i, (SCmpEQU64, SCBranchSCC0, SCBranchSCC1, SCmpGeU32, VCmpXEqU32,
                               GlobalStoreB32, SEndpgm, Label))]
    assert order == [SCmpEQU64, SCBranchSCC1, SCmpGeU32, VCmpXEqU32, GlobalStoreB32,
                     SCBranchSCC0, SEndpgm, Label]
    store = next(line for i, line in zip(items, lines) if isinstance(i, GlobalStoreB32))
    assert store.startswith("global_store_dword ")
    assert " sc0" not in store and " sc1" not in store and " nt" not in store
    labelName = next(i for i in items if isinstance(i, Label)).getLabelName()
    assert all(labelName in line for i, line in zip(items, lines)
               if isinstance(i, (SCBranchSCC0, SCBranchSCC1)))

    # Raw pre-remap WG id (StreamKTileIdx snapshot) against ProbeGrid, the
    # dword after ProbeEpoch in the second load.
    cmp = next(line for i, line in zip(items, lines) if isinstance(i, SCmpGeU32))
    gridLoad = re.match(r"s_load_dwordx2 s\[(\d+):(\d+)\]", loads[1][1])
    assert re.match(r"s_cmp_ge_u32 s\[sgprStreamKTileIdx\], s%s\b" % gridLoad.group(2), cmp)

    # Nothing between the compare and the exit branch writes SCC.
    cmpIdx = next(n for n, i in enumerate(items) if isinstance(i, SCmpGeU32))
    brIdx = next(n for n, i in enumerate(items) if isinstance(i, SCBranchSCC0))
    between = lines[cmpIdx + 1:brIdx]
    assert all(re.match(r"(s_mov_b(32|64)|v_cmpx_\w+|s_nop|global_store_dword) ", l) for l in between)

    # Exec is narrowed by v_cmpx and restored from the same pair after the store.
    cmpx = next(line for i, line in zip(items, lines) if isinstance(i, VCmpXEqU32))
    assert cmpx.startswith("v_cmpx_eq_u32 exec,")
    storeIdx = next(n for n, i in enumerate(items) if isinstance(i, GlobalStoreB32))
    save = [line for line in lines[:storeIdx] if re.match(r"s_mov_b64 s\[\d+:\d+\], exec\b", line)]
    restore = [line for line in lines[storeIdx + 1:] if line.startswith("s_mov_b64 exec, ")]
    assert len(save) == 1 and len(restore) == 1
    assert restore[0].split()[2] == save[0].split()[1].rstrip(",")

    # Four SGPRs at most; temps are returned to the pools.
    sgprs = set()
    for line in lines:
        for lo, hi in re.findall(r"\bs\[(\d+):(\d+)\]", line):
            sgprs.update(range(int(lo), int(hi) + 1))
        sgprs.update(int(n) for n in re.findall(r"\bs(\d+)\b", line))
    assert len(sgprs) <= 4
    assert kwa.sgprPool.available() == 16
    assert kwa.vgprPool.available() == 7


def _signature(numU32):
    sig = SignatureBase(kernelName="probe_layout", kernArgsVersion=3, codeObjectVersion="5",
                        groupSegmentSize=0, sgprWorkGroup=[1, 1, 1], vgprWorkItem=0,
                        flatWorkGroupSize=256)
    for n in range(numU32):
        sig.addArg("arg%u" % n, SVK.SIG_VALUE, "u32")
    return sig


@pytest.mark.parametrize("numU32, padded", [(53, True), (54, False)])
def test_occupancy_probe_signature_layout(numU32, padded):
    commonArgsSize = 16
    writer = SimpleNamespace(states=SimpleNamespace(probeKernArgOffset=-1))
    sig = _signature(numU32)
    assert (sig.offset % 8 == 4) is padded
    addOccupancyProbeArgs(writer, sig, commonArgsSize)
    meta = str(sig)
    args = dict(re.findall(r"- \.name:\s+(\w+)\n\s+\.size:\s+\d+\n\s+\.offset:\s+(\d+)", meta))
    addr = writer.states.probeKernArgOffset + commonArgsSize
    assert ("ProbePad" in args) is padded
    assert addr % 8 == 0
    assert int(args["ProbeAddr"]) == addr
    assert int(args["ProbeEpoch"]) == addr + 8
    assert int(args["ProbeGrid"]) == addr + 12
    assert sig.offset == addr + 16
    # Host packing: appendAligned<void*> then 2 x append<uint32_t> from numU32 * 4 bytes.
    hostAddr = (numU32 * 4 + 7) // 8 * 8
    assert hostAddr == addr


def test_occupancy_probe_kernarg_matches_loads():
    """End-to-end SK5 gfx950 emit: metadata offsets match the probe s_loads."""
    from Tensile.Tests.unit.characterization._codegen.codegen_harness import emit_kernels_from_logic
    import Tensile.Tests.unit.characterization._codegen.codegen_harness as harness

    logic = os.path.join(os.path.dirname(__file__), "characterization", "_codegen", "data",
                         "bigfiles", "equality_gfx950_HSS_big.yaml")
    orig = harness._isolated_globals

    @contextlib.contextmanager
    def withProbe():
        with orig():
            globalParameters["EmitOccupancyProbe"] = True
            yield

    harness._isolated_globals = withProbe
    try:
        results = emit_kernels_from_logic(logic, limit=1)
    finally:
        harness._isolated_globals = orig
    (_, src, err), = results
    assert err == 0
    args = dict(re.findall(r"- \.name:\s+(\w+)\n\s+\.size:\s+\d+\n\s+\.offset:\s+(\d+)", src))
    loads = re.findall(r"s_load_dword\w* s\S+, s\[sgprKernArgAddress:sgprKernArgAddress\+1\], "
                       r"(0x[0-9a-f]+)\s+// load (ProbeAddr|ProbeEpoch, ProbeGrid)", src)
    offsets = {name: int(off, 16) + 16 for off, name in loads}
    assert int(args["ProbeAddr"]) % 8 == 0
    assert offsets == {"ProbeAddr": int(args["ProbeAddr"]),
                       "ProbeEpoch, ProbeGrid": int(args["ProbeEpoch"])}
    assert int(args["ProbeGrid"]) == int(args["ProbeEpoch"]) + 4
    size = int(re.search(r"kernarg_segment_size:\s+(\d+)", src).group(1))
    assert size == (int(args["ProbeGrid"]) + 4 + 7) // 8 * 8

    # The raw WG id is snapshotted before the wgmXCC remap and read by the
    # probe exit; nothing between kernel entry and the exit stores, does an
    # atomic or touches the work queue.
    body = src[src.index(".text") if ".text" in src else 0:]
    snap = body.index("snapshot raw pre-remap launch WG id")
    remap = body.index("remap workgroup to XCCs")
    exit_ = body.index("probe-only WG: exit before any other side effect")
    assert snap < remap < exit_
    prologue = body[:exit_]
    stores = re.findall(r"^\s*((?:global|buffer|flat|scratch)_(?:store|atomic)\w*|s_store\w*|s_atomic\w*|ds_\w+)",
                        prologue, re.M)
    assert stores == ["global_store_dword"]
