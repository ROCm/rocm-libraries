# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the CU-occupancy probe (SupportOccupancyProbe) codegen."""

import os
import shutil
from types import SimpleNamespace

import pytest
import yaml

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly

import rocisa
from rocisa.code import Label, SignatureBase
from rocisa.instruction import GlobalStoreB32, SCBranchSCC0, SCBranchSCC1, SLoadB64, SNop
from rocisa.label import LabelManager
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType, SignatureValueKind as SVK

from Tensile.Common.GlobalParameters import defaultInternalSupportParams, globalParameters
from Tensile.Components.Signature import addOccupancyProbeArgs
from Tensile.Contractions import InternalArgsSupport
from Tensile.ExecutionPolicy import normalize_execution_policy
from Tensile.SolutionStructs.Solution import _supportOccupancyProbe

pytestmark = pytest.mark.unit

PROBE_OFFSET = 200


def _state(streamk=5, isa=(9, 5, 0), grouped=False, custom=None):
    state = normalize_execution_policy({"StreamK": streamk})
    state["ISA"] = isa
    state["ProblemType"] = {"GroupedGemm": grouped}
    if custom is not None:
        state["CustomKernel"] = custom
    return state


@pytest.mark.parametrize(
    "flag, kwargs, expected",
    [
        (False, {}, False),
        (True, {}, True),
        (True, {"isa": (9, 4, 2)}, False),
        (True, {"isa": (12, 5, 0)}, False),
        (True, {"streamk": 3}, False),
        (True, {"streamk": 4}, False),
        (True, {"grouped": True}, False),
        (True, {"custom": {"name": "handwritten"}}, False),
        (True, {"custom": {"name": "gen", "generated": True}}, True),
    ],
)
def test_support_occupancy_probe_gate(monkeypatch, flag, kwargs, expected):
    monkeypatch.setitem(globalParameters, "EmitOccupancyProbe", flag)
    assert _supportOccupancyProbe(_state(**kwargs)) is expected


@pytest.mark.parametrize("support", [False, True])
def test_internal_args_support_state_emits_probe_only_when_set(support):
    isp = dict(defaultInternalSupportParams, SupportOccupancyProbe=support)
    state = InternalArgsSupport.FromOriginalState({"InternalSupportParams": isp}).state()
    assert state.get("occupancyProbe") is (True if support else None)


def _probe_writer(offset):
    kwa = KernelWriterAssembly.__new__(KernelWriterAssembly)
    kwa.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    kwa.sgprPool.add(0, 16, "unit")
    kwa.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    kwa.vgprPool.add(0, 8, "unit")
    kwa.vgprPool.checkOut(1, "Serial")
    kwa.labels = LabelManager()
    kwa.db = {"AssertOnSgprOverflow": True}
    kwa.states = SimpleNamespace(probeKernArgOffset=offset, laneSGPRCount=2,
                                 regCaps={"MaxSgpr": 102})
    return kwa


def test_occupancy_probe_absent_without_offset():
    module = _probe_writer(-1).occupancyProbe({"WavefrontSize": 64})
    assert list(module.flatitems()) == []


def test_occupancy_probe_emission():
    ri = rocisa.rocIsa.getInstance()
    ri.init((9, 5, 0), shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++")
    ri.setKernel((9, 5, 0), 64)
    kwa = _probe_writer(PROBE_OFFSET)
    items = list(kwa.occupancyProbe({"WavefrontSize": 64}).flatitems())
    lines = [str(i) for i in items]

    # ProbeAddr, then ProbeEpoch + ProbeGrid, both from KernArgAddress.
    loads = [i for i in items if isinstance(i, SLoadB64)]
    assert len(loads) == 2
    for load, offset in zip(loads, (PROBE_OFFSET, PROBE_OFFSET + 8)):
        params = [str(p) for p in load.getParams()]
        assert "sgprKernArgAddress" in params[1]
        assert int(params[2], 0) == offset

    # A plain store; every branch goes to the one skip label.
    store = next(line for i, line in zip(items, lines) if isinstance(i, GlobalStoreB32))
    assert store.startswith("global_store_dword ")
    assert " sc0" not in store and " sc1" not in store and " nt" not in store
    labelName = next(i for i in items if isinstance(i, Label)).getLabelName()
    assert all(labelName in line for i, line in zip(items, lines)
               if isinstance(i, (SCBranchSCC0, SCBranchSCC1)))

    # Exec is narrowed by v_cmpx writing exec and reset to all ones after the
    # store, with one s_nop between the last v_cmpx and the store; VCC is untouched.
    cmpx = [l for l in lines if l.startswith("v_cmp")]
    assert len(cmpx) == 2 and all(l.startswith("v_cmpx_") and l.split()[1] == "exec," for l in cmpx)
    storeIdx = lines.index(store)
    movs = [(n, l.split("//")[0].strip()) for n, l in enumerate(lines) if l.startswith("s_mov_b64 ")]
    assert [m for _, m in movs] == ["s_mov_b64 exec, -1"]
    assert movs[0][0] > storeIdx
    assert not any("vcc" in l.split("//")[0] for l in lines)
    assert sum(isinstance(i, SNop) for i in items[lines.index(cmpx[-1]):storeIdx]) == 1

    # Temps are returned to the pools.
    assert kwa.sgprPool.available() == 16
    assert kwa.vgprPool.available() == 7


def _kernarg_metadata(src):
    metadata = yaml.safe_load(src.split(".amdgpu_metadata", 1)[1].split(".end_amdgpu_metadata", 1)[0])
    kernel = metadata["amdhsa.kernels"][0]
    return {arg[".name"]: arg[".offset"] for arg in kernel[".args"]}, kernel[".kernarg_segment_size"]


@pytest.mark.parametrize("numU32, padded", [(53, True), (54, False)])
def test_occupancy_probe_signature_layout(numU32, padded):
    commonArgsSize = 16
    writer = SimpleNamespace(states=SimpleNamespace(probeKernArgOffset=-1))
    sig = SignatureBase(kernelName="probe_layout", kernArgsVersion=3, codeObjectVersion="5",
                        groupSegmentSize=0, sgprWorkGroup=[1, 1, 1], vgprWorkItem=0,
                        flatWorkGroupSize=256)
    for n in range(numU32):
        sig.addArg("arg%u" % n, SVK.SIG_VALUE, "u32")
    assert (sig.offset % 8 == 4) is padded

    addOccupancyProbeArgs(writer, sig, commonArgsSize)
    args, _ = _kernarg_metadata(str(sig))
    addr = writer.states.probeKernArgOffset + commonArgsSize
    assert ("ProbePad" in args) is padded
    assert addr % 8 == 0
    assert args["ProbeAddr"] == addr
    assert args["ProbeEpoch"] == addr + 8
    assert args["ProbeGrid"] == addr + 12
    assert sig.offset == addr + 16


def test_occupancy_probe_kernarg_matches_loads(monkeypatch):
    """End-to-end SK5 gfx950 emit: metadata offsets match the probe s_loads."""
    from codegen_harness import emit_kernels_from_logic

    monkeypatch.setitem(globalParameters, "EmitOccupancyProbe", True)
    logic = os.path.join(os.path.dirname(__file__), "characterization", "_codegen", "data",
                         "bigfiles", "equality_gfx950_HSS_big.yaml")
    (_, src, err), = emit_kernels_from_logic(logic, limit=1)
    assert err == 0
    args, size = _kernarg_metadata(src)
    assert "ProbeAddr" in args, "logic file no longer yields an SK5 kernel; repoint the test"

    # KernArgAddress is shifted past the 16-byte common args when the probe loads.
    loads = {}
    for line in src.splitlines():
        if "// load ProbeAddr" in line or "// load ProbeEpoch, ProbeGrid" in line:
            ops = line.split("//")[0].split(", ")
            assert ops[1].strip() == "s[sgprKernArgAddress:sgprKernArgAddress+1]"
            loads[line.split("// load ")[1].strip()] = int(ops[2], 0) + 16
    assert loads == {"ProbeAddr": args["ProbeAddr"], "ProbeEpoch, ProbeGrid": args["ProbeEpoch"]}
    assert size == (args["ProbeGrid"] + 4 + 7) // 8 * 8

    # The raw WG id is snapshotted before the wgmXCC remap and read by the
    # probe exit; the probe store is the only store, atomic or LDS access
    # between kernel entry and the exit.
    body = src[src.index(".text") if ".text" in src else 0:]
    snap = body.index("snapshot raw pre-remap launch WG id")
    remap = body.index("remap workgroup to XCCs")
    exit_ = body.index("probe-only WG: exit before any other side effect")
    assert snap < remap < exit_
    prologue = [l.split()[0] for l in body[:exit_].splitlines() if l.strip()]
    stores = [op for op in prologue
              if op.startswith(("global_store", "global_atomic", "buffer_store", "buffer_atomic",
                                "flat_", "scratch_", "s_store", "s_atomic", "ds_"))]
    assert stores == ["global_store_dword"]
