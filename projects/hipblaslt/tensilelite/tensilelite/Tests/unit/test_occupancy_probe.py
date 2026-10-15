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
from rocisa.instruction import GlobalStoreB32, SCBranchSCC0, SCBranchSCC1, SLoadB32, SLoadB64, SNop
from rocisa.label import LabelManager
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
        (True, {"isa": (9, 4, 2)}, False),
        (True, {"streamk": 3}, False),
        (True, {"streamk": 4}, False),
        (True, {"grouped": True}, False),
        (True, {"isa": (12, 5, 0)}, False),
    ],
)
def test_support_occupancy_probe_gate(emit_probe_flag, flag, kwargs, expected):
    emit_probe_flag(flag)
    assert _supportOccupancyProbe(_state(**kwargs)) is expected


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


def _emit_probe():
    """Emits the probe for a gfx950 wave64 kernel; returns (writer, items, lines)."""
    with preserve_rocisa_kernel_state():
        ri = rocisa.rocIsa.getInstance()
        ri.init((9, 5, 0), shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++")
        ri.setKernel((9, 5, 0), 64)
        kwa = _probe_writer(200)
        items = list(kwa.occupancyProbe({"WavefrontSize": 64}).flatitems())
        return kwa, items, [str(i) for i in items]


def test_occupancy_probe_absent_without_offset():
    module = _probe_writer(-1).occupancyProbe({"WavefrontSize": 64})
    assert list(module.flatitems()) == []


def test_occupancy_probe_emission():
    kwa, items, lines = _emit_probe()

    loads = [line for i, line in zip(items, lines) if isinstance(i, (SLoadB64, SLoadB32))]
    assert len(loads) == 2 and all(l.startswith("s_load_dwordx2 ") for l in loads)
    assert "0xc8" in loads[0] and "0xd0" in loads[1]

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

    # Four SGPRs at most; temps are returned to the pools.
    sgprs = set()
    for line in lines:
        for lo, hi in re.findall(r"\bs\[(\d+):(\d+)\]", line):
            sgprs.update(range(int(lo), int(hi) + 1))
        sgprs.update(int(n) for n in re.findall(r"\bs(\d+)\b", line))
    assert len(sgprs) <= 4
    assert kwa.sgprPool.available() == 16
    assert kwa.vgprPool.available() == 7
    assert len(set(re.findall(r"\bv(\d+)\b", "\n".join(lines)))) <= 2


def _run_probe(lines, hwId, xccId, serial, rawId, grid, addr=0x1000):
    """Interprets the emitted probe for one lane; returns (store offset or None, exits)."""
    s = {"sgprStreamKTileIdx": rawId}
    v = {"vgprSerial": serial}
    exec_, scc, stored = 1, None, None
    hwregs = {"HW_REG_HW_ID": hwId, "HW_REG_XCC_ID": xccId}

    def val(op):
        op = op.rstrip(",")
        m = re.fullmatch(r"s\[(\w+)(?::\d+)?\]|s(\d+)", op)
        if m:
            return s[m.group(1) or m.group(2)]
        m = re.fullmatch(r"v\[(\w+)\]|v(\d+)", op)
        if m:
            return v[m.group(1) or m.group(2)]
        return int(op, 0)

    def reg(op):
        return re.fullmatch(r"[sv]\[?(\w+?)(?::\d+)?\]?,?", op).group(1)

    for line in lines:
        if not line.strip() or line.lstrip().startswith("/*"):
            continue
        ins, *ops = line.split("//")[0].replace(", ", ",").split()
        ops = re.split(r",(?![^(]*\))", ops[0]) if ops else []
        if ins in ("s_load_dwordx2",):
            r = int(reg(ops[0]))
            if "0xc8" in line:
                s[str(r)], s[str(r + 1)] = addr, 0
            else:
                s[str(r)], s[str(r + 1)] = 77, grid
        elif ins == "s_cmp_eq_u64":
            scc = s[reg(ops[0])] == 0 and ops[1] == "0"
        elif ins == "v_mov_b32":
            v[reg(ops[0])] = val(ops[1])
        elif ins == "s_getreg_b32":
            name, off, size = re.match(r"hwreg\((\w+),(\d+),(\d+)\)", ops[1]).groups()
            s[reg(ops[0])] = hwregs[name] >> int(off) & ((1 << int(size)) - 1)
        elif ins == "v_lshlrev_b32":
            v[reg(ops[0])] = val(ops[2]) << val(ops[1]) & 0xffffffff
        elif ins == "s_lshl_b32":
            s[reg(ops[0])] = val(ops[1]) << val(ops[2]) & 0xffffffff
            scc = s[reg(ops[0])] != 0
        elif ins == "s_and_b32":
            s[reg(ops[0])] = val(ops[1]) & val(ops[2])
            scc = s[reg(ops[0])] != 0
        elif ins == "s_cselect_b32":
            s[reg(ops[0])] = val(ops[1]) if scc else val(ops[2])
        elif ins == "v_bfi_b32":
            m = val(ops[1])
            v[reg(ops[0])] = (m & val(ops[2])) | (~m & val(ops[3]) & 0xffffffff)
        elif ins == "v_lshl_or_b32":
            v[reg(ops[0])] = (val(ops[1]) << val(ops[2]) | val(ops[3])) & 0xffffffff
        elif ins == "s_mov_b64" and ops == ["exec", "-1"]:
            exec_ = 1
        elif ins == "v_cmpx_eq_u32":
            exec_ &= val(ops[1]) == val(ops[2])
        elif ins == "v_cmpx_gt_u32":
            exec_ &= val(ops[1]) > val(ops[2])
        elif ins == "global_store_dword":
            if exec_:
                assert s[reg(ops[2])] == addr and val(ops[1]) == 77
                stored = val(ops[0])
        elif ins == "s_cmp_ge_u32":
            scc = val(ops[0]) >= val(ops[1])
        elif ins == "s_cbranch_scc0":
            if not scc:
                assert exec_ == 1
                return stored, False
        elif ins == "s_cbranch_scc1":
            if scc:
                assert exec_ == 1
                return stored, False
        elif ins == "s_endpgm":
            return stored, True
        else:
            assert ins in ("s_waitcnt", "s_nop") or ins.endswith(":"), line
    return stored, False


def test_occupancy_probe_slot_map():
    """Slot = XCC << 6 | SE << 4 | CU for SH == 0, SE < 4, XCC < 8; others skip the store.

    Only thread 0 stores; WGs with raw id >= ProbeGrid then exit; a null
    ProbeAddr skips both.
    """
    _, _, lines = _emit_probe()

    slots = {}
    for xcc in range(16):
        for hw in range(256):
            cu, sh, se = hw & 15, hw >> 4 & 1, hw >> 5
            # HW_ID[7:0] (wave/SIMD/pipe) and bits above 15 must not leak in.
            hwId = 0xa5 | hw << 8 | 0x5a0000
            for rawId, grid, exits in ((3, 4, False), (4, 4, True), (9, 4, True)):
                stored, exited = _run_probe(lines, hwId, 0x30 | xcc, 0, rawId, grid)
                assert exited is exits
                if sh or se >= 4 or xcc >= 8:
                    assert stored is None
                else:
                    assert stored == 4 * (xcc << 6 | se << 4 | cu)
            assert _run_probe(lines, hwId, xcc, 1, 9, 4) == (None, True)
            assert _run_probe(lines, hwId, xcc, 64, 3, 4) == (None, False)
            assert _run_probe(lines, hwId, xcc, 0, 9, 4, addr=0) == (None, False)
            if stored is not None:
                assert slots.setdefault(stored, (xcc, hw)) == (xcc, hw)
    assert sorted(slots) == [4 * k for k in range(512)]


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
