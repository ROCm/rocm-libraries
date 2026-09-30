# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute descriptor setup on the CPU and compare PAP with full initialization.

This models scalar descriptor words, not tensor-load or GPU scheduling behavior.
The reference uses the full initializer; the candidate starts from a descriptor
advanced and disabled by the current tile, with the next tile's identity loaded.
"""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import re
import struct
from types import SimpleNamespace

import pytest

from rocisa.code import Module
from rocisa.container import ContinuousRegister, sgpr
from rocisa.enum import RegisterType
from Tensile.Component import Component
from Tensile.Components.PersistentLoop import PersistentLoopOn
from Tensile.Common import INDEX_CHARS
from Tensile.Common.DataType import DataType
from rocisa.register import RegisterPool
from Tensile.Components.ClusterLoad import ClusterLoadTDM
from Tensile.Components.TensorDataMover import TensorDataMoverLoad
from Tensile.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


class _Writer(KernelWriterAssembly):
    def __init__(self, kernel):
        self.states = SimpleNamespace(kernel=kernel, laneSGPRCount=1, unrollIdx=0,
                                      asmCaps={"HasSMulHi": True},
                                      indexChars=INDEX_CHARS, IncLdsBufSwitch=False,
                                      staggerUCode=False, numSgprSizesSum=1,
                                      batchOffsetAKernArgOffset=8, batchOffsetBKernArgOffset=16)
        self.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False)
        self._tmp = 200
        self.max_tmp = self._tmp
        self._label = 0
        self.labels = SimpleNamespace(getNameInc=self._next_label)

    def _next_label(self, name):
        self._label += 1
        return f"{name}_{self._label}"

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        previous = self._tmp
        base = (previous + alignment - 1) // alignment * alignment
        self._tmp = base + size
        self.max_tmp = max(self.max_tmp, self._tmp)
        try:
            yield ContinuousRegister(base, size)
        finally:
            self._tmp = previous

    def strideRef(self, tc, dim):
        return sgpr(f"Stride{tc}{INDEX_CHARS[dim]}")


class _DescriptorMachine:
    """Small strict interpreter with physical aliases and 32/64-bit operands."""
    def __init__(self):
        self.names = {"tdmAGroup0": 0, "tdmBGroup0": 0,
                      "tdmAGroup1": 4, "tdmBGroup1": 4,
                      "tdmMXSAGroup0": 16, "tdmMXSBGroup0": 16,
                      "tdmMXSAGroup1": 20, "tdmMXSBGroup1": 20,
                      "tdmAGroup2": 32, "tdmBGroup2": 32,
                      "tdmAGroup3": 32, "tdmBGroup3": 32}
        self.words = {}
        self.memory = {}
        self.scc = False
        self.vcc = False
        self.writes = []

    def index(self, expression):
        name, _, offset = expression.removeprefix("sgpr").removeprefix("vgpr").partition("+")
        if name.isdecimal():
            base = int(name)
        else:
            if name not in self.names:
                self.names[name] = 600 + len(self.names) * 8
            base = self.names[name]
        return base + sum(int(term) for term in (offset or "0").split("+"))

    def registers(self, operand):
        match = re.fullmatch(r"[sv](?:\[([^]]+)\]|(\d+))", operand)
        assert match, operand
        first, _, last = (match[1] or match[2]).partition(":")
        base = self.index(first)
        return range(base, self.index(last) + 1 if last else base + 1)

    def value(self, operand):
        if re.fullmatch(r"[sv](?:\[[^]]+\]|\d+)", operand):
            return sum(self.words.get(index, 0) << (32 * i)
                       for i, index in enumerate(self.registers(operand)))
        return 0 if operand == "0.0" else int(operand, 0)

    def put(self, operand, value):
        for i, index in enumerate(self.registers(operand)):
            self.words[index] = (value >> (32 * i)) & 0xffffffff
            self.writes.append(index)

    def set(self, name, value, size=1):
        self.put(str(sgpr(name, size)), value)

    def get(self, name, size=1):
        return self.value(str(sgpr(name, size)))

    def run(self, module):
        lines = [line.split("//", 1)[0].strip() for line in str(module).splitlines()]
        lines = [line for line in lines if line and not line.startswith(("/*", "*"))]
        labels = {line[:-1]: index for index, line in enumerate(lines) if line.endswith(":")}
        pc = 0
        while pc < len(lines):
            line = lines[pc]
            pc += 1
            if line.endswith(":"):
                continue
            op, _, operands = line.partition(" ")
            args = [arg.strip() for arg in operands.split(",")]
            if op in ("s_branch", "s_cbranch_scc0", "s_cbranch_scc1"):
                if op == "s_branch" or self.scc == (op == "s_cbranch_scc1"):
                    pc = labels[args[0]]
                continue
            if op in ("s_cbranch_vccz", "s_cbranch_vccnz"):
                if self.vcc == (op == "s_cbranch_vccnz"):
                    pc = labels[args[0]]
                continue
            if op == "v_cmp_eq_f32":
                values = [struct.unpack("f", struct.pack("I", self.value(arg)))[0]
                          for arg in args[1:]]
                self.vcc = values[0] == values[1]
                continue
            if op == "s_nop" or op.startswith("s_wait"):
                continue
            if op in ("s_load_b64", "s_load_dwordx2"):
                address = self.value(args[1]) + int(args[2].split()[0], 0)
                self.put(args[0], self.memory[address])
                continue
            if op.startswith("s_cmp_"):
                left, right = map(self.value, args)
                predicate = op.split("_")[2]
                self.scc = {"eq": left == right, "ge": left >= right,
                            "gt": left > right, "lt": left < right,
                            "le": left <= right}[predicate]
                continue
            if op == "s_bitcmp1_b32":
                self.scc = bool(self.value(args[0]) & (1 << self.value(args[1])))
                continue
            src = [self.value(arg) for arg in args[1:]]
            if op in ("s_mov_b32", "s_mov_b64", "v_readfirstlane_b32"):
                result = src[0]
            elif op == "s_cmov_b32":
                if not self.scc:
                    continue
                result = src[0]
            elif op == "s_cselect_b32":
                result = src[0] if self.scc else src[1]
            elif op in ("s_add_u32", "s_addc_u32"):
                result = sum(src) + (int(self.scc) if op == "s_addc_u32" else 0)
                self.scc = result > 0xffffffff
            elif op in ("s_sub_u32", "s_sub_i32", "s_add_i32"):
                result = src[0] + src[1] if op == "s_add_i32" else src[0] - src[1]
                self.scc = result < 0 if op == "s_sub_u32" else False
            elif op in ("s_mul_i32", "s_mul_hi_u32"):
                result = src[0] * src[1]
                if op == "s_mul_hi_u32":
                    result >>= 32
            elif op in ("s_and_b32", "s_or_b32", "s_xor_b32"):
                result = (src[0] & src[1] if op == "s_and_b32" else
                          src[0] | src[1] if op == "s_or_b32" else src[0] ^ src[1])
                self.scc = bool(result)
            elif op in ("s_lshr_b32", "s_lshl_b32", "s_lshr_b64", "s_lshl_b64"):
                shift = src[1] & (63 if op.endswith("64") else 31)
                result = src[0] >> shift if "lshr" in op else src[0] << shift
            elif op in ("s_min_u32", "s_max_u32"):
                result = min(src) if op == "s_min_u32" else max(src)
            else:
                raise AssertionError("Uninterpreted descriptor instruction: " + line)
            self.put(args[0], result)

    def descriptor(self, scales):
        indices = list(range(12)) + list(range(32, 36))
        if scales:
            indices += list(range(16, 28))
        return {index: self.words.get(index, 0) for index in indices}


def _kernel(dtype="H", *, iterate=False, depth=256, scales=False, tlu=False,
            cluster=(1, 1), waves=4):
    pt = {"DataTypeA": DataType(dtype), "DataTypeB": DataType(dtype),
          "DataTypeMXSA": DataType("E8"), "DataTypeMXSB": DataType("E8"),
          "MXBlockA": 32 if scales else 0, "MXBlockB": 32 if scales else 0,
          "Sparse": 0, "SupportUserArgs": False, "Batched": True,
          "StridedBatched": True, "GroupedGemm": False, "IndicesSummation": [3],
          "ComputeDataType": DataType("S")}
    kernel = {"TileProcessingStrategy": "DataParallel", "WorkAssignment": "StaticGrid",
              "TDMInst": 3, "TDMFuse": 0, "enableTDMA": True, "enableTDMB": True,
              "NumWaves": waves, "WavefrontSize": 32, "UseSubtileImpl": False,
              "TDMSplit": False, "TDMPlusLdsBuf": 0, "GlobalSplitU": 0,
              "DepthU": depth, "MacroTile0": 256, "MacroTile1": 128,
              "MatrixInstK": 128, "ClusterDim": list(cluster),
              "Multicast": cluster != (1, 1), "ProblemType": pt,
              "LdsOffsetA_Blk": 0x18000, "1LDSBuffer": 0, "StoreSwapAddr": True,
              "StaggerU": 0, "ReuseAcrossPersistent": 0,
              "PrefetchGlobalRead": 2, "PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 2,
              "_TDMIterateModeA": iterate, "_TDMIterateModeB": iterate,
              "LDSSegmentInterleave": 0}
    if isinstance(iterate, tuple):
        kernel["_TDMIterateModeA"], kernel["_TDMIterateModeB"] = iterate
    if cluster != (1, 1):
        # Include relocated MX bases and a padded interleaved data layout.
        kernel["LDSSegmentInterleave"] = 1
        kernel["LDSSegInterleaveOffsets"] = {"writeStrideBytes": 0x5000,
            "footprintPacked": True, "ldsBaseA": 0, "ldsBaseB": 0x1000,
            "ldsBaseMXSA": 0x10000, "ldsBaseMXSB": 0x12000}
    tensors = {}
    for tc, idx in (("A", 0), ("B", 1), ("MXSA", 0), ("MXSB", 1)):
        is_scale = tc.startswith("MX")
        kernel[f"LdsOffset{tc}"] = 0x10000 if is_scale else 0
        kernel[f"LdsBlockSizePerPad{tc}"] = 256 if is_scale else 2048
        kernel[f"LdsPad{tc}"] = 16
        tensors[tc] = {"tensorChar": tc, "idx": idx, "tileChar": INDEX_CHARS[idx],
                       "bpeGR": pt[f"DataType{tc}"].numBytes(), "tlu": tlu,
                       "isA": tc == "A", "isB": tc == "B", "isM": False,
                       "ia": [3, idx, 2], "is_sparse": False}
    return kernel, tensors


def _machine(kernel, tensors, wave):
    machine = _DescriptorMachine()
    machine.set("Serial", wave * kernel["WavefrontSize"])
    machine.set("WaveIdx", wave)
    for name, value in {"WorkGroup0": 0, "WorkGroup1": 0, "WorkGroup2": 0,
                        "SizeI": 520, "SizeJ": 264, "SizeL": 1568,
                        "SizesSum": 1568, "Alpha": 0x3f800000,
                        "MulticastMaskA": 0x55, "MulticastMaskB": 0x0f,
                        "MulticastMask": 0x55 if wave % 2 == 0 else 0x0f}.items():
        machine.set(name, value)
    for i, (tc, tp) in enumerate(tensors.items()):
        machine.set(f"Address{tc}", 0x1234fffffe00 + (i << 36), 2)
        free = machine.get("SizeI" if tp["idx"] == 0 else "SizeJ")
        machine.set(f"Stride{tc}{INDEX_CHARS[tp['idx']]}", 1 if tp["tlu"] else 1568)
        machine.set(f"Stride{tc}L", free if tp["tlu"] else 1)
        machine.set(f"Stride{tc}K", free * 1568)
    return machine


def _full(writer, kernel, tensors):
    module = Module("reference full tile descriptors")
    pairs = [("A", "B")]
    if kernel["ProblemType"]["MXBlockA"]:
        pairs.append(("MXSA", "MXSB"))
    for a, b in pairs:
        module.add(writer.initTDMDescriptorWaveSeparated(kernel, tensors[a], tensors[b]))
        module.add(writer.tdmGlobalOffsetWaveSeparated(kernel, tensors[a], tensors[b]))
    return module


@pytest.fixture(autouse=True)
def _descriptor_components(monkeypatch):
    monkeypatch.setattr(TensorDataMoverLoad, "find", lambda writer: TensorDataMoverLoad())
    monkeypatch.setattr(ClusterLoadTDM, "find", lambda writer: ClusterLoadTDM())


@pytest.mark.parametrize("dtype,iterate,depth,scales,tlu", [
    ("H", False, 256, False, False), ("H", False, 256, False, True),
    ("H", True, 256, False, False), ("F8", False, 256, True, False),
    ("F8", True, 256, True, False), ("F4", True, 256, True, False),
    ("F8", False, 128, True, False), ("F4", True, 128, True, False),
    ("F8", (True, False), 256, True, False),
    ("F8", (False, True), 256, True, False),
    ("F6", False, 256, False, False),
])
@pytest.mark.parametrize("cluster", [(1, 1), (2, 4)])
@pytest.mark.parametrize("bank", [0, 1])
def test_incremental_matches_full_after_current_tile(dtype, iterate, depth, scales, tlu, cluster, bank):
    kernel, tensors = _kernel(dtype, iterate=iterate, depth=depth, scales=scales,
                              tlu=tlu, cluster=cluster)
    writer = _Writer(kernel)
    reference_code = _full(writer, kernel, tensors)
    incremental = Module("PAP incremental refresh")
    pairs = [("A", "B")] + ([("MXSA", "MXSB")] if scales else [])
    for a, b in pairs:
        incremental.add(writer.papTdmUpdateDescriptor(kernel, tensors[a], tensors[b]))
    assert "PAP: re-enable TDM for the next tile" in str(incremental)
    assert "set tile0" not in str(incremental)
    assert "set lds increment" not in str(incremental)
    # No second descriptor or other persistent register allocation.
    assert writer.max_tmp <= 207

    for wave in range(kernel["NumWaves"]):
        current = _machine(kernel, tensors, wave)
        current.run(reference_code)
        for a, b in pairs:
            tp = tensors[b if wave % 2 else a]
            inc = int(depth * tp["bpeGR"])
            if tp["tlu"]:
                inc *= current.get(f"Stride{tp['tensorChar']}L")
            if a == "MXSA":
                inc = depth // 32 * current.get("SizeJ" if wave % 2 else "SizeI")
            current.set(f"tdm{a}{b}Incs", inc)
            advance = writer.tdmIncrementABWaveSperated(kernel, tensors[a], tensors[b])
            for _ in range(6):
                current.run(advance)
            if bank:
                current.run(writer.tdmSwapLdsOffset(kernel, tensors[a]))
            current.set(f"tdm{a}Group0", 0)  # HalfPLR's final-trip NULL descriptor.

        for wg0, wg1, batch in ((1, 1, 0), (2, 2, 0), (0, 0, 1), (2, 2, 3)):
            candidate = deepcopy(current)
            expected = _machine(kernel, tensors, wave)
            for machine in (candidate, expected):
                machine.set("WorkGroup0", wg0)
                machine.set("WorkGroup1", wg1)
                machine.set("WorkGroup2", batch)
            expected.run(reference_code)
            for a, b in pairs:
                if bank:
                    expected.run(writer.tdmSwapLdsOffset(kernel, tensors[a]))
            invariant_before = {index: candidate.words.get(index, 0)
                                for index in (4, 8, 9, 10, 11, 32, 33, 34,
                                              20, 24, 25, 26, 27)}
            candidate.writes.clear()
            candidate.run(incremental)
            assert candidate.descriptor(scales) == expected.descriptor(scales), (wave, wg0, wg1, batch)
            assert all(candidate.words.get(index, 0) == value
                       for index, value in invariant_before.items())
            assert not set(invariant_before).intersection(candidate.writes)
            assert candidate.get("tdmAGroup0") == 1
            assert candidate.get("tdmAGroup0+1") == current.get("tdmAGroup0+1")
            # Empty high-component edge waves must not retain their old row count.
            if not tlu and wg0 == 2 and wave == 2:
                dim1 = ((candidate.get("tdmAGroup1+2") >> 16)
                        | ((candidate.get("tdmAGroup1+3") & 0xffff) << 16))
                assert dim1 == 0
                if kernel["_TDMIterateModeA"]:
                    assert candidate.get("tdmAGroup2+3") == 0
            # The next case includes an edge -> interior transition, so stale
            # zero/clipped dimensions and iterate counts must grow again.
            current = candidate


@pytest.mark.parametrize("arg_type", [0, 3])
def test_incremental_reresolves_general_batched_base(arg_type):
    kernel, tensors = _kernel(iterate=True)
    kernel["ProblemType"]["SupportUserArgs"] = True
    writer = _Writer(kernel)
    full = _full(writer, kernel, tensors)
    incremental = writer.papTdmUpdateDescriptor(kernel, tensors["A"], tensors["B"])
    for wave in range(4):
        current = _machine(kernel, tensors, wave)
        current.set("ArgType", (wave << 8) | arg_type)
        current.set("KernArgAddress", 0x100000000, 2)
        for tc, offset in (("A", 8), ("B", 16)):
            array = current.get(f"Address{tc}", 2)
            current.memory[array] = 0x2345fffffff0
            current.memory[array + 8] = 0x3456ffffffc0
            current.memory[0x100000000 + offset] = 0x80
        current.run(full)
        expected = deepcopy(current)
        for machine in (current, expected):
            machine.set("WorkGroup0", 1)
            machine.set("WorkGroup1", 1)
            machine.set("WorkGroup2", 1)
        current.run(incremental)
        expected.run(full)
        assert current.descriptor(False) == expected.descriptor(False)
        assert current.get("tdmAGroup0+3") >> 30 == 2


@pytest.mark.parametrize("overrides,preserve", [
    ({"TileProcessingStrategy": "StreamK"}, True),
    ({"TDMFuse": 1}, True), ({"TDMFuse": 2}, True),
    ({"UseSubtileImpl": True}, True), ({"TDMSplit": True}, True),
    ({"TDMPlusLdsBuf": 1}, True), ({"StaggerU": 1}, True),
    ({"NumWaves": 1}, True), ({"ProblemType": {"Sparse": 1}}, True), ({}, False),
])
def test_unsupported_refresh_keeps_full_initializer(overrides, preserve):
    kernel, tensors = _kernel(scales=True)
    kernel.update(overrides)
    writer = _Writer(kernel)
    calls = []
    writer.initTDMDescriptorWaveSeparated = lambda *args, **kwargs: calls.append(kwargs) or Module()
    writer.tdmGlobalOffsetWaveSeparated = lambda *args: Module()
    writer.tdmApplyTileKOffsetWaveSeparated = lambda *args: Module()
    writer.papTdmUpdateDescriptor(kernel, tensors["A"], tensors["B"], preservePapBank=preserve)
    assert calls == [{}]


@pytest.mark.parametrize("no_tail,half_plr,expected_refreshes", [
    (False, False, [True, False]), (True, False, [True]), (False, True, [True]),
])
def test_current_descriptor_restore_stays_omitted_without_tail(
        monkeypatch, no_tail, half_plr, expected_refreshes):
    kernel, tensors = _kernel()
    kernel.update(NoTailLoop=no_tail, HalfPLR=half_plr)
    writer = _Writer(kernel)
    writer.states.currentTileWork = SimpleNamespace(k_start=None)
    writer.states.rapInPapNextTilePrefetch = False
    writer.isPrefetchAcrossPersistentEnabled = lambda kernel: True
    writer.allocPapTileIdentity = lambda *args, **kwargs: nullcontext({})
    writer.papCheckpointCurrentTileIdentity = lambda *args, **kwargs: Module()
    writer.papRestoreCurrentTileIdentity = lambda *args, **kwargs: Module()
    writer.loopCounterName = lambda *args: "LoopCounterL"
    writer.calculateLoopNumIter = lambda *args: Module()
    writer.setupPrefetchAcrossPersistentLoads = lambda *args, **kwargs: Module()
    writer.papTdmSaveLdsBank = lambda *args: Module()
    calls = []
    writer.papTdmUpdateDescriptor = lambda *args, preservePapBank=True: (
        calls.append(preservePapBank) or Module())
    processing = SimpleNamespace(prefetchEligibility=lambda *args: Module(),
        prefetchAcrossPersistentSetupNextTile=lambda *args, **kwargs: Module())
    assignment = SimpleNamespace(reserveNext=lambda *args: Module())
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda writer: processing)
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda writer: assignment)
    PersistentLoopOn().prefetch(writer, kernel, tensors["A"], tensors["B"])
    assert calls == expected_refreshes
