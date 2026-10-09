# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute the extracted persistent control-flow emitters in small GPU kernels."""

import struct
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.container import ContinuousRegister, sgpr
from rocisa.enum import RegisterType
from rocisa.instruction import SCmpEQU32
from rocisa.register import RegisterPool

from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.Components.TileProcessingStrategy import TileProcessingStrategy
from Tensile.Components.WorkAssignment import XCCMappingOn, _extract_hybrid_mode
from gpu_test_helpers import (
    GPU_MARKS, assemble_kernel, generate_kernel_asm, generate_load_params,
    init_rocisa, run_on_gpu, run_scalar_kernel,
)

pytestmark = [pytest.mark.unit, *GPU_MARKS]


@pytest.fixture(autouse=True)
def _gpu_instruction_set(_isolate_rocisa_state):
    init_rocisa(wavesize=64)


@pytest.mark.parametrize("k,tail,max_unit,gsu,runtime_gsu,expected", [
    (0, False, 1, 0, 1, 0),
    (16, False, 1, 0, 1, 1),
    (31, False, 1, 0, 1, 1),
    (0, True, 1, 0, 1, 0),
    (16, True, 1, 0, 1, 1),
    (17, True, 1, 0, 1, 2),
    (20, True, 4, 0, 1, 2),
    (18, True, 4, 0, 1, 1),
    (17, True, 1, 2, 2, 1),
    (17, True, 1, 2, 1, 2),
])
def test_ordinary_loop_count_respects_tail_and_gsu(
    tmp_path, k, tail, max_unit, gsu, runtime_gsu, expected,
):
    writer = SimpleNamespace(
        states=SimpleNamespace(tailloopInNll=tail, tailloopInNllmaxUnit=max_unit, unrollIdx=0),
        gsuMaskHex=lambda kernel: "0x3fff",
        sgprs={"LoopCounter": 6, "SizesSum": 8, "GSU": 9},
    )
    module = KernelWriterAssembly._calculateOrdinaryLoopNumIter(
        writer, {"DepthU": 16, "GlobalSplitU": gsu}, "LoopCounter", 0,
        ContinuousRegister(32, 3),
    )
    observed = run_scalar_kernel(
        writer, module, {"SizesSum": k, "GSU": runtime_gsu}, ["LoopCounter"], tmp_path,
    )
    assert observed["LoopCounter"] == (expected,) * 64


@pytest.mark.parametrize("gsu,runtime_gsu,arg_type,keep_address", [
    (0, 1, 0, True),
    (0, 1, 3, False),
    (2, 1, 3, False),
    (2, 2, 3, True),
])
def test_ordinary_srd_routes_general_batch_and_gsu(
    monkeypatch, tmp_path, gsu, runtime_gsu, arg_type, keep_address,
):
    def reject_strategy_lookup(*args, **kwargs):
        pytest.fail("Ordinary writer emission must not request a persistent strategy")

    monkeypatch.setattr(TileProcessingStrategy, "find", reject_strategy_lookup)
    kernel = {
        "TileProcessingStrategy": "None", "GlobalSplitU": gsu,
        "GlobalSplitUAlgorithm": "MultipleBuffer", "_GlobalAccumulation": "MultipleBuffer",
        "ProblemType": {"SupportUserArgs": True},
    }

    @contextmanager
    def alloc_tmp(*args, **kwargs):
        yield ContinuousRegister(32, 1)

    writer = SimpleNamespace(
        states=SimpleNamespace(kernel=kernel), allocTmpSgpr=alloc_tmp,
        shiftSrd=lambda ch: Module("gfx950 requires no SRD shift"),
        cmpNamedArgTypeEq=lambda module, value, comment: module.add(
            SCmpEQU32(src0=sgpr("ArgType"), src1=value, comment=comment)),
        sgprs={"GSU": 6, "ArgType": 7, "AddressD": 8, "SrdD": 12},
    )
    module = ".set BufferOOB, 0xffffffff\n.set Srd127_96, 0\n"
    module += str(KernelWriterAssembly.allocPostLoopSrd(writer, "D", kernel))
    observed = run_scalar_kernel(
        writer, module,
        {"GSU": runtime_gsu, "ArgType": arg_type,
         "AddressD+0": 0x9abcdef0, "AddressD+1": 0x12345678},
        ["SrdD+0", "SrdD+1"], tmp_path,
    )
    # Verify both halves so a truncated or incorrectly copied pointer fails.
    assert observed["SrdD+0"] == (0x9abcdef0 if keep_address else 0,) * 64
    assert observed["SrdD+1"] == (0x12345678 if keep_address else 0,) * 64


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("uso", [0, 1])
def test_hybrid_mode_extraction_preserves_uniform_summation_order(tmp_path, mode, uso):
    writer = SimpleNamespace(sgprs={"MagicShiftItersPerTile": 6, "WorkAssignmentMode": 7})
    packed = (mode << 30) | (uso << 29) | 17
    observed = run_scalar_kernel(
        writer, _extract_hybrid_mode(), {"MagicShiftItersPerTile": packed},
        ["WorkAssignmentMode", "MagicShiftItersPerTile"], tmp_path,
    )
    assert observed["WorkAssignmentMode"] == (mode,) * 64
    assert observed["MagicShiftItersPerTile"] == ((uso << 29) | 17,) * 64


class _MappingWriter:
    def __init__(self, xcc):
        self.states = SimpleNamespace(kernel={
            "TileProcessingStrategy": "StreamK", "WorkAssignment": "Hybrid",
            "PersistentXCCMapping": xcc,
        })
        self.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
        self.sgprPool.add(0, 64, "mapping registers")
        self.sgprPool.checkOut(6, "launch arguments and output pointer")
        self.sgprs = {name: self.sgprPool.checkOut(1, name) for name in (
            "ItersPerTile", "MagicNumberItersPerTile", "MagicShiftItersPerTile",
            "SKItersPerWG", "skGrid", "skTiles", "WorkGroup0",
            "ModeWordAfterMapping", "WorkAssignmentMode", "TraceOffset",
        )}
        # The actual Hybrid ABI overlays dynamic SKGrid on the static skTiles slot.
        self.sgprs["SKGrid"] = self.sgprs["skTiles"]

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        index = self.sgprPool.checkOutAligned(size, alignment or 1, tag)
        try:
            yield ContinuousRegister(index, size)
        finally:
            self.sgprPool.checkIn(index)

    def isPersistentConstantsToVgprEnabled(self, kernel):
        return False

    def acquirePersistentConstSgpr(self, kernel, name):
        return name

    def releasePersistentConstSgpr(self, name):
        assert isinstance(name, str), "Mapping scratch belongs to allocTmpSgpr"


@pytest.mark.parametrize("mode,inactive_slot", [(0, 95), (0, 4), (1, 3)],
                         ids=["static_tree", "static_parallel", "dynamic"])
@pytest.mark.parametrize("grid,xcc", [(1, 8), (64, 8), (65, 8), (63, 3), (64, 3)])
@pytest.mark.parametrize("other_bits", [0, 0xa0000000], ids=["plain", "uso_and_magic_add"])
def test_hybrid_xcc_mapping_uses_active_grid_slot(tmp_path, mode, inactive_slot, grid, xcc, other_bits):
    writer = _MappingWriter(xcc)
    packed_mode = other_bits | (mode << 30) | 17
    inputs = {
        "ItersPerTile": 8, "MagicNumberItersPerTile": 0x10000000,
        "MagicShiftItersPerTile": packed_mode, "SKItersPerWG": 3,
        "skGrid": grid if mode == 0 else inactive_slot,
        "skTiles": inactive_slot if mode == 0 else grid,
    }
    outputs = ("WorkGroup0", "ModeWordAfterMapping", "WorkAssignmentMode",
               "MagicShiftItersPerTile", "skGrid", "skTiles")
    record_bytes = len(outputs) * 64 * 4
    args = [("output", 8, "global_buffer", "u32")]
    loads = [(4, 2, 0, "output pointer")]
    for i, name in enumerate(inputs):
        args.append((name, 4, "by_value", "u32"))
        loads.append((name, 1, 8 + 4 * i, name))
    body = [str(generate_load_params(loads)),
            "s_mov_b32 s[sgprWorkGroup0], s2",
            str(XCCMappingOn()(writer, writer.states.kernel)),
            "s_mov_b32 s[sgprModeWordAfterMapping], s[sgprMagicShiftItersPerTile]",
            str(_extract_hybrid_mode()),
            f"s_mul_i32 s[sgprTraceOffset], s2, {record_bytes}"]
    for i, name in enumerate(outputs):
        body.extend([
            "v_lshlrev_b32 v1, 2, v0",
            "v_add_u32 v1, s[sgprTraceOffset], v1",
            f"v_add_u32 v1, {i * 64 * 4}, v1",
            f"v_mov_b32 v2, s[sgpr{name}]",
            "global_store_dword v1, v2, s[4:5]",
            "s_waitcnt vmcnt(0)",
        ])
    asm = generate_kernel_asm("\n".join(body), writer, args, num_threads=64)
    (tmp_path / "hybrid_mapping.s").write_text(asm)
    co = str(tmp_path / "hybrid_mapping.co")
    assemble_kernel(asm, co)
    raw = run_on_gpu(co, grid * record_bytes, scalars=tuple(inputs.values()),
                     grid=grid, num_threads=64)
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    mapped = []
    for rank in range(grid):
        group = rank % xcc
        # Concatenate the groups' round-robin rank lists in XCC order.
        expected_rank = group * (grid // xcc) + min(group, grid % xcc) + rank // xcc
        expected = (expected_rank, packed_mode, mode, other_bits | 17,
                    inputs["skGrid"], inputs["skTiles"])
        for i, value in enumerate(expected):
            start = (rank * len(outputs) + i) * 64
            assert words[start:start + 64] == (value,) * 64, (rank, outputs[i])
        mapped.append(words[rank * len(outputs) * 64])
    assert sorted(mapped) == list(range(grid))
