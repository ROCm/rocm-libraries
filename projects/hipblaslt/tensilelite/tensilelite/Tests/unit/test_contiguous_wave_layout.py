# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check emitted store coordinates against logical wave/lane ownership."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa import rocIsa
from rocisa.enum import RegisterType
from rocisa.container import ContinuousRegister
from rocisa.register import RegisterPool

from tensilelite.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401 (component registry)
from tensilelite.Components.ComputeStoreVgprs import ComputeStoreVgprsMFMA, ComputeStoreVgprsMFMASwap
from tensilelite.AsmStoreState import StoreState
from tensilelite.Components.NotLocalFullTileElements import NotLocalFullTileElementsMFMA

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("gfx1250_iim")]


@contextmanager
def _temporary_scalar(*args, **kwargs):
    yield ContinuousRegister(90, 1)


def _execute(module, registers):
    registers = dict(registers)

    def value(operand):
        return registers[operand] if operand in registers else int(operand, 0)

    for line in str(module).splitlines():
        code = line.partition("//")[0].strip()
        if not code or code.startswith("/*"):
            continue
        op, destination, *sources = code.replace(",", " ").split()
        if op in ("v_mov_b32", "s_mov_b32"):
            result = value(sources[0])
        else:
            a, b = map(value, sources[:2])
            if op in ("v_add_u32", "v_add_nc_u32"):
                result = a + b
            elif op in ("v_mul_lo_u32", "s_mul_i32"):
                result = a * b
            elif op == "v_and_b32":
                result = a & b
            elif op == "v_lshrrev_b32":
                result = b >> a
            elif op == "v_lshlrev_b32":
                result = b << a
            elif op == "v_add_lshl_u32":
                result = (a + b) << value(sources[2])
            else:
                pytest.fail(f"Unsupported coordinate instruction: {line}")
        registers[destination] = result & 0xFFFFFFFF
    return registers


def _component(wave_tile_a, width_a, source_swap, contiguous, store_width=None, width_b=8, matrix=(16, 16)):
    rocIsa.getInstance().setKernel((12, 5, 0), 32)
    kernel = dict(
        WavefrontSize=32, MIWaveGroup=[2, 2], MIWaveTile=[wave_tile_a, 8],
        MatrixInstM=matrix[0], MatrixInstN=matrix[1], MatrixInstBM=1, MatrixInstBN=1,
        MIOutputVectorWidth=8, VectorWidthA=width_a, VectorWidthB=width_b,
        MacroTile0=2 * matrix[0] * wave_tile_a, MacroTile1=16 * matrix[1],
        BufferStore=False, StoreRemapVectorWidth=0, LocalSplitU=1,
        EnableMatrixInstruction=True, SourceSwap=source_swap, _VectorStore=True,
        StoreVectorWidth=width_a if source_swap else 8 * width_a, GlobalSplitU=1, GlobalSplitUAlgorithm="MultipleBuffer",
        WaveTileContiguousA=contiguous,
    )
    if contiguous:
        kernel["StoreVectorWidth"] = 1 if source_swap else 8
    if store_width is not None:
        kernel["StoreVectorWidth"] = store_width
    writer = SimpleNamespace(
        states=SimpleNamespace(kernel=kernel, subtileRecomputeCoords=False),
        vgprs=SimpleNamespace(), vgprPool=RegisterPool(0, RegisterType.Vgpr, False),
        allocTmpSgpr=_temporary_scalar, maxGwvw=lambda _: 16,
    )
    component = ComputeStoreVgprsMFMASwap if source_swap else ComputeStoreVgprsMFMA
    module = component()(writer, kernel)
    return writer, kernel, module


@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("width_b", [2, 4, 8])
@pytest.mark.parametrize("wave_tile,width", [(7, 1), (6, 1), (6, 2), (8, 4), (10, 1), (10, 2)])
def test_wave_output_coverage(wave_tile, width, width_b, contiguous, source_swap):
    writer, kernel, module = _component(wave_tile, width, source_swap, contiguous, width_b=width_b)
    widths, element_lists = NotLocalFullTileElementsMFMA().getElements(writer, kernel)
    # Include scalar edge stores as well as full vector stores.
    store = SimpleNamespace(kernelWriter=writer, kernel=kernel, firstBatch=True,
                            lastCoordOffset1=0)
    for store_width, elements in zip(widths, element_lists):
        offsets_m, offsets_n = StoreState.getStoreElementsInfoForBatch(store, kernel, elements)
        all_outputs = set()
        for wave in range(4):
            wave_outputs = set()
            for lane in range(32):
                values = _execute(module, {"v[vgprSerial]": wave * 32 + lane,
                                          "s[sgprWorkGroup0]": 0, "s[sgprWorkGroup1]": 0})
                base_m = values[f"v{writer.vgprs.coord0}"]
                base_n = values[f"v{writer.vgprs.coord1}"]
                for offset_m, offset_n in zip(offsets_m, offsets_n):
                    for element in range(store_width):
                        point = (base_m + offset_m + element, base_n + offset_n)
                        assert point not in wave_outputs, (wave, lane, point)
                        wave_outputs.add(point)
            assert len(wave_outputs) == 16 * wave_tile * 128
            if contiguous:
                rows = range((wave % 2) * 16 * wave_tile, ((wave % 2) + 1) * 16 * wave_tile)
            else:
                rows = [row for vector in range(wave_tile // width)
                        for row in range((wave % 2 + 2 * vector) * 16 * width,
                                         (wave % 2 + 2 * vector + 1) * 16 * width)]
            assert {m for m, _ in wave_outputs} == set(rows)
            columns = [column for vector in range(8 // width_b)
                       for column in range((wave // 2 + 2 * vector) * 16 * width_b,
                                           (wave // 2 + 2 * vector + 1) * 16 * width_b)]
            assert {n for _, n in wave_outputs} == set(columns)
            assert all_outputs.isdisjoint(wave_outputs)
            all_outputs.update(wave_outputs)
        assert all_outputs == {(m, n) for m in range(32 * wave_tile) for n in range(256)}


@pytest.mark.parametrize("rectangular", [False, True])
@pytest.mark.parametrize("source_swap,store_width", [
    (False, None), (False, 1), (False, 2), (False, 4), (False, 8),
    (True, None), (True, 1),
])
@pytest.mark.parametrize("wave_tile,width", [(7, 1), (6, 1), (6, 2), (6, 3), (8, 4), (12, 6), (16, 8), (10, 1), (10, 2)])
@pytest.mark.parametrize("width_b", [2, 4, 8])
def test_native_wmma_results_store_without_lane_exchange(wave_tile, width, source_swap, store_width, width_b, rectangular):
    from tensilelite.KernelWriterModules import accToArchMapper

    matrix = ((16, 32) if source_swap else (32, 16)) if rectangular else (16, 16)
    outputs = matrix[0] * matrix[1] // 32
    writer, kernel, coordinates = _component(wave_tile, width, source_swap, True, store_width, width_b, matrix)
    states = []
    for serial in range(128):
        wave, lane = divmod(serial, 32)
        state = {"v[vgprSerial]": serial}
        for j in range(8):
            for i in range(wave_tile):
                for t in range(outputs):
                    # WMMA result coordinates before the store address mapping.
                    output = 16 * (t // 8) + 8 * (lane // 16) + t % 8
                    m, n = (lane % 16, output) if source_swap else (output, lane % 16)
                    point = (wave % 2 * matrix[0] * wave_tile + matrix[0] * i + m,
                             matrix[1] * width_b * (wave // 2 + 2 * (j // width_b)) + width_b * n + j % width_b)
                    state[f"C{outputs * (i + wave_tile * j) + t}"] = point
        states.append(state)
    _, arch2acc = accToArchMapper(kernel)
    widths, element_lists = NotLocalFullTileElementsMFMA().getElements(writer, kernel)
    assert widths[0] == (store_width if store_width is not None else (1 if source_swap else 8))
    store = SimpleNamespace(kernelWriter=writer, kernel=kernel, firstBatch=True, lastCoordOffset1=0)
    for store_width, elements in zip(widths, element_lists):
        offsets_m, offsets_n = StoreState.getStoreElementsInfoForBatch(store, kernel, elements)
        for serial, state in enumerate(states):
            values = _execute(coordinates, {"v[vgprSerial]": serial,
                              "s[sgprWorkGroup0]": 0, "s[sgprWorkGroup1]": 0})
            base_m, base_n = values[f"v{writer.vgprs.coord0}"], values[f"v{writer.vgprs.coord1}"]
            # Actual element enumeration is in increasing architectural-C order.
            for index, (dm, dn) in enumerate(zip(offsets_m, offsets_n)):
                for v in range(store_width):
                    reg = arch2acc[index * store_width + v]
                    assert state[f"C{reg}"] == (base_m + dm + v, base_n + dn)
