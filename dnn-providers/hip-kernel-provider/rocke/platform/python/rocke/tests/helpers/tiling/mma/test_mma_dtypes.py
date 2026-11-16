# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""TileMma with IR-typed 8-bit dtypes, from selection to LLVM IR, and the visualizer's view of an
MMA's dtypes.

A TileMma names its dtypes as IR types (``FP8E4M3``, ``BF8E5M2``); these tests check that an fp8 or
bf8 wave tile lowers to one MFMA call per atom, that its fragments move through global memory and
LDS with vector accesses rather than one byte at a time, each claiming the alignment of its own
size, and that ``MmaTee.from_mma`` sizes and labels each operand from those dtypes.
Every kernel is lowered by each engine (``rocke.tests.engines.lower_with_engine`` over
``ENGINES``), and the C++ engine must emit the Python engine's IR; where the ``rocke_engine``
binding is not built, only the C++ cases skip. Where the host clang targets the arch, a separate
test compiles and links the LDS round trip once (compile-verified, never run).
"""

from __future__ import annotations

import re

import pytest

from rocke.core.ir import (
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I8,
    IRBuilder,
    PtrType,
)
from rocke.helpers.tiling import (
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.mma import TileMma, Tiling
from rocke.helpers.tiling.visualization import layout_render as lr
from rocke.tests.engines import ENGINES, lower_with_engine
from rocke.tests.ir_validity import assert_kernel_target_accepts


def _wave_tile_kernel(dtype, mma: TileMma):
    """One wave computes C = A * B^T for the MMA's wave tile, loading A and B from global memory."""
    wave_m, wave_n, wave_k = mma.shape
    b = IRBuilder(f"mma_dtypes_{dtype.name}")
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size
    a_ptr = b.param("A", PtrType(dtype, "global"), noalias=True, readonly=True)
    b_ptr = b.param("B", PtrType(dtype, "global"), noalias=True, readonly=True)
    c_ptr = b.param("C", PtrType(F32, "global"), noalias=True, writeonly=True)
    lane = b.thread_id_x()
    zero = b.const_i32(0)
    a_td = make_tensor_desc((wave_m, wave_k), (wave_k, 1), dtype)
    b_td = make_tensor_desc((wave_n, wave_k), (wave_k, 1), dtype)
    c_td = make_tensor_desc((wave_m, wave_n), (wave_n, 1), F32)
    acc = make_fragment(mma.c_desc, F32)
    fill_fragment(b, acc, 0)
    fa = load_fragment(b, a_ptr, make_window(a_td, (zero, zero)), mma.a_desc, lane)
    fb = load_fragment(b, b_ptr, make_window(b_td, (zero, zero)), mma.b_desc, lane)
    acc = mma(b, fa, fb, acc)
    store_fragment(b, c_ptr, make_window(c_td, (zero, zero)), acc, lane)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "dtype,intrinsic",
    [
        (FP8E4M3, "@llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8"),
        (BF8E5M2, "@llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8"),
    ],
)
def test_8bit_wave_tile_issues_one_mfma_per_atom(dtype, intrinsic, engine) -> None:
    # A 32x32x64 wave tile over the gfx942 16x16x32 atom is a 2 x 2 x 2 atom grid: eight calls.
    mma = TileMma(
        (32, 32, 64),
        a=dtype,
        b=dtype,
        c=F32,
        target="gfx942",
        tiling=Tiling(atom_shape=(16, 16, 32)),
    )
    assert mma.a_dtype is dtype and mma.b_dtype is dtype and mma.c_dtype is F32
    ir = lower_with_engine(_wave_tile_kernel(dtype, mma), "gfx942", engine=engine)
    calls = [line for line in ir.splitlines() if "call" in line and intrinsic in line]
    assert len(calls) == 8


def _accesses(ir: str, space: int) -> list[str]:
    """Every load and store of address space `space` in `ir`, as ``"load <type>, align <n>"`` or
    ``"store <type>, align <n>"``."""
    loads = re.findall(
        rf"= load ([^,]+), ptr addrspace\({space}\) [^,]+, align (\d+)", ir
    )
    stores = re.findall(
        rf"store (<[^>]+>|\S+) \S+, ptr addrspace\({space}\) [^,]+, align (\d+)", ir
    )
    return [f"load {t}, align {a}" for t, a in loads] + [
        f"store {t}, align {a}" for t, a in stores
    ]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("dtype", [FP8E4M3, BF8E5M2])
def test_8bit_wave_tile_loads_its_operands_with_vector_loads(dtype, engine) -> None:
    # Each lane holds 8 consecutive K elements of A and of B per atom, so the 32x32x64 tile reads
    # its operands as 8 loads of 8 bytes (A and B, 2 atoms along M or N, 2 along K) instead of 64
    # one-byte loads.
    mma = TileMma(
        (32, 32, 64),
        a=dtype,
        b=dtype,
        c=F32,
        target="gfx942",
        tiling=Tiling(atom_shape=(16, 16, 32)),
    )
    ir = lower_with_engine(_wave_tile_kernel(dtype, mma), "gfx942", engine=engine)
    loads = [access for access in _accesses(ir, 1) if access.startswith("load")]
    assert loads == ["load <8 x i8>, align 8"] * 8


def _lds_round_trip_kernel(dtype):
    """One wave loads the A operand of an fp8 32x32x64 tile from global memory as ``dtype``, stages
    it through an LDS buffer (store_fragment, then load_fragment) and writes it back to global.
    """
    mma = TileMma(
        (32, 32, 64),
        a=FP8E4M3,
        b=FP8E4M3,
        c=F32,
        target="gfx942",
        tiling=Tiling(atom_shape=(16, 16, 32)),
    )
    wave_m, _, wave_k = mma.shape
    b = IRBuilder(f"lds_stage_{dtype.name}")
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size
    src = b.param("A", PtrType(dtype, "global"), noalias=True, readonly=True)
    dst = b.param("Out", PtrType(dtype, "global"), noalias=True, writeonly=True)
    lane = b.thread_id_x()
    zero = b.const_i32(0)
    td = make_tensor_desc((wave_m, wave_k), (wave_k, 1), dtype)
    window = make_window(td, (zero, zero))
    lds = b.smem_alloc(dtype, [td.element_span])
    staged = load_fragment(b, src, window, mma.a_desc, lane)
    store_fragment(b, lds, window, staged, lane)
    loaded = load_fragment(b, lds, window, mma.a_desc, lane)
    store_fragment(b, dst, window, loaded, lane)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("dtype", [FP8E4M3, BF8E5M2, I8])
def test_8bit_fragment_moves_through_lds_with_vector_accesses(dtype, engine) -> None:
    # The 8 consecutive bytes a lane holds per atom are one LDS store and one LDS load each; no LDS
    # access moves a single byte, and each claims 8-byte alignment, which an 8-byte run of the
    # 8-element-aligned buffer has.
    kernel = _lds_round_trip_kernel(dtype)
    ir = lower_with_engine(kernel, "gfx942", engine=engine)
    assert (
        _accesses(ir, 3)
        == ["load <8 x i8>, align 8"] * 4 + ["store <8 x i8>, align 8"] * 4
    )


@pytest.mark.parametrize("dtype", [FP8E4M3, BF8E5M2, I8])
def test_8bit_lds_round_trip_compiles(dtype) -> None:
    assert_kernel_target_accepts(_lds_round_trip_kernel(dtype), "gfx942")


def test_mma_tee_sizes_and_labels_operands_from_the_mma_dtypes() -> None:
    fp8 = lr.MmaTee.from_mma(
        TileMma((16, 16, 32), a=FP8E4M3, b=FP8E4M3, c=F32, target="gfx942")
    )
    assert (fp8.a_dtype_bits, fp8.b_dtype_bits, fp8.c_dtype_bits) == (8, 8, 32)
    assert (fp8.in_dtype, fp8.out_dtype) == ("fp8e4m3", "f32")
    f16 = lr.MmaTee.from_mma(
        TileMma((16, 16, 16), a=F16, b=F16, c=F32, target="gfx90a")
    )
    assert (f16.a_dtype_bits, f16.c_dtype_bits) == (16, 32)
    assert (f16.in_dtype, f16.out_dtype) == ("f16", "f32")


def test_mma_input_label_spells_both_inputs_only_when_they_differ() -> None:
    assert lr.mma_input_label(F16, F16) == "f16"
    assert lr.mma_input_label(FP8E4M3, BF8E5M2) == "fp8e4m3/bf8e5m2"
