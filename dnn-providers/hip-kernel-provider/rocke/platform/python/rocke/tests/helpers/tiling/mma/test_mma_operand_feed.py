# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""How the TileMma driver feeds A and B operands from their fragments' storage registers.

A fragment holds its registers in its storage type: TF32 as ``i32``, packed sub-byte dtypes (fp4,
fp6, i4) as 32-bit words, every other dtype as its logical vector. The driver slices each
instruction's operand out of those registers (whole words for a packed dtype) and hands ``b.mma``
the vector type the instruction's A or B dtype takes. A TF32 instruction also reads F32 fragments,
dropping their low mantissa bits, so an F32 tensor feeds it with no conversion. A u8 operand is sliced
as the i8 registers of the same bits and handed over as u8.

Lowering tests pin the Python engine (``_lower_kernel_to_llvm_python``); the C++ engine is checked by
the byte-identity gate. Where the host clang targets the arch, the module is also compiled and linked
(compile-verified, never run).
"""

from __future__ import annotations

import pytest

from rocke.core.ir import (
    F16,
    F32,
    FP4E2M1,
    I8,
    I32,
    TF32,
    U8,
    IRBuilder,
    PtrType,
    VectorType,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.tiling import (
    Fragment,
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.mma import TileMma, Tiling

from .test_mma_issue import _calls, _compile_or_skip


def _wave_tile_kernel(mma: TileMma, tensor_dtype, pointer_dtype):
    """One wave computes C = A * B^T for the MMA's wave tile, with A (M, K) and B (N, K) read from
    global tensors of ``tensor_dtype`` behind ``pointer_dtype`` pointers and the f32 result stored.
    Returns the kernel and the A fragment the driver was given."""
    wave_m, wave_n, wave_k = mma.shape
    b = IRBuilder("operand_feed")
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size
    a_ptr = b.param("A", PtrType(pointer_dtype, "global"), noalias=True, readonly=True)
    b_ptr = b.param("B", PtrType(pointer_dtype, "global"), noalias=True, readonly=True)
    c_ptr = b.param("C", PtrType(F32, "global"), noalias=True, writeonly=True)
    lane = b.thread_id_x()
    zero = b.const_i32(0)
    a_td = make_tensor_desc((wave_m, wave_k), (wave_k, 1), tensor_dtype)
    b_td = make_tensor_desc((wave_n, wave_k), (wave_k, 1), tensor_dtype)
    c_td = make_tensor_desc((wave_m, wave_n), (wave_n, 1), F32)
    acc = make_fragment(mma.c_desc, F32)
    fill_fragment(b, acc, 0)
    fa = load_fragment(b, a_ptr, make_window(a_td, (zero, zero)), mma.a_desc, lane)
    fb = load_fragment(b, b_ptr, make_window(b_td, (zero, zero)), mma.b_desc, lane)
    result = mma(b, fa, fb, acc)
    store_fragment(b, c_ptr, make_window(c_td, (zero, zero)), result, lane)
    b.ret()
    return b.kernel, fa


def _tf32_mma(shape, atom):
    tiling = Tiling(atom_shape=atom) if atom is not None else None
    return TileMma(shape, a=TF32, b=TF32, c=F32, target="gfx942", tiling=tiling)


# The xf32 tiles: (wave tile, atom hint, instructions issued). One atom, a 2 x 2 grid of C
# subtiles, a 4-step K walk, and a 2 x 2 x 2 grid on the 32x32x4 instruction.
_XF32_TILES = [
    pytest.param((16, 16, 8), None, 1, id="16x16x8"),
    pytest.param((32, 32, 8), (16, 16, 8), 4, id="32x32x8-atom16x16x8"),
    pytest.param((16, 16, 32), (16, 16, 8), 4, id="16x16x32-atom16x16x8"),
    pytest.param((64, 64, 8), (32, 32, 4), 8, id="64x64x8-atom32x32x4"),
]


@pytest.mark.parametrize("shape,atom,calls", _XF32_TILES)
@pytest.mark.parametrize(
    "tensor_dtype",
    [
        pytest.param(F32, id="from-f32-tensors"),
        pytest.param(TF32, id="from-tf32-tensors"),
    ],
)
def test_tf32_tile_issues_one_xf32_mfma_per_atom(
    shape, atom, calls, tensor_dtype
) -> None:
    # F32 fragments feed the TF32 instruction directly (it truncates); TF32 fragments, held as i32,
    # are reinterpreted as tf32. Either way every atom is one xf32 MFMA.
    mma = _tf32_mma(shape, atom)
    kernel, fa = _wave_tile_kernel(mma, tensor_dtype, tensor_dtype)
    assert fa.dtype == tensor_dtype
    ir = _lower_kernel_to_llvm_python(kernel, arch="gfx942")
    assert len(_calls(ir, mma.traits.llvm_intrinsic)) == calls
    _compile_or_skip(kernel, "gfx942")


def test_tf32_tile_refuses_an_f16_fragment() -> None:
    mma = _tf32_mma((16, 16, 8), None)
    b = IRBuilder("tf32_f16_operand")
    fa = make_fragment(mma.a_desc, F16)
    fb = make_fragment(mma.b_desc, TF32)
    acc = make_fragment(mma.c_desc, F32)
    for fragment in (fa, fb, acc):
        fill_fragment(b, fragment, 0)
    with pytest.raises(ValueError) as err:
        mma(b, fa, fb, acc)
    assert str(err.value) == (
        "MMA operand dtype mismatch -- operand=A, fragment=f16, expected tf32 or f32 (tf32 math "
        "also reads f32 fragments, dropping their low mantissa bits)"
    )


def test_operand_without_a_truncating_source_names_only_its_dtype() -> None:
    mma = TileMma((16, 16, 16), a=F16, b=F16, c=F32, target="gfx90a")
    b = IRBuilder("f16_f32_operand")
    fa = make_fragment(mma.a_desc, F16)
    fb = make_fragment(mma.b_desc, F32)
    acc = make_fragment(mma.c_desc, F32)
    for fragment in (fa, fb, acc):
        fill_fragment(b, fragment, 0)
    with pytest.raises(ValueError) as err:
        mma(b, fa, fb, acc)
    assert str(err.value) == (
        "MMA operand dtype mismatch -- operand=B, fragment=f32, expected f16"
    )


def _extract_indices(kernel, vector) -> list[int]:
    """The register indices read out of ``vector``, in program order."""
    return [
        op.attrs["index"]
        for op in kernel.body.ops
        if op.name == "vector.extract" and op.operands[0] is vector
    ]


def test_packed_tile_feeds_each_instruction_its_own_words() -> None:
    # 256 fp4 K elements over two 16x16x128 instructions: each lane's A fragment is 64 elements in 8
    # words, and the instructions take words [0, 4) and [4, 8), whole, with no repacking.
    mma = TileMma(
        (16, 16, 256),
        a=FP4E2M1,
        b=FP4E2M1,
        c=F32,
        target="gfx950",
        tiling=Tiling(atom_shape=(16, 16, 128)),
    )
    kernel, fa = _wave_tile_kernel(mma, FP4E2M1, I8)
    assert fa.storage_type == VectorType(I32, 8)
    assert _extract_indices(kernel, fa.value) == list(range(8))
    mma_ops = [op for op in kernel.body.ops if op.name == "tile.mma"]
    assert len(mma_ops) == 2
    assert all(op.operands[0].type == VectorType(I32, 4) for op in mma_ops)
    ir = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
    assert len(_calls(ir, mma.traits.llvm_intrinsic)) == 2
    _compile_or_skip(kernel, "gfx950")


def _unsigned_tile_kernel(mma: TileMma):
    """One wave computes C = A * B^T for a u8 x i8 wave tile with an i32 result. A and B are read
    from i8 tensors: load_fragment refuses u8, so A's registers are loaded as i8 and reinterpreted as
    u8, the same bits."""
    wave_m, wave_n, wave_k = mma.shape
    b = IRBuilder("unsigned_operand_feed")
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size
    a_ptr = b.param("A", PtrType(I8, "global"), noalias=True, readonly=True)
    b_ptr = b.param("B", PtrType(I8, "global"), noalias=True, readonly=True)
    c_ptr = b.param("C", PtrType(I32, "global"), noalias=True, writeonly=True)
    lane = b.thread_id_x()
    zero = b.const_i32(0)
    a_td = make_tensor_desc((wave_m, wave_k), (wave_k, 1), I8)
    b_td = make_tensor_desc((wave_n, wave_k), (wave_k, 1), I8)
    c_td = make_tensor_desc((wave_m, wave_n), (wave_n, 1), I32)
    acc = make_fragment(mma.c_desc, I32)
    fill_fragment(b, acc, 0)
    a_i8 = load_fragment(b, a_ptr, make_window(a_td, (zero, zero)), mma.a_desc, lane)
    fa = Fragment(
        mma.a_desc, U8, b.bitcast(a_i8.value, VectorType(U8, a_i8.storage_type.count))
    )
    fb = load_fragment(b, b_ptr, make_window(b_td, (zero, zero)), mma.b_desc, lane)
    result = mma(b, fa, fb, acc)
    store_fragment(b, c_ptr, make_window(c_td, (zero, zero)), result, lane)
    b.ret()
    return b.kernel


# u8 x i8 tiles of the gfx1250 16x16x64 instruction: (wave tile, instructions issued). Two steps
# along K, then a 2 x 2 grid of C subtiles, then both.
_UNSIGNED_TILES = [
    pytest.param((16, 16, 128), 2, id="16x16x128-two-k-steps"),
    pytest.param((32, 32, 64), 4, id="32x32x64-four-c-subtiles"),
    pytest.param((32, 32, 128), 8, id="32x32x128-both"),
]


@pytest.mark.parametrize("shape,calls", _UNSIGNED_TILES)
def test_unsigned_tile_of_several_instructions_issues_each_one(shape, calls) -> None:
    # Every instruction takes a u8 A operand sliced from the u8 fragment.
    mma = TileMma(
        shape, a=U8, b=I8, target="gfx1250", atom_override="wmma_i32_16x16x64_iu8"
    )
    kernel = _unsigned_tile_kernel(mma)
    mma_ops = [op for op in kernel.body.ops if op.name == "tile.mma"]
    assert len(mma_ops) == calls
    assert all(op.operands[0].type == VectorType(U8, 32) for op in mma_ops)
    assert all(op.operands[1].type == VectorType(I8, 32) for op in mma_ops)
    ir = _lower_kernel_to_llvm_python(kernel, arch="gfx1250")
    assert len(_calls(ir, mma.traits.llvm_intrinsic)) == calls
    _compile_or_skip(kernel, "gfx1250")
