# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The f32, i8, f64 and bf16 (without the ``_1k`` suffix) MFMAs on gfx90a hardware.

Each case runs one MFMA on one wave and compares the result with numpy. The
host places a random A (M x K) and B (K x N) into each lane's registers through
the instruction's own A and B lane maps, and reads the accumulator back through
its C lane map, so a wrong map fails the comparison. The inputs are small
integers, exact in every format, so the comparison is exact. Machines without
a gfx90a skip.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.core.arch import ArchTarget
from rocke.core.ir import (
    I32,
    IRBuilder,
    PtrType,
    VectorType,
    dtype_to_ir_type,
    type_bits,
)
from rocke.tests.gpu import run_on_device
from rocke.tests.lane_maps import lane_placements

_WAVE = 64

# gfx90a catalog rows checked here.
_OPS = (
    "mfma_f32_16x16x4_f32",
    "mfma_f32_32x32x2_f32",
    "mfma_i32_16x16x16_i8",
    "mfma_i32_32x32x8_i8",
    "mfma_f64_16x16x4_f64",
    "mfma_f32_16x16x8_bf16",
    "mfma_f32_32x32x4_bf16",
)

# Catalog dtype -> the host array that holds its values (bf16 values are built
# as f32 and keep the high half).
_HOST = {
    "fp32": np.float32,
    "fp64": np.float64,
    "bf16": np.float32,
    "i8": np.int8,
    "i32": np.int32,
}


def _operand(b: IRBuilder, name: str, register):
    """Each lane's operand, loaded as 32-bit words and reinterpreted."""
    words = type_bits(register) // 32
    ptr = b.param(name, PtrType(I32, "global"))
    index = b.mul(b.thread_id_x(), b.const_i32(words))
    loaded = (
        b.global_load(ptr, index, I32)
        if words == 1
        else b.global_load_vN(ptr, index, I32, words)
    )
    return b.bitcast(loaded, register)


def _kernel(op):
    """D = A * B with C = 0; each lane stores its accumulator slots in order."""
    b = IRBuilder(f"gfx90a_{op.op_id}")
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    a = _operand(b, "A", VectorType(dtype_to_ir_type(op.a_dtype), op.a_frag_len))
    bv = _operand(b, "B", VectorType(dtype_to_ir_type(op.b_dtype), op.b_frag_len))
    c_elem = dtype_to_ir_type(op.c_dtype)
    out = b.param("D", PtrType(c_elem, "global"))
    d = b.mma(op, a, bv, b.zero_vec(c_elem, op.c_frag_len))
    base = b.mul(b.thread_id_x(), b.const_i32(op.c_frag_len))
    for slot in range(op.c_frag_len):
        b.global_store(out, b.add(base, b.const_i32(slot)), b.vec_extract(d, slot))
    b.ret()
    return b.kernel


def _registers(layout, matrix: np.ndarray, dtype: str) -> np.ndarray:
    """Each lane's operand: the matrix elements its slots hold, as 32-bit words."""
    lanes = np.array(
        [matrix[coord] for coord in lane_placements(layout, _WAVE)],
        dtype=_HOST[dtype],
    )
    if dtype == "bf16":
        # A bf16 value is the high half of the f32 with the same value; the
        # small integers used here are exact in both.
        lanes = (lanes.view(np.uint32) >> 16).astype(np.uint16)
    return np.ascontiguousarray(lanes).view(np.int32)


@pytest.mark.parametrize("op_id", _OPS)
def test_the_mfma_matches_numpy(op_id):
    op = ArchTarget.from_gfx("gfx90a").mma.by_op_id(op_id)
    rng = np.random.default_rng(sum(map(ord, op_id)))
    low, high = (-128, 128) if op.a_dtype == "i8" else (-4, 5)
    a = rng.integers(low, high, size=(op.m, op.k))
    b = rng.integers(low, high, size=(op.k, op.n))
    out = np.zeros(_WAVE * op.c_frag_len, dtype=_HOST[op.c_dtype])
    run_on_device(
        _kernel(op),
        {
            "A": _registers(op.a_layout(), a, op.a_dtype),
            "B": _registers(op.b_layout(), b, op.b_dtype),
            "D": out,
        },
        threads=_WAVE,
        arches=("gfx90a",),
    )
    got = np.zeros((op.m, op.n), dtype=np.float64)
    for value, coord in zip(out, lane_placements(op.c_layout(), _WAVE)):
        got[coord] = value
    np.testing.assert_array_equal(got, a @ b)
