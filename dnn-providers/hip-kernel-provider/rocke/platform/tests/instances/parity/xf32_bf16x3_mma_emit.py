#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/xf32_bf16x3_mma_emit.py -- Python reference emitter for the
# emulated TF32 MMA atoms, which each engine expands into BF16x3 at lowering.
#
# No shipped kernel family selects an emulated atom yet, so without this family the
# byte-identity gate never compares the two engines on the expansion at all.
# The engine-parity case in tests/core/test_xf32_bf16x3_lowering.py needs the
# rocke_engine pybind module; this pair needs only the archive, so run_diff.py
# byte-compares it wherever the gate runs.
#
# Both sides build through the bare op_id -- rocke_b_mma has no MmaOp form --
# which also keeps each builder's logical-op_id fragment lookup under the gate.
#
# Operands are logical TF32 vectors, like the native atoms': fp32 fragments loaded
# as-is and wrapped with a bitcast, so no payload bit is rounded away.
#
# arch = gfx950 (the only target with emulated TF32 rows), llvm_flavor = AUTO,
# matching the C side.
from rocke.core.ir import F32, TF32, IRBuilder, KernelDef, PtrType, VectorType

from _emit_common import run_emit

# (logical op_id, A/B fragment length, accumulator fragment length). The lengths
# are the physical bf16 atom's: the logical atom borrows its fragment.
CONFIGS = [
    ("mfma_f32_16x16x32_xf32", 8, 4),
    ("mfma_f32_32x32x16_xf32", 8, 16),
]


def _spec(idx: int):
    """Config selector: the atom for `idx`, pinned to gfx950.

    The arch is pinned rather than defaulted because it is part of the case:
    gfx950 is the only target whose catalog carries these rows.
    """
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    return CONFIGS[idx], "gfx950"


def _build(spec, *, arch: str = "gfx950") -> KernelDef:
    """Load both fp32 fragments, issue the emulated tile.mma, store every lane.

    Each builder call is its own statement so the order the C side has to match
    is spelled out rather than left to argument evaluation.
    """
    op_id, ab_len, c_len = spec
    b = IRBuilder("xf32_bf16x3_mma")
    b.kernel.attrs["max_workgroup_size"] = 64
    a_ptr = b.param("a", PtrType(F32, "global"))
    b_ptr = b.param("b", PtrType(F32, "global"))
    out_ptr = b.param("out", PtrType(F32, "global"))
    tid = b.thread_id_x()
    ab_stride = b.const_i32(ab_len)
    ab_base = b.mul(tid, ab_stride)
    a = b.global_load_vN(a_ptr, ab_base, F32, ab_len)
    bv = b.global_load_vN(b_ptr, ab_base, F32, ab_len)
    a = b.bitcast(a, VectorType(TF32, ab_len))
    bv = b.bitcast(bv, VectorType(TF32, ab_len))
    zero = b.const_f32(0.0)
    c = b.vector_splat(zero, c_len)
    d = b.mma(op_id, a, bv, c)
    c_stride = b.const_i32(c_len)
    c_base = b.mul(tid, c_stride)
    for slot in range(c_len):
        offset = b.const_i32(slot)
        idx = b.add(c_base, offset)
        lane = b.vec_extract(d, slot)
        b.global_store(out_ptr, idx, lane)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: xf32_bf16x3_mma_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
