# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for ``scf_if_else`` with results.

Config index = case * len(ARCHES) + arch_index, with CASES below.
- "branch": an if/else on ``lane < 24`` yields an f32 and a ``vec<i32x2>``
  (the then branch loads ``X[lane]`` and splats ``lane``; the else branch yields
  constants), and both results are stored.
- "loop": an if/else inside a loop that carries an i32 sum yields 1 or 10 per
  trip; the loop yields the sum plus the if/else's result.
"""

from rocke.core.ir import F32, I32, IRBuilder, PtrType, VectorType

from _emit_common import run_emit

ARCHES = ("gfx942", "gfx950", "gfx1201")
CASES = ("branch", "loop")


def build_branch(b: IRBuilder) -> None:
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    z = b.param("Z", PtrType(I32, "global"))
    lane = b.thread_id_x()
    branch = b.scf_if_else(
        b.cmp_lt(lane, b.const_i32(24)), result_types=[F32, VectorType(I32, 2)]
    )
    with branch as (then_ctx, else_ctx):
        with then_ctx:
            b.scf_yield(b.global_load(x, lane, F32), b.vector_splat(lane, 2))
        with else_ctx:
            b.scf_yield(b.const_f32(7.5), b.vector_splat(b.const_i32(-1), 2))
    value, pair = branch.results
    b.global_store(y, lane, value)
    b.global_store(z, lane, b.vec_extract(pair, 1))
    b.ret()


def build_loop(b: IRBuilder) -> None:
    w = b.param("W", PtrType(I32, "global"))
    lane = b.thread_id_x()
    taken = b.mod(lane, b.const_i32(4))
    loop = b.scf_for_iter(
        b.const_i32(0), b.const_i32(4), b.const_i32(1), [("acc", b.const_i32(0))]
    )
    with loop as (k, (acc,)):
        step = b.scf_if_else(b.cmp_lt(k, taken), result_types=[I32])
        with step as (then_ctx, else_ctx):
            with then_ctx:
                b.scf_yield(b.const_i32(1))
            with else_ctx:
                b.scf_yield(b.const_i32(10))
        b.scf_yield(b.add(acc, step.results[0]))
    b.global_store(w, lane, loop.results[0])
    b.ret()


def _spec(idx):
    if not 0 <= idx < len(CASES) * len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return CASES[idx // len(ARCHES)], ARCHES[idx % len(ARCHES)]


def _build(spec, *, arch):
    b = IRBuilder("scf_if_else_results")
    b.kernel.attrs["max_workgroup_size"] = 64
    {"branch": build_branch, "loop": build_loop}[spec](b)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
