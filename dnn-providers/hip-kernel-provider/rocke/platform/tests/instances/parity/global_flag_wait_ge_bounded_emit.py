# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for the bounded flag spin
(``global_flag_wait_ge_bounded``) and the ``scope`` of ``global_atomic_add``.

Each case adds into an i32 and an f32 slot with the given atomic scope (None =
the unscoped op), spins on the flag with the given wait scope / ordering and
stores the last observed flag value.
"""

from rocke.core.ir import F32, I32, IRBuilder, PtrType

from _emit_common import run_emit

# (atomic scope, wait scope, wait ordering)
CASES = (
    (None, "agent", "acquire"),
    ("workgroup", "workgroup", "monotonic"),
    ("agent", "system", "acquire"),
    ("system", "agent", "monotonic"),
)


def build_flag_wait(b: IRBuilder, atomic_scope, wait_scope, wait_ordering) -> None:
    flag = b.param("Flag", PtrType(I32, "global"))
    acc = b.param("Acc", PtrType(F32, "global"))
    idx = b.param("Idx", I32)
    expect = b.param("Expect", I32)
    polls = b.param("Polls", I32)
    one = b.const_i32(1)
    b.global_atomic_add(flag, idx, one, ordering="release", scope=atomic_scope)
    b.global_atomic_add(acc, idx, b.const_f32(1.0), scope=atomic_scope)
    seen = b.global_flag_wait_ge_bounded(
        flag, idx, expect, polls, scope=wait_scope, ordering=wait_ordering
    )
    b.global_store(flag, b.add(idx, one), seen)
    b.ret()


def _spec(idx):
    if not 0 <= idx < len(CASES):
        raise SystemExit(f"unknown config index {idx}")
    return (CASES[idx], "gfx950")


def _build(spec, *, arch):
    b = IRBuilder("flag_wait_ge_bounded")
    b.kernel.attrs["max_workgroup_size"] = 64
    build_flag_wait(b, *spec)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
