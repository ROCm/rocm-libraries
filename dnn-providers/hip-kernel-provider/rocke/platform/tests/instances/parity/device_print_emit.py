#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python reference emitter for the device-print prototype."""

from rocke.core.ir import F32, IRBuilder, KernelDef, PrintValue, PtrType

from _emit_common import run_emit


def _spec(idx: int) -> int:
    if not 0 <= idx <= 2:
        raise SystemExit(f"unknown config index {idx}")
    return idx


def _build(idx: int, *, arch: str = "gfx950") -> KernelDef:
    del arch
    b = IRBuilder("device_print")
    if idx == 0:
        pointer = b.param("p", PtrType(F32, "global"))
        integer = b.const_i32(-5)
        floating = b.const_f32(6.5)
        predicate = b.cmp_eq(integer, integer)
        b.device_print(
            "state=",
            integer,
            " unsigned=",
            PrintValue(integer, "u32"),
            " f=",
            floating,
            " ok=",
            predicate,
            " p=",
            pointer,
            predicate=predicate,
        )
    else:
        count = 7 if idx == 1 else 8
        value = b.const_i32(1)
        b.device_print(*(PrintValue(value, "i32") for _ in range(count)))
    return b.kernel


if __name__ == "__main__":
    raise SystemExit(run_emit(_spec, _build))
