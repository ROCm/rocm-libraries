#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/mx_scale_decode_emit.py -- Python reference emitter for the E8M0
# scale decoder in isolation.
#
# The kernel loads one raw E8M0 byte, decodes it with decode_mx_scale_e8m0, and
# stores the f32 multiplier. It has no thread-id dependency, so a host build of
# the emitted IR and a one-thread GPU launch both run it once per code. The
# numeric tests execute all 256 codes through this kernel. Byte-compared against
# mx_scale_decode_emit.c.
#
# Modes: "ll" (default), "ir", "verify" are the gate modes. "hip" prints the HIP
# lowering of the kernel; the numeric tests build it, and the gate does not
# compare it.
#
# Every builder call is its own statement in the same order the C emitter
# issues it: each call consumes an SSA id, so a reordering emits the same IR
# under different names.
#
# arch = gfx950, llvm_flavor = AUTO, matching the C side.
import sys

from rocke.core.ir import F32, I8, IRBuilder, KernelDef, PtrType
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.helpers.mx_scale import decode_mx_scale_e8m0

from _emit_common import run_emit

KERNEL_NAME = "rocke_mx_scale_decode"


def build_mx_scale_decode(_spec=None, *, arch: str = "gfx950") -> KernelDef:
    """Build ``out[0] = decode_mx_scale_e8m0(codes[0])``."""
    b = IRBuilder(KERNEL_NAME)
    b.kernel.attrs["max_workgroup_size"] = 1
    codes = b.param("codes", PtrType(I8, "global"), readonly=True, align=1)
    out = b.param("out", PtrType(F32, "global"), writeonly=True, align=4)
    idx = b.const_i32(0)
    code = b.global_load_i8(codes, idx)
    scale = decode_mx_scale_e8m0(b, code)
    b.global_store(out, idx, scale, align=4)
    b.ret()
    return b.kernel


def _spec(idx: int):
    if idx != 0:
        raise SystemExit(f"unknown config index {idx}")
    return None


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[2] == "hip":
        _spec(int(sys.argv[1]))
        sys.stdout.write(lower_kernel_to_hip(build_mx_scale_decode(), arch="gfx950"))
        return 0
    return run_emit(
        _spec,
        build_mx_scale_decode,
        usage="usage: mx_scale_decode_emit.py <config_index 0> [ll|ir|verify|hip]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
