#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fmha_appendkv_emit.py -- Python reference emitter for the
# fmha_appendkv parity harness. Selects one of 6 FmhaAppendKvSpec configs by
# argv[1] (0..5), builds via build_fmha_fwd_appendkv and prints
# lower_kernel_to_llvm(arch='gfx950') so it can be byte-compared with the C
# emitter fmha_appendkv_emit.c.
import sys

from ck_dsl.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from ck_dsl.instances.common.fmha_appendkv import (
    FmhaAppendKvSpec,
    build_fmha_fwd_appendkv,
)
from ck_dsl.helpers.rotary import RotarySpec
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> FmhaAppendKvSpec:
    if idx == 0:
        return FmhaAppendKvSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=4, num_kv_heads=2),
                dtype="f16",
            ),
            batch=1,
            rotary=None,
            block_size=256,
        )
    if idx == 1:
        return FmhaAppendKvSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=4, num_kv_heads=2),
                dtype="f16",
            ),
            batch=2,
            rotary=RotarySpec(head_size=128, layout="half"),
            block_size=256,
        )
    if idx == 2:
        return FmhaAppendKvSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(
                    head_size=128,
                    num_query_heads=8,
                    num_kv_heads=4,
                    block_size_q=16,
                    block_size_k=64,
                ),
                dtype="bf16",
            ),
            batch=2,
            rotary=RotarySpec(head_size=128, layout="interleaved"),
            block_size=128,
        )
    if idx == 3:
        return FmhaAppendKvSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=4, num_kv_heads=2),
                dtype="f16",
            ),
            batch=1,
            rotary=RotarySpec(head_size=64, layout="half"),
            block_size=256,
        )
    if idx == 4:
        return FmhaAppendKvSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=32, num_query_heads=8, num_kv_heads=8),
                dtype="bf16",
            ),
            batch=2,
            rotary=None,
            block_size=256,
        )
    if idx == 5:
        return FmhaAppendKvSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=192, num_query_heads=4, num_kv_heads=2),
                dtype="f16",
            ),
            batch=1,
            rotary=RotarySpec(head_size=192, layout="half"),
            block_size=256,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: fmha_appendkv_emit.py <config_index 0..5>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_fmha_fwd_appendkv(spec, arch="gfx950")
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
