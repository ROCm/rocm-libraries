#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/streamk_gemm_emit.py -- Python reference emitter for the
# streamk-GEMM parity harness. Selects one of the sampled configs by argv[1]
# (the config index), builds the StreamKGemmSpec, builds the kernel via
# build_streamk_gemm and prints lower_kernel_to_llvm(arch='gfx950') to stdout
# so it can be byte-compared with the C emitter streamk_gemm_emit.c.
import sys

from ck_dsl.instances.common.streamk_gemm import (
    StreamKGemmSpec,
    build_streamk_gemm,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> StreamKGemmSpec:
    if idx == 0:
        return StreamKGemmSpec(
            M=32,
            N=32,
            K=32,
            tile_m=16,
            tile_n=16,
            tile_k=16,
            dtype="f16",
            num_cus=8,
            blocks_per_cu=1,
            persistent=False,
        )
    if idx == 1:
        return StreamKGemmSpec(
            M=64,
            N=64,
            K=64,
            tile_m=16,
            tile_n=16,
            tile_k=32,
            dtype="f16",
            num_cus=8,
            blocks_per_cu=1,
            persistent=False,
        )
    if idx == 2:
        return StreamKGemmSpec(
            M=128,
            N=128,
            K=128,
            tile_m=16,
            tile_n=16,
            tile_k=32,
            dtype="f16",
            num_cus=256,
            blocks_per_cu=1,
            persistent=True,
        )
    if idx == 3:
        return StreamKGemmSpec(
            M=256,
            N=256,
            K=256,
            tile_m=32,
            tile_n=32,
            tile_k=16,
            dtype="f16",
            num_cus=304,
            blocks_per_cu=1,
            persistent=False,
        )
    if idx == 4:
        return StreamKGemmSpec(
            M=128,
            N=128,
            K=64,
            tile_m=32,
            tile_n=32,
            tile_k=16,
            dtype="f16",
            num_cus=256,
            blocks_per_cu=1,
            persistent=False,
        )
    if idx == 5:
        return StreamKGemmSpec(
            M=512,
            N=512,
            K=512,
            tile_m=16,
            tile_n=16,
            tile_k=32,
            dtype="f16",
            num_cus=304,
            blocks_per_cu=2,
            persistent=True,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: streamk_gemm_emit.py <config_index 0..5> [mode]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    spec = _spec(idx)
    kernel = build_streamk_gemm(spec, arch="gfx950")
    if mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        text = lower_kernel_to_llvm(kernel, arch="gfx950")
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
