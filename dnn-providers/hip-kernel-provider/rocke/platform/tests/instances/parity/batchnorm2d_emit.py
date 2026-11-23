#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/batchnorm2d_emit.py -- Python reference emitter for the
# BatchNorm2D parity STRESS harness. Selects one of N configs by argv[1]
# (the config index) and prints _native_lower(arch='gfx950') to stdout
# so it can be byte-compared with the C emitter batchnorm2d_emit.c.
import sys

from rocke.instances.common.batchnorm2d import (
    BatchNorm2DSpec,
    build_batchnorm2d,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# (n, c, h, w, block_size, vec, dtype, training, save_mean_invstd, channels_per_block,
#  n_groups, fast_reduce)
CONFIGS = [
    # --- small single-pass, training ---
    (2, 8, 16, 16, 64, 2, "f16", True, False, 1, 1, False),  # 0
    (2, 8, 16, 16, 64, 2, "bf16", True, False, 1, 1, False),  # 1
    (2, 8, 16, 16, 64, 2, "f16", True, True, 1, 1, False),  # 2  save mean/invstd
    # --- inference (no reduction, no LDS) ---
    (2, 8, 16, 16, 64, 2, "f16", False, False, 1, 1, False),  # 3
    (2, 8, 16, 16, 64, 2, "bf16", False, False, 1, 1, False),  # 4
    # --- vec / block sweep ---
    (2, 8, 32, 32, 128, 4, "f16", True, False, 1, 1, False),  # 5
    (4, 4, 16, 16, 64, 4, "bf16", True, True, 1, 1, False),  # 6
    (1, 16, 8, 8, 32, 2, "f16", True, False, 1, 1, False),  # 7  N=1 (single sample)
    # --- fp16 alias ---
    (2, 8, 16, 16, 64, 2, "fp16", True, False, 1, 1, False),  # 8
    # --- two-pass path: elems_per_thread > 64 ---
    (8, 4, 32, 32, 64, 2, "f16", True, False, 1, 1, False),  # 9  elems=128 -> two-pass
    (8, 4, 32, 32, 64, 2, "bf16", False, False, 1, 1, False),  # 10 two-pass shape, inference
    (8, 4, 32, 32, 64, 4, "f16", True, True, 1, 1, False),  # 11 two-pass + save mv
    # --- channels_per_block > 1 lever ---
    (2, 8, 16, 16, 64, 2, "f16", True, False, 2, 1, False),  # 12 training, cpb=2
    (2, 8, 16, 16, 64, 2, "bf16", False, False, 4, 1, False),  # 13 inference, cpb=4 (all channels, 1 CTA)
    (4, 4, 16, 16, 64, 4, "bf16", True, True, 2, 1, False),  # 14 training + save mv, cpb=2
    (8, 4, 32, 32, 64, 2, "f16", True, False, 4, 1, False),  # 15 two-pass + cpb=4
    # --- n_groups > 1 lever (inference) ---
    (4, 4, 16, 16, 256, 2, "f16", False, False, 1, 2, False),  # 16 ng=2
    (4, 4, 16, 16, 256, 2, "bf16", False, False, 1, 4, False),  # 17 ng=4
    (4, 8, 16, 16, 256, 2, "f16", False, False, 2, 2, False),  # 18 cpb=2 + ng=2
    # --- fast_reduce lever (training-only): wave-shuffle-first Welford ---
    (2, 8, 16, 16, 64, 2, "f16", True, False, 1, 1, True),  # 19 single-wave (block_size==wave_size)
    (4, 4, 16, 16, 128, 2, "bf16", True, True, 1, 1, True),  # 20 cross-warp (num_warps=2) + save mv
    (2, 8, 16, 16, 64, 2, "f16", True, False, 2, 1, True),  # 21 fast_reduce + cpb=2
    # --- HW tail (not divisible by block_size*vec): real CNN shapes ---
    (1, 16, 14, 14, 64, 2, "f16", True, False, 1, 1, False),  # 22 HW=196, training
    (1, 16, 14, 14, 64, 2, "f16", False, False, 1, 1, False),  # 23 HW=196, inference
    (1, 144, 56, 56, 256, 4, "f16", True, False, 1, 1, False),  # 24 HW=3136, training
    (1, 480, 24, 24, 128, 4, "bf16", True, True, 1, 1, False),  # 25 HW=576, save mean/invstd
    (1, 16, 14, 14, 64, 2, "f16", True, False, 1, 1, True),  # 26 HW tail + fast_reduce
    (1, 16, 14, 14, 64, 2, "f16", True, False, 2, 1, False),  # 27 HW tail + cpb=2
    (2, 16, 14, 14, 64, 2, "f16", True, False, 1, 1, False),  # 28 HW tail + N=2
    # --- n_groups > 1 lever (training): each thread accumulates its own
    # Welford partial over its (og, tid_within_row) slice, merged via the
    # same welford_block_reduce_stable/_wave_prologue call the n_groups==1
    # path uses -- always reloads X (no register cache) across pass 1/2 ---
    (4, 4, 16, 16, 256, 2, "f16", True, False, 1, 2, False),  # 29 ng=2, training
    (4, 4, 16, 16, 256, 2, "bf16", True, False, 1, 4, False),  # 30 ng=4, training
    (4, 8, 16, 16, 256, 2, "f16", True, False, 2, 2, False),  # 31 cpb=2 + ng=2, training
    (4, 4, 16, 16, 256, 2, "f16", True, True, 1, 2, False),  # 32 ng=2, training + save mv
    (4, 4, 16, 16, 128, 2, "f16", True, False, 1, 2, True),  # 33 ng=2 + fast_reduce, training
    # --- n_groups > 1, masked tail: HW does not divide hw_threads*vec
    # exactly (7x7=49, 14x14=196 -- shapes no power-of-two hw_threads can
    # tile exactly), exercising the new tail_chunks_ng path ---
    (4, 4, 7, 7, 256, 2, "f16", True, False, 1, 4, False),  # 34 HW=49 pure tail (ept_row_ng=0), training
    (4, 4, 14, 14, 256, 2, "f16", True, False, 1, 4, False),  # 35 HW=196 mixed main+tail, training
    (4, 4, 7, 7, 256, 2, "f16", False, False, 1, 4, False),  # 36 HW=49 pure tail, inference
    (4, 4, 7, 7, 256, 2, "f16", True, True, 1, 4, False),  # 37 HW=49 tail + save mean/invstd
    (4, 4, 7, 7, 128, 2, "f16", True, False, 1, 2, True),  # 38 HW=49 tail + fast_reduce
    (4, 4, 7, 7, 256, 2, "bf16", True, False, 1, 4, False),  # 39 HW=49 tail, bf16
    # --- dynamic (AOT-generic) variant: requires cpb==1, ng==1 ---
    (2, 8, 14, 14, 256, 4, "f16", True, False, 1, 1, False, True),  # 40 training, HW tail
    (1, 16, 8, 8, 64, 2, "f16", False, False, 1, 1, False, True),  # 41 inference
    (2, 8, 16, 16, 64, 2, "f16", True, True, 1, 1, True, True),  # 42 save_mv + fast_reduce
]


def _spec(idx: int) -> BatchNorm2DSpec:
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    cfg = CONFIGS[idx]
    n, c, h, w, bs, v, dt, training, smv, cpb, ng, fr = cfg[:12]
    dynamic = cfg[12] if len(cfg) > 12 else False
    return BatchNorm2DSpec(
        n=n,
        c=c,
        h=h,
        w=w,
        block_size=bs,
        vec=v,
        dtype=dt,
        training=training,
        save_mean_invstd=smv,
        channels_per_block=cpb,
        n_groups=ng,
        fast_reduce=fr,
        dynamic=dynamic,
    )


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            f"usage: batchnorm2d_emit.py <config_index 0..{len(CONFIGS) - 1}> [ll|ir|verify]\n"
        )
        return 2
    if sys.argv[1] == "--count":
        sys.stdout.write(str(len(CONFIGS)) + "\n")
        return 0
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    spec = _spec(idx)
    kernel = build_batchnorm2d(spec)
    if mode == "ll":
        text = _native_lower(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
