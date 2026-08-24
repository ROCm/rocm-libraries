#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/layernorm2d_emit.py -- Python reference emitter for the
# LayerNorm2D parity STRESS harness. Selects one of N configs by argv[1]
# (the config index) and prints _native_lower(arch='gfx950') to stdout
# so it can be byte-compared with the C emitter layernorm2d_emit.c.
import sys

from rocke.instances.common.layernorm2d import (
    LayerNorm2DSpec,
    build_layernorm2d,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# (n_per_block, block_size, vec, dtype, save_mean_invstd)
CONFIGS = [
    # --- original 6 sampled configs (regression anchor) ---
    (4096, 256, 4, "f16", False),  # 0
    (4096, 256, 8, "f16", False),  # 1
    (4096, 256, 4, "bf16", False),  # 2
    (2048, 128, 4, "f16", True),  # 3
    (8192, 256, 8, "f16", False),  # 4
    (1024, 256, 2, "bf16", True),  # 5
    # --- tiny / minimal shapes (edge: smallest valid N) ---
    (128, 64, 2, "f16", False),  # 6  N = block*vec, elems=2
    (256, 64, 4, "f16", True),  # 7
    (512, 64, 8, "bf16", False),  # 8
    (128, 64, 2, "bf16", True),  # 9
    # --- all block sizes ---
    (4096, 512, 4, "f16", False),  # 10
    (8192, 1024, 8, "f16", False),  # 11  block_size at the 1024 cap
    (2048, 1024, 2, "bf16", True),  # 12
    (1024, 128, 8, "f16", False),  # 13
    # --- fp16 alias (validate_io allows "fp16"; Python Literal is f16/bf16) ---
    (4096, 256, 4, "fp16", False),  # 14  ADVERSARIAL: aliasing
    (2048, 128, 2, "fp16", True),  # 15
    # --- odd / non-power-of-2 multipliers (edge, still divisible) ---
    (1536, 256, 2, "f16", False),  # 16  1536 = 256*2*3
    (3072, 256, 4, "bf16", False),  # 17  3072 = 256*4*3
    (5120, 256, 4, "f16", True),  # 18  5120 = 256*4*5 (prime factor 5)
    (1792, 128, 2, "bf16", False),  # 19  1792 = 128*2*7 (prime factor 7)
    (2816, 128, 2, "f16", False),  # 20  2816 = 128*2*11 (prime factor 11)
    (6656, 256, 2, "bf16", True),  # 21  6656 = 256*2*13 (prime factor 13)
    # --- two-pass path: elems_per_thread > 64 (re-stream X in pass 2) ---
    (16384, 256, 8, "f16", False),  # 22  elems=64 (boundary, single-pass)
    (33280, 256, 2, "f16", False),  # 23  elems=130 -> TWO-PASS (256*2*65)
    (32768, 256, 8, "bf16", False),  # 24  elems=128 -> TWO-PASS
    (65536, 256, 8, "f16", True),  # 25  elems=256 -> TWO-PASS, save mv
    (131072, 512, 8, "bf16", False),  # 26  elems=256 -> TWO-PASS, big block
    (34816, 256, 2, "bf16", True),  # 27  elems=136 two-pass + save mv bf16 (256*2*68)
    # --- very large N single-pass at high block_size ---
    (65536, 1024, 8, "f16", False),  # 28  elems=64 boundary at 1024 block
    (133120, 1024, 2, "bf16", False),  # 29 elems=130 two-pass at 1024 block (1024*2*65)
    # --- vec sweep at fixed shape ---
    (2048, 256, 2, "f16", False),  # 30
    (4096, 256, 4, "f16", True),  # 31  same as 0 but save mv
    (8192, 256, 8, "bf16", True),  # 32
    # --- block 512 + two-pass ---
    (66560, 512, 2, "f16", False),  # 33  elems=130 two-pass, block 512 (512*2*65)
    (133120, 512, 4, "bf16", True),  # 34  elems=260 two-pass + save mv (512*4*65)
]


def _spec(idx: int) -> LayerNorm2DSpec:
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    n, bs, v, dt, smv = CONFIGS[idx]
    return LayerNorm2DSpec(
        n_per_block=n, block_size=bs, vec=v, dtype=dt, save_mean_invstd=smv
    )


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            f"usage: layernorm2d_emit.py <config_index 0..{len(CONFIGS) - 1}> [ll|ir|verify]\n"
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
    kernel = build_layernorm2d(spec)
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
