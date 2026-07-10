#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/moe_sorting_emit.py -- Python reference emitter for the
# MoE-sorting parity harness. Selects one of the sampled configs by argv[2]
# (the config index) and the phase by argv[1] ("hist"/"scan"/"scatter"),
# builds the MoeSortingSpec, builds the kernel via the matching
# build_moe_sort_* and prints _native_lower(arch='gfx950') to stdout so
# it can be byte-compared with the C emitter moe_sorting_emit.c.
import sys

from rocke.instances.common.moe_sorting import (
    MoeSortingSpec,
    build_moe_sort_histogram,
    build_moe_sort_scan,
    build_moe_sort_scatter,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


def _spec(idx: int) -> MoeSortingSpec:
    if idx == 0:
        return MoeSortingSpec(tokens=2, topk=8, experts=8, block_size=64)
    if idx == 1:
        return MoeSortingSpec(tokens=16, topk=4, experts=32, block_size=256)
    if idx == 2:
        return MoeSortingSpec(tokens=32, topk=8, experts=64, block_size=256)
    if idx == 3:
        return MoeSortingSpec(tokens=128, topk=2, experts=32, block_size=512)
    if idx == 4:
        return MoeSortingSpec(tokens=8, topk=16, experts=16, block_size=128)
    if idx == 5:
        return MoeSortingSpec(tokens=2, topk=8, experts=64, block_size=256)
    raise SystemExit(f"unknown config index {idx}")


# Flat config index encoding (mirrors C emitter):
#   0.. 5 = hist    / spec 0..5
#   6..11 = scan    / spec 0..5
#  12..17 = scatter / spec 0..5
_PHASES = ["hist", "scan", "scatter"]
_BUILD = {
    "hist": build_moe_sort_histogram,
    "scan": build_moe_sort_scan,
    "scatter": build_moe_sort_scatter,
}


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: moe_sorting_emit.py <flat_config_index 0..17> [mode]\n"
        )
        return 2
    flat = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    phase_idx = flat // 6
    if phase_idx >= len(_PHASES):
        raise SystemExit(f"unknown config index {flat}")
    phase = _PHASES[phase_idx]
    idx = flat % 6
    spec = _spec(idx)
    kernel = _BUILD[phase](spec, arch="gfx950")
    if mode == "ll":
        text = _native_lower(kernel, arch="gfx950")
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
