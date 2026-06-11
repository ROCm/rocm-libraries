#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/moe_sorting_emit.py -- Python reference emitter for the
# MoE-sorting parity harness. Selects one of the sampled configs by argv[2]
# (the config index) and the phase by argv[1] ("hist"/"scan"/"scatter"),
# builds the MoeSortingSpec, builds the kernel via the matching
# build_moe_sort_* and prints lower_kernel_to_llvm(arch='gfx950') to stdout so
# it can be byte-compared with the C emitter moe_sorting_emit.c.
import sys

from ck_dsl.instances.common.moe_sorting import (
    MoeSortingSpec,
    build_moe_sort_histogram,
    build_moe_sort_scan,
    build_moe_sort_scatter,
)
from ck_dsl import lower_kernel_to_llvm


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


_BUILD = {
    "hist": build_moe_sort_histogram,
    "scan": build_moe_sort_scan,
    "scatter": build_moe_sort_scatter,
}


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write(
            "usage: moe_sorting_emit.py <phase hist|scan|scatter> <config_index 0..5>\n"
        )
        return 2
    phase = sys.argv[1]
    idx = int(sys.argv[2])
    if phase not in _BUILD:
        raise SystemExit(f"unknown phase {phase}")
    spec = _spec(idx)
    kernel = _BUILD[phase](spec, arch="gfx950")
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
