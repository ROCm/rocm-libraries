#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python reference emitter for compact routed gather plus FP8 quantization."""

from __future__ import annotations

from _emit_common import run_emit
from rocke.instances.common.moe_compact_gather_quant import (
    MoeCompactGatherQuantSpec,
    build_moe_compact_gather_quant,
)


def _spec(index: int) -> MoeCompactGatherQuantSpec:
    if index == 0:
        return MoeCompactGatherQuantSpec(
            tokens=8,
            hidden=3584,
            max_blocks=128,
            input_dtype="bf16",
        )
    if index == 1:
        return MoeCompactGatherQuantSpec(
            tokens=32,
            hidden=4096,
            max_blocks=64,
            input_dtype="f16",
            block_size=128,
        )
    raise SystemExit(f"unknown config index {index}")


def _build(spec: MoeCompactGatherQuantSpec, arch: str = "gfx950"):
    return build_moe_compact_gather_quant(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage=(
            "usage: moe_compact_gather_quant_emit.py "
            "<config_index 0..1> [ll|ir|verify]\n"
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
