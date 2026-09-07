#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python reference emitter for fused top-k active-expert packing."""

from _emit_common import run_emit
from rocke.instances.common.moe_topk_active_pack import (
    MoeTopkActivePackSpec,
    build_moe_topk_active_pack,
)


def _spec(index: int) -> MoeTopkActivePackSpec:
    if index == 0:
        return MoeTopkActivePackSpec(tokens=1, experts=896, topk=16)
    if index == 1:
        return MoeTopkActivePackSpec(tokens=8, experts=896, topk=16)
    raise SystemExit(f"unknown config index {index}")


def main() -> int:
    return run_emit(
        _spec,
        build_moe_topk_active_pack,
        usage="usage: moe_topk_active_pack_emit.py <config_index 0..1> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
