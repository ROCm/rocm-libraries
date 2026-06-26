#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_wmma_attention_fwd_emit.py -- Python reference emitter for
# the gfx1250 dense WMMA FMHA forward parity harness.
from rocke.instances.gfx1250.wmma_attention_fwd import (
    WmmaAttentionFwdSpec,
    build_wmma_attention_fwd,
)
from _emit_common import run_emit


def _spec(idx: int) -> WmmaAttentionFwdSpec:
    if idx == 0:
        return WmmaAttentionFwdSpec(head_size=64, num_query_heads=8)
    if idx == 1:
        return WmmaAttentionFwdSpec(head_size=128, num_query_heads=16)
    if idx == 2:
        return WmmaAttentionFwdSpec(head_size=64, num_query_heads=32, num_kv_heads=8)
    if idx == 3:
        return WmmaAttentionFwdSpec(head_size=64, num_query_heads=8, mask_mode="causal")
    if idx == 4:
        return WmmaAttentionFwdSpec(
            head_size=256, num_query_heads=4, mask_mode="causal"
        )
    if idx == 5:
        return WmmaAttentionFwdSpec(head_size=128, num_query_heads=16, num_kv_heads=2)
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_wmma_attention_fwd(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_wmma_attention_fwd_emit.py <config_index 0..5>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
