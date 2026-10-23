# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paired paged/strided segment cases for both tiled decode targets, plus a
strided case with swizzled K/V LDS tiles per target (12: gfx942, 13: gfx950)."""

from importlib import import_module

from _emit_common import run_emit


def spec_for_index(idx):
    if not 0 <= idx < 14:
        raise SystemExit(f"unknown config index {idx}")
    swizzle = idx >= 12
    if swizzle:
        arch = "gfx942" if idx == 12 else "gfx950"
        case = 1
    else:
        arch = "gfx942" if idx % 6 < 3 else "gfx950"
        case = idx % 3
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    spec = module.UnifiedAttention3DTiledSpec(
        head_size=(64, 128, 256)[case],
        block_size=(16, 32, 64)[case],
        num_query_heads=(4, 8, 8)[case],
        num_kv_heads=(2, 2, 1)[case],
        dtype="bf16" if case == 1 else "fp16",
        use_sinks=False,
        sliding_window=17 if case == 2 else 0,
        has_softcap=False,
        num_segments=4,
        num_seqs=3,
        kv_layout="strided" if idx < 6 or swizzle else "paged",
        use_kv_lds_swizzle=swizzle,
    )
    return spec, arch


def build(spec, arch):
    return import_module(
        f"kernels.{arch}.attention_tiled_3d"
    ).build_unified_attention_3d_tiled(spec, arch=arch)


def main() -> int:
    return run_emit(spec_for_index, build)


if __name__ == "__main__":
    raise SystemExit(main())
