# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Page/compute decoupling across targets, working dtypes and head dimensions."""

from importlib import import_module


def spec_for_index(idx):
    if not 0 <= idx < 48:
        raise SystemExit(f"unknown config index {idx}")
    arch = "gfx942" if idx < 24 else "gfx950"
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    spec = module.UnifiedAttention3DTiledSpec(
        head_size=(64, 128, 256)[idx // 4 % 3],
        block_size=(1, 16, 32, 64)[idx % 4],
        num_query_heads=8,
        num_kv_heads=2,
        dtype="fp16" if idx // 12 % 2 == 0 else "bf16",
        use_sinks=False,
        sliding_window=17 if idx % 2 else 0,
        has_softcap=False,
        num_segments=8,
        num_seqs=3,
        tile_size_override=32,
        waves_per_eu=3,
    )
    return spec, arch


def build(spec, arch):
    return import_module(
        f"kernels.{arch}.attention_tiled_3d"
    ).build_unified_attention_3d_tiled(spec, arch=arch)


if __name__ == "__main__":
    from _emit_common import run_emit

    raise SystemExit(run_emit(spec_for_index, build))
