# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Page/compute decoupling across targets, working dtypes and head dimensions."""

from dataclasses import replace
from importlib import import_module

CONFIG_COUNT = 98


def spec_for_index(idx):
    if not 0 <= idx < CONFIG_COUNT:
        raise SystemExit(f"unknown config index {idx}")
    if idx >= 66:
        case = idx - 66
        arch = "gfx942" if case < 16 else "gfx950"
        module = import_module(f"kernels.{arch}.attention_tiled_3d")
        local = case % 16
        if local >= 12:
            return (
                module.UnifiedAttentionReduceTiledSpec(
                    head_size=(64, 128, 256, 64)[local - 12],
                    num_query_heads=8,
                    num_kv_heads=2,
                    dtype="bf16",
                    num_segments=8,
                    waves_per_eu=3,
                    metadata_containment=True,
                ),
                arch,
            )
        return (
            module.UnifiedAttention3DTiledSpec(
                head_size=(64, 128, 256)[local // 4],
                block_size=(1, 16, 32, 64)[local % 4],
                num_query_heads=8,
                num_kv_heads=2,
                dtype="bf16" if local % 2 else "fp16",
                use_sinks=local >= 8,
                sliding_window=33 if local >= 8 else 0,
                has_softcap=local >= 8,
                use_alibi=local >= 8,
                use_qq_bias=local >= 8,
                num_segments=8,
                num_seqs=3,
                tile_size_override=32,
                waves_per_eu=3,
                use_decode_grid=True,
                metadata_containment=True,
            ),
            arch,
        )
    feature_case = (idx - 48) % 9 if idx >= 48 else None
    arch = "gfx942" if idx < 24 or 48 <= idx < 57 else "gfx950"
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
    if feature_case is not None:
        combined = feature_case >= 4
        spec = replace(
            spec,
            head_size=128,
            block_size=(
                (1, 16, 32, 64)[feature_case - 4] if 4 <= feature_case < 8 else 1
            ),
            dtype="bf16" if feature_case == 8 else "fp16",
            sliding_window=33,
            has_softcap=combined or feature_case == 0,
            use_sinks=combined or feature_case == 1,
            use_alibi=combined or feature_case == 2,
            use_qq_bias=combined or feature_case == 3,
        )
    return spec, arch


def build(spec, arch):
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    builder = (
        module.build_unified_attention_3d_tiled
        if hasattr(spec, "block_size")
        else module.build_unified_attention_reduce_tiled
    )
    return builder(spec, arch=arch)


def main():
    from _emit_common import run_emit

    return run_emit(spec_for_index, build)


if __name__ == "__main__":
    raise SystemExit(main())
