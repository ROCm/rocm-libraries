#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx950_attention_tiled_2d_fastkv_regp_emit.py -- Python reference
# emitter for the experimental gfx950 "fast paged-KV descriptor + register-P"
# tiled-2D unified-attention kernel parity harness.
#
# Selects one of the sampled configs by argv[1], builds a
# UnifiedAttention2DTiledSpec with the experiment knobs, emits the kernel via
# build_unified_attention_2d_fastkv_register_p(spec, arch="gfx950") and prints
# lower_kernel_to_llvm(kernel, arch="gfx950") to stdout so it can be
# byte-compared with the C emitter gfx950_attention_tiled_2d_fastkv_regp_emit.c.
from rocke.instances.gfx950.attention_tiled_2d import UnifiedAttention2DTiledSpec
from rocke.instances.gfx950.attention_tiled_2d_fastkv_regp import (
    build_unified_attention_2d_fastkv_register_p,
)
from _emit_common import run_emit


# Shared base for the bf16 d64_b32_h64kv8 / T=64 / num_warps=4 experiment family.
_BASE = dict(
    head_size=64,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_warps=4,
    waves_per_eu=2,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_fast_paged_kv_desc=True,
)


_CONFIGS = {
    0: dict(_BASE),
    1: dict(_BASE, use_transposed_half_local_pv=True),
    2: dict(_BASE, use_mfma32_skip_legacy_qreg=True),
    3: dict(_BASE, use_transposed_half_local_pv=True, use_mfma32_skip_legacy_qreg=True),
    4: dict(_BASE, use_agpr_alloc_zero=True, use_transposed_half_local_pv=True),
    5: dict(_BASE, use_grouped_kv2_softmax=True),
}


def _kernel(idx: int):
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    spec = UnifiedAttention2DTiledSpec(**_CONFIGS[idx])
    return build_unified_attention_2d_fastkv_register_p(spec, arch="gfx950")


def _emit_build(kernel, arch=None):
    return kernel


def main() -> int:
    return run_emit(
        _kernel,
        _emit_build,
        usage="usage: gfx950_attention_tiled_2d_fastkv_regp_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
