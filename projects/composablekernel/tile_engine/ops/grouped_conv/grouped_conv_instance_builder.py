#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Grouped Convolution kernel sweep builder for the tile engine.

Expands JSON sweep configs into complete GroupedConvKernelConfig lists,
applying trait-based filtering to control kernel generation.

Usage:
    python grouped_conv_instance_builder.py configs/forward.json --arch gfx950
    python grouped_conv_instance_builder.py configs/receipt0_forward.json --arch gfx950 --list
    python grouped_conv_instance_builder.py configs/forward_ci.json --filter "c.tile_n >= 128"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))

from grouped_conv_utils import GroupedConvKernelConfig  # noqa: E402


# =============================================================================
# Architecture-specific configurations
# =============================================================================

# Data types supported per architecture
ARCH_DTYPES = {
    "gfx950": ["fp16", "bf16", "fp32", "fp8", "bf8", "int8"],
    "gfx942": ["fp16", "bf16", "fp32", "fp8", "bf8", "int8"],
    "gfx90a": ["fp16", "bf16", "fp32"],
    "gfx908": ["fp16", "fp32"],
}

# Common tile configurations used across variants
# Format: (tile_m, tile_n, tile_k)
COMMON_TILES = [
    (1, 64, 64),
    (1, 128, 128),
    (1, 256, 256),
    (2, 128, 128),
    (2, 256, 128),
]

# Wave configurations per tile
# Key: (tile_m, tile_n, tile_k) -> (wave_m, wave_n, wave_k)
TILE_TO_WAVE = {
    (1, 64, 64): (2, 2, 1),
    (1, 128, 128): (2, 2, 1),
    (1, 256, 256): (2, 2, 1),
    (2, 128, 128): (2, 2, 1),
    (2, 256, 128): (2, 2, 1),
}

# Warp tile configurations
# Key: (tile_m, tile_n, tile_k) -> (warp_m, warp_n, warp_k)
TILE_TO_WARP = {
    (1, 64, 64): (32, 32, 16),
    (1, 128, 128): (32, 32, 16),
    (1, 256, 256): (32, 32, 16),
    (2, 128, 128): (32, 32, 16),
    (2, 256, 128): (32, 32, 16),
}

# Vector sizes per tile
# Key: (tile_m, tile_n, tile_k) -> (vec_a, vec_b, vec_c)
TILE_TO_VECTOR = {
    (1, 64, 64): (4, 8, 8),
    (1, 128, 128): (4, 8, 8),
    (1, 256, 256): (4, 8, 8),
    (2, 128, 128): (4, 8, 8),
    (2, 256, 128): (4, 8, 8),
}

# Valid pipelines per variant
VARIANT_PIPELINES = {
    "forward": ["compv3", "compv4"],
    "bwd_data": ["compv3", "mem"],
    "bwd_weight": ["compv3", "mem"],
}

# Valid schedulers
VALID_SCHEDULERS = ["intrawave", "interwave"]

# Valid epilogues
VALID_EPILOGUES = ["cshuffle"]

# Valid layouts
VALID_LAYOUTS = ["nhwgc"]


# =============================================================================
# Helper functions
# =============================================================================


def _get_wave_config(tile: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Get wave configuration for a tile."""
    return TILE_TO_WAVE.get(tile, (2, 2, 1))


def _get_warp_config(tile: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Get warp tile configuration for a tile."""
    return TILE_TO_WARP.get(tile, (32, 32, 16))


def _get_vector_sizes(tile: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Get vector sizes for a tile."""
    return TILE_TO_VECTOR.get(tile, (4, 8, 8))


# =============================================================================
# Sweep expansion
# =============================================================================


def expand_sweep(
    config_path: str, arch: str, ndim_override: int = 0
) -> List[GroupedConvKernelConfig]:
    """Expand JSON sweep config into GroupedConvKernelConfig list.

    The JSON trait_config acts as an allow-list filter: if a trait key
    is present, only the listed values survive. If absent, all values pass.

    This means:
      - receipt0_forward.json (minimal trait_config) -> full kernel set
      - forward_ci.json (restricted to fp16, compv3) -> small subset

    Args:
        config_path: Path to JSON config file
        arch: GPU architecture (e.g., "gfx950")
        ndim_override: If > 0, override ndim_spatial from config

    Returns:
        List of GroupedConvKernelConfig objects
    """
    with open(config_path) as f:
        config = json.load(f)

    variant = config["variant"]
    trait_cfg = config.get("trait_config", {})

    # Build allow-list filters from JSON trait_config
    def _allow(key: str, default=None):
        entry = trait_cfg.get(key)
        if entry is None:
            return default
        return set(entry.get("values", []))

    allowed_dtypes = _allow("data_type")
    allowed_pipelines = _allow("pipeline")
    allowed_schedulers = _allow("scheduler")
    allowed_ndims = _allow("ndim_spatial")

    # Intersect requested dtypes with arch support
    arch_dtypes = set(ARCH_DTYPES.get(arch, ARCH_DTYPES.get("gfx950", [])))
    if allowed_dtypes is not None:
        dtypes = sorted(allowed_dtypes & arch_dtypes)
    else:
        dtypes = sorted(arch_dtypes)

    # Pipelines
    variant_pipes = VARIANT_PIPELINES.get(variant, ["compv3"])
    if allowed_pipelines is not None:
        pipelines = [p for p in variant_pipes if p in allowed_pipelines]
    else:
        pipelines = variant_pipes

    # Schedulers
    if allowed_schedulers is not None:
        schedulers = [s for s in VALID_SCHEDULERS if s in allowed_schedulers]
    else:
        schedulers = VALID_SCHEDULERS

    # Ndim spatial
    if ndim_override > 0:
        ndims = [ndim_override]
    elif allowed_ndims is not None:
        ndims = sorted(allowed_ndims)
    else:
        ndims = [2]  # Default to 2D

    # Epilogues (always cshuffle for now)
    epilogues = VALID_EPILOGUES

    # Layouts (always nhwgc for now)
    layouts = VALID_LAYOUTS

    # Generate all combinations
    configs: List[GroupedConvKernelConfig] = []

    for dtype in dtypes:
        for ndim in ndims:
            for layout in layouts:
                for tile in COMMON_TILES:
                    tile_m, tile_n, tile_k = tile
                    wave_m, wave_n, wave_k = _get_wave_config(tile)
                    warp_m, warp_n, warp_k = _get_warp_config(tile)
                    vec_a, vec_b, vec_c = _get_vector_sizes(tile)

                    for pipeline in pipelines:
                        for scheduler in schedulers:
                            for epilogue in epilogues:
                                configs.append(
                                    GroupedConvKernelConfig(
                                        variant=variant,
                                        ndim_spatial=ndim,
                                        dtype=dtype,
                                        layout=layout,
                                        arch=arch,
                                        tile_m=tile_m,
                                        tile_n=tile_n,
                                        tile_k=tile_k,
                                        wave_m=wave_m,
                                        wave_n=wave_n,
                                        wave_k=wave_k,
                                        warp_tile_m=warp_m,
                                        warp_tile_n=warp_n,
                                        warp_tile_k=warp_k,
                                        pipeline=pipeline,
                                        epilogue=epilogue,
                                        scheduler=scheduler,
                                        vector_size_a=vec_a,
                                        vector_size_b=vec_b,
                                        vector_size_c=vec_c,
                                        pad_m=True,
                                        pad_n=True,
                                        pad_k=True,
                                        block_per_cu=1,
                                        num_wave_groups=1,
                                        num_groups_to_merge=1,
                                    )
                                )

    # Dedup by name (same name = same compiled kernel)
    seen: Set[str] = set()
    unique: List[GroupedConvKernelConfig] = []
    for c in configs:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)

    return unique


def apply_filter(
    configs: List[GroupedConvKernelConfig], expr: str = "", filter_file: str = ""
) -> List[GroupedConvKernelConfig]:
    """Apply user-defined filters to a config list.

    Args:
        expr: Python expression evaluated per config with 'c' as the config.
              Example: "c.tile_n >= 128 and c.pipeline == 'compv4'"
        filter_file: Path to a .py file defining filter_config(c) -> bool.

    Both can be combined (AND logic).
    """
    result = configs

    if filter_file:
        import importlib.util

        spec = importlib.util.spec_from_file_location("user_filter", filter_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "filter_config")
        result = [c for c in result if fn(c)]

    if expr:
        # Developer-only CLI flag -- not user-facing, not exposed via web APIs.
        result = [c for c in result if eval(expr, {"c": c})]  # noqa: S307

    return result


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Grouped Convolution tile engine sweep builder"
    )
    parser.add_argument("config", help="Sweep config JSON")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--ndim", type=int, default=0, help="Override ndim_spatial")
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default="",
        help='Python expression per config, e.g. "c.tile_n >= 128"',
    )
    parser.add_argument(
        "--filter-file",
        default="",
        help="Path to .py file with filter_config(c) -> bool",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument(
        "--export-json",
        type=str,
        default="",
        help="Export kernel configs to JSON file",
    )
    args = parser.parse_args()

    configs = expand_sweep(args.config, args.arch, args.ndim)
    before = len(configs)
    configs = apply_filter(configs, args.filter_expr, args.filter_file)
    filtered = before - len(configs)

    print(
        f"Expanded {args.config} -> {before} configs"
        f"{f' (filtered {filtered}, kept {len(configs)})' if filtered else ''}"
    )

    if args.count_only:
        return

    if args.list:
        for i, c in enumerate(configs):
            print(f"  [{i}] {c.name}")

    if args.export_json:
        export = {
            "metadata": {
                "config_file": args.config,
                "arch": args.arch,
                "count": len(configs),
            },
            "kernels": [c.to_json_obj() for c in configs],
        }
        with open(args.export_json, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\nExported {len(configs)} configs to {args.export_json}")


if __name__ == "__main__":
    main()
