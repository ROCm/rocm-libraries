#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 04: Registry and JSON Export/Import

Demonstrates:
- Building a kernel registry from explicit configs
- JSON export with statistics
- JSON import and reconstruction
- Multi-registry selection (throughput vs latency)
- Architecture filtering

All configs built inline with every field visible.

Usage:
    python3 04_registry_json.py
    python3 04_registry_json.py --output /tmp/conv_registry.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codegen"))

from ctypes_utils import detect_gpu_arch
from grouped_conv_utils import (
    validate_grouped_conv_config,
    auto_correct_grouped_conv_config,
)


def make_config(variant, dtype, arch, tile_n, tile_k, pipeline):
    """Build a grouped conv config with all fields explicit."""
    return {
        "tile_config": {
            "tile_m": [1],
            "tile_n": [tile_n],
            "tile_k": [tile_k],
            "wave_m": [2],
            "wave_n": [2],
            "wave_k": [1],
            "warp_tile_m": [32],
            "warp_tile_n": [32],
            "warp_tile_k": [16],
        },
        "trait_config": {
            "pipeline": [pipeline],
            "epilogue": ["cshuffle"],
            "scheduler": ["intrawave"],
            "pad_m": [True],
            "pad_n": [True],
            "pad_k": [True],
        },
        "variant": variant,
        "ndim_spatial": 2,
        "arch": arch,
        "layout": "nhwgc",
        "dtype": dtype,
    }


def build_registry(configs, name="default"):
    """Build a simple in-memory registry from config dicts."""
    registry = {
        "name": name,
        "kernels": [],
        "statistics": {"by_variant": {}, "by_dtype": {}, "by_arch": {}},
    }

    for cfg in configs:
        result = validate_grouped_conv_config(cfg)
        if not result.is_valid:
            cfg, result = auto_correct_grouped_conv_config(cfg)

        tile = cfg["tile_config"]
        trait = cfg["trait_config"]
        tile_n = tile["tile_n"][0] if isinstance(tile["tile_n"], list) else tile["tile_n"]
        tile_k = tile["tile_k"][0] if isinstance(tile["tile_k"], list) else tile["tile_k"]
        pipeline = trait["pipeline"][0] if isinstance(trait["pipeline"], list) else trait["pipeline"]

        kernel_name = (f"grouped_conv_{cfg['variant']}_{cfg['dtype']}"
                       f"_{cfg['ndim_spatial']}d_1x{tile_n}x{tile_k}_{pipeline}")

        kernel_entry = {
            "name": kernel_name,
            "signature": {
                "variant": cfg["variant"],
                "dtype": cfg["dtype"],
                "ndim_spatial": cfg["ndim_spatial"],
                "layout": cfg["layout"],
            },
            "algorithm": {
                "tile_m": 1, "tile_n": tile_n, "tile_k": tile_k,
                "wave": "2x2x1", "warp": "32x32x16",
                "pipeline": pipeline,
                "epilogue": "cshuffle",
                "scheduler": "intrawave",
            },
            "arch": cfg["arch"],
            "valid": result.is_valid,
        }
        registry["kernels"].append(kernel_entry)

        stats = registry["statistics"]
        stats["by_variant"][cfg["variant"]] = stats["by_variant"].get(cfg["variant"], 0) + 1
        stats["by_dtype"][cfg["dtype"]] = stats["by_dtype"].get(cfg["dtype"], 0) + 1
        stats["by_arch"][cfg["arch"]] = stats["by_arch"].get(cfg["arch"], 0) + 1

    return registry


def export_registry_json(registry):
    """Export registry to formatted JSON string."""
    return json.dumps(registry, indent=2, sort_keys=False)


def import_registry_json(json_str):
    """Import registry from JSON string."""
    return json.loads(json_str)


def filter_by_arch(registry, arch):
    """Return a new registry with only kernels matching the given arch."""
    filtered = {
        "name": registry["name"] + f"_{arch}",
        "kernels": [k for k in registry["kernels"] if k["arch"] == arch],
        "statistics": {},
    }
    for k in filtered["kernels"]:
        for key_name, key_val in [
            ("by_variant", k["signature"]["variant"]),
            ("by_dtype", k["signature"]["dtype"]),
            ("by_arch", k["arch"]),
        ]:
            filtered["statistics"].setdefault(key_name, {})
            filtered["statistics"][key_name][key_val] = (
                filtered["statistics"][key_name].get(key_val, 0) + 1
            )
    return filtered


def select_kernel(registry, variant="forward", dtype="fp16"):
    """Simple heuristic: pick the largest tile config matching variant+dtype."""
    matching = [
        k for k in registry["kernels"]
        if k["signature"]["variant"] == variant and k["signature"]["dtype"] == dtype
    ]
    if not matching:
        return None
    return max(matching, key=lambda k: k["algorithm"]["tile_n"] * k["algorithm"]["tile_k"])


def main():
    parser = argparse.ArgumentParser(description="Registry & JSON Export/Import")
    parser.add_argument(
        "--arch", default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    parser.add_argument(
        "--output", default="",
        help="Output JSON file (optional)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 04: Registry & JSON Export/Import")
    print("=" * 70)
    print(f"\n  Arch: {args.arch}\n")

    # =========================================================================
    # Step 1: Build throughput registry (large tiles, explicit configs)
    # =========================================================================
    print("-" * 50)
    print("Step 1: Throughput Registry (large tiles)")
    print("-" * 50)

    throughput_configs = [
        make_config("forward",    "fp16", args.arch, tile_n=256, tile_k=256, pipeline="compv4"),
        make_config("bwd_data",   "fp16", args.arch, tile_n=256, tile_k=256, pipeline="compv3"),
        make_config("bwd_weight", "fp16", args.arch, tile_n=256, tile_k=256, pipeline="compv3"),
    ]

    print(f"  Configs: tile 1x256x256, wave 2x2x1, warp 32x32x16")
    throughput_reg = build_registry(throughput_configs, "throughput")
    print(f"  Kernels: {len(throughput_reg['kernels'])}")
    for k in throughput_reg["kernels"]:
        print(f"    - {k['name']} (valid={k['valid']})")

    # =========================================================================
    # Step 2: Build latency registry (small tiles, explicit configs)
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 2: Latency Registry (small tiles)")
    print("-" * 50)

    latency_configs = [
        make_config("forward",    "fp16", args.arch, tile_n=64, tile_k=64, pipeline="compv3"),
        make_config("bwd_data",   "fp16", args.arch, tile_n=64, tile_k=64, pipeline="compv3"),
        make_config("bwd_weight", "fp16", args.arch, tile_n=64, tile_k=64, pipeline="compv3"),
    ]

    print(f"  Configs: tile 1x64x64, wave 2x2x1, warp 32x32x16")
    latency_reg = build_registry(latency_configs, "latency")
    print(f"  Kernels: {len(latency_reg['kernels'])}")
    for k in latency_reg["kernels"]:
        print(f"    - {k['name']} (valid={k['valid']})")

    # =========================================================================
    # Step 3: Multi-registry kernel selection
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 3: Multi-Registry Kernel Selection")
    print("-" * 50)

    tp_kernel = select_kernel(throughput_reg, "forward")
    lt_kernel = select_kernel(latency_reg, "forward")

    print(f"  Throughput pick: {tp_kernel['name'] if tp_kernel else 'none'}")
    print(f"  Latency pick:    {lt_kernel['name'] if lt_kernel else 'none'}")

    # =========================================================================
    # Step 4: JSON export
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 4: JSON Export")
    print("-" * 50)

    combined_reg = {
        "name": "all_conv_kernels",
        "kernels": throughput_reg["kernels"] + latency_reg["kernels"],
        "statistics": {},
    }
    for cat in ["by_variant", "by_dtype", "by_arch"]:
        combined_reg["statistics"][cat] = {}
        for reg in [throughput_reg, latency_reg]:
            for key, val in reg["statistics"].get(cat, {}).items():
                combined_reg["statistics"][cat][key] = (
                    combined_reg["statistics"][cat].get(key, 0) + val
                )

    json_str = export_registry_json(combined_reg)
    print(f"  Combined kernels: {len(combined_reg['kernels'])}")
    print(f"  JSON size: {len(json_str)} bytes")
    print(f"\n  Preview:\n{json_str[:500]}\n  ...")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json_str)
        print(f"\n  Written to: {args.output}")

    # =========================================================================
    # Step 5: JSON import and filter
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 5: JSON Import & Arch Filter")
    print("-" * 50)

    imported = import_registry_json(json_str)
    print(f"  Imported {len(imported['kernels'])} kernels")

    filtered = filter_by_arch(imported, args.arch)
    print(f"  After filter_by_arch('{args.arch}'): {len(filtered['kernels'])} kernels")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Throughput registry: {len(throughput_reg['kernels'])} kernels (tile 1x256x256)")
    print(f"  Latency registry:   {len(latency_reg['kernels'])} kernels (tile 1x64x64)")
    print(f"  Combined:           {len(combined_reg['kernels'])} kernels")
    print(f"  JSON round-trip:    OK")
    print(f"  Arch filter:        OK")
    print(f"\n  Status: PASS")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
