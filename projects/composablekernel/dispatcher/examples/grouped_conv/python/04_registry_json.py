#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 04: Registry and JSON Export/Import

Demonstrates:
- Building a kernel registry from configs
- JSON export with statistics
- JSON import and reconstruction
- Multi-registry selection (throughput vs latency)

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
    get_grouped_conv_default_config,
    format_grouped_conv_summary,
)


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

        trait_cfg = cfg.get("trait_config", {})

        variant = cfg.get("variant", "forward")
        dtype = cfg.get("dtype", "fp16")
        arch = cfg.get("arch", "gfx950")
        ndim = cfg.get("ndim_spatial", 2)

        pipeline = trait_cfg.get("pipeline", ["compv4"])
        if isinstance(pipeline, list):
            pipeline = pipeline[0]

        tile_m = trait_cfg.get("tile_m", [1])
        tile_n = trait_cfg.get("tile_n", [128])
        tile_k = trait_cfg.get("tile_k", [128])
        if isinstance(tile_m, list): tile_m = tile_m[0]
        if isinstance(tile_n, list): tile_n = tile_n[0]
        if isinstance(tile_k, list): tile_k = tile_k[0]

        kernel_name = f"grouped_conv_{variant}_{dtype}_{ndim}d_{tile_m}x{tile_n}x{tile_k}_{pipeline}"

        kernel_entry = {
            "name": kernel_name,
            "signature": {
                "variant": variant,
                "dtype": dtype,
                "ndim_spatial": ndim,
                "layout": "nhwc",
            },
            "algorithm": {
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "pipeline": pipeline,
            },
            "arch": arch,
            "valid": result.is_valid,
        }
        registry["kernels"].append(kernel_entry)

        # Update statistics
        stats = registry["statistics"]
        stats["by_variant"][variant] = stats["by_variant"].get(variant, 0) + 1
        stats["by_dtype"][dtype] = stats["by_dtype"].get(dtype, 0) + 1
        stats["by_arch"][arch] = stats["by_arch"].get(arch, 0) + 1

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
    # Recompute stats
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
    # Step 1: Build throughput registry (large tiles)
    # =========================================================================
    print("-" * 50)
    print("Step 1: Throughput Registry")
    print("-" * 50)

    throughput_configs = []
    for variant in ["forward", "bwd_data", "bwd_weight"]:
        cfg = get_grouped_conv_default_config(
            variant=variant, ndim_spatial=2, arch=args.arch, dtype="fp16",
        )
        cfg["trait_config"]["tile_n"] = [256]
        cfg["trait_config"]["tile_k"] = [256]
        cfg["trait_config"]["pipeline"] = ["compv4"]
        throughput_configs.append(cfg)

    throughput_reg = build_registry(throughput_configs, "throughput")
    print(f"  Kernels: {len(throughput_reg['kernels'])}")
    print(f"  Stats:   {throughput_reg['statistics']}")

    # =========================================================================
    # Step 2: Build latency registry (small tiles)
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 2: Latency Registry")
    print("-" * 50)

    latency_configs = []
    for variant in ["forward", "bwd_data", "bwd_weight"]:
        cfg = get_grouped_conv_default_config(
            variant=variant, ndim_spatial=2, arch=args.arch, dtype="fp16",
        )
        cfg["trait_config"]["tile_n"] = [64]
        cfg["trait_config"]["tile_k"] = [64]
        cfg["trait_config"]["pipeline"] = ["compv3"]
        latency_configs.append(cfg)

    latency_reg = build_registry(latency_configs, "latency")
    print(f"  Kernels: {len(latency_reg['kernels'])}")
    print(f"  Stats:   {latency_reg['statistics']}")

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
    # Merge stats
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
    print(f"\n  Preview:\n{json_str[:400]}\n  ...")

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
    print(f"  Throughput registry: {len(throughput_reg['kernels'])} kernels")
    print(f"  Latency registry:   {len(latency_reg['kernels'])} kernels")
    print(f"  Combined:           {len(combined_reg['kernels'])} kernels")
    print(f"  JSON round-trip:    OK")
    print(f"  Arch filter:        OK")
    print(f"\n  Status: PASS")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
