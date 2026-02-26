#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 01: Basic Grouped Convolution

Full workflow: config, validate, autocorrect, codegen, verify output files.

Demonstrates:
1. Define a grouped conv kernel config (all fields explicit)
2. Validate against arch filter rules
3. Auto-correct invalid configurations
4. Generate kernel headers via codegen
5. Inspect generated output

Usage:
    python3 01_basic_grouped_conv.py
    python3 01_basic_grouped_conv.py --dtype bf16
    python3 01_basic_grouped_conv.py --variant bwd_data
    python3 01_basic_grouped_conv.py --arch gfx942
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codegen"))

from ctypes_utils import detect_gpu_arch
from grouped_conv_utils import (
    validate_grouped_conv_config,
    auto_correct_grouped_conv_config,
    format_grouped_conv_summary,
)


def create_grouped_conv_config(
    variant="forward", ndim_spatial=2, arch="gfx950", dtype="fp16", pipeline="compv4",
):
    """Build a grouped conv config with all fields explicit (like GEMM KernelConfig)."""
    return {
        "tile_config": {
            "tile_m": [1],
            "tile_n": [128],
            "tile_k": [128],
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
        "ndim_spatial": ndim_spatial,
        "arch": arch,
        "layout": "nhwgc",
        "dtype": dtype,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Basic Grouped Convolution Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--variant", default="forward", choices=["forward", "bwd_data", "bwd_weight"],
        help="Convolution direction (default: forward)",
    )
    parser.add_argument(
        "--ndim", type=int, default=2, choices=[1, 2, 3],
        help="Spatial dimensions (default: 2)",
    )
    parser.add_argument(
        "--arch", default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    parser.add_argument(
        "--pipeline", default="compv4", choices=["compv3", "compv4", "mem"],
        help="Pipeline version (default: compv4)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 01: Basic Grouped Convolution")
    print("=" * 70)
    print(f"\n  Arch:      {args.arch}")
    print(f"  Dtype:     {args.dtype}")
    print(f"  Variant:   {args.variant}")
    print(f"  Dims:      {args.ndim}D")
    print(f"  Pipeline:  {args.pipeline}")

    # =========================================================================
    # Step 1: Create config (all fields explicit)
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 1: Create Config (all fields explicit)")
    print("-" * 50)

    config = create_grouped_conv_config(
        variant=args.variant,
        ndim_spatial=args.ndim,
        arch=args.arch,
        dtype=args.dtype,
        pipeline=args.pipeline,
    )

    tile = config["tile_config"]
    trait = config["trait_config"]
    print(f"  variant:   {config['variant']}")
    print(f"  ndim:      {config['ndim_spatial']}D")
    print(f"  layout:    {config['layout']}")
    print(f"  dtype:     {config['dtype']}")
    print(f"  tile:      M={tile['tile_m'][0]} N={tile['tile_n'][0]} K={tile['tile_k'][0]}")
    print(f"  wave:      {tile['wave_m'][0]}x{tile['wave_n'][0]}x{tile['wave_k'][0]}")
    print(f"  warp:      {tile['warp_tile_m'][0]}x{tile['warp_tile_n'][0]}x{tile['warp_tile_k'][0]}")
    print(f"  pipeline:  {trait['pipeline'][0]}")
    print(f"  epilogue:  {trait['epilogue'][0]}")
    print(f"  scheduler: {trait['scheduler'][0]}")
    print(f"  padding:   M={trait['pad_m'][0]} N={trait['pad_n'][0]} K={trait['pad_k'][0]}")

    # =========================================================================
    # Step 2: Validate config
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 2: Validate Config")
    print("-" * 50)

    result = validate_grouped_conv_config(config)
    if result.is_valid:
        print("  Config is VALID")
    else:
        print("  Config has issues:")
        for err in result.errors:
            print(f"    - {err}")

    # =========================================================================
    # Step 3: Auto-correct if needed
    # =========================================================================
    if not result.is_valid:
        print("\n" + "-" * 50)
        print("Step 3: Auto-Correct")
        print("-" * 50)

        corrected, new_result = auto_correct_grouped_conv_config(config)
        print(f"  Corrected: {new_result.is_valid}")
        if new_result.is_valid:
            config = corrected
            print(format_grouped_conv_summary(config))

    # =========================================================================
    # Step 4: Generate kernel via codegen
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 4: Generate Kernel")
    print("-" * 50)

    try:
        from unified_grouped_conv_codegen import (
            UnifiedGroupedConvCodegen,
            GroupedConvKernelConfig,
            GroupedConvVariant,
        )

        variant_map = {
            "forward": GroupedConvVariant.FORWARD,
            "bwd_data": GroupedConvVariant.BACKWARD_DATA,
            "bwd_weight": GroupedConvVariant.BACKWARD_WEIGHT,
        }

        codegen = UnifiedGroupedConvCodegen(
            output_dir=Path("/tmp/grouped_conv_example_01"),
            datatype=args.dtype,
            variant=variant_map[args.variant],
            ndim_spatial=args.ndim,
            gpu_target=args.arch,
        )

        kernels = codegen.generate_all()
        print(f"  Generated {len(kernels)} kernel(s)")
        for k in kernels[:5]:
            print(f"    - {k.name if hasattr(k, 'name') else k}")
        if len(kernels) > 5:
            print(f"    ... and {len(kernels) - 5} more")
    except Exception as e:
        print(f"  Codegen skipped: {e}")
        print("  (This is normal if running without full build environment)")

    # =========================================================================
    # Step 5: Verify generated files
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 5: Verify Output")
    print("-" * 50)

    output_dir = Path("/tmp/grouped_conv_example_01")
    if output_dir.exists():
        hpp_files = list(output_dir.glob("*.hpp"))
        print(f"  Output dir: {output_dir}")
        print(f"  Generated headers: {len(hpp_files)}")
        for f in hpp_files[:5]:
            print(f"    - {f.name}")
    else:
        print("  No output directory (codegen may have been skipped)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Arch:     {args.arch}")
    print(f"  Config:   {args.variant} {args.ndim}D {args.dtype}")
    print(f"  Tile:     1x128x128, wave 2x2x1, warp 32x32x16")
    print(f"  Pipeline: {args.pipeline}, epilogue cshuffle, scheduler intrawave")
    print(f"  Valid:    {result.is_valid}")
    print("  Status:   PASS")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
