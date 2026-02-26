#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 01: Basic Grouped Convolution

Full workflow: config, validate, autocorrect, codegen, verify output files.

Demonstrates:
1. Define a grouped conv kernel config
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
    get_grouped_conv_default_config,
    format_grouped_conv_summary,
)


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
    # Step 1: Create default config
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 1: Create Default Config")
    print("-" * 50)

    config = get_grouped_conv_default_config(
        variant=args.variant,
        ndim_spatial=args.ndim,
        arch=args.arch,
        dtype=args.dtype,
    )
    config["trait_config"]["pipeline"] = [args.pipeline]

    print(format_grouped_conv_summary(config))

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
    print(f"  Valid:    {result.is_valid}")
    print("  Status:   PASS")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
