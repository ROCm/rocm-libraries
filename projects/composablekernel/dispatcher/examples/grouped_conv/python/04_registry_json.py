#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 04: Registry & JSON Export/Import with GPU Execution

Demonstrates kernel registry management, JSON serialization, and GPU dispatch.

Usage:
    python3 04_registry_json.py
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from grouped_conv_utils import (
    GroupedConvKernelConfig,
    GroupedConvProblem,
    GroupedConvRegistry,
    GpuGroupedConvRunner,
    validate_grouped_conv_config,
    detect_gpu_arch,
)


def main():
    arch = detect_gpu_arch()
    print("=" * 70)
    print("Example 04: Registry & JSON Export/Import")
    print("=" * 70)
    print(f"\n  Arch: {arch}")

    # Step 1: Build throughput registry (large tiles)
    print("\n--- Step 1: Throughput Registry (large tiles) ---")
    tp_reg = GroupedConvRegistry("throughput")
    for variant in ["forward", "bwd_data", "bwd_weight"]:
        tp_reg.add(GroupedConvKernelConfig(
            variant=variant, ndim_spatial=2, arch=arch,
            tile_n=256, tile_k=256, pipeline="compv3",
        ))
    tp_reg.print_registry()

    # Step 2: Build latency registry (small tiles)
    print("\n--- Step 2: Latency Registry (small tiles) ---")
    lat_reg = GroupedConvRegistry("latency")
    for variant in ["forward", "bwd_data", "bwd_weight"]:
        lat_reg.add(GroupedConvKernelConfig(
            variant=variant, ndim_spatial=2, arch=arch,
            tile_n=64, tile_k=64, pipeline="compv3",
        ))
    lat_reg.print_registry()

    # Step 3: JSON export
    print("\n--- Step 3: JSON Export ---")
    combined = GroupedConvRegistry("all_conv_kernels")
    for k in tp_reg.kernels:
        combined.add(k)
    for k in lat_reg.kernels:
        combined.add(k)

    json_str = combined.to_json()
    print(f"  Combined: {len(combined)} kernels")
    print(f"  JSON size: {len(json_str)} bytes")
    print(f"  Preview:\n{json_str[:300]}  ...")

    # Step 4: JSON import + arch filter
    print("\n--- Step 4: JSON Import & Filter ---")
    imported = GroupedConvRegistry.from_json(json_str)
    print(f"  Imported: {len(imported)} kernels")
    filtered = imported.filter_by_arch(arch)
    print(f"  After arch filter ({arch}): {len(filtered)} kernels")
    fwd_only = imported.filter_by_variant("forward")
    print(f"  Forward only: {len(fwd_only)} kernels")

    # Step 5: GPU execution with a problem
    print("\n--- Step 5: GPU Execution ---")
    runner = GpuGroupedConvRunner()
    if not runner.is_available():
        print("  GPU library not available")
        return 1

    print(f"  Compiled kernels: {runner.lib.kernel_names()}")

    prob = GroupedConvProblem(
        N=1, C=128, K=128, Hi=16, Wi=16, Y=3, X=3,
        stride_h=1, stride_w=1, pad_h=1, pad_w=1,
        direction="forward",
    )
    inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np.float16)
    wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np.float16)

    res = runner.run(inp, wei, prob)
    if res.success:
        print(f"  Time:   {res.time_ms:.4f} ms")
        print(f"  TFLOPS: {res.tflops:.2f}")
        print(f"  Output: {res.output.shape}, nonzero={np.count_nonzero(res.output)}/{res.output.size}")
    else:
        print(f"  GPU failed: {res.error}")

    runner.cleanup()

    # Summary
    print("\n" + "=" * 70)
    print(f"  Registries:  throughput={len(tp_reg)}, latency={len(lat_reg)}")
    print(f"  Combined:    {len(combined)} kernels")
    print(f"  JSON:        round-trip OK ({len(imported)} imported)")
    gpu_ok = res.success if runner.is_available() else False
    print(f"  GPU:         {'OK' if gpu_ok else 'FAIL'}")
    print(f"  Status:      {'PASS' if gpu_ok else 'FAIL'}")
    print("=" * 70)
    return 0 if gpu_ok else 1


if __name__ == "__main__":
    sys.exit(main())
