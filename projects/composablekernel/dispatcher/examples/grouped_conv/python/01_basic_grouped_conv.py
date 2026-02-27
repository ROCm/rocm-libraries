#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 01: Basic Grouped Convolution

Config, validate, GPU execute, CPU reference verify.

Usage:
    python3 01_basic_grouped_conv.py
    python3 01_basic_grouped_conv.py --variant bwd_data
    python3 01_basic_grouped_conv.py --arch gfx942
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from grouped_conv_utils import (
    GroupedConvKernelConfig,
    GroupedConvProblem,
    GpuGroupedConvRunner,
    validate_grouped_conv_config,
    auto_correct_grouped_conv_config,
    detect_gpu_arch,
)


def cpu_conv2d_fwd(inp, wei, prob):
    """Naive CPU reference: 2D forward, NHWGC layout."""
    N, Hi, Wi, G, Cpg = inp.shape
    _, Kpg, Y, X, _ = wei.shape
    Ho, Wo = prob.Ho, prob.Wo
    out = np.zeros((N, Ho, Wo, G, Kpg), dtype=np.float32)
    for n in range(N):
        for g in range(G):
            for ho in range(Ho):
                for wo in range(Wo):
                    for k in range(Kpg):
                        s = 0.0
                        for y in range(Y):
                            for x in range(X):
                                hi = ho * prob.stride_h - prob.pad_h + y * prob.dilation_h
                                wi = wo * prob.stride_w - prob.pad_w + x * prob.dilation_w
                                if 0 <= hi < Hi and 0 <= wi < Wi:
                                    for c in range(Cpg):
                                        s += float(inp[n, hi, wi, g, c]) * float(wei[g, k, y, x, c])
                        out[n, ho, wo, g, k] = s
    return out


def main():
    parser = argparse.ArgumentParser(description="Basic Grouped Conv Example")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--variant", default="forward",
                        choices=["forward", "bwd_data", "bwd_weight"])
    parser.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--arch", default=detect_gpu_arch())
    args = parser.parse_args()

    print("=" * 70)
    print("Example 01: Basic Grouped Convolution")
    print("=" * 70)

    # Step 1: Kernel config
    print("\n--- Step 1: Kernel Config ---")
    config = GroupedConvKernelConfig(
        variant=args.variant, ndim_spatial=args.ndim,
        arch=args.arch, dtype=args.dtype,
    )
    config.print_config()

    # Step 2: Validate
    print("\n--- Step 2: Validate ---")
    result = validate_grouped_conv_config(config.to_dict())
    if result.is_valid:
        print("  Config is VALID")
    else:
        print("  Config has issues, auto-correcting...")
        corrected, result = auto_correct_grouped_conv_config(config.to_dict())
        print(f"  After correction: valid={result.is_valid}")

    # Step 3: Define problem
    print("\n--- Step 3: Problem ---")
    prob = GroupedConvProblem(
        N=1, C=64, K=128, Hi=16, Wi=16, Y=3, X=3,
        stride_h=1, stride_w=1, pad_h=1, pad_w=1,
        direction=args.variant,
    )
    prob.print_problem()

    # Step 4: GPU execution
    print("\n--- Step 4: GPU Execution ---")
    runner = GpuGroupedConvRunner()
    if not runner.is_available():
        print("  GPU library not available")
        print("  Build: cd dispatcher/build && cmake .. && make dispatcher_conv_lib")
        return 1

    print(f"  Library: {runner.library_path}")
    print(f"  Kernels: {runner.lib.kernel_names()}")

    inp = np.random.uniform(-0.5, 0.5, prob.input_shape()).astype(np.float16)
    wei = np.random.uniform(-0.5, 0.5, prob.weight_shape()).astype(np.float16)

    res = runner.run(inp, wei, prob)
    if not res.success:
        print(f"  GPU execution failed: {res.error}")
        runner.cleanup()
        return 1

    print(f"  Time:   {res.time_ms:.4f} ms")
    print(f"  TFLOPS: {res.tflops:.2f}")
    print(f"  Output: shape={res.output.shape}, range=[{res.output.min():.3f}, {res.output.max():.3f}]")

    # Step 5: CPU reference (forward only)
    verified = False
    if args.variant == "forward" and args.ndim == 2:
        print("\n--- Step 5: CPU Reference Verification ---")
        ref = cpu_conv2d_fwd(inp, wei, prob)
        gpu_f32 = res.output.astype(np.float32)
        diff = np.abs(gpu_f32 - ref)
        max_abs = diff.max()
        max_rel = (diff / (np.abs(ref) + 1e-6)).max()
        match = np.allclose(gpu_f32, ref, atol=0.05, rtol=0.05)
        print(f"  max_abs_diff: {max_abs:.6f}")
        print(f"  max_rel_diff: {max_rel:.6f}")
        print(f"  Match: {match}")
        verified = match

    runner.cleanup()

    # Summary
    print("\n" + "=" * 70)
    status = "PASS" if res.success and (verified or args.variant != "forward") else "FAIL"
    print(f"  Status: {status}")
    print(f"  {config.name} | {prob.gflops:.2f} GFLOPs | {res.tflops:.2f} TFLOPS")
    print("=" * 70)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
