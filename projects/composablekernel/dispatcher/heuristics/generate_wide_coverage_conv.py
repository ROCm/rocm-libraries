#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Wide-coverage conv-fwd shape generator.

Emits a CSV of 13-tuple grouped-conv forward shapes covering the same
breadth as generate_wide_coverage.py does for GEMM:
  - pow-2 square spatial dims + common channel counts
  - skinny / tall / deep-K ladders (mapped to C/K/spatial)
  - prime spatial dims (worst-case tiling)
  - filter sizes 1x1 / 3x3 / 5x5 / 7x7
  - stride 1 and 2, pad 0 / 1 / 2 / 3
  - LLM-derived channel sizes (attention head widths, FFN widths)
  - grouped convolutions (G=1,2,4,8) and depthwise (G=C=K)

Output CSV columns (13 fields):
    N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w, direction

All shapes satisfy:
    C % G == 0, K % G == 0, (C/G) % 8 == 0, (K/G) % 8 == 0
    Ho = (Hi + 2*pad_h - Y) // stride_h + 1 >= 1
    Wo = (Wi + 2*pad_w - X) // stride_w + 1 >= 1

Usage:
    python3 generate_wide_coverage_conv.py --out wide_coverage.csv
    python3 generate_wide_coverage_conv.py --out wide_coverage.csv --max_shapes 200
"""

import argparse
import csv
import sys
from pathlib import Path

HEADER = ["N", "G", "C", "K", "Hi", "Wi", "Y", "X", "stride_h", "stride_w", "pad_h", "pad_w", "direction"]


def _valid(N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w):
    if C % G != 0 or K % G != 0:
        return False
    if (C // G) % 8 != 0 or (K // G) % 8 != 0:
        return False
    Ho = (Hi + 2 * pad_h - Y) // stride_h + 1
    Wo = (Wi + 2 * pad_w - X) // stride_w + 1
    return Ho >= 1 and Wo >= 1


def generate_shapes():
    shapes = set()

    # --- 1. Pow-2 square spatial + standard channel sweeps ---
    # Mirrors GEMM "square shapes" category
    for log_hw in range(3, 8):  # 8x8 to 128x128
        hw = 2 ** log_hw
        for C in [64, 128, 256, 512]:
            for K in [64, 128, 256, 512]:
                for N in [1, 4, 16]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if _valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                            shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 2. Large channel, small spatial (compute-bound) ---
    # Analog of GEMM "deep K"
    for C in [512, 1024]:
        for K in [512, 1024]:
            for hw in [4, 7, 8, 14, 16]:
                for N in [1, 8, 32]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if _valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                            shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 3. Large spatial, small channel (memory-bound) ---
    # Analog of GEMM "shallow K"
    for C in [64, 128]:
        for K in [64, 128]:
            for hw in [56, 112, 224]:
                for N in [1, 4, 8]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if _valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                            shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 4. Asymmetric C/K (channel expansion / reduction) ---
    for Hi, Wi in [(8, 8), (14, 14), (28, 28), (56, 56)]:
        for C, K in [(64, 256), (128, 512), (256, 64), (512, 128), (256, 1024), (1024, 256)]:
            for N in [1, 4, 16]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if _valid(N, 1, C, K, Hi, Wi, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, Hi, Wi, Y, X, 1, 1, pad_h, pad_w))

    # --- 5. Stride-2 downsampling ---
    for Hi, Wi in [(28, 28), (56, 56), (112, 112)]:
        for C, K in [(64, 128), (128, 256), (256, 512)]:
            for N in [1, 4, 8]:
                # 3x3 stride-2 (standard ResNet transition)
                if _valid(N, 1, C, K, Hi, Wi, 3, 3, 2, 2, 1, 1):
                    shapes.add((N, 1, C, K, Hi, Wi, 3, 3, 2, 2, 1, 1))
                # 1x1 stride-2 projection shortcut
                if _valid(N, 1, C, K, Hi, Wi, 1, 1, 2, 2, 0, 0):
                    shapes.add((N, 1, C, K, Hi, Wi, 1, 1, 2, 2, 0, 0))

    # --- 6. Large filter sizes: 5x5, 7x7 ---
    for Hi, Wi in [(16, 16), (32, 32), (56, 56)]:
        for C, K in [(64, 64), (128, 128), (64, 128)]:
            for N in [1, 4]:
                # 5x5 stride-1 pad-2
                if _valid(N, 1, C, K, Hi, Wi, 5, 5, 1, 1, 2, 2):
                    shapes.add((N, 1, C, K, Hi, Wi, 5, 5, 1, 1, 2, 2))
                # 7x7 stride-1 pad-3 (ResNet stem style)
                if _valid(N, 1, C, K, Hi, Wi, 7, 7, 1, 1, 3, 3):
                    shapes.add((N, 1, C, K, Hi, Wi, 7, 7, 1, 1, 3, 3))
                # 7x7 stride-2 pad-3
                if _valid(N, 1, C, K, Hi, Wi, 7, 7, 2, 2, 3, 3):
                    shapes.add((N, 1, C, K, Hi, Wi, 7, 7, 2, 2, 3, 3))

    # --- 7. Prime spatial dims (worst-case tiling) ---
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47]
    for hw in primes:
        for C, K in [(64, 64), (128, 128), (64, 128)]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if _valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 8. LLM-derived channel widths ---
    # Attention head widths and FFN widths cross-producted with small spatial
    # dims representative of sequence-as-image layouts.
    llm_channels = [64, 128, 256, 512, 1024, 2048, 4096]
    for C in llm_channels:
        for K in llm_channels:
            for hw in [1, 4, 8]:
                for N in [1, 8]:
                    if _valid(N, 1, C, K, hw, hw, 1, 1, 1, 1, 0, 0):
                        shapes.add((N, 1, C, K, hw, hw, 1, 1, 1, 1, 0, 0))

    # --- 9. Grouped convolutions (G > 1) ---
    for G in [2, 4, 8]:
        for base in [16, 32, 64]:
            C = base * G
            K = base * G
            for hw in [14, 28, 56]:
                for N in [1, 4, 8]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if _valid(N, G, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                            shapes.add((N, G, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 10. Depthwise (G = C = K) ---
    for C in [64, 128, 256, 512]:
        for hw in [14, 28, 56, 112]:
            for N in [1, 4, 8]:
                G = C
                K = C
                if _valid(N, G, C, K, hw, hw, 3, 3, 1, 1, 1, 1):
                    shapes.add((N, G, C, K, hw, hw, 3, 3, 1, 1, 1, 1))
                if _valid(N, G, C, K, hw, hw, 3, 3, 2, 2, 1, 1):
                    shapes.add((N, G, C, K, hw, hw, 3, 3, 2, 2, 1, 1))

    # --- 11. Very small batch (inference) ---
    for N in [1, 2]:
        for hw in [7, 14, 28]:
            for C, K in [(64, 128), (128, 256), (256, 512), (512, 1024)]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if _valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 12. Large batch (training) ---
    for N in [64, 128]:
        for hw in [8, 14, 28]:
            for C, K in [(64, 64), (128, 128), (256, 256)]:
                if _valid(N, 1, C, K, hw, hw, 3, 3, 1, 1, 1, 1):
                    shapes.add((N, 1, C, K, hw, hw, 3, 3, 1, 1, 1, 1))

    # --- 13. Non-pow-2 common spatial sizes ---
    for hw in [6, 10, 12, 18, 24, 36, 48, 60, 75, 96]:
        for C, K in [(64, 64), (128, 128), (64, 128)]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if _valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    return sorted(shapes)


def main():
    parser = argparse.ArgumentParser(description="Generate wide-coverage conv-fwd shapes")
    parser.add_argument("--out", default="wide_coverage_conv.csv", help="Output CSV path")
    parser.add_argument("--max_shapes", type=int, default=None, help="Limit shape count (for testing)")
    args = parser.parse_args()

    shapes = generate_shapes()
    if args.max_shapes:
        shapes = shapes[: args.max_shapes]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for s in shapes:
            writer.writerow(list(s) + ["forward"])

    print(f"Wrote {len(shapes)} shapes to {out_path}", file=sys.stderr)

    # breakdown
    by_filter = {}
    for s in shapes:
        key = (s[6], s[7])  # Y, X
        by_filter[key] = by_filter.get(key, 0) + 1
    by_stride = {}
    for s in shapes:
        key = s[8]  # stride_h
        by_stride[key] = by_stride.get(key, 0) + 1
    grouped = sum(1 for s in shapes if s[1] > 1 and s[1] < s[2])
    depthwise = sum(1 for s in shapes if s[1] == s[2] == s[3])

    print(f"  Filter sizes: { {f'{y}x{x}': n for (y,x),n in sorted(by_filter.items())} }", file=sys.stderr)
    print(f"  Stride 1: {by_stride.get(1, 0)}, Stride 2: {by_stride.get(2, 0)}", file=sys.stderr)
    print(f"  Grouped (G>1, G<C): {grouped}, Depthwise (G=C=K): {depthwise}", file=sys.stderr)


if __name__ == "__main__":
    main()
