#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Edge-case conv-fwd shape generator.

Analog of generate_edge_dims.py (GEMM N=1 / K=1 cases) for conv:
  - N=1 (single-image inference)
  - K=8 (minimum output channels -- stresses output-tile padding)
  - C=8 (minimum input channels)
  - Tiny spatial: Hi/Wi in {1, 2, 3, 4, 5, 6, 7}
  - 1x1 output (Hi=Y, Wi=X, no padding, stride=1 -- Ho=Wo=1)
  - Filter larger than input (valid only with sufficient padding)
  - Stride larger than typical (stride=3, stride=4)

Output CSV columns (13 fields):
    N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w, direction

All shapes satisfy C % G == 0, K % G == 0, (C/G) % 8 == 0, (K/G) % 8 == 0,
Ho >= 1, Wo >= 1.

Usage:
    python3 generate_edge_dims_conv.py --out edge_dims_conv.csv
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

    channel_vals = [8, 16, 32, 64, 128, 256, 512]
    tiny_hw = [1, 2, 3, 4, 5, 6, 7]
    small_hw = [8, 9, 10, 11, 12, 13, 14, 15, 16]

    # --- 1. N=1 (single-image inference) across a range of shapes ---
    for C in channel_vals:
        for K in channel_vals:
            for hw in [4, 7, 8, 14, 16, 28]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if _valid(1, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((1, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 2. K=8 (minimum output channels -- output-tile padding stress) ---
    K = 8
    for C in [8, 16, 32, 64, 128, 256]:
        for hw in [4, 7, 14, 28, 56]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if _valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 3. C=8 (minimum input channels -- input-tile padding stress) ---
    C = 8
    for K in [8, 16, 32, 64, 128, 256]:
        for hw in [4, 7, 14, 28, 56]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if _valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 4. Tiny spatial (Hi, Wi in 1-7) ---
    for hw in tiny_hw:
        for C in [8, 16, 64, 128]:
            for K in [8, 16, 64, 128]:
                for N in [1, 4]:
                    # 1x1 always valid for tiny hw
                    if _valid(N, 1, C, K, hw, hw, 1, 1, 1, 1, 0, 0):
                        shapes.add((N, 1, C, K, hw, hw, 1, 1, 1, 1, 0, 0))
                    # 3x3 only if hw >= 3
                    if hw >= 3 and _valid(N, 1, C, K, hw, hw, 3, 3, 1, 1, 1, 1):
                        shapes.add((N, 1, C, K, hw, hw, 3, 3, 1, 1, 1, 1))

    # --- 5. Small spatial (8-16) with minimal channels ---
    for hw in small_hw:
        for C in [8, 16, 32]:
            for K in [8, 16, 32]:
                for N in [1, 2]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if _valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                            shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # --- 6. Ho=Wo=1 (output collapses to a single spatial point) ---
    # Achieved by setting Hi=Y, Wi=X, pad=0, stride=1
    for Y in [1, 3, 5, 7]:
        for C in [64, 128, 256]:
            for K in [64, 128, 256]:
                Hi = Y
                Wi = Y
                if _valid(1, 1, C, K, Hi, Wi, Y, Y, 1, 1, 0, 0):
                    shapes.add((1, 1, C, K, Hi, Wi, Y, Y, 1, 1, 0, 0))

    # --- 7. Large stride (stride=3, stride=4) ---
    for stride in [3, 4]:
        for hw in [12, 16, 24, 32, 48]:
            for C, K in [(64, 64), (128, 128), (64, 128)]:
                for N in [1, 4]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if _valid(N, 1, C, K, hw, hw, Y, X, stride, stride, pad_h, pad_w):
                            shapes.add((N, 1, C, K, hw, hw, Y, X, stride, stride, pad_h, pad_w))

    # --- 8. Asymmetric spatial (Hi != Wi) ---
    for Hi, Wi in [(1, 28), (28, 1), (4, 112), (112, 4), (7, 56), (56, 7), (3, 48), (48, 3)]:
        for C, K in [(64, 64), (128, 128)]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if _valid(N, 1, C, K, Hi, Wi, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, Hi, Wi, Y, X, 1, 1, pad_h, pad_w))

    # --- 9. C=K=8 (absolute minimum channels, various spatial) ---
    for hw in [1, 2, 3, 4, 5, 6, 7, 8, 14, 28]:
        for N in [1, 2, 4]:
            if _valid(N, 1, 8, 8, hw, hw, 1, 1, 1, 1, 0, 0):
                shapes.add((N, 1, 8, 8, hw, hw, 1, 1, 1, 1, 0, 0))
            if hw >= 3 and _valid(N, 1, 8, 8, hw, hw, 3, 3, 1, 1, 1, 1):
                shapes.add((N, 1, 8, 8, hw, hw, 3, 3, 1, 1, 1, 1))

    # --- 10. N=1, C=K=8, tiny spatial (combined extremes) ---
    for hw in tiny_hw:
        if _valid(1, 1, 8, 8, hw, hw, 1, 1, 1, 1, 0, 0):
            shapes.add((1, 1, 8, 8, hw, hw, 1, 1, 1, 1, 0, 0))

    return sorted(shapes)


def main():
    parser = argparse.ArgumentParser(description="Generate edge-case conv-fwd shapes")
    parser.add_argument("--out", default="edge_dims_conv.csv", help="Output CSV path")
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

    n1 = sum(1 for s in shapes if s[0] == 1)
    k8 = sum(1 for s in shapes if s[3] == 8)
    c8 = sum(1 for s in shapes if s[2] == 8)
    tiny = sum(1 for s in shapes if s[4] <= 7 and s[5] <= 7)

    print(f"Wrote {len(shapes)} shapes to {out_path}", file=sys.stderr)
    print(f"  N=1: {n1}, K=8: {k8}, C=8: {c8}, tiny spatial (<=7): {tiny}", file=sys.stderr)


if __name__ == "__main__":
    main()
