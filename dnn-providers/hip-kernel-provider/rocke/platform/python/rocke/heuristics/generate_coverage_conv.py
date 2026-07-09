#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Conv-fwd shape generator covering both wide and edge-case shapes.

Modes:
  wide  -- broad coverage: pow-2 spatial, channel ladders, prime dims, LLM widths,
           grouped/depthwise, stride-2, large filters
  edge  -- edge cases: N=1, K/C=8, tiny spatial, Ho=Wo=1, large strides, asymmetric HW
  all   -- union of both (default)

Output CSV columns (13 fields):
    N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w, direction

All shapes satisfy:
    C % G == 0, K % G == 0, (C/G) % 8 == 0, (K/G) % 8 == 0
    Ho = (Hi + 2*pad_h - Y) // stride_h + 1 >= 1
    Wo = (Wi + 2*pad_w - X) // stride_w + 1 >= 1

Usage:
    python3 generate_coverage_conv.py --out coverage.csv
    python3 generate_coverage_conv.py --mode wide --out wide.csv
    python3 generate_coverage_conv.py --mode edge --out edge.csv --max_shapes 500
"""

import argparse
import csv
import sys
from pathlib import Path

HEADER = [
    "N",
    "G",
    "C",
    "K",
    "Hi",
    "Wi",
    "Y",
    "X",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
    "direction",
]

SHAPE_COLS = HEADER[:-1]


MIN_TILE = 32


def conv_shape_valid(N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w):
    if C % G != 0 or K % G != 0:
        return False
    if (C // G) % 8 != 0 or (K // G) % 8 != 0:
        return False
    Ho = (Hi + 2 * pad_h - Y) // stride_h + 1
    Wo = (Wi + 2 * pad_w - X) // stride_w + 1
    if Ho < 1 or Wo < 1:
        return False
    # Tileability: GEMM dimensions must meet minimum tile size.
    M = N * Ho * Wo
    N_gemm = K
    K_gemm = (C // G) * Y * X
    return M >= MIN_TILE and N_gemm >= MIN_TILE and K_gemm >= MIN_TILE


def generate_wide_shapes():
    """Broad coverage: standard channel/spatial sweeps, grouped, depthwise, LLM widths."""
    shapes = set()

    # 1. Pow-2 square spatial + standard channel sweeps
    for log_hw in range(3, 8):  # 8x8 to 128x128
        hw = 2**log_hw
        for C in [64, 128, 256, 512]:
            for K in [64, 128, 256, 512]:
                for N in [1, 4, 16]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if conv_shape_valid(
                            N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w
                        ):
                            shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 2. Large channel, small spatial (compute-bound)
    for C in [512, 1024]:
        for K in [512, 1024]:
            for hw in [4, 7, 8, 14, 16]:
                for N in [1, 8, 32]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if conv_shape_valid(
                            N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w
                        ):
                            shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 3. Large spatial, small channel (memory-bound)
    for C in [64, 128]:
        for K in [64, 128]:
            for hw in [56, 112, 224]:
                for N in [1, 4, 8]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if conv_shape_valid(
                            N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w
                        ):
                            shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 4. Asymmetric C/K (channel expansion / reduction)
    for Hi, Wi in [(8, 8), (14, 14), (28, 28), (56, 56)]:
        for C, K in [
            (64, 256),
            (128, 512),
            (256, 64),
            (512, 128),
            (256, 1024),
            (1024, 256),
        ]:
            for N in [1, 4, 16]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if conv_shape_valid(N, 1, C, K, Hi, Wi, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, Hi, Wi, Y, X, 1, 1, pad_h, pad_w))

    # 5. Stride-2 downsampling
    for Hi, Wi in [(28, 28), (56, 56), (112, 112)]:
        for C, K in [(64, 128), (128, 256), (256, 512)]:
            for N in [1, 4, 8]:
                if conv_shape_valid(N, 1, C, K, Hi, Wi, 3, 3, 2, 2, 1, 1):
                    shapes.add((N, 1, C, K, Hi, Wi, 3, 3, 2, 2, 1, 1))
                if conv_shape_valid(N, 1, C, K, Hi, Wi, 1, 1, 2, 2, 0, 0):
                    shapes.add((N, 1, C, K, Hi, Wi, 1, 1, 2, 2, 0, 0))

    # 6. Large filter sizes: 5x5, 7x7
    for Hi, Wi in [(16, 16), (32, 32), (56, 56)]:
        for C, K in [(64, 64), (128, 128), (64, 128)]:
            for N in [1, 4]:
                if conv_shape_valid(N, 1, C, K, Hi, Wi, 5, 5, 1, 1, 2, 2):
                    shapes.add((N, 1, C, K, Hi, Wi, 5, 5, 1, 1, 2, 2))
                if conv_shape_valid(N, 1, C, K, Hi, Wi, 7, 7, 1, 1, 3, 3):
                    shapes.add((N, 1, C, K, Hi, Wi, 7, 7, 1, 1, 3, 3))
                if conv_shape_valid(N, 1, C, K, Hi, Wi, 7, 7, 2, 2, 3, 3):
                    shapes.add((N, 1, C, K, Hi, Wi, 7, 7, 2, 2, 3, 3))

    # 7. Prime spatial dims (worst-case tiling)
    for hw in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47]:
        for C, K in [(64, 64), (128, 128), (64, 128)]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if conv_shape_valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 8. LLM-derived channel widths
    for C in [64, 128, 256, 512, 1024, 2048, 4096]:
        for K in [64, 128, 256, 512, 1024, 2048, 4096]:
            for hw in [1, 4, 8]:
                for N in [1, 8]:
                    if conv_shape_valid(N, 1, C, K, hw, hw, 1, 1, 1, 1, 0, 0):
                        shapes.add((N, 1, C, K, hw, hw, 1, 1, 1, 1, 0, 0))

    # 9. Grouped convolutions (G > 1)
    for G in [2, 4, 8]:
        for base in [16, 32, 64]:
            C = base * G
            K = base * G
            for hw in [14, 28, 56]:
                for N in [1, 4, 8]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if conv_shape_valid(
                            N, G, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w
                        ):
                            shapes.add((N, G, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 10. Very small batch (inference)
    for N in [1, 2]:
        for hw in [7, 14, 28]:
            for C, K in [(64, 128), (128, 256), (256, 512), (512, 1024)]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if conv_shape_valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 11. Large batch (training)
    for N in [64, 128]:
        for hw in [8, 14, 28]:
            for C, K in [(64, 64), (128, 128), (256, 256)]:
                if conv_shape_valid(N, 1, C, K, hw, hw, 3, 3, 1, 1, 1, 1):
                    shapes.add((N, 1, C, K, hw, hw, 3, 3, 1, 1, 1, 1))

    # 12. Non-pow-2 common spatial sizes
    for hw in [6, 10, 12, 18, 24, 36, 48, 60, 75, 96]:
        for C, K in [(64, 64), (128, 128), (64, 128)]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if conv_shape_valid(N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    return shapes


def generate_edge_shapes():
    """Edge cases: N=1, minimal channels, tiny spatial, Ho=Wo=1, large strides."""
    shapes = set()

    channel_vals = [8, 16, 32, 64, 128, 256, 512]
    tiny_hw = [1, 2, 3, 4, 5, 6, 7]
    small_hw = [8, 9, 10, 11, 12, 13, 14, 15, 16]

    # 1. N=1 (single-image inference) across a range of shapes
    for C in channel_vals:
        for K in channel_vals:
            for hw in [4, 7, 8, 14, 16, 28]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if conv_shape_valid(1, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((1, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 2. K=8 (minimum output channels -- output-tile padding stress)
    for C in [8, 16, 32, 64, 128, 256]:
        for hw in [4, 7, 14, 28, 56]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if conv_shape_valid(N, 1, C, 8, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, 8, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 3. C=8 (minimum input channels -- input-tile padding stress)
    for K in [8, 16, 32, 64, 128, 256]:
        for hw in [4, 7, 14, 28, 56]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if conv_shape_valid(N, 1, 8, K, hw, hw, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, 8, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 4. Tiny spatial (Hi, Wi in 1-7)
    for hw in tiny_hw:
        for C in [8, 16, 64, 128]:
            for K in [8, 16, 64, 128]:
                for N in [1, 4]:
                    if conv_shape_valid(N, 1, C, K, hw, hw, 1, 1, 1, 1, 0, 0):
                        shapes.add((N, 1, C, K, hw, hw, 1, 1, 1, 1, 0, 0))
                    if hw >= 3 and conv_shape_valid(
                        N, 1, C, K, hw, hw, 3, 3, 1, 1, 1, 1
                    ):
                        shapes.add((N, 1, C, K, hw, hw, 3, 3, 1, 1, 1, 1))

    # 5. Small spatial (8-16) with minimal channels
    for hw in small_hw:
        for C in [8, 16, 32]:
            for K in [8, 16, 32]:
                for N in [1, 2]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if conv_shape_valid(
                            N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w
                        ):
                            shapes.add((N, 1, C, K, hw, hw, Y, X, 1, 1, pad_h, pad_w))

    # 6. Ho=Wo=1 (output collapses to a single spatial point)
    for Y in [1, 3, 5, 7]:
        for C in [64, 128, 256]:
            for K in [64, 128, 256]:
                if conv_shape_valid(1, 1, C, K, Y, Y, Y, Y, 1, 1, 0, 0):
                    shapes.add((1, 1, C, K, Y, Y, Y, Y, 1, 1, 0, 0))

    # 7. Large stride (stride=3, stride=4)
    for stride in [3, 4]:
        for hw in [12, 16, 24, 32, 48]:
            for C, K in [(64, 64), (128, 128), (64, 128)]:
                for N in [1, 4]:
                    for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        if conv_shape_valid(
                            N, 1, C, K, hw, hw, Y, X, stride, stride, pad_h, pad_w
                        ):
                            shapes.add(
                                (N, 1, C, K, hw, hw, Y, X, stride, stride, pad_h, pad_w)
                            )

    # 8. Asymmetric spatial (Hi != Wi)
    for Hi, Wi in [
        (1, 28),
        (28, 1),
        (4, 112),
        (112, 4),
        (7, 56),
        (56, 7),
        (3, 48),
        (48, 3),
    ]:
        for C, K in [(64, 64), (128, 128)]:
            for N in [1, 4]:
                for Y, X, pad_h, pad_w in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    if conv_shape_valid(N, 1, C, K, Hi, Wi, Y, X, 1, 1, pad_h, pad_w):
                        shapes.add((N, 1, C, K, Hi, Wi, Y, X, 1, 1, pad_h, pad_w))

    # 9. C=K=8 (absolute minimum channels, various spatial)
    for hw in [1, 2, 3, 4, 5, 6, 7, 8, 14, 28]:
        for N in [1, 2, 4]:
            if conv_shape_valid(N, 1, 8, 8, hw, hw, 1, 1, 1, 1, 0, 0):
                shapes.add((N, 1, 8, 8, hw, hw, 1, 1, 1, 1, 0, 0))
            if hw >= 3 and conv_shape_valid(N, 1, 8, 8, hw, hw, 3, 3, 1, 1, 1, 1):
                shapes.add((N, 1, 8, 8, hw, hw, 3, 3, 1, 1, 1, 1))

    # 10. N=1, C=K=8, tiny spatial (combined extremes)
    for hw in tiny_hw:
        if conv_shape_valid(1, 1, 8, 8, hw, hw, 1, 1, 1, 1, 0, 0):
            shapes.add((1, 1, 8, 8, hw, hw, 1, 1, 1, 1, 0, 0))

    return shapes


def main():
    parser = argparse.ArgumentParser(
        description="Generate conv-fwd shapes for ML training"
    )
    parser.add_argument(
        "--mode",
        choices=["wide", "edge", "all"],
        default="all",
        help="Shape set to generate (default: all)",
    )
    parser.add_argument(
        "--out", default=None, help="Output CSV path (default: <mode>_conv.csv)"
    )
    parser.add_argument(
        "--max_shapes", type=int, default=None, help="Limit shape count"
    )
    args = parser.parse_args()

    if args.mode == "wide":
        shapes = generate_wide_shapes()
    elif args.mode == "edge":
        shapes = generate_edge_shapes()
    else:
        shapes = generate_wide_shapes() | generate_edge_shapes()

    shapes = sorted(shapes)
    if args.max_shapes:
        shapes = shapes[: args.max_shapes]

    out_path = Path(args.out if args.out else f"{args.mode}_conv.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for s in shapes:
            writer.writerow(list(s) + ["forward"])

    print(f"Wrote {len(shapes)} shapes to {out_path}", file=sys.stderr)

    by_filter = {}
    for s in shapes:
        key = (s[6], s[7])
        by_filter[key] = by_filter.get(key, 0) + 1
    by_stride = {}
    for s in shapes:
        key = s[8]
        by_stride[key] = by_stride.get(key, 0) + 1
    n1 = sum(1 for s in shapes if s[0] == 1)
    grouped = sum(1 for s in shapes if s[1] > 1 and s[1] < s[2])
    depthwise = sum(1 for s in shapes if s[1] == s[2] == s[3])
    tiny = sum(1 for s in shapes if s[4] <= 7 and s[5] <= 7)

    print(
        f"  Filter sizes: { {f'{y}x{x}': n for (y,x),n in sorted(by_filter.items())} }",
        file=sys.stderr,
    )
    print(
        f"  Stride 1: {by_stride.get(1,0)}, Stride 2: {by_stride.get(2,0)}, "
        f"Stride >=3: {sum(v for k,v in by_stride.items() if k >= 3)}",
        file=sys.stderr,
    )
    print(
        f"  N=1: {n1}, Grouped (G>1,G<C): {grouped}, Depthwise (G=C=K): {depthwise}, "
        f"Tiny spatial (<=7): {tiny}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
