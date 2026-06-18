#!/usr/bin/env python3
"""
Generate a non-overlapping validation shape set for the fp16/gfx942 DSL model.

Complements generate_wide_coverage_conv.py and generate_edge_dims_conv.py in
projects/composablekernel/dispatcher/heuristics/ — those define the training
distribution; this script produces shapes outside that set for held-out
evaluation.

Produces two categories:
  - IN-DISTRIBUTION  : parameter ranges covered by the training generators,
                       but specific (N,G,C,K,Hi,Wi,Y,X,stride,pad) tuples not
                       in the training parquet.
  - OUT-OF-DISTRIBUTION: parameter values the training generators never emit
                         (non-pow-2 N, non-pow-2 C/K, large filter on large
                         spatial, very large spatial, asymmetric Hi!=Wi, etc.)

Zero overlap with the training parquet is asserted before writing output.

Usage:
    python3 dnn-providers/ck-dsl-provider/heuristics/scripts/generate_validation_shapes_conv.py \\
        --parquet data/gfx942_fp16_dsl_v2/conv_fp16_gfx942_dsl.parquet \\
        --out     shapes/gfx942_fp16_dsl_validation/all_shapes.csv \\
        --shards  8
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import pandas as pd

HEADER = ["N", "G", "C", "K", "Hi", "Wi", "Y", "X",
          "stride_h", "stride_w", "pad_h", "pad_w", "direction"]


def _valid(N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w, dilation_h=1, dilation_w=1):
    if C % G != 0 or K % G != 0:
        return False
    # Per-group channel counts must be 8-aligned (DSL kernel alignment requirement).
    if (C // G) % 8 != 0 or (K // G) % 8 != 0:
        return False
    eff_Y = (Y - 1) * dilation_h + 1
    eff_X = (X - 1) * dilation_w + 1
    Ho = (Hi + 2 * pad_h - eff_Y) // stride_h + 1
    Wo = (Wi + 2 * pad_w - eff_X) // stride_w + 1
    return Ho >= 1 and Wo >= 1


def load_training_shapes(parquet_path):
    df = pd.read_parquet(parquet_path)
    cols = ["N", "G", "C", "K", "Hi", "Wi", "Y", "X",
            "stride_h", "stride_w", "pad_h", "pad_w"]
    return set(
        tuple(int(v) for v in row)
        for row in df[cols].drop_duplicates().values.tolist()
    )


def generate_in_distribution(training):
    """
    Same parameter ranges as the training generators, different combinations.
    Uses values present in the generators but cross-products not emitted there.
    """
    shapes = set()

    def add(N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw):
        t = (N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw)
        if _valid(*t) and t not in training:
            shapes.add(t)

    # Pow-2 spatial with N values not heavily used in training (2, 8, 32, 64)
    for hw in [8, 16, 32, 64, 128]:
        for C in [64, 128, 256, 512]:
            for K in [64, 128, 256, 512]:
                for N in [2, 8, 32, 64]:
                    for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        add(N, 1, C, K, hw, hw, Y, X, 1, 1, ph, pw)

    # Asymmetric C/K ratios not in training generators
    for Hi, Wi in [(14, 14), (28, 28), (56, 56)]:
        for C, K in [(64, 512), (512, 64), (128, 1024), (1024, 128),
                     (256, 512), (512, 256)]:
            for N in [2, 8]:
                for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    add(N, 1, C, K, Hi, Wi, Y, X, 1, 1, ph, pw)

    # Prime spatial dims with channel sizes not in training
    primes = [7, 11, 13, 17, 19, 23, 29, 31]
    for hw in primes:
        for C, K in [(256, 256), (512, 512), (128, 256), (256, 128)]:
            for N in [2, 8]:
                for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    add(N, 1, C, K, hw, hw, Y, X, 1, 1, ph, pw)

    # Non-pow-2 spatial sizes with channels not in training generators
    for hw in [6, 10, 12, 18, 24, 36, 48, 60, 75, 96]:
        for C, K in [(256, 256), (512, 512), (128, 512), (512, 128)]:
            for N in [2, 8]:
                for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    add(N, 1, C, K, hw, hw, Y, X, 1, 1, ph, pw)

    # Stride-2 with spatial/channel combos not in training stride-2 set
    for Hi, Wi in [(14, 14), (28, 28), (56, 56), (112, 112)]:
        for C, K in [(128, 512), (256, 1024), (64, 256)]:
            for N in [2, 8]:
                add(N, 1, C, K, Hi, Wi, 3, 3, 2, 2, 1, 1)
                add(N, 1, C, K, Hi, Wi, 1, 1, 2, 2, 0, 0)

    # Large batch with shapes not in training large-batch set
    for N in [64, 128]:
        for hw in [7, 14, 56]:
            for C, K in [(128, 256), (256, 512), (64, 128)]:
                add(N, 1, C, K, hw, hw, 3, 3, 1, 1, 1, 1)
                add(N, 1, C, K, hw, hw, 1, 1, 1, 1, 0, 0)

    # Grouped conv combinations not in training (G=2,4,8 with different base sizes)
    for G in [2, 4, 8]:
        for base in [8, 16, 32, 64, 128]:
            C = base * G
            K = base * G
            if (C // G) % 8 != 0 or (K // G) % 8 != 0:
                continue
            for hw in [7, 14, 28, 56]:
                for N in [2, 8]:
                    for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        add(N, G, C, K, hw, hw, Y, X, 1, 1, ph, pw)

    return shapes


def generate_out_of_distribution(training):
    """
    Shapes with parameter values the training generators never emit:
    - N values: 3, 6, 12, 24, 48, 96 (non-pow-2, not in generators)
    - Non-pow-2 C/K not in generators: 96, 160, 192, 320, 384, 640, 768, 1536, 3072
    - Large filter (5x5, 7x7) on large spatial (>56) not in generators
    - Asymmetric Hi!=Wi with larger spatial not in edge generator
    - Very large N (256, 512) training batch sizes
    - Very large spatial (256, 320, 448)
    """
    shapes = set()

    def add(N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw):
        t = (N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw)
        if _valid(*t) and t not in training:
            shapes.add(t)

    # Non-pow-2 N values never in generators
    for N in [3, 6, 12, 24, 48, 96]:
        for hw in [14, 28, 56]:
            for C, K in [(64, 64), (128, 128), (256, 256), (64, 128)]:
                for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    add(N, 1, C, K, hw, hw, Y, X, 1, 1, ph, pw)

    # Non-pow-2 channel sizes never in generators
    ood_channels = [96, 160, 192, 320, 384, 640, 768, 1536, 3072]
    for C in ood_channels:
        for K in [64, 128, 256, 512]:
            for hw in [14, 28]:
                for N in [1, 4, 8]:
                    for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        add(N, 1, C, K, hw, hw, Y, X, 1, 1, ph, pw)
    for K in ood_channels:
        for C in [64, 128, 256, 512]:
            for hw in [14, 28]:
                for N in [1, 4, 8]:
                    for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                        add(N, 1, C, K, hw, hw, Y, X, 1, 1, ph, pw)

    # 5x5 and 7x7 filters on large spatial (>56) — generators cap at Hi=56 for large filters
    for Hi, Wi in [(64, 64), (112, 112), (128, 128)]:
        for C, K in [(64, 64), (128, 128), (64, 128)]:
            for N in [1, 4]:
                add(N, 1, C, K, Hi, Wi, 5, 5, 1, 1, 2, 2)
                add(N, 1, C, K, Hi, Wi, 7, 7, 1, 1, 3, 3)

    # Very large spatial never in generators
    for hw in [256, 320, 448]:
        for C, K in [(32, 32), (64, 64), (32, 64)]:
            for N in [1, 2]:
                for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    add(N, 1, C, K, hw, hw, Y, X, 1, 1, ph, pw)

    # Very large batch never in generators (256, 512)
    for N in [256, 512]:
        for hw in [7, 14]:
            for C, K in [(64, 64), (128, 128)]:
                for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    add(N, 1, C, K, hw, hw, Y, X, 1, 1, ph, pw)

    # Asymmetric Hi!=Wi with larger spatial not in edge generator
    for Hi, Wi in [(3, 224), (224, 3), (7, 112), (112, 7),
                   (14, 224), (224, 14), (28, 112), (112, 28)]:
        for C, K in [(64, 64), (128, 128)]:
            for N in [1, 4]:
                for Y, X, ph, pw in [(1, 1, 0, 0), (3, 3, 1, 1)]:
                    add(N, 1, C, K, Hi, Wi, Y, X, 1, 1, ph, pw)

    # Dilation > 1 — generators never emit dilation != 1
    # Skipped: ConvCandidateSweep uses dilation=1 always; no dilation column in sweep CSV.

    return shapes


def shard(shapes, n_shards, out_dir):
    shapes = sorted(shapes)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = math.ceil(len(shapes) / n_shards)
    for i in range(n_shards):
        shard_shapes = shapes[i * size: (i + 1) * size]
        if not shard_shapes:
            continue
        p = out_dir / f"shard_{i:02d}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for s in shard_shapes:
                w.writerow(list(s) + ["forward"])
    return len(shapes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, type=Path,
                    help="Training parquet to exclude from validation set")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output all_shapes.csv path (includes 'split' column)")
    ap.add_argument("--shards", type=int, default=8,
                    help="Number of shard CSVs to write alongside all_shapes.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print("Loading training shapes...", file=sys.stderr)
    training = load_training_shapes(args.parquet)
    print(f"  {len(training)} training shapes loaded", file=sys.stderr)

    print("Generating in-distribution validation shapes...", file=sys.stderr)
    in_dist = generate_in_distribution(training)
    print(f"  {len(in_dist)} in-distribution shapes (non-overlapping)", file=sys.stderr)

    print("Generating out-of-distribution validation shapes...", file=sys.stderr)
    out_dist = generate_out_of_distribution(training)
    print(f"  {len(out_dist)} out-of-distribution shapes", file=sys.stderr)

    all_shapes = sorted(in_dist | out_dist)
    print(f"  {len(all_shapes)} total validation shapes", file=sys.stderr)

    # Verify zero overlap with training
    overlap = set(all_shapes) & training
    assert len(overlap) == 0, f"BUG: {len(overlap)} shapes overlap with training set"

    # Write all_shapes.csv with a 'split' column for analysis
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    in_set = in_dist
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER + ["split"])
        for s in all_shapes:
            split = "in_distribution" if s in in_set else "out_of_distribution"
            w.writerow(list(s) + ["forward", split])

    print(f"Wrote {len(all_shapes)} shapes to {out_path}", file=sys.stderr)

    # Write shards without split column — ConvCandidateSweep expects standard CSV format
    shard_dir = out_path.parent
    shard(all_shapes, args.shards, shard_dir)
    print(f"Wrote {args.shards} shards to {shard_dir}/", file=sys.stderr)

    in_count = len(in_dist)
    out_count = len(out_dist)
    print(f"\nSummary:", file=sys.stderr)
    print(f"  In-distribution    : {in_count} shapes", file=sys.stderr)
    print(f"  Out-of-distribution: {out_count} shapes", file=sys.stderr)
    print(f"  Total              : {len(all_shapes)} shapes", file=sys.stderr)
    print(f"  Shards             : {args.shards}", file=sys.stderr)


if __name__ == "__main__":
    main()
