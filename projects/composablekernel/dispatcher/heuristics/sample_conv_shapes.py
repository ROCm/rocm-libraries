#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Merge, deduplicate, and stratified-sample conv-fwd shape CSVs.

Reads one or more shape CSVs produced by generate_wide_coverage_conv.py and
generate_edge_dims_conv.py, deduplicates, buckets shapes by coverage
dimensions, then round-robins across buckets to hit a target count.  This
preserves representative coverage across all axes rather than accidentally
dropping entire corners by hand-thinning generator ranges.

Bucket key: (filter_size, stride_cat, group_type, spatial_bucket, channel_bucket)
  filter_size  : (Y, X)                           -- 1x1 / 3x3 / 5x5 / 7x7 / other
  stride_cat   : 1 | 2+
  group_type   : "standard" | "grouped" | "depthwise"
  spatial_bucket: tiny (Hi<=4) | small (5-16) | medium (17-64) | large (65+)
  channel_bucket: tiny (min(C,K)<=16) | small (17-64) | medium (65-256) | large (257+)

Usage:
    # merge wide + edge, sample to 2000
    python3 sample_conv_shapes.py \\
        --inputs wide_coverage_conv.csv edge_dims_conv.csv \\
        --out all_shapes.csv --target 2000

    # also write per-shard CSVs for the slurm array job
    python3 sample_conv_shapes.py \\
        --inputs wide_coverage_conv.csv edge_dims_conv.csv \\
        --out all_shapes.csv --target 2000 --shards 32 --shard_dir shards/
"""

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HEADER = ["N", "G", "C", "K", "Hi", "Wi", "Y", "X",
          "stride_h", "stride_w", "pad_h", "pad_w", "direction"]


def _spatial_bucket(Hi: int) -> str:
    if Hi <= 4:
        return "tiny"
    if Hi <= 16:
        return "small"
    if Hi <= 64:
        return "medium"
    return "large"


def _channel_bucket(C: int, K: int) -> str:
    m = min(C, K)
    if m <= 16:
        return "tiny"
    if m <= 64:
        return "small"
    if m <= 256:
        return "medium"
    return "large"


def _group_type(G: int, C: int, K: int) -> str:
    if G == C and G == K:
        return "depthwise"
    if G > 1:
        return "grouped"
    return "standard"


def bucket_key(row: tuple) -> tuple:
    N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw = row
    return (
        (Y, X),
        1 if sh == 1 else 2,
        _group_type(G, C, K),
        _spatial_bucket(Hi),
        _channel_bucket(C, K),
    )


def load_csv(path: Path) -> list[tuple]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if not line:
                continue
            N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw, direction = line
            rows.append((int(N), int(G), int(C), int(K),
                         int(Hi), int(Wi), int(Y), int(X),
                         int(sh), int(sw), int(ph), int(pw)))
    return rows


def stratified_sample(shapes: list[tuple], target: int, seed: int = 42) -> list[tuple]:
    """Round-robin across buckets until target is reached."""
    rng = random.Random(seed)

    buckets: dict[tuple, list[tuple]] = defaultdict(list)
    for s in shapes:
        buckets[bucket_key(s)].append(s)

    # Shuffle within each bucket so round-robin gives variety
    for v in buckets.values():
        rng.shuffle(v)

    # Compute per-bucket quota proportional to bucket size, minimum 1
    total = len(shapes)
    quotas: dict[tuple, int] = {}
    allocated = 0
    keys = sorted(buckets.keys())
    for k in keys:
        q = max(1, round(len(buckets[k]) / total * target))
        quotas[k] = q
        allocated += q

    # Adjust largest bucket(s) to hit target exactly
    delta = target - allocated
    sorted_by_size = sorted(keys, key=lambda k: len(buckets[k]), reverse=True)
    for k in sorted_by_size:
        if delta == 0:
            break
        step = 1 if delta > 0 else -1
        quotas[k] = max(1, quotas[k] + step)
        delta -= step

    selected = []
    for k in keys:
        bucket = buckets[k]
        q = min(quotas[k], len(bucket))
        selected.extend(bucket[:q])

    rng.shuffle(selected)
    return selected


def write_csv(rows: list[tuple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for r in rows:
            writer.writerow(list(r) + ["forward"])


def print_stats(rows: list[tuple], label: str) -> None:
    buckets: dict[tuple, int] = defaultdict(int)
    for r in rows:
        buckets[bucket_key(r)] += 1

    filter_counts: dict[tuple, int] = defaultdict(int)
    stride_counts: dict[int, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    spatial_counts: dict[str, int] = defaultdict(int)
    channel_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw = r
        filter_counts[(Y, X)] += 1
        stride_counts[sh] += 1
        group_counts[_group_type(G, C, K)] += 1
        spatial_counts[_spatial_bucket(Hi)] += 1
        channel_counts[_channel_bucket(C, K)] += 1

    print(f"\n{label}: {len(rows)} shapes across {len(buckets)} buckets",
          file=sys.stderr)
    print(f"  Filters:  { {f'{y}x{x}': n for (y,x),n in sorted(filter_counts.items())} }",
          file=sys.stderr)
    print(f"  Strides:  { dict(sorted(stride_counts.items())) }", file=sys.stderr)
    print(f"  Groups:   { dict(sorted(group_counts.items())) }", file=sys.stderr)
    print(f"  Spatial:  { dict(sorted(spatial_counts.items())) }", file=sys.stderr)
    print(f"  Channels: { dict(sorted(channel_counts.items())) }", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Merge and stratified-sample conv shapes")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input CSVs (wide_coverage_conv.csv, edge_dims_conv.csv, ...)")
    parser.add_argument("--out", required=True, help="Output merged CSV (all_shapes.csv)")
    parser.add_argument("--target", type=int, default=None,
                        help="Target shape count after sampling (default: use all shapes)")
    parser.add_argument("--shards", type=int, default=0,
                        help="If >0, also write shard_00.csv ... shard_NN.csv")
    parser.add_argument("--shard_dir", default="shards",
                        help="Directory for shard CSVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tile", type=int, default=32,
                        help="Minimum GEMM dimension (M, N, K_gemm) required for a shape to be "
                             "tileable. Matches the smallest entry in the sweep's tile-size set "
                             "(kTileSizes[] in ConvCandidateSweep.cpp). Default: 32")
    args = parser.parse_args()

    # Load and deduplicate
    seen: set[tuple] = set()
    all_shapes: list[tuple] = []
    for p in args.inputs:
        rows = load_csv(Path(p))
        before = len(seen)
        for r in rows:
            if r not in seen:
                seen.add(r)
                all_shapes.append(r)
        print(f"Loaded {p}: {len(rows)} rows, "
              f"{len(seen) - before} new unique shapes", file=sys.stderr)

    print(f"Total unique shapes before sampling: {len(all_shapes)}", file=sys.stderr)

    # Drop shapes where any GEMM dimension is smaller than the sweep's minimum tile size.
    #   M      = N * Ho * Wo   (output pixels per image)
    #   N_gemm = K             (output channels)
    #   K_gemm = (C/G) * R * S (input channels × filter area, per group)
    min_tile = args.min_tile
    tileable = []
    for r in all_shapes:
        N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw = r
        Ho = (Hi + 2 * ph - Y) // sh + 1
        Wo = (Wi + 2 * pw - X) // sw + 1
        M      = N * Ho * Wo
        N_gemm = K
        K_gemm = (C // G) * Y * X
        if M >= min_tile and N_gemm >= min_tile and K_gemm >= min_tile:
            tileable.append(r)
    n_dropped = len(all_shapes) - len(tileable)
    print(f"Dropped {n_dropped} untileable shapes (M/N/K_gemm < {min_tile}); "
          f"{len(tileable)} remain", file=sys.stderr)
    all_shapes = tileable

    print_stats(all_shapes, "Before sampling")

    if args.target is None or len(all_shapes) <= args.target:
        sampled = all_shapes
        print(f"Using all {len(all_shapes)} shapes (no sampling).", file=sys.stderr)
    else:
        sampled = stratified_sample(all_shapes, args.target, seed=args.seed)

    print_stats(sampled, "After sampling")

    write_csv(sampled, Path(args.out))
    print(f"\nWrote {len(sampled)} shapes to {args.out}", file=sys.stderr)

    if args.shards > 0:
        shard_dir = Path(args.shard_dir)
        n = len(sampled)
        shard_size = math.ceil(n / args.shards)
        for i in range(args.shards):
            chunk = sampled[i * shard_size: (i + 1) * shard_size]
            if not chunk:
                break
            shard_path = shard_dir / f"shard_{i:02d}.csv"
            write_csv(chunk, shard_path)
        actual_shards = min(args.shards, math.ceil(n / shard_size))
        print(f"Wrote {actual_shards} shards to {shard_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
