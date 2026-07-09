#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Generic multi-op shape sampler and sharder.

Merge, deduplicate, optionally stratified-sample, and shard shape CSVs for any
rocke op family.  Ops with registered bucketing (conv, gemm) get proportional
stratified sampling that preserves representative coverage across the shape
space.  Ops without registered bucketing (moe, norm, or any new op) get naive
round-robin sharding.

Usage:
    # conv: stratified sample + shard
    python3 -m rocke.heuristics.shard_shapes \\
        --op conv --inputs wide.csv edge.csv \\
        --out all_shapes.csv --target 2000 --shards 32 --shard_dir shards/

    # gemm: stratified sample by aspect/K/scale buckets
    python3 -m rocke.heuristics.shard_shapes \\
        --op gemm --inputs gemm_shapes.csv \\
        --out sampled.csv --target 500 --shards 16 --shard_dir shards/

    # moe/norm/unknown: round-robin shard only (no bucketing)
    python3 -m rocke.heuristics.shard_shapes \\
        --op moe --inputs moe_shapes.csv \\
        --out all.csv --shards 4 --shard_dir shards/
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ── Op shape configuration ──────────────────────────────────────────────────


@dataclass
class OpShapeConfig:
    header: List[str]
    bucket_fn: Optional[Callable[[tuple], tuple]] = None
    extra_columns: Dict[str, str] = field(default_factory=dict)


# ── Conv bucketing ───────────────────────────────────────────────────────────


def _conv_spatial_bucket(Hi: int) -> str:
    if Hi <= 4:
        return "tiny"
    if Hi <= 16:
        return "small"
    if Hi <= 64:
        return "medium"
    return "large"


def _conv_channel_bucket(C: int, K: int) -> str:
    m = min(C, K)
    if m <= 16:
        return "tiny"
    if m <= 64:
        return "small"
    if m <= 256:
        return "medium"
    return "large"


def _conv_group_type(G: int, C: int, K: int) -> str:
    if G == C and G == K:
        return "depthwise"
    if G > 1:
        return "grouped"
    return "standard"


def conv_bucket_key(row: tuple) -> tuple:
    N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw = row
    return (
        (Y, X),
        1 if sh == 1 else 2,
        _conv_group_type(G, C, K),
        _conv_spatial_bucket(Hi),
        _conv_channel_bucket(C, K),
    )



# ── Gemm bucketing ──────────────────────────────────────────────────────────


def _gemm_aspect_bucket(M: int, N: int) -> str:
    if M == 0 or N == 0:
        return "degenerate"
    ratio = math.log2(max(M, 1) / max(N, 1))
    if ratio < -2:
        return "tall"
    if ratio < -0.5:
        return "tallish"
    if ratio <= 0.5:
        return "square"
    if ratio <= 2:
        return "widish"
    return "wide"


def _gemm_k_bucket(K: int) -> str:
    if K <= 16:
        return "tiny"
    if K <= 128:
        return "small"
    if K <= 2048:
        return "medium"
    return "large"


def _gemm_scale_bucket(M: int, N: int) -> str:
    mn = M * N
    if mn <= 2**14:
        return "tiny"
    if mn <= 2**20:
        return "small"
    if mn <= 2**26:
        return "medium"
    return "large"


def gemm_bucket_key(row: tuple) -> tuple:
    M, N, K = row
    return (
        _gemm_aspect_bucket(M, N),
        _gemm_k_bucket(K),
        _gemm_scale_bucket(M, N),
    )


# ── Config registry ─────────────────────────────────────────────────────────


CONV_HEADER = [
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

CONV_SHAPE_COLS = CONV_HEADER[:-1]

GEMM_HEADER = ["M", "N", "K"]


OP_CONFIGS: Dict[str, OpShapeConfig] = {
    "conv": OpShapeConfig(
        header=CONV_HEADER,
        bucket_fn=conv_bucket_key,
        extra_columns={"direction": "forward"},
    ),
    "gemm": OpShapeConfig(
        header=GEMM_HEADER,
        bucket_fn=gemm_bucket_key,
    ),
}


def get_config(op: str) -> Optional[OpShapeConfig]:
    return OP_CONFIGS.get(op)


# ── CSV I/O ──────────────────────────────────────────────────────────────────


def load_csv(path: Path, config: OpShapeConfig) -> List[tuple]:
    """Load shapes from a CSV, parsing only the numeric shape columns."""
    n_shape_cols = len(config.header) - len(config.extra_columns)
    rows: List[tuple] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            if not line:
                continue
            rows.append(tuple(int(v) for v in line[:n_shape_cols]))
    return rows


def load_csv_generic(path: Path) -> Tuple[List[str], List[tuple]]:
    """Load a CSV with unknown schema. Returns (header, rows)."""
    rows: List[tuple] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if not line:
                continue
            vals = []
            for v in line:
                try:
                    vals.append(int(v))
                except ValueError:
                    vals.append(v)
            rows.append(tuple(vals))
    return header, rows


def write_csv(rows: List[tuple], path: Path, config: OpShapeConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(config.header)
        extra_vals = list(config.extra_columns.values())
        for r in rows:
            writer.writerow(list(r) + extra_vals)


def write_csv_generic(rows: List[tuple], path: Path, header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(list(r))


# ── Sampling and sharding ───────────────────────────────────────────────────


def stratified_sample(
    shapes: List[tuple],
    target: int,
    bucket_fn: Optional[Callable[[tuple], tuple]],
    seed: int = 42,
) -> List[tuple]:
    """Proportional stratified sample across buckets.

    When bucket_fn is None, all shapes land in one bucket → random sample.
    """
    rng = random.Random(seed)

    if bucket_fn is None:
        bucket_fn = lambda _: ("all",)

    buckets: Dict[tuple, List[tuple]] = defaultdict(list)
    for s in shapes:
        buckets[bucket_fn(s)].append(s)

    for v in buckets.values():
        rng.shuffle(v)

    total = len(shapes)
    quotas: Dict[tuple, int] = {}
    allocated = 0
    keys = sorted(buckets.keys())
    for k in keys:
        q = max(1, round(len(buckets[k]) / total * target))
        quotas[k] = q
        allocated += q

    delta = target - allocated
    sorted_by_size = sorted(keys, key=lambda k: len(buckets[k]), reverse=True)
    for k in sorted_by_size:
        if delta == 0:
            break
        step = 1 if delta > 0 else -1
        quotas[k] = max(1, quotas[k] + step)
        delta -= step

    selected: List[tuple] = []
    for k in keys:
        bucket = buckets[k]
        q = min(quotas[k], len(bucket))
        selected.extend(bucket[:q])

    rng.shuffle(selected)
    return selected


def shard_shapes(
    shapes: List[tuple],
    num_shards: int,
    bucket_fn: Optional[Callable[[tuple], tuple]] = None,
) -> List[List[tuple]]:
    """Split shapes into num_shards chunks, stratified by bucket when possible."""
    if bucket_fn is None:
        # No bucketing — sequential chunks.
        chunk_size = math.ceil(len(shapes) / num_shards)
        return [
            shapes[i * chunk_size : (i + 1) * chunk_size]
            for i in range(num_shards)
            if shapes[i * chunk_size : (i + 1) * chunk_size]
        ]

    # Group by bucket, then round-robin across shards.
    buckets: Dict[tuple, List[tuple]] = defaultdict(list)
    for s in shapes:
        buckets[bucket_fn(s)].append(s)

    shards: List[List[tuple]] = [[] for _ in range(num_shards)]
    idx = 0
    for _key in sorted(buckets):
        for s in buckets[_key]:
            shards[idx % num_shards].append(s)
            idx += 1

    return [s for s in shards if s]


# ── Stats printing ───────────────────────────────────────────────────────────


def print_stats(
    rows: List[tuple],
    label: str,
    bucket_fn: Optional[Callable[[tuple], tuple]],
) -> None:
    if not bucket_fn:
        print(f"\n{label}: {len(rows)} shapes", file=sys.stderr)
        return

    buckets: Dict[tuple, int] = defaultdict(int)
    for r in rows:
        buckets[bucket_fn(r)] += 1

    print(
        f"\n{label}: {len(rows)} shapes across {len(buckets)} buckets",
        file=sys.stderr,
    )

    # Per-dimension distribution
    dim_counts: Dict[int, Dict[object, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = bucket_fn(r)
        for i, v in enumerate(key):
            dim_counts[i][v] += 1

    for i, counts in sorted(dim_counts.items()):
        print(f"  dim[{i}]: {dict(sorted(counts.items(), key=str))}", file=sys.stderr)


def print_conv_stats(rows: List[tuple], label: str) -> None:
    """Conv-specific stats with named bucket dimensions."""
    filter_counts: Dict[tuple, int] = defaultdict(int)
    stride_counts: Dict[int, int] = defaultdict(int)
    group_counts: Dict[str, int] = defaultdict(int)
    spatial_counts: Dict[str, int] = defaultdict(int)
    channel_counts: Dict[str, int] = defaultdict(int)

    for r in rows:
        N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw = r
        filter_counts[(Y, X)] += 1
        stride_counts[sh] += 1
        group_counts[_conv_group_type(G, C, K)] += 1
        spatial_counts[_conv_spatial_bucket(Hi)] += 1
        channel_counts[_conv_channel_bucket(C, K)] += 1

    buckets: Dict[tuple, int] = defaultdict(int)
    for r in rows:
        buckets[conv_bucket_key(r)] += 1

    print(
        f"\n{label}: {len(rows)} shapes across {len(buckets)} buckets",
        file=sys.stderr,
    )
    print(
        f"  Filters:  { {f'{y}x{x}': n for (y,x),n in sorted(filter_counts.items())} }",
        file=sys.stderr,
    )
    print(f"  Strides:  { dict(sorted(stride_counts.items())) }", file=sys.stderr)
    print(f"  Groups:   { dict(sorted(group_counts.items())) }", file=sys.stderr)
    print(f"  Spatial:  { dict(sorted(spatial_counts.items())) }", file=sys.stderr)
    print(f"  Channels: { dict(sorted(channel_counts.items())) }", file=sys.stderr)


def print_gemm_stats(rows: List[tuple], label: str) -> None:
    """Gemm-specific stats with named bucket dimensions."""
    aspect_counts: Dict[str, int] = defaultdict(int)
    k_counts: Dict[str, int] = defaultdict(int)
    scale_counts: Dict[str, int] = defaultdict(int)

    for r in rows:
        M, N, K = r
        aspect_counts[_gemm_aspect_bucket(M, N)] += 1
        k_counts[_gemm_k_bucket(K)] += 1
        scale_counts[_gemm_scale_bucket(M, N)] += 1

    buckets: Dict[tuple, int] = defaultdict(int)
    for r in rows:
        buckets[gemm_bucket_key(r)] += 1

    print(
        f"\n{label}: {len(rows)} shapes across {len(buckets)} buckets",
        file=sys.stderr,
    )
    print(f"  Aspect:  {dict(sorted(aspect_counts.items()))}", file=sys.stderr)
    print(f"  K-scale: {dict(sorted(k_counts.items()))}", file=sys.stderr)
    print(f"  Output:  {dict(sorted(scale_counts.items()))}", file=sys.stderr)


_STATS_FN = {
    "conv": print_conv_stats,
    "gemm": print_gemm_stats,
}


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge, stratified-sample, and shard shapes for any rocke op."
    )
    parser.add_argument(
        "--op", required=True, help="Op family (conv, gemm, moe, norm, ...)"
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Input shape CSVs")
    parser.add_argument("--out", required=True, help="Output merged CSV")
    parser.add_argument(
        "--target", type=int, default=None, help="Target shape count after sampling"
    )
    shard_group = parser.add_mutually_exclusive_group()
    shard_group.add_argument(
        "--shards",
        type=int,
        default=0,
        help="If >0, write shard_00.csv ... shard_NN.csv",
    )
    shard_group.add_argument(
        "--shapes-per-shard",
        type=int,
        default=None,
        help="Max shapes per shard; computes --shards automatically",
    )
    parser.add_argument(
        "--shard_dir", default="shards", help="Directory for shard CSVs"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config = get_config(args.op)

    # Generic fallback: read header from first input, no bucketing/filtering
    generic_header: Optional[List[str]] = None
    if config is None:
        generic_header, _ = load_csv_generic(Path(args.inputs[0]))
        config = OpShapeConfig(header=generic_header)
        print(
            f"Op '{args.op}' has no registered config; using generic CSV mode",
            file=sys.stderr,
        )

    # Load and deduplicate
    seen: set[tuple] = set()
    all_shapes: List[tuple] = []
    for p in args.inputs:
        rows = load_csv(Path(p), config)
        before = len(seen)
        for r in rows:
            if r not in seen:
                seen.add(r)
                all_shapes.append(r)
        print(
            f"Loaded {p}: {len(rows)} rows, {len(seen) - before} new unique shapes",
            file=sys.stderr,
        )

    print(f"Total unique shapes: {len(all_shapes)}", file=sys.stderr)

    # Stats before sampling
    stats_fn = _STATS_FN.get(
        args.op, lambda rows, label: print_stats(rows, label, config.bucket_fn)
    )
    stats_fn(all_shapes, "Before sampling")

    # Sample
    if args.target is None or len(all_shapes) <= args.target:
        sampled = all_shapes
        print(f"Using all {len(all_shapes)} shapes (no sampling).", file=sys.stderr)
    else:
        sampled = stratified_sample(
            all_shapes, args.target, config.bucket_fn, seed=args.seed
        )

    stats_fn(sampled, "After sampling")

    # Write output
    write_csv(sampled, Path(args.out), config)
    print(f"\nWrote {len(sampled)} shapes to {args.out}", file=sys.stderr)

    # Resolve --shapes-per-shard into a shard count
    if args.shapes_per_shard is not None and args.shapes_per_shard > 0:
        args.shards = math.ceil(len(sampled) / args.shapes_per_shard)
        print(f"shapes_per_shard={args.shapes_per_shard} -> {args.shards} shards", file=sys.stderr)

    # Shard
    if args.shards > 0:
        shard_dir = Path(args.shard_dir)
        chunks = shard_shapes(sampled, args.shards, bucket_fn=config.bucket_fn)
        for i, chunk in enumerate(chunks):
            shard_path = shard_dir / f"shard_{i:02d}.csv"
            write_csv(chunk, shard_path, config)
        print(f"Wrote {len(chunks)} shards to {shard_dir}/", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
