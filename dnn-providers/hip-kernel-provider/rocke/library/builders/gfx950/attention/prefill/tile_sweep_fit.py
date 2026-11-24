# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fits a tile-selection rule to the gfx950 dense sweep, and says how well it fits.

``tile_sweep.py`` times every legal tile per shape. This reads those rows and
answers three questions in order:

1. **Is there anything to win?** The current scorer is shape-blind: it prefers
   256x64 wherever that tile is legal. If its geomean against the best per-shape
   tile is already near 1.0, the heuristic buys nothing and that is the finding.
2. **What separates the winners?** The winning tile is reported against the
   features a rule could read: grid fill against CU count, head size, dtype,
   mask, GQA ratio.
3. **How good is a candidate rule?** Each rule is scored the way the acceptance
   criterion is written: geomean of (rule's pick / best per shape), and the worst
   single shape against the current default.

Ratios only in the report. The sweep's absolute times stay in its JSONL for
analysis and do not belong in anything published (``platform/AGENTS.md``,
Compliance).

A rule here is a plain function ``(shape, tiles) -> tile``, so adding a
hypothesis is adding a function to ``RULES`` rather than editing the machinery.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

# The gfx950 part the sweep runs on. Only used when a row carries no CU count.
DEFAULT_CU_COUNT = 256

# What the shipped scorer prefers, from Gfx950AttentionDenseNative.cpp.
BASELINE_TILE = (256, 64)


def load(path):
    """Rows the fit may use: timed, and passing their own correctness check.

    A row that errored or failed the reference check is dropped and counted, so a
    sweep that silently lost half its configs cannot look like a clean fit.
    """
    rows, dropped = [], collections.Counter()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            dropped["error"] += 1
            continue
        err = row.get("max_abs_err")
        if err is None or math.isnan(err):
            dropped["unchecked"] += 1
            continue
        rows.append(row)
    return rows, dropped


def per_shape(rows):
    """``{shape_id: {"features": ..., "tiles": {(bm, bn): median_ms}}}``.

    Median across rounds, not mean: one slow round from a scheduling hiccup
    should not move the winner, and with three rounds the median is the middle
    measurement rather than an average of an outlier.
    """
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    features = {}
    for row in rows:
        key = row["shape"]
        grouped[key][(row["block_m"], row["block_n"])].append(row["ms"])
        features.setdefault(
            key,
            {
                "batch": row["batch"],
                "seqlen_q": row["seqlen_q"],
                "seqlen_kv": row["seqlen_kv"],
                "num_query_heads": row["num_query_heads"],
                "num_kv_heads": row["num_kv_heads"],
                "head_size": row["head_size"],
                "dtype": row["dtype"],
                "causal": row["causal"],
                "source": row.get("source", ""),
                "cu_count": row.get("cu_count", DEFAULT_CU_COUNT),
            },
        )
    return {
        key: {
            "features": features[key],
            "tiles": {t: statistics.median(v) for t, v in tiles.items()},
        }
        for key, tiles in grouped.items()
    }


def cta_count(shape, block_m: int) -> int:
    """Work items the tile launches: query blocks x query heads x batch.

    The same arithmetic the rocKE dispatcher uses (``gfx950_dense.py::_base_spec``)
    and the quantity a grid-fill rule compares against the CU count.
    """
    query_blocks = -(-shape["seqlen_q"] // block_m)
    return query_blocks * shape["num_query_heads"] * shape["batch"]


# --- candidate rules: (shape, legal tiles) -> chosen tile -------------------


def rule_current(shape, tiles):
    """What ships today: 256x64 where legal, else ascending (block_m, block_n)."""
    if BASELINE_TILE in tiles:
        return BASELINE_TILE
    return min(tiles)


def rule_largest(shape, tiles):
    """Biggest tile that fits. The naive alternative, worth measuring so a more
    complicated rule has to beat something other than the status quo."""
    return max(tiles)


def _grid_fill_rule(threshold: float):
    """Prefer the largest tile that still fills the device.

    A tile is 'filling' when its CTA count reaches ``threshold`` x CU count.
    Among filling tiles take the largest block_m (fewest, fattest CTAs); if none
    fill, take the smallest block_m, which makes the most work items out of a
    launch too small to occupy the part.
    """

    def rule(shape, tiles):
        cus = shape["cu_count"] or DEFAULT_CU_COUNT
        filling = [t for t in tiles if cta_count(shape, t[0]) >= threshold * cus]
        if filling:
            return max(filling)
        return min(tiles, key=lambda t: (t[0], -t[1]))

    return rule


def rule_oracle(shape, tiles):
    """The best tile, by definition. The ceiling every rule is measured against."""
    return min(tiles, key=lambda t: tiles[t])


RULES = {
    "current (shipped)": rule_current,
    "largest tile": rule_largest,
    "grid-fill x1.0": _grid_fill_rule(1.0),
    "grid-fill x2.0": _grid_fill_rule(2.0),
    "grid-fill x4.0": _grid_fill_rule(4.0),
}


def evaluate(rule, shapes):
    """Geomean of (rule's pick / best) and the worst shape, over ``shapes``.

    1.0 means the rule picks the fastest tile everywhere. The acceptance criterion
    is written against this number.
    """
    ratios, worst = [], (1.0, None)
    for key, data in shapes.items():
        tiles = data["tiles"]
        if not tiles:
            continue
        best = min(tiles.values())
        if best <= 0:
            continue
        chosen = rule(data["features"], tiles)
        ratio = tiles[chosen] / best
        ratios.append(ratio)
        if ratio > worst[0]:
            worst = (ratio, key)
    if not ratios:
        return None
    geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return {
        "geomean": geomean,
        "worst": worst[0],
        "worst_shape": worst[1],
        "exact": sum(1 for r in ratios if r <= 1.0 + 1e-9),
        "count": len(ratios),
    }


def describe_winners(shapes):
    """How the winning tile varies with the features a rule could read."""
    by_tile = collections.Counter()
    by_fill = collections.defaultdict(collections.Counter)
    for data in shapes.values():
        tiles, feat = data["tiles"], data["features"]
        if not tiles:
            continue
        win = min(tiles, key=lambda t: tiles[t])
        by_tile[win] += 1
        cus = feat["cu_count"] or DEFAULT_CU_COUNT
        ratio = cta_count(feat, win[0]) / cus
        bucket = (
            "<1x"
            if ratio < 1
            else "1-2x" if ratio < 2 else "2-4x" if ratio < 4 else ">=4x"
        )
        by_fill[bucket][win] += 1
    return by_tile, by_fill


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", help="output of tile_sweep.py")
    ap.add_argument(
        "--source",
        default="",
        help="restrict to one shape source ('suite' for the geomean claim)",
    )
    args = ap.parse_args(argv)

    rows, dropped = load(args.jsonl)
    if dropped:
        print("dropped rows:", dict(dropped))
    shapes = per_shape(rows)
    if args.source:
        shapes = {
            k: v for k, v in shapes.items() if v["features"]["source"] == args.source
        }
    if not shapes:
        print("no usable rows")
        return 1
    print(f"shapes: {len(shapes)}   timed rows: {len(rows)}")
    print()

    by_tile, by_fill = describe_winners(shapes)
    print("winning tile, overall:")
    for tile, n in by_tile.most_common():
        print(f"   {tile[0]:>4}x{tile[1]:<4} {n:4d}  ({100 * n / len(shapes):.0f}%)")
    print()
    print("winning tile, by grid fill (CTAs / CUs at the winning block_m):")
    for bucket in ("<1x", "1-2x", "2-4x", ">=4x"):
        if bucket not in by_fill:
            continue
        inner = ", ".join(
            f"{t[0]}x{t[1]}:{n}" for t, n in by_fill[bucket].most_common(4)
        )
        print(f"   {bucket:<5} {inner}")
    print()

    print(f"{'rule':<22} {'geomean':>8} {'worst':>8} {'exact':>12}")
    baseline = evaluate(rule_current, shapes)
    for name, rule in RULES.items():
        got = evaluate(rule, shapes)
        if got is None:
            continue
        print(
            f"{name:<22} {got['geomean']:8.4f} {got['worst']:8.4f} "
            f"{got['exact']:>5}/{got['count']:<6}"
        )
    print()
    print(
        "geomean 1.0 = picks the fastest tile everywhere. "
        f"Headroom over the shipped scorer: {baseline['geomean']:.4f}x"
    )
    if baseline["geomean"] < 1.02:
        print(
            "  The shipped scorer is already within 2% of best. "
            "A heuristic buys little here, which is itself the finding."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
