#!/usr/bin/env python3
"""Generate a targeted top-up shape set for the DSL conv heuristic.

Reads OOF predictions from train.py (oof_predictions.parquet) to identify
shape subsets where the model is inaccurate, then generates a dense grid of
new shapes covering only those subsets.  Produced shapes are guaranteed to
have zero overlap with the existing training parquet so they can be fed
directly to the sweep binary and merged into the next training run.

Typical usage
-------------
  # Identify hard subsets and generate targeted shapes
  python3 generate_targeted_shapes_conv.py \\
      --oof   oof_predictions.parquet \\
      --train conv_fp16_<arch>_dsl.parquet \\
      --out   all_shapes.csv \\
      --shards 32

  # Full analytics report (global stats, worst shapes) without generating shapes
  python3 generate_targeted_shapes_conv.py \\
      --oof   oof_predictions.parquet \\
      --train training.parquet \\
      --analytics --dry-run [--analytics-out shape_report.csv]

  # Constrain generation to specific subsets regardless of OOF error
  python3 generate_targeted_shapes_conv.py \\
      --oof   oof_predictions.parquet \\
      --train training.parquet \\
      --out   shapes.csv \\
      --force-subsets "N=2" "N=2,grouped"

  # Dry run: print subset analysis only, do not write files
  python3 generate_targeted_shapes_conv.py \\
      --oof   oof_predictions.parquet \\
      --train training.parquet \\
      --out   shapes.csv \\
      --dry-run

The OOF parquet must contain columns produced by train.py:
  N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w, tflops,
  oof_pred_tflops

The training parquet must contain the same shape columns.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = ["N", "G", "C", "K", "Hi", "Wi", "Y", "X",
          "stride_h", "stride_w", "pad_h", "pad_w", "direction"]

SHAPE_COLS = ["N", "G", "C", "K", "Hi", "Wi", "Y", "X",
              "stride_h", "stride_w", "pad_h", "pad_w"]

# Canonical tile-size set used by ConvCandidateSweep — drop shapes where any
# GEMM dimension is smaller than the smallest tile.
MIN_TILE = 32

# Hard-coded dimension pools for the generator grid.
# These span the realistic operating range for DSL grouped-conv kernels.
_N_VALUES    = [1, 2, 4, 8, 16, 32, 64, 128]
_C_VALUES    = [32, 64, 128, 256, 512, 1024, 2048]
_K_VALUES    = [32, 64, 128, 256, 512, 1024, 2048]
_HW_VALUES   = [7, 14, 28, 56, 112, 224]
_FILTER_PADS = [(1, 1, 0, 0), (3, 3, 1, 1), (5, 5, 2, 2), (7, 7, 3, 3)]
_STRIDES     = [1, 2]
_G_VALUES    = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Shape categorisation helpers (mirrors sample_conv_shapes.py buckets)
# ---------------------------------------------------------------------------

def _spatial_bucket(Hi: int) -> str:
    if Hi <= 4:   return "tiny"
    if Hi <= 16:  return "small"
    if Hi <= 64:  return "medium"
    return "large"


def _channel_bucket(C: int, K: int) -> str:
    m = min(C, K)
    if m <= 16:  return "tiny"
    if m <= 64:  return "small"
    if m <= 256: return "medium"
    return "large"


def _group_type(G: int, C: int, K: int) -> str:
    if G == C == K: return "depthwise"
    if G > 1:       return "grouped"
    return "standard"


# ---------------------------------------------------------------------------
# OOF analysis
# ---------------------------------------------------------------------------

def _efficiency(pred_tflops, actual_tflops):
    """Per-candidate tflops efficiency (pred/oracle, clipped to [0,1])."""
    return np.clip(pred_tflops / np.maximum(actual_tflops, 1e-6), 0.0, 1.0)


def _build_per_shape(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per shape: oracle_tflops, pred_best_tflops, efficiency, subset labels."""
    df = oof_df.copy()
    df["oracle_tflops"] = df.groupby(SHAPE_COLS)["tflops"].transform("max")
    df["pred_rank"] = df.groupby(SHAPE_COLS)["oof_pred_tflops"].rank(
        ascending=False, method="first"
    )
    best = df[df["pred_rank"] == 1].copy()
    best["efficiency"] = _efficiency(best["tflops"].values, best["oracle_tflops"].values)
    best["group_type"]  = best.apply(lambda r: _group_type(int(r.G), int(r.C), int(r.K)), axis=1)
    best["spatial_bkt"] = best["Hi"].apply(lambda h: _spatial_bucket(int(h)))
    best["channel_bkt"] = best.apply(lambda r: _channel_bucket(int(r.C), int(r.K)), axis=1)
    return best.sort_values("efficiency").reset_index(drop=True)


def _build_summary(per_shape: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-shape efficiency into subset summary, sorted worst first."""
    return (
        per_shape.groupby(["N", "group_type", "spatial_bkt", "channel_bkt"])
        .agg(
            n_shapes       = ("efficiency", "count"),
            mean_efficiency= ("efficiency", "mean"),
            p10_efficiency = ("efficiency", lambda x: float(np.percentile(x, 10))),
            p50_efficiency = ("efficiency", "median"),
            min_efficiency = ("efficiency", "min"),
        )
        .reset_index()
        .sort_values("mean_efficiency")
    )


def print_analysis(summary: pd.DataFrame, threshold: float) -> None:
    print("\n=== OOF Subset Analysis (worst first) ===", file=sys.stderr)
    print(f"{'N':>4}  {'group_type':<12}  {'spatial':<8}  {'channel':<8}  "
          f"{'n_shapes':>8}  {'mean_eff':>8}  {'p10_eff':>7}  {'p50_eff':>7}",
          file=sys.stderr)
    print("-" * 75, file=sys.stderr)
    for _, r in summary.iterrows():
        flag = " <-- TARGETED" if r.mean_efficiency < threshold else ""
        print(f"{int(r.N):>4}  {r.group_type:<12}  {r.spatial_bkt:<8}  {r.channel_bkt:<8}  "
              f"{int(r.n_shapes):>8}  {r.mean_efficiency:>8.3f}  "
              f"{r.p10_efficiency:>7.3f}  {r.p50_efficiency:>7.3f}{flag}",
              file=sys.stderr)
    print(file=sys.stderr)


def print_analytics(per_shape: pd.DataFrame, summary: pd.DataFrame,
                    analytics_out: Path | None) -> None:
    """Print global stats and worst individual shapes; optionally write per-shape CSV."""
    eff = per_shape["efficiency"].values
    print("\n=== Global OOF Efficiency ===")
    print(f"  Shapes analysed : {len(eff):,}")
    print(f"  Mean efficiency : {eff.mean():.4f}")
    print(f"  P10  efficiency : {float(np.percentile(eff, 10)):.4f}")
    print(f"  P50  efficiency : {float(np.percentile(eff, 50)):.4f}")
    print(f"  P90  efficiency : {float(np.percentile(eff, 90)):.4f}")
    print(f"  Min  efficiency : {eff.min():.4f}")

    print("\n=== Worst 20 Shapes ===")
    worst = per_shape.head(20)
    print(worst[SHAPE_COLS + ["oracle_tflops", "tflops", "efficiency"]]
          .rename(columns={"tflops": "pred_best_tflops"})
          .to_string(index=False))

    if analytics_out is not None:
        out_cols = SHAPE_COLS + ["oracle_tflops", "tflops", "efficiency",
                                  "group_type", "spatial_bkt", "channel_bkt"]
        analytics_out.parent.mkdir(parents=True, exist_ok=True)
        per_shape[out_cols].rename(columns={"tflops": "pred_best_tflops"}).to_csv(
            analytics_out, index=False
        )
        print(f"\nPer-shape analysis written to {analytics_out}")


# ---------------------------------------------------------------------------
# Shape validity + GEMM dimension check
# ---------------------------------------------------------------------------

def _valid(N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw) -> bool:
    if C % G != 0 or K % G != 0:
        return False
    # Per-group channel counts must be 8-aligned (DSL kernel alignment).
    if (C // G) % 8 != 0 or (K // G) % 8 != 0:
        return False
    Ho = (Hi + 2 * ph - Y) // sh + 1
    Wo = (Wi + 2 * pw - X) // sw + 1
    if Ho < 1 or Wo < 1:
        return False
    # GEMM dims must all be >= MIN_TILE
    M      = N * Ho * Wo
    N_gemm = K
    K_gemm = (C // G) * Y * X
    return M >= MIN_TILE and N_gemm >= MIN_TILE and K_gemm >= MIN_TILE


# ---------------------------------------------------------------------------
# Targeted shape generator
# ---------------------------------------------------------------------------

def _parse_force_subset(token: str) -> dict:
    """Parse a token like 'N=2' or 'N=2,grouped' into a predicate dict."""
    pred = {}
    for part in token.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            pred[k.strip()] = v.strip()
        else:
            # bare word interpreted as group_type
            pred["group_type"] = part
    return pred


def _matches_subset(N, G, C, K, Hi, group_type, spatial_bkt, channel_bkt,
                    pred: dict) -> bool:
    """Return True if (N, …) satisfies all predicates in pred."""
    if "N" in pred and int(pred["N"]) != N:
        return False
    if "group_type" in pred and pred["group_type"] != group_type:
        return False
    if "spatial" in pred and pred["spatial"] != spatial_bkt:
        return False
    if "channel" in pred and pred["channel"] != channel_bkt:
        return False
    return True


def generate_targeted(
    targeted_subsets: list[dict],
    training_set: set[tuple],
    density: int = 1,
) -> list[tuple]:
    """
    Generate shapes that fall in at least one targeted subset and are not in
    training_set.

    density: multiplier on the default grid resolution (1 = standard, 2 = 2x
    more N/spatial/channel combinations — useful when a subset is very sparse).
    """
    shapes: set[tuple] = set()

    # Expand dimension pools with density multiplier
    n_pool  = _N_VALUES
    c_pool  = _C_VALUES
    k_pool  = _K_VALUES
    hw_pool = _HW_VALUES
    if density >= 2:
        n_pool  = sorted(set(n_pool  + [3, 6, 12, 24, 48, 96]))
        hw_pool = sorted(set(hw_pool + [10, 12, 18, 24, 36, 48]))
        c_pool  = sorted(set(c_pool  + [96, 192, 384, 768, 1536]))
        k_pool  = sorted(set(k_pool  + [96, 192, 384, 768, 1536]))

    def try_add(N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw):
        t = (N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw)
        if not _valid(*t):
            return
        if t in training_set or t in shapes:
            return
        gt  = _group_type(G, C, K)
        sb  = _spatial_bucket(Hi)
        cb  = _channel_bucket(C, K)
        for pred in targeted_subsets:
            if _matches_subset(N, G, C, K, Hi, gt, sb, cb, pred):
                shapes.add(t)
                return

    for N in n_pool:
        for G in _G_VALUES:
            for C in c_pool:
                if C % G != 0:
                    continue
                if (C // G) % 8 != 0:
                    continue
                for K in k_pool:
                    if K % G != 0:
                        continue
                    if (K // G) % 8 != 0:
                        continue
                    for Hi in hw_pool:
                        Wi = Hi  # symmetric by default
                        for Y, X, ph, pw in _FILTER_PADS:
                            for sh in _STRIDES:
                                sw = sh
                                try_add(N, G, C, K, Hi, Wi, Y, X, sh, sw, ph, pw)

    return sorted(shapes)


# ---------------------------------------------------------------------------
# Shard writer (same format as other shape generators)
# ---------------------------------------------------------------------------

def write_csv(rows: list[tuple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(list(r) + ["forward"])


def write_shards(shapes: list[tuple], n_shards: int, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    size = math.ceil(len(shapes) / n_shards)
    actual = 0
    for i in range(n_shards):
        chunk = shapes[i * size: (i + 1) * size]
        if not chunk:
            break
        write_csv(chunk, out_dir / f"shard_{i:02d}.csv")
        actual += 1
    return actual


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--oof", required=True, type=Path,
                    help="OOF predictions parquet from train.py")
    ap.add_argument("--train", required=True, type=Path,
                    help="Training parquet to exclude from generated shapes")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output all_shapes.csv path (required unless --dry-run)")
    ap.add_argument("--shards", type=int, default=32,
                    help="Number of shard CSVs to write alongside all_shapes.csv (default: 32)")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="Mean efficiency below which a subset is targeted (default: 0.90)")
    ap.add_argument("--force-subsets", nargs="*", default=[],
                    metavar="PRED",
                    help="Override threshold and always target these subsets. "
                         "Format: 'N=2' or 'N=2,grouped' or 'spatial=small'. "
                         "Multiple tokens allowed.")
    ap.add_argument("--density", type=int, default=1, choices=[1, 2, 3],
                    help="Grid density multiplier (1=default, 2=denser, default: 1)")
    ap.add_argument("--analytics", action="store_true",
                    help="Print global efficiency stats and worst 20 shapes")
    ap.add_argument("--analytics-out", type=Path, default=None,
                    metavar="CSV",
                    help="Write per-shape efficiency analysis to this CSV (implies --analytics)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print subset analysis only; do not generate or write shapes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out is None and not args.dry_run:
        ap.error("--out is required unless --dry-run is set")

    # Load data
    print(f"Loading OOF predictions from {args.oof} ...", file=sys.stderr)
    oof_df = pd.read_parquet(args.oof)
    if "oof_pred_tflops" not in oof_df.columns:
        ap.error("OOF parquet missing 'oof_pred_tflops' column — was it produced by train.py?")

    print(f"  {len(oof_df):,} rows, "
          f"{oof_df.groupby(SHAPE_COLS).ngroups:,} unique shapes", file=sys.stderr)

    print(f"Loading training shapes from {args.train} ...", file=sys.stderr)
    train_df = pd.read_parquet(args.train)
    training_set: set[tuple] = set(
        tuple(int(v) for v in row)
        for row in train_df[SHAPE_COLS].drop_duplicates().values.tolist()
    )
    print(f"  {len(training_set):,} unique training shapes", file=sys.stderr)

    # OOF analysis
    per_shape = _build_per_shape(oof_df)
    summary   = _build_summary(per_shape)
    print_analysis(summary, args.threshold)

    if args.analytics or args.analytics_out is not None:
        print_analytics(per_shape, summary, args.analytics_out)

    # Determine targeted subsets
    targeted_subsets: list[dict] = []

    bad = summary[summary["mean_efficiency"] < args.threshold]
    for _, r in bad.iterrows():
        targeted_subsets.append({
            "N":          str(int(r.N)),
            "group_type": r.group_type,
            "spatial":    r.spatial_bkt,
            "channel":    r.channel_bkt,
        })

    for token in args.force_subsets:
        pred = _parse_force_subset(token)
        targeted_subsets.append(pred)
        print(f"  Force-targeting subset: {pred}", file=sys.stderr)

    if not targeted_subsets:
        print("No subsets below threshold and no --force-subsets specified. "
              "Nothing to generate.", file=sys.stderr)
        return

    print(f"\nTargeting {len(targeted_subsets)} subset(s):", file=sys.stderr)
    for p in targeted_subsets:
        print(f"  {p}", file=sys.stderr)

    if args.dry_run:
        print("\n--dry-run: stopping before shape generation.", file=sys.stderr)
        return

    # Generate
    print(f"\nGenerating shapes (density={args.density}) ...", file=sys.stderr)
    shapes = generate_targeted(targeted_subsets, training_set, density=args.density)
    print(f"  Generated {len(shapes):,} new shapes (zero overlap with training)",
          file=sys.stderr)

    if not shapes:
        print("No shapes generated — all candidates already in training set "
              "or failed geometry checks.", file=sys.stderr)
        return

    overlap = set(shapes) & training_set
    assert len(overlap) == 0, f"BUG: {len(overlap)} shapes overlap with training set"

    write_csv(shapes, args.out)
    print(f"Wrote {len(shapes):,} shapes to {args.out}", file=sys.stderr)

    n_shards = write_shards(shapes, args.shards, args.out.parent)  # type: ignore[union-attr]
    print(f"Wrote {n_shards} shards to {args.out.parent}/", file=sys.stderr)

    n_by_N: dict[int, int] = defaultdict(int)
    n_by_gt: dict[str, int] = defaultdict(int)
    n_by_sp: dict[str, int] = defaultdict(int)
    for s in shapes:
        N, G, C, K, Hi = s[0], s[1], s[2], s[3], s[4]
        n_by_N[N] += 1
        n_by_gt[_group_type(G, C, K)] += 1
        n_by_sp[_spatial_bucket(Hi)] += 1

    print("\nShape breakdown:", file=sys.stderr)
    print(f"  By N:          {dict(sorted(n_by_N.items()))}", file=sys.stderr)
    print(f"  By group_type: {dict(sorted(n_by_gt.items()))}", file=sys.stderr)
    print(f"  By spatial:    {dict(sorted(n_by_sp.items()))}", file=sys.stderr)


if __name__ == "__main__":
    main()
