#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# AICK-1303: run the fused-VectorSize conv kernels on MI450 (gfx1250) across a few
# shapes and compare each fused sel-path against its solo. Both are vanilla
# (as-compiled): fused_conv is pinned to the VS1 VGPR budget, so a positive
# fused-vs-solo overhead is the runtime cost of plain fusion - the cost dynamic VGPR
# would remove. ~0 overhead means the kernel is not occupancy-bound for that shape, so
# consolidation is free without dynamic VGPR. Sweep shapes to see if any is sensitive.
#
# Prereq: dvgpr/build_runnable.sh has produced out/harness, on an MI450 node.
# Env: REPS (default 1000), SHAPES override (see below).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
HARNESS="${HARNESS:-$OUT/harness}"
REPS="${REPS:-1000}"
SUMMARY="$OUT/summary.csv"
[[ -x "$HARNESS" ]] || { echo "missing $HARNESS - run build_runnable.sh first" >&2; exit 2; }

# shape label -> env. Escalating size / occupancy stress (C,K multiples of 8).
declare -A SHAPES=(
  [small]="CONV_N=64 CONV_K=128 CONV_C=128 CONV_HI=28 CONV_WI=28"
  [large]="CONV_N=256 CONV_K=256 CONV_C=256 CONV_HI=56 CONV_WI=56"
  [spatial]="CONV_N=128 CONV_K=64 CONV_C=64 CONV_HI=112 CONV_WI=112"
  [deep]="CONV_N=64 CONV_K=512 CONV_C=512 CONV_HI=28 CONV_WI=28"
)
SHAPE_ORDER=(small large spatial deep)

JOBS=("solo1 0" "solo2 0" "solo4 0" "solo8 0"
      "fused_conv 0" "fused_conv 1" "fused_conv 2" "fused_conv 3")

echo "shape,kernel,sel,avg_us,checksum" > "$SUMMARY"
for shape in "${SHAPE_ORDER[@]}"; do
  echo ">>> shape=$shape (${SHAPES[$shape]}), REPS=$REPS"
  for j in "${JOBS[@]}"; do
    read -r k s <<<"$j"
    printf '   %-10s sel=%s ... ' "$k" "$s"
    line=$(env ${SHAPES[$shape]} "$HARNESS" "$k" "$s" "$REPS" 2>/dev/null) \
      || { echo "FAILED (shape unsupported?)"; continue; }
    echo "$line"
    echo "$shape,$line" >> "$SUMMARY"
  done
done

echo ">>> raw: $SUMMARY"
echo ">>> fused path vs its solo, per shape (overhead = plain-fusion runtime cost):"
duckdb -markdown -c "
  WITH s AS (SELECT shape, kernel, avg_us solo_us, checksum solo_sum
             FROM read_csv_auto('$SUMMARY') WHERE kernel LIKE 'solo%'),
       f AS (SELECT shape, sel, avg_us fused_us, checksum fused_sum,
                    CASE sel WHEN 0 THEN 'solo1' WHEN 1 THEN 'solo2' WHEN 2 THEN 'solo4' ELSE 'solo8' END AS solo
             FROM read_csv_auto('$SUMMARY') WHERE kernel='fused_conv')
  SELECT f.shape, f.sel, f.solo, s.solo_us, f.fused_us,
         round(100.0*(f.fused_us - s.solo_us)/s.solo_us, 1) AS fused_overhead_pct,
         (abs(f.fused_sum - s.solo_sum) < 1e-3*abs(s.solo_sum)) AS checksum_match
  FROM f JOIN s ON s.shape = f.shape AND s.kernel = f.solo
  ORDER BY f.shape, f.sel"
