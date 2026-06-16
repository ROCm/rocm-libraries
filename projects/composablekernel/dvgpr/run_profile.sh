#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# AICK-1303: run the fused-VectorSize conv kernels on MI450 (gfx1250) and compare each
# fused sel-path against its solo kernel. Both are vanilla (as-compiled); fused_conv is
# pinned to the VS1 VGPR budget, so fused-path-i slower than solo-i quantifies the
# occupancy cost of plain fusion - the cost dynamic VGPR would remove (dynamic VGPR is
# unsupported on this ROCm; see README). With PROFILE=1, also captures rocprofv3.
#
# Prereq: dvgpr/build_runnable.sh has produced out/harness, on an MI450 node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
HARNESS="${HARNESS:-$OUT/harness}"
REPS="${REPS:-100}"
PROFILE="${PROFILE:-0}"
ROCPROF="${ROCPROF:-rocprofv3}"
SUMMARY="$OUT/summary.csv"
[[ -x "$HARNESS" ]] || { echo "missing $HARNESS - run build_runnable.sh first" >&2; exit 2; }

# kernel sel ; solos first (per-VS baselines), then the fused paths.
JOBS=("solo1 0" "solo2 0" "solo4 0" "solo8 0"
      "fused_conv 0" "fused_conv 1" "fused_conv 2" "fused_conv 3")

echo "kernel,sel,avg_us,checksum" > "$SUMMARY"
STEP=1; TOTAL=${#JOBS[@]}
echo ">>> running $TOTAL kernels (REPS=$REPS, PROFILE=$PROFILE) ..."
for j in "${JOBS[@]}"; do
  read -r k s <<<"$j"
  printf '[%d/%d] %-10s sel=%s ... ' "$STEP" "$TOTAL" "$k" "$s"; STEP=$((STEP+1))
  line=$("$HARNESS" "$k" "$s" "$REPS") || { echo "FAILED"; continue; }
  echo "$line"
  echo "$line" >> "$SUMMARY"
  if [[ "$PROFILE" == "1" ]]; then
    dir="$OUT/prof/${k}_sel${s}"; mkdir -p "$dir"
    "$ROCPROF" --kernel-trace --output-format csv -d "$dir" -- "$HARNESS" "$k" "$s" "$REPS" >/dev/null 2>&1 || true
  fi
done

echo ">>> raw: $SUMMARY"
echo ">>> fused path vs its solo (overhead = plain-fusion occupancy cost; checksum must match):"
duckdb -markdown -c "
  WITH s AS (SELECT kernel, avg_us solo_us, checksum solo_sum FROM read_csv_auto('$SUMMARY') WHERE kernel LIKE 'solo%'),
       f AS (SELECT sel, avg_us fused_us, checksum fused_sum,
                    CASE sel WHEN 0 THEN 'solo1' WHEN 1 THEN 'solo2' WHEN 2 THEN 'solo4' ELSE 'solo8' END AS solo
             FROM read_csv_auto('$SUMMARY') WHERE kernel='fused_conv')
  SELECT f.sel, f.solo, s.solo_us, f.fused_us,
         round(100.0*(f.fused_us - s.solo_us)/s.solo_us, 1) AS fused_overhead_pct,
         (abs(f.fused_sum - s.solo_sum) < 1e-3*abs(s.solo_sum)) AS checksum_match
  FROM f JOIN s ON s.kernel = f.solo ORDER BY f.sel"
