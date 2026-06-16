#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# AICK-1303: run the dynamic-VGPR comparison on MI450 (gfx1250). For each kernel and
# the fused sel-paths, runs the harness against the vanilla and patched code objects
# and collects per-launch time + an output checksum (correctness proxy). With
# PROFILE=1 it also captures rocprofv3 occupancy/duration counters.
#
#   {solo1,solo2,solo4,solo8,fused_conv(sel 0..3)} x {vanilla,patched}
#
# Prereqs: dvgpr/build_runnable.sh has produced out/{harness,vanilla.hsaco,patched.hsaco}
# and you are on an MI450 node. The patched run only differs if the CP/MES honors
# ENABLE_DYNAMIC_VGPR; if results mismatch or it fails, see dynamic-vgpr-feasibility.md
# (s_alloc_vgpr SCC-retry, segment granularity).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
HARNESS="${HARNESS:-$OUT/harness}"
VANILLA_HSACO="${VANILLA_HSACO:-$OUT/vanilla.hsaco}"
PATCHED_HSACO="${PATCHED_HSACO:-$OUT/patched.hsaco}"
REPS="${REPS:-100}"
PROFILE="${PROFILE:-0}"
ROCPROF="${ROCPROF:-rocprofv3}"
SUMMARY="$OUT/summary.csv"

for f in "$HARNESS" "$VANILLA_HSACO" "$PATCHED_HSACO"; do
  [[ -f "$f" ]] || { echo "missing $f - run build_runnable.sh first" >&2; exit 2; }
done

# matrix: "kernel sel"
JOBS=("fused_conv 0" "fused_conv 1" "fused_conv 2" "fused_conv 3"
      "solo1 0" "solo2 0" "solo4 0" "solo8 0")

echo "variant,kernel,sel,avg_us,checksum" > "$SUMMARY"
run() {  # label hsaco kernel sel
  local label="$1" hsaco="$2" kern="$3" sel="$4"
  local line; line=$("$HARNESS" "$hsaco" "$kern" "$sel" "$REPS") || { echo "  run failed: $label $kern $sel" >&2; return 0; }
  echo "$label,$line" >> "$SUMMARY"
  if [[ "$PROFILE" == "1" ]]; then
    local dir="$OUT/prof/${label}_${kern}_sel${sel}"; mkdir -p "$dir"
    "$ROCPROF" --kernel-trace --output-format csv -d "$dir" -- \
        "$HARNESS" "$hsaco" "$kern" "$sel" "$REPS" >/dev/null 2>&1 || true
  fi
}

for j in "${JOBS[@]}"; do
  read -r k s <<<"$j"
  run vanilla "$VANILLA_HSACO" "$k" "$s"
  run patched "$PATCHED_HSACO" "$k" "$s"
done

echo ">>> raw: $SUMMARY"
echo ">>> vanilla vs patched (avg us) + checksum match:"
duckdb -markdown -c "
  WITH v AS (SELECT kernel,sel,avg_us van_us,checksum van_sum FROM read_csv_auto('$SUMMARY') WHERE variant='vanilla'),
       p AS (SELECT kernel,sel,avg_us pat_us,checksum pat_sum FROM read_csv_auto('$SUMMARY') WHERE variant='patched')
  SELECT v.kernel, v.sel, v.van_us, p.pat_us,
         round(100.0*(v.van_us-p.pat_us)/v.van_us, 1) AS pct_faster,
         (abs(v.van_sum-p.pat_sum) < 1e-3*abs(v.van_sum)) AS checksum_match
  FROM v JOIN p USING (kernel, sel) ORDER BY v.kernel, v.sel"
