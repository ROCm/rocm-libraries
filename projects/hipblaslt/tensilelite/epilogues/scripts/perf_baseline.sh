#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Perf-baseline script for the SubtileMegaFusedEmit batched/prefetched redesign.
# Runs the pinned single-solution baseline YAML and prints per-shape timing so
# BEFORE and AFTER numbers are easy to compare across redesign increments.
#
# Usage: bash epilogues/scripts/perf_baseline.sh [output-dir]
#   output-dir defaults to /tmp/redesign-baseline.
# The caller must source the venv before invoking this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
YAML_DIR="$SCRIPT_DIR/../YAMLs"

export TENSILE_DISABLE_HELPER_CACHE=1
cd "$TENSILE_DIR"

OUT_DIR="${1:-/tmp/redesign-baseline}"
YAML="$YAML_DIR/redesign_baseline_bf16.yaml"
RAW_LOG="/tmp/perf_baseline_raw.txt"

rm -rf "$OUT_DIR"

echo "=== perf_baseline: redesign_baseline_bf16 ===" | tee "$RAW_LOG"
echo "Output dir : $OUT_DIR" | tee -a "$RAW_LOG"
echo "YAML       : $YAML"   | tee -a "$RAW_LOG"
echo "" | tee -a "$RAW_LOG"

./Tensile/bin/Tensile "$YAML" "$OUT_DIR" 2>&1 | tee -a "$RAW_LOG"
RC=$?

echo ""
echo "=== Timing lines (shapes 2048x16384 and 4096x4096) ==="

# Tensile benchmark client emits per-problem CSV rows that contain the solution name
# (includes "Contraction_") and the kernel time in microseconds.  Extract those lines
# so the winning kernel time-us for each shape is immediately visible.
grep -E "(Contraction_|time.?us|GBench|Winner|2048.*16384|4096.*4096|us)" "$RAW_LOG" | \
  grep -v "^==="

echo ""
echo "=== Raw CSV output (all solver lines) ==="
grep -E "Contraction_" "$RAW_LOG" | head -40

exit $RC
