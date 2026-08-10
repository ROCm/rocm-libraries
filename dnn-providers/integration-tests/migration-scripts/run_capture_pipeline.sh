#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Orchestrate the full C++ graph -> bundle migration pipeline.
#
#   Hop A  capture   C++ --capture-bundles serializes each graph to disk
#   Hop B  compress  place_bundles.py -> template + sweep
#   Hop C  verify    verify_migration.py re-expands every case, byte-diffs vs A
#   Hop D  cover     diff_coverage.py: pass_set(bundles) ⊇ pass_set(c++)
#
# Hop C proves the bytes survived (static). Hop D proves the behavior survived
# (the migrated bundles actually run and pass wherever the C++ tests passed).
# Two supporting layers run between C and D: a bundle-loader smoke test and an
# idempotency check.
#
# Hops A-C and the supporting layers need only CPU. Hop D needs a GPU host.
#
# Usage:
#   run_capture_pipeline.sh <binary> [work_dir]
#   run_capture_pipeline.sh <binary> [work_dir] --skip-hopd   (alias: --skip-layer4)
#
set -euo pipefail

BINARY="${1:?Usage: $0 <integration_tests_binary> [work_dir] [--skip-hopd]}"
WORK="${2:-/tmp/almiopen2279}"
SKIP_HOPD="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$(cd "$SCRIPT_DIR/.." && pwd)/integration-test-bundles"

# Template-sweep bundles register under gtest suites named {tier}_{Op}_{Topology}
# (e.g. quick_ConvolutionFwd_Default) — NOT containing the word "Bundle". Filter
# on the tier prefixes so the loader-facing checks target exactly our bundles.
BUNDLE_FILTER='quick_*:full_*:standard_*:comprehensive_*'

echo "=== Migration Pipeline ==="
echo "  binary:   $BINARY"
echo "  work dir: $WORK"
echo "  output:   $OUT"
echo ""

mkdir -p "$WORK"

# ── Hop A: Census + Capture ──────────────────────────────────────────────

echo "--- Step 1: Census ---"
python3 "$SCRIPT_DIR/census.py" "$BINARY" --graph-only --out "$WORK/census.json"

echo ""
echo "--- Step 2: Capture (Hop A) ---"
"$BINARY" --capture-bundles "$WORK/captured" --gtest_filter='*IntegrationGpu*'

echo ""
echo "--- Step 3: Place (Hop B) ---"
python3 "$SCRIPT_DIR/place_bundles.py" \
    --capture-dir "$WORK/captured" \
    --output-dir "$OUT"

# ── Hop C: Verify (byte-level round-trip) ───────────────────────────────

echo ""
echo "--- Hop C: Verify migration (byte-diff vs capture) ---"
python3 "$SCRIPT_DIR/verify_migration.py" \
    --census "$WORK/census.json" \
    --capture-dir "$WORK/captured" \
    --bundle-dir "$OUT"

# ── Supporting layer: real binary smoke ─────────────────────────────────

echo ""
echo "--- Supporting: real binary smoke (bundle loader) ---"
SMOKE_LOG="$WORK/smoke.log"
if ! "$BINARY" --allow-bundles --gd "$OUT" --gtest_filter="$BUNDLE_FILTER" 2>&1 | tee "$SMOKE_LOG"; then
    if grep -q "Failed to load bundle" "$SMOKE_LOG"; then
        echo "  FAIL: a bundle failed to load — this is a real regression, not a missing-GPU skip" >&2
        exit 1
    fi
    echo "  WARN: smoke returned non-zero (may need GPU)" >&2
fi

# ── Supporting layer: idempotency ───────────────────────────────────────

echo ""
echo "--- Supporting: idempotency check ---"
"$BINARY" --capture-bundles "$WORK/captured2" --gtest_filter='*IntegrationGpu*'
python3 "$SCRIPT_DIR/place_bundles.py" \
    --capture-dir "$WORK/captured2" \
    --output-dir "$OUT"
if ! git -C "$OUT" rev-parse --git-dir > /dev/null 2>&1; then
    echo "  SKIP: $OUT is not in a git repo — cannot check idempotency via diff"
elif git diff --exit-code -- "$OUT" > /dev/null 2>&1; then
    echo "  OK: idempotent (no diff)"
else
    echo "  FAIL: pipeline is not idempotent — git diff follows:" >&2
    git diff --stat -- "$OUT" >&2
    exit 1
fi

# ── Hop D: Differential coverage (GPU only) ─────────────────────────────

if [ "$SKIP_HOPD" = "--skip-hopd" ] || [ "$SKIP_HOPD" = "--skip-layer4" ]; then
    echo ""
    echo "--- Hop D: SKIPPED ($SKIP_HOPD) ---"
else
    echo ""
    echo "--- Hop D: Differential coverage (behavior-level) ---"
    "$BINARY" --gtest_output=json:"$WORK/cpp.json" \
        --gtest_filter='*IntegrationGpu*' || true
    "$BINARY" --allow-bundles --gd "$OUT" \
        --gtest_output=json:"$WORK/bundle.json" \
        --gtest_filter="$BUNDLE_FILTER" || true
    python3 "$SCRIPT_DIR/diff_coverage.py" \
        --cpp "$WORK/cpp.json" \
        --bundle "$WORK/bundle.json" \
        --bundle-dir "$OUT"
fi

echo ""
echo "=== Pipeline complete ==="
