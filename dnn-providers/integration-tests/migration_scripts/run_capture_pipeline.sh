#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Orchestrate the full C++ graph -> bundle migration pipeline.
#
# Layers 1-3 (census, capture, place, verify, idempotency) need only CPU.
# Layer 4 (differential coverage) needs a GPU host.
#
# Usage:
#   run_capture_pipeline.sh <binary> [work_dir]
#   run_capture_pipeline.sh <binary> [work_dir] --skip-layer4
#
set -euo pipefail

BINARY="${1:?Usage: $0 <integration_tests_binary> [work_dir] [--skip-layer4]}"
WORK="${2:-/tmp/almiopen2279}"
SKIP_LAYER4="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$(cd "$SCRIPT_DIR/.." && pwd)/integration_test_bundles"

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

# ── Layer 1: Verify (Hop C) ─────────────────────────────────────────────

echo ""
echo "--- Layer 1: Verify migration (Hop C) ---"
python3 "$SCRIPT_DIR/verify_migration.py" \
    --census "$WORK/census.json" \
    --capture-dir "$WORK/captured" \
    --bundle-dir "$OUT"

# ── Layer 2: Real binary smoke ──────────────────────────────────────────

echo ""
echo "--- Layer 2: Real binary smoke (bundle loader) ---"
"$BINARY" --allow-bundles --gd "$OUT" --gtest_filter='*Bundle*' || {
    echo "  WARN: Layer 2 smoke returned non-zero (may need GPU)" >&2
}

# ── Layer 3: Idempotency ────────────────────────────────────────────────

echo ""
echo "--- Layer 3: Idempotency check ---"
"$BINARY" --capture-bundles "$WORK/captured2" --gtest_filter='*IntegrationGpu*'
python3 "$SCRIPT_DIR/place_bundles.py" \
    --capture-dir "$WORK/captured2" \
    --output-dir "$OUT"
if git diff --exit-code -- "$OUT" > /dev/null 2>&1; then
    echo "  OK: idempotent (no diff)"
else
    echo "  FAIL: pipeline is not idempotent — git diff follows:" >&2
    git diff --stat -- "$OUT" >&2
    exit 1
fi

# ── Layer 4: Differential coverage (GPU only) ───────────────────────────

if [ "$SKIP_LAYER4" = "--skip-layer4" ]; then
    echo ""
    echo "--- Layer 4: SKIPPED (--skip-layer4) ---"
else
    echo ""
    echo "--- Layer 4: Differential coverage ---"
    "$BINARY" --gtest_output=json:"$WORK/cpp.json" \
        --gtest_filter='*IntegrationGpu*' || true
    "$BINARY" --allow-bundles --gd "$OUT" \
        --gtest_output=json:"$WORK/bundle.json" \
        --gtest_filter='*Bundle*' || true
    python3 "$SCRIPT_DIR/diff_coverage.py" \
        --cpp "$WORK/cpp.json" \
        --bundle "$WORK/bundle.json" \
        --bundle-dir "$OUT"
fi

echo ""
echo "=== Pipeline complete ==="
