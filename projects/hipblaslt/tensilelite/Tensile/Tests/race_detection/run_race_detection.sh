#!/usr/bin/env bash
# Run Tensile GEMM kernels through the rocjitsu emulator with race
# detection enabled. Fails if any race is detected or validation fails.
#
# Usage:
#   run_race_detection.sh <rocjitsu_kmd_lib> <tensilelite_client> <rj_config>
#
# Environment:
#   PYTHONPATH must include the rocisa build and tensilelite source.
#
# All .yaml files in this script's directory are treated as test configs.

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <rocjitsu_kmd_lib> <tensilelite_client> <rj_config>"
  exit 1
fi

KMD_LIB="$(realpath "$1")"
CLIENT="$(realpath "$2")"
RJ_CONFIG="$(realpath "$3")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TENSILE="$SCRIPT_DIR/../../bin/Tensile"
OUTPUT_DIR="${RACE_OUTPUT_DIR:-/tmp/rj-race-output}"

failed=0
for yaml in "$SCRIPT_DIR"/*.yaml; do
  name=$(basename "$yaml" .yaml)
  outdir="$OUTPUT_DIR/$name"
  sinkdir="$OUTPUT_DIR/race-reports/$name"
  mkdir -p "$outdir" "$sinkdir"

  echo "=== Running $name ==="
  RJ_CONFIG="$RJ_CONFIG" \
  RJ_LOG=1 \
  RJ_RACE=1 \
  RJ_SINKS=stderr,file \
  RJ_SINK_DIR="$sinkdir" \
  LD_PRELOAD="$KMD_LIB" \
  timeout 60 python "$TENSILE" \
    "$yaml" "$outdir" \
    --prebuilt-client "$CLIENT" \
    2>&1 | tee "$OUTPUT_DIR/$name.log"

  # Check validation
  if ! grep -q "PASSED" "$OUTPUT_DIR/$name.log"; then
    echo "FAIL: $name — validation did not pass"
    failed=1
  fi

  # Check for races
  if grep -q "^RACE " "$sinkdir/race.log" 2>/dev/null; then
    echo "FAIL: $name — race(s) detected:"
    cat "$sinkdir/race.log"
    failed=1
  else
    echo "PASS: $name — no races detected"
  fi
  echo ""
done

if [ "$failed" -ne 0 ]; then
  echo "One or more tests failed."
  exit 1
fi

echo "All tests passed."
