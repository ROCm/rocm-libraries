#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
YAML_DIR="$SCRIPT_DIR/../YAMLs"

export TENSILE_DISABLE_HELPER_CACHE=1
cd "$TENSILE_DIR"

# bf16 MegaFused path (PartialRMS + ResidualAdd + StoreBf16D).
rm -rf /tmp/fused-harness
./Tensile/bin/Tensile "$YAML_DIR/correctness_bf16.yaml" /tmp/fused-harness
FUSED_RC=$?

# MXFP8 MegaFused path (PartialRMS + ResidualAdd + MXFP8 DynQuant + StoreBf16D).
rm -rf /tmp/fused-mxfp8-harness
./Tensile/bin/Tensile "$YAML_DIR/correctness_mxfp8.yaml" /tmp/fused-mxfp8-harness
MXFP8_RC=$?

# Non-zero if either path failed so callers can gate on a single exit code.
if [[ $FUSED_RC -ne 0 || $MXFP8_RC -ne 0 ]]; then
  echo "correctness_harness FAILED: fused_rc=$FUSED_RC mxfp8_rc=$MXFP8_RC"
  exit 1
fi
echo "correctness_harness PASSED: fused and mxfp8 mega-fused configs both OK"
