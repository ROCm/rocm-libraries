#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# AICK-1303: build the runnable harness for MI450 (gfx1250). The harness compiles
# the fused-VectorSize kernels in and launches them via <<<>>> (vanilla / as-compiled
# static VGPR). Produces out/harness. Run with run_profile.sh.
# (Dynamic-VGPR patched code objects are not runnable on this ROCm - the runtime does
# not enable dynamic-VGPR dispatch; see dvgpr/README.md. For the static analysis use
# build_variants.sh + extract_metrics.py.)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
OUT="$HERE/out"; mkdir -p "$OUT"
CLANG="/opt/rocm/bin/amdclang++"
EX="example/ck_tile/20_grouped_convolution"

# Reuse the example's compile flags (includes/defines/arch); drop -c and the object
# output, add an rpath to the ROCm libs so the binary runs without LD_LIBRARY_PATH.
CMD=$(jq -r '.[] | select(.file | endswith("grouped_convolution_forward.cpp")) | .command' \
        "$BUILD/compile_commands.json" | head -1)
FLAGS=$(echo "$CMD" | sed -E "s#^[^ ]+ ##; s# -c # #; s# -o [^ ]+\.o # #; s#[^ ]*grouped_convolution_forward.cpp##")
ROCM_LIB=$(dirname "$(find /opt/rocm* -name 'libamdhip64.so.7' 2>/dev/null | head -1)")
ROCM_LIB="${ROCM_LIB:-/opt/rocm/lib}"

echo ">>> building harness (rpath: $ROCM_LIB) ..."
( cd "$ROOT" && "$CLANG" $FLAGS -Wl,-rpath,"$ROCM_LIB" "$EX/fused_vectorsize_harness.cpp" -o "$OUT/harness" ) \
    2>&1 | tee "$ROOT/.cline_output.log" | tail -5

echo ">>> done:"; ls -la "$OUT/harness"
echo "    run: bash $HERE/run_profile.sh"
