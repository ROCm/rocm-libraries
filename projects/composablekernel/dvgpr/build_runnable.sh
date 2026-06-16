#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# AICK-1303: build the runnable dynamic-VGPR comparison for MI450 (gfx1250):
#   out/vanilla.hsaco  - code object, kernels as compiled (static VGPR)
#   out/patched.hsaco  - code object, kernels transformed to dynamic VGPR
#   out/harness        - host executable that loads a .hsaco and launches a kernel
# Run with run_profile.sh. Requires a configured build/ (for compile flags) and the
# ROCm toolchain. The .hsaco builds need no GPU; running the harness needs MI450.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
OUT="$HERE/out"; mkdir -p "$OUT"
CLANG="/opt/rocm/bin/amdclang++"
KERNELS="solo1 solo2 solo4 solo8 fused_conv"
EX="example/ck_tile/20_grouped_convolution"

# 1. Produce vanilla.s / patched.s (+ the static-analysis objects).
bash "$HERE/build_variants.sh"

# 2. Code objects by linking the relocatable objects from build_variants. patched.o
#    already carries ENABLE_DYNAMIC_VGPR (set on the .o); linking preserves it.
echo ">>> building code objects ..."
"$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx1250 "$OUT/vanilla.o" -o "$OUT/vanilla.hsaco"
"$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx1250 "$OUT/patched.o" -o "$OUT/patched.hsaco"
echo "    verifying ENABLE_DYNAMIC_VGPR in patched.hsaco (read-only):"
python3 "$HERE/verify_dvgpr.py" "$OUT/patched.hsaco" "$OUT/vanilla.hsaco"

# 3. Harness executable. Reuse the example's compile flags (includes/defines/arch),
#    drop -c and the object output, link as an executable.
CMD=$(jq -r '.[] | select(.file | endswith("grouped_convolution_forward.cpp")) | .command' \
        "$BUILD/compile_commands.json" | head -1)
FLAGS=$(echo "$CMD" | sed -E "s#^[^ ]+ ##; s# -c # #; s# -o [^ ]+\.o # #; s#[^ ]*grouped_convolution_forward.cpp##")
echo ">>> building harness ..."
( cd "$ROOT" && "$CLANG" $FLAGS "$EX/fused_vectorsize_harness.cpp" -o "$OUT/harness" ) \
    2>&1 | tee "$ROOT/.cline_output.log" | tail -5

echo ">>> done:"
ls -la "$OUT/vanilla.hsaco" "$OUT/patched.hsaco" "$OUT/harness"
echo "    run: VANILLA_HSACO=$OUT/vanilla.hsaco PATCHED_HSACO=$OUT/patched.hsaco bash $HERE/run_profile.sh"
