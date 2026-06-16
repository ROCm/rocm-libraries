#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# AICK-1303: compile the fused-VectorSize probe and produce two gfx1250 device
# objects for the dynamic-VGPR comparison:
#   out/vanilla.o  - all kernels as compiled (solo1/2/4/8 + fused_conv, static VGPR)
#   out/patched.o  - same kernels transformed to dynamic VGPR (per-path s_alloc_vgpr
#                    + small creation count + DEALLOC removed + ENABLE_DYNAMIC_VGPR)
# The two objects give the "with and without patch" axis; the five kernels give the
# "each solo plus fused" axis. Static only (assemble + descriptor); running/profiling
# needs a launch harness + MI450 (see run_profile.sh).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
MC="/opt/rocm/lib/llvm/bin/llvm-mc"
PROBE_REL="example/ck_tile/20_grouped_convolution/fused_vectorsize_probe.cpp"
KERNELS="solo1 solo2 solo4 solo8 fused_conv"
OUT="$HERE/out"
mkdir -p "$OUT"

# 1. Derive the probe compile command from the build's grouped_convolution_forward.cpp
#    entry (same flags/arch), swap in the probe, emit the gfx1250 device .s.
SRC_REL="example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp"
CMD=$(jq -r ".[] | select(.file | endswith(\"$SRC_REL\")) | .command" "$BUILD/compile_commands.json" | head -1)
CMD="${CMD//$SRC_REL/$PROBE_REL}"
CMD="${CMD/ -c / --save-temps=obj -c }"
CMD=$(echo "$CMD" | sed -E "s# -o [^ ]+grouped_convolution_forward.cpp.o# -o $OUT/probe.o#")

echo ">>> compiling probe for gfx1250 ..."
( cd "$BUILD" && eval "$CMD" ) 2>&1 | tee "$ROOT/.cline_output.log" | tail -3
# --save-temps=obj writes temps next to the -o object (i.e. in $OUT).
DEV_S="$OUT/fused_vectorsize_probe-hip-amdgcn-amd-amdhsa-gfx1250.s"
cp "$DEV_S" "$OUT/vanilla.s"
echo "    device asm: $OUT/vanilla.s"

# 2. Vanilla object.
"$MC" -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 "$OUT/vanilla.s" --filetype=obj -o "$OUT/vanilla.o"
echo "    assembled $OUT/vanilla.o"

# 3. Patched object: transform each kernel, assemble, set ENABLE_DYNAMIC_VGPR on the five.
cp "$OUT/vanilla.s" "$OUT/patched.s"
for k in $KERNELS; do
  python3 "$HERE/dvgpr_transform.py" "$OUT/patched.s" "$OUT/patched.s" "$k"
done
"$MC" -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 "$OUT/patched.s" --filetype=obj -o "$OUT/patched.o"
python3 "$HERE/patch_dvgpr.py" "$OUT/patched.o" "${KERNELS// /,}"
echo "    assembled + patched $OUT/patched.o"

echo ">>> done. Static comparison: python3 $HERE/extract_metrics.py $OUT/vanilla.o $OUT/patched.o"
