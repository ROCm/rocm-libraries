#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_attn2d_recipe_demo.sh -- assemble the FULL production unified-attention 2D
# kernel into a recipe and verify byte-identical HSACO through the C recipe VM.
#
# The real build_unified_attention_2d (untouched) is run to produce its
# KernelDef; kerneldef_to_recipe converts that exact op stream to a recipe; the
# pure-C recipe VM re-emits + lowers it; comgr compiles both the VM output and
# the production lowering and the HSACOs are compared byte-for-byte.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_attn2d_recipe"
ARCH="${1:-gfx950}"
D="${2:-128}"
DTYPE="${3:-fp16}"
ROCM="${ROCM_PATH:-/opt/rocm}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}"

echo ">> building ckc lib + recipe_run + comgr tool"
( cd "$OUT" && cc -std=c99 -O2 -I "$CKC/include" -c "$CKC"/src/*.c 2>/dev/null && ar rcs libckc.a ./*.o ) || {
    echo "ckc lib build FAILED"; exit 1; }
cc -std=c99 -O2 -I "$CKC/include" "$HERE/recipe_run.c" "$OUT/libckc.a" -lm -o "$OUT/recipe_run" || {
    echo "recipe_run build FAILED"; exit 1; }
cc -std=c99 -O2 -I "$ROCM/include" "$HERE/comgr_compile_ll.c" -L"$ROCM/lib" -lamd_comgr -o "$OUT/comgr" || {
    echo "comgr tool build FAILED"; exit 1; }

echo ""
echo ">> converting production build_unified_attention_2d ($DTYPE D$D) -> recipe"
python3 -m ck_dsl.portable_ir.kerneldef_to_recipe --emit recipe --D "$D" --dtype "$DTYPE" --arch "$ARCH" \
    > "$OUT/attn2d.recipe.json" 2> "$OUT/conv.err" || { echo "convert FAIL"; cat "$OUT/conv.err"; exit 1; }
echo "   recipe: $(wc -c < "$OUT/attn2d.recipe.json") bytes"

echo ">> recipe VM -> .ll"
"$OUT/recipe_run" "$OUT/attn2d.recipe.json" --arch "$ARCH" > "$OUT/vm.ll" 2> "$OUT/vm.err" || {
    echo "VM FAIL: $(cat "$OUT/vm.err")"; exit 1; }
echo ">> production lower -> .ll"
python3 -m ck_dsl.portable_ir.kerneldef_to_recipe --emit ll --D "$D" --dtype "$DTYPE" --arch "$ARCH" \
    > "$OUT/ref.ll" 2> "$OUT/ref.err" || { echo "REF FAIL"; cat "$OUT/ref.err"; exit 1; }

echo ">> comgr both -> HSACO"
vmsz=$("$OUT/comgr" "$OUT/vm.ll" "$OUT/vm.hsaco" "$ARCH") || { echo "comgr VM FAIL"; exit 1; }
refsz=$("$OUT/comgr" "$OUT/ref.ll" "$OUT/ref.hsaco" "$ARCH") || { echo "comgr REF FAIL"; exit 1; }
vmsha=$(sha256sum "$OUT/vm.hsaco" | cut -d' ' -f1)
refsha=$(sha256sum "$OUT/ref.hsaco" | cut -d' ' -f1)
echo ""
echo "production .ll lines : $(wc -l < "$OUT/ref.ll")"
echo "recipe-VM   HSACO    : ${vmsz}B  sha=${vmsha:0:16}"
echo "production  HSACO    : ${refsz}B  sha=${refsha:0:16}"
echo ""
if [ "$vmsha" = "$refsha" ]; then
    echo "PASS: full unified-attention 2D ($DTYPE D$D) recipe -> C VM HSACO byte-identical to production."
    exit 0
else
    echo "FAIL: HSACO differs"
    exit 1
fi
