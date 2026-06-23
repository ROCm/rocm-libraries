#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_attn2d_parametric_demo.sh -- land #2 end-to-end with the storage benefit:
# ONE parametric recipe (rolled over head_size) for the unified-attention 2D
# kernel, expanded by the C VM per D, verified HSACO byte-identical to
# production for D64/D128/D256.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_attn2d_param"
ARCH="${1:-gfx950}"
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

echo ">> rolling production attention-2D into ONE parametric recipe"
python3 -m ck_dsl.portable_ir.drivers.roll_recipe --emit recipe > "$OUT/attn2d_param.recipe.json" 2> "$OUT/roll.err" || {
    echo "roll FAIL"; cat "$OUT/roll.err"; exit 1; }
echo "   parametric recipe: $(wc -c < "$OUT/attn2d_param.recipe.json") bytes (covers all D)"
echo ""
printf "%-6s  %-12s  %-12s  %-10s  %s\n" "D" "vm_hsaco" "ref_hsaco" "match" "concrete_recipe_KB"
rc=0
for D in 64 128 256; do
    "$OUT/recipe_run" "$OUT/attn2d_param.recipe.json" --arch "$ARCH" --int "D=$D" --str dtype=fp16 \
        > "$OUT/vm_$D.ll" 2> "$OUT/vm_$D.err" || { echo "VM FAIL D=$D: $(cat "$OUT/vm_$D.err")"; rc=1; continue; }
    python3 -m ck_dsl.portable_ir.src.kerneldef_to_recipe --emit ll --D "$D" --dtype fp16 --arch "$ARCH" \
        > "$OUT/ref_$D.ll" 2> "$OUT/ref_$D.err" || { echo "REF FAIL D=$D"; rc=1; continue; }
    "$OUT/comgr" "$OUT/vm_$D.ll" "$OUT/vm_$D.hsaco" "$ARCH" >/dev/null || { echo "comgr VM FAIL D=$D"; rc=1; continue; }
    "$OUT/comgr" "$OUT/ref_$D.ll" "$OUT/ref_$D.hsaco" "$ARCH" >/dev/null || { echo "comgr REF FAIL D=$D"; rc=1; continue; }
    vmsha=$(sha256sum "$OUT/vm_$D.hsaco" | cut -d' ' -f1)
    refsha=$(sha256sum "$OUT/ref_$D.hsaco" | cut -d' ' -f1)
    # size of the per-shape concrete recipe for contrast
    ckb=$(python3 -m ck_dsl.portable_ir.src.kerneldef_to_recipe --emit recipe --D "$D" --dtype fp16 2>/dev/null | wc -c)
    if [ "$vmsha" = "$refsha" ]; then m="IDENTICAL"; else m="DIFFER"; rc=1; fi
    printf "%-6s  %-12s  %-12s  %-10s  %s\n" "$D" "$(wc -c <"$OUT/vm_$D.hsaco")B" "$(wc -c <"$OUT/ref_$D.hsaco")B" "$m" "$((ckb/1024))"
done
echo ""
[ $rc -eq 0 ] && echo "PASS: ONE parametric recipe -> D64/D128/D256, each HSACO byte-identical to production." || echo "FAIL"
exit $rc
