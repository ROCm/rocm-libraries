#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_qk_block_demo.sh -- prove the head_size-parametric crux of the attention
# kernel records into ONE recipe that the C VM expands per head dim, byte-
# identical to the Python reference.
#
# qk_block mirrors the production QK vec8 dot-product (the reason head_size is
# structural). The recorder authors it once with a rolled static_for over
# spec(D)//8 and a parametric head stride; the C VM expands it for D in
# {64,128,256}; each HSACO is compared to the Python reference build_qk_block(D).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_qk_block"
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

python3 "$HERE/qk_block.py" --emit recipe > "$OUT/qk.recipe.json"
echo ""
echo "ONE recipe artifact: $(wc -c < "$OUT/qk.recipe.json") bytes (rolled static_for over spec D//8)"
echo ""
printf "%-6s  %-12s  %-12s  %-10s  %-12s\n" "D" "vm_hsaco" "ref_hsaco" "match" "ref_ll_lines"
rc=0
for D in 64 128 256; do
    "$OUT/recipe_run" "$OUT/qk.recipe.json" --arch "$ARCH" --int "D=$D" --str dtype=f16 \
        > "$OUT/vm_$D.ll" 2> "$OUT/vm_$D.err" || { echo "VM FAIL D=$D: $(cat "$OUT/vm_$D.err")"; rc=1; continue; }
    python3 "$HERE/qk_block.py" --emit ll --D "$D" --dtype f16 --arch "$ARCH" \
        > "$OUT/ref_$D.ll" 2> "$OUT/ref_$D.err" || { echo "REF FAIL D=$D"; rc=1; continue; }
    "$OUT/comgr" "$OUT/vm_$D.ll" "$OUT/vm_$D.hsaco" "$ARCH" >/dev/null || { echo "comgr VM FAIL"; rc=1; continue; }
    "$OUT/comgr" "$OUT/ref_$D.ll" "$OUT/ref_$D.hsaco" "$ARCH" >/dev/null || { echo "comgr REF FAIL"; rc=1; continue; }
    vmsha=$(sha256sum "$OUT/vm_$D.hsaco" | cut -d' ' -f1)
    refsha=$(sha256sum "$OUT/ref_$D.hsaco" | cut -d' ' -f1)
    lines=$(wc -l < "$OUT/ref_$D.ll")
    if [ "$vmsha" = "$refsha" ]; then m="IDENTICAL"; else m="DIFFER"; rc=1; fi
    printf "%-6s  %-12s  %-12s  %-10s  %-12s\n" "$D" "$(wc -c < "$OUT/vm_$D.hsaco")B" "$(wc -c < "$OUT/ref_$D.hsaco")B" "$m" "$lines"
done
echo ""
[ $rc -eq 0 ] && echo "PASS: one rolled recipe -> D64/D128/D256, each HSACO byte-identical to the Python reference." || echo "FAIL"
exit $rc
