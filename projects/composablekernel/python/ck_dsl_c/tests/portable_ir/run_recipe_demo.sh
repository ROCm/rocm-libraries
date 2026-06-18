#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_recipe_demo.sh -- demonstrate the builder-recipe / mini-VM path:
#   Python authoring convenience + compact per-builder artifact
#   + runtime shape flexibility  + no CPython in the JIT.
#
# ONE recipe artifact (a few KB) is emitted from Python. The pure-C recipe VM
# then specializes it on D at runtime (D in {64,128,256}) and lowers + comgr-
# compiles each to a gfx950 HSACO. For each D, the recipe-VM HSACO is compared
# byte-for-byte against the HSACO of the equivalent Python-authored reference
# kernel (build_toy), proving the VM reproduces the Python build exactly.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_recipe_demo"
ARCH="${1:-gfx950}"
ROCM="${ROCM_PATH:-/opt/rocm}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

echo ">> building ckc lib + recipe_run + comgr tool"
( cd "$OUT" && cc -std=c99 -O2 -I "$CKC/include" -c "$CKC"/src/*.c 2>/dev/null && ar rcs libckc.a ./*.o ) || {
    echo "ckc lib build FAILED"; exit 1; }
cc -std=c99 -O2 -I "$CKC/include" "$HERE/recipe_run.c" "$OUT/libckc.a" -lm -o "$OUT/recipe_run" || {
    echo "recipe_run build FAILED"; exit 1; }
cc -std=c99 -O2 -I "$ROCM/include" "$HERE/comgr_compile_ll.c" -L"$ROCM/lib" -lamd_comgr -o "$OUT/comgr" || {
    echo "comgr tool build FAILED"; exit 1; }

python3 "$HERE/recipe_toy.py" --emit recipe > "$OUT/toy.recipe.json"
echo ""
echo "ONE recipe artifact: $(wc -c < "$OUT/toy.recipe.json") bytes (covers every D)"
echo ""
printf "%-6s  %-12s  %-12s  %-10s  %s\n" "D" "vm_hsaco" "ref_hsaco" "match" "vm_ll_lines"
rc=0
for D in 64 128 256; do
    # Recipe VM (pure C): one recipe + runtime D -> specialized kernel -> .ll
    "$OUT/recipe_run" "$OUT/toy.recipe.json" --arch "$ARCH" --int "D=$D" --str dtype=f32 > "$OUT/vm_$D.ll" 2> "$OUT/vm_$D.err" || { echo "VM FAIL D=$D: $(cat "$OUT/vm_$D.err")"; rc=1; continue; }
    # Python reference kernel -> .ll
    python3 "$HERE/recipe_toy.py" --emit ll --D "$D" --dtype f32 --arch "$ARCH" > "$OUT/ref_$D.ll" 2> "$OUT/ref_$D.err" || { echo "REF FAIL D=$D"; rc=1; continue; }
    # comgr-compile both to HSACO and compare bytes.
    vmsz=$("$OUT/comgr" "$OUT/vm_$D.ll" "$OUT/vm_$D.hsaco" "$ARCH") || { echo "comgr VM FAIL D=$D"; rc=1; continue; }
    refsz=$("$OUT/comgr" "$OUT/ref_$D.ll" "$OUT/ref_$D.hsaco" "$ARCH") || { echo "comgr REF FAIL D=$D"; rc=1; continue; }
    vmsha=$(sha256sum "$OUT/vm_$D.hsaco" | cut -d' ' -f1)
    refsha=$(sha256sum "$OUT/ref_$D.hsaco" | cut -d' ' -f1)
    lines=$(wc -l < "$OUT/vm_$D.ll")
    if [ "$vmsha" = "$refsha" ]; then m="IDENTICAL"; else m="DIFFER"; rc=1; fi
    printf "%-6s  %-12s  %-12s  %-10s  %s\n" "$D" "${vmsz}B" "${refsz}B" "$m" "$lines"
done
echo ""
[ $rc -eq 0 ] && echo "PASS: one recipe -> D64/D128/D256, each HSACO byte-identical to the Python reference." || echo "FAIL"
exit $rc
