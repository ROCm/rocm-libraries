#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_recorder_demo.sh -- prove the Python recorder auto-generates a correct
# recipe. mini_attn is authored idiomatically against recipe_recorder.py (no
# hand-written recipe JSON); the recorder emits the recipe; the pure-C VM
# specializes it on use_norm at JIT time; each HSACO is compared byte-for-byte
# to the Python reference kernel (mini_attn.py::build_mini_attn).
#
# Also reports the recorded recipe size vs the hand-written one (they should be
# functionally equivalent; HSACO is the ground truth).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_recorder"
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

# Recorder auto-emits the recipe from idiomatic authoring; compare size to the
# hand-written recipe.
python3 -m ck_dsl.portable_ir.recipe_recorder --emit recipe > "$OUT/recorded.recipe.json"
python3 -m ck_dsl.portable_ir.mini_attn       --emit recipe > "$OUT/handwritten.recipe.json"
echo ""
echo "recorded recipe:    $(wc -c < "$OUT/recorded.recipe.json") bytes  (auto-emitted from idiomatic authoring)"
echo "hand-written recipe:$(wc -c < "$OUT/handwritten.recipe.json") bytes"
echo ""
printf "%-10s  %-12s  %-12s  %-10s\n" "use_norm" "vm_hsaco" "ref_hsaco" "match"
rc=0
for U in 0 1; do
    "$OUT/recipe_run" "$OUT/recorded.recipe.json" --arch "$ARCH" --int "use_norm=$U" --str dtype=f32 \
        > "$OUT/vm_$U.ll" 2> "$OUT/vm_$U.err" || { echo "VM FAIL use_norm=$U: $(cat "$OUT/vm_$U.err")"; rc=1; continue; }
    python3 -m ck_dsl.portable_ir.mini_attn --emit ll --use-norm "$U" --dtype f32 --arch "$ARCH" \
        > "$OUT/ref_$U.ll" 2> "$OUT/ref_$U.err" || { echo "REF FAIL"; rc=1; continue; }
    "$OUT/comgr" "$OUT/vm_$U.ll" "$OUT/vm_$U.hsaco" "$ARCH" >/dev/null || { echo "comgr VM FAIL"; rc=1; continue; }
    "$OUT/comgr" "$OUT/ref_$U.ll" "$OUT/ref_$U.hsaco" "$ARCH" >/dev/null || { echo "comgr REF FAIL"; rc=1; continue; }
    vmsha=$(sha256sum "$OUT/vm_$U.hsaco" | cut -d' ' -f1)
    refsha=$(sha256sum "$OUT/ref_$U.hsaco" | cut -d' ' -f1)
    if [ "$vmsha" = "$refsha" ]; then m="IDENTICAL"; else m="DIFFER"; rc=1; fi
    printf "%-10s  %-12s  %-12s  %-10s\n" "$U" "$(wc -c < "$OUT/vm_$U.hsaco")B" "$(wc -c < "$OUT/ref_$U.hsaco")B" "$m"
done
echo ""
[ $rc -eq 0 ] && echo "PASS: recorder-emitted recipe -> use_norm {0,1}, HSACO byte-identical to the Python reference." || echo "FAIL"
exit $rc
