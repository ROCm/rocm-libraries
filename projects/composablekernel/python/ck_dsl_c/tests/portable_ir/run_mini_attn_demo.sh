#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_mini_attn_demo.sh -- evaluate whether the recipe VM ISA can express the
# attention kernel's control structure. mini_attn uses a RUNTIME scf.for
# online-softmax accumulation, a RUNTIME scf.if store guard, and a COMPILE-TIME
# static_if on the spec `use_norm` -- the same structural features the real
# unified-attention 2D scalar kernel uses.
#
# ONE recipe specializes on use_norm in {0,1} at JIT time (pure C); each
# comgr-compiled HSACO is compared byte-for-byte to the Python-authored
# reference (build_mini_attn).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_mini_attn"
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

python3 "$HERE/mini_attn.py" --emit recipe > "$OUT/mini_attn.recipe.json"
echo ""
echo "ONE recipe artifact: $(wc -c < "$OUT/mini_attn.recipe.json") bytes (covers use_norm=0 and 1)"
echo "  features exercised: runtime scf.for (3 iter-args) + runtime scf.if + compile-time static_if + exp2/fmax/rcp"
echo ""
printf "%-10s  %-12s  %-12s  %-10s  %s\n" "use_norm" "vm_hsaco" "ref_hsaco" "match" "vm_ll_lines"
rc=0
for U in 0 1; do
    "$OUT/recipe_run" "$OUT/mini_attn.recipe.json" --arch "$ARCH" --int "use_norm=$U" --str dtype=f32 \
        > "$OUT/vm_$U.ll" 2> "$OUT/vm_$U.err" || { echo "VM FAIL use_norm=$U: $(cat "$OUT/vm_$U.err")"; rc=1; continue; }
    python3 "$HERE/mini_attn.py" --emit ll --use-norm "$U" --dtype f32 --arch "$ARCH" \
        > "$OUT/ref_$U.ll" 2> "$OUT/ref_$U.err" || { echo "REF FAIL use_norm=$U"; rc=1; continue; }
    vmsz=$("$OUT/comgr" "$OUT/vm_$U.ll" "$OUT/vm_$U.hsaco" "$ARCH") || { echo "comgr VM FAIL"; rc=1; continue; }
    refsz=$("$OUT/comgr" "$OUT/ref_$U.ll" "$OUT/ref_$U.hsaco" "$ARCH") || { echo "comgr REF FAIL"; rc=1; continue; }
    vmsha=$(sha256sum "$OUT/vm_$U.hsaco" | cut -d' ' -f1)
    refsha=$(sha256sum "$OUT/ref_$U.hsaco" | cut -d' ' -f1)
    lines=$(wc -l < "$OUT/vm_$U.ll")
    if [ "$vmsha" = "$refsha" ]; then m="IDENTICAL"; else m="DIFFER"; rc=1; fi
    printf "%-10s  %-12s  %-12s  %-10s  %s\n" "$U" "${vmsz}B" "${refsz}B" "$m" "$lines"
done
echo ""
[ $rc -eq 0 ] && echo "PASS: one recipe -> use_norm {0,1}, each HSACO byte-identical to the Python reference." || echo "FAIL"
exit $rc
