#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_gemm_cshuffle_demo.sh -- prove the universal GEMM (CShuffle epilogue) rolls
# over tile_n into ONE parametric recipe that the C VM expands per tile_n,
# HSACO byte-identical to the Python reference.
#
# This exercises the full parametric VM surface: rolled scf.for iter-args/results
# (the variable loop-carry fan), format register names (acc_m{lane}_n0), rolled
# scf.yield operands, intexpr in attrs (sched_group_barrier count) and in result
# TYPES (smem buffer shape), and exact smem-alloc LDS naming.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_gemm_cshuffle"
ARCH="${1:-gfx950}"
ROCM="${ROCM_PATH:-/opt/rocm}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}"

echo ">> building ckc lib + recipe_run + comgr tool"
( cd "$OUT" && cc -std=c99 -O2 -I "$CKC/include" -c "$CKC"/src/portable_ir/*.c 2>/dev/null && ar rcs libckc.a ./*.o ) || {
    echo "ckc lib build FAILED"; exit 1; }
cc -std=c99 -O2 -I "$CKC/include" "$HERE/recipe_run.c" "$OUT/libckc.a" -lm -o "$OUT/recipe_run" || {
    echo "recipe_run build FAILED"; exit 1; }
cc -std=c99 -O2 -I "$ROCM/include" "$HERE/comgr_compile_ll.c" -L"$ROCM/lib" -lamd_comgr -o "$OUT/comgr" || {
    echo "comgr tool build FAILED"; exit 1; }

python3 -m ck_dsl.portable_ir.examples.export_gemm_cshuffle --emit recipe > "$OUT/gemm.recipe.json"
echo ""
echo "ONE recipe artifact: $(wc -c < "$OUT/gemm.recipe.json") bytes (covers all tile_n)"
echo ""
printf "%-6s  %-12s  %-12s  %-10s\n" "TN" "vm_hsaco" "ref_hsaco" "match"
rc=0
for TN in 32 64 128 256; do
    "$OUT/recipe_run" "$OUT/gemm.recipe.json" --arch "$ARCH" --int "TN=$TN" \
        > "$OUT/vm_$TN.ll" 2> "$OUT/vm_$TN.err" || { echo "VM FAIL TN=$TN: $(cat "$OUT/vm_$TN.err")"; rc=1; continue; }
    python3 -m ck_dsl.portable_ir.examples.export_gemm_cshuffle --emit ll --TN "$TN" --arch "$ARCH" \
        > "$OUT/ref_$TN.ll" 2> "$OUT/ref_$TN.err" || { echo "REF FAIL TN=$TN"; rc=1; continue; }
    "$OUT/comgr" "$OUT/vm_$TN.ll" "$OUT/vm_$TN.hsaco" "$ARCH" >/dev/null || { echo "comgr VM FAIL"; rc=1; continue; }
    "$OUT/comgr" "$OUT/ref_$TN.ll" "$OUT/ref_$TN.hsaco" "$ARCH" >/dev/null || { echo "comgr REF FAIL"; rc=1; continue; }
    vmsha=$(sha256sum "$OUT/vm_$TN.hsaco" | cut -d' ' -f1)
    refsha=$(sha256sum "$OUT/ref_$TN.hsaco" | cut -d' ' -f1)
    if [ "$vmsha" = "$refsha" ]; then m="IDENTICAL"; else m="DIFFER"; rc=1; fi
    printf "%-6s  %-12s  %-12s  %-10s\n" "$TN" "$(wc -c < "$OUT/vm_$TN.hsaco")B" "$(wc -c < "$OUT/ref_$TN.hsaco")B" "$m"
done
echo ""
[ $rc -eq 0 ] && echo "PASS: one rolled GEMM-CShuffle recipe -> tile_n {32,64,128,256}, each HSACO byte-identical to production." || echo "FAIL"
exit $rc
