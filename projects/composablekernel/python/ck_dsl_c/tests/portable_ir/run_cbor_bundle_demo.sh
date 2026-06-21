#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_cbor_bundle_demo.sh -- productization plumbing end-to-end:
#
#   ONE recipe is shipped three ways -- JSON (authoring), a compact CBOR blob,
#   and inside a CBOR BUNDLE (schema ck.dsl.bundle/v1) that packs many recipes
#   keyed by (key, arch). The pure-C recipe VM consumes all three through the
#   same DOM (json_dom / cbor_dom) and the same rv_run_root core. For each D we
#   prove:
#     (1) JSON, CBOR and bundle inputs lower to byte-identical LLVM IR; and
#     (2) the resulting HSACO is byte-identical to the Python reference kernel.
#
# This is the shipping form: the runtime loads one CBOR bundle and serves any
# kernel/shape by key with no CPython and no per-recipe JSON files.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_cbor_bundle_demo"
ARCH="${1:-gfx950}"
ROCM="${ROCM_PATH:-/opt/rocm}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

echo ">> building ckc lib (incl cbor_dom.c) + recipe_run + comgr tool"
( cd "$OUT" && cc -std=c99 -O2 -I "$CKC/include" -c "$CKC"/src/*.c 2>/dev/null && ar rcs libckc.a ./*.o ) || {
    echo "ckc lib build FAILED"; exit 1; }
cc -std=c99 -O2 -I "$CKC/include" "$HERE/recipe_run.c" "$OUT/libckc.a" -lm -o "$OUT/recipe_run" || {
    echo "recipe_run build FAILED"; exit 1; }
HAVE_COMGR=1
cc -std=c99 -O2 -I "$ROCM/include" "$HERE/comgr_compile_ll.c" -L"$ROCM/lib" -lamd_comgr -o "$OUT/comgr" 2>/dev/null || HAVE_COMGR=0

echo ">> emitting recipe (JSON) and packing CBOR + bundle"
python3 -m ck_dsl.portable_ir.recipe_toy --emit recipe > "$OUT/toy.recipe.json"
python3 -m ck_dsl.portable_ir.recipe_bundle encode "$OUT/toy.recipe.json" "$OUT/toy.recipe.cbor"
python3 -m ck_dsl.portable_ir.recipe_bundle bundle "$OUT/bundle.cbor" "$OUT/toy.recipe.json:toy:$ARCH"
js=$(wc -c < "$OUT/toy.recipe.json"); cb=$(wc -c < "$OUT/toy.recipe.cbor"); bn=$(wc -c < "$OUT/bundle.cbor")
echo "   recipe: JSON=${js}B  CBOR=${cb}B  bundle=${bn}B"
echo ""

printf "%-6s  %-18s  %-18s  %-18s  %s\n" "D" "json==cbor==bundle" "vm_hsaco" "ref_hsaco" "match"
rc=0
for D in 64 128 256; do
    "$OUT/recipe_run" "$OUT/toy.recipe.json" --arch "$ARCH" --int "D=$D" --str dtype=f32 > "$OUT/j_$D.ll" 2> "$OUT/j_$D.err" || { echo "JSON FAIL D=$D: $(cat "$OUT/j_$D.err")"; rc=1; continue; }
    "$OUT/recipe_run" "$OUT/toy.recipe.cbor" --cbor --arch "$ARCH" --int "D=$D" --str dtype=f32 > "$OUT/c_$D.ll" 2> "$OUT/c_$D.err" || { echo "CBOR FAIL D=$D: $(cat "$OUT/c_$D.err")"; rc=1; continue; }
    "$OUT/recipe_run" "$OUT/bundle.cbor" --bundle toy --arch "$ARCH" --int "D=$D" --str dtype=f32 > "$OUT/b_$D.ll" 2> "$OUT/b_$D.err" || { echo "BUNDLE FAIL D=$D: $(cat "$OUT/b_$D.err")"; rc=1; continue; }
    ll_match="DIFFER"
    if diff -q "$OUT/j_$D.ll" "$OUT/c_$D.ll" >/dev/null && diff -q "$OUT/j_$D.ll" "$OUT/b_$D.ll" >/dev/null; then ll_match="IDENTICAL"; else rc=1; fi

    if [ "$HAVE_COMGR" = "1" ]; then
        python3 -m ck_dsl.portable_ir.recipe_toy --emit ll --D "$D" --dtype f32 --arch "$ARCH" > "$OUT/ref_$D.ll" 2> "$OUT/ref_$D.err" || { echo "REF FAIL D=$D"; rc=1; continue; }
        vmsz=$("$OUT/comgr" "$OUT/b_$D.ll" "$OUT/b_$D.hsaco" "$ARCH") || { echo "comgr VM FAIL D=$D"; rc=1; continue; }
        refsz=$("$OUT/comgr" "$OUT/ref_$D.ll" "$OUT/ref_$D.hsaco" "$ARCH") || { echo "comgr REF FAIL D=$D"; rc=1; continue; }
        vmsha=$(sha256sum "$OUT/b_$D.hsaco" | cut -d' ' -f1)
        refsha=$(sha256sum "$OUT/ref_$D.hsaco" | cut -d' ' -f1)
        if [ "$vmsha" = "$refsha" ]; then m="IDENTICAL"; else m="DIFFER"; rc=1; fi
        printf "%-6s  %-18s  %-18s  %-18s  %s\n" "$D" "$ll_match" "${vmsz}B" "${refsz}B" "$m"
    else
        printf "%-6s  %-18s  %-18s  %-18s  %s\n" "$D" "$ll_match" "(no comgr)" "(no comgr)" "-"
    fi
done
echo ""

# --------------------------------------------------------------------------
# Part B: the CONCRETE RECORD path -- record a production SET into ONE bundle
# (keyed by kernel name) and have the C VM serve each by key. Exercises the
# multi-result ("outs") VM limit too (ckdsl_multi_result_i32). HSACO is the
# ground truth: each served kernel must match its Python reference byte-for-byte.
# --------------------------------------------------------------------------
echo ">> concrete-record bundle (record many kernels -> one CBOR bundle, serve by key)"
python3 -m ck_dsl.portable_ir.recipe_bundle record-demo "$OUT/concrete.bundle.cbor" --arch "$ARCH"
echo ""
printf "%-32s  %-12s  %-12s  %s\n" "key (kernel)" "vm_hsaco" "ref_hsaco" "match"
# key                              python-reference emitter + args
run_concrete() {
    key="$1"; shift
    "$OUT/recipe_run" "$OUT/concrete.bundle.cbor" --bundle "$key" --arch "$ARCH" \
        > "$OUT/cc_$key.ll" 2> "$OUT/cc_$key.err" || { echo "VM FAIL $key: $(cat "$OUT/cc_$key.err")"; rc=1; return; }
    if [ "$HAVE_COMGR" = "1" ]; then
        python3 -m "$@" --arch "$ARCH" > "$OUT/cc_${key}_ref.ll" 2> "$OUT/cc_${key}_ref.err" || { echo "REF FAIL $key"; rc=1; return; }
        vmsz=$("$OUT/comgr" "$OUT/cc_$key.ll" "$OUT/cc_$key.hsaco" "$ARCH") || { echo "comgr VM FAIL $key"; rc=1; return; }
        refsz=$("$OUT/comgr" "$OUT/cc_${key}_ref.ll" "$OUT/cc_${key}_ref.hsaco" "$ARCH") || { echo "comgr REF FAIL $key"; rc=1; return; }
        vmsha=$(sha256sum "$OUT/cc_$key.hsaco" | cut -d' ' -f1)
        refsha=$(sha256sum "$OUT/cc_${key}_ref.hsaco" | cut -d' ' -f1)
        if [ "$vmsha" = "$refsha" ]; then m="IDENTICAL"; else m="DIFFER"; rc=1; fi
        printf "%-32s  %-12s  %-12s  %s\n" "$key" "${vmsz}B" "${refsz}B" "$m"
    else
        printf "%-32s  %-12s  %-12s  %s\n" "$key" "(no comgr)" "(no comgr)" "lowered-ok"
    fi
}
run_concrete ckdsl_mini_attn_norm0_f32 ck_dsl.portable_ir.mini_attn --emit ll --use-norm 0 --dtype f32
run_concrete ckdsl_mini_attn_norm1_f32 ck_dsl.portable_ir.mini_attn --emit ll --use-norm 1 --dtype f32
run_concrete ckdsl_multi_result_i32    ck_dsl.portable_ir.recipe_multi_result --emit ll --dtype i32
echo ""
[ $rc -eq 0 ] && echo "PASS: CBOR + bundle (rolled and concrete-record) reproduce the Python build exactly (byte-identical)." || echo "FAIL"
exit $rc
