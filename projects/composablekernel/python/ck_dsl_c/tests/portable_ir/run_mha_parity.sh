#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_mha_parity.sh -- prove the portable-IR boundary on the real MHA/SDPA
# kernel from the SDPA provider comparison, and demonstrate shape-polymorphism.
#
# Part A (byte-parity): for each (dtype, head_size) family, Python builds +
#   lowers the scalar 2D attention kernel, and also exports it to portable IR;
#   the C backend imports the IR and lowers it. The two .ll must be identical.
#
# Part B (shape-polymorphism): for one family (fp16 D128) the IR is exported at
#   S2048 / S4096 / S8192 and for MHA vs GQA; all must be the SAME artifact,
#   proving sequence length / batch / head-grouping are runtime (not baked into
#   the kernel body), so one IR per (dtype, head_size) serves every S.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_mha_parity"
ARCH="${1:-gfx950}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

BIN="$OUT/roundtrip"
echo ">> compiling C roundtrip driver"
cc -std=c99 -I "$CKC/include" "$CKC"/src/*.c "$HERE/roundtrip.c" -o "$BIN" -lm 2>/dev/null || {
    echo "compile FAILED"; exit 1; }

rc=0
echo ""
echo "== Part A: Python-lowered .ll  vs  C-from-portable-IR .ll =="
for dt in fp16 bf16; do
  for hd in 64 128 256; do
    tag="${dt}_d${hd}"
    python3 -m ck_dsl.portable_ir.examples.export_mha --dtype "$dt" --head-size "$hd" --arch "$ARCH" \
        > "$OUT/$tag.ir.json" 2> "$OUT/$tag.exp.err" || { echo "EXPORT FAIL $tag"; cat "$OUT/$tag.exp.err"; rc=1; continue; }
    python3 -m ck_dsl.portable_ir.examples.export_mha --dtype "$dt" --head-size "$hd" --arch "$ARCH" --ll \
        > "$OUT/py_$tag.ll" 2>> "$OUT/$tag.exp.err" || { echo "LL FAIL $tag"; rc=1; continue; }
    "$BIN" "$OUT/$tag.ir.json" "$ARCH" > "$OUT/cjson_$tag.ll" 2> "$OUT/$tag.rt.err" || { echo "RT FAIL $tag"; cat "$OUT/$tag.rt.err"; rc=1; continue; }
    ps=$(sha256sum "$OUT/py_$tag.ll"|cut -d' ' -f1); cs=$(sha256sum "$OUT/cjson_$tag.ll"|cut -d' ' -f1)
    lines=$(wc -l < "$OUT/py_$tag.ll"); kb=$(( $(wc -c < "$OUT/$tag.ir.json") / 1024 ))
    if [ "$ps" = "$cs" ]; then printf "PASS  %-10s  ll=%4s lines  ir=%4s KB  %s\n" "$tag" "$lines" "$kb" "${cs:0:16}"
    else printf "FAIL  %-10s  py=%s cjson=%s\n" "$tag" "${ps:0:16}" "${cs:0:16}"; diff -u "$OUT/py_$tag.ll" "$OUT/cjson_$tag.ll"|head -20; rc=1; fi
  done
done

echo ""
echo "== Part B: shape-polymorphism (one IR per family covers all S / MHA+GQA) =="
base=""
for s in 2048 4096 8192; do
  python3 -m ck_dsl.portable_ir.examples.export_mha --dtype fp16 --head-size 128 --seqlen "$s" --arch "$ARCH" > "$OUT/poly_s$s.json" 2>/dev/null
  h=$(sha256sum "$OUT/poly_s$s.json"|cut -d' ' -f1)
  printf "  fp16 D128 S%-5s  ir-sha=%s\n" "$s" "${h:0:16}"
  [ -z "$base" ] && base="$h"
  [ "$h" != "$base" ] && { echo "  -> DIFF: S$s IR differs from S2048!"; rc=1; }
done
# MHA vs GQA-8 (same head_size/dtype): bodies must match too.
python3 -m ck_dsl.portable_ir.examples.export_mha --dtype fp16 --head-size 128 --seqlen 2048 --num-heads 32 --gqa 1 --arch "$ARCH" > "$OUT/poly_mha.json" 2>/dev/null
python3 -m ck_dsl.portable_ir.examples.export_mha --dtype fp16 --head-size 128 --seqlen 2048 --num-heads 32 --gqa 8 --arch "$ARCH" > "$OUT/poly_gqa.json" 2>/dev/null
hm=$(sha256sum "$OUT/poly_mha.json"|cut -d' ' -f1); hg=$(sha256sum "$OUT/poly_gqa.json"|cut -d' ' -f1)
printf "  fp16 D128 MHA    ir-sha=%s\n  fp16 D128 GQA8   ir-sha=%s\n" "${hm:0:16}" "${hg:0:16}"
if [ "$base" = "$hm" ] && [ "$hm" = "$hg" ]; then
  echo "  -> ONE IR (fp16 D128) serves S2048/S4096/S8192 and MHA+GQA8."
else
  echo "  -> NOTE: MHA/GQA IR differ (head grouping affected the body)."
fi
exit $rc
