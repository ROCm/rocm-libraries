#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_three_way.sh -- compile-time comparison on the same gfx950 host of the
# unified-attention 2D kernel (fp16 MHA), three ways:
#   NATIVE C    : ckc_build_* (build IR in C) + lower + comgr      [bench_compare]
#   PORTABLE IR : import per-shape portable IR + lower + comgr     [bench_compare]
#   RECORD+ROLL : expand ONE parametric recipe (spec D) + lower + comgr [bench_recipe]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_threeway"
ARCH="${1:-gfx950}"
ITERS="${2:-10}"
ROCM="${ROCM_PATH:-/opt/rocm}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

echo ">> building"
( cd "$OUT" && cc -std=c99 -O2 -I "$CKC/include" -c "$CKC"/src/*.c 2>/dev/null && ar rcs libckc.a ./*.o ) || { echo lib FAIL; exit 1; }
g++ -std=c++17 -O2 -I "$CKC/include" -I "$ROCM/include" "$HERE/bench_compare.cpp" "$OUT/libckc.a" -L"$ROCM/lib" -lamd_comgr -o "$OUT/bench_compare" 2>/dev/null || { echo bench_compare FAIL; exit 1; }
g++ -std=c++17 -O2 -I "$CKC/include" -I "$ROCM/include" "$HERE/bench_recipe.cpp"  "$OUT/libckc.a" -L"$ROCM/lib" -lamd_comgr -o "$OUT/bench_recipe"  2>/dev/null || { echo bench_recipe FAIL; exit 1; }

echo ">> rolling ONE parametric recipe (covers all D)"
python3 -m ck_dsl.portable_ir.drivers.roll_recipe --emit recipe > "$OUT/param.recipe.json" 2>/dev/null

for D in 64 128 256; do
  echo ""
  echo "== fp16 MHA D$D =="
  python3 -m ck_dsl.portable_ir.examples.export_mha --dtype fp16 --head-size "$D" --gqa 1 --num-heads 32 --arch "$ARCH" > "$OUT/pir_$D.json" 2>/dev/null
  "$OUT/bench_compare" --dtype fp16 --head-size "$D" --nqh 32 --nkv 32 --seqlen 2048 --json "$OUT/pir_$D.json" --arch "$ARCH" --iters "$ITERS"
  "$OUT/bench_recipe" "$OUT/param.recipe.json" "$D" fp16 "$ARCH" "$ITERS"
done
