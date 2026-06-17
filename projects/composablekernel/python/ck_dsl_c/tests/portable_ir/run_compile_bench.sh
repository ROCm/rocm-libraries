#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_compile_bench.sh -- compile-time benchmark of the portable-IR path across
# the MHA/SDPA variants from dsl_docs/architecture/SDPA_CKDSL_Provider_Comparison.
#
# Each row exports the Python-authored attention kernel to portable IR, then
# times the online provider steps in C: import (JSON->IR) + lower (IR->.ll) +
# comgr (.ll->gfx950 HSACO), N samples, median reported. This is the same
# "cold compile per shape" metric as the comparison doc, but the kernel source
# is now a Python-authored portable-IR artifact lowered by the C backend.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_compile_bench"
ARCH="${1:-gfx950}"
ITERS="${2:-10}"
ROCM="${ROCM_PATH:-/opt/rocm}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

echo ">> building ckc static lib + bench (links libamd_comgr)"
( cd "$OUT" && cc -std=c99 -O2 -I "$CKC/include" -c "$CKC"/src/*.c 2>/dev/null && ar rcs libckc.a ./*.o ) || {
    echo "ckc lib build FAILED"; exit 1; }
g++ -std=c++17 -O2 -I "$CKC/include" -I "$ROCM/include" "$HERE/bench_compile.cpp" \
    "$OUT/libckc.a" -L"$ROCM/lib" -lamd_comgr -o "$OUT/bench" || {
    echo "bench build FAILED (need libamd_comgr at $ROCM/lib)"; exit 1; }

# (dtype, head_size, gqa-ratio, label) covering the comparison's dense families.
VARIANTS=(
  "fp16 64  1  CorrectnessD64"
  "fp16 128 1  PrefillMHA_D128"
  "fp16 256 1  PrefillD256"
  "bf16 128 1  PrefillD128_bf16"
  "fp16 128 4  Llama3_GQA4_D128"
  "fp16 128 8  GQA8_D128"
  "bf16 256 4  LargeHead_GQA4_D256"
)

echo ""
echo "== Portable-IR compile time (Python-authored IR -> C import -> lower -> comgr gfx950) =="
echo "   median of $ITERS samples; warm-up discarded"
echo ""
for v in "${VARIANTS[@]}"; do
  set -- $v
  dt="$1"; hd="$2"; gqa="$3"; label="$4"
  json="$OUT/${label}.ir.json"
  python3 "$HERE/export_mha.py" --dtype "$dt" --head-size "$hd" --gqa "$gqa" --num-heads 32 \
      --arch "$ARCH" > "$json" 2>"$OUT/${label}.err" || { echo "export FAIL $label"; cat "$OUT/${label}.err"; continue; }
  printf "%-22s " "$label"
  "$OUT/bench" "$json" "$ARCH" "$ITERS"
done
