#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_compare.sh -- head-to-head compile-time comparison (same host, same
# lower+comgr path) of the pure C-interface (native C build) vs the portable-IR
# import path, over the MHA/SDPA variants from Section 3 (kernel performance vs
# AOTriton) of dsl_docs/architecture/SDPA_CKDSL_Provider_Comparison 1.md.
#
# For each variant: export the Python-authored kernel to portable IR, then time
# both paths -> lower -> comgr (gfx950 HSACO). The two .ll are asserted
# byte-identical, so the only thing the timings isolate is "native C build" vs
# "JSON import".
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_compare"
ARCH="${1:-gfx950}"
ITERS="${2:-10}"
ROCM="${ROCM_PATH:-/opt/rocm}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

echo ">> building ckc static lib + bench"
( cd "$OUT" && cc -std=c99 -O2 -I "$CKC/include" -c "$CKC"/src/portable_ir/*.c 2>/dev/null && ar rcs libckc.a ./*.o ) || {
    echo "ckc lib build FAILED"; exit 1; }
g++ -std=c++17 -O2 -I "$CKC/include" -I "$ROCM/include" "$HERE/bench_compare.cpp" \
    "$OUT/libckc.a" -L"$ROCM/lib" -lamd_comgr -o "$OUT/bench" || {
    echo "bench build FAILED"; exit 1; }

# Section-3 variants -> (label, dtype, head_size, gqa_ratio, seqlen, batch). num
# query heads fixed at 32; num_kv_heads = 32 / gqa_ratio.
VARIANTS=(
  "InFamily_GQA8_D64_S2016_B32_bf16       bf16 64  8 2016 32"
  "Prefill_GQA8_D64_S2048_fp16            fp16 64  8 2048 1"
  "Prefill_GQA8_D128_S2048_fp16           fp16 128 8 2048 1"
  "Prefill_MHA_D128_S2048_fp16            fp16 128 1 2048 1"
  "Prefill_GQA8_D128_S2048_bf16           bf16 128 8 2048 1"
  "Prefill_GQA8_D256_S2048_fp16           fp16 256 8 2048 1"
  "Prefill_GQA8_D128_B4_S2048_fp16        fp16 128 8 2048 4"
  "Prefill_GQA8_D128_B8_S2048_fp16        fp16 128 8 2048 8"
  "Prefill_GQA8_D128_S4096_fp16           fp16 128 8 4096 1"
  "Prefill_GQA8_D128_B4_S4096_fp16        fp16 128 8 4096 4"
  "Prefill_GQA8_D128_S8192_fp16           fp16 128 8 8192 1"
  "Llama2_7B_MHA_D128_S2048               fp16 128 1 2048 1"
  "Llama3_8B_GQA4_D128_S4096_bf16         bf16 128 4 4096 1"
  "Llama3_70B_GQA8_D128_S8192_bf16        bf16 128 8 8192 1"
  "LargeHead_GQA4_D256_S2048_bf16         bf16 256 4 2048 1"
  "TrainingMicrobatch_GQA4_D128_B8_bf16   bf16 128 4 2048 8"
)

for v in "${VARIANTS[@]}"; do
  set -- $v
  label="$1"; dt="$2"; hd="$3"; gqa="$4"; s="$5"; b="$6"
  nkv=$(( 32 / gqa ))
  json="$OUT/${label}.ir.json"
  python3 -m ck_dsl.portable_ir.examples.export_mha --dtype "$dt" --head-size "$hd" --num-heads 32 --gqa "$gqa" \
      --seqlen "$s" --batch "$b" --arch "$ARCH" > "$json" 2>"$OUT/${label}.err" || { echo "export FAIL $label"; cat "$OUT/${label}.err"; continue; }
  echo ""
  echo "== $label  ($dt D$hd  qh32/kv$nkv  B$b S$s) =="
  "$OUT/bench" --dtype "$dt" --head-size "$hd" --nqh 32 --nkv "$nkv" --seqlen "$s" \
      --batch "$b" \
      --json "$json" --arch "$ARCH" --iters "$ITERS"
done
