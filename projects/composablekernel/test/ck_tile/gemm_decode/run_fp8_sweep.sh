#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# FP8 per-tensor head-to-head: drive all four M-sweep harnesses across the
# decode shape set and emit one fp8_compare table per shape, then aggregate.
#
#   gemm_decode_fp8_best   bench_gemm_decode_msweep_fp8   (warp-per-scalar, the SUT)
#   aiter_wvsplitkq        wvsplitk_msweep.py --fp8       (VALU peer)
#   aiter_gemm_a8w8_ck     wvsplitk_msweep.py --fp8       (classic-CK MFMA fallback)
#   ck_gemm_quant_tensor   gemm_quant_tensor_msweep.py    (CKTile MFMA M=16 ceiling)
#
# All harnesses run WARM (buffer reuse, no cache flush) so GB/s is comparable.
# Usage:
#   ./run_fp8_sweep.sh [OUTDIR]
# Env overrides: SHAPES_N, K, MMAX, WARMUP, REPEAT, AITER_DIR, BIN_DIR, PY.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CK_ROOT="$(cd "$HERE/../../.." && pwd)"

OUTDIR="${1:-/tmp/fp8_sweep_$(date +%Y%m%d_%H%M%S)}"
BIN_DIR="${BIN_DIR:-$CK_ROOT/build/bin}"
PY="${PY:-/opt/venv/bin/python3}"
AITER_DIR="${AITER_DIR:-/home/AMD/samremes/dev/aiter}"
SHAPES_N="${SHAPES_N:-2048 4096 7168 8192 16384}"
K="${K:-7168}"
MMAX="${MMAX:-8}"
WARMUP="${WARMUP:-25}"
REPEAT="${REPEAT:-200}"

GD_BIN="$BIN_DIR/bench_gemm_decode_msweep_fp8"
CKQ_BIN="$BIN_DIR/tile_example_gemm_quant"
mkdir -p "$OUTDIR"

echo "== FP8 per-tensor sweep =="
echo "   out=$OUTDIR  N={$SHAPES_N} K=$K Mmax=$MMAX warmup=$WARMUP repeat=$REPEAT"
echo "   gd_bin=$GD_BIN"
echo "   ckq_bin=$CKQ_BIN"
echo "   aiter=$AITER_DIR  py=$PY"
[ -x "$GD_BIN" ] || { echo "FATAL: $GD_BIN missing/not built"; exit 1; }

run_step() {  # label logfile cmd...
  local label="$1" log="$2"; shift 2
  echo "  -> $label"
  if "$@" >"$log" 2>&1; then echo "     ok ($log)"; else
    echo "     FAILED rc=$? (see $log); tail:"; tail -8 "$log" | sed 's/^/        /'
  fi
}

SUMMARY="$OUTDIR/summary.md"
: >"$SUMMARY"

for N in $SHAPES_N; do
  echo; echo "==== N=$N K=$K ===="
  GD="$OUTDIR/gd_${N}_${K}.csv"
  AI="$OUTDIR/aiter_${N}_${K}.csv"
  CKQ="$OUTDIR/ckq_${N}_${K}.csv"
  MD="$OUTDIR/cmp_${N}_${K}.md"

  # 1) gemm_decode FP8 autotuned sweep (positional: warmup repeat N K Mmax)
  run_step "gemm_decode FP8 sweep" "$OUTDIR/gd_${N}_${K}.log" \
    bash -c "'$GD_BIN' $WARMUP $REPEAT $N $K $MMAX > '$GD'"

  # 2) AITER FP8 competitors (wvSplitKQ + gemm_a8w8_CK). First call may JIT-build.
  run_step "AITER wvSplitKQ + gemm_a8w8_CK" "$OUTDIR/aiter_${N}_${K}.log" \
    "$PY" "$HERE/wvsplitk_msweep.py" --fp8 --aiter-dir "$AITER_DIR" \
      --N "$N" --K "$K" --mmax "$MMAX" --warmup 10 --repeat 100 --csv-out "$AI"

  # 3) CKTile gemm_quant TensorQuant (MFMA M=16 ceiling)
  run_step "CKTile gemm_quant tensor" "$OUTDIR/ckq_${N}_${K}.log" \
    "$PY" "$HERE/gemm_quant_tensor_msweep.py" --exe "$CKQ_BIN" \
      --N "$N" --K "$K" --mmax "$MMAX" --warmup "$WARMUP" --repeat "$REPEAT" \
      --csv-out "$CKQ"

  # 4) Join + verdict
  CMP_ARGS=(--gemm-decode-csv "$GD" --mmax "$MMAX" --md-out "$MD")
  [ -s "$AI" ]  && CMP_ARGS+=(--aiter-csv "$AI")
  [ -s "$CKQ" ] && CMP_ARGS+=(--ckquant-csv "$CKQ")
  echo "  -> fp8_compare"
  "$PY" "$HERE/fp8_compare.py" "${CMP_ARGS[@]}" || echo "     compare FAILED"
  [ -s "$MD" ] && { cat "$MD" >>"$SUMMARY"; echo >>"$SUMMARY"; }
done

echo; echo "== DONE =="; echo "Per-shape tables + aggregate: $SUMMARY"
ls -1 "$OUTDIR"
