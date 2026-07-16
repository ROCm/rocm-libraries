#!/bin/bash
# rocprofv3 counter comparison: rocke dense vs flyDSL 0.2.4, GQA Hq128/Hkv8 S8192 causal.
# Each metric runs in its own single-pass rocprofv3 invocation (fast, no app replay).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/tmp/prof_cmp
rm -rf "$OUT"; mkdir -p "$OUT"
ATOM=/workspace/atom-venv/bin/python
FLY=/tmp/flydsl024-venv/bin/python

METRICS=(MfmaUtil MeanOccupancyPerCU OccupancyPercent FetchSize WriteSize LDSBankConflict MemUnitStalled SQ_WAVES)

dense_cmd=($ATOM "$HERE/_prof_dense.py" 8192 128 8 128 1 8)
fly_cmd=($FLY "$HERE/_flydsl_bench.py" dense 1 8192 128 8 128 1 1 8)

for M in "${METRICS[@]}"; do
  rocprofv3 --pmc "$M" --kernel-include-regex "rocke_attention_dense" \
    -d "$OUT/dense_$M" --output-format csv -- "${dense_cmd[@]}" \
    >/dev/null 2>&1
  PYTHONPATH=/workspace/flydsl-main rocprofv3 --pmc "$M" \
    --kernel-include-regex "flash_attn_dualwave_swp" \
    -d "$OUT/flydsl_$M" --output-format csv -- "${fly_cmd[@]}" \
    >/dev/null 2>&1
  echo "done $M"
done
echo "ALL_DONE"
