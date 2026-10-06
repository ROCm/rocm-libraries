#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# CPU wall-clock of tilewright's ranking call for one deployed model:
#
#   bench_selection_time.sh <config.yaml> <model.tilewright.bin> <library-dir> <out-dir>
#
# Builds bench_selection_time against the engine in this checkout, writes the
# inputs (the library's kernel pool and the shapes) and runs it. Everything
# goes to <out-dir>. Environment: SHAPES_FILE, ITERS (200), WARMUP (50),
# MIN_SCORED (1), DEVICE (0, for the KFD hardware values), PYTHON (python3).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${1:?usage: bench_selection_time.sh <config.yaml> <model> <library-dir> <out-dir>}"
MODEL="${2:?missing model file}"
LIBDIR="${3:?missing library directory}"
OUT="${4:?missing output directory}"

mkdir -p "$OUT"
cmake -S "$HERE" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release > "$OUT/configure.log"
cmake --build "$OUT/build" --target bench_selection_time > "$OUT/build.log"

shapes=()
if [[ -n "${SHAPES_FILE:-}" ]]; then
  shapes=(--shapes-file "$SHAPES_FILE")
fi
"${PYTHON:-python3}" "$HERE/bench_selection_time.py" \
  --config-yaml "$CFG" --library-dir "$LIBDIR" --device "${DEVICE:-0}" \
  --out "$OUT/inputs.txt" ${shapes[@]+"${shapes[@]}"}

"$OUT/build/bench_selection_time" \
  --model "$MODEL" --inputs "$OUT/inputs.txt" \
  --iters "${ITERS:-200}" --warmup "${WARMUP:-50}" --min-scored "${MIN_SCORED:-1}" \
  | tee "$OUT/results.txt"
