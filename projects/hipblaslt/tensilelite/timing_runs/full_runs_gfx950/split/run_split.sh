#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# gfx950 build / run split of the full common set (build on --build-only, then --use-cache run),
# same conditions as ../../full_runs/run.sh: 25 cores, 4 xdist workers, 5400s per-test timeout,
# msgpack client. Gives per-test RUN-phase durations, which is what a client-launch-batching change
# can affect (the combined pass is dominated by kernel builds).
# Env: TL, CLIENT, ART, TENSILE_CACHE, TENSILELITE_CACHE (see ../measure_env.sh),
#      OUT (default $TL/timing_runs/full_runs_gfx950/split). ART is deleted and rebuilt, so it must
#      be set: the default ART is the measured baseline, which the script refuses to delete.
source "$(dirname "${BASH_SOURCE[0]}")/../measure_env.sh" || exit 1
OUT=${OUT:-$TL/timing_runs/full_runs_gfx950/split}
check_rm_target TENSILE_CACHE TENSILELITE_CACHE ART
cd "$TL" || exit 1
mkdir -p "$OUT"
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
COMMON=(tensilelite/Tests/common -v --durations=0 --timeout=5400 -n 4 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=--cxx-compiler,/usr/bin/amdclang++")
phase() { # name, extra args...
  local name=$1; shift
  clear_caches
  echo "START $name $(date -u +%FT%TZ)" >> "$OUT/times.txt"
  local s=$(date +%s)
  taskset -c 0-24 pytest "${COMMON[@]}" "$@" --junit-xml="$OUT/$name.xml" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc seconds=$(( $(date +%s)-s )) $(date -u +%FT%TZ)" >> "$OUT/times.txt"
}
rm -rf -- "$ART"; mkdir -p "$ART"
phase build --build-only --gpu-targets gfx950 --artifact-dir "$ART" --prebuilt-client="$CLIENT"
du -sh "$ART" >> "$OUT/times.txt"
phase run --use-cache --artifact-dir "$ART" --prebuilt-client="$CLIENT"
echo ALLDONE >> "$OUT/times.txt"
