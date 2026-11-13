#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# gfx950 build / run split of the full common set (build on --build-only, then --use-cache run),
# same conditions as ../../full_runs/run.sh: 25 cores, 4 xdist workers, 5400s per-test timeout,
# msgpack client. Gives per-test RUN-phase durations, which is what a client-launch-batching change
# can affect (the combined pass is dominated by kernel builds).
# Env: TL, CLIENT, ART, TENSILE_CACHE, TENSILELITE_CACHE (see ../measure_env.sh),
#      GPU_TARGET, CORES, GPU (see ../measure_env.sh),
#      OUT (default $TL/timing_runs/full_runs_gfx950/split). ART is deleted and rebuilt, so it must
#      be set: the default ART is the measured baseline, which the script refuses to delete.
#      DIST          pytest-xdist scheduler (default worksteal, as the hw-common CI category; the
#                    committed split numbers were measured with load, xdist's default)
#      CLEAR_RUN_CACHE  1 also clears the caches before the run phase (default 0: the run phase
#                    starts from the cache the build phase left; the committed split numbers were
#                    measured with 1). The build phase always starts from empty caches.
source "$(dirname "${BASH_SOURCE[0]}")/../measure_env.sh" || exit 1
OUT=${OUT:-$TL/timing_runs/full_runs_gfx950/split}
DIST=${DIST:-worksteal}
CLEAR_RUN_CACHE=${CLEAR_RUN_CACHE:-0}
case $CLEAR_RUN_CACHE in
  0|1) ;;
  *) echo "CLEAR_RUN_CACHE='$CLEAR_RUN_CACHE' must be 0 or 1" >&2; exit 1 ;;
esac
check_rm_target TENSILE_CACHE TENSILELITE_CACHE ART
cd "$TL" || exit 1
mkdir -p "$OUT"
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
COMMON=(tensilelite/Tests/common -v --durations=0 --timeout=5400 -n 4 --dist="$DIST" -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=--cxx-compiler,/usr/bin/amdclang++")
phase() { # name, clear caches first (0|1), extra args...
  local name=$1 clear=$2; shift 2
  if [ "$clear" = 1 ]; then clear_caches; fi
  echo "START $name dist=$DIST caches_cleared=$clear $(date -u +%FT%TZ)" >> "$OUT/times.txt"
  local s=$(date +%s)
  taskset -c "$CORES" pytest "${COMMON[@]}" "$@" --junit-xml="$OUT/$name.xml" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc seconds=$(( $(date +%s)-s )) $(date -u +%FT%TZ)" >> "$OUT/times.txt"
}
rm -rf -- "$ART"; mkdir -p "$ART"
phase build 1 --build-only --gpu-targets "$GPU_TARGET" --artifact-dir "$ART" --prebuilt-client="$CLIENT"
du -sh "$ART" >> "$OUT/times.txt"
phase run "$CLEAR_RUN_CACHE" --use-cache --artifact-dir "$ART" --prebuilt-client="$CLIENT"
echo ALLDONE >> "$OUT/times.txt"
