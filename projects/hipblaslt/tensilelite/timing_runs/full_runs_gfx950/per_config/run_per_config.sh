#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run step of every gfx950 common config that passed in the split run phase, one at a time
# (serial, no xdist), with TimingInstrumentation on, and save per config:
#   <name>/<name>.log.gz            raw instrumented pytest/client log
#   <name>/<name>_analysis.txt      scripts/analyze_timing.py output
#   <name>/times.txt                pytest wall seconds + return code
# Build artifacts come from the split build phase (../split) or from
# run_per_config_build.sh with KEEP_ARTIFACTS=1. Caches (~/.tensile, ~/.tensilelite) are cleared
# before each config, as in run_cold_warm.sh / the split phases, unless CLEAR_CACHE=once
# (or KEEP_CACHE=1): then they are cleared once before the first config and kept between configs
# (a normal run, where e.g. the helper-kernel cache is shared by all configs). CLEAR_CACHE=never
# keeps the cache a preceding build pass left (a CI-like build-then-run on one node).
# When every config is DONE, scripts/aggregate_timing.py sums the logs of ALL configs (failed ones
# included) into $OUT/AGGREGATE_analysis.txt. Re-running the script after a finished pass skips the
# runs and only regenerates the aggregate.
# Usage: bash run_per_config.sh [configs.txt]   (one tensilelite/Tests/... yaml path per line)
# Env: TL, CLIENT, ART, TENSILE_CACHE, TENSILELITE_CACHE (see ../measure_env.sh),
#      CORES, GPU, ANALYZE_DIR, CLEAR_CACHE (see ../measure_env.sh),
#      OUT (default $TL/timing_runs/full_runs_gfx950/per_config), KEEP_CACHE.
source "$(dirname "${BASH_SOURCE[0]}")/../measure_env.sh" || exit 1
OUT=${OUT:-$TL/timing_runs/full_runs_gfx950/per_config}
LIST=${1:-$OUT/configs.txt}
check_rm_target TENSILE_CACHE TENSILELITE_CACHE
cd "$TL" || exit 1
mkdir -p "$OUT"
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
TO="--cxx-compiler,/usr/bin/amdclang++,--global-parameters,TimingInstrumentation=True"
COMMON=(-s -v -n 0 --timeout=5400 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=$TO" -p no:warnings -p no:cacheprovider)

n=0; total=$(wc -l < "$LIST")
if [ "$CLEAR_CACHE" = once ]; then
  clear_caches
  echo "CLEAR_CACHE=once: caches cleared once at $(date -u +%FT%TZ)" >> "$OUT/progress.txt"
elif [ "$CLEAR_CACHE" = never ]; then
  echo "CLEAR_CACHE=never: caches kept at $(date -u +%FT%TZ)" >> "$OUT/progress.txt"
fi
while read -r cfg; do
  [ -z "$cfg" ] && continue
  n=$((n+1))
  name=$(echo "${cfg#tensilelite/Tests/}" | sed -e 's/\.yaml$//' -e 's#/#__#g')
  D=$OUT/$name
  if [ -f "$D/times.txt" ] && grep -q DONE "$D/times.txt"; then continue; fi   # resumable
  mkdir -p "$D"
  RID="tensilelite/Tests/common/test_config_run.py::test_config_run[$cfg]"
  if [ "$CLEAR_CACHE" = per-config ]; then
    clear_caches
  fi
  s=$(date +%s)
  taskset -c "$CORES" pytest "$RID" --use-cache --artifact-dir "$ART" --prebuilt-client="$CLIENT" \
    "${COMMON[@]}" > "$D/$name.log" 2>&1
  rc=$?
  echo "run rc=$rc seconds=$(( $(date +%s)-s )) ($n/$total)" > "$D/times.txt"
  python "$ANALYZE_DIR/analyze_timing.py" "$D/$name.log" > "$D/${name}_analysis.txt" 2>&1
  gzip -f "$D/$name.log"
  echo DONE >> "$D/times.txt"
  echo "$(date -u +%T) $n/$total $cfg rc=$rc" >> "$OUT/progress.txt"
done < "$LIST"
python "$ANALYZE_DIR/aggregate_timing.py" "$OUT"/*/*.log.gz > "$OUT/AGGREGATE_analysis.txt" 2>&1
echo "aggregate rc=$? logs=$(ls "$OUT"/*/*.log.gz | wc -l)" >> "$OUT/progress.txt"
echo ALLDONE > "$OUT/done.txt"
