#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build step of every common config, one at a time (serial, no xdist), with
# TimingInstrumentation on, COLD: the caches (~/.tensile, ~/.tensilelite) are cleared before every
# config, so each build pays the full helper-kernel / assembler / library cost. With
# CLEAR_CACHE=once they are cleared before the first config only (a CI-like build that starts
# with an empty cache). Counterpart of run_per_config.sh (run step). Saves per config:
#   <name>/<name>_build.log.gz        raw instrumented pytest log
#   <name>/<name>_build_analysis.txt  scripts/analyze_timing.py output
#   <name>/times.txt                  pytest wall seconds + return code
# Artifacts go to a per-config scratch dir that is deleted after the config's log is saved (they
# are not used by anything else; the run phase uses $ART). With KEEP_ARTIFACTS=1 they go to $ART
# instead (emptied once before the first config) and are kept, for run_per_config.sh.
# When every config is DONE, scripts/aggregate_timing.py sums the logs of ALL configs (failed ones
# included) into $OUT/AGGREGATE_analysis.txt. Re-running the script after a finished pass skips the
# builds and only regenerates the aggregate.
# Usage: bash run_per_config_build.sh [configs.txt]   (one tensilelite/Tests/... yaml path per line)
# Env: TL, CLIENT, TENSILE_CACHE, TENSILELITE_CACHE (see ../measure_env.sh),
#      GPU_TARGET, CORES, GPU, ANALYZE_DIR, CLEAR_CACHE (see ../measure_env.sh), KEEP_ARTIFACTS, ART,
#      OUT (default .../per_config_build), SCRATCH (default /tmp/tl_build_scratch_gfx950)
source "$(dirname "${BASH_SOURCE[0]}")/../measure_env.sh" || exit 1
OUT=${OUT:-$TL/timing_runs/full_runs_gfx950/per_config_build}
LIST=${1:-$TL/timing_runs/full_runs_gfx950/per_config/configs.txt}
SCRATCH=${SCRATCH:-/tmp/tl_build_scratch_gfx950}
check_rm_target TENSILE_CACHE TENSILELITE_CACHE SCRATCH
if [ "${KEEP_ARTIFACTS:-0}" = "1" ]; then check_rm_target ART; fi
if [ "$CLEAR_CACHE" = never ]; then
  echo "CLEAR_CACHE=never is not supported by the build pass" >&2; exit 1
fi
cd "$TL" || exit 1
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
TO="--cxx-compiler,/usr/bin/amdclang++,--global-parameters,TimingInstrumentation=True"
COMMON=(-s -v -n 0 --timeout=5400 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=$TO" -p no:warnings -p no:cacheprovider)

mkdir -p "$OUT" "$SCRATCH"
n=0; total=$(wc -l < "$LIST")
if [ ! -f "$OUT/started.txt" ]; then   # first start of the pass, not a resume
  if [ "$CLEAR_CACHE" = once ]; then
    clear_caches
    echo "CLEAR_CACHE=once: caches cleared once at $(date -u +%FT%TZ)" >> "$OUT/progress.txt"
  fi
  if [ "${KEEP_ARTIFACTS:-0}" = "1" ]; then rm -rf -- "$ART"; fi
  date -u +%FT%TZ > "$OUT/started.txt"
fi
if [ "${KEEP_ARTIFACTS:-0}" = "1" ]; then mkdir -p "$ART"; fi
while read -r cfg; do
  [ -z "$cfg" ] && continue
  n=$((n+1))
  name=$(echo "${cfg#tensilelite/Tests/}" | sed -e 's/\.yaml$//' -e 's#/#__#g')
  D=$OUT/$name
  if [ -f "$D/times.txt" ] && grep -q DONE "$D/times.txt"; then continue; fi   # resumable
  mkdir -p "$D"
  A=$SCRATCH/$name
  if [ "${KEEP_ARTIFACTS:-0}" = "1" ]; then A=$ART; else rm -rf -- "$A"; fi
  BID="tensilelite/Tests/common/test_config_build.py::test_config_build[$cfg]"
  if [ "$CLEAR_CACHE" = per-config ]; then clear_caches; fi   # cold cache for every build
  s=$(date +%s)
  taskset -c "$CORES" pytest "$BID" --build-only --gpu-targets "$GPU_TARGET" --artifact-dir "$A" \
    --prebuilt-client="$CLIENT" "${COMMON[@]}" > "$D/${name}_build.log" 2>&1   # ductile builds run the client
  rc=$?
  echo "build_cold rc=$rc seconds=$(( $(date +%s)-s )) ($n/$total)" > "$D/times.txt"
  python "$ANALYZE_DIR/analyze_timing.py" "$D/${name}_build.log" > "$D/${name}_build_analysis.txt" 2>&1
  gzip -f "$D/${name}_build.log"
  if [ "${KEEP_ARTIFACTS:-0}" != "1" ]; then rm -rf -- "$A"; fi
  echo DONE >> "$D/times.txt"
  echo "$(date -u +%T) $n/$total $cfg rc=$rc" >> "$OUT/progress.txt"
done < "$LIST"
python "$ANALYZE_DIR/aggregate_timing.py" "$OUT"/*/*_build.log.gz > "$OUT/AGGREGATE_analysis.txt" 2>&1
echo "aggregate rc=$? logs=$(ls "$OUT"/*/*_build.log.gz | wc -l)" >> "$OUT/progress.txt"
echo ALLDONE > "$OUT/done.txt"
