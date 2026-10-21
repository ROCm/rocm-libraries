#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run step of every gfx950 common config that passed in the split run phase, one at a time
# (serial, no xdist), with TimingInstrumentation on, and save per config:
#   <name>/<name>.log.gz            raw instrumented pytest/client log
#   <name>/<name>_analysis.txt      scripts/analyze_timing.py output
#   <name>/times.txt                pytest wall seconds + return code
# Build artifacts come from the split build phase (../split). Caches (~/.tensile, ~/.tensilelite)
# are cleared before each config, as in run_cold_warm.sh / the split phases, unless KEEP_CACHE=1:
# then they are cleared once before the first config and kept between configs (a normal run, where
# e.g. the helper-kernel cache is shared by all configs).
# Usage: bash run_per_config.sh [configs.txt]   (one tensilelite/Tests/... yaml path per line)
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
OUT=${OUT:-$TL/timing_runs/full_runs_gfx950/per_config}
LIST=${1:-$OUT/configs.txt}
ART=/tmp/tl_split_gfx950/artifacts
CLIENT=/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client
cd $TL || exit 1
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
TO="--cxx-compiler,/usr/bin/amdclang++,--global-parameters,TimingInstrumentation=True"
COMMON=(-s -v -n 0 --timeout=5400 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=$TO" -p no:warnings -p no:cacheprovider)

n=0; total=$(wc -l < $LIST)
if [ "${KEEP_CACHE:-0}" = "1" ]; then
  rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite
  echo "KEEP_CACHE=1: caches cleared once at $(date -u +%FT%TZ)" >> $OUT/progress.txt
fi
while read -r cfg; do
  [ -z "$cfg" ] && continue
  n=$((n+1))
  name=$(echo "${cfg#tensilelite/Tests/}" | sed -e 's/\.yaml$//' -e 's#/#__#g')
  D=$OUT/$name
  if [ -f $D/times.txt ] && grep -q DONE $D/times.txt; then continue; fi   # resumable
  mkdir -p $D
  RID="tensilelite/Tests/common/test_config_run.py::test_config_run[$cfg]"
  if [ "${KEEP_CACHE:-0}" != "1" ]; then
    rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite
  fi
  s=$(date +%s)
  taskset -c 0-24 pytest "$RID" --use-cache --artifact-dir $ART --prebuilt-client=$CLIENT \
    "${COMMON[@]}" > $D/$name.log 2>&1
  rc=$?
  echo "run rc=$rc seconds=$(( $(date +%s)-s )) ($n/$total)" > $D/times.txt
  python scripts/analyze_timing.py $D/$name.log > $D/${name}_analysis.txt 2>&1
  gzip -f $D/$name.log
  echo DONE >> $D/times.txt
  echo "$(date -u +%T) $n/$total $cfg rc=$rc" >> $OUT/progress.txt
done < $LIST
echo ALLDONE > $OUT/done.txt
