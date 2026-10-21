#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# gfx950 build / run split of the full common set (build on --build-only, then --use-cache run),
# same conditions as ../../full_runs/run.sh: 25 cores, 4 xdist workers, 5400s per-test timeout,
# msgpack client. Gives per-test RUN-phase durations, which is what a client-launch-batching change
# can affect (the combined pass is dominated by kernel builds).
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
OUT=$TL/timing_runs/full_runs_gfx950/split
ART=/tmp/tl_split_gfx950/artifacts
CLIENT=/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client
cd $TL || exit 1
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
COMMON=(tensilelite/Tests/common -v --durations=0 --timeout=5400 -n 4 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=--cxx-compiler,/usr/bin/amdclang++")
phase() { # name, extra args...
  local name=$1; shift
  rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite
  echo "START $name $(date -u +%FT%TZ)" >> $OUT/times.txt
  local s=$(date +%s)
  taskset -c 0-24 pytest "${COMMON[@]}" "$@" --junit-xml=$OUT/$name.xml > $OUT/$name.log 2>&1
  local rc=$?
  echo "END $name rc=$rc seconds=$(( $(date +%s)-s )) $(date -u +%FT%TZ)" >> $OUT/times.txt
}
rm -rf $ART; mkdir -p $ART
phase build --build-only --gpu-targets gfx950 --artifact-dir $ART --prebuilt-client=$CLIENT
du -sh $ART >> $OUT/times.txt
phase run --use-cache --artifact-dir $ART --prebuilt-client=$CLIENT
echo ALLDONE >> $OUT/times.txt
