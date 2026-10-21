#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# gfx950 counterpart of ../full_runs/run.sh (gfx942). Same CI-like conditions:
# 1 GPU, 25 cores, 4 xdist workers, 5400s per-test timeout, msgpack client.
# Single combined pass (what CI runs): build + run in one pytest invocation.
# Per-test timings come from pytest's --durations=0.
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
OUT=$TL/timing_runs/full_runs_gfx950
CLIENT=/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client
cd $TL || exit 1
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
COMMON=(tensilelite/Tests/common -v --durations=0 --timeout=5400 -n 4 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=--cxx-compiler,/usr/bin/amdclang++")

rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite
{
  echo "START combined $(date -u +%FT%TZ)"
  rocminfo | grep -m1 -o 'gfx950'
  git rev-parse HEAD
} >> $OUT/times.txt
s=$(date +%s)
taskset -c 0-24 pytest "${COMMON[@]}" --prebuilt-client=$CLIENT --junit-xml=$OUT/combined.xml > $OUT/combined.log 2>&1
rc=$?
echo "END combined rc=$rc seconds=$(( $(date +%s)-s )) $(date -u +%FT%TZ)" >> $OUT/times.txt
# Extract the per-test duration table
sed -n '/slowest durations/,/^=* .* in [0-9.]*s/p' $OUT/combined.log > $OUT/durations.txt
echo ALLDONE >> $OUT/times.txt
