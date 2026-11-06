#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Runs the gfx950 MX common configs with TENSILELITE_VERIFY_REFERENCE_REUSE=1, i.e. every time the
# client skips a CPU-reference recompute because DataInitialization's input epoch did not move,
# it recomputes anyway and fails the run if the result is not bitwise identical.
# CI-like conditions as in ../full_runs/run.sh (combined build+run pass).
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
OUT=$TL/timing_runs/full_runs_gfx950/verify_mx
CLIENT=/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client
cd $TL || exit 1
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export TENSILELITE_VERIFY_REFERENCE_REUSE=1
rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite
s=$(date +%s)
taskset -c 0-24 pytest tensilelite/Tests/common/test_config.py -v --durations=0 --timeout=5400 -n 4 \
  -m "common and not skip-gfx94X" -k "mx and gfx950" \
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=--cxx-compiler,/usr/bin/amdclang++" \
  --prebuilt-client=$CLIENT --junit-xml=$OUT/verify_mx.xml > $OUT/verify_mx.log 2>&1
echo "rc=$? seconds=$(( $(date +%s)-s ))" > $OUT/result.txt
