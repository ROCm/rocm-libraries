#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Repeat of the run stage from run_profile.sh, to measure run-to-run variation.
# Reuses the build artifacts from run_profile.sh; caches are cleared before the run, as before.
# Usage: bash run_rerun.sh <rerun-tag> gemm/gfx950/subtile_mxfp4 gemm/gfx950/dtl gemm/bufferLoadStore0
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
CL=/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client
cd $TL || exit 1
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
TO="--cxx-compiler,/usr/bin/amdclang++,--global-parameters,TimingInstrumentation=True"
COMMON=(-s -v -n 0 --durations=0 --timeout=5400 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=$TO" -p no:warnings)
TAG=$1; shift

for cfg in "$@"; do
  name=$(basename $cfg)
  D=$TL/timing_runs/full_runs_gfx950/profiling/$name
  A=/tmp/tl_timing_art_gfx950_$name
  RID="tensilelite/Tests/common/test_config_run.py::test_config_run[tensilelite/Tests/common/$cfg.yaml]"

  rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite
  s=$(date +%s)
  taskset -c 0-24 pytest "$RID" --use-cache --artifact-dir $A --prebuilt-client=$CL \
    "${COMMON[@]}" > $D/${name}_3_run_$TAG.log 2>&1
  echo "run_$TAG rc=$? $(( $(date +%s)-s ))s" >> $D/times.txt
  python scripts/analyze_timing.py $D/${name}_3_run_$TAG.log > $D/${name}_3_run_${TAG}_analysis.txt 2>&1
done
echo ALLDONE > $TL/timing_runs/full_runs_gfx950/profiling/done_$TAG.txt
