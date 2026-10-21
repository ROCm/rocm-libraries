#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Does the LibraryClient phase hit the helper-kernel cache when ~/.tensile is NOT cleared between runs?
# Clears ~/.tensile once, then runs the same config twice (cache cold, then warm), saving the analysis
# of each. Usage: bash warm_cache_check.sh gemm/gfx950/subtile_mxfp8_tail_loop_smoke
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
CFG=$1
NAME=$(basename $CFG)
OUT=$TL/timing_runs/full_runs_gfx950/per_config_rerun/warm_cache_check/$NAME
ART=/tmp/tl_split_gfx950/artifacts
CLIENT=/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client
mkdir -p $OUT
cd $TL || exit 1
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
TO="--cxx-compiler,/usr/bin/amdclang++,--global-parameters,TimingInstrumentation=True"
RID="tensilelite/Tests/common/test_config_run.py::test_config_run[tensilelite/Tests/common/$CFG.yaml]"
rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite
for pass in cold warm; do
  taskset -c 0-24 pytest "$RID" --use-cache --artifact-dir $ART --prebuilt-client=$CLIENT \
    -s -v -n 0 --timeout=5400 -m "common and not skip-gfx94X" \
    "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=$TO" -p no:warnings -p no:cacheprovider \
    > $OUT/$pass.log 2>&1
  echo "$pass rc=$?" >> $OUT/times.txt
  python scripts/analyze_timing.py $OUT/$pass.log > $OUT/${pass}_analysis.txt 2>&1
  ls /home/alvasile/.tensile/helper_cache >> $OUT/times.txt 2>&1
done
