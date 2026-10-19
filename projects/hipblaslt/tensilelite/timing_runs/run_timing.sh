# Timing-instrumented single-config run: build phase then run phase, CI-like conditions.
# Usage: bash run_timing.sh [config-name-substring]
T=${1:-fp16_gradient_bias}
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
D=$TL/timing_runs
CL=/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client
A=/tmp/tl_timing_art
cd $TL || exit 1
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
TO="--cxx-compiler,/usr/bin/amdclang++,--global-parameters,TimingInstrumentation=True"
COMMON=(-s -v -n 0 --timeout=5400 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=$TO" -p no:warnings -k $T)
rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite $A
: > $D/times.txt
s=$(date +%s)
taskset -c 0-24 pytest tensilelite/Tests/common/test_config_build.py --build-only --gpu-targets gfx942 \
  --artifact-dir $A "${COMMON[@]}" > $D/build_$T.log 2>&1
echo "build rc=$? $(( $(date +%s)-s ))s" >> $D/times.txt
rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite
s=$(date +%s)
taskset -c 0-24 pytest tensilelite/Tests/common/test_config_run.py --use-cache --artifact-dir $A \
  --prebuilt-client=$CL "${COMMON[@]}" > $D/run_$T.log 2>&1
echo "run rc=$? $(( $(date +%s)-s ))s" >> $D/times.txt
python scripts/analyze_timing.py $D/build_$T.log > $D/build_${T}_analysis.txt 2>&1
python scripts/analyze_timing.py $D/run_$T.log > $D/run_${T}_analysis.txt 2>&1
echo DONE >> $D/times.txt
