# For each config: cold build (caches deleted), warm build (caches kept), run (caches deleted
# first, as on a fresh GPU node). Raw logs + analyze_timing.py output per stage.
# Usage: bash run_cold_warm.sh gemm/dtv gemm/cgemm
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
CL=/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client
cd $TL || exit 1
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
TO="--cxx-compiler,/usr/bin/amdclang++,--global-parameters,TimingInstrumentation=True"
COMMON=(-s -v -n 0 --durations=0 --timeout=5400 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=$TO" -p no:warnings)
clear_caches() { rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite; }

for cfg in "$@"; do
  name=$(basename $cfg)
  D=$TL/timing_runs/$name
  A=/tmp/tl_timing_art_$name
  mkdir -p $D
  rm -rf $A
  BID="tensilelite/Tests/common/test_config_build.py::test_config_build[tensilelite/Tests/common/$cfg.yaml]"
  RID="tensilelite/Tests/common/test_config_run.py::test_config_run[tensilelite/Tests/common/$cfg.yaml]"
  : > $D/times.txt

  clear_caches
  s=$(date +%s)
  taskset -c 0-24 pytest "$BID" --build-only --gpu-targets gfx942 --artifact-dir $A/cold \
    "${COMMON[@]}" > $D/${name}_1_build_cold.log 2>&1
  echo "build_cold rc=$? $(( $(date +%s)-s ))s" >> $D/times.txt

  s=$(date +%s)   # caches kept from the cold build
  taskset -c 0-24 pytest "$BID" --build-only --gpu-targets gfx942 --artifact-dir $A/warm \
    "${COMMON[@]}" > $D/${name}_2_build_warm.log 2>&1
  echo "build_warm rc=$? $(( $(date +%s)-s ))s" >> $D/times.txt

  clear_caches
  s=$(date +%s)
  taskset -c 0-24 pytest "$RID" --use-cache --artifact-dir $A/cold --prebuilt-client=$CL \
    "${COMMON[@]}" > $D/${name}_3_run.log 2>&1
  echo "run rc=$? $(( $(date +%s)-s ))s" >> $D/times.txt

  for st in 1_build_cold 2_build_warm 3_run; do
    python scripts/analyze_timing.py $D/${name}_$st.log > $D/${name}_${st}_analysis.txt 2>&1
  done
  echo DONE >> $D/times.txt
done
echo ALLDONE > $TL/timing_runs/cold_warm_done.txt
