# Build-only of the CMS (UseCustomMainLoopSchedule) configs, targeting gfx950, cold caches.
TL=/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite
D=$TL/timing_runs/cms
cd $TL || exit 1
mkdir -p $D
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
TO="--cxx-compiler,/usr/bin/amdclang++,--global-parameters,TimingInstrumentation=True"
COMMON=(-s -v -n 0 --durations=0 --timeout=14400 "--global-parameters=LibraryFormat='msgpack'"
  "--tensile-options=$TO" -p no:warnings)
: > $D/times.txt
for cfg in gemm/gfx950/custom_mainloop_scheduling gemm/gfx950/custom_mainloop_scheduling_tf32 ductile/gfx950/f8bf16_tn_cms; do
  name=$(basename $cfg)
  rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite /tmp/tl_cms_art_$name
  BID="tensilelite/Tests/common/test_config_build.py::test_config_build[tensilelite/Tests/common/$cfg.yaml]"
  s=$(date +%s)
  /usr/bin/time -v -o $D/${name}_time.txt taskset -c 0-24 pytest "$BID" --build-only --gpu-targets gfx950 \
    --artifact-dir /tmp/tl_cms_art_$name "${COMMON[@]}" > $D/${name}_build.log 2>&1
  echo "$name rc=$? $(( $(date +%s)-s ))s" >> $D/times.txt
  python scripts/analyze_timing.py $D/${name}_build.log > $D/${name}_build_analysis.txt 2>&1
done
echo DONE >> $D/times.txt
