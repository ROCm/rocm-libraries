#!/bin/bash
# CI-like conditions: 1 GPU, 25 cores, 4 xdist workers, 5400s per-test timeout.
cd /home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite || exit 1
W=/tmp/tl_split
ART=$W/artifacts
CLIENT=/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client
export OMP_NUM_THREADS=25 OPENBLAS_NUM_THREADS=25 ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
COMMON=(tensilelite/Tests/common -v --durations=0 --timeout=5400 -n 4 -m "common and not skip-gfx94X"
  "--global-parameters=LibraryFormat='msgpack'" "--tensile-options=--cxx-compiler,/usr/bin/amdclang++")
phase() { # name, extra args...
  local name=$1; shift
  rm -rf /home/alvasile/.tensile /home/alvasile/.tensilelite
  echo "START $name $(date -u +%FT%TZ)" >> $W/times.txt
  local s=$(date +%s)
  taskset -c 0-24 pytest "${COMMON[@]}" "$@" --junit-xml=$W/$name.xml > $W/$name.log 2>&1
  local rc=$?
  echo "END $name rc=$rc seconds=$(( $(date +%s)-s )) $(date -u +%FT%TZ)" >> $W/times.txt
}
rm -rf $ART; mkdir -p $ART
# Emulate a CPU-only node: the 5 GPU-gated flags tests would skip there.
F=tensilelite/Tests/common/flags/gfx942
phase build --build-only --gpu-targets gfx942 --artifact-dir $ART --prebuilt-client=$CLIENT \
  --deselect $F/test_build_only.py::test_use_cache \
  --deselect $F/test_use_cache.py::test_compile_and_benchmark \
  --deselect $F/test_use_cache.py::test_use_cache \
  --deselect $F/test_use_cache.py::test_use_cache_creates_cache_dir \
  --deselect $F/test_use_cache.py::test_use_cache_different_params
du -sh $ART >> $W/times.txt; ls $ART | wc -l >> $W/times.txt
phase run --use-cache --artifact-dir $ART --prebuilt-client=$CLIENT
phase combined --prebuilt-client=$CLIENT
echo ALLDONE >> $W/times.txt
