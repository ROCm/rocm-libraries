#!/usr/bin/env bash
# Build, validate (10 sizes) and time (5 runs) each TENSILE_YAML given on the command line, on GPU1 (Device: 1 in the yaml, no visibility mask).
# usage: bench_variants.sh configs/a.yaml [configs/b.yaml ...]   (run under gpu-guard)
set -uo pipefail
BASE=$(cd "$(dirname "$0")" && pwd)
cd "$BASE"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_ID
export LD_LIBRARY_PATH=$BASE/../../build_tmp/tensilelite:/opt/rocm/lib/llvm/lib/x86_64-unknown-linux-gnu
CLIENT=$BASE/../../build_tmp/tensilelite/client/tensilelite-client
for Y in "$@"; do
  name=$(basename "$Y" .yaml)
  TENSILE_YAML=$Y ./run.sh tensile > /tmp/bv_$name.build.log 2>&1
  dir=$(rg -o 'done +(\S+)' -r '$1' /tmp/bv_$name.build.log | tail -1)
  if [[ -z "$dir" ]] || ! rg -q 'clientExit=0' /tmp/bv_$name.build.log; then
    echo "== $name: BUILD/RUN FAILED (see /tmp/bv_$name.build.log)"; continue
  fi
  ini=$(ls "$dir"/tensile/1_BenchmarkProblems/*/00_Final/caches/*/source/ClientParameters.ini)
  echo "== $name  ($dir)"
  val=$(python3 validate.py "$dir" --device 1 2>&1)
  echo "   validate: $(echo "$val" | rg -c PASSED)/10 PASSED $(echo "$val" | rg -v PASSED | tr '\n' ' ')"
  for i in 1 2 3 4 5; do
    "$CLIENT" --config-file "$ini" 2>&1 | rg '^0,0/0' | \
      python3 -c 'import sys,csv; r=next(csv.reader(sys.stdin)); print(f"   time {float(r[10]):8.2f} us  {float(r[11])/1e3:6.0f} TF/s")'
  done
done
