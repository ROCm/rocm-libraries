#!/bin/bash
# RT ATT job (q38_job.sh --mode venv; stages rocke-rt-<v>, rt-h-<v>): WaveScope (ATT) captures of saved code objects
# with the ROCm 7.2 user-space subset (rt72, cluster rule 88) and rocprof-trace-decoder (rules 99, 107), then the
# per-class stall / latency attribution (GO/harness/att_attr.py; the direct objects carry no line info).
#   RT_ATT (space-separated), each <results-path>/<file>.hsaco:<epi>:<rows>:<m>  (paths below q38/results/)
#   RT_ROCKE (the rocke stage name), RT_H (the harness stage name)
set -uo pipefail
A=$HOME/rocke-sdpa-claude
GO=$A/GO
H=$Q38_STAGES/${RT_H:?}
RK=$Q38_STAGES/${RT_ROCKE:?}
O=$Q38_OUT
GFX=gfx950
. $GO/harness/jobs/rt72.sh
export ROCPROF_TRACE_DECODER_LIB=$GO/tools/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib
export ROCKE_HIP_LIB=$(ls $ROCM_PATH/lib/libamdhip64.so.7 2> /dev/null || ls $ROCM_PATH/lib/libamdhip64.so | head -1)
export PYTHONPATH=$RK/platform/python:$RK/library:$H/rust-track:$H/rust-pilot
export ROCKE_BACKEND=python PYTHONDONTWRITEBYTECODE=1
WS=$RK/platform/dsl_docs/optimization/utilities/tools/wavescope
echo "== node $(hostname) rt72 $ROCM_PATH python $(command -v python)"
rc=0
for item in ${RT_ATT:?}; do
  IFS=: read -r path epi rows m <<< "$item"
  id=$(basename "${path%.hsaco}")_m$m
  mkdir -p "$O/$id"
  echo "=== $id $(date -u +%T)"
  timeout -s KILL 600 python "$WS/capture_wavescope_trace.py" --output-dir "$Q38_JOB/att_$id" --kernel-regex mxgemv --no-source \
    --target-cu 1 --buffer-size 0x10000000 -- python "$H/rust-track/rt_att_head.py" --hsaco "$A/q38/results/$path" \
    --epi "$epi" --rows "$rows" --m "$m" --launches 6 > "$O/$id/capture.log" 2>&1
  crc=$?
  echo "   capture rc=$crc"; grep -aE "ATTHEAD|decoded|error|Error" "$O/$id/capture.log" | head -6
  [ $crc = 0 ] || rc=1
  python3 $GO/harness/att_attr.py "$Q38_JOB/att_$id" --top 40 --out "$O/$id/attr.json" > "$O/$id/attr.txt" 2>&1
  head -45 "$O/$id/attr.txt"
  # keep the decoded per-instruction table (code.json) of each dispatch for later diffs (node /tmp is cleared)
  find "$Q38_JOB/att_$id" -name code.json | head -3 | while read -r cj; do
    cp "$cj" "$O/$id/$(basename "$(dirname "$cj")")_code.json"
  done
done
echo "== RTATT rc=$rc $(date -u +%T)"
exit $rc
