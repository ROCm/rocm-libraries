#!/bin/bash
# Capture an ATT trace of the dense prefill kernel and emit WaveScope sidecars.
#
# The ROCm 7.2 image predates the bundled ATT decoder, so the decoder lives in
# the user experiment area and is pointed at via ROCPROF_TRACE_DECODER_LIB.
#
# Usage: capture_dense_prefill_wavescope.sh [--which baseline|candidate] [--results DIR]
set -eu

U="${USER:-yraparti}"
SHARED="/ossci-storage/spur/${U}"
CAND="${SHARED}/src/rocke-dense-opt/rocke"
BASE="${SHARED}/src/rocke-dense-baseline"
DECODER_DIR="${SHARED}/tools/rocprof-trace-decoder/lib"
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
RUN_CONTAINER="${HERE}/run_in_rocm_container.sh"

WHICH=baseline
RESULTS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --which) WHICH="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done
[ -n "${RESULTS}" ] || RESULTS="${SHARED}/results/wavescope_$(date +%Y%m%d_%H%M%S)"

case "${WHICH}" in
    baseline) TREE="${BASE}"; SHAPE=llama3_8b_dense_prefill_baseline_shape.json ;;
    candidate) TREE="${CAND}"; SHAPE=llama3_8b_dense_prefill_shape.json ;;
    *) echo "--which must be baseline|candidate"; exit 2 ;;
esac

BENCH="${TREE}/library/benchmarks/gfx950/attention/prefill"
TOOLS="${TREE}/platform/dsl_docs/optimization/utilities/tools"
OUT="${RESULTS}/att_${WHICH}"
mkdir -p "${OUT}"

[ -f "${DECODER_DIR}/librocprof-trace-decoder.so" ] || {
    echo "missing decoder in ${DECODER_DIR}"; exit 1; }
[ -x "${RUN_CONTAINER}" ] || {
    echo "missing container helper ${RUN_CONTAINER}"; exit 1; }

ROCKE_CONTAINER_CWD="${TREE}/library" \
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
ROCKE_DENSE_VPAD=32 \
ROCKE_DENSE_NBUF=2 \
ROCKE_LLVM_FLAVOR=llvm22 \
ROCKE_DEBUG_LOC=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${TREE}/library:${TREE}/platform/python" \
ROCPROF_TRACE_DECODER_LIB="${DECODER_DIR}" \
"${RUN_CONTAINER}" bash -lc "
set -eu
# The image's own torch is linked against this /opt/rocm, so rocprofv3 loads one
# rocprofiler. A pip torch wheel bundles a second copy and aborts in
# rocprofiler_configure, so the venv is deliberately not used for capture
# (rocke itself is pure Python and comes in over PYTHONPATH).
PY=/opt/venv/bin/python
\${PY} '${TOOLS}/wavescope/capture_wavescope_trace.py' \
  --output-dir '${OUT}' \
  --kernel-regex 'rocke_attention_dense' \
  -- \${PY} '${BENCH}/benchmark_dense_prefill_exact.py' \
     --shape-json '${BENCH}/${SHAPE}' \
     --warmup 2 --iters 3 --no-check \
     --output-json '${RESULTS}/capture_benchmark_${WHICH}.json'
" 2>&1 | tee "${RESULTS}/capture_${WHICH}.log"

echo "=== dispatch folders ==="
find "${OUT}" -type d -name 'ui_output_*_dispatch_*' | tee "${RESULTS}/dispatches_${WHICH}.txt"
echo "results=${RESULTS}"
