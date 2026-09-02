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
VENV="${SHARED}/rocke-venv"
DECODER_DIR="${SHARED}/tools/rocprof-trace-decoder/lib"
IMAGE="docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"

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
    baseline) TREE="${BASE}"; SHAPE=llama3_8b_dense_prefill_baseline_shape.json; EXTRA="" ;;
    candidate) TREE="${CAND}"; SHAPE=llama3_8b_dense_prefill_shape.json; EXTRA="--q-reload" ;;
    *) echo "--which must be baseline|candidate"; exit 2 ;;
esac

BENCH="${TREE}/library/benchmarks/gfx950/attention/prefill"
TOOLS="${TREE}/platform/dsl_docs/optimization/utilities/tools"
OUT="${RESULTS}/att_${WHICH}"
mkdir -p "${OUT}"

[ -f "${DECODER_DIR}/librocprof-trace-decoder.so" ] || {
    echo "missing decoder in ${DECODER_DIR}"; exit 1; }

RENDER_GID="$(getent group render | cut -d: -f3)"
VIDEO_GID="$(getent group video | cut -d: -f3)"

docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --user "$(id -u):$(id -g)" \
    --group-add "${VIDEO_GID}" \
    --group-add "${RENDER_GID}" \
    --ipc=host \
    --network=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /ossci-storage:/ossci-storage \
    -w "${TREE}/library" \
    -e HOME=/tmp \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    -e ROCKE_DENSE_VPAD=32 \
    -e ROCKE_LLVM_FLAVOR=llvm22 \
    -e ROCKE_DEBUG_LOC=1 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH="${TREE}/library:${TREE}/platform/python" \
    -e ROCPROF_TRACE_DECODER_LIB="${DECODER_DIR}" \
    "${IMAGE}" \
    bash -lc "
set -eu
PY='${VENV}/bin/python'
\${PY} '${TOOLS}/wavescope/capture_wavescope_trace.py' \
  --output-dir '${OUT}' \
  --kernel-regex 'rocke_attention_dense' \
  -- \${PY} builders/gfx950/attention/prefill/attention_dense_prefill.py \
     --exact-shape --sq 8192 --hq 32 --hkv 8 --dtype fp16 --d 128 \
     --persistent --np 256 --bn 64 --wpe 2 ${EXTRA} \
     --warmup 2 --iters 3 --no-check
" 2>&1 | tee "${RESULTS}/capture_${WHICH}.log"

echo "=== dispatch folders ==="
find "${OUT}" -type d -name 'ui_output_*_dispatch_*' | tee "${RESULTS}/dispatches_${WHICH}.txt"
echo "results=${RESULTS}"
