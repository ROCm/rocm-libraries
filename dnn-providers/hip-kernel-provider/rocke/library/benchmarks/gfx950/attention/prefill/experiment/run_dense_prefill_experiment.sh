#!/bin/bash
# Direct-SSH experiment runner (no Spur sbatch): docker + rocke-venv on Conductor n01.
#
#   stage 1  calibrate legal knobs on the unchanged baseline kernel
#   stage 2  single-shot baseline and candidate at the exact shape
#   stage 3  paired ABBA baseline vs candidate
#
# Usage: run_dense_prefill_experiment.sh [--stage calib|bench|abba|all] [--rounds N]
set -eu

U="${USER:-yraparti}"
SHARED="/ossci-storage/spur/${U}"
CAND="${SHARED}/src/rocke-dense-opt/rocke"
BASE="${SHARED}/src/rocke-dense-baseline"
VENV="${SHARED}/rocke-venv"
IMAGE="docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"

STAGE=all
ROUNDS=3
RESULTS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done
[ -n "${RESULTS}" ] || RESULTS="${SHARED}/results/dense_prefill_$(date +%Y%m%d_%H%M%S)"

[ -f "${VENV}/bin/activate" ] || { echo "missing ${VENV}; run setup_rocke_venv.sh"; exit 1; }
[ -d "${CAND}/library" ] || { echo "missing candidate tree ${CAND}"; exit 1; }
[ -d "${BASE}/library" ] || { echo "missing baseline tree ${BASE}"; exit 1; }
mkdir -p "${RESULTS}"

# Run a command in the ROCm container with the venv active. $1 = tree whose
# library/ + platform/python go on PYTHONPATH, rest = command.
in_container() {
    tree="$1"; shift
    docker run --rm \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add=video \
        --ipc=host \
        --network=host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v /ossci-storage:/ossci-storage \
        -w "${tree}/library" \
        -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
        -e ROCKE_DENSE_VPAD=32 \
        -e ROCKE_LLVM_FLAVOR=llvm22 \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -e PYTHONPATH="${tree}/library:${tree}/platform/python" \
        "${IMAGE}" \
        "${VENV}/bin/python" "$@"
}

BENCH_C="${CAND}/library/benchmarks/gfx950/attention/prefill"
SHAPE_BASE="${BENCH_C}/llama3_8b_dense_prefill_baseline_shape.json"
SHAPE_CAND="${BENCH_C}/llama3_8b_dense_prefill_shape.json"

echo "host=$(hostname -s) results=${RESULTS} stage=${STAGE}"
echo "candidate=${CAND}"
echo "baseline=${BASE}"

if [ "${STAGE}" = all ] || [ "${STAGE}" = calib ]; then
    echo "=== stage 1: knob calibration (unchanged baseline kernel) ==="
    in_container "${BASE}" \
        "${BASE}/library/benchmarks/gfx950/attention/prefill/calibrate_dense_prefill_knobs.py" \
        --shape-json "${SHAPE_BASE}" \
        --warmup 10 --iters 30 --repeats 3 --seed 0 \
        --output-json "${RESULTS}/calibration.json" \
        2>&1 | tee "${RESULTS}/calibration.log"
fi

if [ "${STAGE}" = all ] || [ "${STAGE}" = bench ]; then
    echo "=== stage 2a: baseline exact shape ==="
    in_container "${BASE}" \
        "${BASE}/library/benchmarks/gfx950/attention/prefill/benchmark_dense_prefill_exact.py" \
        --shape-json "${SHAPE_BASE}" \
        --warmup 20 --iters 50 --seed 0 \
        --output-json "${RESULTS}/baseline.json" \
        2>&1 | tee "${RESULTS}/baseline.log"

    echo "=== stage 2b: candidate exact shape ==="
    in_container "${CAND}" \
        "${BENCH_C}/benchmark_dense_prefill_exact.py" \
        --shape-json "${SHAPE_CAND}" \
        --warmup 20 --iters 50 --seed 0 \
        --output-json "${RESULTS}/candidate.json" \
        2>&1 | tee "${RESULTS}/candidate.log"
fi

if [ "${STAGE}" = all ] || [ "${STAGE}" = abba ]; then
    echo "=== stage 3: paired ABBA (${ROUNDS} rounds) ==="
    in_container "${CAND}" \
        "${BENCH_C}/experiment_dense_prefill_abba.py" \
        --baseline-root "${BASE}" \
        --candidate-root "${CAND}" \
        --baseline-shape-json "${SHAPE_BASE}" \
        --candidate-shape-json "${SHAPE_CAND}" \
        --rounds "${ROUNDS}" --warmup 20 --iters 50 --seed 0 \
        --output-json "${RESULTS}/abba.json" \
        2>&1 | tee "${RESULTS}/abba.log"
fi

echo "results=${RESULTS}"
ls -la "${RESULTS}"
