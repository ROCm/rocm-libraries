#!/bin/bash
# Direct-SSH experiment runner (no Spur submission). It uses Docker where
# available, or rootless runc over Spur's imported image on n07.
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
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
RUN_CONTAINER="${HERE}/run_in_rocm_container.sh"

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

[ -x "${RUN_CONTAINER}" ] || { echo "missing container helper ${RUN_CONTAINER}"; exit 1; }
[ -d "${CAND}/library" ] || { echo "missing candidate tree ${CAND}"; exit 1; }
[ -d "${BASE}/library" ] || { echo "missing baseline tree ${BASE}"; exit 1; }
mkdir -p "${RESULTS}"

# Run a command in the exact ROCm 7.2.4 image. $1 = tree whose
# library/ + platform/python go on PYTHONPATH, rest = command.
in_container() {
    tree="$1"; shift
    ROCKE_CONTAINER_CWD="${tree}/library" \
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    ROCKE_DENSE_VPAD="${ROCKE_DENSE_VPAD:-32}" \
    ROCKE_DENSE_NBUF="${ROCKE_DENSE_NBUF:-2}" \
    ROCKE_LLVM_FLAVOR=llvm22 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${tree}/library:${tree}/platform/python" \
        "${RUN_CONTAINER}" /opt/venv/bin/python "$@"
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
