#!/bin/bash
#SBATCH --job-name=dense-prefill-exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mi355x:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

set -euo pipefail

U=$(id -un)
ROCKE="/ossci-storage/spur/${U}/src/rocke-dense-opt"
BASELINE="/ossci-storage/spur/${U}/src/rocke-dense-baseline"
BENCH="${ROCKE}/library/benchmarks/gfx950/attention/prefill"
TOOLS="${ROCKE}/platform/dsl_docs/optimization/utilities/tools"
RESULTS="/ossci-storage/spur/${U}/results/dense_prefill_exp_${SLURM_JOB_ID:-manual}"
BUILD="/tmp/rocke_dense_build_${SLURM_JOB_ID:-$$}"

mkdir -p "${RESULTS}" "${BUILD}"
export PYTHONPATH="${ROCKE}/platform/python:${ROCKE}/library"
export PYTHONDONTWRITEBYTECODE=1
export ROCKE_DENSE_VPAD=32
export HIP_VISIBLE_DEVICES=0

echo "node=$(hostname -s) user=${U} job=${SLURM_JOB_ID:-manual}"
echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo none)"
echo "gpus=${ROCR_VISIBLE_DEVICES:-unset}"
echo "candidate=${ROCKE}"
echo "baseline=${BASELINE}"

PY=""
for cand in $(command -v python3) /opt/conda/bin/python /opt/conda/envs/*/bin/python \
            /usr/bin/python3 /opt/venv/bin/python; do
    [ -x "${cand}" ] || continue
    if "${cand}" -c 'import torch' >/dev/null 2>&1; then
        PY="${cand}"
        break
    fi
done
if [ -z "${PY}" ]; then
    echo "no interpreter with torch found"
    exit 1
fi
echo "python=${PY}"

"${PY}" -c '
import torch
assert torch.cuda.is_available(), "no HIP device visible"
props = torch.cuda.get_device_properties(0)
print(f"torch={torch.__version__} hip={torch.version.hip}")
print(f"device={props.name} arch={props.gcnArchName} cu={props.multi_processor_count}")
assert "gfx950" in props.gcnArchName.lower(), props.gcnArchName
'

if [ ! -d "${BASELINE}/library" ]; then
    echo "staging baseline tree from candidate (develop attention_dense.py expected)"
    rm -rf "${BASELINE}"
    cp -a "${ROCKE}" "${BASELINE}"
fi

cd "${ROCKE}/library"

echo "=== knob calibration (candidate tree, unchanged knobs) ==="
for bn in 64 128; do
  for wpe in 1 2; do
    for pers in 0 1; do
      tag="bn${bn}_wpe${wpe}_pers${pers}"
      extra=()
      if [ "${pers}" = 1 ]; then extra+=(--persistent); fi
      "${PY}" builders/gfx950/attention/prefill/attention_dense_prefill.py \
        --exact-shape --sq 8192 --hq 32 --hkv 8 --dtype fp16 --d 128 \
        --bn "${bn}" --wpe "${wpe}" --warmup 10 --iters 30 --seed 0 \
        "${extra[@]}" \
        --json-out "${RESULTS}/calib_${tag}.json" || true
    done
  done
done

echo "=== baseline exact shape ==="
"${PY}" "${BENCH}/benchmark_dense_prefill_exact.py" \
  --shape-json "${BENCH}/llama3_8b_dense_prefill_baseline_shape.json" \
  --warmup 20 --iters 50 --seed 0 \
  --output-json "${RESULTS}/baseline.json"

echo "=== candidate exact shape ==="
"${PY}" "${BENCH}/benchmark_dense_prefill_exact.py" \
  --shape-json "${BENCH}/llama3_8b_dense_prefill_shape.json" \
  --warmup 20 --iters 50 --seed 0 \
  --output-json "${RESULTS}/candidate.json"

echo "=== ABBA paired comparison ==="
"${PY}" "${BENCH}/experiment_dense_prefill_abba.py" \
  --baseline-root "${BASELINE}" \
  --candidate-root "${ROCKE}" \
  --baseline-shape-json "${BENCH}/llama3_8b_dense_prefill_baseline_shape.json" \
  --candidate-shape-json "${BENCH}/llama3_8b_dense_prefill_shape.json" \
  --rounds 5 --warmup 20 --iters 50 --seed 0 \
  --output-json "${RESULTS}/abba.json"

echo "=== numeric tests (candidate) ==="
"${PY}" -m pytest -q -p no:cacheprovider \
  tests/test_attention_dense_gfx950_numeric.py \
  tests/dispatch/attention/test_gfx950_dense_wiring.py \
  2>&1 | tee "${RESULTS}/tests.log" || true

echo "=== WaveScope baseline capture ==="
ATT_OUT="${RESULTS}/att_baseline"
mkdir -p "${ATT_OUT}"
if command -v rocprofv3 >/dev/null 2>&1; then
  export ROCKE_DEBUG_LOC=1
  DECODER="${HOME}/.rocke/librocprof-trace-decoder.so"
  if [ -f "${DECODER}" ]; then export ROCPROF_TRACE_DECODER_LIB="${DECODER}"; fi
  "${PY}" "${TOOLS}/wavescope/capture_wavescope_trace.py" \
    --output-dir "${ATT_OUT}" \
    --kernel-regex 'rocke_attention_dense.*persist256' \
    --iteration-range '[1, [2-3]]' \
    -- "${PY}" builders/gfx950/attention/prefill/attention_dense_prefill.py \
      --exact-shape --sq 8192 --hq 32 --hkv 8 --dtype fp16 --d 128 \
      --persistent --warmup 5 --iters 10 --no-check || true
  DISPATCH=$(find "${ATT_OUT}" -type d -name 'ui_output_*_dispatch_*' | head -1 || true)
  if [ -n "${DISPATCH}" ]; then
    "${PY}" "${BENCH}/emit_wavescope_annotations.py" \
      "${DISPATCH}" --title "baseline Llama3-8B dense prefill Sq=8192 Hq=32"
  fi
fi

echo "=== WaveScope candidate capture ==="
ATT_OUT_C="${RESULTS}/att_candidate"
mkdir -p "${ATT_OUT_C}"
if command -v rocprofv3 >/dev/null 2>&1; then
  "${PY}" "${TOOLS}/wavescope/capture_wavescope_trace.py" \
    --output-dir "${ATT_OUT_C}" \
    --kernel-regex 'rocke_attention_dense.*qreload' \
    --iteration-range '[1, [2-3]]' \
    -- "${PY}" builders/gfx950/attention/prefill/attention_dense_prefill.py \
      --exact-shape --sq 8192 --hq 32 --hkv 8 --dtype fp16 --d 128 \
      --persistent --q-reload --warmup 5 --iters 10 --no-check || true
  DISPATCH=$(find "${ATT_OUT_C}" -type d -name 'ui_output_*_dispatch_*' | head -1 || true)
  if [ -n "${DISPATCH}" ]; then
    "${PY}" "${BENCH}/emit_wavescope_annotations.py" \
      "${DISPATCH}" --title "candidate q_reload dense prefill Sq=8192 Hq=32"
  fi
fi

echo "results=${RESULTS}"
ls -la "${RESULTS}"
