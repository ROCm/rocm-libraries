#!/bin/bash
# Bootstrap /ossci-storage/spur/$USER/rocke-venv using the downloaded PyTorch ROCm image.
# Intended for direct SSH on Conductor n01 (docker + GPU available without Spur sbatch).
set -eu

U="${USER:-yraparti}"
SHARED="/ossci-storage/spur/${U}"
ROCKE_SRC="${SHARED}/src/rocke-dense-opt/rocke"
VENV="${SHARED}/rocke-venv"
IMAGE="docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"
LOG="${SHARED}/logs/setup_rocke_venv.log"

mkdir -p "${SHARED}/logs"

docker_run() {
    docker run --rm \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add=video \
        --ipc=host \
        --network=host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v /ossci-storage:/ossci-storage \
        -w "${SHARED}" \
        -e ROCKE_LLVM_FLAVOR=llvm22 \
        "${IMAGE}" \
        "$@"
}

{
    echo "=== setup_rocke_venv $(date -Is) host=$(hostname -s) ==="
    [ -d "${ROCKE_SRC}/library" ] || { echo "missing ${ROCKE_SRC}"; exit 1; }

    docker_run bash -lc "
set -eu
PY=
for cand in /opt/venv/bin/python /opt/venv/bin/python3 /opt/conda/bin/python /opt/conda/envs/*/bin/python python3 python; do
  [ -x \"\${cand}\" ] || continue
  if \"\${cand}\" -c 'import torch' >/dev/null 2>&1; then PY=\"\${cand}\"; break; fi
done
[ -n \"\${PY}\" ] || { echo 'no torch python in image'; exit 1; }
echo using=\${PY}
\"\${PY}\" -m venv '${VENV}'
# shellcheck disable=SC1091
source '${VENV}/bin/activate'
python -m pip install -U pip wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
python -m pip install -r '${ROCKE_SRC}/platform/requirements.txt'
python -m pip install -e '${ROCKE_SRC}/platform'
export PYTHONPATH='${ROCKE_SRC}/library:${ROCKE_SRC}/platform/python'
python -c 'import torch, rocke, numpy; print(\"torch\", torch.__version__, \"hip\", torch.version.hip, \"cuda\", torch.cuda.is_available())'
python -c 'import builders.gfx950.attention.prefill.attention_dense_prefill as adp; print(\"adp_ok\")'
"

    echo "venv=${VENV}"
    echo "run benchmarks with: bash ${SHARED}/src/rocke-dense-opt/scripts/run_dense_prefill_experiment.sh"
} 2>&1 | tee -a "${LOG}"
