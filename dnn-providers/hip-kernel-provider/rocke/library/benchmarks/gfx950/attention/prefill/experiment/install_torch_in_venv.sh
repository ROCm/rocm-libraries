#!/bin/bash
# Install ROCm torch into an existing rocke-venv (run on Conductor n01 via docker).
set -eu

U="${USER:-yraparti}"
SHARED="/ossci-storage/spur/${U}"
VENV="${SHARED}/rocke-venv"
IMAGE="docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"

[ -f "${VENV}/bin/activate" ] || { echo "missing ${VENV}"; exit 1; }

docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --ipc=host \
    --network=host \
    -v /ossci-storage:/ossci-storage \
    -w "${SHARED}" \
    "${IMAGE}" \
    bash -lc "
set -eu
source '${VENV}/bin/activate'
python -m pip install -U pip wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
"
