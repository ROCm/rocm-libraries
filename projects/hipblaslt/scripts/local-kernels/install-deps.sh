#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Install the host packages and Python environment the build uses, including
# what the hipBLASLt clients need: a Fortran compiler, LAPACK, GoogleTest with
# GoogleMock, and NUMA.
# Locally: bash projects/hipblaslt/scripts/local-kernels/install-deps.sh
# The other scripts use the virtualenv at ${VENV_DIR:-$PWD/.venv}.

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    cmake ninja-build g++ gfortran python3-dev python3-venv \
    libmsgpack-cxx-dev libomp-dev zlib1g-dev \
    libblas-dev liblapack-dev libgtest-dev libgmock-dev libnuma-dev
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT}/TheRock/requirements.txt" \
    -r "${ROOT}/projects/hipblaslt/tensilelite/requirements.txt"
if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "${VENV_DIR}/bin" >> "${GITHUB_PATH}"
fi
echo "Python environment: ${VENV_DIR}"
