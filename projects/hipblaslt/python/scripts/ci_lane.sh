#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# One Component CI lane for the hipBLASLt Python bindings: install a pinned TheRock nightly
# into a fresh venv, build the bindings against it, and run their tests from outside the
# source tree. It runs on Linux and in Git Bash on Windows, and works the same locally:
#
#   HIPBLASLT_PY_ROCM_VERSION=10.3.0a20261116 HIPBLASLT_PY_DEVICE=gfx942 \
#     bash projects/hipblaslt/python/scripts/ci_lane.sh
#
# Environment:
#   HIPBLASLT_PY_ROCM_VERSION  the nightly to install (required), e.g. 10.3.0a20261116
#   HIPBLASLT_PY_ROCM_INDEX    the wheel index (default: TheRock's nightly index)
#   HIPBLASLT_PY_DEVICE        a device wheel to add, e.g. gfx942; empty for a CPU-only lane
#   HIPBLASLT_PY_REQUIRE_GPU   1 on GPU lanes, so "no device" skips become failures
#   PYTHON                     the interpreter to build the venv from (default: python3)

set -euo pipefail

: "${HIPBLASLT_PY_ROCM_VERSION:?set HIPBLASLT_PY_ROCM_VERSION to the pinned nightly}"
index="${HIPBLASLT_PY_ROCM_INDEX:-https://nightly.repo.amd.com/rocm/whl-next/}"
device="${HIPBLASLT_PY_DEVICE:-}"
python="${PYTHON:-python3}"

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d)"
# Leave the directory before removing it; Windows can't delete the current directory.
trap 'cd / && rm -rf "${work_dir}"' EXIT
venv="${work_dir}/venv"
"${python}" -m venv "${venv}"
if [[ -x "${venv}/bin/python" ]]; then
    vpy="${venv}/bin/python"
    rocm_sdk="${venv}/bin/rocm-sdk"
else
    vpy="${venv}/Scripts/python.exe"
    rocm_sdk="${venv}/Scripts/rocm-sdk.exe"
fi

extras="libraries,devel"
if [[ -n "${device}" ]]; then
    extras="${extras},device-${device}"
fi
"${vpy}" -m pip install --upgrade pip
"${vpy}" -m pip install --index-url "${index}" "rocm[${extras}]==${HIPBLASLT_PY_ROCM_VERSION}"
"${vpy}" -m pip install pytest

ROCM_PATH="$("${rocm_sdk}" path --root)"
export ROCM_PATH
echo "ROCm ${HIPBLASLT_PY_ROCM_VERSION} at ${ROCM_PATH}"

# hipBLASLt's headers need clang on Windows; MSVC can't compile them.
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
    export CXX="${ROCM_PATH}/lib/llvm/bin/amdclang++.exe"
    export CC="${ROCM_PATH}/lib/llvm/bin/amdclang.exe"
    export CMAKE_GENERATOR=Ninja
fi

"${vpy}" -m pip install -v "${package_dir}"

# Run from outside the source tree, so the tests import the installed package.
cd "${work_dir}"
"${vpy}" -m pytest "${package_dir}/tests" -ra -v
