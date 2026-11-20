#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The steps of one Component CI lane for the hipBLASLt Python bindings: install a pinned TheRock
# nightly into a fresh venv, build the bindings against it, and run their tests from outside the
# source tree. The workflow runs each step on its own, so a failed download, build or test shows
# up as that step. It runs on Linux and in Git Bash on Windows. With no step named, it runs them
# all in a temporary directory, which is how to reproduce a lane locally:
#
#   HIPBLASLT_PY_ROCM_VERSION=10.3.0a20261116 HIPBLASLT_PY_DEVICE=gfx942 \
#     bash projects/hipblaslt/python/scripts/ci_lane.sh
#
# Usage: ci_lane.sh [venv|rocm|build|test|clean]...
#   venv   create a fresh venv with pytest
#   rocm   install the pinned nightly (and the device wheel) into it
#   build  build and install the bindings against that ROCm
#   test   run the bindings' tests
#   clean  remove the venv
#
# Environment:
#   HIPBLASLT_PY_ROCM_VERSION  the nightly to install (required by rocm), e.g. 10.3.0a20261116
#   HIPBLASLT_PY_ROCM_INDEX    the wheel index (default: TheRock's nightly index)
#   HIPBLASLT_PY_DEVICE        a device wheel to add, e.g. gfx942; empty for a CPU-only lane
#   HIPBLASLT_PY_REQUIRE_GPU   1 on GPU lanes, so "no device" skips become failures
#   HIPBLASLT_PY_WORK_DIR      where the steps keep the venv (default: $RUNNER_TEMP/hipblaslt-py
#                              in CI, or a temporary directory when no step is named)
#   PYTHON                     the interpreter to build the venv from (default: python3)

set -euo pipefail

index="${HIPBLASLT_PY_ROCM_INDEX:-https://nightly.repo.amd.com/rocm/whl-next/}"
device="${HIPBLASLT_PY_DEVICE:-}"
python="${PYTHON:-python3}"
package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

work_dir="${HIPBLASLT_PY_WORK_DIR:-${RUNNER_TEMP:+${RUNNER_TEMP}/hipblaslt-py}}"
steps=("$@")
if [[ ${#steps[@]} -eq 0 ]]; then
    steps=(venv rocm build test)
    if [[ -z "${work_dir}" ]]; then
        work_dir="$(mktemp -d)"
    fi
    # Leave the directory before removing it; Windows can't delete the current directory.
    trap 'cd / && rm -rf "${work_dir}"' EXIT
fi
: "${work_dir:?set HIPBLASLT_PY_WORK_DIR, or run every step at once}"
if command -v cygpath >/dev/null 2>&1; then
    work_dir="$(cygpath -u "${work_dir}")"
fi
venv="${work_dir}/venv"

# The venv's own tools, which live in bin/ on Linux and Scripts/ on Windows.
venv_tool() {
    local name="$1"
    shift
    if [[ -x "${venv}/bin/${name}" ]]; then
        "${venv}/bin/${name}" "$@"
    else
        "${venv}/Scripts/${name}.exe" "$@"
    fi
}

use_rocm() {
    ROCM_PATH="$(venv_tool rocm-sdk path --root)"
    export ROCM_PATH
    # hipBLASLt's headers need clang on Windows; MSVC can't compile them.
    if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
        export CXX="${ROCM_PATH}/lib/llvm/bin/amdclang++.exe"
        export CC="${ROCM_PATH}/lib/llvm/bin/amdclang.exe"
        export CMAKE_GENERATOR=Ninja
    fi
}

step_venv() {
    rm -rf "${work_dir}"
    mkdir -p "${work_dir}"
    "${python}" -m venv "${venv}"
    venv_tool python -m pip install --upgrade pip
    venv_tool python -m pip install pytest
}

step_rocm() {
    : "${HIPBLASLT_PY_ROCM_VERSION:?set HIPBLASLT_PY_ROCM_VERSION to the pinned nightly}"
    local extras="libraries,devel"
    if [[ -n "${device}" ]]; then
        extras="${extras},device-${device}"
    fi
    venv_tool python -m pip install --index-url "${index}" "rocm[${extras}]==${HIPBLASLT_PY_ROCM_VERSION}"
    use_rocm
    echo "ROCm ${HIPBLASLT_PY_ROCM_VERSION} at ${ROCM_PATH}"
}

step_build() {
    use_rocm
    venv_tool python -m pip install -v "${package_dir}"
}

step_test() {
    use_rocm
    # Run from outside the source tree, so the tests import the installed package.
    cd "${work_dir}"
    venv_tool python -m pytest "${package_dir}/tests" -ra -v
}

step_clean() {
    cd /
    rm -rf "${work_dir}"
}

for step in "${steps[@]}"; do
    case "${step}" in
        venv | rocm | build | test | clean) "step_${step}" ;;
        *)
            echo "ci_lane.sh: unknown step '${step}'; expected venv, rocm, build, test or clean" >&2
            exit 2
            ;;
    esac
done
