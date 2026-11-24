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
#   test   run the bindings' tests (only the CPU ones when no device wheel is named)
#   clean  remove the venv
#
# Each command is echoed before it runs, so the log shows exactly what a lane did.
#
# Environment:
#   HIPBLASLT_PY_ROCM_VERSION  the nightly to install (required by rocm), e.g. 10.3.0a20261116
#   HIPBLASLT_PY_ROCM_INDEX    the wheel index (default: TheRock's nightly index)
#   HIPBLASLT_PY_DEVICE        a device wheel to add, e.g. gfx942; empty for a CPU-only lane
#   HIPBLASLT_PY_REQUIRE_GPU   1 on GPU lanes, so "no device" skips become failures
#   HIPBLASLT_PY_WORK_DIR      where the steps keep the venv, as WORK_DIR/venv (default:
#                              $RUNNER_TEMP/hipblaslt-py in CI, or a temporary directory when no
#                              step is named). Only the venv inside it is ever removed.
#   PYTHON                     the interpreter to build the venv from (default: python3)

set -euo pipefail

index="${HIPBLASLT_PY_ROCM_INDEX:-https://nightly.repo.amd.com/rocm/whl-next/}"
device="${HIPBLASLT_PY_DEVICE:-}"
python="${PYTHON:-python3}"
package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Echo a command (to stderr, so captured output stays clean), then run it.
run() {
    {
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    } >&2
    "$@"
}

work_dir="${HIPBLASLT_PY_WORK_DIR:-${RUNNER_TEMP:+${RUNNER_TEMP}/hipblaslt-py}}"
# A directory this script made, and so may remove whole. A caller's directory never is.
own_dir=""
steps=("$@")
if [[ ${#steps[@]} -eq 0 ]]; then
    steps=(venv rocm build test)
    if [[ -z "${work_dir}" ]]; then
        work_dir="$(mktemp -d)"
        own_dir="${work_dir}"
    fi
    trap step_clean EXIT
fi
: "${work_dir:?set HIPBLASLT_PY_WORK_DIR, or run every step at once}"
if command -v cygpath >/dev/null 2>&1; then
    work_dir="$(cygpath -u "${work_dir}")"
    [[ -z "${own_dir}" ]] || own_dir="${work_dir}"
fi
# Absolute before any step changes directory, so test and clean find the same place.
if [[ "${work_dir}" != /* ]]; then
    work_dir="${PWD}/${work_dir}"
fi
venv="${work_dir}/venv"

# The venv's own tools, which live in bin/ on Linux and Scripts/ on Windows.
venv_tool() {
    local name="$1"
    shift
    if [[ -x "${venv}/bin/${name}" ]]; then
        run "${venv}/bin/${name}" "$@"
    else
        run "${venv}/Scripts/${name}.exe" "$@"
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
    run rm -rf "${venv}"
    run mkdir -p "${work_dir}"
    run "${python}" -m venv "${venv}"
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
    local select=()
    if [[ -z "${device}" ]]; then
        # A CPU lane runs the CPU tests even on a machine that happens to have a GPU.
        select=(-m "not gpu")
    fi
    # Run from outside the source tree, so the tests import the installed package.
    run cd "${work_dir}"
    venv_tool python -m pytest "${package_dir}/tests" -ra -v ${select[@]+"${select[@]}"}
}

step_clean() {
    # Leave the directory before removing it; Windows can't delete the current directory.
    cd /
    if [[ -n "${own_dir}" ]]; then
        run rm -rf "${own_dir}"
    else
        run rm -rf "${venv}"
    fi
}

echo "ci_lane.sh: steps '${steps[*]}', work dir ${work_dir}, device '${device:-none}'," \
    "ROCm ${HIPBLASLT_PY_ROCM_VERSION:-unset}"
for step in "${steps[@]}"; do
    echo "==> ${step}"
    case "${step}" in
        venv | rocm | build | test | clean) "step_${step}" ;;
        *)
            echo "ci_lane.sh: unknown step '${step}'; expected venv, rocm, build, test or clean" >&2
            exit 2
            ;;
    esac
done
