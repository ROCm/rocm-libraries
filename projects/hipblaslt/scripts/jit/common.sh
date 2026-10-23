#!/usr/bin/env bash
# Shared paths for the hipBLASLt JIT CI scripts. Source this file; do not execute it.
# A local run can export any of these before invoking a script. Defaults match the
# GitHub Actions layout: the repository root, a build directory, a virtualenv and
# a ROCm prefix populated by fetch-sdk.sh.

if [[ -n "${JIT_COMMON_LOADED:-}" ]]; then
    return 0
fi
JIT_COMMON_LOADED=1

_jit_scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/jit -> hipblaslt -> projects -> repository root
ROOT="$(cd "${_jit_scripts}/../../../.." && pwd)"

: "${GITHUB_WORKSPACE:=${ROOT}}"
: "${JIT_BUILD_DIR:=${GITHUB_WORKSPACE}/build/jit}"
: "${JIT_DEPS_DIR:=${GITHUB_WORKSPACE}/build/jit-deps}"
: "${VENV_DIR:=${GITHUB_WORKSPACE}/.venv}"
: "${ROCM_PATH:=${GITHUB_WORKSPACE}/build/rocm-deps}"
: "${JIT_LOG_DIR:=${GITHUB_WORKSPACE}/jit-test-logs}"
: "${JIT_ARCHITECTURE:=gfx950}"

export ROOT GITHUB_WORKSPACE JIT_BUILD_DIR JIT_DEPS_DIR VENV_DIR ROCM_PATH JIT_LOG_DIR JIT_ARCHITECTURE

if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s\n' \
        "JIT_BUILD_DIR=${JIT_BUILD_DIR}" \
        "JIT_DEPS_DIR=${JIT_DEPS_DIR}" \
        "VENV_DIR=${VENV_DIR}" \
        "ROCM_PATH=${ROCM_PATH}" \
        "JIT_LOG_DIR=${JIT_LOG_DIR}" \
        "JIT_ARCHITECTURE=${JIT_ARCHITECTURE}" >> "${GITHUB_ENV}"
fi

jit_log() {
    local name="$1"
    shift
    mkdir -p "${JIT_LOG_DIR}"
    "$@" 2>&1 | tee "${JIT_LOG_DIR}/${name}"
}
