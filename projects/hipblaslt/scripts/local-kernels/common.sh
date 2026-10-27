#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Shared paths for the hipBLASLt local kernel CI scripts. Source this file; do
# not execute it. A local run can export any of these before invoking a script.
# Defaults match the GitHub Actions layout: the repository root, a build
# directory, a virtualenv and a ROCm prefix populated by fetch-sdk.sh.

if [[ -n "${LOCAL_KERNELS_COMMON_LOADED:-}" ]]; then
    return 0
fi
LOCAL_KERNELS_COMMON_LOADED=1

_local_kernels_scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/local-kernels -> hipblaslt -> projects -> repository root
ROOT="$(cd "${_local_kernels_scripts}/../../../.." && pwd)"

: "${GITHUB_WORKSPACE:=${ROOT}}"
: "${LOCAL_KERNELS_BUILD_DIR:=${GITHUB_WORKSPACE}/build/local-kernels}"
: "${LOCAL_KERNELS_DEPS_DIR:=${GITHUB_WORKSPACE}/build/local-kernels-deps}"
: "${VENV_DIR:=${GITHUB_WORKSPACE}/.venv}"
: "${ROCM_PATH:=${GITHUB_WORKSPACE}/build/rocm-deps}"
: "${LOCAL_KERNELS_LOG_DIR:=${GITHUB_WORKSPACE}/local-kernels-test-logs}"
: "${LOCAL_KERNELS_ARCHITECTURE:=gfx942}"

export ROOT GITHUB_WORKSPACE LOCAL_KERNELS_BUILD_DIR LOCAL_KERNELS_DEPS_DIR VENV_DIR ROCM_PATH \
    LOCAL_KERNELS_LOG_DIR LOCAL_KERNELS_ARCHITECTURE

if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s\n' \
        "LOCAL_KERNELS_BUILD_DIR=${LOCAL_KERNELS_BUILD_DIR}" \
        "LOCAL_KERNELS_DEPS_DIR=${LOCAL_KERNELS_DEPS_DIR}" \
        "VENV_DIR=${VENV_DIR}" \
        "ROCM_PATH=${ROCM_PATH}" \
        "LOCAL_KERNELS_LOG_DIR=${LOCAL_KERNELS_LOG_DIR}" \
        "LOCAL_KERNELS_ARCHITECTURE=${LOCAL_KERNELS_ARCHITECTURE}" >> "${GITHUB_ENV}"
fi

mkdir -p "${LOCAL_KERNELS_LOG_DIR}"
