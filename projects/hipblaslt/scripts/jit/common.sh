#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Shared paths for the hipBLASLt JIT CI scripts. Source this file; do not execute it.
# A local run can export any of these before invoking a script. Defaults match the
# GitHub Actions layout: the repository root, a build directory, a virtualenv and
# a ROCm prefix populated by fetch-sdk.sh.

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
