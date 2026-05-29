#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build the single-kernel parity harness (deliverable d) against one generated
# kernel header. Compiling needs hipcc + the CK Tile include tree; RUNNING the
# resulting binary needs a GPU (this environment has none -- build here, run on a
# GPU node).
#
# Usage:
#   ./build_harness.sh [generated_kernel_header.hpp] [gfx_arch]
#
# With no args it auto-picks the single gemm_*.hpp under generated/parity_single
# (what drive_codegen.py emits by default) and targets gfx942.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CK_INCLUDE="$(cd "${HERE}/../../include" && pwd)"

HEADER="${1:-}"
ARCH="${2:-gfx942}"

if [[ -z "${HEADER}" ]]; then
    mapfile -t found < <(find "${HERE}/generated" -name 'gemm_*.hpp' -not -path '*/dispatcher_wrappers/*' 2>/dev/null)
    if [[ "${#found[@]}" -eq 0 ]]; then
        echo "error: no generated gemm_*.hpp found. Run drive_codegen.py first." >&2
        exit 1
    fi
    if [[ "${#found[@]}" -gt 1 ]]; then
        echo "error: multiple generated kernels found; pass one explicitly:" >&2
        printf '  %s\n' "${found[@]}" >&2
        exit 1
    fi
    HEADER="${found[0]}"
fi

HEADER="$(cd "$(dirname "${HEADER}")" && pwd)/$(basename "${HEADER}")"
OUT="${HERE}/harness"

echo "header : ${HEADER}"
echo "arch   : ${ARCH}"
echo "include: ${CK_INCLUDE}"
echo "building ${OUT} ..."

set -x
hipcc -std=c++17 \
    --offload-arch="${ARCH}" \
    -I "${CK_INCLUDE}" \
    -DCK_TILE_SINGLE_KERNEL_INCLUDE \
    -DPARITY_KERNEL_HEADER="\"${HEADER}\"" \
    "${HERE}/harness.cpp" \
    -o "${OUT}"
set +x

echo "built: ${OUT}"
echo "run on a GPU node, e.g.:  ${OUT} -m=512 -n=512 -k=512 -verify=1"
