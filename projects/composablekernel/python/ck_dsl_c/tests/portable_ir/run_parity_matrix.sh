#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_parity_matrix.sh -- build a fresh shared libckc and run the backend-path
# parity matrix across ALL kernel instances x archs on both paths (engine import
# + recipe VM) vs the Python lowerer, with one LLVM flavor pinned on every path.
#
#   run_parity_matrix.sh [flavor=llvm20] [archs=gfx942,gfx950]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_parity_matrix"
FLAVOR="${1:-llvm20}"
ARCHES="${2:-gfx942,gfx950}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

echo ">> building C++ core (libckc_core.a) via CMake, then linking the flat-C portable-IR tooling into libckc.so"
cmake -S "$CKC" -B "$OUT/core" -DCMAKE_BUILD_TYPE=Debug >"$OUT/cmake.log" 2>&1 || { echo "cmake configure FAILED (see $OUT/cmake.log)"; exit 1; }
cmake --build "$OUT/core" --target ckc_core -j "$(nproc)" >>"$OUT/cmake.log" 2>&1 || { echo "ckc_core build FAILED (see $OUT/cmake.log)"; exit 1; }
mkdir -p "$OUT/obj"
( cd "$OUT/obj" && cc -std=c99 -O2 -fPIC -I "$CKC/include" -c "$CKC"/src/*.c ) || { echo "tooling compile FAILED"; exit 1; }
c++ -shared -fPIC "$OUT"/obj/*.o -Wl,--whole-archive "$OUT/core/libckc_core.a" -Wl,--no-whole-archive -lm -o "$OUT/libckc.so" || {
    echo "libckc.so link FAILED"; exit 1; }

export CKC_LIB="$OUT/libckc.so" CK_DSL_LLVM_FLAVOR="$FLAVOR" ARCHES="$ARCHES"
python3 -m ck_dsl.portable_ir.drivers.parity_matrix "${@:3}"
