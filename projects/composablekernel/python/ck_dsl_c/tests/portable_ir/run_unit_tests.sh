#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_unit_tests.sh -- build + run the C unit tests for the recipe-replay DOM
# decoders (src/portable_ir/cbor_dom.cpp, json_dom.cpp). The decoders are part of
# the C++ engine core archive, so we build libckc_core.a (CMake) and link the C
# test (which calls the extern "C" API) against it.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_unit_tests"
rm -rf "$OUT/obj"
mkdir -p "$OUT/obj"

echo ">> building C++ core (libckc_core.a) for the arena"
cmake -S "$CKC" -B "$OUT/core" -DCMAKE_BUILD_TYPE=Debug >"$OUT/cmake.log" 2>&1 || {
    echo "cmake configure FAILED (see $OUT/cmake.log)"; exit 1; }
cmake --build "$OUT/core" --target ckc_core -j "$(nproc)" >>"$OUT/cmake.log" 2>&1 || {
    echo "ckc_core build FAILED (see $OUT/cmake.log)"; exit 1; }

echo ">> compiling the C test (decoders are in the core archive)"
rc=0
( cd "$OUT/obj" && cc -std=c99 -O2 -Wall -Wextra -I "$CKC/include" -c "$HERE/test_cbor_dom.c" ) \
    || { echo "compile FAILED"; exit 1; }
c++ "$OUT"/obj/*.o -Wl,--whole-archive "$OUT/core/libckc_core.a" -Wl,--no-whole-archive \
    -lm -o "$OUT/test_cbor_dom" || { echo "link FAILED"; exit 1; }

echo ">> running"
"$OUT/test_cbor_dom" || rc=1
exit $rc
