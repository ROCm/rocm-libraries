#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# End-to-end driver for the ck-dsl GEMM dispatcher C++<->Python selection-parity
# test. CPU-only: builds the C++ selection harness, runs the REAL ck_dsl::Dispatcher
# over the REAL shipped per-arch manifest bundle, then compares against the Python
# ck_dsl.dispatch dispatcher over the same shape corpus.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDER="$(cd "$HERE/../.." && pwd)"
REPO="$(cd "$PROVIDER/../.." && pwd)"
PYDIR="$REPO/projects/composablekernel/python"
ARCH="${1:-gfx950}"
BUNDLE="$PROVIDER/kernels/$ARCH"

echo "[run] arch=$ARCH bundle=$BUNDLE"
echo "[run] building cpp_select ..."
hipcc -std=c++17 -O1 -I "$PROVIDER/runtime/include" \
  "$HERE/cpp_select.cpp" -o "$HERE/cpp_select"

echo "[run] running C++ Dispatcher ..."
"$HERE/cpp_select" --bundle "$BUNDLE" --shapes "$HERE/shapes.txt" --arch "$ARCH" \
  > "$HERE/cpp_picks.jsonl"

echo "[run] running Python dispatcher + parity compare ..."
PYTHONPATH="$PYDIR" python3 "$HERE/parity_check.py" \
  --cpp-jsonl "$HERE/cpp_picks.jsonl" --shapes "$HERE/shapes.txt" --arch "$ARCH"
