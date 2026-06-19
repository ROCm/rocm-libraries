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

# Pick a compiler: hipcc if present, else a plain C++17 host compiler (selection
# is CPU-only and header-only, so no HIP toolchain is strictly required).
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
  if command -v hipcc >/dev/null 2>&1; then CXX=hipcc
  elif command -v g++ >/dev/null 2>&1; then CXX=g++
  else CXX=c++; fi
fi
echo "[run] building cpp_select with $CXX ..."
"$CXX" -std=c++17 -O1 -I "$PROVIDER/runtime/include" \
  "$HERE/cpp_select.cpp" -o "$HERE/cpp_select"

run_one() {
  local dtype="$1" bundle="$2"
  echo "[run] === dtype=$dtype bundle=$bundle ==="
  "$HERE/cpp_select" --bundle "$bundle" --shapes "$HERE/shapes.txt" \
    --arch "$ARCH" --dtype "$dtype" > "$HERE/cpp_picks_$dtype.jsonl"
  PYTHONPATH="$PYDIR" python3 "$HERE/parity_check.py" \
    --cpp-jsonl "$HERE/cpp_picks_$dtype.jsonl" --shapes "$HERE/shapes.txt" \
    --arch "$ARCH" --dtype "$dtype"
}

# fp16: the shipped per-arch bundle.
run_one fp16 "$BUNDLE"

# bf16 (new family): no shipped HSACO bundle yet, so synthesize a manifest-only
# bundle from the Python bf16 candidates and run the REAL C++ Dispatcher over it.
BF16_BUNDLE="$HERE/.bf16_bundle_$ARCH"
echo "[run] generating bf16 manifest bundle ..."
PYTHONPATH="$PYDIR" python3 "$HERE/gen_bf16_bundle.py" --arch "$ARCH" --out "$BF16_BUNDLE"
run_one bf16 "$BF16_BUNDLE"

# ---- norm family (rmsnorm / layernorm), op=norm ----------------------------
run_norm() {
  local dtype="$1" bundle="$2"
  echo "[run] === op=norm dtype=$dtype bundle=$bundle ==="
  "$HERE/cpp_select" --op norm --bundle "$bundle" --shapes "$HERE/shapes_norm.txt" \
    --arch "$ARCH" --dtype "$dtype" > "$HERE/cpp_picks_norm.jsonl"
  PYTHONPATH="$PYDIR" python3 "$HERE/parity_check_norm.py" \
    --cpp-jsonl "$HERE/cpp_picks_norm.jsonl" --shapes "$HERE/shapes_norm.txt" \
    --arch "$ARCH" --dtype "$dtype"
}
NORM_BUNDLE="$HERE/.norm_bundle_$ARCH"
echo "[run] generating norm manifest bundle ..."
PYTHONPATH="$PYDIR" python3 "$HERE/gen_norm_bundle.py" --arch "$ARCH" --out "$NORM_BUNDLE"
run_norm fp16 "$NORM_BUNDLE"

# ---- conv family (forward implicit-GEMM), op=conv --------------------------
run_conv() {
  local dtype="$1" bundle="$2"
  echo "[run] === op=conv dtype=$dtype bundle=$bundle ==="
  "$HERE/cpp_select" --op conv --bundle "$bundle" --shapes "$HERE/shapes_conv.txt" \
    --arch "$ARCH" --dtype "$dtype" > "$HERE/cpp_picks_conv.jsonl"
  PYTHONPATH="$PYDIR" python3 "$HERE/parity_check_conv.py" \
    --cpp-jsonl "$HERE/cpp_picks_conv.jsonl" --shapes "$HERE/shapes_conv.txt" \
    --arch "$ARCH" --dtype "$dtype"
}
CONV_BUNDLE="$HERE/.conv_bundle_$ARCH"
echo "[run] generating conv manifest bundle ..."
PYTHONPATH="$PYDIR" python3 "$HERE/gen_conv_bundle.py" --arch "$ARCH" --out "$CONV_BUNDLE"
run_conv fp16 "$CONV_BUNDLE"

# ---- attention family (unified FMHA path selection), op=attention ----------
run_attn() {
  local dtype="$1" bundle="$2"
  echo "[run] === op=attention dtype=$dtype bundle=$bundle ==="
  "$HERE/cpp_select" --op attention --bundle "$bundle" \
    --shapes "$HERE/shapes_attention.txt" --arch "$ARCH" --dtype "$dtype" \
    > "$HERE/cpp_picks_attention.jsonl"
  PYTHONPATH="$PYDIR" python3 "$HERE/parity_check_attention.py" \
    --cpp-jsonl "$HERE/cpp_picks_attention.jsonl" \
    --shapes "$HERE/shapes_attention.txt" --arch "$ARCH" --dtype "$dtype"
}
ATTN_BUNDLE="$HERE/.attn_bundle_$ARCH"
echo "[run] generating attention manifest bundle ..."
PYTHONPATH="$PYDIR" python3 "$HERE/gen_attention_bundle.py" --arch "$ARCH" --out "$ATTN_BUNDLE"
run_attn fp16 "$ATTN_BUNDLE"

# ---- fused MoE family (mega-kernel element path), op=moe -------------------
run_moe() {
  local bundle="$1"
  echo "[run] === op=moe bundle=$bundle ==="
  # dtype is per-line in shapes_moe.txt; the --dtype flag is unused for moe.
  "$HERE/cpp_select" --op moe --bundle "$bundle" --shapes "$HERE/shapes_moe.txt" \
    --arch "$ARCH" > "$HERE/cpp_picks_moe.jsonl"
  PYTHONPATH="$PYDIR" python3 "$HERE/parity_check_moe.py" \
    --cpp-jsonl "$HERE/cpp_picks_moe.jsonl" --shapes "$HERE/shapes_moe.txt" \
    --arch "$ARCH"
}
MOE_BUNDLE="$HERE/.moe_bundle_$ARCH"
echo "[run] generating moe manifest bundle ..."
PYTHONPATH="$PYDIR" python3 "$HERE/gen_moe_bundle.py" --arch "$ARCH" --out "$MOE_BUNDLE"
run_moe "$MOE_BUNDLE"
