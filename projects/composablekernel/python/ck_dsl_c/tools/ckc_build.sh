#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Canonical, single-entrypoint build for the ck_dsl C engine, the
# ck-dsl-provider hipDNN plugin, and the hipDNN graph demos.
#
# This is the ONE supported build path. It bakes in every flag, include path,
# and dependency that has historically been forgotten and produced a phantom
# failure:
#   * the engine archive is ALWAYS freshly built and the provider is pointed at
#     that fresh archive (never a stale checked-in one);
#   * the provider gets -DHIPDNN_ROOT / -DHIPDNN_BUILD_DIR / -DCMAKE_PREFIX_PATH
#     / -DCK_DSL_PROVIDER_C_JIT=ON;
#   * the demos get -D__HIP_PLATFORM_AMD__, the full hipDNN frontend/backend/
#     data_sdk/flatbuffers_sdk include set, the build-tree generated headers
#     (backend export header, frontend generated include), and the _deps
#     spdlog/json/flatbuffers includes; SDPA additionally gets
#     -DHIPDNN_ENABLE_SDPA.
#
# All build output goes under a single build root (default: a local /tmp dir;
# never NFS -- gcc/comgr are pathologically slow on NFS). Nothing is written
# back into the source tree.
#
# Usage:
#   ckc_build.sh [options]
#
# Options (all have sane defaults; override via flag or env):
#   --build-root DIR     Build output root         (env BUILD_ROOT)
#   --hipdnn-root DIR    hipDNN source tree        (env HIPDNN_ROOT)
#   --hipdnn-build DIR   hipDNN build dir          (env HIPDNN_BUILD_DIR)
#   --rocm DIR           ROCm prefix               (env ROCM_PATH, default /opt/rocm)
#   --arch GFX           GPU arch for demos        (env GPU_ARCH, default gfx950)
#   --sanitize           Build the engine with -DCKC_SANITIZE=ON
#   --no-demos           Skip the hipDNN graph demos (engine + provider only)
#   --run-gemm           Run the built GEMM demo (needs a GPU; M N K optional)
#   -h | --help          This help.
set -euo pipefail

# -------- locate the source trees relative to this script -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                       # .../ck_dsl_c
REPO_ROOT="$(cd "$ENGINE_DIR/../../../.." && pwd)"              # repo root
PROVIDER_DIR="$REPO_ROOT/dnn-providers/ck-dsl-provider"

# -------- defaults ----------------------------------------------------------
BUILD_ROOT="${BUILD_ROOT:-/tmp/ck_dsl_build}"
HIPDNN_ROOT="${HIPDNN_ROOT:-$REPO_ROOT/projects/hipdnn}"
HIPDNN_BUILD_DIR="${HIPDNN_BUILD_DIR:-}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx950}"
DO_SANITIZE=0
DO_DEMOS=1
DO_RUN_GEMM=0
RUN_GEMM_ARGS=()

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build-root)   BUILD_ROOT="$2"; shift 2 ;;
    --hipdnn-root)  HIPDNN_ROOT="$2"; shift 2 ;;
    --hipdnn-build) HIPDNN_BUILD_DIR="$2"; shift 2 ;;
    --rocm)         ROCM_PATH="$2"; shift 2 ;;
    --arch)         GPU_ARCH="$2"; shift 2 ;;
    --sanitize)     DO_SANITIZE=1; shift ;;
    --no-demos)     DO_DEMOS=0; shift ;;
    --run-gemm)     DO_RUN_GEMM=1; shift
                    while [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; do
                      RUN_GEMM_ARGS+=("$1"); shift; done ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# Resolve the hipDNN build dir the same way the provider cmake does.
if [ -z "$HIPDNN_BUILD_DIR" ]; then
  if [ -d "$HIPDNN_ROOT/build/release" ]; then
    HIPDNN_BUILD_DIR="$HIPDNN_ROOT/build/release"
  else
    HIPDNN_BUILD_DIR="$HIPDNN_ROOT/build"
  fi
fi

JOBS="$(nproc 2>/dev/null || echo 4)"
ENGINE_BUILD="$BUILD_ROOT/engine"
PROVIDER_BUILD="$BUILD_ROOT/provider"
DEMO_BUILD="$BUILD_ROOT/demos"
mkdir -p "$ENGINE_BUILD" "$PROVIDER_BUILD" "$DEMO_BUILD"

echo "=========================================================================="
echo "ck_dsl canonical build"
echo "  repo root      : $REPO_ROOT"
echo "  engine src     : $ENGINE_DIR"
echo "  provider src   : $PROVIDER_DIR"
echo "  build root     : $BUILD_ROOT"
echo "  hipDNN root    : $HIPDNN_ROOT"
echo "  hipDNN build   : $HIPDNN_BUILD_DIR"
echo "  ROCm           : $ROCM_PATH"
echo "  arch (demos)   : $GPU_ARCH"
echo "  sanitize       : $DO_SANITIZE   demos: $DO_DEMOS"
echo "=========================================================================="

# ---------------------------------------------------------------------------
# 1) Engine: always a FRESH static archive. This is the authority -- the
#    provider links THIS, never a checked-in build/ archive.
# ---------------------------------------------------------------------------
echo "[1/3] building ck_dsl engine (ckc_core) ..."
ENGINE_CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
[ "$DO_SANITIZE" -eq 1 ] && ENGINE_CMAKE_ARGS+=(-DCKC_SANITIZE=ON)
cmake -S "$ENGINE_DIR" -B "$ENGINE_BUILD" "${ENGINE_CMAKE_ARGS[@]}"
cmake --build "$ENGINE_BUILD" --target ckc_core -j"$JOBS"

CKC_LIB="$ENGINE_BUILD/libckc_core.a"
if [ ! -f "$CKC_LIB" ]; then
  echo "ERROR: engine build did not produce $CKC_LIB" >&2
  exit 1
fi
echo "      fresh engine archive: $CKC_LIB"

# ---------------------------------------------------------------------------
# 2) Provider plugin, linked against the FRESH engine archive.
# ---------------------------------------------------------------------------
echo "[2/3] building ck-dsl-provider plugin ..."
cmake -S "$PROVIDER_DIR" -B "$PROVIDER_BUILD" \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DHIPDNN_ROOT="$HIPDNN_ROOT" \
  -DHIPDNN_BUILD_DIR="$HIPDNN_BUILD_DIR" \
  -DCK_DSL_PROVIDER_C_JIT=ON \
  -DCKC_LIB="$CKC_LIB"
cmake --build "$PROVIDER_BUILD" -j"$JOBS"

PLUGIN_SO="$(find "$PROVIDER_BUILD" -name 'libck_dsl_provider_plugin.so' | head -1)"
echo "      plugin: ${PLUGIN_SO:-<not found>}"

# ---------------------------------------------------------------------------
# 3) hipDNN graph demos. Standalone programs that load the plugin at runtime;
#    they compile against the hipDNN frontend + the generated build-tree
#    headers and link the hipDNN backend .so.
# ---------------------------------------------------------------------------
if [ "$DO_DEMOS" -eq 0 ]; then
  echo "[3/3] demos skipped (--no-demos)"
  echo "DONE."
  exit 0
fi

echo "[3/3] building hipDNN graph demos ..."

# The complete, do-not-forget include set (source + generated build-tree).
DEMO_INCLUDES=(
  -I"$HIPDNN_ROOT/frontend/include"
  -I"$HIPDNN_BUILD_DIR/frontend/include"
  -I"$HIPDNN_ROOT/backend/include"
  -I"$HIPDNN_BUILD_DIR/backend/src/backend/include"   # generated export header
  -I"$HIPDNN_ROOT/data_sdk/include"
  -I"$HIPDNN_BUILD_DIR/data_sdk/include"
  -I"$HIPDNN_ROOT/flatbuffers_sdk/include"
  -I"$HIPDNN_ROOT/plugin_sdk/include"
  -I"$HIPDNN_BUILD_DIR/_deps/spdlog-src/include"
  -I"$HIPDNN_BUILD_DIR/_deps/json-src/include"
  -I"$HIPDNN_BUILD_DIR/_deps/flatbuffers-src/include"
  -I"$PROVIDER_DIR/runtime/include"
  -I"$PROVIDER_DIR/src"
)
DEMO_CXXFLAGS=(-std=c++17 -O2 -D__HIP_PLATFORM_AMD__)
DEMO_LINK=(-L"$HIPDNN_BUILD_DIR/lib" -lhipdnn_backend
           -Wl,-rpath,"$HIPDNN_BUILD_DIR/lib")

build_demo() {
  local name="$1" src="$2"; shift 2
  local extra_defs=("$@")
  echo "      -> $name"
  hipcc "${DEMO_CXXFLAGS[@]}" "${extra_defs[@]}" "${DEMO_INCLUDES[@]}" \
    -c "$src" -o "$DEMO_BUILD/$name.o"
  hipcc -std=c++17 "$DEMO_BUILD/$name.o" -o "$DEMO_BUILD/$name" "${DEMO_LINK[@]}"
}

build_demo gemm_demo "$PROVIDER_DIR/integration_tests/EndToEndGemmDemo.cpp"
build_demo conv_demo "$PROVIDER_DIR/integration_tests/EndToEndConvDemo.cpp"
# SDPA demo needs the SDPA frontend surface enabled.
build_demo sdpa_demo "$PROVIDER_DIR/integration_tests/EndToEndSdpaDemo.cpp" \
  -DHIPDNN_ENABLE_SDPA

echo "      demos in: $DEMO_BUILD"

# ---------------------------------------------------------------------------
# Optional: run the GEMM demo (needs a GPU; the plugin must be discoverable).
# ---------------------------------------------------------------------------
if [ "$DO_RUN_GEMM" -eq 1 ]; then
  echo "[run] gemm_demo ${RUN_GEMM_ARGS[*]:-} (plugin=$PLUGIN_SO)"
  CK_DSL_C_JIT="${CK_DSL_C_JIT:-1}" \
  CK_DSL_KERNEL_LIB_PATH="${CK_DSL_KERNEL_LIB_PATH:-$PROVIDER_DIR/kernels/$GPU_ARCH}" \
  HIPDNN_PLUGIN_PATH="${HIPDNN_PLUGIN_PATH:-$(dirname "$PLUGIN_SO")}" \
    "$DEMO_BUILD/gemm_demo" "${RUN_GEMM_ARGS[@]}"
fi

echo "DONE."
