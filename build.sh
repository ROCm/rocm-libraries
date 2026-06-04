#!/bin/bash
# build.sh — 编译 ROCm Libraries meta-repo 全部可用组件
set -e

SRC_DIR="/data/ROCm/rocm-libraries"
BUILD_DIR="/data/ROCm/rocm-libraries/build"

export CC=/opt/llvm-23/bin/clang
export CXX=/opt/llvm-23/bin/clang++
export PATH=/opt/llvm-23/bin:/opt/rocm/bin:$PATH

echo "==========================================="
echo "ROCm Libraries 全量编译"
echo "编译器: $(clang --version | head -1)"
echo "==========================================="

CMAKE_FLAGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=/opt/rocm
    -DCMAKE_PREFIX_PATH=/opt/rocm
    -DGPU_TARGETS=gfx1200
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran
    -DBUILD_TESTING=OFF
    -DROCROLLER_BUILD_TESTING=OFF \
    -DROCM_LIBS_ENABLE_COMPONENTS="mxdatagenerator;origami;rocprim;rocrand"
)

cmake -S "$SRC_DIR" -B "$BUILD_DIR" "${CMAKE_FLAGS[@]}"
ninja -C "$BUILD_DIR" -j24
echo "✅ 编译完成"
