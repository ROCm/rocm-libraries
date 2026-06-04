#!/bin/bash
# build-rocm-optimized.sh — 编译 rocWMMA + rocBLAS 使用我们自己的 LLVM 23
set -e

export CC=/opt/llvm-23/bin/clang
export CXX=/opt/llvm-23/bin/clang++
export HIP_CLANG_PATH=/opt/llvm-23/bin
export ROCM_PATH=/opt/rocm
JOBS=24

echo "=== 编译器 ===" 
$CXX --version | head -1

build_rocwmma() {
    echo ""
    echo "=== 编译 rocWMMA ==="
    mkdir -p /data/ROCm/rocWMMA/build
    cd /data/ROCm/rocWMMA/build
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DGPU_TARGETS=gfx1200 \
        -DROCWMMA_BUILD_TESTS=OFF \
        -DROCWMMA_BUILD_SAMPLES=OFF
    ninja -j$JOBS
    echo "✅ rocWMMA 完成"
}

build_rocblas() {
    echo ""
    echo "=== 编译 rocBLAS ==="
    mkdir -p /data/ROCm/rocBLAS/build
    cd /data/ROCm/rocBLAS/build
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DGPU_TARGETS=gfx1200 \
        -DBUILD_WITH_TENSILE=OFF \
        -DBUILD_CLIENTS_TESTS=OFF \
        -DBUILD_CLIENTS_SAMPLES=OFF
    ninja -j$JOBS
    echo "✅ rocBLAS 完成"
}

case "${1:-all}" in
    rocwmma) build_rocwmma ;;
    rocblas) build_rocblas ;;
    all)
        build_rocwmma
        build_rocblas
        ;;
esac
