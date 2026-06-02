# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Standalone FP64 emulation accuracy / runtime test.
#
# Build instructions (from this directory):
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release \
#       [-DHIPBLASLT_BUILD_DIR=/path/to/hipblaslt/build/release]
#   cmake --build build -j$(nproc)
#   ./build/fp64_emul_accuracy [options]
#
# Default path assumes this project lives at projects/fp64_emul_test/
# and hipblaslt has been built at projects/hipblaslt/build/release/.

cmake_minimum_required(VERSION 3.25)

# HIP must be listed in LANGUAGES so CMake sets up CMAKE_HIP_* variables
# and GPU architecture detection.  GPU arch defaults to gfx942 (MI300X) but
# can be overridden via -DCMAKE_HIP_ARCHITECTURES=gfx...
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "HIP GPU architectures")
endif()

project(fp64_emul_accuracy LANGUAGES CXX HIP)

# ── ROCm / HIP ────────────────────────────────────────────────────────────────
find_package(hip REQUIRED)

# ── hipBLASLt build tree ──────────────────────────────────────────────────────
# We need the build-tree libhipblaslt.so for the public hipBLASLt API.
set(HIPBLASLT_BUILD_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../hipblaslt/build/release"
    CACHE PATH "Path to the hipblaslt build/release directory")

# Sanity check
if(NOT EXISTS "${HIPBLASLT_BUILD_DIR}/library/libhipblaslt.so")
    message(FATAL_ERROR
        "libhipblaslt.so not found.\n"
        "  Looked in: ${HIPBLASLT_BUILD_DIR}/library/\n"
        "  Set -DHIPBLASLT_BUILD_DIR=<path-to-build/release>")
endif()

# ── Imported target: build-tree libhipblaslt.so ───────────────────────────────
# The install-tree headers (hipblaslt-install/include) are always up-to-date
# after "cmake --install"; the build-tree copy may be stale.
add_library(hipblaslt_bt SHARED IMPORTED)
set_target_properties(hipblaslt_bt PROPERTIES
    IMPORTED_LOCATION
        "${HIPBLASLT_BUILD_DIR}/library/libhipblaslt.so"
    INTERFACE_INCLUDE_DIRECTORIES
        "${CMAKE_CURRENT_SOURCE_DIR}/../hipblaslt/hipblaslt-install/include")

# ── Executable ────────────────────────────────────────────────────────────────
# fp64_emul_accuracy.cpp contains __global__ kernels (DD reference GEMM, error
# computation) and must be compiled as HIP.  It uses only the public hipBLASLt
# API — no internal headers or extra source files are needed.
add_executable(fp64_emul_accuracy
    src/fp64_emul_accuracy.cpp)

set_source_files_properties(
    src/fp64_emul_accuracy.cpp
    PROPERTIES LANGUAGE HIP)

target_link_libraries(fp64_emul_accuracy
    PRIVATE
        hipblaslt_bt   # public hipBLASLt API
        hip::device)

target_include_directories(fp64_emul_accuracy
    PRIVATE
        # System ROCm (hip, hipblas-common, etc.)
        "/opt/rocm/include")

target_compile_features(fp64_emul_accuracy PRIVATE cxx_std_17)

# RPATH: make the binary find libhipblaslt.so from the build tree at runtime.
set_target_properties(fp64_emul_accuracy PROPERTIES
    BUILD_RPATH  "${HIPBLASLT_BUILD_DIR}/library"
    INSTALL_RPATH "${HIPBLASLT_BUILD_DIR}/library")
