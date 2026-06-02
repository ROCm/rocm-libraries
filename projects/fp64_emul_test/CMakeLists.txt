# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Standalone FP64 emulation accuracy / runtime test.
#
# Build instructions (from this directory):
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release \
#       [-DHIPBLASLT_BUILD_DIR=/path/to/hipblaslt/build/release] \
#       [-DHIPBLASLT_SRC_DIR=/path/to/hipblaslt]
#   cmake --build build -j$(nproc)
#   ./build/fp64_emul_accuracy [options]
#
# Default paths assume this project lives at projects/fp64_emul_test/
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
# We need:
#   (a) the build-tree libhipblaslt.so — for the public hipBLASLt API (matmul, etc.)
#   (b) the source-tree fp64_emulation.cpp — compiled directly into our binary
#       because fp64EmulatedGemm is a local (non-exported) symbol in the .so.
set(HIPBLASLT_BUILD_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../hipblaslt/build/release"
    CACHE PATH "Path to the hipblaslt build/release directory")

set(HIPBLASLT_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../hipblaslt"
    CACHE PATH "Path to the hipblaslt source directory")

# Sanity checks
if(NOT EXISTS "${HIPBLASLT_BUILD_DIR}/library/libhipblaslt.so")
    message(FATAL_ERROR
        "libhipblaslt.so not found.\n"
        "  Looked in: ${HIPBLASLT_BUILD_DIR}/library/\n"
        "  Set -DHIPBLASLT_BUILD_DIR=<path-to-build/release>")
endif()

set(_FP64_EMUL_SRC
    "${HIPBLASLT_SRC_DIR}/library/src/amd_detail/rocblaslt/src/fp64_emulation.cpp")
if(NOT EXISTS "${_FP64_EMUL_SRC}")
    message(FATAL_ERROR
        "fp64_emulation.cpp not found.\n"
        "  Looked in: ${_FP64_EMUL_SRC}\n"
        "  Set -DHIPBLASLT_SRC_DIR=<path-to-hipblaslt-source>")
endif()

# ── Imported target: build-tree libhipblaslt.so ───────────────────────────────
add_library(hipblaslt_bt SHARED IMPORTED)
set_target_properties(hipblaslt_bt PROPERTIES
    IMPORTED_LOCATION
        "${HIPBLASLT_BUILD_DIR}/library/libhipblaslt.so"
    INTERFACE_INCLUDE_DIRECTORIES
        "${HIPBLASLT_BUILD_DIR}/library/include")

# ── Executable ────────────────────────────────────────────────────────────────
# Compile fp64_emulation.cpp directly so we get fp64EmulatedGemm and the
# helper functions without needing them exported from the shared library.
add_executable(fp64_emul_accuracy
    src/fp64_emul_accuracy.cpp
    "${_FP64_EMUL_SRC}")

# Both sources contain __global__ kernels — must be compiled as HIP.
set_source_files_properties(
    src/fp64_emul_accuracy.cpp
    "${_FP64_EMUL_SRC}"
    PROPERTIES LANGUAGE HIP)

target_link_libraries(fp64_emul_accuracy
    PRIVATE
        hipblaslt_bt   # public hipBLASLt API (hipblasLtMatmul, layouts, etc.)
        hip::device)

target_compile_definitions(fp64_emul_accuracy
    PRIVATE ROCBLASLT_INTERNAL_API)

target_include_directories(fp64_emul_accuracy
    PRIVATE
        # Internal rocblaslt headers (fp64_emulation.hpp + rocblaslt.h)
        "${HIPBLASLT_SRC_DIR}/library/src/amd_detail/rocblaslt/src/include"
        "${HIPBLASLT_SRC_DIR}/library/src/amd_detail/rocblaslt/include"
        # Installed hipblaslt public headers (hipblaslt.h with hipblasLtBatchMode_t etc.)
        # These must come before /opt/rocm since the source tree is newer than the system install.
        "${HIPBLASLT_SRC_DIR}/hipblaslt-install/include"
        # Build-tree public hipblaslt headers
        "${HIPBLASLT_BUILD_DIR}/library/include"
        # System ROCm (hip, hipblas, hipblas-common, etc.)
        "/opt/rocm/include")

target_compile_features(fp64_emul_accuracy PRIVATE cxx_std_17)

# RPATH: make the binary find libhipblaslt.so from the build tree at runtime.
set_target_properties(fp64_emul_accuracy PROPERTIES
    BUILD_RPATH  "${HIPBLASLT_BUILD_DIR}/library"
    INSTALL_RPATH "${HIPBLASLT_BUILD_DIR}/library")
