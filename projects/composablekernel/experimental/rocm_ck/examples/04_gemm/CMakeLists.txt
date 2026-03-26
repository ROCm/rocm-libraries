# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build for the kpack GEMM example. Compiles fp32/fp16/bf16 GEMM kernel variants
# per architecture, packs into a .kpack archive, and builds the host-side loader.
#
# Usage:
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#       -DCMAKE_PREFIX_PATH=/opt/rocm \
#       -DGPU_TARGETS="gfx90a;gfx942"
#   ninja -C build

cmake_minimum_required(VERSION 3.21)
project(kpack_gemm LANGUAGES CXX)

enable_language(HIP)
find_package(hip REQUIRED)

set(GPU_TARGETS "gfx90a;gfx942;gfx950" CACHE STRING "GPU architectures to compile .hsaco for")
set(KPACK_ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path (for --rocm-path)")

# CK Tile headers — four levels up from this example directory
set(CK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
get_filename_component(CK_INCLUDE_DIR "${CK_INCLUDE_DIR}" ABSOLUTE)

# Pull in the vendored kpack runtime library and shared rocm_ck headers
# Guards allow this example to work both standalone and as part of the umbrella build.
if(NOT TARGET rocm_kpack)
    add_subdirectory(../../rocm_kpack rocm_kpack_build)
endif()
if(NOT TARGET rocm_ck)
    add_subdirectory(../../include rocm_ck_build)
endif()

# --- Kernel variants: fp32, fp16, bf16, fp16 with 32×32 warp tile, fp16+add, fp16+add+relu ---
set(KERNEL_VARIANTS gemm_fp32 gemm_fp16 gemm_bf16 gemm_fp16_w32 gemm_fp16_add gemm_fp16_add_relu)

# rocm_ck headers for device code (DataType, Layout enums used by gemm_api.hpp)
set(ROCM_CK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
get_filename_component(ROCM_CK_INCLUDE_DIR "${ROCM_CK_INCLUDE_DIR}" ABSOLUTE)

# --- Compile each variant x architecture into .hsaco code objects ---
# Uses --cuda-device-only to produce raw code objects without host stubs.
# -std=c++20 is required for CK Tile headers and GemmKernel schema (consteval, structural NTTPs).
set(HSACO_FILES "")
foreach(variant IN LISTS KERNEL_VARIANTS)
    foreach(arch IN LISTS GPU_TARGETS)
        set(hsaco_output "${CMAKE_CURRENT_BINARY_DIR}/${variant}_${arch}.hsaco")
        add_custom_command(
            OUTPUT ${hsaco_output}
            COMMAND ${CMAKE_HIP_COMPILER}
                    --offload-arch=${arch}
                    --cuda-device-only
                    -x hip
                    -std=c++20
                    --rocm-path=${KPACK_ROCM_PATH}
                    -I${CK_INCLUDE_DIR}
                    -I${ROCM_CK_INCLUDE_DIR}
                    -I${CMAKE_CURRENT_SOURCE_DIR}
                    -o ${hsaco_output}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${variant}.hip
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${variant}.hip
                    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_kernel.hpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_dev.hpp
            COMMENT "Compiling ${variant}.hip -> ${arch}.hsaco"
        )
        list(APPEND HSACO_FILES ${hsaco_output})
    endforeach()
endforeach()

# --- Pack .hsaco files into a kpack archive ---
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gemm.kpack
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pack.py ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${HSACO_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
    COMMENT "Creating gemm.kpack archive"
)
add_custom_target(gemm_kpack_archive ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gemm.kpack)

# --- Build the host-side loader executable ---
# Links against hip::host (host-only HIP runtime) and rocm_kpack (archive reader).
# Device code is loaded at runtime via hipModuleLoadData, not compiled into this binary.
add_executable(kpack_gemm main.cpp cpu_ref.cpp)
target_compile_features(kpack_gemm PRIVATE cxx_std_20)
target_include_directories(kpack_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kpack_gemm PRIVATE rocm_ck rocm_kpack hip::host)
add_dependencies(kpack_gemm gemm_kpack_archive)
