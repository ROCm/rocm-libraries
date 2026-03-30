# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build for the rocm_ck kpack vector-add example with multiple variants.
# Compiles multiple type × block-size variants per architecture, packs into a .kpack
# archive, and builds the host-side loader.
#
# Usage:
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#       -DCMAKE_PREFIX_PATH=/opt/rocm \
#       -DGPU_TARGETS="gfx90a;gfx942"
#   ninja -C build

cmake_minimum_required(VERSION 3.21)
project(kpack_rocm_ck_vector_add LANGUAGES CXX)

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

# --- Kernel variants ---
set(KERNEL_VARIANTS
    vector_add_fp32_b256  vector_add_fp32_b512  vector_add_fp32_b1024
    vector_add_fp16_b512  vector_add_fp16_b1024
    vector_add_bf16_b512  vector_add_bf16_b1024
    vector_add_fp32_b2048_w8
    vector_add_fp16_b1024_w2
    # Mixed-type variants
    vector_add_fp16_fp32_b1024
    vector_add_fp32_fp16_b1024
    vector_add_bf16_fp32_b1024
)

# --- Compile each variant × architecture into .hsaco code objects ---
# Uses --cuda-device-only to produce raw code objects without host stubs.
# -std=c++20 is required for struct NTTPs and consteval.
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
                    -I${CMAKE_CURRENT_SOURCE_DIR}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    -o ${hsaco_output}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${variant}.hip
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${variant}.hip
                    ${CMAKE_CURRENT_SOURCE_DIR}/vector_add_variants.hpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/rocm_ck/elementwise_dev.hpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/rocm_ck/elementwise_spec.hpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/rocm_ck/datatype_utils.hpp
            COMMENT "Compiling ${variant}.hip -> ${arch}.hsaco"
        )
        list(APPEND HSACO_FILES ${hsaco_output})
    endforeach()
endforeach()

# --- Pack .hsaco files into a kpack archive ---
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.kpack
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pack.py ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${HSACO_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
    COMMENT "Creating kernels.kpack archive (${KERNEL_VARIANTS})"
)
add_custom_target(rocm_ck_vector_add_kpack_archive ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernels.kpack)

# --- Build the host-side loader executable ---
# Links against hip::host (host-only HIP runtime) and rocm_kpack (archive reader).
# Device code is loaded at runtime via hipModuleLoadData, not compiled into this binary.
add_executable(kpack_rocm_ck_vector_add main.cpp)
target_compile_features(kpack_rocm_ck_vector_add PRIVATE cxx_std_20)
target_include_directories(kpack_rocm_ck_vector_add PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kpack_rocm_ck_vector_add PRIVATE rocm_ck rocm_kpack hip::host)
add_dependencies(kpack_rocm_ck_vector_add rocm_ck_vector_add_kpack_archive)
