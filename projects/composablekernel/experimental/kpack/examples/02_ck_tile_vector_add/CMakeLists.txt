# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Standalone build for the CK Tile kpack vector-add example.
# Compiles ck_tile_add.hip per-architecture with CK Tile headers,
# packs into a .kpack archive, and builds the host-side loader.
#
# Usage:
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#       -DCMAKE_PREFIX_PATH=/opt/rocm \
#       -DGPU_TARGETS="gfx90a;gfx942"
#   ninja -C build

cmake_minimum_required(VERSION 3.21)
project(kpack_ck_tile_vector_add LANGUAGES CXX)

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

# --- Compile ck_tile_add.hip into per-architecture .hsaco code objects ---
# Uses --cuda-device-only to produce raw code objects without host stubs.
# The -I flag makes CK Tile headers available for device compilation.
set(HSACO_FILES "")
foreach(arch IN LISTS GPU_TARGETS)
    set(hsaco_output "${CMAKE_CURRENT_BINARY_DIR}/ck_tile_add_${arch}.hsaco")
    add_custom_command(
        OUTPUT ${hsaco_output}
        COMMAND ${CMAKE_HIP_COMPILER}
                --offload-arch=${arch}
                --cuda-device-only
                -x hip
                --rocm-path=${KPACK_ROCM_PATH}
                -I${CK_INCLUDE_DIR}
                -o ${hsaco_output}
                ${CMAKE_CURRENT_SOURCE_DIR}/ck_tile_add.hip
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ck_tile_add.hip
        COMMENT "Compiling ck_tile_add.hip -> ${arch}.hsaco"
    )
    list(APPEND HSACO_FILES ${hsaco_output})
endforeach()

# --- Pack .hsaco files into a kpack archive ---
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.kpack
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pack.py ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${HSACO_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/pack.py
    COMMENT "Creating kernels.kpack archive"
)
add_custom_target(ck_tile_vector_add_kpack_archive ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernels.kpack)

# --- Build the host-side loader executable ---
# Links against hip::host (host-only HIP runtime) and rocm_kpack (archive reader).
# Device code is loaded at runtime via hipModuleLoadData, not compiled into this binary.
add_executable(kpack_ck_tile_vector_add main.cpp)
target_link_libraries(kpack_ck_tile_vector_add PRIVATE rocm_ck rocm_kpack hip::host)
add_dependencies(kpack_ck_tile_vector_add ck_tile_vector_add_kpack_archive)
