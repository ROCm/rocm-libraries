# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Kpack experiment — see examples/01_hello_world/ for a standalone build.
#
# The 01_hello_world example has its own CMakeLists.txt and can be built
# independently. Build it directly:
#
#   cd examples/01_hello_world
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#       -DCMAKE_PREFIX_PATH=/opt/rocm \
#       -DGPU_TARGETS="gfx90a;gfx942"
#   ninja -C build

cmake_minimum_required(VERSION 3.21)
project(kpack_experiment)

add_subdirectory(rocm_kpack)
add_subdirectory(include)
add_subdirectory(examples/01_hello_world)
add_subdirectory(examples/02_ck_tile_vector_add)
add_subdirectory(examples/03_rocm_ck_vector_add)
add_subdirectory(examples/04_gemm)
