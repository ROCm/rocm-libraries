# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

# hip::device also needs a compilation target when no GPU is available.
set(GPU_TARGETS gfx1100 CACHE STRING "HIP package compilation targets")

project(rocwmma_openmp_consumer LANGUAGES CXX)

find_package(hip REQUIRED)
# The consumer supplies OpenMP::OpenMP_CXX through FindOpenMP. Any other
# exported dependency, such as OpenMP::omp, must be resolved by rocWMMA itself.
find_package(OpenMP MODULE REQUIRED COMPONENTS CXX)
find_package(rocwmma CONFIG REQUIRED)

add_executable(rocwmma_openmp_consumer test_openmp_consumer.cpp)
target_link_libraries(rocwmma_openmp_consumer PRIVATE roc::rocwmma)

enable_testing()
add_test(NAME rocwmma_openmp_consumer COMMAND rocwmma_openmp_consumer)
