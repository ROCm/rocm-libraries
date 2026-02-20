# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEMP_CLIENTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../clients)

cmake_minimum_required(VERSION 3.16.8)
project(clients LANGUAGES C CXX)

add_subdirectory(common)

if(ROCBLAS_ENABLE_BENCHMARKS)
  add_subdirectory(bench)
  if(ROCBLAS_ENABLE_TENSILE)
    add_subdirectory(gemm-tune)
  endif()
endif()

if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(ROCBLAS_ENABLE_SAMPLES)
  add_subdirectory(samples)
endif()
