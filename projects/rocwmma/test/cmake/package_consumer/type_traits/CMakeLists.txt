# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "HIP compilation targets")
endif()
if(NOT DEFINED GPU_TARGETS)
  set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "HIP package targets")
endif()
project(rocwmma_type_traits_consumer LANGUAGES CXX HIP)

find_package(rocwmma REQUIRED CONFIG)

# This standalone consumer intentionally does not inherit the test suite's
# explicit opt-out: it must exercise the behavior shipped to an unmodified user.
add_executable(rocwmma_type_traits_consumer test_type_traits.hip)
set_target_properties(rocwmma_type_traits_consumer PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_link_libraries(rocwmma_type_traits_consumer PRIVATE roc::rocwmma)

enable_testing()
add_test(NAME rocwmma_type_traits_host COMMAND rocwmma_type_traits_consumer)
add_test(NAME rocwmma_type_traits_gemm COMMAND rocwmma_type_traits_consumer --gemm)
set_tests_properties(rocwmma_type_traits_gemm PROPERTIES SKIP_RETURN_CODE 77)
