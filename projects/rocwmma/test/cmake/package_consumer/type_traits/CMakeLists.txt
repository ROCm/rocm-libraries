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
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Leave compatibility defines unset to exercise an unmodified package consumer.
add_executable(rocwmma_type_traits_consumer test_type_traits.hip)
set_target_properties(rocwmma_type_traits_consumer PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_link_libraries(rocwmma_type_traits_consumer PRIVATE roc::rocwmma)

get_target_property(_include_dirs roc::rocwmma INTERFACE_INCLUDE_DIRECTORIES)
foreach(_dir IN LISTS _include_dirs)
  if(EXISTS "${_dir}/rocwmma/rocwmma.hpp")
    set(_rocwmma_include "${_dir}")
    break()
  endif()
endforeach()
if(NOT _rocwmma_include)
  message(FATAL_ERROR "Cannot locate the installed rocWMMA public headers")
endif()
list(GET CMAKE_HIP_ARCHITECTURES 0 _test_arch)

enable_testing()
add_test(NAME rocwmma_type_traits_host COMMAND rocwmma_type_traits_consumer)
add_test(NAME rocwmma_type_traits_gemm COMMAND rocwmma_type_traits_consumer --gemm)
set_tests_properties(rocwmma_type_traits_gemm PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME rocwmma_type_traits_compiler
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_type_traits.py"
    --compiler "${CMAKE_HIP_COMPILER}" --arch "${_test_arch}"
    --include "${_rocwmma_include}")
