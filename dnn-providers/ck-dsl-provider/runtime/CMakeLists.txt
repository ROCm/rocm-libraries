# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# ck_dsl_runtime: a minimal-dependency (HIP + libamd_comgr only), Python-free
# C++ runtime that loads, comgr-compiles, and launches ck_dsl-generated kernels.
# Header-only library + a standalone GPU test.
cmake_minimum_required(VERSION 3.21)
project(ck_dsl_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CK_DSL_RUNTIME_BUILD_TESTS "Build ck_dsl_runtime standalone test" ON)

find_package(hip REQUIRED)

# libamd_comgr (+ its header). Allow override via -DAMD_COMGR_ROOT.
find_library(AMD_COMGR_LIB amd_comgr
    HINTS ${AMD_COMGR_ROOT}/lib /opt/rocm/lib /opt/rocm-7.2.0/lib)
find_path(AMD_COMGR_INCLUDE amd_comgr/amd_comgr.h
    HINTS ${AMD_COMGR_ROOT}/include /opt/rocm/include /opt/rocm-7.2.0/include)
if(NOT AMD_COMGR_LIB OR NOT AMD_COMGR_INCLUDE)
    message(FATAL_ERROR "libamd_comgr not found (set AMD_COMGR_ROOT)")
endif()

add_library(ck_dsl_runtime INTERFACE)
target_include_directories(ck_dsl_runtime INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${AMD_COMGR_INCLUDE})
target_link_libraries(ck_dsl_runtime INTERFACE hip::host ${AMD_COMGR_LIB})

if(CK_DSL_RUNTIME_BUILD_TESTS)
    add_executable(ck_dsl_runtime_test tests/test_runtime.cpp)
    target_link_libraries(ck_dsl_runtime_test PRIVATE ck_dsl_runtime)
endif()
