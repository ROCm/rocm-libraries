# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

project(mosaic VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MOSAIC_BUILD_TESTS "Build mosaic gtest suite" ON)
option(MOSAIC_BUILD_EXAMPLES "Build host-side C++ examples" ON)
option(MOSAIC_BUILD_HIP_EXAMPLES "Build HIP/GPU examples (needs a HIP toolchain)" OFF)

# include/mosaic/utils.hpp and the opt-in HIP examples include
# <hip/hip_runtime.h>. hip::host provides that include path.
find_package(hip REQUIRED)

add_library(mosaic INTERFACE)
add_library(roc::mosaic ALIAS mosaic)
target_include_directories(mosaic INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(mosaic INTERFACE cxx_std_20)
target_link_libraries(mosaic INTERFACE hip::host)

if(MOSAIC_BUILD_HIP_EXAMPLES)
    enable_language(HIP)
endif()

if(MOSAIC_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()

if(MOSAIC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
