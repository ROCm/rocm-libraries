# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(hipblaslt_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# scikit-build-core pre-installs nanobind and injects its CMake path.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind REQUIRED CONFIG)

# Locate HIP and hipBLASLt. ROCM_PATH may be passed by tasks.py; fall back to /opt/rocm.
if(ROCM_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
endif()
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)

nanobind_add_module(_core NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/module.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/enums.cpp"
)
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(_core PRIVATE hip::host roc::hipblaslt)

install(TARGETS _core DESTINATION hipblaslt)
