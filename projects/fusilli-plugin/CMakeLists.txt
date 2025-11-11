# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.28)

project(fusilli-plugin
  VERSION 0.1.0
  DESCRIPTION "Fusilli-Plugin: A Fusilli/IREE powered hipDNN plugin for graph JIT compilation."
  LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Local Includes
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)
include(FusilliPluginDependencyUtils)

# Global constants
set(FUSILLI_PLUGIN_NAME fusilli_plugin)
set(FUSILLI_PLUGIN_ENGINE_ID 1001)

# Dependencies
set(HIP_PLATFORM "amd")
find_package(hip REQUIRED)
find_package(IREERuntime CONFIG REQUIRED)
find_package(hipdnn_sdk CONFIG REQUIRED)
find_package(Fusilli CONFIG REQUIRED)

# GTest is still fetched since TheRock doesn't provide it
fusilli_plugin_dependency(GTest GTEST_VERSION 1.16.0)

# Define hipDNN variables that were previously provided by building hipDNN
# These are needed since hipdnn_sdk config doesn't export them
if(NOT DEFINED HIPDNN_PLUGIN_ENGINE_SUBDIR)
    set(HIPDNN_PLUGIN_ENGINE_SUBDIR "hipdnn_plugins/engines")
endif()
if(NOT DEFINED HIPDNN_BUILD_PLUGIN_ENGINE_DIR)
    set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}")
endif()

# Includes
include_directories(include)

# Plugin definition
add_library(${FUSILLI_PLUGIN_NAME} SHARED
    src/fusilli_plugin.cpp
)
target_link_libraries(${FUSILLI_PLUGIN_NAME} PRIVATE hipdnn_sdk hip::host fusilli::fusilli)
target_compile_definitions(${FUSILLI_PLUGIN_NAME} PRIVATE
  FUSILLI_PLUGIN_NAME="${FUSILLI_PLUGIN_NAME}"
  FUSILLI_PLUGIN_ENGINE_ID=${FUSILLI_PLUGIN_ENGINE_ID}
)
set_target_properties(${FUSILLI_PLUGIN_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
)

# Tests
enable_testing()
add_subdirectory(test)
