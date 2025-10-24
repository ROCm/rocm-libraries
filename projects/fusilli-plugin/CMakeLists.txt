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
set(FUSILLI_HASH "5970f13835942213c7d4fd6171b936ee9fe9be73" CACHE STRING "Git hash for Fusilli")
set(HIP_DNN_HASH "4e0a0452cfcb8fdb86e9c40a6e43debab4d4ecbc" CACHE STRING "Git hash for hipDNN")

# Dependencies
set(HIP_PLATFORM "amd")
find_package(hip REQUIRED)
fusilli_plugin_dependency(GTest GTEST_VERSION 1.16.0)
fusilli_plugin_dependency(IREERuntime)
# Use local path if user set -DHIPDNN_LOCAL_PATH otherwise hash.
if(HIPDNN_LOCAL_PATH)
    fusilli_plugin_dependency(hipdnn_frontend LOCAL_PATH ${HIPDNN_LOCAL_PATH})
else()
    fusilli_plugin_dependency(hipdnn_frontend HIP_DNN_HASH ${HIP_DNN_HASH})
endif()
# Use local path if user set -DFUSILLI_LOCAL_PATH otherwise hash.
if(FUSILLI_LOCAL_PATH)
    fusilli_plugin_dependency(Fusilli LOCAL_PATH ${FUSILLI_LOCAL_PATH})
else()
    fusilli_plugin_dependency(Fusilli FUSILLI_HASH ${FUSILLI_HASH})
endif()

# Includes
include_directories(include)

# Plugin definition
add_library(${FUSILLI_PLUGIN_NAME} SHARED
    src/fusilli_plugin.cpp
)
target_compile_options(${FUSILLI_PLUGIN_NAME} PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
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
