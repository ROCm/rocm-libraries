# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.24)

project(fusilli-plugin
  VERSION 0.1.0
  DESCRIPTION "Fusilli-Plugin: A Fusilli/IREE powered hipDNN plugin for graph JIT compilation."
  LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# CMake includes
include(GNUInstallDirs)
include(FetchContent)

# IREE Runtime Dependency - build from source, same pattern as fusilli/CMakeLists.txt
if(NOT IREE_SOURCE_DIR)
  message(FATAL_ERROR "IREE_SOURCE_DIR must be provided. Set -DIREE_SOURCE_DIR=<path> to the IREE source directory.")
endif()
message(STATUS "Using existing IREE sources: ${IREE_SOURCE_DIR}")
# Set IREE build flags
set(IREE_VISIBILITY_HIDDEN OFF)
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_HIP ON)
set(IREE_HIP_TEST_TARGET_CHIP "" CACHE STRING "")
# Build IREERuntime as part of plugin
FetchContent_Declare(
  IREERuntime
  SOURCE_DIR ${IREE_SOURCE_DIR}
  # Fusilli's config file requires IREE runtime via
  # find_dependency(IREERuntime), which triggers find_package(IREERuntime). The
  # dependency can only be fulfilled via source build: FetchContent_Declare or
  # add_subdirectory. Of those options, only FetchContent_Declare provides a way
  # of intercepting the find_package call and redirecting to the source build.
  OVERRIDE_FIND_PACKAGE
  SYSTEM
)
FetchContent_MakeAvailable(IREERuntime)

# Other dependencies
find_package(hip CONFIG REQUIRED)
find_package(hipdnn_data_sdk CONFIG REQUIRED)
find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_test_sdk CONFIG REQUIRED)
find_package(Fusilli CONFIG REQUIRED)
# Fetch GTest if system install isn't available
find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  message(STATUS "GTest not found on system: Fetching from GitHub")
  FetchContent_Declare(
    googletest
    QUIET
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.16.0
  )
  FetchContent_MakeAvailable(googletest)
endif()


# Global constants
set(FUSILLI_PLUGIN_TARGET fusilli_plugin)
# Relative path from build/install prefix to plugin location.
# HIPDNN_PLUGIN_ENGINE_SUBDIR is provided by hipdnn_sdk.
set(FUSILLI_PLUGIN_RELATIVE_PATH "${CMAKE_INSTALL_LIBDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}")

# Useful compile warnings
list(PREPEND FUSILLI_PLUGIN_WARNING_COMPILE_OPTIONS
    -Werror                                  # Treat all warnings as errors
    -Wall                                    # Enable most common warnings
    -Wextra                                  # Enable additional warnings not covered by -Wall
    -Wpedantic                               # Enforce strict ISO C++ compliance
    -Wshadow                                 # Warn about variable shadowing
    -Wnon-virtual-dtor                       # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast                         # Warn about C-style casts
    -Wcast-align                             # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual                     # Warn if a base class function is hidden by a derived class function with the same name
    -Wconversion                             # Warn about implicit type conversions that may alter a value
    -Wsign-conversion                        # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference                       # Warn about dereferencing null pointers
    -Wdouble-promotion                       # Warn when a float is implicitly promoted to a double
    -Wformat=2                               # Enable stricter format string checks
    -Winit-self                              # Warn about variables initialized with itself
    -Wunreachable-code                       # Warn about unreachable code
    -Wno-error=unused-command-line-argument  # Disable error for unused command line arguments
    -Wno-gnu-statement-expression            # Disable gnu error for statement expression, this will need to change on Windows.
    -Wswitch-default                         # Warn if a switch statement does not have a default case
)

# Includes
include_directories(include)

# Plugin definition
add_library(${FUSILLI_PLUGIN_TARGET} SHARED
  src/fusilli_plugin.cpp
)
target_compile_options(${FUSILLI_PLUGIN_TARGET} PRIVATE ${FUSILLI_PLUGIN_WARNING_COMPILE_OPTIONS})
target_link_libraries(${FUSILLI_PLUGIN_TARGET} PRIVATE hipdnn_plugin_sdk hipdnn_data_sdk hip::host fusilli::fusilli)
set_target_properties(${FUSILLI_PLUGIN_TARGET} PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${FUSILLI_PLUGIN_RELATIVE_PATH}"
)
# Version script hides all symbol definitions not staring with "hipdnn"
target_link_options(
  ${FUSILLI_PLUGIN_TARGET} PRIVATE "LINKER:--version-script=${CMAKE_SOURCE_DIR}/exports.map"
)

# Installation
install(TARGETS ${FUSILLI_PLUGIN_TARGET}
  LIBRARY DESTINATION "${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

# Tests
enable_testing()
add_subdirectory(test)
