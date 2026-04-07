# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)

project(hipdnn_example_provider VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Output directories
# =============================================================================
# Collect all executables into build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =============================================================================
# Options
# =============================================================================
option(HIPDNN_EXAMPLE_PROVIDER_BUILD_UNIT_TESTS "Build unit tests" ON)
option(HIPDNN_EXAMPLE_PROVIDER_BUILD_SAMPLE "Build sample application (requires hipdnn_frontend)" ON)

# =============================================================================
# Compiler flags
# =============================================================================
# Use an interface library to apply warning flags only to project targets (not to
# dependencies like GoogleTest).
add_library(example_provider_compile_options INTERFACE)
target_compile_options(example_provider_compile_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Werror>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wextra>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<CXX_COMPILER_ID:MSVC>:/WX>
)

# HIP platform definition (enables HIP types without requiring a GPU compiler)
add_compile_definitions(__HIP_PLATFORM_AMD__)

# =============================================================================
# ROCM_PATH
# =============================================================================
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

# =============================================================================
# Dependencies
#
# The plugin SDK and data SDK are header-only libraries installed by hipDNN.
# find_package locates them via their CMake config files and transitively
# brings in the HIP, flatbuffers, and nlohmann_json include paths.
# =============================================================================
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)

find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
# hipdnn_data_sdk is found transitively through hipdnn_plugin_sdk
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)

# =============================================================================
# Subdirectories
# =============================================================================
# Kernel embedding must come first so KERNEL_GENERATED_SOURCES is set
add_subdirectory(kernels)
add_subdirectory(src)

# Install the plugin shared library
if(NOT DEFINED HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR)
    set(HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR "lib/hipdnn_plugins/engines")
endif()
install(TARGETS example_provider_plugin LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR})

if(HIPDNN_EXAMPLE_PROVIDER_BUILD_UNIT_TESTS OR HIPDNN_EXAMPLE_PROVIDER_BUILD_SAMPLE)
    enable_testing()
endif()

if(HIPDNN_EXAMPLE_PROVIDER_BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()

if(HIPDNN_EXAMPLE_PROVIDER_BUILD_SAMPLE)
    add_subdirectory(sample)
endif()
