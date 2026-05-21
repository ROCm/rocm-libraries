# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

# =============================================================================
# Versioning
# =============================================================================
include(${CMAKE_CURRENT_LIST_DIR}/cmake/VersionUtils.cmake)
example_provider_setup_version(example_provider)

project(hipdnn-example-provider VERSION ${EXAMPLE_PROVIDER_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Sanitizer support for Address Sanitizer when enabled in the enviornment.
option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)
if(BUILD_ADDRESS_SANITIZER AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
        OUTPUT_VARIABLE CLANG_RESOURCE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    link_directories(${CLANG_RESOURCE_DIR}/lib/linux)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fno-omit-frame-pointer -shared-libasan)
    add_compile_definitions(ADDRESS_SANITIZER)
endif()

# =============================================================================
# Output directories
# =============================================================================
# Collect all executables into build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =============================================================================
# Options
# =============================================================================
option(EXAMPLEPROVIDER_BUILD_UNIT_TESTS "Build unit tests" ON)
option(EXAMPLEPROVIDER_BUILD_SAMPLE "Build sample application (requires hipdnn_frontend)" ON)

# To enable clang-tidy checks during compilation, place a .clang-tidy
# configuration file in this directory (or a parent directory) with your
# desired checks, ensure clang-tidy is in your system PATH (or CMAKE_PREFIX_PATH),
# then configure with -DEXAMPLEPROVIDER_ENABLE_CLANG_TIDY=ON.
option(EXAMPLEPROVIDER_ENABLE_CLANG_TIDY "Enable clang-tidy checks during compilation" OFF)
if(EXAMPLEPROVIDER_ENABLE_CLANG_TIDY AND NOT DEFINED CMAKE_CXX_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy-20 clang-tidy)
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    else()
        message(FATAL_ERROR "EXAMPLEPROVIDER_ENABLE_CLANG_TIDY is ON but clang-tidy was not found")
    endif()
endif()

if(CMAKE_CXX_CLANG_TIDY)
    message(STATUS "clang-tidy enabled: ${CMAKE_CXX_CLANG_TIDY}")
else()
    message(STATUS "clang-tidy disabled.")
endif()

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
# Dependencies
#
# The plugin SDK and data SDK are header-only libraries installed by hipDNN.
# find_package locates them via their CMake config files and transitively
# brings in the HIP, flatbuffers, and nlohmann_json include paths.
# =============================================================================
find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
# hipdnn_flatbuffers_sdk is found transitively through hipdnn_plugin_sdk
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)

# =============================================================================
# Subdirectories
# =============================================================================
# Kernel embedding must come first so KERNEL_GENERATED_SOURCES is set
add_subdirectory(kernels)
add_subdirectory(src)

# Generate version header from version.h.in
example_provider_generate_version_header(example_provider example_provider_impl)

# Install the plugin shared library
# HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR is the standard ROCm distribution subfolder
# where hipDNN plugins are located (e.g. lib/hipdnn_plugins/engines/).
if(NOT DEFINED HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR)
    message(FATAL_ERROR "HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR was not set by hipdnn_data_sdk package import.")
endif()
install(TARGETS example_provider_plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR})
if(EXAMPLEPROVIDER_BUILD_UNIT_TESTS OR EXAMPLEPROVIDER_BUILD_SAMPLE)
    enable_testing()
endif()

if(EXAMPLEPROVIDER_BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()

if(EXAMPLEPROVIDER_BUILD_SAMPLE)
    add_subdirectory(sample)
endif()
