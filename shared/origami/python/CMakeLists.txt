# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(origami-python VERSION 0.1.0 LANGUAGES CXX)

# Find Python
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Option to control origami library source
option(ORIGAMI_BUILD_FROM_SOURCE "Build origami from source instead of finding installed version" ON)

# Get ROCM_PATH - fail if not found
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
        message(STATUS "Using ROCM_PATH from environment: ${ROCM_PATH}")
    else()
        set(ROCM_PATH "/opt/rocm")
        message(STATUS "ROCM_PATH not set, using default: ${ROCM_PATH}")
    endif()
endif()

# Verify ROCM_PATH exists
if(NOT EXISTS "${ROCM_PATH}")
    message(WARNING "ROCM_PATH '${ROCM_PATH}' does not exist. Build may fail if HIP is not found elsewhere.")
endif()

# Determine how to obtain origami library - no fallback logic
# Check if the target is already available (when added as subdirectory from parent)
if(TARGET roc::origami)
    message(STATUS "Using origami target from parent CMakeLists.txt")
    set(ORIGAMI_TARGET roc::origami)
elseif(ORIGAMI_BUILD_FROM_SOURCE)
    message(STATUS "Building origami from source (ORIGAMI_BUILD_FROM_SOURCE=ON)")
    
    # Get parent directory (shared/origami) - go up one level from python/
    get_filename_component(ORIGAMI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    
    # Verify the CMakeLists.txt exists
    if(NOT EXISTS "${ORIGAMI_ROOT}/CMakeLists.txt")
        message(FATAL_ERROR 
            "ORIGAMI_BUILD_FROM_SOURCE is ON but could not find origami CMakeLists.txt at:\n"
            "  ${ORIGAMI_ROOT}/CMakeLists.txt\n"
            "Please ensure you are building from the correct directory structure.\n"
            "Expected structure: shared/origami/python/CMakeLists.txt\n"
            "Or set ORIGAMI_BUILD_FROM_SOURCE=OFF to use an installed origami package.")
    endif()
    
    # Verify source directory structure
    if(NOT EXISTS "${ORIGAMI_ROOT}/src/origami")
        message(FATAL_ERROR 
            "ORIGAMI_BUILD_FROM_SOURCE is ON but could not find origami source directory at:\n"
            "  ${ORIGAMI_ROOT}/src/origami\n"
            "Please ensure the origami source code is present.\n"
            "Or set ORIGAMI_BUILD_FROM_SOURCE=OFF to use an installed origami package.")
    endif()
    
    # Add origami as a subdirectory
    add_subdirectory("${ORIGAMI_ROOT}" origami_build EXCLUDE_FROM_ALL)
    
    # Verify target was created
    if(NOT TARGET roc::origami)
        message(FATAL_ERROR 
            "ORIGAMI_BUILD_FROM_SOURCE is ON but failed to create roc::origami target.\n"
            "Check the parent CMakeLists.txt at ${ORIGAMI_ROOT}/CMakeLists.txt for errors.")
    endif()
    
    set(ORIGAMI_TARGET roc::origami)
    message(STATUS "Successfully configured origami build from source")
else()
    message(STATUS "Looking for installed origami package (ORIGAMI_BUILD_FROM_SOURCE=OFF)")
    find_package(origami REQUIRED)

    if(NOT TARGET roc::origami)
        message(FATAL_ERROR
            "ORIGAMI_BUILD_FROM_SOURCE is OFF but installed origami package does not provide roc::origami target.\n"
            "Please install origami or set ORIGAMI_BUILD_FROM_SOURCE=ON to build from source.\n"
            "To install origami, build and install the parent project first.")
    endif()

    set(ORIGAMI_TARGET roc::origami)
    message(STATUS "Using installed origami package")
endif()

# Find HIP - add ROCM_PATH to CMAKE_PREFIX_PATH if not already there
if(NOT ROCM_PATH IN_LIST CMAKE_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    message(STATUS "Added ROCM_PATH to CMAKE_PREFIX_PATH: ${ROCM_PATH}")
endif()

# Find HIP
find_package(hip REQUIRED)

# Fetch nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v2.0.0
)
FetchContent_MakeAvailable(nanobind)

find_package(nanobind CONFIG REQUIRED)

# Set Python package directory
set(PY_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/origami")

# Add nanobind module
nanobind_add_module(_pyorigami "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/bindings.cpp")

set_target_properties(_pyorigami PROPERTIES OUTPUT_NAME "origami")

# Link against origami library and HIP
target_link_libraries(_pyorigami PRIVATE ${ORIGAMI_TARGET} hip::host)

# Set compile options
target_compile_options(_pyorigami PRIVATE
    -D__HIP_PLATFORM_AMD__
    -fPIC
    -std=c++17
    -O3
    -Wall
    -DNB_SHARED
    -fvisibility=hidden
)

# Set link directories for ROCm libraries
target_link_directories(_pyorigami PRIVATE "${ROCM_PATH}/lib")

# Install the compiled module to the package directory
install(TARGETS _pyorigami LIBRARY DESTINATION "${PY_PACKAGE_DIR}")

# Install Python package files
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/__init__.py"
    DESTINATION "${PY_PACKAGE_DIR}"
)

# Copy the built module to the source directory for testing (when not installing)
# This allows the relative import in __init__.py to work during development/testing
add_custom_command(TARGET _pyorigami POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:_pyorigami>
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/"
    COMMENT "Copying Python module to source directory for testing"
)
