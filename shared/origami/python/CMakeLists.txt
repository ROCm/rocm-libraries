################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.15)

project(origami-python VERSION 0.1.0 LANGUAGES CXX)

# Find Python
find_package(Python 3.7 COMPONENTS Interpreter Development.Module REQUIRED)

# Get ROCM_PATH from environment or use default
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
else()
    set(ROCM_PATH "/opt/rocm")
endif()

# Find or build origami library
# First try to find it as an installed package
find_package(origami QUIET)

if(NOT origami_FOUND)
    # If not found, build it from parent directory
    message(STATUS "origami not found, building from source")
    
    # Get parent directory (shared/origami) - go up one level from python/
    get_filename_component(ORIGAMI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    
    # Verify the CMakeLists.txt exists
    if(EXISTS "${ORIGAMI_ROOT}/CMakeLists.txt")
        # Add origami as a subdirectory
        add_subdirectory("${ORIGAMI_ROOT}" origami_build EXCLUDE_FROM_ALL)
        
        # Set origami target
        set(ORIGAMI_TARGET roc::origami)
    else()
        message(FATAL_ERROR "Could not find origami CMakeLists.txt at ${ORIGAMI_ROOT}/CMakeLists.txt")
    endif()
else()
    message(STATUS "Found origami package")
    set(ORIGAMI_TARGET roc::origami)
endif()

# Find HIP - add ROCM_PATH to CMAKE_PREFIX_PATH if not already there
if(NOT ROCM_PATH IN_LIST CMAKE_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
endif()
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
nanobind_add_module(_core "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/bindings.cpp")

set_target_properties(_core PROPERTIES OUTPUT_NAME "origami")

# Link against origami library and HIP
target_link_libraries(_core PRIVATE ${ORIGAMI_TARGET} hip::host)

# Set compile options
target_compile_options(_core PRIVATE
    -D__HIP_PLATFORM_AMD__
    -fPIC
    -std=c++17
    -O3
    -Wall
    -DNB_SHARED
    -fvisibility=hidden
)

# Set link directories for ROCm libraries
target_link_directories(_core PRIVATE "${ROCM_PATH}/lib")

# Install the compiled module to the package directory
install(TARGETS _core LIBRARY DESTINATION "${PY_PACKAGE_DIR}")

# Install Python package files
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/__init__.py"
    DESTINATION "${PY_PACKAGE_DIR}"
)
