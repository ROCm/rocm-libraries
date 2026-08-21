# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(Origami VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set cpack license file here to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

include(dependencies)

rocm_setup_version(VERSION ${PROJECT_VERSION})

option(ORIGAMI_BUILD_SHARED_LIBS "Build shared libraries." ON)
option(ORIGAMI_ENABLE_PYTHON "Enable Python bindings." OFF)
option(ORIGAMI_BUILD_TESTING "Build tests." OFF)
option(ORIGAMI_ENABLE_FETCH "Auto-fetch dependencies with FetchContent" ON)

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
endif()

add_subdirectory(origami-cpp)

if(ORIGAMI_ENABLE_PYTHON)
    add_subdirectory(python)
endif()
