# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(Origami VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(ORIGAMI_BUILD_FROM_SOURCE "Build liborigami from this tree instead of finding an installed one." ON)
option(ORIGAMI_BUILD_SHARED_LIBS "Build shared libraries." ON)
option(ORIGAMI_ENABLE_PYTHON "Enable Python bindings." OFF)
option(ORIGAMI_PYTHON_ENABLE_WARNINGS "Enable compiler warnings for Python extension build" OFF)
option(ORIGAMI_BUILD_TESTING "Build tests." OFF)
option(ORIGAMI_ENABLE_FETCH "Auto-fetch dependencies with FetchContent" ON)

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
endif()

if(ORIGAMI_BUILD_FROM_SOURCE)
    if(ORIGAMI_ENABLE_PYTHON)
        set(ORIGAMI_BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries." FORCE)
    endif()

    # Set cpack license file here to avoid rocm-cmake warning
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
    include(dependencies)
    rocm_setup_version(VERSION ${PROJECT_VERSION})
    add_subdirectory(origami-cpp)
else()
    find_package(origami REQUIRED)

    get_target_property(_origami_library_type roc::origami TYPE)
    if(_origami_library_type STREQUAL "STATIC_LIBRARY")
        message(
            FATAL_ERROR
                "The installed roc::origami target is static. Install a shared liborigami or configure with ORIGAMI_BUILD_FROM_SOURCE=ON."
        )
    endif()
endif()

if(ORIGAMI_ENABLE_PYTHON)
    add_subdirectory(python)
endif()
