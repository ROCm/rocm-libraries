# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(Origami VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ORIGAMI_STANDALONE ON)
else()
    set(ORIGAMI_STANDALONE OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set cpack license file here to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

include(dependencies)

rocm_setup_version(VERSION ${PROJECT_VERSION})

option(ORIGAMI_BUILD_SHARED_LIBS "Build shared libraries." ${ORIGAMI_STANDALONE})
option(ORIGAMI_ENABLE_PYTHON "Enable Python bindings." OFF)
option(ORIGAMI_BUILD_TESTING "Enable Python binding tests." OFF)
option(ORIGAMI_ENABLE_INSTALL "Configure origami installation" ON)
option(ORIGAMI_ENABLE_FETCH "Auto-fetch dependencies with FetchContent" ON)

# Check compiler
if(CMAKE_CXX_COMPILER)
    get_filename_component(COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)
    message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER} (${COMPILER_NAME})")
    
    # Warn if not using amdclang++ for ROCm builds
    if(NOT COMPILER_NAME MATCHES "amdclang\\+\\+" AND NOT COMPILER_NAME MATCHES "hipcc")
        message(WARNING 
            "Using ${COMPILER_NAME} instead of amdclang++. "
            "For ROCm builds, it is recommended to use: "
            "-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++")
    endif()
else()
    message(WARNING "C++ compiler not explicitly set")
endif()

# Check CMAKE_PREFIX_PATH for ROCm
if(CMAKE_PREFIX_PATH)
    message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
    list(FIND CMAKE_PREFIX_PATH "/opt/rocm" ROCM_IN_PREFIX)
    if(ROCM_IN_PREFIX EQUAL -1)
        # Check if any path contains rocm
        set(ROCM_FOUND_IN_PREFIX FALSE)
        foreach(PATH ${CMAKE_PREFIX_PATH})
            if(PATH MATCHES "rocm")
                set(ROCM_FOUND_IN_PREFIX TRUE)
                break()
            endif()
        endforeach()
        if(NOT ROCM_FOUND_IN_PREFIX)
            message(WARNING 
                "ROCm path not found in CMAKE_PREFIX_PATH. "
                "Consider adding: -DCMAKE_PREFIX_PATH=/opt/rocm")
        endif()
    endif()
else()
    message(WARNING "CMAKE_PREFIX_PATH not set. Consider: -DCMAKE_PREFIX_PATH=/opt/rocm")
endif()

# Check ROCM_PATH environment variable
if(DEFINED ENV{ROCM_PATH})
    message(STATUS "ROCM_PATH (environment): $ENV{ROCM_PATH}")
    if(NOT EXISTS "$ENV{ROCM_PATH}")
        message(WARNING "ROCM_PATH environment variable points to non-existent directory: $ENV{ROCM_PATH}")
    endif()
else()
    message(STATUS "ROCM_PATH (environment): not set")
endif()

find_package(hip REQUIRED)

# Validate HIP installation
if(hip_FOUND)
    message(STATUS "HIP found: ${hip_VERSION}")
    if(hip_VERSION)
        message(STATUS "HIP version: ${hip_VERSION}")
    endif()
    
    # Check for HIP headers
    find_path(HIP_HEADER_PATH "hip/hip_runtime.h" 
        PATHS ${CMAKE_PREFIX_PATH} 
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
    )
    if(HIP_HEADER_PATH)
        message(STATUS "HIP headers found: ${HIP_HEADER_PATH}")
    else()
        message(WARNING "HIP headers not found in expected locations")
    endif()
else()
    message(FATAL_ERROR 
        "HIP not found. Please ensure:\n"
        "  1. ROCm is installed\n"
        "  2. CMAKE_PREFIX_PATH includes ROCm path (e.g., /opt/rocm)\n"
        "  3. HIP is properly installed")
endif()

# Validate Python if Python bindings are enabled
if(ORIGAMI_ENABLE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    if(Python3_FOUND)
        message(STATUS "Python found: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")
        
        # Check for nanobind
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import nanobind; print(nanobind.__version__)"
            OUTPUT_VARIABLE NANOBIND_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE NANOBIND_CHECK
        )
        if(NANOBIND_CHECK EQUAL 0)
            message(STATUS "nanobind found: ${NANOBIND_VERSION}")
        else()
            message(WARNING 
                "nanobind not found in Python environment. "
                "Install with: pip install nanobind")
        endif()
    else()
        message(WARNING "Python3 not found but ORIGAMI_ENABLE_PYTHON=ON")
    endif()
endif()

if(ORIGAMI_BUILD_SHARED_LIBS OR (BUILD_SHARED_LIBS AND ORIGAMI_STANDALONE))
    add_library(origami SHARED)
else()
    add_library(origami STATIC)
endif()

rocm_set_soversion(origami "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

add_library(roc::origami ALIAS origami)

set_target_properties(origami PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(origami PUBLIC cxx_std_17)

add_library(origami-headers INTERFACE)
add_library(roc::origami-headers ALIAS origami-headers)

target_compile_features(origami-headers INTERFACE cxx_std_17)

target_include_directories(
    origami-headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(
    origami-headers
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/gemm.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/hardware.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/math.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/origami.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/streamk.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/types.hpp>
)

target_link_libraries(origami PUBLIC roc::origami-headers)

target_sources(
    origami
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/gemm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/hardware.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/origami.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/streamk.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/types.cpp"
)

target_link_libraries(origami PUBLIC hip::host)

if(ORIGAMI_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()

    add_subdirectory(tests)

    if(ORIGAMI_ENABLE_PYTHON)
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

        # Check if pytest is available, install if not
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pytest --version
            OUTPUT_QUIET
            ERROR_QUIET
            RESULT_VARIABLE pytest_available
        )

        if(pytest_available)
            # Try to install pytest
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -m pip install pytest pytest-cov
                OUTPUT_QUIET
                ERROR_QUIET
                RESULT_VARIABLE pytest_install_result
            )
        endif()

        # Run pytest tests
        add_test(
            NAME "Python: test_bindings"
            COMMAND ${Python3_EXECUTABLE} -m pytest tests/ -v
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        )

        set_tests_properties(
            "Python: test_bindings"
            PROPERTIES
                ENVIRONMENT
                "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/python:$ENV{PYTHONPATH}"
                DEPENDS _pyorigami
        )
    endif()
endif()

if(ORIGAMI_ENABLE_INSTALL OR ORIGAMI_STANDALONE)
    rocm_install(TARGETS origami origami-headers)

    rocm_export_targets(
        TARGETS
        roc::origami
        roc::origami-headers
        DEPENDS
        PACKAGE
        hip
        NAMESPACE
        roc::
    )

    if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
        rocm_install(TARGETS origami-tests COMPONENT tests)

    endif()

    rocm_install(
        DIRECTORY
        include/
        DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT
        devel
        FILES_MATCHING
        PATTERN
        "*.hpp"
        PATTERN
        "*.h"
    )

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/origami-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/origami-config.cmake" @ONLY
    )

    rocm_install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/origami-config.cmake" DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/origami" COMPONENT devel
    )

    set(BUILD_SHARED_LIBS ${ORIGAMI_BUILD_SHARED_LIBS})
    set(ORIGAMI_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        CACHE PATH "Path placed into ldconfig file"
    )

    rocm_create_package(
        NAME origami
        DESCRIPTION "Origami: Analytical Solution Selection for GEMM"
        MAINTAINER "Origami Maintainer <origami-maintainer@amd.com>"
        LDCONFIG
        LDCONFIG_DIR ${ORIGAMI_CONFIG_DIR}
    )
endif()
