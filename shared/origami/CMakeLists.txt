# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.24.4)

project(Origami VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ORIGAMI_STANDALONE ON)
else()
    set(ORIGAMI_STANDALONE OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(dependencies)

rocm_setup_version(VERSION ${PROJECT_VERSION})

option(ORIGAMI_BUILD_SHARED_LIBS "Build shared libraries." ${ORIGAMI_STANDALONE})
option(ORIGAMI_ENABLE_PYTHON "Enable Python bindings." OFF)
option(ORIGAMI_BUILD_TESTING "Enable Python binding tests." OFF)
option(ORIGAMI_ENABLE_INSTALL "Configure origami installation" ON)

find_package(hip REQUIRED)

if(ORIGAMI_BUILD_SHARED_LIBS OR (BUILD_SHARED_LIBS AND ORIGAMI_STANDALONE))
    add_library(origami SHARED) 
else()
    add_library(origami STATIC)
endif()

rocm_set_soversion(origami "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

add_library(roc::origami ALIAS origami)

set_target_properties(origami PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(origami PUBLIC cxx_std_17)

add_library(origami-headers INTERFACE)
add_library(roc::origami-headers ALIAS origami-headers)

target_include_directories(origami-headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(origami-headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/gemm.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/hardware.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/utils.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/streamk.hpp>
)

target_link_libraries(origami PUBLIC roc::origami-headers)

target_sources(origami
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/gemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/utils.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/streamk.cpp"
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

        add_test(
            NAME origami_python_test
            COMMAND "${Python3_EXECUTABLE}" "origami_test.py"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python"
        )
        
        add_test(
            NAME origami_python_grid_test  
            COMMAND "${Python3_EXECUTABLE}" "origami_grid_test.py"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python"
        )

        set_tests_properties(origami_python_test origami_python_grid_test
            PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/python:$ENV{PYTHONPATH}"
        )

        set_tests_properties(origami_python_test origami_python_grid_test
            PROPERTIES
                DEPENDS origami_python
        )
    endif()
endif()

if(ORIGAMI_ENABLE_INSTALL OR ORIGAMI_STANDALONE)
    rocm_install(TARGETS origami origami-headers)

    rocm_export_targets(
        TARGETS roc::origami roc::origami-headers
        DEPENDS PACKAGE hip
        NAMESPACE roc::
    )

    if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
        rocm_install(TARGETS origami-tests
            COMPONENT tests
        )
        
        rocm_install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/origami_gtest.yaml"
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT tests
        )
    endif()

    rocm_install(
        DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
    )

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/origami-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/origami-config.cmake"
        @ONLY
    )

    rocm_install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/origami-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/origami"
        COMPONENT devel
    )

    set(BUILD_SHARED_LIBS ${ORIGAMI_BUILD_SHARED_LIBS})
    set(ORIGAMI_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

    rocm_create_package(
        NAME origami
        DESCRIPTION "Origami C++ library for analytical GEMM computations"
        MAINTAINER "Origami Maintainer <origami-maintainer@amd.com>"
        LDCONFIG
        LDCONFIG_DIR ${ORIGAMI_CONFIG_DIR}
    )
endif()
