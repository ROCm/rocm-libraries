# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.24.4)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(Origami VERSION 1.0.0 LANGUAGES CXX)

    set(ORIGAMI_STANDALONE ON)
else()
    set(ORIGAMI_STANDALONE OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(CMakeDependentOption)

include(dependencies)

rocm_setup_version(VERSION ${PROJECT_VERSION})

option(ORIGAMI_BUILD_SHARED_LIBS "Build shared libraries." ${ORIGAMI_STANDALONE})
option(ORIGAMI_ENABLE_PYTHON "Enable Python bindings." OFF)
option(ORIGAMI_ENABLE_TESTS "Enable Python binding tests." OFF)

find_package(hip REQUIRED)

if(ORIGAMI_BUILD_SHARED_LIBS OR (BUILD_SHARED_LIBS AND ORIGAMI_STANDALONE))
    add_library(origami SHARED) 
else()
    add_library(origami STATIC)
endif()

rocm_set_soversion(origami ${PROJECT_VERSION_MAJOR})

add_library(origami::origami ALIAS origami)

target_compile_features(origami PUBLIC cxx_std_17)

target_include_directories(origami
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(origami
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/AnalyticalGemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/Hardware.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/Utils.cpp"
)

target_link_libraries(origami PUBLIC hip::host)

if(ORIGAMI_ENABLE_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 QUIET)

    pybind11_add_module(origami_python 
        "${CMAKE_CURRENT_SOURCE_DIR}/python/origami_module.cpp"
    )
    
    set_target_properties(origami_python PROPERTIES
        OUTPUT_NAME "origami"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    
    target_link_libraries(origami_python PRIVATE origami::origami)

endif()

if(ORIGAMI_ENABLE_TESTS)
    enable_testing()
    
    if(ORIGAMI_ENABLE_PYTHON)
        add_test(
            NAME origami_python_basic_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/origami_test.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        )
        
        add_test(
            NAME origami_python_grid_test  
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/origami_grid_test.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        )

        set_tests_properties(origami_python_basic_test origami_python_grid_test
            PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/python:$ENV{PYTHONPATH}"
        )

        set_tests_properties(origami_python_basic_test origami_python_grid_test
            PROPERTIES
                DEPENDS origami_python
        )
    endif()
endif()


if(ORIGAMI_STANDALONE)
    rocm_install(TARGETS origami)

    rocm_export_targets(
        TARGETS origami::origami
        DEPENDS PACKAGE hip
        NAMESPACE origami::
    )

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

