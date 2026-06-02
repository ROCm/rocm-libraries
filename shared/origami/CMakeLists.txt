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
option(ORIGAMI_BUILD_TESTING "Build tests." OFF)
option(ORIGAMI_ENABLE_INSTALL "Configure origami installation" ON)
option(ORIGAMI_ENABLE_FETCH "Auto-fetch dependencies with FetchContent" ON)

find_package(hip REQUIRED)

if(ORIGAMI_BUILD_SHARED_LIBS OR (BUILD_SHARED_LIBS AND ORIGAMI_STANDALONE))
    set(ORIGAMI_LIBRARY_TYPE SHARED)
else()
    set(ORIGAMI_LIBRARY_TYPE STATIC)
endif()
add_library(origami ${ORIGAMI_LIBRARY_TYPE})

rocm_set_soversion(origami "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

add_library(roc::origami ALIAS origami)

# Control the exported symbol surface explicitly: hide everything by default and
# export only what is annotated with ORIGAMI_EXPORT (generated below). This replaces
# the previous "export everything" behaviour (default visibility +
# WINDOWS_EXPORT_ALL_SYMBOLS) so the public ABI is the public API and nothing more.
set_target_properties(origami PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_features(origami PUBLIC cxx_std_17)

# Generate origami/origami_export.h defining the ORIGAMI_EXPORT visibility macro
# (and ORIGAMI_STATIC for static builds). Public headers include it to annotate the
# exported API. The header is generated at configure time into the build tree.
include(GenerateExportHeader)
generate_export_header(origami
    BASE_NAME ORIGAMI
    EXPORT_MACRO_NAME ORIGAMI_EXPORT
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/origami/origami_export.h"
    STATIC_DEFINE ORIGAMI_STATIC
)
if(ORIGAMI_LIBRARY_TYPE STREQUAL "STATIC")
    # A static origami has no import library; consumers must see ORIGAMI_EXPORT as empty.
    target_compile_definitions(origami PUBLIC ORIGAMI_STATIC)
endif()

add_library(origami-headers INTERFACE)
add_library(roc::origami-headers ALIAS origami-headers)

target_compile_features(origami-headers INTERFACE cxx_std_17)

target_include_directories(
    origami-headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(
    origami-headers
    INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/origami/origami_export.h>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/gemm.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/hardware.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/heuristics.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/logger.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/math.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/origami.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/streamk.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/types.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast_simulator.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast.hpp>
)

target_link_libraries(origami PUBLIC roc::origami-headers)

target_sources(
    origami
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/gemm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/hardware.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/heuristics.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/logger.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/origami.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/streamk.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/types.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simulator/tensilelite/formocast_simulator.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simulator/tensilelite/formocast.cpp"
)

target_link_libraries(origami PUBLIC hip::host)

if(ORIGAMI_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
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

    # The export header is generated into the build tree, so it is not covered by the
    # include/ directory install above; install it next to the hand-authored headers.
    rocm_install(
        FILES "${PROJECT_BINARY_DIR}/include/origami/origami_export.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/origami"
        COMPONENT devel
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
