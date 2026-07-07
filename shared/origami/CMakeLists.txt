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
    add_library(origami SHARED)
else()
    add_library(origami STATIC)
endif()

rocm_set_soversion(origami "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

add_library(roc::origami ALIAS origami)

set_target_properties(origami PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
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
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/architecture.hpp>
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

# ── origami::comm — analytical communication cost model ──────────────────────
# Header-only INTERFACE library. Deliberately does NOT link hip::host: the comm
# model is pure C++17 and usable without a ROCm toolchain, so consumers that only
# need collective cost prediction can depend on it independently of roc::origami.
add_library(origami-comm-headers INTERFACE)
add_library(roc::origami-comm-headers ALIAS origami-comm-headers)

target_compile_features(origami-comm-headers INTERFACE cxx_std_17)

target_include_directories(
    origami-comm-headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(
    origami-comm-headers
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/architecture.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/origami_comm.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/types.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/hardware.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/heuristics.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/primitives.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms/base.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms/direct.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms/ring.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms/resolve.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/latency.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/collective.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/tensor.hpp>
)

# The HIP-dependent device bridge header is registered separately: it is not part
# of the HIP-free header set above (it includes origami/hardware.hpp). It is
# installed via the include/ directory glob below and consumed through
# roc::origami-comm-device.

# ── origami::comm compiled library ───────────────────────────────────────────
# The non-constexpr bodies of the comm model are compiled once here, instead of
# being re-parsed inline in every consumer. The declarations and all constexpr
# code still come from the headers (roc::origami-comm-headers). Like that
# interface target, this library is pure C++17 and deliberately does NOT link
# hip::host, so the collective cost model remains usable without a ROCm toolchain.
if(ORIGAMI_BUILD_SHARED_LIBS OR (BUILD_SHARED_LIBS AND ORIGAMI_STANDALONE))
    add_library(origami-comm SHARED)
else()
    add_library(origami-comm STATIC)
endif()

rocm_set_soversion(origami-comm "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

add_library(roc::origami-comm ALIAS origami-comm)

set_target_properties(origami-comm PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_compile_features(origami-comm PUBLIC cxx_std_17)
target_link_libraries(origami-comm PUBLIC roc::origami-comm-headers)

target_sources(
    origami-comm
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/comm/primitives.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/comm/latency.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/comm/algorithms/direct.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/comm/algorithms/ring.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/comm/algorithms/resolve.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/comm/collective.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/comm/tensor.cpp"
)

# ── origami::comm device bridge — origami::hardware_t → comm system_t ─────────
# Unlike the rest of origami::comm, this small bridge IS HIP-dependent: it reads
# live device topology (origami::hardware_t / hipDeviceProp_t) so a comm system_t
# can describe the exact device about to run, including CPX partitions. It is a
# separate target precisely so the core cost model (roc::origami-comm) stays
# HIP-free and usable without a ROCm toolchain. Consumers that want device
# introspection link roc::origami-comm-device in addition.
if(ORIGAMI_BUILD_SHARED_LIBS OR (BUILD_SHARED_LIBS AND ORIGAMI_STANDALONE))
    add_library(origami-comm-device SHARED)
else()
    add_library(origami-comm-device STATIC)
endif()

rocm_set_soversion(origami-comm-device "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

add_library(roc::origami-comm-device ALIAS origami-comm-device)

set_target_properties(origami-comm-device PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_compile_features(origami-comm-device PUBLIC cxx_std_17)
target_link_libraries(origami-comm-device
    PUBLIC roc::origami-comm-headers roc::origami hip::host
)

target_sources(
    origami-comm-device
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/comm/hardware_device.cpp"
)

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
    rocm_install(TARGETS origami origami-headers origami-comm-headers origami-comm
                 origami-comm-device)

    rocm_export_targets(
        TARGETS
        roc::origami
        roc::origami-headers
        roc::origami-comm-headers
        roc::origami-comm
        roc::origami-comm-device
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
