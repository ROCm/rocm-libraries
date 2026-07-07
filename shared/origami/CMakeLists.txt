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

set_target_properties(origami PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_features(origami PUBLIC cxx_std_17)

include(GenerateExportHeader)
generate_export_header(origami
    BASE_NAME ORIGAMI
    EXPORT_MACRO_NAME ORIGAMI_EXPORT
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/origami/origami_export.h"
    STATIC_DEFINE ORIGAMI_STATIC
)
if(ORIGAMI_LIBRARY_TYPE STREQUAL "STATIC")
    target_compile_definitions(origami PUBLIC ORIGAMI_STATIC)
endif()

add_library(origami-headers INTERFACE)
add_library(roc::origami-headers ALIAS origami-headers)

target_compile_features(origami-headers INTERFACE cxx_std_17)

target_sources(origami-headers
    INTERFACE
    FILE_SET HEADERS
    BASE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${PROJECT_BINARY_DIR}/include"
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/architecture.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/attention.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/gemm.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/hardware.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/heuristics.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/logger.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/math.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/origami.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/streamk.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/types.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast_simulator.hpp"
        "${PROJECT_BINARY_DIR}/include/origami/origami_export.h"
)
target_include_directories(
    origami-headers INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(origami-headers INTERFACE hip::host)

set_target_properties(origami-headers PROPERTIES VERIFY_INTERFACE_HEADER_SETS ON)

target_link_libraries(origami PUBLIC roc::origami-headers)

add_dependencies(origami origami-headers_verify_interface_header_sets)

# ── origami::comm — analytical communication cost model ──────────────────────
# Header-only INTERFACE library. Deliberately does NOT link hip::host: the comm
# model is pure C++17 and usable without a ROCm toolchain, so consumers that only
# need collective cost prediction can depend on it independently of roc::origami.
add_library(origami-comm-headers INTERFACE)
add_library(roc::origami-comm-headers ALIAS origami-comm-headers)

target_compile_features(origami-comm-headers INTERFACE cxx_std_17)

target_sources(origami-comm-headers
    INTERFACE
    FILE_SET HEADERS
    BASE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/architecture.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/origami_comm.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/types.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/hardware.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/heuristics.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/primitives.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms/base.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms/direct.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms/ring.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/algorithms/resolve.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/latency.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/collective.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/comm/tensor.hpp"
)
# The comm headers transitively include origami/types.hpp, which pulls in the
# generated origami/origami_export.h under the build tree. Expose that generated
# include dir on the build interface (the FILE_SET BASE_DIRS above already covers
# the source include/ dir) so consumers compile without needing roc::origami.
target_include_directories(
    origami-comm-headers INTERFACE
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ── origami::comm compiled library ───────────────────────────────────────────
# The non-constexpr bodies of the comm model are compiled once here, instead of
# being re-parsed inline in every consumer. The declarations and all constexpr
# code still come from the headers (roc::origami-comm-headers). Like that
# interface target, this library is pure C++17 and deliberately does NOT link
# hip::host, so the collective cost model remains usable without a ROCm toolchain.
add_library(origami-comm ${ORIGAMI_LIBRARY_TYPE})

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
add_library(origami-comm-device ${ORIGAMI_LIBRARY_TYPE})

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
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/attention.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/gemm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/hardware.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/heuristics.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/logger.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/origami.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/streamk.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/types.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simulator/tensilelite/formocast_simulator.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simulator/tensilelite/formocast.cpp"
)

target_link_libraries(origami PRIVATE hip::host)

if(ORIGAMI_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

rocm_install(TARGETS origami origami-comm origami-comm-device)

install(TARGETS origami-headers origami-comm-headers
    EXPORT origami-targets
    FILE_SET HEADERS
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT devel
)

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
