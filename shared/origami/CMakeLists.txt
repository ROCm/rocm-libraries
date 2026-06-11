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
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/gemm.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/hardware.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/heuristics.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/logger.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/math.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/ml_recommender.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/origami.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/streamk.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/types.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast_simulator.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast.hpp>
)

target_link_libraries(origami PUBLIC roc::origami-headers)

# The MLREC_v6 engine now lives in the standalone, framework-agnostic `mosaic`
# library (../mosaic). To keep liborigami.a self-contained for direct-linking
# consumers (hipBLASLt, the F1 microbench) with ZERO link-topology changes, we
# compile the mosaic engine TU directly into the origami target. mosaic also
# ships its own CMakeLists.txt for standalone / future-backend (Triton/CK)
# builds. origami's ml_recommender.cpp is now a thin origami<->mosaic adapter.
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/ml_recommender.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../mosaic/src/mosaic/model.cpp"
    PROPERTIES COMPILE_OPTIONS
    "-O3;-march=native;-mavx512f;-mavx512bw;-mfma;-ffast-math;-funroll-loops;-fopenmp-simd"
)

# mosaic is self-contained (no origami headers); the adapter needs its headers.
target_include_directories(origami PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../mosaic/include"
)

target_sources(
    origami
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/gemm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/hardware.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/heuristics.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/logger.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/ml_recommender.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/../mosaic/src/mosaic/model.cpp"
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

    # The mosaic engine (compiled into liborigami here, rather than
    # add_subdirectory'd) auto-discovers its weights relative to the library, so
    # install them to the mosaic data root. (Step 2 moves this integration so
    # hipBLASLt links mosaic directly and mosaic installs its own weights.)
    set(MOSAIC_WEIGHTS_BIN "${CMAKE_CURRENT_SOURCE_DIR}/../mosaic/backends/hipblaslt/data/mosaic_weights.bin")
    if(EXISTS "${MOSAIC_WEIGHTS_BIN}")
        rocm_install(
            FILES "${MOSAIC_WEIGHTS_BIN}"
            DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/mosaic/backends/hipblaslt/data"
            COMPONENT runtime
        )
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
endif()
