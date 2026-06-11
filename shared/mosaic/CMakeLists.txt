# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# mosaic: standalone, framework-agnostic kernel-recommender (MLREC_v6 runtime).
# Builds a STATIC library with NO origami / hipBLASLt dependency.

cmake_minimum_required(VERSION 3.25.2)

project(Mosaic VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MOSAIC_STANDALONE ON)
else()
    set(MOSAIC_STANDALONE OFF)
endif()

option(MOSAIC_ENABLE_INSTALL "Configure mosaic installation" ON)

# ── the static library ─────────────────────────────────────────────────────
add_library(mosaic STATIC)
add_library(roc::mosaic ALIAS mosaic)

set_target_properties(mosaic PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_compile_features(mosaic PUBLIC cxx_std_17)

# Public headers (mosaic is self-contained: no origami / HIP headers).
add_library(mosaic-headers INTERFACE)
add_library(roc::mosaic-headers ALIAS mosaic-headers)
target_compile_features(mosaic-headers INTERFACE cxx_std_17)
target_include_directories(
    mosaic-headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(
    mosaic-headers
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mosaic/mosaic.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mosaic/types.hpp>
)

target_include_directories(mosaic PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mosaic PUBLIC roc::mosaic-headers)

# The engine is hot CPU inference code: same aggressive flags origami used for
# the reconstructed ml_recommender.cpp. AVX512 paths are guarded by __AVX512F__
# so the scalar fallback is used where the ISA is unavailable.
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mosaic/model.cpp"
    PROPERTIES COMPILE_OPTIONS
    "-O3;-march=native;-mavx512f;-mavx512bw;-mfma;-ffast-math;-funroll-loops;-fopenmp-simd"
)

target_sources(
    mosaic
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/mosaic/model.cpp"
)

# dladdr-based weight auto-discovery needs libdl on Unix.
if(UNIX)
    target_link_libraries(mosaic PUBLIC ${CMAKE_DL_LIBS})
endif()

# ── install ─────────────────────────────────────────────────────────────────
if(MOSAIC_ENABLE_INSTALL OR MOSAIC_STANDALONE)
    include(GNUInstallDirs)

    install(TARGETS mosaic mosaic-headers EXPORT mosaic-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )

    install(
        DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
    )

    # Weights ship under share/mosaic/data/hipblaslt (matches the auto-discovery
    # search path in model.cpp). The parent moves the .bin into data/hipblaslt.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/hipblaslt/ml_recommender_weights.bin")
        install(
            FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/hipblaslt/ml_recommender_weights.bin"
            DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/mosaic/data/hipblaslt"
        )
    endif()
endif()
