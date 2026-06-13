# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# mosaic: standalone, framework-agnostic kernel-recommender (MLREC_v1 runtime).
# Builds a STATIC library with NO GEMM-framework dependency.

cmake_minimum_required(VERSION 3.25.2)

project(Mosaic VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MOSAIC_STANDALONE ON)
else()
    set(MOSAIC_STANDALONE OFF)
endif()

option(MOSAIC_ENABLE_INSTALL "Configure mosaic installation" ON)
option(MOSAIC_ENABLE_PYTHON "Enable Python bindings." OFF)
option(MOSAIC_BUILD_TESTING "Build tests." OFF)
option(MOSAIC_ENABLE_FETCH "Auto-fetch dependencies (nanobind/Catch2) with FetchContent" ON)
# The hipBLASLt backend is header-only glue that references origami headers; it
# is enabled when mosaic is built inside hipBLASLt (where roc::origami-headers
# exists), and OFF for the framework-agnostic standalone build.
option(MOSAIC_ENABLE_HIPBLASLT_BACKEND "Build the hipBLASLt backend adapter (needs origami headers)" OFF)

# ── the static library ─────────────────────────────────────────────────────
add_library(mosaic STATIC)
add_library(roc::mosaic ALIAS mosaic)

set_target_properties(mosaic PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_compile_features(mosaic PUBLIC cxx_std_17)

# Public headers (mosaic is self-contained: no GEMM-framework / HIP headers).
add_library(mosaic-headers INTERFACE)
add_library(roc::mosaic-headers ALIAS mosaic-headers)
target_compile_features(mosaic-headers INTERFACE cxx_std_17)
target_include_directories(
    mosaic-headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(
    mosaic-headers
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mosaic/model.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mosaic/types.hpp>
)

target_include_directories(mosaic PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mosaic PUBLIC roc::mosaic-headers)

# The engine is hot CPU inference code: aggressive flags for the scoring loop.
#
# IMPORTANT (portability): the AVX512 code paths in model.cpp are guarded by the
# __AVX512F__ predefined macro and the runtime CPUID dispatch in select_dot_fn()
# is meant to fall back to scalar code on CPUs without AVX512. That contract is
# only honest if we do NOT *force* AVX512 codegen here: an unconditional
# `-mavx512f` makes __AVX512F__ true and emits AVX512 instructions in EVERY
# function (including the scalar fallbacks the compiler may auto-vectorize), so
# the resulting object requires AVX512 to run and SIGILLs (illegal instruction)
# on any non-AVX512 host -- e.g. Zen2/Zen3 EPYC servers commonly paired with the
# GPUs hipBLASLt targets. We therefore let `-march=native` decide the ISA from
# the build host: it enables AVX512 (and FMA) exactly when the host supports it,
# keeping the scalar path for hosts that don't. This mirrors how the rest of the
# repo (e.g. composablekernel) gates -mavx512f behind a host-capability check.
#
# NOTE (parity): -ffast-math relaxes IEEE float semantics (reassociation, FMA
# contraction, finite-math). It is kept here because the deployed weights' C++
# <-> Python parity was validated with it, but it technically conflicts with the
# byte-exact parity contract documented in model.cpp; re-validate parity
# (deploy stage07) if this flag set changes.
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mosaic/model.cpp"
    PROPERTIES COMPILE_OPTIONS
    "-O3;-march=native;-ffast-math;-funroll-loops;-fopenmp-simd"
)

target_sources(
    mosaic
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/mosaic/model.cpp"
)

# dladdr-based weight auto-discovery needs libdl on Unix.
if(UNIX)
    target_link_libraries(mosaic PUBLIC ${CMAKE_DL_LIBS})
endif()

# ── framework backends (per-GEMM-library glue) ──────────────────────────────
if(MOSAIC_ENABLE_HIPBLASLT_BACKEND)
    add_subdirectory(backends/hipblaslt)
endif()

# ── python bindings + tests (mirroring the framework libs) ──────────────────
if(MOSAIC_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(MOSAIC_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
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

    # Per-library weights ship under share/mosaic/backends/hipblaslt/data: one
    # .mosaic.bin per library-logic stem plus the mosaic_index that maps stems
    # -> binaries. Each PredictionLibrary resolves its own model via the index.
    file(GLOB MOSAIC_WEIGHT_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/hipblaslt/data/*.mosaic.bin")
    if(MOSAIC_WEIGHT_FILES)
        install(
            FILES ${MOSAIC_WEIGHT_FILES}
            DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/mosaic/backends/hipblaslt/data"
        )
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/backends/hipblaslt/data/mosaic_index")
        install(
            FILES "${CMAKE_CURRENT_SOURCE_DIR}/backends/hipblaslt/data/mosaic_index"
            DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/mosaic/backends/hipblaslt/data"
        )
    endif()
endif()
