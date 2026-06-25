# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tilewright: standalone, framework-agnostic kernel-recommender (MLREC_v1 runtime).
# Builds a STATIC library with NO GEMM-framework dependency.

cmake_minimum_required(VERSION 3.25.2)

project(Tilewright VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TILEWRIGHT_STANDALONE ON)
else()
    set(TILEWRIGHT_STANDALONE OFF)
endif()

option(TILEWRIGHT_ENABLE_INSTALL "Configure tilewright installation" ON)
option(TILEWRIGHT_ENABLE_PYTHON "Enable Python bindings." OFF)
option(TILEWRIGHT_BUILD_TESTING "Build tests." OFF)
option(TILEWRIGHT_ENABLE_FETCH "Auto-fetch dependencies (nanobind/Catch2) with FetchContent" ON)
# The hipBLASLt backend is header-only glue that references origami headers; it
# is enabled when tilewright is built inside hipBLASLt (where roc::origami-headers
# exists), and OFF for the framework-agnostic standalone build.
option(TILEWRIGHT_ENABLE_HIPBLASLT_BACKEND "Build the hipBLASLt backend adapter (needs origami headers)" OFF)

# ── the static library ─────────────────────────────────────────────────────
add_library(tilewright STATIC)
add_library(roc::tilewright ALIAS tilewright)

set_target_properties(tilewright PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_compile_features(tilewright PUBLIC cxx_std_17)

# Public headers (tilewright is self-contained: no GEMM-framework / HIP headers).
add_library(tilewright-headers INTERFACE)
add_library(roc::tilewright-headers ALIAS tilewright-headers)
target_compile_features(tilewright-headers INTERFACE cxx_std_17)
target_include_directories(
    tilewright-headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(
    tilewright-headers
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/tilewright/model.hpp>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/tilewright/types.hpp>
)

target_include_directories(tilewright PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(tilewright PUBLIC roc::tilewright-headers)

# The engine is hot CPU inference code. Use portable optimization flags by
# default; AVX512 is OPT-IN (TILEWRIGHT_ENABLE_AVX512) and only added when the
# compiler accepts the flags. AVX512 must NOT be enabled from -march=native: that
# bakes in the build host's ISA and would crash on deploy hosts without AVX512.
# The AVX512 code paths are guarded by __AVX512F__/__AVX512BF16__, so without
# these flags the engine compiles to the portable scalar fp32 path.
option(TILEWRIGHT_ENABLE_AVX512 "Build the scorer with AVX512 (x86 deploy hosts that support it)" OFF)
set(_tw_model_flags -O3 -ffast-math -funroll-loops -fopenmp-simd)
if(TILEWRIGHT_ENABLE_AVX512)
    include(CheckCXXCompilerFlag)
    foreach(_f "-mavx512f" "-mavx512bw" "-mavx512vnni" "-mavx512bf16" "-mfma")
        string(MAKE_C_IDENTIFIER "TW_HAS${_f}" _v)
        check_cxx_compiler_flag("${_f}" ${_v})
        if(${_v})
            list(APPEND _tw_model_flags "${_f}")
        endif()
    endforeach()
endif()
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/model.cpp"
    PROPERTIES COMPILE_OPTIONS "${_tw_model_flags}"
)

target_sources(
    tilewright
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/model.cpp"
)

# dladdr-based weight auto-discovery needs libdl on Unix.
if(UNIX)
    target_link_libraries(tilewright PUBLIC ${CMAKE_DL_LIBS})
endif()

# ── framework backends (per-GEMM-library glue) ──────────────────────────────
if(TILEWRIGHT_ENABLE_HIPBLASLT_BACKEND)
    add_subdirectory(backends/hipblaslt)
endif()

# ── python bindings + tests (mirroring the framework libs) ──────────────────
if(TILEWRIGHT_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(TILEWRIGHT_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── install ─────────────────────────────────────────────────────────────────
if(TILEWRIGHT_ENABLE_INSTALL OR TILEWRIGHT_STANDALONE)
    include(GNUInstallDirs)

    install(TARGETS tilewright tilewright-headers EXPORT tilewright-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )

    install(
        DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING
        PATTERN "*.hpp"
    )

    # Per-library weights are NOT installed here. The framework backend's build
    # co-locates them with its kernel library (e.g. hipBLASLt copies the
    # tilewright_index + *.tilewright.bin into its per-arch Tensile library dir),
    # so they are discovered and packaged exactly like the kernel library.
endif()
