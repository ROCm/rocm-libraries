# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# race-emulator: a CPU-side emulator for AMD GPU assembly that detects race
# conditions (missing s_waitcnt, s_barrier, etc.) without requiring a GPU.
#
# This library is used by hipblaslt and tensilelite to validate kernel assembly.
# It has no dependencies beyond the C++ standard library.
#
# When built as part of the rocm-libraries monorepo, the target roc::race-emulator
# is made available by the top-level CMakeLists.txt via add_subdirectory. When
# hipblaslt is built standalone, it pulls this in directly from shared/.
#
# To build tests: cmake -DRACE_EMULATOR_BUILD_TESTING=ON ...

cmake_minimum_required(VERSION 3.25.2)
project(race-emulator VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(RACE_EMULATOR_BUILD_TESTING "Build race-emulator unit tests (CPU-only, no GPU required)" OFF)
set(RACE_EMULATOR_BUILD_HW_TESTING "" CACHE STRING
    "GPU target for hardware validation tests, e.g. gfx942 (requires hipcc and a GPU)")

# Default to static when built as a subdirectory (e.g. from hipblaslt),
# shared when built standalone. Matches the origami convention.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    option(RACE_EMULATOR_BUILD_SHARED_LIBS "Build as shared library" ON)
else()
    option(RACE_EMULATOR_BUILD_SHARED_LIBS "Build as shared library" OFF)
endif()

if(RACE_EMULATOR_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    add_library(race-emulator SHARED)
else()
    add_library(race-emulator STATIC)
endif()

add_library(roc::race-emulator ALIAS race-emulator)

target_compile_features(race-emulator PUBLIC cxx_std_20)
set_target_properties(race-emulator PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD_REQUIRED ON
)

target_sources(race-emulator PRIVATE
    src/ControlFlowInstructions.cpp
    src/Emulator.cpp
    src/Instruction.cpp
    src/LDS.cpp
    src/MemoryInstructions.cpp
    src/MfmaInstructions.cpp
    src/Parsing.cpp
    src/Profiler.cpp
    src/RaceDetector.cpp
    src/ScalarInstructions.cpp
    src/Util.cpp
    src/VectorInstructions.cpp
    src/Wave.cpp
    src/WaveRaceState.cpp
    src/Workgroup.cpp
)

target_include_directories(race-emulator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(race-emulator PRIVATE -Wall -Wextra -pedantic)
    # F16C: hardware f16<->f32 conversion via vcvtph2ps/vcvtps2ph.
    # Available on x86-64 since Ivy Bridge (2012). Without this flag the
    # compiler targets a generic x86-64 baseline and cannot emit F16C
    # instructions, so our inline f16ToFloat/floatToF16 (FloatTypes.h) fall
    # back to multi-instruction software bit-manipulation (~4x slower).
    #
    # We enable -mf16c specifically rather than -march=native because
    # -march=native ties the binary to the build machine's full feature set
    # (AVX-512, etc.), which breaks if CI caches artifacts across different
    # CPU types. -mf16c is a single, safe, nearly universal extension.
    #
    # PUBLIC because FloatTypes.h is a public header with inline conversion
    # functions — consumers need the flag for consistent codegen.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-mf16c HAS_F16C)
    if(HAS_F16C)
        target_compile_options(race-emulator PUBLIC -mf16c)
    endif()
endif()

# race-emulator uses self-registering static globals to populate the instruction
# registry. With a static library the linker strips object files that have no
# externally referenced symbols, so those constructors never run. INTERFACE link
# options ensure that any consumer automatically gets --whole-archive.
if(NOT RACE_EMULATOR_BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
    if(MSVC)
        target_link_options(race-emulator INTERFACE "/WHOLEARCHIVE:race-emulator")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_link_options(race-emulator INTERFACE
            "LINKER:--whole-archive" $<TARGET_FILE:race-emulator> "LINKER:--no-whole-archive")
    endif()
endif()

# Tests
if(RACE_EMULATOR_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install rules
install(TARGETS race-emulator
    EXPORT race-emulator-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/race-emulatorConfig-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/race-emulatorConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/race-emulatorConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/race-emulator
)

install(EXPORT race-emulator-targets
    FILE race-emulator-targets.cmake
    NAMESPACE roc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/race-emulator
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/race-emulatorConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/race-emulatorConfig-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/race-emulator
)
