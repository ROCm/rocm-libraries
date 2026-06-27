# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/modules/default_amdclang.cmake)
endif()

set(STINKYTOFU_VERSION "0.1.0")
project(stinkytofu VERSION ${STINKYTOFU_VERSION} LANGUAGES CXX)

set(STINKYTOFU_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(STINKYTOFU_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(STINKYTOFU_VERSION_PATCH ${PROJECT_VERSION_PATCH})

find_package(ROCmCMakeBuildTools QUIET CONFIG)
set(STINKYTOFU_VERSION_TWEAK "")
if(ROCmCMakeBuildTools_FOUND)
    include(ROCMSetupVersion)
    rocm_get_git_commit_tag(STINKYTOFU_VERSION_TWEAK)
else()
    message(WARNING "ROCmCMakeBuildTools not found — version TWEAK (git hash) will be empty.")
endif()
message(STATUS "StinkyTofu version: ${STINKYTOFU_VERSION} (${STINKYTOFU_VERSION_TWEAK})")

configure_file(
    include/stinkytofu/Version.h.in
    ${PROJECT_BINARY_DIR}/include/stinkytofu/Version.h
    @ONLY)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable assertions even in Release builds
# Remove NDEBUG from Release build types
foreach(flag_var
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "(^| )-DNDEBUG($| )" " " ${flag_var} "${${flag_var}}")
    string(STRIP "${${flag_var}}" ${flag_var})
endforeach()

add_compile_options(-fno-rtti -fno-exceptions)

# Hide all symbols by default; only STINKYTOFU_EXPORT symbols are visible.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# On Windows, DLLs must be next to executables at runtime. Place all
# binaries in a single directory so tools/tests find stinkytofu.dll.
if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

set(STINKYTOFU_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(STINKYTOFU_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)

##===----------------------------------------------------------------------===##
# Architectures
##===----------------------------------------------------------------------===##
include(cmake/StinkytofuArchList.cmake)

set(STINKYTOFU_ARCHS_TO_BUILD "all"
    CACHE STRING "Semicolon-separated list of backends to build, or \"all\".")

if (STINKYTOFU_ARCHS_TO_BUILD STREQUAL "all" )
    set(STINKYTOFU_ARCHS_TO_BUILD ${STINKYTOFU_ALL_ARCHS})
endif()

foreach(arch ${STINKYTOFU_ARCHS_TO_BUILD})
    list(APPEND STINKYTOFU_ARCHS "${arch}")
    set(STINKYTOFU_ENUM_ARCHS "${STINKYTOFU_ENUM_ARCHS}STINKYTOFU_ARCH(${arch})\n")

    # Set architecture variable for #cmakedefine
    string(TOUPPER "${arch}" ARCH_UPPER)
    set(STINKYTOFU_ARCH_${ARCH_UPPER} 1)
endforeach(arch)

message(STATUS "Using archs: ${STINKYTOFU_ARCHS}")

# AMD comgr for runtime assembler capability probing (must be before configure_file)
# Prefer the ROCM_PATH cache variable (-DROCM_PATH=...) over the environment.
if(ROCM_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
elseif(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
endif()
find_package(amd_comgr CONFIG)
if(NOT amd_comgr_FOUND)
    message(FATAL_ERROR
        "amd_comgr not found. Set ROCM_PATH or CMAKE_PREFIX_PATH to your ROCm installation.")
endif()
set(STINKYTOFU_HAS_COMGR 1)

configure_file(hardware/hardware.def.in include/Config/Archs.def @ONLY)
configure_file(include/stinkytofu/Config.h.in include/stinkytofu/Config/Config.h)

##===----------------------------------------------------------------------===##
# Warning flags (single source of truth for all stinkytofu targets)
##===----------------------------------------------------------------------===##
option(STINKYTOFU_ENABLE_WERROR "Treat warnings as errors" OFF)
add_library(stinkytofu-warnings INTERFACE)
target_compile_options(stinkytofu-warnings INTERFACE -Wall)
if(STINKYTOFU_ENABLE_WERROR)
    target_compile_options(stinkytofu-warnings INTERFACE
        -Werror -Wno-error=deprecated-declarations -Wno-error=comment)
endif()

##===----------------------------------------------------------------------===##
# Code coverage
##===----------------------------------------------------------------------===##
# LLVM source-based coverage for amdclang/Clang (gcov-style for GCC). Applied
# globally (before any targets are defined) so the library, tools, and tests are
# all instrumented — the FileCheck and Python tests exercise library code via
# separate processes, so the library itself must carry the instrumentation.
# Drive it with `invoke coverage`; reports filter out tests/examples/deps.
option(STINKYTOFU_CODE_COVERAGE "Build with code coverage instrumentation" OFF)
if(STINKYTOFU_CODE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
        add_link_options(-fprofile-instr-generate -fcoverage-mapping)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(--coverage)
        add_link_options(--coverage)
    else()
        message(FATAL_ERROR
            "STINKYTOFU_CODE_COVERAGE is not supported for compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()
    message(STATUS "StinkyTofu: code coverage instrumentation enabled")
endif()

##===----------------------------------------------------------------------===##
# Hardware library and tablegen tools
##===----------------------------------------------------------------------===##
# Add subdirectories first to define gfxisa, tablegen, and tablegen_generated targets
add_subdirectory(hardware)
add_subdirectory(tools/tablegen)

# gfxisa needs generated instruction .inc files (hardware/generated/*.inc)
target_include_directories(gfxisa PUBLIC ${TABLEGEN_OUTPUT_DIR})
add_dependencies(gfxisa instruction_generated)

##===----------------------------------------------------------------------===##
# StinkyTofu library
##===----------------------------------------------------------------------===##
# Note: Stinkytofu library should be built with:
# (1) No RTTI
# (2) No exceptions
#
# => clang/gcc: -fno-rtti -fno-exceptions
# => MSVC: /GR- /EHs-c-

# Now create stinkytofu library which depends on tablegen_generated
# Add generated source files
set(STINKYTOFU_GENERATED_SOURCES
    "${TABLEGEN_OUTPUT_DIR}/stinkytofu/ir/logical/LogicalOpcode.cpp"
)

# Mark generated files so CMake knows they will be created during build
set_source_files_properties(${STINKYTOFU_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Custom RPATH: bake link-dependency dirs (e.g. amd_comgr) into the installed
# libstinkytofu when those deps are not on the loader's default search path.
option(STINKYTOFU_INSTALL_RPATH_USE_LINK_PATH
    "Use link-dependency dirs as the installed libstinkytofu RPATH" OFF)

# dladdr() used in IntrinsicRegistry.cpp requires -ldl
# Threads required for std::call_once (separate libpthread on glibc < 2.34)
find_package(Threads REQUIRED)

# Private link deps shared by stinkytofu_objs, stinkytofu, and stinkytofu_static.
# All three need the same libraries for linking/include propagation.
set(STINKYTOFU_PRIVATE_LINK_LIBS
    stinkytofu-warnings
    amd_comgr
    ${CMAKE_DL_LIBS}
    Threads::Threads
)

# Public include dirs shared by stinkytofu and stinkytofu_static.
set(STINKYTOFU_PUBLIC_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

# All sources are compiled once into an OBJECT library. The shared/static
# stinkytofu targets and the test-only stinkytofu_static target all reuse
# these object files, avoiding redundant compilation.
add_library(stinkytofu_objs OBJECT ${STINKYTOFU_GENERATED_SOURCES})
# OBJECT libraries don't inherit -fPIC from their consumers on Linux; set it explicitly.
set_property(TARGET stinkytofu_objs PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(stinkytofu_objs PRIVATE ${STINKYTOFU_PRIVATE_LINK_LIBS})
# CMake auto-defines <target>_EXPORTS for SHARED/STATIC targets but not for
# OBJECT libraries. Explicitly define stinkytofu_EXPORTS so STINKYTOFU_EXPORT
# expands to dllexport (not dllimport) when compiling the object files.
target_compile_definitions(stinkytofu_objs PRIVATE stinkytofu_EXPORTS)
target_include_directories(stinkytofu_objs
    PRIVATE
        ${STINKYTOFU_PUBLIC_INCLUDE_DIRS}
        "${TABLEGEN_OUTPUT_DIR}"
        "${HW_ARCH_HEADERS_DIR}"
)
add_dependencies(stinkytofu_objs tablegen_generated)

# stinkytofu: shared (default) or static depending on BUILD_SHARED_LIBS.
# External consumers (rocisa, stinkytofu-opt) link this target.
# stinkytofu_objs is PRIVATE so it does not appear in the install export set.
# Public include dirs are set directly on stinkytofu so consumers get them.
add_library(stinkytofu $<TARGET_OBJECTS:stinkytofu_objs>)
target_link_libraries(stinkytofu PRIVATE ${STINKYTOFU_PRIVATE_LINK_LIBS})
target_include_directories(stinkytofu
    PUBLIC
        ${STINKYTOFU_PUBLIC_INCLUDE_DIRS}
        $<INSTALL_INTERFACE:include/stinkytofu>
)
set_target_properties(stinkytofu PROPERTIES
    VERSION "${STINKYTOFU_VERSION_MAJOR}.${STINKYTOFU_VERSION_MINOR}.${STINKYTOFU_VERSION_PATCH}"
    SOVERSION "${STINKYTOFU_VERSION_MAJOR}"
    INSTALL_RPATH_USE_LINK_PATH ${STINKYTOFU_INSTALL_RPATH_USE_LINK_PATH})
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(stinkytofu PUBLIC STINKYTOFU_STATIC)
endif()

# Namespaced alias so add_subdirectory consumers use the same target name as
# find_package consumers (the install export uses NAMESPACE stinkytofu::).
add_library(stinkytofu::stinkytofu ALIAS stinkytofu)

# On Windows, delay-load amd_comgr.dll so it is resolved at first use rather
# than at process startup.  A delay-load hook in ComgrProbe.cpp uses
# LoadLibraryExA to load the DLL from ROCM_PATH\bin.
if(WIN32)
    if(BUILD_SHARED_LIBS)
        target_link_options(stinkytofu PRIVATE "LINKER:/DELAYLOAD:$<TARGET_FILE_NAME:amd_comgr>")
        target_link_libraries(stinkytofu PRIVATE delayimp)
    else()
        target_link_options(stinkytofu INTERFACE "LINKER:/DELAYLOAD:$<TARGET_FILE_NAME:amd_comgr>")
        target_link_libraries(stinkytofu INTERFACE delayimp)
    endif()
endif()

# Add source files from src/ subdirectory
add_subdirectory(src)

# When building shared + tests: provide a static twin so unit_tests can access
# non-exported symbols for white-box coverage testing. Not needed when
# BUILD_SHARED_LIBS=OFF because stinkytofu itself is already a static lib.
if(BUILD_SHARED_LIBS AND STINKYTOFU_BUILD_TESTS)
    add_library(stinkytofu_static STATIC $<TARGET_OBJECTS:stinkytofu_objs>)
    target_link_libraries(stinkytofu_static PRIVATE ${STINKYTOFU_PRIVATE_LINK_LIBS})
    target_include_directories(stinkytofu_static PUBLIC ${STINKYTOFU_PUBLIC_INCLUDE_DIRS})
    target_compile_definitions(stinkytofu_static PUBLIC STINKYTOFU_STATIC)
    if(WIN32)
        target_link_options(stinkytofu_static INTERFACE "LINKER:/DELAYLOAD:$<TARGET_FILE_NAME:amd_comgr>")
        target_link_libraries(stinkytofu_static INTERFACE delayimp)
    endif()
endif()

##===----------------------------------------------------------------------===##
# Example plugins
##===----------------------------------------------------------------------===##
# Single switch for everything example-related: the example plugins themselves and
# the unit tests that exercise them. Default OFF so integrated/ROCm builds never
# build or package examples; standalone builds opt in via tasks.py. Added before
# tests/, which gate their plugin coverage on this same flag. New example tests
# should hook in under STINKYTOFU_BUILD_EXAMPLES too.
option(STINKYTOFU_BUILD_EXAMPLES "Build StinkyTofu example plugins (and their tests)" OFF)

# Single source of truth for where plugins live relative to libstinkytofu's lib dir.
# Shared by the install rule, the compiled-in runtime lookup (PassBuilder::examplePluginPath),
# and consumers that vendor libstinkytofu (rocisa) so they install plugins to a matching path.
set(STINKYTOFU_PLUGIN_INSTALL_RELDIR "stinkytofu/plugins"
    CACHE STRING "Plugin directory relative to the libstinkytofu lib dir")

if(STINKYTOFU_BUILD_EXAMPLES)
    add_subdirectory(examples/plugins)
endif()

##===----------------------------------------------------------------------===##
# Unit tests
##===----------------------------------------------------------------------===##
# Option to build unit tests
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_stinkytofu_build_tests_default ON)
else()
    set(_stinkytofu_build_tests_default OFF)
endif()
option(STINKYTOFU_BUILD_TESTS "Build StinkyTofu unit tests" ${_stinkytofu_build_tests_default})
if(STINKYTOFU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

##===----------------------------------------------------------------------===##
# Python bindings
##===----------------------------------------------------------------------===##
# Option to build Python bindings (default OFF when used as sub-project)
option(STINKYTOFU_BUILD_PYTHON "Build StinkyTofu Python bindings" ${_stinkytofu_build_tests_default})
if(STINKYTOFU_BUILD_PYTHON)
    add_subdirectory(python_module)
endif()

##===----------------------------------------------------------------------===##
# Tools (stinkytofu-opt, stinkytofu-check, intrinsic-compiler)
##===----------------------------------------------------------------------===##
# Added after the stinkytofu library. tablegen is added separately above
# because it sets TABLEGEN_OUTPUT_DIR needed before gfxisa.
add_subdirectory(tools)

##===----------------------------------------------------------------------===##
# Install targets
##===----------------------------------------------------------------------===##

install(TARGETS stinkytofu stinkytofu-warnings
    EXPORT stinkytofu-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(EXPORT stinkytofu-targets
    FILE stinkytofu-targets.cmake
    NAMESPACE stinkytofu::
    DESTINATION lib/cmake/stinkytofu
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/stinkytofuConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/stinkytofuConfig.cmake
    INSTALL_DESTINATION lib/cmake/stinkytofu
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/stinkytofuConfigVersion.cmake
    VERSION "${STINKYTOFU_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/stinkytofuConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/stinkytofuConfigVersion.cmake
    DESTINATION lib/cmake/stinkytofu
)

# --------------------------------------------------------------------------- #
# clang-tidy
# --------------------------------------------------------------------------- #
option(ENABLE_CLANG_TIDY "Enable clang-tidy targets" OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/modules")
include(ClangTidy)
add_clang_tidy_custom_target()

# Build-tree export so find_package() works without installing.
# Write targets to stinkytofu-targets.cmake (not stinkytofuConfig.cmake) to
# avoid clobbering the configure_package_config_file output above.
export(EXPORT stinkytofu-targets NAMESPACE stinkytofu:: FILE stinkytofu-targets.cmake)

# Install public header files with stinkytofu/ namespace
# Exclude ir/python/ (Python-specific, internal headers)
# cmake-lint: disable=E1122
install(DIRECTORY include/
    DESTINATION include/stinkytofu
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.def"
    PATTERN "ir/python" EXCLUDE
)
# Install generated header files
install(FILES
    ${PROJECT_BINARY_DIR}/stinkytofu/ir/logical/LogicalInstructions_generated.hpp
    DESTINATION include/stinkytofu/ir/logical
)
# Generated headers included by public headers but produced at build time.
# Installed at the location each public #include resolves to (root: include/stinkytofu).
#   hardware/GfxIsa.hpp -> "Config/Archs.def"
#   *.cpp               -> "stinkytofu/Config/Config.h"
#   ir/logical/LogicalOpcode.hpp -> "LogicalOpcodes_generated.inc"
#   ir/asm/StinkyAsmIR.hpp       -> "hardware/gfxIsa.inc"
install(FILES
    ${PROJECT_BINARY_DIR}/include/Config/Archs.def
    DESTINATION include/stinkytofu/Config
)
install(FILES
    ${PROJECT_BINARY_DIR}/include/stinkytofu/Config/Config.h
    DESTINATION include/stinkytofu/stinkytofu/Config
)
install(FILES
    ${TABLEGEN_OUTPUT_DIR}/LogicalOpcodes_generated.inc
    DESTINATION include/stinkytofu
)
install(FILES
    ${TABLEGEN_OUTPUT_DIR}/hardware/gfxIsa.inc
    DESTINATION include/stinkytofu/hardware
)
# Per-arch rocisa conversion mappings, included by the rocisa conversion glue
# (src/conversion/rocisa/*ArchInfo.hpp) which rocisa compiles into _rocisa.
install(DIRECTORY ${TABLEGEN_OUTPUT_DIR}/stinkytofu/ir/rocisa/
    DESTINATION include/stinkytofu/stinkytofu/ir/rocisa
    FILES_MATCHING PATTERN "*.inc"
)
install(FILES
    ${PROJECT_BINARY_DIR}/include/stinkytofu/Version.h
    DESTINATION include/stinkytofu
)
# Tablegen-generated file needed by the Python module when built standalone
# against an installed stinkytofu.
install(FILES
    ${TABLEGEN_OUTPUT_DIR}/PythonBindings_generated.inc
    DESTINATION include/stinkytofu/bindings/python
)
