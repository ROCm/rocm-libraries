# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/modules/default_amdclang.cmake)
endif()

project(stinkytofu)

set(STINKYTOFU_VERSION "0.1.0")
string(REPLACE "." ";" _ver_list ${STINKYTOFU_VERSION})
list(GET _ver_list 0 STINKYTOFU_VERSION_MAJOR)
list(GET _ver_list 1 STINKYTOFU_VERSION_MINOR)
list(GET _ver_list 2 STINKYTOFU_VERSION_PATCH)

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

configure_file(hardware/hardware.def.in include/Config/Archs.def @ONLY)
configure_file(include/stinkytofu/Config.h.in include/stinkytofu/Config/Config.h)

##===----------------------------------------------------------------------===##
# ROCm ISA to StinkyTofu conversion
##===----------------------------------------------------------------------===##
# Include rocisa conversion sources (separate target, not stinkytofu library)
# Note: Do not add rocisa sources to the stinkytofu library target.
# They are built as a separate rocisa target defined in rocisa/CMakeLists.txt.
#
# This is a temporary solution until:
# (1) stinkytofu designs a new rocisa IR to replace the original ones, or
# (2) stinkytofu completely replaces rocisa IR with stinkyAsm IR.
if(TARGET _rocisa)
    add_subdirectory(src/conversion/rocisa)
endif()

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
add_library(stinkytofu ${STINKYTOFU_GENERATED_SOURCES})
target_link_libraries(stinkytofu PRIVATE stinkytofu-warnings)
set_target_properties(stinkytofu PROPERTIES
    VERSION "${STINKYTOFU_VERSION_MAJOR}.${STINKYTOFU_VERSION_MINOR}.${STINKYTOFU_VERSION_PATCH}"
    SOVERSION "${STINKYTOFU_VERSION_MAJOR}")
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(stinkytofu PUBLIC STINKYTOFU_STATIC)
endif()
target_include_directories(stinkytofu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include/stinkytofu>
    PRIVATE
        # Make the generated files' directories available to the library
        # This allows source files to #include the generated .inc files
        "${TABLEGEN_OUTPUT_DIR}"
        # Generated GfxXXX.hpp and ArchHelper_includes.inc (from hardware/CMakeLists.txt)
        # PRIVATE: do not expose to consumers of stinkytofu
        "${HW_ARCH_HEADERS_DIR}"
)

# Make stinkytofu depend on the generated ISA definition files from tablegen
# This ensures the build order is: gfxisa -> tablegen -> run tablegen -> stinkytofu
add_dependencies(stinkytofu tablegen_generated)

# dladdr() used in IntrinsicRegistry.cpp requires -ldl
# Threads required for std::call_once (separate libpthread on glibc < 2.34)
find_package(Threads REQUIRED)
target_link_libraries(stinkytofu PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

# Add source files from src/ subdirectory
add_subdirectory(src)

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

# Install the stinkytofu library and export targets for find_package().
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
export(TARGETS stinkytofu stinkytofu-warnings NAMESPACE stinkytofu:: FILE stinkytofuConfig.cmake)

# Install public header files with stinkytofu/ namespace
# Exclude ir/python/ (Python-specific, internal headers)
# cmake-lint: disable=E1122
install(DIRECTORY include/
    DESTINATION include/stinkytofu
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    PATTERN "ir/python" EXCLUDE
)
# Install generated header files
install(FILES
    ${PROJECT_BINARY_DIR}/stinkytofu/ir/logical/LogicalInstructions_generated.hpp
    DESTINATION include/stinkytofu/ir/logical
)
install(FILES
    ${PROJECT_BINARY_DIR}/include/stinkytofu/Version.h
    DESTINATION include/stinkytofu
)
