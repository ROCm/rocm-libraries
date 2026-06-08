# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
message(STATUS "Building CK DSL provider plugin")
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Version handling: read version.json + git SHA into CK_DSL_PROVIDER_*
# variables before project() so they can drive the project version.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CkDslProviderVersion.cmake)
ck_dsl_provider_setup_version()

project(ck-dsl-provider VERSION ${CK_DSL_PROVIDER_VERSION} LANGUAGES CXX)

include(CMakePackageConfigHelpers)

if(NOT WIN32)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
    endif()
    message(STATUS "CK DSL provider install prefix: ${CMAKE_INSTALL_PREFIX}")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
    CACHE BOOL "Add paths to linker search and installed rpath"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${ROCM_PATH}/lib/cmake/hip ${HIP_DIR}/cmake
)

option(CKDSLPROVIDER_SKIP_TESTS "Skip building tests" OFF)

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files;
# remove when fixed. Added to prevent issues with not finding hip cmake.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

find_package(hip REQUIRED)

# --- Embedded Python (python-build-standalone + pybind11) --------------------
# The plugin embeds a self-contained CPython so the provider can drive
# the CK DSL compile pipeline at runtime with no dependency on the host
# system Python. The interpreter is provisioned from astral's
# python-build-standalone (a pinned, relocatable prefix), discovered via
# find_package below, and bundled beside the plugin at install time.
# See cmake/CkDslPython.cmake. Provisioning honors -DCKDSL_PYTHON_DIST_DIR
# for air-gapped / pre-staged builds, else downloads the pinned tarball.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CkDslPython.cmake)
ck_dsl_provider_provision_python()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "CK DSL provider Python3: ${Python3_EXECUTABLE} (${Python3_VERSION})")

# pybind11: prefer an installed CMake config; otherwise fetch a pinned
# release. pybind11 is header-only for the embed use case and resolves
# the provisioned interpreter through Python3_ROOT_DIR (set by the
# provisioning step above), so no host `pip install` is required.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    set(PYBIND11_FINDPYTHON ON CACHE INTERNAL "")
    fetchcontent_declare(
        pybind11 URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
                     DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    fetchcontent_makeavailable(pybind11)
endif()
message(STATUS "CK DSL provider pybind11: ${pybind11_VERSION}")

# hipdnn SDK dependencies - use targets directly if available (superbuild),
# otherwise find installed packages (true standalone).
if(NOT TARGET hipdnn_data_sdk)
    find_package(hipdnn_data_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_flatbuffers_sdk)
    find_package(hipdnn_flatbuffers_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

if(WIN32)
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "bin")
else()
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "lib")
endif()

# Both HIPDNN_PLUGIN_ENGINE_SUBDIR and HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR
# are populated by the hipdnn_data_sdk package import above. Guard
# defensively so a standalone build (or a misconfigured SDK install)
# fails with a clear message instead of silently producing
# `build/lib//libfoo.so`.
if(NOT DEFINED HIPDNN_PLUGIN_ENGINE_SUBDIR)
    message(FATAL_ERROR "HIPDNN_PLUGIN_ENGINE_SUBDIR was not set by hipdnn_data_sdk package import.")
endif()

set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
)

message(STATUS "CK DSL provider plugin build directory: ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}")
message(STATUS "CK DSL provider plugin install directory: ${HIPDNN_INSTALL_PLUGIN_ENGINE_DIR}")
file(MAKE_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})

# Resolve absolute paths to the in-tree ck_dsl Python package and the
# provider-local ck_dsl_provider package, then bake them into a
# generated header that the embedded interpreter uses to prepend them
# to sys.path (PREP_FINDINGS §P-4).
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CkDslProviderPaths.cmake)
ck_dsl_provider_resolve_python_paths()

set(_ckDslPathsHeader "${CMAKE_CURRENT_BINARY_DIR}/include/ckdsl_provider_paths.h")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/templates/ckdsl_provider_paths.h.in"
    "${_ckDslPathsHeader}"
    @ONLY
)

add_subdirectory(src)

# Make the generated path header visible to every TU in the OBJECT
# library (and therefore to the static + shared downstream targets).
target_include_directories(ck_dsl_provider_impl
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

# Generate version.h on top of the plugin's OBJECT library so every
# translation unit sees CK_DSL_PROVIDER_VERSION_STRING.
ck_dsl_provider_generate_version_header(ck_dsl_provider_impl)

# Unit and integration tests for the provider. Guarded so packagers
# and the superbuild can opt out via -DCKDSLPROVIDER_SKIP_TESTS=ON.
# Both lanes share the GTest dependency fetched here.
#
# Test labels (see ../cmake/Tests.cmake):
#   * tests/  uses add_unit_test_target -> label "unit_test"
#     -> ninja ck-dsl-provider-unit-check
#   * integration_tests/ uses add_integration_test_target
#     -> label "integration_test"
#     -> ninja ck-dsl-provider-integration-check
# finalize_test_targets wires the prefixed ctest targets that the M1
# capstone (I-11) advertises.
if(NOT CKDSLPROVIDER_SKIP_TESTS)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/Dependencies.cmake)
    fetch_gtest_dependency()
    add_subdirectory(tests)
    add_subdirectory(integration_tests)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/Tests.cmake)
    finalize_test_targets("ck-dsl-provider")
endif()

if(NOT DEFINED HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR)
    message(FATAL_ERROR "HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR was not set by hipdnn_data_sdk package import.")
endif()

install(TARGETS ck_dsl_provider_plugin EXPORT ck-dsl-provider-plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)
