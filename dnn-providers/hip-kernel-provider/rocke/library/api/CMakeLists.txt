# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/VersionUtils.cmake)
dnn_provider_setup_version(rocke_client ${CMAKE_CURRENT_LIST_DIR})

project(rocke-client VERSION ${ROCKE_CLIENT_VERSION} LANGUAGES CXX)

if(NOT WIN32 AND DEFINED ROCM_PATH)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
    endif()
    message(STATUS "Setting CMAKE_INSTALL_PREFIX to ${CMAKE_INSTALL_PREFIX}")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
    CACHE BOOL "Add paths to linker search and installed rpath"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake ${ROCM_PATH}/lib/cmake/hip ${HIP_DIR}/cmake
)

if(DEFINED HIPKERNELPROVIDER_ENABLE_TESTS)
    set(_ROCKE_CLIENT_ENABLE_TESTS_DEFAULT ${HIPKERNELPROVIDER_ENABLE_TESTS})
else()
    set(_ROCKE_CLIENT_ENABLE_TESTS_DEFAULT ON)
endif()

option(ROCKE_CLIENT_ENABLE_TESTS "Build rocke-client smoke tests" ${_ROCKE_CLIENT_ENABLE_TESTS_DEFAULT})
option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)
option(ENABLE_CLANG_FORMAT "Enable Clang-Format checks" ON)
if(WIN32)
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" OFF)
else()
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)
endif()

if(NOT DEFINED HIPDNN_WARNING_COMPILE_OPTIONS)
    list(
        PREPEND
        HIPDNN_WARNING_COMPILE_OPTIONS
        -Werror
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Winit-self
        -Wunreachable-code
        -Wno-error=unused-command-line-argument
        -Wswitch-default
    )
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

find_package(hip REQUIRED)

if(NOT TARGET hipdnn_data_sdk)
    find_package(hipdnn_data_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_flatbuffers_sdk)
    find_package(hipdnn_flatbuffers_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

include(CheckToolVersion)
include(ClangTidy)
include(Sanitizers)

if(ROCKE_CLIENT_ENABLE_TESTS)
    include(Dependencies)
    fetch_gtest_dependency()
    include(Tests)
endif()

if(WIN32)
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "bin")
else()
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "lib")
endif()

set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
)

message(STATUS "rocke-client plugin build directory: ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}")
message(STATUS "rocke-client plugin install directory: ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}")
file(MAKE_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})

add_subdirectory(src)

# WORKAROUND (superbuild only): scope clang-tidy to rocke's own headers.
#
# The rocke api is C++20 while the hipDNN components it includes (data /
# flatbuffers / plugin / test SDKs, frontend, backend) are still C++17. clang-tidy
# would otherwise demand C++20-only idioms (designated initializers, ranges,
# contains, requires) inside those C++17 headers and fail the -Werror tidy gate.
#
# This only happens because rocke is built as a CHILD of hip-kernel-provider in
# the superbuild, which has two effects a standalone rocke build does not:
#   1. hipDNN is an in-source peer, so its headers arrive as non-system `-I`
#      includes and clang-tidy analyzes them. A standalone build consumes hipDNN
#      via find_package (IMPORTED targets), whose includes are `-isystem` and are
#      skipped by clang-tidy automatically -- so this workaround is a no-op there.
#   2. The provider's ClangTidy.cmake forces `-config-file=<provider>/.clang-tidy`,
#      which stops clang-tidy walking the tree and so bypasses the rocke-local
#      .clang-tidy. That is why this scoping lives here in CMake (appended to the
#      tidy invocation) rather than in a rocke .clang-tidy file.
#
# `-header-filter` limits only which HEADERS report findings; every rocke .cpp is
# still fully analyzed. Remove this once rocke and hipDNN share a C++ standard.
if(ENABLE_CLANG_TIDY)
    foreach(_rocke_tidy_target rocke_client_impl rocke_client rocke_client_tests
                               rocke_client_integration_tests)
        if(TARGET ${_rocke_tidy_target})
            set_property(TARGET ${_rocke_tidy_target} APPEND PROPERTY
                         CXX_CLANG_TIDY "-header-filter=rocke/library/api")
        endif()
    endforeach()
endif()

install(TARGETS rocke_client EXPORT rocke-client
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)

if(ROCKE_CLIENT_ENABLE_TESTS)
    if(DEFINED ROCKE_CLIENT_PARENT_PROJECT)
        # Embedded inside hip-kernel-provider: finalize_test_targets is NOT called
        # here — the parent owns that call, and calling it twice creates duplicate
        # alias targets (check, unit-check, ...) and fails configure.
        #
        # Rename the installed binaries to match the parent provider's artifact
        # glob ("bin/<parent>*_test*") so TheRock's packaging captures them.
        # Without this they install as "bin/rocke_client_tests" and miss the glob.
        # Use EXTERNAL_TEST_INSTALL_STAGING rather than _TEST_TARGETS so the
        # CTestTestfile entries reference the renamed binary paths, not the raw
        # CMake target names.
        string(REPLACE "-" "_" _rocke_client_parent_prefix "${ROCKE_CLIENT_PARENT_PROJECT}")
        set_target_properties(rocke_client_tests PROPERTIES
            OUTPUT_NAME "${_rocke_client_parent_prefix}_rocke_client_tests_test")
        set_target_properties(rocke_client_integration_tests PROPERTIES
            OUTPUT_NAME "${_rocke_client_parent_prefix}_rocke_client_integration_tests_test")
        set_property(GLOBAL APPEND_STRING PROPERTY
            "EXTERNAL_TEST_INSTALL_STAGING_${_rocke_client_parent_prefix}"
            "\n# rocke-client installed tests\n"
            "add_test(rocke_client_tests \"../${_rocke_client_parent_prefix}_rocke_client_tests_test\")\n"
            "set_tests_properties(rocke_client_tests PROPERTIES LABELS \"unit_test;quick\")\n"
            "add_test(rocke_client_integration_tests \"../${_rocke_client_parent_prefix}_rocke_client_integration_tests_test\")\n"
            "set_tests_properties(rocke_client_integration_tests PROPERTIES LABELS \"integration_test;quick\")\n"
        )
    else()
        # Standalone build: rocke-client is the top-level project, so it owns
        # finalize_test_targets and its own installed CTestTestfile under
        # bin/rocke-client/ (reached from ctest via "../<binary>"). Provider-mode
        # CTest staging is handled by the embedded branch above; this branch only
        # emits the standalone metadata.
        finalize_test_targets("${PROJECT_NAME}")

        option(ROCKE_INSTALL_STANDALONE_CTEST "Install standalone rocke-client CTest metadata" ON)
        if(ROCKE_INSTALL_STANDALONE_CTEST)
            set(_ROCKE_CLIENT_CTEST_ENTRIES "")
            string(APPEND _ROCKE_CLIENT_CTEST_ENTRIES
                   "add_test(rocke_client_tests \"../rocke_client_tests\")\n")
            string(APPEND _ROCKE_CLIENT_CTEST_ENTRIES
                   "set_tests_properties(rocke_client_tests PROPERTIES LABELS \"unit_test;quick\")\n")
            string(APPEND _ROCKE_CLIENT_CTEST_ENTRIES
                   "add_test(rocke_client_integration_tests \"../rocke_client_integration_tests\")\n")
            string(APPEND _ROCKE_CLIENT_CTEST_ENTRIES
                   "set_tests_properties(rocke_client_integration_tests PROPERTIES LABELS \"integration_test;quick\")\n")

            set(_ROCKE_CLIENT_INSTALLED_CTEST "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")
            file(WRITE "${_ROCKE_CLIENT_INSTALLED_CTEST}"
                 "# Autogenerated CTestTestfile for installed rocke-client tests\n")
            file(APPEND "${_ROCKE_CLIENT_INSTALLED_CTEST}"
                 "# Generated by ${PROJECT_NAME} build system\n\n")
            file(APPEND "${_ROCKE_CLIENT_INSTALLED_CTEST}" "${_ROCKE_CLIENT_CTEST_ENTRIES}")
            install(FILES "${_ROCKE_CLIENT_INSTALLED_CTEST}"
                    DESTINATION ${CMAKE_INSTALL_BINDIR}/rocke-client
                    RENAME CTestTestfile.cmake)
        endif()
    endif()
endif()
