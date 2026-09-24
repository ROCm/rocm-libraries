# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(Origami VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(dependencies)

rocm_setup_version(VERSION ${PROJECT_VERSION})

option(ORIGAMI_BUILD_SHARED_LIBS "Build Origami libraries as shared vs static." ON)
option(ORIGAMI_ENABLE_PYTHON "Enable Python bindings." OFF)
option(ORIGAMI_BUILD_TESTING "Build tests." OFF)
option(ORIGAMI_ENABLE_FETCH "Auto-fetch dependencies with FetchContent" ON)

find_package(hip REQUIRED)

# Configure the Windows version resource (PE VERSIONINFO) embedded into
# origami.dll. This mirrors the sibling rocBLAS/hipBLAS/hipBLASLt libraries so
# the DLL reports file/product version metadata in its Windows file
# properties. A resource compiler is required to compile version.rc; the
# ROCm toolchain detects one (llvm-rc/rc) or sets CMAKE_RC_COMPILER to a
# NOTREQUIRED/NOTFOUND sentinel when none is available. When no real
# compiler is present, warn and skip the resource rather than fail
# configuration, so the DLL still builds (without version metadata) on
# machines lacking a resource compiler.
if(WIN32 AND CMAKE_RC_COMPILER AND NOT CMAKE_RC_COMPILER MATCHES "NOTREQUIRED|NOTFOUND")
    set(ORIGAMI_VERSION_COMMIT_ID "unknown")
    if(GIT_EXECUTABLE)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _origami_git_commit_id
            RESULT_VARIABLE _origami_git_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_origami_git_result EQUAL 0 AND _origami_git_commit_id)
            set(ORIGAMI_VERSION_COMMIT_ID "${_origami_git_commit_id}")
        endif()
    endif()
    string(TIMESTAMP ORIGAMI_COPYRIGHT_YEAR "%Y")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/version.rc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/version.rc"
        @ONLY)
    set(_origami_have_version_rc TRUE)
elseif(WIN32)
    message(WARNING
        "Origami: no Windows resource compiler (llvm-rc/rc) detected; "
        "origami.dll will be built WITHOUT VERSIONINFO metadata. Install "
        "the Windows SDK or make llvm-rc available (e.g. on PATH or in "
        "the ROCm llvm/bin directory) to embed version info.")
endif()

set(BUILD_SHARED_LIBS "${ORIGAMI_BUILD_SHARED_LIBS}")
add_library(origami)
if(WIN32 AND BUILD_SHARED_LIBS AND _origami_have_version_rc)
    target_sources(origami PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/version.rc")
endif()

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
if(NOT BUILD_SHARED_LIBS)
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

rocm_install(TARGETS origami)

install(TARGETS origami-headers
    EXPORT origami-targets
    FILE_SET HEADERS
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT devel
)

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
