# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

# Used by the shared CTest test categorisation helpers
# (shared/ctest/TestCategories.cmake) to locate the parser script.
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
    set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

# hipDNN dependency versions to use
set(_HIPDNN_DEFAULT_FLATBUFFERS_VERSION "25.9.23")

set(HIPDNN_SPDLOG_VERSION "1.15.3" CACHE STRING "Version of spdlog to use")
set(HIPDNN_FLATBUFFERS_VERSION "${_HIPDNN_DEFAULT_FLATBUFFERS_VERSION}" CACHE STRING "Version of flatbuffers to use")
set(HIPDNN_NLOHMANN_JSON_VERSION "3.12.0" CACHE STRING "Version of nlohmann_json to use")
set(HIPDNN_GTEST_VERSION "1.16.0" CACHE STRING "Version of googletest to use")
set(HIPDNN_NANOBIND_VERSION "2.12.0" CACHE STRING "Version of nanobind to use")
set(HIPDNN_TSL_ROBIN_MAP_VERSION "1.4.1" CACHE STRING "Version of tsl-robin-map to use")

option(HIPDNN_SKIP_JSON_LIB "Defines whether SDK public headers should not include nlohmann/json.hpp." OFF)

# Status message
if(HIPDNN_SKIP_JSON_LIB)
    message(STATUS "hipDNN: JSON serialization disabled in SDK public headers")
else()
    message(STATUS "hipDNN: JSON serialization enabled in SDK public headers (requires nlohmann_json)")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_definitions(__HIP_PLATFORM_AMD__)

if(NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ClangToolChain.cmake"
        CACHE FILEPATH "Toolchain file"
    )
endif()
set(HIPDNN_VERSION "0.0.1")

project(hipdnn VERSION ${HIPDNN_VERSION} LANGUAGES CXX)

# ----------------------------------------------------------------------------
# TheRock Feature Flags
#
# See: https://github.com/ROCm/TheRock/blob/main/docs/development/flags.md
#
# These flags are injected during the project() call above via
# CMAKE_PROJECT_TOP_LEVEL_INCLUDES (TheRock's build system), which only
# populates the cache during that call. They must therefore be declared AFTER
# project(): declared before, option() captures the pre-injection default
# (typically OFF) and the injected value is silently ignored, leaving the
# feature compiled out despite the flag being set.
# ----------------------------------------------------------------------------
option(HIPDNN_ENABLE_SDPA "Enable SDPA (Scaled Dot-Product Attention) support" OFF)
if(HIPDNN_ENABLE_SDPA)
    message(STATUS "hipDNN: SDPA support enabled")
    add_compile_definitions(HIPDNN_ENABLE_SDPA)
else()
    message(STATUS "hipDNN: SDPA support disabled")
endif()

# Gates install of the cuDNN-compatibility shim headers and build/run of its
# tests (RFC 0012; ALMIOPEN-2034). The shim sources always exist in the tree;
# this option controls only whether they are installed and tested.
option(HIPDNN_ENABLE_CUDNN_COMPATIBILITY
    "Enable the cuDNN compatibility shim (headers + tests)" OFF
)
if(HIPDNN_ENABLE_CUDNN_COMPATIBILITY)
    message(STATUS "hipDNN: cuDNN compatibility shim enabled")
else()
    message(STATUS "hipDNN: cuDNN compatibility shim disabled")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/CheckPreCommit.cmake)

if(NOT WIN32)
    if(CMAKE_GENERATOR STREQUAL "Ninja")
        message(STATUS "The CMake generator is Ninja.")
    else()
        message(WARNING "The CMake generator is not Ninja. Ninja is preferred over Make.")
    endif()
endif()

# Only set if not already specified by user
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND ROCM_PATH)
    set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
    message(STATUS "CMAKE_INSTALL_PREFIX set using ROCM_PATH: ${CMAKE_INSTALL_PREFIX}")
elseif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND ROCM_CMAKE_PATH)
    set(CMAKE_INSTALL_PREFIX "${ROCM_CMAKE_PATH}" CACHE PATH "Install path prefix" FORCE)
    message(STATUS "CMAKE_INSTALL_PREFIX set using ROCM_CMAKE_PATH: ${CMAKE_INSTALL_PREFIX}")
else()
    message(STATUS "CMAKE_INSTALL_PREFIX set to default: ${CMAKE_INSTALL_PREFIX}")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

if(WIN32)
    set(HIPDNN_PLUGIN_ROOTDIR "${CMAKE_INSTALL_BINDIR}" CACHE STRING "Root location for plugins")
    set(HIPDNN_PLUGIN_FULL_ROOTDIR "${CMAKE_INSTALL_FULL_BINDIR}" CACHE STRING
                                                                        "Root location for plugins"
    )
else()
    set(HIPDNN_PLUGIN_ROOTDIR "${CMAKE_INSTALL_LIBDIR}" CACHE STRING "Root location for plugins")
    set(HIPDNN_PLUGIN_FULL_ROOTDIR "${CMAKE_INSTALL_FULL_LIBDIR}" CACHE STRING
                                                                        "Root location for plugins"
    )
endif()

set(HIPDNN_PLUGIN_ENGINE_SUBDIR "hipdnn_plugins/engines"
    CACHE STRING "Subdirectory for official hipDNN engine plugins"
)
set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
    CACHE PATH "Build directory for official hipDNN engine plugins"
)
set(HIPDNN_FULL_INSTALL_PLUGIN_ENGINE_DIR
    "${HIPDNN_PLUGIN_FULL_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
    CACHE
        STRING
        "Full install directory for official hipDNN engine plugins. Uses the CMAKE_INSTALL_PREFIX set at configure time (defaults to /opt/rocm/)."
)
set(HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR
    "${HIPDNN_PLUGIN_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
    CACHE STRING "Relative install directory for official hipDNN engine plugins"
)
set(HIPDNN_PLUGIN_HEURISTIC_SUBDIR "hipdnn_plugins/heuristics"
    CACHE STRING "Subdirectory for official hipDNN heuristic plugins"
)
set(HIPDNN_BUILD_PLUGIN_HEURISTIC_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ROOTDIR}/${HIPDNN_PLUGIN_HEURISTIC_SUBDIR}"
    CACHE PATH "Build directory for official hipDNN heuristic plugins"
)
set(HIPDNN_FULL_INSTALL_PLUGIN_HEURISTIC_DIR
    "${HIPDNN_PLUGIN_FULL_ROOTDIR}/${HIPDNN_PLUGIN_HEURISTIC_SUBDIR}"
    CACHE
        STRING
        "Full install directory for official hipDNN heuristic plugins. Uses the CMAKE_INSTALL_PREFIX set at configure time (defaults to /opt/rocm/)."
)
set(HIPDNN_RELATIVE_INSTALL_PLUGIN_HEURISTIC_DIR
    "${HIPDNN_PLUGIN_ROOTDIR}/${HIPDNN_PLUGIN_HEURISTIC_SUBDIR}"
    CACHE STRING "Relative install directory for official hipDNN heuristic plugins"
)
set(HIPDNN_TEST_PLUGIN_DIR "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ROOTDIR}/test_plugins"
    CACHE INTERNAL "Build directory for test plugins"
)
set(HIPDNN_TEST_PLUGIN_INSTALL_DIR "${HIPDNN_PLUGIN_ROOTDIR}/test_plugins"
    CACHE INTERNAL "Install directory for test plugins"
)

# Migrate HIP_DNN_BUILD_BACKEND -> HIPDNN_BUILD_BACKEND
if(DEFINED HIP_DNN_BUILD_BACKEND)
    set(HIPDNN_BUILD_BACKEND ${HIP_DNN_BUILD_BACKEND} CACHE BOOL "Build the backend code" FORCE)
    unset(HIP_DNN_BUILD_BACKEND CACHE)
    message(DEPRECATION "HIP_DNN_BUILD_BACKEND has been renamed to HIPDNN_BUILD_BACKEND. Please update your build scripts to use -DHIPDNN_BUILD_BACKEND instead.")
endif()
option(HIPDNN_BUILD_BACKEND "Build the backend code" ON)

# Migrate HIP_DNN_BUILD_FRONTEND -> HIPDNN_BUILD_FRONTEND
if(DEFINED HIP_DNN_BUILD_FRONTEND)
    set(HIPDNN_BUILD_FRONTEND ${HIP_DNN_BUILD_FRONTEND} CACHE BOOL "Build the frontend code" FORCE)
    unset(HIP_DNN_BUILD_FRONTEND CACHE)
    message(DEPRECATION "HIP_DNN_BUILD_FRONTEND has been renamed to HIPDNN_BUILD_FRONTEND. Please update your build scripts to use -DHIPDNN_BUILD_FRONTEND instead.")
endif()
option(HIPDNN_BUILD_FRONTEND "Build the frontend code" ON)

# Migrate HIP_DNN_GENERATE_SDK_HEADERS -> HIPDNN_GENERATE_SDK_HEADERS
if(DEFINED HIP_DNN_GENERATE_SDK_HEADERS)
    set(HIPDNN_GENERATE_SDK_HEADERS ${HIP_DNN_GENERATE_SDK_HEADERS} CACHE BOOL "Automatically generates sdk headers from schema during build" FORCE)
    unset(HIP_DNN_GENERATE_SDK_HEADERS CACHE)
    message(DEPRECATION "HIP_DNN_GENERATE_SDK_HEADERS has been renamed to HIPDNN_GENERATE_SDK_HEADERS. Please update your build scripts to use -DHIPDNN_GENERATE_SDK_HEADERS instead.")
endif()
option(HIPDNN_GENERATE_SDK_HEADERS "Automatically generates sdk headers from schema during build" ON)

# Migrate HIP_DNN_SKIP_TESTS -> HIPDNN_SKIP_TESTS
if(DEFINED HIP_DNN_SKIP_TESTS)
    set(HIPDNN_SKIP_TESTS ${HIP_DNN_SKIP_TESTS} CACHE BOOL "Skips building all tests" FORCE)
    unset(HIP_DNN_SKIP_TESTS CACHE)
    message(DEPRECATION "HIP_DNN_SKIP_TESTS has been renamed to HIPDNN_SKIP_TESTS. Please update your build scripts to use -DHIPDNN_SKIP_TESTS instead.")
endif()
option(HIPDNN_SKIP_TESTS "Skips building all tests" OFF)

option(HIPDNN_BUILD_PYTHON_BINDINGS "Build Python bindings (requires Python and nanobind)" OFF)

option(HIPDNN_ENABLE_COVERAGE "Build with code coverage flags" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)
option(BUILD_THREAD_SANITIZER "Build with Thread Sanitizer enabled" OFF)
option(ENABLE_CLANG_FORMAT "Enable Clang-Format checks" ON)
if(WIN32)
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" OFF)
else()
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)
endif()

if(ENABLE_CLANG_TIDY)
    if(WIN32)
        message(WARNING "Clang-Tidy enabled on Windows - this significantly increases build time.")
    endif()
else()
    message(WARNING "Clang-Tidy is disabled - some build issues may be missed.")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL
                                                 "Add paths to linker search and installed rpath"
)

include(cmake/Dependencies.cmake)
include(cmake/VersionUtils.cmake)
include(cmake/CheckToolVersion.cmake)
include(cmake/ClangCheck.cmake)
include(cmake/ClangTidy.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/Tests.cmake)
include(cmake/TestPluginNames.cmake)
include(cmake/Spdlog.cmake)

# Add global compile options
set(EXTRA_COMPILE_OPTIONS)
set(EXTRA_LINK_OPTIONS)
if(HIPDNN_ENABLE_COVERAGE)
    message(VERBOSE "Enabling compile flags for coverage-mapping")
    list(PREPEND EXTRA_COMPILE_OPTIONS -fprofile-instr-generate -fcoverage-mapping)
    list(PREPEND EXTRA_LINK_OPTIONS -fprofile-instr-generate -fcoverage-mapping)
endif()

list(
    PREPEND
    HIPDNN_WARNING_COMPILE_OPTIONS
    -Werror # Treat all warnings as errors
    -Wall # Enable most common warnings
    -Wextra # Enable additional warnings not covered by -Wall
    -Wpedantic # Enforce strict ISO C++ compliance
    -Wshadow # Warn about variable shadowing
    -Wnon-virtual-dtor # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast # Warn about C-style casts
    -Wcast-align # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual # Warn if a base class function is hidden by a derived class function with
                         # the same name
    -Wconversion # Warn about implicit type conversions that may alter a value
    -Wsign-conversion # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference # Warn about dereferencing null pointers
    -Wdouble-promotion # Warn when a float is implicitly promoted to a double
    -Wformat=2 # Enable stricter format string checks
    -Winit-self # Warn about variables initialized with itself
    -Wunreachable-code # Warn about unreachable code
    $<$<CXX_COMPILER_ID:Clang>:-Wno-error=unused-command-line-argument> # Clang: disable error for unused command line arguments
    -Wno-error=null-dereference # Downgrade null-dereference to warning (false positives with flatbuffers in backend/data_sdk)
    -Wswitch-default # Warn if a switch statement does not have a default case
)

add_compile_options(${EXTRA_COMPILE_OPTIONS})
add_link_options(${EXTRA_LINK_OPTIONS})

# Disable RTTI in production code and RTTI-free test targets.
# Public headers must not use typeid, dynamic_cast, or dynamic_pointer_cast.
set(HIPDNN_NO_RTTI_OPTIONS
    $<$<CXX_COMPILER_ID:MSVC>:/GR->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-rtti>
)

add_subdirectory(data_sdk)
add_subdirectory(flatbuffers_sdk)

if(HIPDNN_BUILD_BACKEND)
    add_subdirectory(backend)
endif()

if(HIPDNN_BUILD_FRONTEND)
    add_subdirectory(frontend)
endif()

add_subdirectory(test_sdk)
add_subdirectory(plugin_sdk)

add_subdirectory(tests)
add_subdirectory(tools)

if(HIPDNN_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

if(NOT HIPDNN_SKIP_TESTS)
    # Keep this after all build folders have been added so they have a chance to register their tests
    # using add_*_test_target()
    finalize_test_targets("${PROJECT_NAME}")
endif()

# keep this after all the targets have been added that could use the HIP language compiler.
add_clang_tidy_custom_target()

# Doxygen documentation target (standalone builds only to avoid target name
# collisions with other projects in the superbuild)
if(PROJECT_IS_TOP_LEVEL)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        add_custom_target(
            doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating Doxygen documentation"
            VERBATIM
            USES_TERMINAL
        )
    else()
        add_custom_target(
            doxygen
            COMMAND ${CMAKE_COMMAND} -E echo
                    "Error: Doxygen not found. Please install it (e.g. sudo apt-get install doxygen)."
            COMMAND ${CMAKE_COMMAND} -E false
            COMMENT "Doxygen not found"
            VERBATIM
            USES_TERMINAL
        )
    endif()
endif()

if(HIPDNN_ENABLE_COVERAGE)
    findandcheckllvmtools()

    set(CODE_COVERAGE_IGNORE_REGEX
        ".*deps.*|.*tests.*|.*data_objects.*|.*HipErrorHandler.*|.*/test_sdk/.*/Mock[^/]*"
    )
    set(CODE_COVERAGE_OBJECTS
        -object
        ./${CMAKE_INSTALL_LIBDIR}/libhipdnn_backend.so
        -object
        ./${CMAKE_INSTALL_BINDIR}/hipdnn_backend_tests
        -object
        ./${CMAKE_INSTALL_BINDIR}/hipdnn_frontend_tests
        -object
        ./${CMAKE_INSTALL_BINDIR}/hipdnn_public_backend_tests
        -object
        ./${CMAKE_INSTALL_BINDIR}/hipdnn_data_sdk_tests
        -object
        ./${CMAKE_INSTALL_BINDIR}/hipdnn_flatbuffers_sdk_tests
        -object
        ./${CMAKE_INSTALL_BINDIR}/hipdnn_plugin_sdk_tests
        -object
        ./${CMAKE_INSTALL_BINDIR}/hipdnn_test_sdk_tests
    )

    # Helper function to create a code coverage target
    # ~~~
    # Parameters:
    #   TARGET_NAME - Name of the coverage target to create
    #   DEPENDS_TARGET - The target that must be built before generating
    #                    coverage (required, use "" for no dependency)
    # ~~~
    function(add_custom_coverage_target TARGET_NAME DEPENDS_TARGET)
        # Build the DEPENDS clause
        set(DEPENDS_CLAUSE "")
        if(NOT "${DEPENDS_TARGET}" STREQUAL "")
            set(DEPENDS_CLAUSE DEPENDS ${DEPENDS_TARGET})
        endif()

        # cmake-format: off
        add_custom_target(
            ${TARGET_NAME}
            COMMAND find . -name "*.profraw" -print0 | xargs -0 ${LLVM_PROFDATA_BINARY} merge
                    -sparse -o ./coverage-report/hipdnn.profdata
            COMMAND
                ${LLVM_COV_BINARY} report ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/hipdnn.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} >
                ./coverage-report/code_cov_hipdnn.report
            COMMAND
                ${LLVM_COV_BINARY} show -Xdemangler=${LLVM_CXXFILT_BINARY} ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/hipdnn.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} --format=html
                --output-dir=coverage-report
            COMMAND
                ${LLVM_COV_BINARY} export ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/hipdnn.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} --format=lcov >
                ./coverage-report/coverage.info
            ${DEPENDS_CLAUSE}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
            VERBATIM
            USES_TERMINAL
        )
        # cmake-format: on
    endfunction()

    if(HIPDNN_SKIP_TESTS)
        message(
            WARNING
                "\nThe code coverage targets depend on test targets but test targets don't exist because HIPDNN_SKIP_TESTS is enabled."
        )
    else()
        add_custom_coverage_target(code_coverage check) # code_coverage depends on 'check'
        add_custom_coverage_target(unit-coverage unit-check) # unit-coverage depends on 'unit-check'
        add_custom_coverage_target(integration-coverage integration-check) # ... 'integration-check'

        # Make alias to target coverage
        add_custom_target(
            coverage DEPENDS code_coverage COMMENT "Alias for code_coverage to match mathci"
        )
    endif()

    # Create a coverage target that does not depend on any other target and will use the coverage
    # data that already exists on disk from prior / manual target runs performed before this target
    # is run.
    add_custom_coverage_target(current-coverage "")

endif()

if(NOT HIPDNN_SKIP_TESTS)
    install_hipdnn_ctest_files()
endif()
