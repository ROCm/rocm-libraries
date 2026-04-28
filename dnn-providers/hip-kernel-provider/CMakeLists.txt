# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/VersionUtils.cmake)
dnn_provider_setup_version(hip_kernel_provider ${CMAKE_CURRENT_LIST_DIR})

project(hip-kernel-provider VERSION ${HIP_KERNEL_PROVIDER_VERSION} LANGUAGES CXX)

if(NOT WIN32 AND DEFINED ROCM_PATH)
    # Only set if not already specified by user
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
    endif()
    message(STATUS "Setting CMAKE_INSTALL_PREFIX to ${CMAKE_INSTALL_PREFIX}")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
    CACHE BOOL "Add paths to linker search and installed rpath"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip
     ${HIP_DIR}/cmake
)

option(HIPKERNELPROVIDER_ENABLE_TESTS "Build tests" ON)
option(ENABLE_ASM_SDPA_ENGINE "Build the ASM SDPA engine" ON)

option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)
option(ENABLE_CLANG_FORMAT "Enable Clang-Format checks" ON)
if(WIN32)
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" OFF)
else()
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)
endif()
option(HIPKERNELPROVIDER_ENABLE_COVERAGE "Build with code coverage flags" OFF)

if(HIPKERNELPROVIDER_ENABLE_COVERAGE)
    message(VERBOSE "Enabling compile flags for coverage-mapping")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

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

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files;
# remove when fixed. Added to prevent issues with not finding hip cmake.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)

# hipdnn SDK dependencies - use targets directly if available (superbuild),
# otherwise find installed packages (true standalone)
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
include(ClangCheck)
include(ClangTidy)

# Setup testing dependencies
include(Dependencies)
include(Tests)

if(WIN32)
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "bin")
else()
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "lib")
endif()

set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
)

message(STATUS "Plugin build directory: ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}")
message(STATUS "Plugin install directory: ${HIPDNN_INSTALL_PLUGIN_ENGINE_DIR}")
file(MAKE_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})

# ============================================================================
# Kernel embedding (must come before src so KERNEL_GENERATED_SOURCES is set)
# ============================================================================
add_subdirectory(kernels)

# ============================================================================
# Plugin library targets and tests
# ============================================================================
add_subdirectory(src)

# ============================================================================
# Code Coverage
# ============================================================================
if(HIPKERNELPROVIDER_ENABLE_COVERAGE)
    findandcheckllvmtools()

    set(CODE_COVERAGE_IGNORE_REGEX ".*_deps.*|.*tests.*|.*integration_tests.*")
    set(CODE_COVERAGE_OBJECTS
        -object ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/libhip_kernel_provider.so
        -object ./${CMAKE_INSTALL_BINDIR}/hip_kernel_provider_tests
        -object ./${CMAKE_INSTALL_BINDIR}/hip_kernel_provider_integration_tests
    )

    # Creates a coverage target that merges profraw files and generates reports.
    #
    # @param TARGET_NAME name of the custom target to create
    # @param DEPENDS_TARGET target to depend on (empty string for no dependency)
    function(add_custom_coverage_target TARGET_NAME DEPENDS_TARGET)
        set(DEPENDS_CLAUSE "")
        if(NOT "${DEPENDS_TARGET}" STREQUAL "")
            set(DEPENDS_CLAUSE DEPENDS ${DEPENDS_TARGET})
        endif()

        # cmake-format: off
        add_custom_target(
            ${TARGET_NAME}
            COMMAND find . -name "*.profraw" -print0 | xargs -0 ${LLVM_PROFDATA_BINARY} merge
                    -sparse -o ./coverage-report/hip_kernel_provider.profdata
            COMMAND
                ${LLVM_COV_BINARY} report ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/hip_kernel_provider.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} >
                ./coverage-report/code_cov_hip_kernel_provider.report
            COMMAND
                ${LLVM_COV_BINARY} show -Xdemangler=${LLVM_CXXFILT_BINARY} ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/hip_kernel_provider.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} --format=html
                --output-dir=coverage-report
            COMMAND
                ${LLVM_COV_BINARY} export ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/hip_kernel_provider.profdata
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

    add_custom_coverage_target(code_coverage check)
    add_custom_coverage_target(unit-coverage unit-check)
    add_custom_coverage_target(integration-coverage integration-check)

    # Alias target
    add_custom_target(coverage DEPENDS code_coverage COMMENT "Alias for code_coverage")

    # Standalone coverage report from existing profdata (does not run tests)
    add_custom_coverage_target(current-coverage "")
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS hip_kernel_provider EXPORT hip-kernel-provider
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)

if(ENABLE_ASM_SDPA_ENGINE)
    install(DIRECTORY src/engines/asm_sdpa_engine/asm/asm_kernels/
            DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/asm_kernels
            FILES_MATCHING PATTERN "*.co")
endif()
