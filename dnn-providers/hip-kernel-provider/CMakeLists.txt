# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/VersionUtils.cmake)

dnn_provider_setup_version(hip_kernel_provider ${CMAKE_CURRENT_LIST_DIR})

project(hip-kernel-provider VERSION ${HIP_KERNEL_PROVIDER_VERSION} LANGUAGES CXX)

# shared/ctest is only present in a full rocm-libraries checkout; guard the check at the
# point of use so standalone / sparse checkouts that lack shared/ctest still build.
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
    set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${ROCM_PATH}/lib/cmake/hip ${HIP_DIR}/cmake
)

option(HIPKERNELPROVIDER_ENABLE_TESTS "Build tests" ON)
option(HIPKERNELPROVIDER_ENABLE_ROCKE "Build rocKE engine and smoke tests" OFF)
option(HIPDNN_ENABLE_KERNEL_INGESTOR "Enable the generic kernel ingestor" OFF)
option(ENABLE_ASM_SDPA_ENGINE "Build the ASM SDPA engine" ON)
option(ENABLE_HIP_MLOPS_ENGINE "Build the HIP kernel engine" ON)

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

if(ENABLE_ASM_SDPA_ENGINE)
    message(STATUS "ASM SDPA engine enabled")
else()
    message(STATUS "ASM SDPA engine disabled")
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    message(STATUS "HIP kernel engine enabled")
else()
    message(STATUS "HIP kernel engine disabled")
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
include(Sanitizers)

# Setup testing dependencies
include(Dependencies)
if(HIPKERNELPROVIDER_ENABLE_TESTS)
    fetch_gtest_dependency()
endif()
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
# YAML-based CTest categorization detection
# ============================================================================
# Hoisted here (rather than src/CMakeLists.txt) so it is available before
# add_subdirectory(rocke) below: rocke/platform/tests/CMakeLists.txt applies
# ROCKE_ENGINE_CTEST_CATEGORIES_YAML to its own pre-registered CTest tests in
# its own directory scope (CMake's set_property(TEST ...) predates 3.28's
# cross-directory support, so this project's 3.25.2 floor requires it run
# where the tests were add_test()'d).
if(HIPKERNELPROVIDER_ENABLE_TESTS)
    set(HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/HIP_MLOPS_ENGINE_test_categories.yaml")
    set(HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/HIP_MLOPS_ENGINE_test_categories_external.yaml")
    set(HIPKERNELPROVIDER_INTEGRATION_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/HIP_MLOPS_ENGINE_test_categories_integration.yaml")
    set(HIPKERNELPROVIDER_ASM_SDPA_INTEGRATION_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/ASM_SDPA_ENGINE_test_categories_integration.yaml")
    set(ROCKE_ENGINE_CTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/ROCKE_ENGINE_test_categories_external.yaml")
    set(HIPKERNELPROVIDER_CTEST_INSTALL_FILE "")
    set(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED FALSE)
    if(EXISTS "${HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML}"
       AND EXISTS "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"
       AND EXISTS "${HIPKERNELPROVIDER_INTEGRATION_CATEGORIES_YAML}"
       AND EXISTS "${HIPKERNELPROVIDER_ASM_SDPA_INTEGRATION_CATEGORIES_YAML}"
       AND EXISTS "${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"
       AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        set(DNN_PROVIDER_CTEST_CATEGORIES_YAML "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}")
        set(DNN_PROVIDER_TEST_CATEGORY_YAMLS
            "${HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML}"
            "${HIPKERNELPROVIDER_INTEGRATION_CATEGORIES_YAML}"
            "${HIPKERNELPROVIDER_ASM_SDPA_INTEGRATION_CATEGORIES_YAML}"
            "${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"
            "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"
        )
        set(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED TRUE)
        message(STATUS "hip-kernel-provider: YAML-based CTest categorization enabled")
    else()
        message(STATUS "hip-kernel-provider: shared/ctest or test category YAML not found, using legacy CTest labels")
    endif()
endif()

if(HIPKERNELPROVIDER_ENABLE_ROCKE)
    message(STATUS "rocKE features enabled")

    # rocke/ = platform (engine + build-time Python). The engine's installed
    # CTest entries stage into EXTERNAL_TEST_INSTALL_STAGING_hip_kernel_provider,
    # so add_subdirectory(rocke) must precede src/ below, which emits
    # bin/hip_kernel_provider/CTestTestfile.cmake.
    set(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR "hip_kernel_provider")
    set(ROCKE_INSTALL_STANDALONE_CTEST OFF)

    # TEMPORARY: do not install the rocKE Python CI tests (rocke_python_import_smoke,
    # rocke_golden_static, rocke_pytest) in provider mode.
    #
    # These three CTest entries import the `rocke` package from the test artifact.
    # ROCm/TheRock#6849 removed the "bin/hip_kernel_provider/rocke/**" artifact
    # include on the assumption that the staged rocKE wheels replaced it, but the
    # wheels are never staged (see below), so neither import source survives and
    # every rocKE Python entry fails with `ModuleNotFoundError: No module named
    # 'rocke'`. rocm-libraries builds this component with
    # HIPKERNELPROVIDER_ENABLE_ROCKE=ON, so the breakage lands on every
    # hipkernelprovider PR here while TheRock's own CI (flag defaults OFF) stays green.
    #
    # This matches what the ROCKE_INSTALL_PYTHON_TESTS option already documents for
    # provider mode; nothing was actually setting it. The C++ rocKE smoke tests are
    # unaffected and keep running.
    #
    # Revert this line once ROCm/TheRock#7145 has landed and the wheel-staging order
    # bug in rocke/CMakeLists.txt is fixed (ROCKE_BUILD_PYENV / ROCKE_WHEEL_DIR are
    # declared after add_subdirectory(platform), so the wheel install() never
    # registers). Tracked by ROCm/rocm-libraries#10497.
    set(ROCKE_INSTALL_PYTHON_TESTS OFF)

    add_subdirectory(rocke)
else()
    message(STATUS "rocKE features disabled")
endif()

add_subdirectory(src)

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    add_subdirectory(descriptor-packaging)
endif()

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
