# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/VersionUtils.cmake)
dnn_provider_setup_version(hipblaslt_provider ${CMAKE_CURRENT_LIST_DIR})

project(hipblaslt-provider VERSION ${HIPBLASLT_PROVIDER_VERSION} LANGUAGES CXX)

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
option(HIPBLASLTPROVIDER_ENABLE_MX_GEMM
    "Enable MX (microscaling) data-type support for GEMM" OFF
)
if(HIPBLASLTPROVIDER_ENABLE_MX_GEMM)
    message(STATUS "hipblaslt-provider: MX GEMM support enabled")
    add_compile_definitions(HIPBLASLTPROVIDER_ENABLE_MX_GEMM)
else()
    message(STATUS "hipblaslt-provider: MX GEMM support disabled")
endif()

# Root of the rocm-libraries monorepo (for shared infrastructure like CTest categories).
# Used by the YAML-based test categorization wiring further down. Guarded with an EXISTS
# check at the point of use so standalone / sparse checkouts that lack shared/ctest still build.
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
    set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

if(NOT WIN32)
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

option(HIPDNN_SKIP_TESTS "Skip building tests" OFF)
option(ENABLE_CLANG_FORMAT "Enable Clang-Format checks" ON)
if(NOT ENABLE_CLANG_FORMAT)
    message(WARNING "Clang-Format checks are disabled. CI requires passing clang-format checks.")
endif()
if(WIN32)
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" OFF)
else()
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)
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
    -Woverloaded-virtual # Warn if a base class function is hidden by a derived class function
                         # with the same name
    -Wconversion # Warn about implicit type conversions that may alter a value
    -Wsign-conversion # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference # Warn about dereferencing null pointers
    -Wdouble-promotion # Warn when a float is implicitly promoted to a double
    -Wformat=2 # Enable stricter format string checks
    -Winit-self # Warn about variables initialized with itself
    -Wunreachable-code # Warn about unreachable code
    -Wno-error=unused-command-line-argument # Disable error for unused command line arguments
    -Wswitch-default # Warn if a switch statement does not have a default case
    -Wno-ignored-attributes # disable warning about unused declspec(dllexport)
)

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files;
# remove when fixed. Added to prevent issues with not finding hip cmake.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

find_package(hip REQUIRED)

# SDK dependencies - use targets directly in superbuild, find_package otherwise
if(NOT TARGET hipdnn_data_sdk)
    find_package(hipdnn_data_sdk CONFIG REQUIRED)
endif()

if(NOT TARGET hipdnn_flatbuffers_sdk)
    find_package(hipdnn_flatbuffers_sdk CONFIG REQUIRED)
endif()

if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

include(ClangTidy)

# Sanitizer support (Address Sanitizer, etc.)
include(Sanitizers)

# Setup testing dependencies, tests & clang format
include(Dependencies)
if (NOT HIPDNN_SKIP_TESTS)
    fetch_gtest_dependency()
    set(SKIP_TEST_NAME_VALIDATION ON)
    include(Tests)
endif()
# TODO: Remove this clang-format probe once TheRock's ml-libs/CMakeLists.txt passes
# -DENABLE_CLANG_FORMAT=OFF for hipblasltprovider (as it already does for miopenprovider).
# The common ClangCheck.cmake calls findAndCheckClangFormat() which FATAL_ERRORs when
# clang-format is missing. TheRock's build container does not provide clang-format, so we
# probe for it here and skip the include if absent. Other providers (miopen, hip-kernel,
# integration-tests) rely on TheRock passing -DENABLE_CLANG_FORMAT=OFF instead.
if(ENABLE_CLANG_FORMAT)
    find_program(_HIPBLASLT_CLANG_FORMAT_PROBE NAMES clang-format-18 clang-format)
    if(_HIPBLASLT_CLANG_FORMAT_PROBE)
        include(ClangCheck)
    else()
        message(WARNING "ENABLE_CLANG_FORMAT=ON but clang-format not found; skipping format targets")
    endif()
endif()

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

find_package(hipblaslt CONFIG REQUIRED)

add_library(
    hipblaslt_plugin_private
    engines/HipblasltEngine.cpp
    EngineManager.cpp
    HipblasltHandleFactory.cpp
    HipblasltContainer.cpp
    HipblasltPlugin.cpp
    HipblasltUtils.cpp
    engines/plans/HipblasltMatmulPlan.cpp
    engines/plans/HipblasltMatmulPlanBuilder.cpp
    $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_MX_GEMM}>:engines/plans/HipblasltMxMatmulPlan.cpp>
    $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_MX_GEMM}>:engines/plans/HipblasltMxMatmulPlanBuilder.cpp>
    HipblasltMatmulDesc.cpp
    HipblasltMatrixLayout.cpp
    HipblasltMatrixTransformDesc.cpp
)
target_include_directories(hipblaslt_plugin_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
dnn_provider_generate_version_header(hipblaslt_provider hipblaslt_plugin_private ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(hipblaslt_plugin_private PUBLIC roc::hipblaslt hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hip::host)

target_compile_options(hipblaslt_plugin_private PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
clang_tidy_check(hipblaslt_plugin_private)

add_library(hipblaslt_plugin SHARED HipblasltPluginPublic.cpp)
target_link_libraries(hipblaslt_plugin PRIVATE hipblaslt_plugin_private)
clang_tidy_check(hipblaslt_plugin)

if(NOT WIN32)
    target_link_libraries(
        hipblaslt_plugin PRIVATE "-Xlinker --exclude-libs=ALL"
    ) # Hide symbols from linked static libs
endif()

target_compile_options(hipblaslt_plugin PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    hipblaslt_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

if(NOT HIPDNN_SKIP_TESTS)
    # Optional: YAML-driven CTest test categorization for TheRock CI's test_runner.py.
    # Must be configured before add_subdirectory(tests)/add_subdirectory(integration_tests)
    # below: add_unit_test_target()/add_integration_test_target() check
    # DNN_PROVIDER_CTEST_CATEGORIES_YAML at registration time and skip their
    # legacy, unlabeled ctest entry in favor of the tiered suites generated
    # further down. Skipped when shared/ctest is not present (e.g. standalone
    # hipblaslt-provider or a sparse checkout that doesn't include
    # rocm-libraries/shared/), in which case we fall back to the project's own
    # install_provider_ctest_files() below.
    set(HIPBLASLTPROVIDER_GTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
    set(HIPBLASLTPROVIDER_CTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_external.yaml")
    set(HIPBLASLTPROVIDER_INTEGRATION_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_integration.yaml")
    set(HIPBLASLTPROVIDER_CTEST_INSTALL_FILE "")
    set(HIPBLASLTPROVIDER_YAML_CATEGORIZATION_ENABLED FALSE)
    if(EXISTS "${HIPBLASLTPROVIDER_GTEST_CATEGORIES_YAML}"
       AND EXISTS "${HIPBLASLTPROVIDER_CTEST_CATEGORIES_YAML}"
       AND EXISTS "${HIPBLASLTPROVIDER_INTEGRATION_CATEGORIES_YAML}"
       AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        set(DNN_PROVIDER_CTEST_CATEGORIES_YAML "${HIPBLASLTPROVIDER_CTEST_CATEGORIES_YAML}")
        set(DNN_PROVIDER_TEST_CATEGORY_YAMLS
            "${HIPBLASLTPROVIDER_GTEST_CATEGORIES_YAML}"
            "${HIPBLASLTPROVIDER_INTEGRATION_CATEGORIES_YAML}"
            "${HIPBLASLTPROVIDER_CTEST_CATEGORIES_YAML}"
        )
        set(HIPBLASLTPROVIDER_YAML_CATEGORIZATION_ENABLED TRUE)
        message(STATUS "hipblaslt-provider: YAML-based CTest categorization enabled")
    else()
        message(STATUS "hipblaslt-provider: shared/ctest or test category YAML not found, using legacy CTest labels")
    endif()

    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    if(NOT TARGET hipdnn_integration_tests)
        find_package(hipdnn_integration_tests CONFIG QUIET)
    endif()

    set(DNN_PROVIDER_TEST_RUN_DEPENDS hipblaslt_plugin_tests hipblaslt_plugin_integration_tests)
    if(TARGET hipdnn_integration_tests)
        list(APPEND DNN_PROVIDER_TEST_RUN_DEPENDS hipblaslt_plugin hipdnn_integration_tests)
    endif()

    # Keep this after all build folders have been added so they have a chance to register their tests
    # using add_*_test_target()
    # Finalize test targets with prefix to avoid collisions in superbuild mode
    finalize_test_targets("${PROJECT_NAME}")

    if(HIPBLASLTPROVIDER_YAML_CATEGORIZATION_ENABLED)
        enable_testing()
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
        # Install-time CTestTestfile uses relative paths so `ctest` can run from the
        # installed location (bin/hipblaslt_plugin/) and reach ../hipblaslt_plugin_tests etc.
        set(HIPBLASLTPROVIDER_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
        file(WRITE "${HIPBLASLTPROVIDER_CTEST_INSTALL_FILE}"
[=[
# This is a test file generated by hipblaslt-provider for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
        )
        # On ASAN builds, HIPDNN_TEST_MIOPEN_CACHE_DIR is set (see Sanitizers.cmake) and the generated
        # suites require the cache-clearing fixture; empty otherwise.
        set(_miopen_cache_fixture "")
        if(HIPDNN_TEST_MIOPEN_CACHE_DIR)
            set(_miopen_cache_fixture hipdnn_clear_miopen_test_cache)
        endif()
        # The GTest YAML applies to both test binaries; each call appends its category suites
        # (e.g. hipblaslt_plugin_tests_quick_suite, hipblaslt_plugin_integration_tests_quick_suite)
        # to the same install-time CTestTestfile.cmake. ENVIRONMENT forwards the CMake-side
        # TEST_ENVIRONMENT (ASAN symbolizer path, coverage LLVM_PROFILE_FILE, set by
        # dnn-providers/cmake/Sanitizers.cmake / _build_test_environment_list_internal) to
        # every generated suite; without it those suites run unsanitized/uninstrumented.
        apply_test_category_labels(
            hipblaslt_plugin_tests
            "${HIPBLASLTPROVIDER_GTEST_CATEGORIES_YAML}"
            "${PROJECT_BINARY_DIR}/staging"
            "${HIPBLASLTPROVIDER_CTEST_INSTALL_FILE}"
            ENVIRONMENT ${TEST_ENVIRONMENT}
            ENVIRONMENT_MODIFICATION ${TEST_ENVIRONMENT_MODIFICATION}
            FIXTURES_REQUIRED ${_miopen_cache_fixture}
        )
        apply_test_category_labels(
            hipblaslt_plugin_integration_tests
            "${HIPBLASLTPROVIDER_GTEST_CATEGORIES_YAML}"
            "${PROJECT_BINARY_DIR}/staging"
            "${HIPBLASLTPROVIDER_CTEST_INSTALL_FILE}"
            ENVIRONMENT ${TEST_ENVIRONMENT}
            ENVIRONMENT_MODIFICATION ${TEST_ENVIRONMENT_MODIFICATION}
            FIXTURES_REQUIRED ${_miopen_cache_fixture}
        )
    endif()

    set(_external_category_args "")
    if(HIPBLASLTPROVIDER_YAML_CATEGORIZATION_ENABLED)
        list(APPEND _external_category_args
            TEST_CATEGORIES_YAML "${HIPBLASLTPROVIDER_INTEGRATION_CATEGORIES_YAML}"
            INSTALL_TEST_FILE "${HIPBLASLTPROVIDER_CTEST_INSTALL_FILE}"
            TEST_NAME_PREFIX ${PROJECT_NAME}-external-integration
            ENVIRONMENT ${TEST_ENVIRONMENT}
            ENVIRONMENT_MODIFICATION ${TEST_ENVIRONMENT_MODIFICATION}
            FIXTURES_REQUIRED ${_miopen_cache_fixture}
        )
    endif()

    if(TARGET hipdnn_integration_tests)
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-external-integration-check
            PLUGIN_TARGET  hipblaslt_plugin
            ENGINE_NAME    HIPBLASLT_ENGINE
            INSTALL_SUBDIR hipblaslt_plugin
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/config/HIPBLASLT_ENGINE.toml
            GTEST_FILTER   "*Matmul*"
            ${_external_category_args}
        )
        message(STATUS "Created ${PROJECT_NAME}-external-integration-check target")

        # This is a pre-registered CTest test (not a GTest binary), so it is
        # labeled via apply_ctest_category_labels()/test_categories_external.yaml.
        if(HIPBLASLTPROVIDER_YAML_CATEGORIZATION_ENABLED)
            if(HIPBLASLTPROVIDER_CTEST_INSTALL_FILE)
                # Fold the staged add_test() into the labeled file before labeling;
                # apply_ctest_category_labels() scans the file's add_test() names.
                get_property(_external_staging GLOBAL
                    PROPERTY "EXTERNAL_TEST_INSTALL_STAGING_hipblaslt_plugin"
                )
                if(_external_staging)
                    file(APPEND "${HIPBLASLTPROVIDER_CTEST_INSTALL_FILE}"
                         "\n# External integration test (cross-provider suite)\n${_external_staging}"
                    )
                endif()
                # Install tree.
                apply_ctest_category_labels(
                    "${HIPBLASLTPROVIDER_CTEST_CATEGORIES_YAML}"
                    "${HIPBLASLTPROVIDER_CTEST_INSTALL_FILE}"
                )
            endif()
            # Build tree.
            apply_ctest_category_labels(
                "${HIPBLASLTPROVIDER_CTEST_CATEGORIES_YAML}"
                EXPLICIT_TESTS ${PROJECT_NAME}-external-integration-check
            )
        endif()
    else()
        message(STATUS "hipdnn_integration_tests not found - skipping external-integration-check")
    endif()

    # Install the ctest files. When YAML-based categorization is active, install the
    # generated install_CTestTestfile.cmake to bin/hipblaslt_plugin/ so TheRock CI's
    # test_runner.py can drive `ctest -L quick` (etc.) from the artifact's install tree.
    # Otherwise, fall back to the project's own install_provider_ctest_files() which
    # emits an unlabeled CTestTestfile.cmake.
    if(HIPBLASLTPROVIDER_CTEST_INSTALL_FILE)
        install(FILES "${HIPBLASLTPROVIDER_CTEST_INSTALL_FILE}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/hipblaslt_plugin"
                RENAME "CTestTestfile.cmake"
        )
    else()
        install_provider_ctest_files("hipblaslt_plugin")
    endif()
endif()

install(TARGETS hipblaslt_plugin EXPORT hipblaslt-plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)
