# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)

project(hipdnn-integration-tests LANGUAGES CXX)

if(NOT DEFINED ROCM_LIBRARIES_ROOT)
    set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
    CACHE BOOL "Add paths to linker search and installed rpath"
)

if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${CMAKE_CURRENT_SOURCE_DIR}/../cmake
)

option(ENABLE_CLANG_FORMAT "Enable Clang-Format checks" ON)
if(NOT ENABLE_CLANG_FORMAT)
    message(WARNING "Clang-Format checks are disabled. CI requires passing clang-format checks.")
endif()

if(WIN32)
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" OFF)
    if(ENABLE_CLANG_TIDY)
        message(WARNING "Clang-Tidy enabled on Windows - this significantly increases build time.")
    endif()
else()
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)
    if(NOT ENABLE_CLANG_TIDY)
        message(WARNING "Clang-Tidy checks are disabled. CI requires passing clang-tidy checks.")
    endif()
endif()

option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)

set(RANDOMIZATION_LIBRARY "rocrand"  CACHE STRING "Name of the randomization library")

# Derive USE_ROCRAND as a boolean based on the RANDOMIZATION_LIBRARY value
if(RANDOMIZATION_LIBRARY STREQUAL "rocrand")
    set(USE_ROCRAND TRUE)
else()
    set(USE_ROCRAND FALSE)
endif()

include(GNUInstallDirs)
include(ClangTidy)
include(ClangCheck)
include(Sanitizers)
include(Tests)
include(HipdnnIntegrationTestHelpers)

list(
    PREPEND
    INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS
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
    -Wswitch-default # Warn if a switch statement does not have a default case
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS
        -Wno-error=unused-command-line-argument # Clang-specific: disable error for unused CLI args
    )
endif()

find_package(hip REQUIRED)

if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk REQUIRED)
endif()

if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk REQUIRED)
endif()

find_package(Threads REQUIRED)

find_package(rocrand QUIET)
if(USE_ROCRAND AND NOT TARGET roc::rocrand)
    message(WARNING "rocRAND not found, falling back to CPU random number generation")
    set(USE_ROCRAND OFF)
endif()

include(Dependencies)
fetch_gtest_dependency()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Copy integration test bundles into the build tree so tests can find them at
# <exe_dir>/../lib/integration-test-bundles/
# NOTE: file(COPY) runs at configure time only. If you add or modify files
# under integration-test-bundles/, re-run cmake to pick up the changes.
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/integration-test-bundles/"
     DESTINATION "${CMAKE_BINARY_DIR}/lib/integration-test-bundles")

if(Python3_FOUND)
    add_test(
        NAME hipdnn_bundle_verifier_python_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/reference-data-scripts/tests/test_verify_golden_bundles.py
    )
    # Labels are applied via the shared YAML category path below (test_categories_external.yaml)
    # so this pre-registered test gets the tier labels.
    set(HIPDNN_IT_BUNDLE_VERIFIER_TEST hipdnn_bundle_verifier_python_tests)
endif()

# Install integration test bundles so installed tests can find them at
# <exe_dir>/../lib/integration-test-bundles/
install(DIRECTORY "${CMAKE_BINARY_DIR}/lib/integration-test-bundles"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set(INTEGRATION_TESTS_EXE hipdnn_integration_tests)

add_executable(${INTEGRATION_TESTS_EXE}
    src/main.cpp
    src/harness/ReferenceExecutorPool.cpp
    src/harness/bundle/BundleReferenceValidationHarness.cpp
    src/harness/bundle/FrontendGraphEngineRunner.cpp
    src/harness/bundle/GraphSession.cpp
    src/harness/bundle/HarnessDependencies.cpp
    src/harness/bundle/IntegrationBundleVerificationHarness.cpp
    src/harness/bundle/OutputComparison.cpp
    src/harness/bundle/ProductionPolicy.cpp
    src/harness/bundle/ReferenceOpCoverage.cpp
    src/harness/bundle/SupportClaimReport.cpp
    src/harness/bundle/SupportClaims.cpp
    src/harness/bundle/SupportVerdict.cpp
    src/harness/bundle/VariantPackBuilder.cpp
    src/harness/input-init/FillInputs.cpp
)

# Golden-data validation: validates our checked-in .bin data against our own
# reference executors. Deliberately a separate binary rather than a mode of
# hipdnn_integration_tests -- it loads no plugin, creates no handle, and builds no
# engine table, so "this run cannot reach an engine" is a property of the link
# rather than of a flag. FrontendGraphEngineRunner.cpp and HarnessDependencies.cpp
# are absent for exactly that reason.
#
# It is also NOT registered through add_external_integration_test_target(): that
# parameterizes a binary over a provider's plugin and engine, and no engine is
# involved here. Registering it once, below, is what stops the same golden data
# being revalidated once per provider.
set(GOLDEN_DATA_TESTS_EXE hipdnn_golden_data_tests)

add_executable(${GOLDEN_DATA_TESTS_EXE}
    src/golden_data_main.cpp
    src/harness/ReferenceExecutorPool.cpp
    src/harness/bundle/BundleReferenceValidationHarness.cpp
    src/harness/bundle/GraphSession.cpp
    src/harness/bundle/OutputComparison.cpp
    src/harness/bundle/ReferenceOpCoverage.cpp
    src/harness/bundle/SupportClaimReport.cpp
    src/harness/bundle/SupportClaims.cpp
    src/harness/bundle/SupportVerdict.cpp
    src/harness/bundle/VariantPackBuilder.cpp
    src/harness/input-init/FillInputs.cpp
)

target_include_directories(${GOLDEN_DATA_TESTS_EXE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${GOLDEN_DATA_TESTS_EXE} SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third-party/argparse
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third-party/tomlplusplus
)

target_link_libraries(${GOLDEN_DATA_TESTS_EXE}
    PRIVATE
        hip::host
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gtest
        Threads::Threads
        $<$<PLATFORM_ID:Windows>:shlwapi>
)

target_compile_options(${GOLDEN_DATA_TESTS_EXE}
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(${GOLDEN_DATA_TESTS_EXE})

add_integration_test_target(${GOLDEN_DATA_TESTS_EXE} ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(src/integration-tests)

target_include_directories(${INTEGRATION_TESTS_EXE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${INTEGRATION_TESTS_EXE} SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third-party/argparse
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third-party/tomlplusplus
)

target_link_libraries(${INTEGRATION_TESTS_EXE}
    PRIVATE
        hip::host
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gtest
        Threads::Threads
        $<$<PLATFORM_ID:Windows>:shlwapi>
)

target_compile_options(${INTEGRATION_TESTS_EXE} PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(${INTEGRATION_TESTS_EXE})

# NOTE: hipdnn_integration_tests is intentionally NOT registered as a tiered
# ctest target here. The bundle sweep tests it carries exercise the plugin, so
# they are driven by each provider's add_external_integration_test_target()
# (see dnn-providers/miopen-provider) which runs this binary against a specific
# engine and selects tests via that provider's test_categories_integration.yaml.

set(HIPDNN_IT_GTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
set(HIPDNN_IT_CTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_external.yaml")
set(HIPDNN_IT_YAML_CATEGORIZATION_ENABLED FALSE)
if(EXISTS "${HIPDNN_IT_GTEST_CATEGORIES_YAML}"
   AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
    include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
    set(DNN_PROVIDER_TEST_CATEGORY_YAMLS "${HIPDNN_IT_GTEST_CATEGORIES_YAML}")
    set(DNN_PROVIDER_CTEST_CATEGORIES_YAML "${HIPDNN_IT_CTEST_CATEGORIES_YAML}")
    set(HIPDNN_IT_YAML_CATEGORIZATION_ENABLED TRUE)
    message(STATUS "hipdnn-integration-tests: YAML-based CTest categorization enabled")
    # Label the pre-registered bundle-verifier ctest test via the shared YAML path
    # (it is registered directly above, not by _create_test_name_validation_target_internal).
    if(HIPDNN_IT_BUNDLE_VERIFIER_TEST)
        _apply_provider_ctest_category_labels(${HIPDNN_IT_BUNDLE_VERIFIER_TEST})
    endif()
else()
    message(STATUS "hipdnn-integration-tests: shared/ctest or test_categories.yaml not found, using legacy CTest labels")
endif()

add_subdirectory(gpu-ref)
add_subdirectory(tests)

# Finalize test targets with prefix to avoid collisions in superbuild mode
# Keep this after all build folders have been added so they have a chance to register their tests
# using add_*_test_target()
finalize_test_targets("${PROJECT_NAME}")

if(HIPDNN_IT_YAML_CATEGORIZATION_ENABLED)
    enable_testing()
    set(HIPDNN_IT_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
    file(WRITE "${HIPDNN_IT_CTEST_INSTALL_FILE}"
         "# Autogenerated CTestTestfile for installed hipdnn-integration-tests categories.\n"
         "# Tests use relative paths to work in the installed tree.\n")

    # The GTest binaries land in CMAKE_BINARY_DIR/bin only once built (RUNTIME_OUTPUT_DIRECTORY,
    # Tests.cmake:_add_test_target_internal), but apply_test_category_labels() validates the
    # working directory at configure time -- create it up front so a from-scratch configure
    # doesn't fail validation before anything has been built.
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

    # No FIXTURES_REQUIRED/hipdnn_clear_miopen_test_cache here: none of these
    # targets link against or invoke MIOpen (they exercise the hipDNN backend, the
    # from-scratch GPU reference implementation, and our own golden data directly),
    # so the ASAN-only MIOpen-cache-clear fixture miopen-provider/hipblaslt-provider
    # wire up does not apply here.
    foreach(_it_target hipdnn_integration_tests_unit_tests hipdnn_gpu_ref_tests
                       hipdnn_golden_data_tests)
        apply_test_category_labels(
            ${_it_target}
            "${HIPDNN_IT_GTEST_CATEGORIES_YAML}"
            "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
            "${HIPDNN_IT_CTEST_INSTALL_FILE}"
            ENVIRONMENT ${TEST_ENVIRONMENT}
            ENVIRONMENT_MODIFICATION ${TEST_ENVIRONMENT_MODIFICATION}
        )
    endforeach()

    # Install the generated (category-labeled) CTestTestfile so
    # `ctest -L quick|standard|...` works from the installed tree.
    install(FILES "${HIPDNN_IT_CTEST_INSTALL_FILE}"
            DESTINATION "${CMAKE_INSTALL_BINDIR}/hipdnn_integration_tests_ctest"
            RENAME "CTestTestfile.cmake")
else()
    install_provider_ctest_files("hipdnn_integration_tests_ctest")
endif()

include(CMakePackageConfigHelpers)

set(HIPDNN_INTEGRATION_TESTS_VERSION "0.1.0")

set_target_properties(
    ${INTEGRATION_TESTS_EXE}
    PROPERTIES
        INSTALL_RPATH
        "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/engines"
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_RPATH_USE_ORIGIN TRUE
)

# Install the executables with an export set
install(TARGETS ${INTEGRATION_TESTS_EXE}
    EXPORT ${INTEGRATION_TESTS_EXE}_targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# --- Package export infrastructure ---

# Generate Config.cmake from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${INTEGRATION_TESTS_EXE}Config.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${INTEGRATION_TESTS_EXE}/${INTEGRATION_TESTS_EXE}Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${INTEGRATION_TESTS_EXE}
)

# Generate ConfigVersion.cmake
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${INTEGRATION_TESTS_EXE}/${INTEGRATION_TESTS_EXE}ConfigVersion.cmake"
    VERSION ${HIPDNN_INTEGRATION_TESTS_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Build-tree export so superbuild consumers can find the package without installing
export(TARGETS ${INTEGRATION_TESTS_EXE}
    FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${INTEGRATION_TESTS_EXE}/${INTEGRATION_TESTS_EXE}Targets.cmake"
)

# Install-tree exports
install(EXPORT ${INTEGRATION_TESTS_EXE}_targets
    FILE ${INTEGRATION_TESTS_EXE}Targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${INTEGRATION_TESTS_EXE}
)

install(FILES
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${INTEGRATION_TESTS_EXE}/${INTEGRATION_TESTS_EXE}Config.cmake"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${INTEGRATION_TESTS_EXE}/${INTEGRATION_TESTS_EXE}ConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HipdnnIntegrationTestHelpers.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${INTEGRATION_TESTS_EXE}
)

add_clang_tidy_custom_target()
