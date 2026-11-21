# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
message(STATUS "Building miopen provider plugin standalone")
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include version utilities (must be before project() call)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/VersionUtils.cmake)

# Version handling
dnn_provider_setup_version(miopen_provider ${CMAKE_CURRENT_LIST_DIR})

# Set project with version
project(miopen-provider VERSION ${MIOPEN_PROVIDER_VERSION} LANGUAGES CXX)

# Root of the rocm-libraries monorepo (for shared infrastructure like CTest categories).
# Used by the YAML-based test categorization wiring further down. Guarded with an EXISTS
# check at the point of use so standalone / sparse checkouts that lack shared/ctest still build.
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
    set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

include(CMakePackageConfigHelpers)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/CheckPreCommit.cmake)

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

# Migrate HIP_DNN_SKIP_TESTS -> MIOPENPROVIDER_SKIP_TESTS
if(DEFINED HIP_DNN_SKIP_TESTS)
    set(MIOPENPROVIDER_SKIP_TESTS ${HIP_DNN_SKIP_TESTS} CACHE BOOL "Skip building tests" FORCE)
    unset(HIP_DNN_SKIP_TESTS CACHE)
    message(DEPRECATION "HIP_DNN_SKIP_TESTS has been renamed to MIOPENPROVIDER_SKIP_TESTS. Please update your build scripts to use -DMIOPENPROVIDER_SKIP_TESTS instead.")
endif()
option(MIOPENPROVIDER_SKIP_TESTS "Skip building tests" OFF)
option(MIOPENPROVIDER_ENABLE_COVERAGE "Build with code coverage flags" OFF)

option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)
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

list(
    PREPEND
    MIOPENPROVIDER_WARNING_COMPILE_OPTIONS
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
)

if(WIN32)
    # AMD clang reports `__declspec(dllimport)` as "not supported" on the
    # x86_64-pc-windows-msvc target, even though the attribute is honored
    # (verified via llvm-readobj --coff-exports on MIOpen.dll). This produces
    # ~150k spurious warnings from the CMake-generated MIOPEN_EXPORT and
    # MIOPEN_INTERNALS_EXPORT macros. Suppress until the compiler issue is
    # resolved upstream.
    list(PREPEND MIOPENPROVIDER_WARNING_COMPILE_OPTIONS -Wno-ignored-attributes)
endif()

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files;
# remove when fixed. Added to prevent issues with not finding hip cmake.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

    find_package(hip REQUIRED)

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
include(WindowsDllStaging)

include(Dependencies)
if(NOT MIOPENPROVIDER_SKIP_TESTS)
    fetch_gtest_dependency()
endif()
include(Tests)
include(LongPathManifest)

if(MIOPENPROVIDER_ENABLE_COVERAGE)
    message(STATUS "Enabling compile flags for coverage-mapping")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
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

# When this provider is built standalone or against an installed MIOpen, locate
# the package on disk. When built as part of a superbuild that also includes
# projects/miopen as a sibling component, the MIOpen target is already in the
# build graph and find_package would either fail or pull in a stale installed
# copy.
if(NOT TARGET MIOpen AND NOT TARGET MIOpen_private)
    find_package(miopen CONFIG REQUIRED)
endif()

# Select which MIOpen library the provider links against.
#
# With the hipDNN public/private split enabled (MIOPEN_ENABLE_HIPDNN_WRAPPER=ON in
# the MIOpen build) MIOpen ships a private implementation library MIOpen_private
# that exports the internal C API under _impl names. The provider links it directly
# and, via the force-included MiopenApiPrivateRename.hpp (below), its public C API
# calls are renamed to those _impl entry points — linking the public
# wrapper would fail to load, since libMIOpen.so exports only the public names. We
# detect that build by the presence of the MIOpen_private target.
#
# When the split is OFF (the default) there is no MIOpen_private target: the
# provider links the single public MIOpen library and its sources call the public
# C API unchanged, exactly as before the split.
if(TARGET MIOpen_private)
    set(MIOPEN_PROVIDER_MIOPEN_LIB MIOpen_private)
else()
    set(MIOPEN_PROVIDER_MIOPEN_LIB MIOpen)
endif()
message(STATUS "miopen-provider: linking MIOpen library target '${MIOPEN_PROVIDER_MIOPEN_LIB}'")

add_library(
    miopen_plugin_impl OBJECT
    engines/MiopenEngine.cpp
    engines/plans/MiopenBatchnormApplicabilityChecks.cpp
    engines/plans/MiopenBatchnormBwdPlan.cpp
    engines/plans/MiopenBatchnormFwdInferencePlan.cpp
    engines/plans/MiopenBatchnormFwdInferenceWithVariancePlan.cpp
    engines/plans/MiopenBatchnormFwdTrainingPlan.cpp
    engines/plans/MiopenBatchnormFwdTrainingPlanBuilder.cpp
    engines/plans/MiopenBatchnormPlanBuilder.cpp
    engines/plans/MiopenBinaryPointwiseChecks.cpp
    engines/plans/MiopenBinaryPointwisePlan.cpp
    engines/plans/MiopenBinaryPointwisePlanBuilder.cpp
    engines/plans/MiopenConvApplicabilityChecks.cpp
    engines/plans/MiopenConvBwdPlan.cpp
    engines/plans/MiopenConvFwdBiasActivPlan.cpp
    engines/plans/MiopenConvFwdBiasActivPlanBuilder.cpp
    engines/plans/MiopenConvFwdPlan.cpp
    engines/plans/MiopenConvPlanBuilder.cpp
    engines/plans/MiopenConvWrwPlan.cpp
    engines/plans/MiopenPointwiseTensorChecks.cpp
    engines/plans/MiopenUnaryActivationChecks.cpp
    engines/plans/MiopenUnaryActivationPlan.cpp
    engines/plans/MiopenUnaryActivationPlanBuilder.cpp
    HipdnnMiopenHandle.cpp
    MiopenActivationDescriptor.cpp
    MiopenContainer.cpp
    MiopenConvDescriptor.cpp
    MiopenPluginPublic.cpp
    MiopenTensor.cpp
    MiopenUtils.cpp
)

target_include_directories(miopen_plugin_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(miopen_plugin_impl PUBLIC ${MIOPEN_PROVIDER_MIOPEN_LIB} hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)

# Mark every header included as <miopen/...> as a system header, so the provider's -Werror and
# clang-tidy ignore them (on Windows, miopen.h trips -Wmicrosoft-enum-value). Marking MIOpen's
# include directories SYSTEM instead would not work: the in-tree target is shared with MIOpen's
# tests, CK plugins and driver, and SYSTEM would put ROCM_PATH/include ahead of this checkout's
# headers for all of them. An installed MIOpen is already system via its imported target.
# PUBLIC so the unit tests inherit it.
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(miopen_provider_system_header_prefix "/clang:--system-header-prefix=miopen/")
else()
    set(miopen_provider_system_header_prefix "--system-header-prefix=miopen/")
endif()
target_compile_options(miopen_plugin_impl PUBLIC "${miopen_provider_system_header_prefix}")
unset(miopen_provider_system_header_prefix)

target_compile_definitions(miopen_plugin_impl PUBLIC MIOPEN_BETA_API)
# When linking the private implementation library, force-include the symbol-rename
# header so the provider's public C API calls bind the _impl entry points exported
# by libMIOpen_private.so. The header is generated from MIOpen's
# include/miopen/miopen.h and src/include/miopen/miopen_internal.h by
# projects/miopen/script/generate_wrapper_sources.py, in the same run that writes
# the src/private/miopen_private_rename.h MIOpen_private is built with, so the two
# renames cannot differ. It keeps the provider sources free of _impl spellings.
# PUBLIC so the force-include also reaches the unit-test targets, which compile
# the provider headers. With the split OFF nothing is force-included and the
# provider calls the public C API unchanged.
#
# The header is named without a directory: it resolves through the PUBLIC include
# directory set above, which keeps the build-machine path -- and any space in it --
# out of a flag that SHELL: splits on spaces. Force-include is spelled -include by
# GNU-style drivers and /FI by MSVC-style ones (cl, clang-cl), and both accept the
# filename as a separate argument.
if(TARGET MIOpen_private)
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(miopen_provider_force_include "/FI")
    else()
        set(miopen_provider_force_include "-include")
    endif()
    target_compile_options(miopen_plugin_impl PUBLIC
        "SHELL:${miopen_provider_force_include} MiopenApiPrivateRename.hpp")
    unset(miopen_provider_force_include)
endif()
target_compile_options(miopen_plugin_impl PRIVATE ${MIOPENPROVIDER_WARNING_COMPILE_OPTIONS})
set_target_properties(miopen_plugin_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# These sources only call MIOpen's host C API, but linking MIOpen pulls in
# hip::device's `-x hip --offload-arch=...` interface flags, so each TU is
# parsed as HIP. clang-tidy resolves to the system clang-tidy (the ROCm one
# reports an unsupported version), which lacks ROCm's device-library bitcode
# and errors with "cannot find ROCm device library". The objects are host-only
# (no kernels, no embedded device code), so tell tidy to skip the device path.
list(APPEND CLANG_TIDY_EXTRA_ARGS -extra-arg=-nogpulib)
clang_tidy_check(miopen_plugin_impl)

# Static library for unit tests to link against (provides access to internal implementation)
add_library(miopen_plugin_private STATIC $<TARGET_OBJECTS:miopen_plugin_impl>)
target_include_directories(miopen_plugin_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link to miopen_plugin_impl to inherit its PUBLIC properties
target_link_libraries(miopen_plugin_private PUBLIC ${MIOPEN_PROVIDER_MIOPEN_LIB} hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk miopen_plugin_impl)

add_library(miopen_plugin SHARED $<TARGET_OBJECTS:miopen_plugin_impl>)
target_link_libraries(miopen_plugin PRIVATE ${MIOPEN_PROVIDER_MIOPEN_LIB} hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)

if(NOT WIN32)
    target_link_libraries(miopen_plugin PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

target_compile_options(miopen_plugin PRIVATE ${MIOPENPROVIDER_WARNING_COMPILE_OPTIONS})

# INSTALL_RPATH applies to the installed plugin only (plain `cmake --install`
# to ${ROCM_PATH}). It is NOT used by TheRock, which overwrites INSTALL_RPATH
# with $ORIGIN-relative paths via its global post-subproject hook.
set_target_properties(
    miopen_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "\$ORIGIN/../.."
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

# Set RPATH origin for TheRock integration. The plugin is installed to
# lib/hipdnn_plugins/engines/ which is 2 levels below lib/, so TheRock needs
# this hint to calculate correct relative RPATHs.
set_target_properties(miopen_plugin PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")

# Generate version header
dnn_provider_generate_version_header(miopen_provider miopen_plugin_impl ${CMAKE_CURRENT_LIST_DIR})

if(NOT MIOPENPROVIDER_SKIP_TESTS)
    set(MIOPENPROVIDER_GTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
    set(MIOPENPROVIDER_CTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_external.yaml")
    set(MIOPENPROVIDER_INTEGRATION_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_integration.yaml")
    set(MIOPENPROVIDER_CTEST_INSTALL_FILE "")
    set(MIOPENPROVIDER_YAML_CATEGORIZATION_ENABLED FALSE)
    if(EXISTS "${MIOPENPROVIDER_GTEST_CATEGORIES_YAML}"
       AND EXISTS "${MIOPENPROVIDER_CTEST_CATEGORIES_YAML}"
       AND EXISTS "${MIOPENPROVIDER_INTEGRATION_CATEGORIES_YAML}"
       AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        set(DNN_PROVIDER_CTEST_CATEGORIES_YAML "${MIOPENPROVIDER_CTEST_CATEGORIES_YAML}")
        set(DNN_PROVIDER_TEST_CATEGORY_YAMLS
            "${MIOPENPROVIDER_GTEST_CATEGORIES_YAML}"
            "${MIOPENPROVIDER_INTEGRATION_CATEGORIES_YAML}"
            "${MIOPENPROVIDER_CTEST_CATEGORIES_YAML}"
        )
        set(MIOPENPROVIDER_YAML_CATEGORIZATION_ENABLED TRUE)
        message(STATUS "miopen-provider: YAML-based CTest categorization enabled")
    else()
        message(STATUS "miopen-provider: shared/ctest or test category YAML not found, using legacy CTest labels")
    endif()

    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    if(NOT TARGET hipdnn_integration_tests)
        find_package(hipdnn_integration_tests CONFIG QUIET)
    endif()

    set(DNN_PROVIDER_TEST_RUN_DEPENDS miopen_plugin_tests miopen_plugin_integration_tests)
    if(TARGET hipdnn_integration_tests)
        list(APPEND DNN_PROVIDER_TEST_RUN_DEPENDS miopen_plugin hipdnn_integration_tests)
    endif()

    # The plugin ABI checks (check_public_abi.py check-plugin and the re-entry test) need
    # MIOpen's private library, Python and the MIOpen scripts. The stand-in plugin below is
    # built only where they are registered.
    set(miopen_plugin_abi_script
        "${ROCM_LIBRARIES_ROOT}/projects/miopen/script/check_public_abi.py")
    set(miopen_plugin_abi_checks FALSE)
    if(TARGET MIOpen_private)
        if(EXISTS "${miopen_plugin_abi_script}" AND Python3_EXECUTABLE)
            set(miopen_plugin_abi_checks TRUE)
        else()
            message(STATUS "miopen-provider: plugin ABI checks not registered: "
                           "${miopen_plugin_abi_script} or Python3 not found")
        endif()
    endif()

    # A plugin built the way an older release or a prebuilt SDK plugin is: these sources,
    # linked against the public libMIOpen.so, with no rename. It is the input that
    # check_public_abi.py check-plugin and MIOpen's re-entry guard must reject. It stays out
    # of the engine plugin directory because hipDNN would load it as a real engine there.
    # Built by default so the tests that read it find it; the check targets also list it in
    # DNN_PROVIDER_TEST_RUN_DEPENDS.
    if(miopen_plugin_abi_checks AND NOT WIN32)
        get_target_property(miopen_plugin_stand_in_sources miopen_plugin_impl SOURCES)
        add_library(miopen_plugin_public_stand_in SHARED ${miopen_plugin_stand_in_sources})
        unset(miopen_plugin_stand_in_sources)
        # Same include directories and definitions as the real plugin. Its compile options and
        # link libraries are not copied: they carry the rename header and MIOpen_private.
        target_include_directories(miopen_plugin_public_stand_in PRIVATE
            $<TARGET_PROPERTY:miopen_plugin_impl,INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(miopen_plugin_public_stand_in PRIVATE
            $<TARGET_PROPERTY:miopen_plugin_impl,INTERFACE_COMPILE_DEFINITIONS>)
        target_link_libraries(miopen_plugin_public_stand_in PRIVATE
            MIOpen hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)
        # --as-needed drops libraries nothing needs. libMIOpen_private.so stays, because the
        # stand-in calls the three *GetWorkSpaceSizeRange functions that only it exports, so
        # check-plugin rejects the stand-in for linking the public library and for needing
        # public entry points, not for missing the private one.
        target_link_options(miopen_plugin_public_stand_in PRIVATE -Wl,--as-needed)
        target_compile_options(miopen_plugin_public_stand_in PRIVATE
            --system-header-prefix=miopen/ ${MIOPENPROVIDER_WARNING_COMPILE_OPTIONS})
        set_target_properties(miopen_plugin_public_stand_in PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            POSITION_INDEPENDENT_CODE ON
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/public_stand_in")
        list(APPEND DNN_PROVIDER_TEST_RUN_DEPENDS miopen_plugin_public_stand_in)
    endif()
    # In a discrete build the re-entry entry runs a small test binary, so the check targets
    # build it. They do not build the combined miopen_gtest.
    if(miopen_plugin_abi_checks AND MIOPEN_TEST_DISCRETE AND TARGET test_hipdnn_reentry)
        list(APPEND DNN_PROVIDER_TEST_RUN_DEPENDS test_hipdnn_reentry)
    endif()

    # Finalize test targets with prefix to avoid collisions in superbuild mode
    # Keep this after all build folders have been added so they have a chance to register their tests
    # using add_*_test_target()
    if(MIOPENPROVIDER_ENABLE_COVERAGE)
        finalize_test_targets("${PROJECT_NAME}" ENABLE_COVERAGE)
    else()
        finalize_test_targets("${PROJECT_NAME}")
    endif()

    if(MIOPENPROVIDER_YAML_CATEGORIZATION_ENABLED)
        enable_testing()
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
        # Install-time CTestTestfile uses relative paths so `ctest` can run from the
        # installed location (bin/miopen_plugin/) and reach ../miopen_plugin_tests etc.
        set(MIOPENPROVIDER_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
        file(WRITE "${MIOPENPROVIDER_CTEST_INSTALL_FILE}"
[=[
# This is a test file generated by miopen-provider for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
        )
        # On ASAN builds, HIPDNN_TEST_MIOPEN_CACHE_DIR is set (see Sanitizers.cmake) and the generated
        # suites must require the cache-clearing fixture; empty otherwise.
        set(_miopen_cache_fixture "")
        if(HIPDNN_TEST_MIOPEN_CACHE_DIR)
            set(_miopen_cache_fixture hipdnn_clear_miopen_test_cache)
        endif()
        # The GTest YAML applies to both test binaries; each call appends its category suites
        # (e.g. miopen_plugin_tests_quick_suite, miopen_plugin_integration_tests_quick_suite)
        # to the same install-time CTestTestfile.cmake. ENVIRONMENT forwards the CMake-side
        # TEST_ENVIRONMENT (ASAN symbolizer path, coverage LLVM_PROFILE_FILE, set by
        # dnn-providers/cmake/Sanitizers.cmake / _build_test_environment_list_internal) to
        # every generated suite; without it those suites run unsanitized/uninstrumented.
        apply_test_category_labels(
            miopen_plugin_tests
            "${MIOPENPROVIDER_GTEST_CATEGORIES_YAML}"
            "${PROJECT_BINARY_DIR}/staging"
            "${MIOPENPROVIDER_CTEST_INSTALL_FILE}"
            ENVIRONMENT ${TEST_ENVIRONMENT}
            ENVIRONMENT_MODIFICATION ${TEST_ENVIRONMENT_MODIFICATION}
            FIXTURES_REQUIRED ${_miopen_cache_fixture}
        )
        apply_test_category_labels(
            miopen_plugin_integration_tests
            "${MIOPENPROVIDER_GTEST_CATEGORIES_YAML}"
            "${PROJECT_BINARY_DIR}/staging"
            "${MIOPENPROVIDER_CTEST_INSTALL_FILE}"
            ENVIRONMENT ${TEST_ENVIRONMENT}
            ENVIRONMENT_MODIFICATION ${TEST_ENVIRONMENT_MODIFICATION}
            FIXTURES_REQUIRED ${_miopen_cache_fixture}
        )
    endif()

    set(_external_category_args "")
    if(MIOPENPROVIDER_YAML_CATEGORIZATION_ENABLED)
        list(APPEND _external_category_args
            TEST_CATEGORIES_YAML "${MIOPENPROVIDER_INTEGRATION_CATEGORIES_YAML}"
            INSTALL_TEST_FILE "${MIOPENPROVIDER_CTEST_INSTALL_FILE}"
            TEST_NAME_PREFIX ${PROJECT_NAME}-external-integration
            ENVIRONMENT ${TEST_ENVIRONMENT}
            ENVIRONMENT_MODIFICATION ${TEST_ENVIRONMENT_MODIFICATION}
            FIXTURES_REQUIRED ${_miopen_cache_fixture}
        )
    endif()

    if(TARGET hipdnn_integration_tests)
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-external-integration-check
            PLUGIN_TARGET  miopen_plugin
            ENGINE_NAME    MIOPEN_ENGINE
            INSTALL_SUBDIR miopen_plugin
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/config/MIOPEN_ENGINE.toml
            # Five small batchnorm inference bundles: a few seconds on one GPU.
            BENCHMARK_REFERENCE_SMOKE_FILTER "quick_BatchnormFwdInference_Inference.*"
            ${_external_category_args}
        )
        message(STATUS "Created ${PROJECT_NAME}-external-integration-check target")

        # This is a pre-registered CTest test (not a GTest binary), so it is
        # labeled via apply_ctest_category_labels()/test_categories_external.yaml.
        if(MIOPENPROVIDER_YAML_CATEGORIZATION_ENABLED)
            if(MIOPENPROVIDER_CTEST_INSTALL_FILE)
                # Fold the staged add_test() into the labeled file before labeling;
                # apply_ctest_category_labels() scans the file's add_test() names.
                get_property(_external_staging GLOBAL
                    PROPERTY "EXTERNAL_TEST_INSTALL_STAGING_miopen_plugin"
                )
                if(_external_staging)
                    file(APPEND "${MIOPENPROVIDER_CTEST_INSTALL_FILE}"
                         "\n# External integration test (cross-provider suite)\n${_external_staging}"
                    )
                endif()
                # Install tree.
                apply_ctest_category_labels(
                    "${MIOPENPROVIDER_CTEST_CATEGORIES_YAML}"
                    "${MIOPENPROVIDER_CTEST_INSTALL_FILE}"
                )
            endif()
            # Build tree.
            apply_ctest_category_labels(
                "${MIOPENPROVIDER_CTEST_CATEGORIES_YAML}"
                EXPLICIT_TESTS ${PROJECT_NAME}-external-integration-check
                               ${PROJECT_NAME}-external-integration-check-benchmark-reference-smoke
            )
        endif()
    else()
        message(STATUS "hipdnn_integration_tests not found - skipping external-integration-check")
    endif()

    # The plugin must bind to libMIOpen_private.so (check_public_abi.py check-plugin). The
    # second entry runs the check on the stand-in and requires the "plugin check: FAIL" line,
    # not just a non-zero exit, so a script error does not count as a rejection.
    if(miopen_plugin_abi_checks)
        enable_testing()
        add_test(NAME ${PROJECT_NAME}-plugin-abi-check
            COMMAND ${Python3_EXECUTABLE} ${miopen_plugin_abi_script} check-plugin
                    $<TARGET_FILE:miopen_plugin> --private-lib $<TARGET_FILE:MIOpen_private>)
        set(miopen_plugin_abi_tests ${PROJECT_NAME}-plugin-abi-check)
        if(TARGET miopen_plugin_public_stand_in)
            add_test(NAME ${PROJECT_NAME}-plugin-abi-check-rejects-public-plugin
                COMMAND ${Python3_EXECUTABLE} ${miopen_plugin_abi_script} check-plugin
                        $<TARGET_FILE:miopen_plugin_public_stand_in>
                        --private-lib $<TARGET_FILE:MIOpen_private>)
            set_tests_properties(${PROJECT_NAME}-plugin-abi-check-rejects-public-plugin
                PROPERTIES PASS_REGULAR_EXPRESSION "plugin check: FAIL")
            list(APPEND miopen_plugin_abi_tests
                 ${PROJECT_NAME}-plugin-abi-check-rejects-public-plugin)
        endif()
        # Runs the re-entry test (test/gtest/hipdnn_reentry.cpp) with the real hipDNN backend and
        # the stand-in as the only plugin; see check_plugin_reentry.py. Needs a built MIOpen
        # test binary and a GPU, so the script skips (exit 77) on a machine that cannot run it,
        # instead of failing.
        if(MIOPEN_TEST_DISCRETE)
            set(miopen_plugin_reentry_gtest test_hipdnn_reentry)
        else()
            set(miopen_plugin_reentry_gtest miopen_gtest)
        endif()
        if(TARGET miopen_plugin_public_stand_in AND TARGET ${miopen_plugin_reentry_gtest})
            add_test(NAME ${PROJECT_NAME}-plugin-reentry-guard
                COMMAND ${Python3_EXECUTABLE}
                        ${ROCM_LIBRARIES_ROOT}/projects/miopen/script/check_plugin_reentry.py
                        --gtest $<TARGET_FILE:${miopen_plugin_reentry_gtest}>
                        --stand-in-dir $<TARGET_FILE_DIR:miopen_plugin_public_stand_in>)
            set_tests_properties(${PROJECT_NAME}-plugin-reentry-guard PROPERTIES
                SKIP_RETURN_CODE 77
                TIMEOUT 300)
            # It runs MIOpen on a GPU, so it takes the same sanitizer environment and cache
            # clearing as the provider's other tests.
            if(TEST_ENVIRONMENT)
                set_tests_properties(${PROJECT_NAME}-plugin-reentry-guard PROPERTIES
                    ENVIRONMENT "${TEST_ENVIRONMENT}")
            endif()
            if(TEST_ENVIRONMENT_MODIFICATION)
                set_tests_properties(${PROJECT_NAME}-plugin-reentry-guard PROPERTIES
                    ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
            endif()
            if(HIPDNN_TEST_MIOPEN_CACHE_DIR)
                set_tests_properties(${PROJECT_NAME}-plugin-reentry-guard PROPERTIES
                    FIXTURES_REQUIRED hipdnn_clear_miopen_test_cache)
            endif()
            list(APPEND miopen_plugin_abi_tests ${PROJECT_NAME}-plugin-reentry-guard)
        elseif(TARGET miopen_plugin_public_stand_in)
            message(STATUS "${PROJECT_NAME}-plugin-reentry-guard not registered: "
                           "MIOpen's test binary '${miopen_plugin_reentry_gtest}' is not "
                           "configured (it needs BUILD_TESTING=ON)")
        endif()
        if(MIOPENPROVIDER_YAML_CATEGORIZATION_ENABLED)
            apply_ctest_category_labels("${MIOPENPROVIDER_CTEST_CATEGORIES_YAML}"
                                        EXPLICIT_TESTS ${miopen_plugin_abi_tests})
        endif()
    endif()

    # Install the ctest files. When YAML-based categorization is active, install the
    # generated install_CTestTestfile.cmake to bin/miopen_plugin/ so TheRock CI's
    # test_runner.py can drive `ctest -L quick` (etc.) from the artifact's install tree.
    # Otherwise, fall back to the project's own install_provider_ctest_files() which
    # emits an unlabeled CTestTestfile.cmake.
    if(MIOPENPROVIDER_CTEST_INSTALL_FILE)
        install(FILES "${MIOPENPROVIDER_CTEST_INSTALL_FILE}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/miopen_plugin"
                RENAME "CTestTestfile.cmake"
        )
    else()
        install_provider_ctest_files("miopen_plugin")
    endif()
endif()

install(TARGETS miopen_plugin EXPORT miopen-plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/${PROJECT_NAME}")

add_clang_tidy_custom_target()

if(MIOPENPROVIDER_ENABLE_COVERAGE)
    findandcheckllvmtools()

    set(CODE_COVERAGE_IGNORE_REGEX ".*deps.*|.*tests.*")
    set(CODE_COVERAGE_OBJECTS
        -object ./${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/libmiopen_plugin.so
        -object ./${CMAKE_INSTALL_BINDIR}/miopen_plugin_tests
        -object ./${CMAKE_INSTALL_BINDIR}/miopen_plugin_integration_tests
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
            COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report
            COMMAND find . -name "*.profraw" -print0 | xargs -0 ${LLVM_PROFDATA_BINARY} merge
                    -sparse -o ./coverage-report/miopen_plugin.profdata
            COMMAND
                ${LLVM_COV_BINARY} report ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/miopen_plugin.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} >
                ./coverage-report/code_cov_miopen_plugin.report
            COMMAND
                ${LLVM_COV_BINARY} show -Xdemangler=${LLVM_CXXFILT_BINARY} ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/miopen_plugin.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} --format=html
                --output-dir=coverage-report
            COMMAND
                ${LLVM_COV_BINARY} export ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/miopen_plugin.profdata
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

    if(MIOPENPROVIDER_SKIP_TESTS)
        message(
            WARNING
                "\nThe code coverage targets depend on test targets but test targets don't exist because MIOPENPROVIDER_SKIP_TESTS is enabled."
        )
    else()
        add_custom_coverage_target(code_coverage check) # code_coverage depends on 'check'
        add_custom_coverage_target(unit-coverage unit-check) # unit-coverage depends on 'unit-check'
        add_custom_coverage_target(integration-coverage integration-check) # ... 'integration-check'

        # Make alias to target coverage
        add_custom_target(coverage DEPENDS code_coverage COMMENT "Alias for code_coverage")
    endif()

    # Create a coverage target that does not depend on any other target and will use the coverage
    # data that already exists on disk from prior / manual target runs performed before this target
    # is run.
    add_custom_coverage_target(current-coverage "")

endif()
hipdnn_apply_long_path_manifest()
