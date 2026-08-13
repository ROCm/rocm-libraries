# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# OBJECT library with all implementation
add_library(
    hip_kernel_provider_impl OBJECT
    core/Handle.cpp
    core/Container.cpp
    core/PluginPublic.cpp
    core/Utils.cpp
    compilation/Kernel.cpp
    compilation/Program.cpp
)

add_subdirectory(engines)

target_include_directories(hip_kernel_provider_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/engines/
)
dnn_provider_generate_version_header(hip_kernel_provider hip_kernel_provider_impl ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(hip_kernel_provider_impl PUBLIC hip::host hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hiprtc::hiprtc)

target_compile_options(hip_kernel_provider_impl PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(hip_kernel_provider_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
clang_tidy_check(hip_kernel_provider_impl)
add_clang_tidy_custom_target()

# Static library for unit tests to link against (provides access to internal implementation)
add_library(hip_kernel_provider_private STATIC $<TARGET_OBJECTS:hip_kernel_provider_impl>)
target_include_directories(hip_kernel_provider_private PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_link_libraries(hip_kernel_provider_private PUBLIC hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hiprtc::hiprtc hip_kernel_provider_impl)
if(ENABLE_ASM_SDPA_ENGINE)
    target_compile_definitions(hip_kernel_provider_private PUBLIC HIPDNN_ENGINE_ASM_SDPA=1)
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    target_compile_definitions(hip_kernel_provider_private PUBLIC HIPDNN_ENGINE_HIP_MLOPS=1)
endif()

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    target_compile_definitions(hip_kernel_provider_private PUBLIC HIPDNN_ENABLE_KERNEL_INGESTOR=1)
    # Only the install path is compiled in. Baking the build tree in as well would ship an
    # absolute path from this machine inside the plugin and prefer it on any host where it
    # happened to exist, which also means nothing would ever exercise the installed copy.
    # Build-tree runs set HIPDNN_DESCRIPTOR_DIR instead; the test targets do it for ctest.
    target_compile_definitions(hip_kernel_provider_impl PRIVATE
        HIPDNN_DESCRIPTOR_INSTALL_DIR="${CMAKE_INSTALL_PREFIX}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPDNN_DESCRIPTOR_SUBDIR}"
    )
endif()

# hip_kernel_provider - SHARED library (the actual plugin)
add_library(hip_kernel_provider SHARED $<TARGET_OBJECTS:hip_kernel_provider_impl>)
target_link_libraries(hip_kernel_provider PRIVATE hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hiprtc::hiprtc hip::host)

if(NOT WIN32)
    target_link_libraries(hip_kernel_provider PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

target_compile_options(hip_kernel_provider PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    hip_kernel_provider
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

# Set RPATH origin for TheRock integration
set_target_properties(hip_kernel_provider PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")

# ============================================================================
# Tests
# ============================================================================
if(HIPKERNELPROVIDER_ENABLE_TESTS)

    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    if(NOT TARGET hipdnn_integration_tests)
        find_package(hipdnn_integration_tests CONFIG QUIET)
    endif()

    set(DNN_PROVIDER_TEST_RUN_DEPENDS hip_kernel_provider_tests hip_kernel_provider_integration_tests)
    if(TARGET hipdnn_integration_tests)
        list(APPEND DNN_PROVIDER_TEST_RUN_DEPENDS hip_kernel_provider hipdnn_integration_tests)
    endif()

    # Keep this after all build folders have been added so they have a chance to register their tests
    # using add_*_test_target()
    if(HIPKERNELPROVIDER_ENABLE_COVERAGE)
        finalize_test_targets("${PROJECT_NAME}" ENABLE_COVERAGE)
    else()
        finalize_test_targets("${PROJECT_NAME}")
    endif()

    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
        enable_testing()
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
        # Install-time CTestTestfile uses relative paths so `ctest` can run from the
        # installed location (bin/hip_kernel_provider/) and reach ../hip_kernel_provider_tests etc.
        set(HIPKERNELPROVIDER_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
        file(WRITE "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
[=[
# This is a test file generated by hip-kernel-provider for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
        )
        # The GTest YAML applies to both test binaries; each call appends its category suites
        # (e.g. hip_kernel_provider_tests_quick_suite, hip_kernel_provider_integration_tests_quick_suite)
        # to the same install-time CTestTestfile.cmake. ENVIRONMENT forwards the CMake-side
        # TEST_ENVIRONMENT (ASAN symbolizer path, coverage LLVM_PROFILE_FILE, set by
        # dnn-providers/cmake/Sanitizers.cmake / _build_test_environment_list_internal) to
        # every generated suite; without it those suites run unsanitized/uninstrumented.
        # The per-target published variable is that same ambient set merged with anything
        # passed to add_unit_test_target() -- HIPDNN_DESCRIPTOR_DIR, without which the
        # descriptor-backed engines are absent and every test that reads one fails.
        apply_test_category_labels(
            hip_kernel_provider_tests
            "${HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML}"
            "${PROJECT_BINARY_DIR}/staging"
            "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
            ENVIRONMENT ${hip_kernel_provider_tests_TEST_ENVIRONMENT}
        )
        # hip_kernel_provider_integration_tests_TEST_ENVIRONMENT is TEST_ENVIRONMENT
        # merged with any per-target ENVIRONMENT passed to add_integration_test_target()
        # (e.g. HIPDNN_AITER_ASM_DIR for the SDPA engine), published by
        # dnn-providers/cmake/Tests.cmake since the raw test name below is never
        # itself registered as a CTest test in this YAML-categorized mode.
        apply_test_category_labels(
            hip_kernel_provider_integration_tests
            "${HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML}"
            "${PROJECT_BINARY_DIR}/staging"
            "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
            ENVIRONMENT ${hip_kernel_provider_integration_tests_TEST_ENVIRONMENT}
        )
    endif()

    set(_external_category_args "")
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
        list(APPEND _external_category_args
            TEST_CATEGORIES_YAML "${HIPKERNELPROVIDER_INTEGRATION_CATEGORIES_YAML}"
            INSTALL_TEST_FILE "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
            TEST_NAME_PREFIX ${PROJECT_NAME}-external-integration
            ENVIRONMENT ${TEST_ENVIRONMENT}
        )
    endif()

    if(TARGET hipdnn_integration_tests)
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-external-integration-check
            PLUGIN_TARGET  hip_kernel_provider
            ENGINE_NAME    HIP_MLOPS_ENGINE
            INSTALL_SUBDIR hip_kernel_provider
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/../config/HIP_MLOPS_ENGINE.toml
            ${_external_category_args}
        )
        message(STATUS "Created ${PROJECT_NAME}-external-integration-check target")

        # This is a pre-registered CTest test (not a GTest binary), so it is
        # labeled via apply_ctest_category_labels()/HIP_MLOPS_ENGINE_test_categories_external.yaml.
        if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
            if(HIPKERNELPROVIDER_CTEST_INSTALL_FILE)
                # Fold the staged add_test() into the labeled file before labeling;
                # apply_ctest_category_labels() scans the file's add_test() names.
                get_property(_external_staging GLOBAL
                    PROPERTY "EXTERNAL_TEST_INSTALL_STAGING_hip_kernel_provider"
                )
                if(_external_staging)
                    file(APPEND "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                         "\n# External integration test (cross-provider suite)\n${_external_staging}"
                    )
                endif()
                # Install tree.
                apply_ctest_category_labels(
                    "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"
                    "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                )
                # Also apply rocKE's pre-registered CTest categories now that
                # the platform's staged add_test() entries (folded above) are
                # present in the file. This overwrites -- rather than merges
                # with -- the ad hoc "ckc;unit;host" / "ckc;python;..." labels
                # rocke/platform/CMakeLists.txt bakes into its install text:
                # ctest's install-tree script interpreter only honors the
                # last set_tests_properties() call per test, and
                # quick/standard/comprehensive/full is the RFC0010-standardized
                # replacement for those ad hoc tags (see ALMIOPEN-2278).
                apply_ctest_category_labels(
                    "${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"
                    "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                )
            endif()
            # Build tree.
            apply_ctest_category_labels(
                "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"
                EXPLICIT_TESTS ${PROJECT_NAME}-external-integration-check
            )
        endif()
    else()
        message(STATUS "hipdnn_integration_tests not found - skipping external-integration-check")
    endif()

    set(_asm_sdpa_external_category_args "")
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
        list(APPEND _asm_sdpa_external_category_args
            TEST_CATEGORIES_YAML "${HIPKERNELPROVIDER_ASM_SDPA_INTEGRATION_CATEGORIES_YAML}"
            INSTALL_TEST_FILE "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
            TEST_NAME_PREFIX ${PROJECT_NAME}-asm-sdpa-external-integration
            ENVIRONMENT ${TEST_ENVIRONMENT}
        )
    endif()

    if(TARGET hipdnn_integration_tests)
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-asm-sdpa-external-integration-check
            PLUGIN_TARGET  hip_kernel_provider
            ENGINE_NAME    ASM_SDPA_ENGINE
            INSTALL_SUBDIR hip_kernel_provider
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/../config/ASM_SDPA_ENGINE.toml
            ${_asm_sdpa_external_category_args}
        )
        message(STATUS "Created ${PROJECT_NAME}-asm-sdpa-external-integration-check target")

        # Unlike the HIP_MLOPS block above, this target does not re-fold
        # EXTERNAL_TEST_INSTALL_STAGING_hip_kernel_provider or re-apply
        # HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML (the provider-wide
        # _external.yaml): that staging/labeling is provider-wide and already
        # applied exactly once above. Re-running it here would re-append the
        # same rocke-staged add_test() entries into
        # HIPKERNELPROVIDER_CTEST_INSTALL_FILE a second time (duplicate CTest
        # test names). ASM_SDPA's own category suites are labeled via
        # TEST_CATEGORIES_YAML/INSTALL_TEST_FILE in
        # _asm_sdpa_external_category_args above.
    else()
        message(STATUS "hipdnn_integration_tests not found - skipping asm-sdpa-external-integration-check")
    endif()

    # Install the ctest files. When YAML-based categorization is active, install the
    # generated install_CTestTestfile.cmake to bin/hip_kernel_provider/ so TheRock CI's
    # test_runner.py can drive `ctest -L quick` (etc.) from the artifact's install tree.
    # Otherwise, fall back to the project's own install_provider_ctest_files() which
    # emits an unlabeled CTestTestfile.cmake.
    if(HIPKERNELPROVIDER_CTEST_INSTALL_FILE)
        install(FILES "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/hip_kernel_provider"
                RENAME "CTestTestfile.cmake"
        )
    else()
        install_provider_ctest_files("hip_kernel_provider")
    endif()
endif()

# Process all kernels marked for inlining
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/KernelEmbedding.cmake")
# Generate embedded kernel sources and includes at configure time
embed_kernel_sources(
    TARGET hip_kernel_provider_impl
    OUTPUT_SRCS_CPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    OUTPUT_SRCS_HPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.hpp
    OUTPUT_INCS_CPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp
    OUTPUT_INCS_HPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.hpp
)

set(HIP_KERNEL_PROVIDER_GENERATED_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp)

set_source_files_properties(${HIP_KERNEL_PROVIDER_GENERATED_SOURCES} TARGET_DIRECTORY hip_kernel_provider_impl PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")

target_sources(hip_kernel_provider_impl PRIVATE "${HIP_KERNEL_PROVIDER_GENERATED_SOURCES}")
