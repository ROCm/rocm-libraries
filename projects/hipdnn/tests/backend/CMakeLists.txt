# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

include(${PROJECT_SOURCE_DIR}/tests/test_plugins/cmake/TestPluginTargets.cmake)

# Main integration test executable WITH HipErrorHandler
add_executable(
    hipdnn_public_backend_tests
    main.cpp
    IntegrationBackendDescriptor.cpp
    IntegrationBackendExecuteApi.cpp
    IntegrationConvOperationDescriptorApi.cpp
    IntegrationGraphDescriptorApi.cpp
    IntegrationEngineApi.cpp
    IntegrationEngineConfigApi.cpp
    IntegrationEngineHeuristicApi.cpp
    IntegrationExecutionPlanApi.cpp
    IntegrationGetLastErrorString.cpp
    IntegrationHandleApi.cpp
    IntegrationKnobsApi.cpp
    IntegrationLoggingPipeline.cpp
    IntegrationPluginLoading.cpp
    IntegrationPluginLogLevel.cpp
    IntegrationSetPluginPathsExt.cpp
    IntegrationSetPluginUnloadingModeExt.cpp
    IntegrationTensorDescriptorApi.cpp
    IntegrationVariantPackApi.cpp
    TestUtil.cpp
)

target_compile_options(hipdnn_public_backend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(
    hipdnn_public_backend_tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        hipdnn_test_sdk
        hipdnn_flatbuffers_sdk
        hipdnn_plugin_sdk
        hipdnn_backend
        hip::host
)

# Enable spdlog support (tests include backend internal headers that use spdlog/fmt)
hipdnn_enable_spdlog(hipdnn_public_backend_tests)

target_include_directories(hipdnn_public_backend_tests PRIVATE ${HIPDNN_TEST_INCLUDE_DIRS})

target_compile_definitions(
    hipdnn_public_backend_tests
    PRIVATE TEST_GOOD_PLUGIN_NAME="${TEST_GOOD_PLUGIN_NAME}"
            TEST_EXECUTE_FAILS_PLUGIN_NAME="${TEST_EXECUTE_FAILS_PLUGIN_NAME}"
            TEST_NO_APPLICABLE_ENGINES_A_PLUGIN_NAME="${TEST_NO_APPLICABLE_ENGINES_A_PLUGIN_NAME}"
            TEST_NO_APPLICABLE_ENGINES_B_PLUGIN_NAME="${TEST_NO_APPLICABLE_ENGINES_B_PLUGIN_NAME}"
            TEST_DUPLICATE_ID_A_PLUGIN_NAME="${TEST_DUPLICATE_ID_A_PLUGIN_NAME}"
            TEST_DUPLICATE_ID_B_PLUGIN_NAME="${TEST_DUPLICATE_ID_B_PLUGIN_NAME}"
            TEST_INCOMPLETE_API_PLUGIN_NAME="${TEST_INCOMPLETE_API_PLUGIN_NAME}"
            TEST_GOOD_DEFAULT_PLUGIN_NAME="${TEST_GOOD_DEFAULT_PLUGIN_NAME}"
            TEST_KNOBS_PLUGIN_NAME="${TEST_KNOBS_PLUGIN_NAME}"
            TEST_KNOB_CONSTRAINT_VALIDATION_PLUGIN_NAME="${TEST_KNOB_CONSTRAINT_VALIDATION_PLUGIN_NAME}"
            TEST_INCOMPATIBLE_VERSION_PLUGIN_NAME="${TEST_INCOMPATIBLE_VERSION_PLUGIN_NAME}"
            TEST_OVERRIDE_IMPLEMENTING_PLUGIN_NAME="${HIPDNN_TEST_OVERRIDE_IMPLEMENTING_PLUGIN_TARGET}"
            TEST_OVERRIDE_OMITTING_PLUGIN_NAME="${HIPDNN_TEST_OVERRIDE_OMITTING_PLUGIN_TARGET}"
            TEST_VERSION_LIAR_PLUGIN_NAME="${HIPDNN_TEST_VERSION_LIAR_PLUGIN_TARGET}"
            TEST_SECOND_OVERRIDE_PLUGIN_NAME="${HIPDNN_TEST_SECOND_OVERRIDE_PLUGIN_TARGET}"
            TEST_MALFORMED_VERSION_PLUGIN_NAME="${HIPDNN_TEST_MALFORMED_VERSION_PLUGIN_TARGET}"
            TEST_VERSION_ZERO_PLUGIN_NAME="${HIPDNN_TEST_VERSION_ZERO_PLUGIN_TARGET}"
)

# Ensure test plugins are built before tests
add_dependencies(
    hipdnn_public_backend_tests
    test_good_plugin
    test_execute_fails_plugin
    test_no_applicable_engines_a_plugin
    test_no_applicable_engines_b_plugin
    test_good_default_plugin
    test_duplicate_id_a_plugin
    test_duplicate_id_b_plugin
    test_incomplete_api_plugin
    test_knobs_plugin
    test_knob_constraint_validation_plugin
    test_incompatible_version_plugin
    ${HIPDNN_TEST_OVERRIDE_IMPLEMENTING_PLUGIN_TARGET}
    ${HIPDNN_TEST_OVERRIDE_OMITTING_PLUGIN_TARGET}
    ${HIPDNN_TEST_VERSION_LIAR_PLUGIN_TARGET}
    ${HIPDNN_TEST_SECOND_OVERRIDE_PLUGIN_TARGET}
    ${HIPDNN_TEST_MALFORMED_VERSION_PLUGIN_TARGET}
    ${HIPDNN_TEST_VERSION_ZERO_PLUGIN_TARGET}
)

clang_tidy_check(hipdnn_public_backend_tests)
add_integration_test_target(hipdnn_public_backend_tests ${CMAKE_CURRENT_BINARY_DIR})
