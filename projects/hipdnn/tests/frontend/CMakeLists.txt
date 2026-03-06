# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    public_hipdnn_frontend_tests
    main.cpp
    IntegrationBatchnormForwardInference.cpp
    IntegrationBatchnormForwardInferenceVarianceExt.cpp
    IntegrationBlockScaleDequantize.cpp
    IntegrationBlockScaleQuantize.cpp
    IntegrationConvForward.cpp
    IntegrationConvolutionBackwardData.cpp
    IntegrationConvolutionBackwardWeight.cpp
    IntegrationFrontendUserLogging.cpp
    IntegrationGraphEngineFiltering.cpp
    IntegrationGraphKnobsApi.cpp
    IntegrationConvFpropDescriptorLowering.cpp
    IntegrationConvolutionDgradDescriptorLowering.cpp
    IntegrationConvolutionWgradDescriptorLowering.cpp
    IntegrationLayernormForward.cpp
    IntegrationMatmul.cpp
    IntegrationRMSNormForward.cpp
)

target_compile_options(public_hipdnn_frontend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(
    public_hipdnn_frontend_tests
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    hipdnn_test_sdk
    hipdnn_backend
    hip::host
)

target_include_directories(public_hipdnn_frontend_tests PRIVATE ${HIPDNN_TEST_INCLUDE_DIRS})

target_compile_definitions(
    public_hipdnn_frontend_tests
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
)

# Ensure test plugins are built before tests
add_dependencies(
    public_hipdnn_frontend_tests
    test_good_plugin
    test_execute_fails_plugin
    test_no_applicable_engines_a_plugin
    test_no_applicable_engines_b_plugin
    test_duplicate_id_a_plugin
    test_duplicate_id_b_plugin
    test_incomplete_api_plugin
    test_knobs_plugin
    test_knob_constraint_validation_plugin
)

clang_tidy_check(public_hipdnn_frontend_tests)
add_integration_test_target(public_hipdnn_frontend_tests ${CMAKE_CURRENT_BINARY_DIR})
