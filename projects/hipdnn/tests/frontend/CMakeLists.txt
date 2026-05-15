# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    hipdnn_public_frontend_tests
    main.cpp
    IntegrationBatchnormBackwardDescriptorLifting.cpp
    IntegrationBatchnormBackwardDescriptorLowering.cpp
    IntegrationBatchnormDescriptorLifting.cpp
    IntegrationBatchnormDescriptorLowering.cpp
    IntegrationBatchnormForwardInference.cpp
    IntegrationBatchnormForwardInferenceVarianceExt.cpp
    IntegrationBatchnormInferenceDescriptorLifting.cpp
    IntegrationBatchnormInferenceDescriptorLowering.cpp
    IntegrationBatchnormInferenceNodeVarianceExtDescriptorLifting.cpp
    IntegrationBatchnormInferenceNodeVarianceExtDescriptorLowering.cpp
    IntegrationBlockScaleDequantize.cpp
    IntegrationBlockScaleDequantizeDescriptorLifting.cpp
    IntegrationBlockScaleDequantizeDescriptorLowering.cpp
    IntegrationBlockScaleQuantize.cpp
    IntegrationBlockScaleQuantizeDescriptorLifting.cpp
    IntegrationBlockScaleQuantizeDescriptorLowering.cpp
    IntegrationConvForward.cpp
    IntegrationConvFpropDescriptorLifting.cpp
    IntegrationConvFpropDescriptorLowering.cpp
    IntegrationConvolutionBackwardData.cpp
    IntegrationConvolutionBackwardWeight.cpp
    IntegrationConvolutionDgradDescriptorLowering.cpp
    IntegrationConvolutionDgradDescriptorLifting.cpp
    IntegrationConvolutionWgradDescriptorLifting.cpp
    IntegrationConvolutionWgradDescriptorLowering.cpp
    IntegrationCustomOpDescriptorLifting.cpp
    IntegrationCustomOpDescriptorLowering.cpp
    IntegrationFrontendUserLogging.cpp
    IntegrationGraphEngineFiltering.cpp
    IntegrationGraphKnobsApi.cpp
    IntegrationGraphKnobSettingsDescriptorLowering.cpp
    IntegrationGraphKnobsDescriptorLifting.cpp
    IntegrationGraphLifting.cpp
    IntegrationGraphSupportCheck.cpp
    IntegrationLayerNormDescriptorLifting.cpp
    IntegrationLayerNormDescriptorLowering.cpp
    IntegrationLayernormForward.cpp
    IntegrationMatmul.cpp
    IntegrationMatmulDescriptorLifting.cpp
    IntegrationMatmulDescriptorLowering.cpp
    IntegrationPointwiseDescriptorLifting.cpp
    IntegrationPointwiseDescriptorLowering.cpp
    IntegrationResampleFwdDescriptorLifting.cpp
    IntegrationResampleFwdDescriptorLowering.cpp
    IntegrationReductionDescriptorLifting.cpp
    IntegrationReductionDescriptorLowering.cpp
    IntegrationRMSNormDescriptorLifting.cpp
    IntegrationRMSNormDescriptorLowering.cpp
    IntegrationRMSNormForward.cpp
    IntegrationRMSNormBackwardDescriptorLowering.cpp
    IntegrationRMSNormBackwardDescriptorLifting.cpp
    IntegrationSetPluginPathsExt.cpp
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:IntegrationSdpaBwdDescriptorLowering.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:IntegrationSdpaBwdLifting.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:IntegrationSdpaFwdDescriptorLifting.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:IntegrationSdpaFwdDescriptorLowering.cpp>
)

target_compile_options(hipdnn_public_frontend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(
    hipdnn_public_frontend_tests
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    hipdnn_test_sdk
    hipdnn_backend
    hip::host
    ${CMAKE_DL_LIBS}
)

target_include_directories(hipdnn_public_frontend_tests PRIVATE ${HIPDNN_TEST_INCLUDE_DIRS})

target_compile_definitions(
    hipdnn_public_frontend_tests
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
    hipdnn_public_frontend_tests
    test_good_plugin
    test_good_default_plugin
    test_execute_fails_plugin
    test_no_applicable_engines_a_plugin
    test_no_applicable_engines_b_plugin
    test_duplicate_id_a_plugin
    test_duplicate_id_b_plugin
    test_incomplete_api_plugin
    test_knobs_plugin
    test_knob_constraint_validation_plugin
)

clang_tidy_check(hipdnn_public_frontend_tests)
add_integration_test_target(hipdnn_public_frontend_tests ${CMAKE_CURRENT_BINARY_DIR})
