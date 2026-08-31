# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    hipdnn_public_frontend_tests
    main.cpp
    IntegrationHipdnnDirectBackendWrapper.cpp
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
    IntegrationHeuristicPolicyEnumeration.cpp
    IntegrationLayerNormDescriptorLifting.cpp
    IntegrationLayerNormDescriptorLowering.cpp
    IntegrationLayernormForward.cpp
    IntegrationLayernormBackward.cpp
    IntegrationLayernormBackwardDescriptorLowering.cpp
    IntegrationLayernormBackwardDescriptorLifting.cpp
    IntegrationMalformedVersionPlugin.cpp
    IntegrationMatmul.cpp
    IntegrationMatmulDescriptorLifting.cpp
    IntegrationMatmulDescriptorLowering.cpp
    IntegrationMoeGroupedMatmul.cpp
    IntegrationMoeGroupedMatmulBwd.cpp
    IntegrationMoeGroupedMatmulBwdDescriptorLifting.cpp
    IntegrationMoeGroupedMatmulBwdDescriptorLowering.cpp
    IntegrationMoeGroupedMatmulDescriptorLifting.cpp
    IntegrationMoeGroupedMatmulDescriptorLowering.cpp
    IntegrationPointwiseDescriptorLifting.cpp
    IntegrationPointwiseDescriptorLowering.cpp
    IntegrationRaggedTensorDescriptorLowering.cpp
    IntegrationPassByValueVersionFilter.cpp
    IntegrationPassByValueDelivery.cpp
    IntegrationResampleFwdDescriptorLifting.cpp
    IntegrationResampleFwdDescriptorLowering.cpp
    IntegrationResampleBwdDescriptorLifting.cpp
    IntegrationResampleBwdDescriptorLowering.cpp
    IntegrationReductionDescriptorLifting.cpp
    IntegrationReductionDescriptorLowering.cpp
    IntegrationRMSNormDescriptorLifting.cpp
    IntegrationRMSNormDescriptorLowering.cpp
    IntegrationRMSNormForward.cpp
    IntegrationRMSNormBackwardDescriptorLowering.cpp
    IntegrationRMSNormBackwardDescriptorLifting.cpp
    IntegrationAutotuneCompiledPlanPath.cpp
    IntegrationAutotuneConfigRoundTrip.cpp
    IntegrationAutotuneConfigRoundTripUnsupportedOp.cpp
    IntegrationAutotuneEndToEnd.cpp
    IntegrationAutotuneExhaustive.cpp
    IntegrationAutotunePersistence.cpp
    IntegrationSetPluginPathsExt.cpp
    # cuDNN-compatibility shim integration coverage
    # only built when the shim is enabled.
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:IntegrationCudnnShimUmbrella.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:IntegrationCudnnShimHandle.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:IntegrationOverrideExecute.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:IntegrationOverrideValidation.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:IntegrationTestPluginInfrastructure.cpp>
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
    hipdnn_plugin_sdk
    hip::host
    ${CMAKE_DL_LIBS}
)

# configures an integration test target that consumes the hipdnn_frontend library
# or the hipdnn_frontend_dynamic library
function(configure_hipdnn_public_frontend_test_target TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE ${HIPDNN_TEST_INCLUDE_DIRS})

    target_compile_definitions(
        ${TARGET_NAME}
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
                TEST_HASHED_NAME_PLUGIN_NAME="${TEST_HASHED_NAME_PLUGIN_NAME}"
                TEST_OVERRIDE_IMPLEMENTING_PLUGIN_NAME="${HIPDNN_TEST_OVERRIDE_IMPLEMENTING_PLUGIN_TARGET}"
                TEST_OVERRIDE_OMITTING_PLUGIN_NAME="${HIPDNN_TEST_OVERRIDE_OMITTING_PLUGIN_TARGET}"
                TEST_VERSION_LIAR_PLUGIN_NAME="${HIPDNN_TEST_VERSION_LIAR_PLUGIN_TARGET}"
                TEST_SECOND_OVERRIDE_PLUGIN_NAME="${HIPDNN_TEST_SECOND_OVERRIDE_PLUGIN_TARGET}"
                TEST_MALFORMED_VERSION_PLUGIN_NAME="${HIPDNN_TEST_MALFORMED_VERSION_PLUGIN_TARGET}"
                TEST_VERSION_ZERO_PLUGIN_NAME="${HIPDNN_TEST_VERSION_ZERO_PLUGIN_TARGET}"
                TEST_PASS_BY_VALUE_PLUGIN_NAME="${HIPDNN_TEST_PASS_BY_VALUE_PLUGIN_TARGET}"
                TEST_PASS_BY_VALUE_RECORDER_PLUGIN_NAME="${HIPDNN_TEST_PASS_BY_VALUE_RECORDER_PLUGIN_TARGET}"
                TEST_GOOD_HEURISTIC_PLUGIN_NAME="${TEST_GOOD_HEURISTIC_PLUGIN_NAME}"
                TEST_AUTOTUNE_PLUGIN_NAME="${TEST_AUTOTUNE_PLUGIN_NAME}"
    )

    # Ensure test plugins are built before tests
    add_dependencies(
        ${TARGET_NAME}
        ${TEST_GOOD_PLUGIN_NAME}
        ${TEST_GOOD_DEFAULT_PLUGIN_NAME}
        ${TEST_EXECUTE_FAILS_PLUGIN_NAME}
        ${TEST_NO_APPLICABLE_ENGINES_A_PLUGIN_NAME}
        ${TEST_NO_APPLICABLE_ENGINES_B_PLUGIN_NAME}
        ${TEST_DUPLICATE_ID_A_PLUGIN_NAME}
        ${TEST_DUPLICATE_ID_B_PLUGIN_NAME}
        ${TEST_INCOMPLETE_API_PLUGIN_NAME}
        ${TEST_HASHED_NAME_PLUGIN_NAME}
        ${TEST_KNOBS_PLUGIN_NAME}
        ${TEST_KNOB_CONSTRAINT_VALIDATION_PLUGIN_NAME}
        # Override-execute fake plugins.
        ${HIPDNN_TEST_OVERRIDE_IMPLEMENTING_PLUGIN_TARGET}
        ${HIPDNN_TEST_OVERRIDE_OMITTING_PLUGIN_TARGET}
        ${HIPDNN_TEST_VERSION_LIAR_PLUGIN_TARGET}
        ${HIPDNN_TEST_SECOND_OVERRIDE_PLUGIN_TARGET}
        ${HIPDNN_TEST_MALFORMED_VERSION_PLUGIN_TARGET}
        ${HIPDNN_TEST_VERSION_ZERO_PLUGIN_TARGET}
        ${HIPDNN_TEST_PASS_BY_VALUE_PLUGIN_TARGET}
        ${HIPDNN_TEST_PASS_BY_VALUE_RECORDER_PLUGIN_TARGET}
        ${TEST_GOOD_HEURISTIC_PLUGIN_NAME}
        test_autotune_plugin
    )
endfunction()

configure_hipdnn_public_frontend_test_target(hipdnn_public_frontend_tests)

clang_tidy_check(hipdnn_public_frontend_tests)
add_hipdnn_test(hipdnn_public_frontend_tests ${CMAKE_CURRENT_BINARY_DIR})

add_executable(
    hipdnn_public_frontend_dynamic_load_tests
    main_dynamic.cpp
    IntegrationHipdnnDynamicBackendWrapper.cpp
)

target_compile_options(
    hipdnn_public_frontend_dynamic_load_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS}
)
target_link_libraries(
    hipdnn_public_frontend_dynamic_load_tests
    GTest::gtest
    Threads::Threads
    hipdnn_test_sdk_dynamic
    hipdnn_plugin_sdk
    hip::host
    ${CMAKE_DL_LIBS}
)

configure_hipdnn_public_frontend_test_target(hipdnn_public_frontend_dynamic_load_tests)
add_dependencies(hipdnn_public_frontend_dynamic_load_tests hipdnn_backend)

clang_tidy_check(hipdnn_public_frontend_dynamic_load_tests)
add_hipdnn_test(hipdnn_public_frontend_dynamic_load_tests ${CMAKE_CURRENT_BINARY_DIR})
