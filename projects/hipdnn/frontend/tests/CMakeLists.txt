# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Frontend Unit tests")
    return()
endif()

find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    hipdnn_frontend_tests
    main.cpp
    TestAttributes.cpp
    TestBackendInterface.cpp
    TestBackendLoggingHelpers.cpp
    TestBlockScaleDequantizeAttributes.cpp
    TestBlockScaleDequantizeNode.cpp
    TestBlockScaleQuantizeAttributes.cpp
    TestBlockScaleQuantizeNode.cpp
    TestBatchnormAttributes.cpp
    TestBatchnormBackwardAttributes.cpp
    TestBatchnormBackwardNode.cpp
    TestBatchnormInferenceAttributes.cpp
    TestBatchnormInferenceAttributesVarianceExt.cpp
    TestBatchnormInferenceNode.cpp
    TestBatchnormInferenceNodeVarianceExt.cpp
    TestBatchnormNode.cpp
    TestCustomOpAttributes.cpp
    TestCustomOpNode.cpp
    TestConvolutionDgradAttributes.cpp
    TestConvolutionFwdAttributes.cpp
    TestConvolutionDgradNode.cpp
    TestConvolutionNode.cpp
    TestConvolutionWgradAttributes.cpp
    TestDescriptorHelpers.cpp
    TestDescriptorUnpackHelpers.cpp
    TestConvolutionWgradNode.cpp
    TestEngineOverrideConfig.cpp
    TestError.cpp
    TestFrontendLogging.cpp
    TestGraph.cpp
    TestGraphAttributes.cpp
    TestGraphOverrideValidation.cpp
    TestHandle.cpp
    TestIncompatibleBackendWrapper.cpp
    TestINode.cpp
    TestKnob.cpp
    TestKnobUnpacker.cpp
    TestLayernormAttributes.cpp
    TestLayerNormNode.cpp
    TestMatmulAttributes.cpp
    TestMatmulNode.cpp
    TestNode.cpp
    TestOperationUnpacker.cpp
    TestSdpaAttributes.cpp
    TestSdpaBackwardAttributes.cpp
    TestSdpaBwdNode.cpp
    TestSdpaFwdNode.cpp
    TestPointwiseAttributes.cpp
    TestPointwiseNode.cpp
    TestReductionAttributes.cpp
    TestReductionNode.cpp
    TestRMSNormAttributes.cpp
    TestRMSNormNode.cpp
    TestScopedHipdnnBackendDescriptor.cpp
    TestSetEnginePluginPaths.cpp
    TestTensorAttributes.cpp
    TestTensorValueAttributes.cpp
    TestTypes.cpp
    TestUtilities.cpp
    TestTopologicalSortingUtils.cpp
    TestSdkLogging.cpp
    TestVersion.cpp
)

target_compile_options(hipdnn_frontend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hipdnn_frontend_tests GTest::gtest GTest::gmock hip::host Threads::Threads hipdnn_test_sdk
)
clang_tidy_check(hipdnn_frontend_tests)

add_unit_test_target(hipdnn_frontend_tests ${CMAKE_CURRENT_BINARY_DIR})
