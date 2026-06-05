# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping Test SDK tests")
    return()
endif()

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    hipdnn_test_sdk_tests
    main.cpp
    utilities/TestCpuFpReferenceBatchnorm.cpp
    utilities/TestCpuFpReferenceBatchnormWithVariance.cpp
    utilities/TestCpuFpReferenceBlockScaleDequantize.cpp
    utilities/TestCpuFpReferenceRMSNorm.cpp
    utilities/TestCpuFpReferenceSdpa.cpp
    utilities/TestCpuFpReferenceConvolution.cpp
    utilities/TestCpuFpReferenceLayernorm.cpp
    utilities/TestCpuFpReferenceMatmul.cpp
    utilities/TestCpuFpReferenceReduction.cpp
    utilities/TestCpuFpReferenceMiopenRmsValidation.cpp
    utilities/TestCpuFpReferenceUtilities.cpp
    utilities/TestCpuFpReferenceValidation.cpp
    utilities/TestCpuTypeUtilities.cpp
    utilities/TestSdkFrontendTypeConversions.cpp
    utilities/TestCpuReferencePointwise.cpp
    utilities/TestDynamicTolerancesBatchNorm.cpp
    utilities/TestDynamicTolerancesCommon.cpp
    utilities/TestDynamicTolerancesConv.cpp
    utilities/TestDynamicTolerancesLayerNorm.cpp
    utilities/TestDynamicTolerancesMatmul.cpp
    utilities/TestDynamicTolerancesPointwise.cpp
    utilities/TestDynamicTolerancesRMSNorm.cpp
    utilities/TestDynamicTolerancesSdpa.cpp
    utilities/TestGraphTestHelpers.cpp
    utilities/TestFlatbufferDatatypeMapping.cpp
    utilities/TestFlatbufferTensorAttributesUtils.cpp
    utilities/TestLoadGraphAndTensors.cpp
    utilities/TestLoggingUtils.cpp
    utilities/TestLogRecorder.cpp
    utilities/TestScopedDirectory.cpp
    utilities/TestScopedEnvironmentVariableSetter.cpp
    utilities/TestSeeds.cpp
    utilities/TestTensorDiff.cpp
    utilities/TestVectorLoggingUtils.cpp
    utilities/cpu_graph_executor/TestBlockScaleDequantizePlan.cpp
    utilities/cpu_graph_executor/TestBlockScaleDequantizeSignatureKey.cpp
    utilities/cpu_graph_executor/TestBatchnormBwdPlan.cpp
    utilities/cpu_graph_executor/TestBatchnormBwdSignatureKey.cpp
    utilities/cpu_graph_executor/TestBatchnormFwdInferencePlan.cpp
    utilities/cpu_graph_executor/TestBatchnormFwdInferenceSignatureKey.cpp
    utilities/cpu_graph_executor/TestBatchnormFwdInferenceWithVariancePlan.cpp
    utilities/cpu_graph_executor/TestBatchnormTrainPlan.cpp
    utilities/cpu_graph_executor/TestBatchnormTrainSignatureKey.cpp
    utilities/cpu_graph_executor/TestConvolutionBwdPlan.cpp
    utilities/cpu_graph_executor/TestConvolutionBwdSignatureKey.cpp
    utilities/cpu_graph_executor/TestConvolutionWrwPlan.cpp
    utilities/cpu_graph_executor/TestConvolutionWrwSignatureKey.cpp
    utilities/cpu_graph_executor/TestConvolutionFwdPlan.cpp
    utilities/cpu_graph_executor/TestConvolutionFwdSignatureKey.cpp
    utilities/cpu_graph_executor/TestCpuReferenceGraphExecutor.cpp
    utilities/cpu_graph_executor/TestFusedOperationsCpuGraphExecutor.cpp
    utilities/cpu_graph_executor/TestLayernormFpropPlan.cpp
    utilities/cpu_graph_executor/TestLayernormFpropSignatureKey.cpp
    utilities/cpu_graph_executor/TestMatmulPlan.cpp
    utilities/cpu_graph_executor/TestMatmulSignatureKey.cpp
    utilities/cpu_graph_executor/TestPointwiseOperationsCpuGraphExecutor.cpp
    utilities/cpu_graph_executor/TestGraphTensorBundle.cpp
    utilities/cpu_graph_executor/TestPlanRegistrySignatureKey.cpp
    utilities/cpu_graph_executor/TestPointwisePlan.cpp
    utilities/cpu_graph_executor/TestPointwiseSignatureKey.cpp
    utilities/cpu_graph_executor/TestReductionPlan.cpp
    utilities/cpu_graph_executor/TestReductionSignatureKey.cpp
    utilities/cpu_graph_executor/TestRMSNormFwdPlan.cpp
    utilities/cpu_graph_executor/TestRMSNormFwdSignatureKey.cpp
    utilities/cpu_graph_executor/TestRMSNormBwdPlan.cpp
    utilities/cpu_graph_executor/TestRMSNormBwdSignatureKey.cpp
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:utilities/cpu_graph_executor/TestSdpaFwdPlan.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:utilities/cpu_graph_executor/TestSdpaFwdSignatureKey.cpp>
    TestVersion.cpp
)

target_compile_options(hipdnn_test_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hipdnn_test_sdk_tests GTest::gtest GTest::gmock Threads::Threads hipdnn_test_sdk hipdnn_plugin_sdk
)

clang_tidy_check(hipdnn_test_sdk_tests)
add_unit_test_target(hipdnn_test_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
