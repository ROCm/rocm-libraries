# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping SDK tests")
    return()
endif()

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)
enable_language(HIP)

add_executable(hipdnn_sdk_tests
    main.cpp
    plugin/TestPluginDataTypeHelpers.cpp
    plugin/TestPluginException.cpp
    plugin/TestPluginHelpers.cpp
    plugin/TestPluginLastErrorManager.cpp
    plugin/flatbuffer_utilities/TestEngineConfigWrapper.cpp
    plugin/flatbuffer_utilities/TestEngineDetailsWrapper.cpp
    plugin/flatbuffer_utilities/TestGraphWrapper.cpp
    plugin/flatbuffer_utilities/TestNodeWrapper.cpp
    test_utilities/TestCpuFpReferenceBatchnorm.cpp
    test_utilities/TestCpuFpReferenceConvolution.cpp
    test_utilities/TestCpuFpReferenceMiopenRmsValidation.cpp
    test_utilities/TestCpuFpReferenceUtilities.cpp
    test_utilities/TestCpuFpReferenceValidation.cpp
    test_utilities/TestCpuReferencePointwise.cpp
    test_utilities/TestFlatbufferDatatypeMapping.cpp
    test_utilities/TestFlatbufferTensorAttributesUtils.cpp
    test_utilities/TestLoggingUtils.cpp
    test_utilities/TestScopedDirectory.cpp
    test_utilities/TestScopedEnvironmentVariableSetter.cpp
    test_utilities/TestSeeds.cpp
    test_utilities/TestVectorLoggingUtils.cpp
    test_utilities/cpu_graph_executor/TestBatchnormBwdPlan.cpp
    test_utilities/cpu_graph_executor/TestBatchnormBwdSignatureKey.cpp
    test_utilities/cpu_graph_executor/TestBatchnormFwdInferencePlan.cpp
    test_utilities/cpu_graph_executor/TestBatchnormFwdInferenceSignatureKey.cpp
    test_utilities/cpu_graph_executor/TestBatchnormTrainPlan.cpp
    test_utilities/cpu_graph_executor/TestBatchnormTrainSignatureKey.cpp
    test_utilities/cpu_graph_executor/TestConvolutionBwdPlan.cpp
    test_utilities/cpu_graph_executor/TestConvolutionBwdSignatureKey.cpp
    test_utilities/cpu_graph_executor/TestConvolutionFwdPlan.cpp
    test_utilities/cpu_graph_executor/TestConvolutionFwdSignatureKey.cpp
    test_utilities/cpu_graph_executor/TestCpuReferenceGraphExecutor.cpp
    test_utilities/cpu_graph_executor/TestFusedOperationsCpuGraphExecutor.cpp
    test_utilities/cpu_graph_executor/TestGraphTensorBundle.cpp
    test_utilities/cpu_graph_executor/TestPlanRegistrySignatureKey.cpp
    test_utilities/cpu_graph_executor/TestPointwisePlan.cpp
    test_utilities/cpu_graph_executor/TestPointwiseSignatureKey.cpp
    utilities/TestAllocators.cpp
    utilities/TestCallbackLogger.cpp
    utilities/TestFloats.cpp
    utilities/TestJson.cpp
    utilities/TestLoadGraphAndTensors.cpp
    utilities/TestMigratableMemory.cpp
    utilities/TestPlatformUtils.cpp
    utilities/TestPointwiseValidation.cpp
    utilities/TestScopedResource.cpp
    utilities/TestShallowHostOnlyMigratableMemory.cpp
    utilities/TestShallowTensor.cpp
    utilities/TestShapeUtilities.cpp
    utilities/TestTensor.cpp
    utilities/TestTensorIterator.cpp
    utilities/TestTensorView.cpp
)

target_compile_options(hipdnn_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(hipdnn_sdk_tests 
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_sdk
    hipdnn_frontend
    hip::host
    ${CMAKE_DL_LIBS}
)

clang_tidy_check(hipdnn_sdk_tests)
add_unit_test_target(hipdnn_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
