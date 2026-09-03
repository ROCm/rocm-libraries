# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# NOTE: FrontendGraphEngineRunner.cpp, ReferenceExecutorPool.cpp and
# HarnessDependencies.cpp are deliberately absent. They are the only places that
# touch the frontend Graph, a hipdnnHandle_t, or a real reference executor, so
# leaving them out makes "these unit tests never reach an engine or a device" a
# link-time guarantee rather than a convention.
add_executable(hipdnn_integration_tests_unit_tests
    main.cpp
    ../src/harness/bundle/BundleReferenceValidationHarness.cpp
    ../src/harness/bundle/GraphSession.cpp
    ../src/harness/bundle/IntegrationBundleVerificationHarness.cpp
    ../src/harness/bundle/OutputComparison.cpp
    ../src/harness/bundle/ProductionPolicy.cpp
    ../src/harness/bundle/ReferenceOpCoverage.cpp
    ../src/harness/bundle/SupportClaimReport.cpp
    ../src/harness/bundle/SupportClaims.cpp
    ../src/harness/bundle/SupportVerdict.cpp
    ../src/harness/bundle/VariantPackBuilder.cpp
    ../src/harness/input-init/FillInputs.cpp
    TestBundleMetadata.cpp
    TestSupportClaims.cpp
    TestGraphDescription.cpp
    TestNodeTypeToString.cpp
    TestPlatformUtils.cpp
    TestSupportMatrixCollector.cpp
    TestTestConfig.cpp
    TestTestSettings.cpp
    TestTomlGuards.cpp
    TestGpuReferenceGraphExecutor.cpp
    TestGpuConvolutionFwdSignatureKey.cpp
    TestGpuConvolutionFwdPlan.cpp
    TestGpuSdpaFwdSignatureKey.cpp
    TestGpuSdpaFwdPlan.cpp
    TestGpuRMSNormSignatureKey.cpp
    TestGpuRMSNormPlan.cpp
    TestReferenceGraphExecutorFactory.cpp
    TestBundleDiscovery.cpp
    TestVerificationPaths.cpp
    TestBundleVerificationHarness.cpp
    TestInputFillRecipes.cpp
    TestFillInputs.cpp
    TestToleranceResolver.cpp
    TestVerificationModePaths.cpp
    TestErrorPaths.cpp
    TestLoadedEngineTable.cpp
    TestReferenceOpCoverage.cpp
    TestSupportClaimReport.cpp
    TestSupportClaimEnforcement.cpp
    TestEnforcementRungs.cpp
    TestSupportVerdict.cpp
    TestUnpinnedRunEnforcement.cpp
    TestBundleReferenceValidationHarness.cpp
    TestProductionPolicy.cpp
    TestGraphSession.cpp
    TestOutputComparison.cpp
)

target_include_directories(hipdnn_integration_tests_unit_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_include_directories(hipdnn_integration_tests_unit_tests SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/third-party/tomlplusplus)

target_link_libraries(hipdnn_integration_tests_unit_tests
    PRIVATE
        hip::host
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gmock
        GTest::gtest
        Threads::Threads
        $<$<PLATFORM_ID:Windows>:shlwapi>
)

target_compile_options(hipdnn_integration_tests_unit_tests
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(hipdnn_integration_tests_unit_tests)

add_unit_test_target(hipdnn_integration_tests_unit_tests ${CMAKE_CURRENT_BINARY_DIR})

add_executable(hipdnn_gpu_ref_tests
    gpu-ref/main.cpp
    gpu-ref/TestGpuFpReferencePointwise.cpp
    gpu-ref/TestGpuFpReferenceConvolution.cpp
    gpu-ref/TestGpuFpReferenceSdpa.cpp
    gpu-ref/TestGpuFpReferenceDgrad.cpp
    gpu-ref/TestGpuFpReferenceLayernormBwd.cpp
    gpu-ref/TestGpuFpReferenceLayernormFwd.cpp
    gpu-ref/TestGpuFpReferenceWgrad.cpp
    gpu-ref/TestGpuFpReferenceRMSNormFwd.cpp
    gpu-ref/TestGpuFpReferenceRMSNormBwd.cpp
    gpu-ref/TestGpuFpReferenceValidation.cpp
    gpu-ref/TestGpuRefHipError.cpp
    gpu-ref/TestGpuRefKernelCompiler.cpp
    gpu-ref/TestShallowDeviceOnlyMigratableMemory.cpp
    gpu-ref/TestShallowGpuTensor.cpp
    gpu-ref/TestGpuFpReferenceCommon.cpp
    gpu-ref/TestGpuFpReferenceBatchnormFwdInf.cpp
)

target_link_libraries(hipdnn_gpu_ref_tests
    PRIVATE
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gtest
        Threads::Threads
        hip::host
        $<$<BOOL:${USE_ROCRAND}>:roc::rocrand>
)

target_compile_options(hipdnn_gpu_ref_tests
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(hipdnn_gpu_ref_tests)

add_integration_test_target(hipdnn_gpu_ref_tests ${CMAKE_CURRENT_BINARY_DIR})

if(USE_ROCRAND)
    target_compile_definitions(hipdnn_gpu_ref_tests PRIVATE USE_ROCRAND)
endif()
