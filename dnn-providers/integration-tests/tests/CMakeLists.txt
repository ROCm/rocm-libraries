# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# NOTE: FrontendGraphEngineRunner.cpp, ReferenceExecutorPool.cpp and
# HarnessDependencies.cpp are deliberately absent. They are the only places that
# touch the frontend Graph or a hipdnnHandle_t, or hand the harness a pooled reference
# executor. CpuReferenceGraphExecutorAdapter.cpp, GpuReferenceGraphExecutor.cpp and
# ReferenceGraphExecutorFactory.cpp ARE linked: the direct executor tests construct
# reference executors themselves.
# Leaving the pool, HarnessDependencies.cpp and GoldenDataRegistration.cpp out keeps
# "harness code paths cannot get a pooled executor" a link-time guarantee rather than
# a convention. GoldenDataRegistration.cpp is absent for the same reason: it registers
# gtest tests and reaches the shared reference executors. Its decisions live in
# GoldenDataPlan.cpp and GoldenOutputProbe.cpp, which are linked and tested here.
add_executable(hipdnn_integration_tests_unit_tests
    main.cpp
    ../src/harness/CpuReferenceGraphExecutorAdapter.cpp
    ../src/harness/ReferenceGraphExecutorFactory.cpp
    ../src/harness/bundle/GraphSession.cpp
    ../src/harness/bundle/IntegrationBundleVerificationHarness.cpp
    ../src/harness/bundle/OutputComparison.cpp
    ../src/harness/bundle/ProductionPolicy.cpp
    ../src/harness/bundle/SupportClaimReport.cpp
    ../src/harness/bundle/SupportClaimWriter.cpp
    ../src/harness/bundle/SupportClaims.cpp
    ../src/harness/bundle/SupportVerdict.cpp
    ../src/harness/bundle/SweepManifestCache.cpp
    ../src/harness/bundle/VariantPackBuilder.cpp
    ../src/harness/gpu-graph-executor/GpuReferenceGraphExecutor.cpp
    ../src/harness/input-init/FillInputs.cpp
    ../src/harness/input-init/ResampleBackwardInputs.cpp
    ../src/harness/input-init/SdpaBackwardInputs.cpp
    ../src/harness/reference-validation/BundleReferenceValidationHarness.cpp
    ../src/harness/reference-validation/GoldenDataPlan.cpp
    ../src/harness/reference-validation/GoldenOutputProbe.cpp
    ../src/harness/reference-validation/ReferenceOpCoverage.cpp
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
    TestGpuBatchnormFwdTrainPlan.cpp
    TestGpuBatchnormFwdTrainSignatureKey.cpp
    TestGpuConvolutionBwdPlan.cpp
    TestGpuConvolutionBwdSignatureKey.cpp
    TestGpuConvolutionFwdSignatureKey.cpp
    TestGpuConvolutionFwdPlan.cpp
    TestGpuConvolutionWrwPlan.cpp
    TestGpuConvolutionWrwSignatureKey.cpp
    TestGpuLayernormBwdSignatureKey.cpp
    TestGpuLayernormBwdPlan.cpp
    TestGpuLayernormFwdSignatureKey.cpp
    TestGpuLayernormFwdPlan.cpp
    TestGpuMatmulSignatureKey.cpp
    TestGpuMatmulPlan.cpp
    TestGpuSdpaFwdSignatureKey.cpp
    TestGpuSdpaBwdPlan.cpp
    TestGpuSdpaFwdPlan.cpp
    TestGpuSdpaFwdPlanZeroDropout.cpp
    TestGpuSdpaFwdPlanVllmForm.cpp
    TestGpuSdpaFwdPlanSeqLenKvOnly.cpp
    TestGpuSdpaPagedFwdPlan.cpp
    TestGpuRaggedSdpaDensify.cpp
    TestGpuRaggedSdpaMapping.cpp
    TestGpuReductionSignatureKey.cpp
    TestGpuReductionPlan.cpp
    TestGpuResampleSignatureKey.cpp
    TestGpuResamplePlan.cpp
    TestGpuRMSNormSignatureKey.cpp
    TestGpuRMSNormPlan.cpp
    TestGpuSdpaRaggedFwdSignatureKey.cpp
    TestGpuSdpaRaggedFwdPlan.cpp
    TestGpuBatchnormFwdInfSignatureKey.cpp
    TestGpuBatchnormFwdInfPlan.cpp
    TestGpuBatchnormFwdInfVarianceSignatureKey.cpp
    TestGpuBatchnormFwdInfVariancePlan.cpp
    TestGpuPlanRegistrySignatureKey.cpp
    TestReferenceGraphExecutorFactory.cpp
    TestBundleDiscovery.cpp
    TestGTestFilter.cpp
    TestVerificationPaths.cpp
    TestBundleVerificationHarness.cpp
    TestGraphVerificationHarness.cpp
    TestBenchmark.cpp
    TestCapture.cpp
    TestInputFillRecipes.cpp
    TestFillInputs.cpp
    TestToleranceResolver.cpp
    TestConvFwdDynamicTolerance.cpp
    TestConvFwdToleranceGrading.cpp
    TestVerificationModePaths.cpp
    TestVerifierTally.cpp
    TestExpectedRejection.cpp
    TestErrorPaths.cpp
    TestLoadedEngineTable.cpp
    TestReferenceOpCoverage.cpp
    TestSupportClaimReport.cpp
    TestSupportClaimEnforcement.cpp
    TestEnforcementRungs.cpp
    TestSupportClaimRoundTrip.cpp
    TestSupportClaimWriter.cpp
    TestObserveSupportOnly.cpp
    TestSupportVerdict.cpp
    TestClaimModeResolution.cpp
    TestUnpinnedRunEnforcement.cpp
    TestBundleReferenceValidationHarness.cpp
    TestGoldenDataPlan.cpp
    TestGoldenOutputProbe.cpp
    TestProductionPolicy.cpp
    TestGraphSession.cpp
    TestOutputComparison.cpp
    TestRngDumpHandoff.cpp
)

target_include_directories(hipdnn_integration_tests_unit_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_include_directories(hipdnn_integration_tests_unit_tests SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/third-party/tomlplusplus)

# The settings files the providers ship, which TestShippedEngineSettings loads: the tests catch a
# change that a device run would show only as failures. Their text is embedded when the tree is
# configured (ShippedEngineTomls.hpp), because the packaged test binary runs where the provider
# sources are not present. Editing one of the files re-runs the configure step.
set(_shipped_toml_names MIOPEN HIPBLASLT ASM_SDPA HIP_MLOPS HIP_FLASH2)
set(_shipped_toml_paths
    ${CMAKE_CURRENT_SOURCE_DIR}/../../miopen-provider/config/MIOPEN_ENGINE.toml
    ${CMAKE_CURRENT_SOURCE_DIR}/../../hipblaslt-provider/config/HIPBLASLT_ENGINE.toml
    ${CMAKE_CURRENT_SOURCE_DIR}/../../hip-kernel-provider/config/ASM_SDPA_ENGINE.toml
    ${CMAKE_CURRENT_SOURCE_DIR}/../../hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml
    ${CMAKE_CURRENT_SOURCE_DIR}/../../hip-kernel-provider/src/engines/hip_flash2_engine/config/HIP_FLASH2_ENGINE.toml
)
foreach(_toml_name _toml_path IN ZIP_LISTS _shipped_toml_names _shipped_toml_paths)
    cmake_path(NORMAL_PATH _toml_path)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_toml_path})
    file(READ ${_toml_path} _toml_content)
    # The files are checked out with CRLF on Windows: embed one line ending everywhere.
    string(REPLACE "\r\n" "\n" _toml_content "${_toml_content}")
    # The text goes into a C++ raw string literal; a file containing its closing delimiter would
    # end the literal early.
    string(FIND "${_toml_content}" ")HIPDNN_TOML\"" _toml_delimiter_pos)
    if(NOT _toml_delimiter_pos EQUAL -1)
        message(FATAL_ERROR "${_toml_path} contains the raw-string delimiter )HIPDNN_TOML\"")
    endif()
    set(HIPDNN_SHIPPED_TOML_${_toml_name} "${_toml_content}")
endforeach()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ShippedEngineTomls.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/ShippedEngineTomls.hpp @ONLY)
target_include_directories(hipdnn_integration_tests_unit_tests
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

target_link_libraries(hipdnn_integration_tests_unit_tests
    PRIVATE
        hip::host
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gmock
        GTest::gtest
        Threads::Threads
        $<$<BOOL:${USE_ROCRAND}>:roc::rocrand>
        $<$<PLATFORM_ID:Windows>:shlwapi>
)

target_compile_options(hipdnn_integration_tests_unit_tests
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(hipdnn_integration_tests_unit_tests)

# FillInputs.cpp generates large inputs with rocRAND (DeviceInputFiller). Only that
# file is built with USE_ROCRAND: rocrand.h declares a global `half`, which is
# ambiguous in any file that also pulls in hipdnn_data_sdk::types::half unqualified.
if(USE_ROCRAND)
    set_source_files_properties(../src/harness/input-init/FillInputs.cpp
        PROPERTIES COMPILE_DEFINITIONS USE_ROCRAND)
endif()

add_unit_test_target(hipdnn_integration_tests_unit_tests ${CMAKE_CURRENT_BINARY_DIR})

add_executable(hipdnn_gpu_ref_tests
    gpu-ref/main.cpp
    gpu-ref/TestGpuFpReferencePointwise.cpp
    gpu-ref/TestGpuFpReferenceConvolution.cpp
    gpu-ref/TestGpuFpReferenceSdpa.cpp
    gpu-ref/TestGpuFpReferenceSdpaBwd.cpp
    gpu-ref/TestGpuFpReferenceSdpaRagged.cpp
    gpu-ref/TestGpuFpReferenceDgrad.cpp
    gpu-ref/TestGpuFpReferenceLayernormBwd.cpp
    gpu-ref/TestGpuFpReferenceLayernormFwd.cpp
    gpu-ref/TestGpuFpReferenceMatmul.cpp
    gpu-ref/TestGpuFpReferenceWgrad.cpp
    gpu-ref/TestGpuFpReferenceReduction.cpp
    gpu-ref/TestGpuFpReferenceResampleBwd.cpp
    gpu-ref/TestGpuFpReferenceResampleFwd.cpp
    gpu-ref/TestGpuFpReferenceRMSNormFwd.cpp
    gpu-ref/TestGpuFpReferenceRMSNormBwd.cpp
    gpu-ref/TestGpuFpReferenceRmsValidation.cpp
    gpu-ref/TestGpuFpReferenceValidation.cpp
    gpu-ref/TestGpuRefHipError.cpp
    gpu-ref/TestGpuRefKernelCompiler.cpp
    gpu-ref/TestShallowDeviceOnlyMigratableMemory.cpp
    gpu-ref/TestShallowGpuTensor.cpp
    gpu-ref/TestGpuFpReferenceCommon.cpp
    gpu-ref/TestGpuFpReferenceBatchnormFwdInf.cpp
    gpu-ref/TestGpuFpReferenceBatchnormFwdInfWithVariance.cpp
    gpu-ref/TestGpuFpReferenceBatchnormFwdTrain.cpp
    gpu-ref/TestGpuFpReferenceBatchnormBwd.cpp
    gpu-ref/TestGpuRefLaunch.cpp

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
