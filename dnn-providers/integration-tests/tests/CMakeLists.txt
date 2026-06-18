# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipdnn_integration_tests_unit_tests
    main.cpp
    TestArchMatch.cpp
    TestBundleMetadata.cpp
    TestGraphDescription.cpp
    TestNodeTypeToString.cpp
    TestPlatformUtils.cpp
    TestSupportMatrixCollector.cpp
    TestTestConfig.cpp
    TestTestSettings.cpp
    TestGpuReferenceGraphExecutor.cpp
    TestGpuConvolutionFwdSignatureKey.cpp
    TestGpuConvolutionFwdPlan.cpp
    TestReferenceGraphExecutorFactory.cpp
    TestBundleDiscovery.cpp
    TestVerificationPaths.cpp
    TestGoldenVerificationHarness.cpp
)

target_include_directories(hipdnn_integration_tests_unit_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_include_directories(hipdnn_integration_tests_unit_tests SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/third_party/tomlplusplus)

target_link_libraries(hipdnn_integration_tests_unit_tests
    PRIVATE
        hip::host
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gtest
        Threads::Threads
        $<$<PLATFORM_ID:Windows>:shlwapi>
)

target_compile_options(hipdnn_integration_tests_unit_tests
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(hipdnn_integration_tests_unit_tests)

add_unit_test_target(hipdnn_integration_tests_unit_tests ${CMAKE_CURRENT_BINARY_DIR})

add_executable(hipdnn_gpu_ref_tests
    gpu_ref/main.cpp
    gpu_ref/TestGpuFpReferenceConvolution.cpp
    gpu_ref/TestGpuFpReferenceDgrad.cpp
    gpu_ref/TestGpuFpReferenceWgrad.cpp
    gpu_ref/TestGpuFpReferenceValidation.cpp
    gpu_ref/TestGpuRefHipError.cpp
    gpu_ref/TestGpuRefKernelCompiler.cpp
    gpu_ref/TestShallowDeviceOnlyMigratableMemory.cpp
    gpu_ref/TestShallowGpuTensor.cpp
)

target_link_libraries(hipdnn_gpu_ref_tests
    PRIVATE
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gtest
        Threads::Threads
        hip::host
)

target_compile_options(hipdnn_gpu_ref_tests
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(hipdnn_gpu_ref_tests)

add_tiered_test_target(hipdnn_gpu_ref_tests ${CMAKE_CURRENT_BINARY_DIR})
