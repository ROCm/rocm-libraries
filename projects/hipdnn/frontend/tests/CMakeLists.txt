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
    TestBackendFactory.cpp
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
    # cuDNN-compatibility shim tests; only built when the shim is enabled.
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimUmbrellaCompiles.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimHandle.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimStatusTranslationStandalone.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimError.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimTensor.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV9Tu.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimGraph.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimGraphNodes.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimNoteTriage.cpp>
    # SDPA node coverage additionally needs the SDPA build.
    $<$<AND:$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>,$<BOOL:${HIPDNN_ENABLE_SDPA}>>:TestCudnnShimGraphSDPA.cpp>
    TestDescriptorHelpers.cpp
    TestDescriptorUnpackHelpers.cpp
    TestConvolutionWgradNode.cpp
    TestError.cpp
    TestFrontendLogging.cpp
    TestGraph.cpp
    TestGraphAttributes.cpp
    TestGraphMatchKey.cpp
    TestGraphOverrideValidation.cpp
    TestHandle.cpp
    TestHeuristicPolicyInfo.cpp
    TestIncompatibleBackendWrapper.cpp
    TestINode.cpp
    TestKnob.cpp
    TestKnobUnpacker.cpp
    TestLayernormAttributes.cpp
    TestLayerNormNode.cpp
    TestLayernormBackwardAttributes.cpp
    TestLayernormBackwardNode.cpp
    TestGraphLayernormBackward.cpp
    TestMatmulAttributes.cpp
    TestMatmulNode.cpp
    TestMoeGroupedMatmulAttributes.cpp
    TestMoeGroupedMatmulNode.cpp
    TestGraphMoeGroupedMatmul.cpp
    TestMoeGroupedMatmulBwdAttributes.cpp
    TestMoeGroupedMatmulBwdNode.cpp
    TestGraphMoeGroupedMatmulBwd.cpp
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
    TestGraphResampleBwd.cpp
    TestResampleBwdAttributes.cpp
    TestResampleBwdNode.cpp
    TestResampleFwdAttributes.cpp
    TestResampleFwdNode.cpp
    TestRMSNormAttributes.cpp
    TestRMSNormNode.cpp
    TestRMSNormBackwardAttributes.cpp
    TestRMSNormBackwardNode.cpp
    TestScopedHipdnnBackendDescriptor.cpp
    TestSetEnginePluginPaths.cpp
    TestTensorAttributes.cpp
    TestTensorValueAttributes.cpp
    TestTypes.cpp
    TestUtilities.cpp
    TestTopologicalSortingUtils.cpp
    TestSdkLogging.cpp
    TestVersion.cpp
    TestAutotune.cpp
    TestAutotuneFileWriter.cpp
    TestAutotuneTypes.cpp
)

target_compile_options(hipdnn_frontend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hipdnn_frontend_tests GTest::gtest GTest::gmock hip::host Threads::Threads hipdnn_test_sdk
)
clang_tidy_check(hipdnn_frontend_tests)

add_hipdnn_test(hipdnn_frontend_tests ${CMAKE_CURRENT_BINARY_DIR})

# Runtime-load static check. Links the shipped hipdnn_frontend_dynamic target,
# which compiles the umbrella with HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND and links
# WITHOUT libhipdnn_backend. A successful link therefore proves that, in
# runtime-load mode, no backend symbol is referenced directly. This exercises the
# exact target a downstream consumer would link.
add_executable(
    hipdnn_frontend_dynamic_load_tests
    TestRuntimeLoadBackendFactory.cpp
    TestRuntimeLoadBackendStaticCheck.cpp
)

target_compile_options(hipdnn_frontend_dynamic_load_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hipdnn_frontend_dynamic_load_tests
    GTest::gtest GTest::gtest_main Threads::Threads hipdnn_frontend_dynamic
)
if(TARGET hipdnn_backend)
    add_dependencies(hipdnn_frontend_dynamic_load_tests hipdnn_backend)
    target_compile_definitions(
        hipdnn_frontend_dynamic_load_tests PRIVATE HIPDNN_TEST_EXPECT_BACKEND_LIBRARY
    )
endif()

add_hipdnn_test(
    hipdnn_frontend_dynamic_load_tests ${CMAKE_CURRENT_BINARY_DIR}
)
# Under ASAN the dlopen interceptor ignores the executable RUNPATH, so
# the bare-name backend .so is only found via LD_LIBRARY_PATH.
if(BUILD_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN")
    set_property(TEST hipdnn_frontend_dynamic_load_tests APPEND PROPERTY
        ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:hipdnn_backend>")
endif()
