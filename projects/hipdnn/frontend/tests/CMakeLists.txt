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
    TestBackendLibraryResolution.cpp
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
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimGraphSupportCheck.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimKnobWrapper.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimNoteTriage.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimPlanSurface.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimLogging.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimUtils.cpp>
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

# Link the shipped hipdnn_frontend_dynamic target without libhipdnn_backend.
# A successful link proves runtime-load mode references no backend symbols directly.
# This is a separate, deliberately small executable rather than a second configuration of
# hipdnn_frontend_tests above: it has exactly the two sources below and links
# hipdnn_frontend_dynamic alone, so nothing else in the suite can drag libhipdnn_backend
# back in and mask the assertion.
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

# HIPDNN_TEST_EXPECT_BACKEND_LIBRARY is defined only when the backend is actually part of
# this build. With it, a backend that fails to load at run time is a test FAILURE; without
# it -- a frontend-only build, where there is no backend to find -- its absence is
# tolerated. Gating on the target rather than on what happens to be installed is what keeps
# a genuinely missing backend from degrading into a silent skip.
if(TARGET hipdnn_backend)
    add_dependencies(hipdnn_frontend_dynamic_load_tests hipdnn_backend)
    target_compile_definitions(
        hipdnn_frontend_dynamic_load_tests PRIVATE HIPDNN_TEST_EXPECT_BACKEND_LIBRARY
    )
endif()

add_hipdnn_test(
    hipdnn_frontend_dynamic_load_tests ${CMAKE_CURRENT_BINARY_DIR}
)

# Compiles the shipped public frontend umbrella as C++20, the standard under which
# std::filesystem::path::u8string() returns std::u8string -- concatenating its result
# with a narrow literal would break C++20 consumers at include time. There is nothing to
# run: the target is object-only, consuming hipdnn_frontend_dynamic for its usage
# requirements and linking nothing -- in particular not the backend, which runtime-load
# mode must never require. hipdnn_frontend_dynamic_load_tests above is the C++17 control.
if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    add_library(hipdnn_frontend_runtime_load_header_cpp20 OBJECT TestRuntimeLoadHeaderCpp20.cpp)
    target_compile_features(hipdnn_frontend_runtime_load_header_cpp20 PRIVATE cxx_std_20)
    # CXX_CLANG_TIDY is cleared because a superbuild sets CMAKE_CXX_CLANG_TIDY globally,
    # which would otherwise lint this C++17 header as if it were C++20 source. Checks
    # like modernize-use-ranges and modernize-use-constraints are gated on the standard,
    # so they stay silent everywhere else and fire only here -- demanding std::ranges and
    # requires in a header that must keep compiling as C++17. This target proves the
    # header compiles at C++20; every other target lints it at the standard it ships as.
    set_target_properties(
        hipdnn_frontend_runtime_load_header_cpp20
        PROPERTIES CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   CXX_CLANG_TIDY ""
    )
    target_compile_options(
        hipdnn_frontend_runtime_load_header_cpp20 PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS}
    )
    target_link_libraries(
        hipdnn_frontend_runtime_load_header_cpp20 PRIVATE hipdnn_frontend_dynamic
    )
    # Attached to the runtime-load test build so targeted validation cannot omit the gate.
    add_dependencies(hipdnn_frontend_dynamic_load_tests hipdnn_frontend_runtime_load_header_cpp20)
else()
    message(STATUS "hipDNN: this C++ compiler advertises no cxx_std_20; the C++20 runtime-load "
                   "header compile target is not created")
endif()
