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
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimNoteTriageTable.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimEngineFilters.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimPlanSurface.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimLogging.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimUtils.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8Identifiers.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8AttributeTable.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BackendApi.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8CApiStandalone.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8TensorDescriptor.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8ParameterDescriptors.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8Operations.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8OperationLowering.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8TypeMapping.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8OperationGraph.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8EngineSelection.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8ExecutionPlan.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8Utils.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8DescriptorBuilders.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8AuxBuilders.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8Tu.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8OperationBuilders.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8MixedTu.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8Heuristics.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8EngineBuilders.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8ExecutionPlanBuilder.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8Filters.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8PlanCache.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8GetPlan.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8FindPlan.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8CoreFactories.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8CoreLowering.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8CorePlans.cpp>
    # The errata filter exists only when the frontend keeps its JSON library.
    $<$<AND:$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>,$<NOT:$<BOOL:${HIPDNN_FRONTEND_SKIP_JSON_LIB}>>>:TestCudnnShimV8Errata.cpp>
    # SDPA node coverage additionally needs the SDPA build.
    $<$<AND:$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>,$<BOOL:${HIPDNN_ENABLE_SDPA}>>:TestCudnnShimGraphSDPA.cpp>
    TestDescriptorHelpers.cpp
    TestDescriptorUnpackHelpers.cpp
    TestConvolutionWgradNode.cpp
    TestEngineQueries.cpp
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

# detail/EngineQueries.hpp decodes FlatBuffers pages; the frontend itself does not link the SDK.
target_link_libraries(
    hipdnn_frontend_tests GTest::gtest GTest::gmock hip::host Threads::Threads hipdnn_test_sdk
    hipdnn_flatbuffers_sdk
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

if(HIPDNN_ENABLE_CUDNN_COMPATIBILITY)
    # build() bodies depend on NV_CUDNN_DISABLE_EXCEPTION, so the status-return path needs its
    # own binary: mixing TUs with and without the macro in one binary is an ODR violation.
    add_executable(
        hipdnn_frontend_cudnn_v8_noexcept_tests
        main.cpp
        TestCudnnShimV8NoExcept.cpp
        TestCudnnShimV8NoexceptOperations.cpp
        TestCudnnShimV8NoexceptPlans.cpp
    )
    target_compile_definitions(
        hipdnn_frontend_cudnn_v8_noexcept_tests PRIVATE NV_CUDNN_DISABLE_EXCEPTION
    )
    target_compile_options(
        hipdnn_frontend_cudnn_v8_noexcept_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS}
    )
    target_link_libraries(
        hipdnn_frontend_cudnn_v8_noexcept_tests
        GTest::gtest GTest::gmock hip::host Threads::Threads hipdnn_test_sdk
    )
    clang_tidy_check(hipdnn_frontend_cudnn_v8_noexcept_tests)
    add_hipdnn_test(hipdnn_frontend_cudnn_v8_noexcept_tests ${CMAKE_CURRENT_BINARY_DIR})

    # Compile-only: the umbrella with each JSON opt-out.
    add_library(hipdnn_frontend_cudnn_v8_skip_cudnn_json OBJECT TestCudnnShimV8SkipCudnnJson.cpp)
    target_compile_definitions(
        hipdnn_frontend_cudnn_v8_skip_cudnn_json PRIVATE CUDNN_FRONTEND_SKIP_JSON_LIB
    )
    target_compile_options(
        hipdnn_frontend_cudnn_v8_skip_cudnn_json PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS}
    )
    target_link_libraries(
        hipdnn_frontend_cudnn_v8_skip_cudnn_json PRIVATE hipdnn_frontend hip::host
    )

    add_library(
        hipdnn_frontend_cudnn_v8_skip_hipdnn_json OBJECT TestCudnnShimV8SkipHipdnnJson.cpp
    )
    target_compile_definitions(
        hipdnn_frontend_cudnn_v8_skip_hipdnn_json PRIVATE HIPDNN_FRONTEND_SKIP_JSON_LIB
    )
    target_compile_options(
        hipdnn_frontend_cudnn_v8_skip_hipdnn_json PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS}
    )
    target_link_libraries(
        hipdnn_frontend_cudnn_v8_skip_hipdnn_json PRIVATE hipdnn_frontend hip::host
    )

    # Compile-only: each autotune header outside the umbrella, first and alone in its TU,
    # plain and under each JSON opt-out (CUDNN_ / HIPDNN_FRONTEND_SKIP_JSON_LIB).
    set(_hipdnn_cudnn_v8_header_find_plan TestCudnnShimV8HeaderFindPlan.cpp)
    set(_hipdnn_cudnn_v8_header_get_plan TestCudnnShimV8HeaderGetPlan.cpp)
    set(_hipdnn_cudnn_v8_header_engine_config_generator
        TestCudnnShimV8HeaderEngineConfigGenerator.cpp
    )
    set(_hipdnn_cudnn_v8_header_targets)
    foreach(_header find_plan get_plan engine_config_generator)
        foreach(_skip_owner NONE CUDNN HIPDNN)
            set(_target hipdnn_frontend_cudnn_v8_header_${_header})
            if(NOT _skip_owner STREQUAL "NONE")
                string(TOLOWER "${_skip_owner}" _owner)
                string(APPEND _target "_skip_${_owner}_json")
            endif()
            add_library(${_target} OBJECT ${_hipdnn_cudnn_v8_header_${_header}})
            if(NOT _skip_owner STREQUAL "NONE")
                target_compile_definitions(
                    ${_target} PRIVATE ${_skip_owner}_FRONTEND_SKIP_JSON_LIB
                )
            endif()
            target_compile_options(${_target} PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
            target_link_libraries(${_target} PRIVATE hipdnn_frontend hip::host)
            list(APPEND _hipdnn_cudnn_v8_header_targets ${_target})
        endforeach()
    endforeach()

    # PyTorch consumes these headers as C++20; every other target here compiles them as C++17.
    # CXX_CLANG_TIDY is cleared for the reason given on the runtime-load C++20 target above.
    set(_hipdnn_cudnn_v8_cpp20_targets)
    if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        add_library(hipdnn_frontend_cudnn_v8_tu_cpp20 OBJECT TestCudnnShimV8Tu.cpp)
        target_compile_features(hipdnn_frontend_cudnn_v8_tu_cpp20 PRIVATE cxx_std_20)
        set_target_properties(
            hipdnn_frontend_cudnn_v8_tu_cpp20
            PROPERTIES CXX_STANDARD 20
                       CXX_STANDARD_REQUIRED ON
                       CXX_EXTENSIONS OFF
                       CXX_CLANG_TIDY ""
        )
        target_compile_options(
            hipdnn_frontend_cudnn_v8_tu_cpp20 PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS}
        )
        target_link_libraries(
            hipdnn_frontend_cudnn_v8_tu_cpp20 PRIVATE hipdnn_frontend hip::host GTest::gtest
        )
        list(APPEND _hipdnn_cudnn_v8_cpp20_targets hipdnn_frontend_cudnn_v8_tu_cpp20)
    else()
        message(STATUS "hipDNN: this C++ compiler advertises no cxx_std_20; the C++20 cuDNN v8 "
                       "shim compile target is not created")
    endif()

    # Attached to the unit binary so targeted builds cannot skip the gates.
    add_dependencies(
        hipdnn_frontend_tests
        hipdnn_frontend_cudnn_v8_skip_cudnn_json
        hipdnn_frontend_cudnn_v8_skip_hipdnn_json
        ${_hipdnn_cudnn_v8_header_targets}
        ${_hipdnn_cudnn_v8_cpp20_targets}
    )
endif()
