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
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8CApiStandalone.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8TypeMapping.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8Utils.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8Tu.cpp>
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
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BuilderTensor.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BuilderParameters.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BuilderOperations.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BuilderLowering.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BuilderGraph.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BuilderEngines.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BuilderPlans.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BuilderInputs.cpp>
    $<$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>:TestCudnnShimV8BuilderContracts.cpp>
    # The errata filter and the shim's json alias exist only when the frontend keeps its JSON
    # library.
    $<$<AND:$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>,$<NOT:$<BOOL:${HIPDNN_FRONTEND_SKIP_JSON_LIB}>>>:TestCudnnShimV8Errata.cpp>
    $<$<AND:$<BOOL:${HIPDNN_ENABLE_CUDNN_COMPATIBILITY}>,$<NOT:$<BOOL:${HIPDNN_FRONTEND_SKIP_JSON_LIB}>>>:TestCudnnShimGlobalJson.cpp>
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
function(hipdnn_frontend_expect_backend_library test_target)
    if(TARGET hipdnn_backend)
        add_dependencies(${test_target} hipdnn_backend)
        target_compile_definitions(${test_target} PRIVATE HIPDNN_TEST_EXPECT_BACKEND_LIBRARY)
    endif()
endfunction()

hipdnn_frontend_expect_backend_library(hipdnn_frontend_dynamic_load_tests)

add_hipdnn_test(
    hipdnn_frontend_dynamic_load_tests ${CMAKE_CURRENT_BINARY_DIR}
)

# A compile-only C++20 object target for `source`. CXX_CLANG_TIDY is cleared because a
# superbuild sets CMAKE_CXX_CLANG_TIDY globally, which would otherwise lint the C++17 headers
# as if they were C++20 source. Checks like modernize-use-ranges and modernize-use-constraints
# are gated on the standard, so they stay silent everywhere else and fire only here --
# demanding std::ranges and requires in headers that must keep compiling as C++17. These
# targets prove the headers compile at C++20; every other target lints them at the standard
# they ship as.
function(hipdnn_frontend_cpp20_object object_target source)
    add_library(${object_target} OBJECT ${source})
    target_compile_features(${object_target} PRIVATE cxx_std_20)
    set_target_properties(
        ${object_target}
        PROPERTIES CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   CXX_CLANG_TIDY ""
    )
    target_compile_options(${object_target} PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
endfunction()

# Compiles the shipped public frontend umbrella as C++20, the standard under which
# std::filesystem::path::u8string() returns std::u8string -- concatenating its result
# with a narrow literal would break C++20 consumers at include time. There is nothing to
# run: the target consumes hipdnn_frontend_dynamic for its usage requirements and links
# nothing -- in particular not the backend, which runtime-load mode must never require.
# hipdnn_frontend_dynamic_load_tests above is the C++17 control.
if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    hipdnn_frontend_cpp20_object(
        hipdnn_frontend_runtime_load_header_cpp20 TestRuntimeLoadHeaderCpp20.cpp
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
        TestCudnnShimV8NoexceptBoundary.cpp
        TestCudnnShimV8NoexceptEngines.cpp
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

    # Link-only gate, like hipdnn_frontend_dynamic_load_tests: the shim links against
    # hipdnn_frontend_dynamic without libhipdnn_backend.
    add_executable(hipdnn_frontend_cudnn_v8_dynamic_link_tests TestCudnnShimV8DynamicLink.cpp)
    target_compile_options(
        hipdnn_frontend_cudnn_v8_dynamic_link_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS}
    )
    target_link_libraries(
        hipdnn_frontend_cudnn_v8_dynamic_link_tests
        GTest::gtest GTest::gtest_main Threads::Threads hipdnn_frontend_dynamic
    )
    hipdnn_frontend_expect_backend_library(hipdnn_frontend_cudnn_v8_dynamic_link_tests)
    clang_tidy_check(hipdnn_frontend_cudnn_v8_dynamic_link_tests)
    add_hipdnn_test(hipdnn_frontend_cudnn_v8_dynamic_link_tests ${CMAKE_CURRENT_BINARY_DIR})

    # Compile-only: the umbrella with logging compiled out and without SDPA. HIPDNN_ENABLE_SDPA
    # is a directory-wide and hipdnn_frontend interface definition, so it is undefined here;
    # compile options follow the definitions on the command line.
    add_library(
        hipdnn_frontend_cudnn_shim_no_logging_no_sdpa OBJECT TestCudnnShimNoLoggingNoSdpa.cpp
    )
    target_compile_definitions(
        hipdnn_frontend_cudnn_shim_no_logging_no_sdpa PRIVATE NV_CUDNN_FRONTEND_DISABLE_LOGGING
    )
    target_compile_options(
        hipdnn_frontend_cudnn_shim_no_logging_no_sdpa
        PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} -UHIPDNN_ENABLE_SDPA
    )
    target_link_libraries(
        hipdnn_frontend_cudnn_shim_no_logging_no_sdpa PRIVATE hipdnn_frontend hip::host
    )

    # Compile-only: the umbrella under each JSON opt-out (CUDNN_ / HIPDNN_FRONTEND_SKIP_JSON_LIB;
    # the plain umbrella is in hipdnn_frontend_tests), and each autotune header outside the
    # umbrella, first and alone in its TU, plain and under each opt-out.
    set(_hipdnn_cudnn_v8_header_umbrella TestCudnnShimV8SkipJson.cpp)
    set(_hipdnn_cudnn_v8_header_find_plan TestCudnnShimV8HeaderFindPlan.cpp)
    set(_hipdnn_cudnn_v8_header_get_plan TestCudnnShimV8HeaderGetPlan.cpp)
    set(_hipdnn_cudnn_v8_header_engine_config_generator
        TestCudnnShimV8HeaderEngineConfigGenerator.cpp
    )
    set(_hipdnn_cudnn_v8_header_targets)
    foreach(_header umbrella find_plan get_plan engine_config_generator)
        foreach(_skip_owner NONE CUDNN HIPDNN)
            if(_header STREQUAL "umbrella" AND _skip_owner STREQUAL "NONE")
                continue()
            endif()
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
    # The wrapper source keeps TestCudnnShimV8Tu.cpp at one compile_commands.json entry, which
    # the -p lookup of the unit binary's tidy step would otherwise also lint as C++20.
    set(_hipdnn_cudnn_v8_cpp20_targets)
    if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        hipdnn_frontend_cpp20_object(hipdnn_frontend_cudnn_v8_tu_cpp20 TestCudnnShimV8TuCpp20.cpp)
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
        hipdnn_frontend_cudnn_shim_no_logging_no_sdpa
        ${_hipdnn_cudnn_v8_header_targets}
        ${_hipdnn_cudnn_v8_cpp20_targets}
    )

    # Build tree only. The suite uses stand-in inputs: no network, upstream tree or GPU.
    if(Python3_FOUND)
        add_test(
            NAME hipdnn_cudnn_v8_extractor_tests
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/data/cudnn_v8/test_extract_cudnn_v8_identifiers.py
        )
        _apply_hipdnn_test_category_labels(hipdnn_cudnn_v8_extractor_tests)
    endif()
endif()
