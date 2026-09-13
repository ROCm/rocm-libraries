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
# Pass the lib/lib64 directory name, not an absolute path, so assertions survive relocation.
target_compile_definitions(
    hipdnn_frontend_dynamic_load_tests
    PRIVATE HIPDNN_TEST_INSTALL_LIBDIR="${CMAKE_INSTALL_LIBDIR}"
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

# The runner owns runtime selection; compiler discovery supplies only a same-image fallback.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
        OUTPUT_VARIABLE _clang_resource_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _clang_resource_result
    )
    set(_asan_preload_runtime "")
    if(_clang_resource_result EQUAL 0 AND NOT _clang_resource_dir STREQUAL "")
        set(_asan_preload_runtime "${_clang_resource_dir}/lib/linux/libclang_rt.asan-x86_64.so")
    endif()
    set(_asan_preload_test hipdnn_frontend_dynamic_load_tests_asan_preload)
    set(_asan_preload_runner "${CMAKE_CURRENT_BINARY_DIR}/RuntimeLoadAsanPreloadRunner.cmake")
    configure_file(RuntimeLoadAsanPreloadRunner.cmake.in "${_asan_preload_runner}" @ONLY)

    add_test(NAME ${_asan_preload_test}
             COMMAND ${CMAKE_COMMAND}
                     "-DHIPDNN_TEST_EXECUTABLE=$<TARGET_FILE:hipdnn_frontend_dynamic_load_tests>"
                     -P "${_asan_preload_runner}"
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    _apply_hipdnn_test_category_labels(${_asan_preload_test})

    set(_asan_preload_snippet "${CMAKE_CURRENT_BINARY_DIR}/RuntimeLoadAsanPreloadTest.cmake")
    file(WRITE "${_asan_preload_snippet}"
         "add_test(${_asan_preload_test} \"${CMAKE_COMMAND}\"\n"
         "    \"-DHIPDNN_TEST_EXECUTABLE=../hipdnn_frontend_dynamic_load_tests\"\n"
         "    -P \"RuntimeLoadAsanPreloadRunner.cmake\")\n")
    get_test_property(${_asan_preload_test} LABELS _asan_preload_labels)
    if(_asan_preload_labels)
        file(APPEND "${_asan_preload_snippet}"
             "set_tests_properties(${_asan_preload_test} PROPERTIES LABELS \"${_asan_preload_labels}\")\n")
    endif()
    install(FILES "${_asan_preload_snippet}" "${_asan_preload_runner}"
            DESTINATION ${CMAKE_INSTALL_BINDIR}/hipdnn)
    set_property(GLOBAL APPEND PROPERTY HIPDNN_INSTALLED_CTEST_INCLUDES
                 RuntimeLoadAsanPreloadTest.cmake)
endif()

if(WIN32)
    set(_native_override_test hipdnn_frontend_dynamic_load_tests_native_override)
    set(_native_override_cleanup "${_native_override_test}_cleanup")
    set(_native_override_fixture hipdnn_frontend_native_override)
    set(_native_override_filter
        "--gtest_filter=TestRuntimeLoadBackendNativeOverride.DISABLED_LoadsBackendFromNativeEnvironmentPath")
    add_test(NAME ${_native_override_test}
             COMMAND hipdnn_frontend_dynamic_load_tests
                     "${_native_override_filter}" --gtest_also_run_disabled_tests
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME ${_native_override_cleanup}
             COMMAND ${CMAKE_COMMAND} -E rm -rf
                     "$<TARGET_FILE_DIR:hipdnn_frontend_dynamic_load_tests>/hipdnn_native_override_fixture")
    set_tests_properties(${_native_override_test} PROPERTIES
                         FIXTURES_REQUIRED ${_native_override_fixture})
    set_tests_properties(${_native_override_cleanup} PROPERTIES
                         FIXTURES_CLEANUP ${_native_override_fixture})
    _apply_hipdnn_test_category_labels(${_native_override_test} ${_native_override_cleanup})

    set(_native_override_snippet "${CMAKE_CURRENT_BINARY_DIR}/RuntimeLoadNativeOverrideTest.cmake")
    file(WRITE "${_native_override_snippet}"
         "add_test(${_native_override_test}\n"
         "    \"../hipdnn_frontend_dynamic_load_tests${CMAKE_EXECUTABLE_SUFFIX}\"\n"
         "    \"${_native_override_filter}\" --gtest_also_run_disabled_tests)\n"
         "add_test(${_native_override_cleanup} \"${CMAKE_COMMAND}\" -E rm -rf\n"
         "    \"../hipdnn_native_override_fixture\")\n"
         "set_tests_properties(${_native_override_test} PROPERTIES\n"
         "    FIXTURES_REQUIRED ${_native_override_fixture})\n"
         "set_tests_properties(${_native_override_cleanup} PROPERTIES\n"
         "    FIXTURES_CLEANUP ${_native_override_fixture})\n")
    if(DEFINED TEST_ENVIRONMENT)
        set_tests_properties(${_native_override_test} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
        file(APPEND "${_native_override_snippet}"
             "set_tests_properties(${_native_override_test} PROPERTIES\n"
             "    ENVIRONMENT [==[${TEST_ENVIRONMENT}]==])\n")
    endif()
    if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
        set_tests_properties(${_native_override_test} PROPERTIES
                             ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
        file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}" _native_override_build_bin)
        file(APPEND "${_native_override_snippet}" "set(_hipdnn_native_override_env_mod)\n")
        foreach(_native_modification IN LISTS TEST_ENVIRONMENT_MODIFICATION)
            # Relocate only the build bin; preserve external paths and extend the runtime PATH.
            if(_native_modification STREQUAL "PATH=path_list_prepend:${_native_override_build_bin}")
                set(_native_installed_modification
                    "\"PATH=path_list_prepend:..\"")
            else()
                set(_native_installed_modification "[==[${_native_modification}]==]")
            endif()
            file(APPEND "${_native_override_snippet}"
                 "list(APPEND _hipdnn_native_override_env_mod ${_native_installed_modification})\n")
        endforeach()
        file(APPEND "${_native_override_snippet}"
             "set_tests_properties(${_native_override_test} PROPERTIES\n"
             "    ENVIRONMENT_MODIFICATION \"\${_hipdnn_native_override_env_mod}\")\n")
    endif()
    foreach(_native_test ${_native_override_test} ${_native_override_cleanup})
        get_test_property(${_native_test} LABELS _native_labels)
        if(_native_labels)
            file(APPEND "${_native_override_snippet}"
                 "set_tests_properties(${_native_test} PROPERTIES LABELS \"${_native_labels}\")\n")
        endif()
    endforeach()
    install(FILES "${_native_override_snippet}" DESTINATION ${CMAKE_INSTALL_BINDIR}/hipdnn)
    set_property(GLOBAL APPEND PROPERTY HIPDNN_INSTALLED_CTEST_INCLUDES
                 RuntimeLoadNativeOverrideTest.cmake)
endif()
