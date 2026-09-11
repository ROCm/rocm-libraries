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
# The resolved-path assertion compares against <exe dir>/../<libdir>, the layout both the
# build tree and the install tree use. The name of that sibling directory is a CMake
# decision (lib vs lib64), so it is handed to the test rather than guessed there. Only the
# directory name travels; no configure-time absolute path is baked in, so the binary stays
# correct in a relocated tree.
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

# Second ctest entry over the same binary, run with the shared ASan runtime preloaded.
# Preloading activates ASan's dlopen interceptor even for an uninstrumented binary, and
# that interceptor resolves a bare soname against the ASan runtime's own RUNPATH instead
# of the caller's -- the configuration the asan-debug artifact tier runs in, and the one
# that has to prove the backend is found by a path the frontend computed. Leak detection
# is off because an uninstrumented binary's allocations are not tracked, and the link-order
# check is off because the runtime arrives through LD_PRELOAD rather than -fsanitize=address.
#
# The runtime sits beside the compiler rather than on any loader search path, so it is
# located through the resource directory. cmake/Sanitizers.cmake asks the same question,
# but its lib/windows subpath is the Windows branch; Linux is lib/linux. Not every
# toolchain ships the shared runtime: when it is absent the entry is not registered, and
# the message below is what keeps that from being an oracle that silently disappears.
#
# The category labels are not decoration -- CI selects by label tier, and an unlabelled
# test is invisible to every tier -- so this hand-registered entry applies the same helper
# add_hipdnn_test() applies to the targets it registers.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
        OUTPUT_VARIABLE _clang_resource_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(_asan_preload_runtime "${_clang_resource_dir}/lib/linux/libclang_rt.asan-x86_64.so")
    set(_asan_preload_test hipdnn_frontend_dynamic_load_tests_asan_preload)

    if(EXISTS "${_asan_preload_runtime}")
        set(_asan_preload_env
            "LD_PRELOAD=${_asan_preload_runtime}"
            "ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0"
        )

        add_test(NAME ${_asan_preload_test}
                 COMMAND hipdnn_frontend_dynamic_load_tests
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_tests_properties(${_asan_preload_test} PROPERTIES
                             ENVIRONMENT "${_asan_preload_env}")
        _apply_hipdnn_test_category_labels(${_asan_preload_test})

        # install_hipdnn_ctest_files() (cmake/Tests.cmake) writes the installed
        # CTestTestfile.cmake from one bare add_test() per registered target and carries no
        # test properties across, so an entry whose entire content is its environment cannot
        # travel that way. Emit it as an installed snippet and register the file name --
        # the mechanism cmake/Tests.cmake provides for entries add_hipdnn_test() cannot
        # express. The preload path baked in here is the one this tree was configured
        # against; the installed tree is exercised by the same container image that built
        # it, so that is the path the runtime is at there too.
        set(_asan_preload_snippet "${CMAKE_CURRENT_BINARY_DIR}/RuntimeLoadAsanPreloadTest.cmake")
        file(WRITE "${_asan_preload_snippet}"
             "# Autogenerated: installed-tree copy of the ${_asan_preload_test} ctest entry\n"
             "# in projects/hipdnn/frontend/tests/CMakeLists.txt. Regenerated from that\n"
             "# entry at configure time -- do not hand-edit.\n\n"
             "add_test(${_asan_preload_test} \"../hipdnn_frontend_dynamic_load_tests\")\n"
             "set_tests_properties(${_asan_preload_test} PROPERTIES\n"
             "    ENVIRONMENT \"${_asan_preload_env}\")\n")

        # Read back rather than retyped, so the installed entry cannot drift from the
        # build-tree one. ctest's CTestTestfile.cmake interpreter understands
        # set_tests_properties() but not the generic set_property(), so labels go through
        # the former.
        get_test_property(${_asan_preload_test} LABELS _asan_preload_labels)
        if(_asan_preload_labels)
            file(APPEND "${_asan_preload_snippet}"
                 "set_tests_properties(${_asan_preload_test} PROPERTIES LABELS \"${_asan_preload_labels}\")\n")
        endif()

        install(FILES "${_asan_preload_snippet}" DESTINATION ${CMAKE_INSTALL_BINDIR}/hipdnn)
        set_property(GLOBAL APPEND PROPERTY HIPDNN_INSTALLED_CTEST_INCLUDES
                     RuntimeLoadAsanPreloadTest.cmake)
    else()
        message(STATUS
                "hipDNN: shared ASan runtime not found at ${_asan_preload_runtime}; "
                "not registering ctest entry ${_asan_preload_test}")
    endif()
endif()
