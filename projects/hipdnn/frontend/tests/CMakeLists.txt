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

# A downstream consumer compiling the shipped runtime-load headers as C++20. Under that
# standard std::filesystem::path::u8string() returns std::u8string, so a diagnostic that
# concatenated it with a narrow literal would break every C++20 consumer at include time.
# This compiles the real shipped header rather than a copy; there is nothing to run, so
# the target is object-only and links nothing -- in particular not the backend, which
# runtime-load mode must never require. hipdnn_frontend_dynamic_load_tests above is the
# C++17 control.
if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    add_library(hipdnn_frontend_runtime_load_header_cpp20 OBJECT TestRuntimeLoadHeaderCpp20.cpp)
    target_compile_features(hipdnn_frontend_runtime_load_header_cpp20 PRIVATE cxx_std_20)
    set_target_properties(
        hipdnn_frontend_runtime_load_header_cpp20
        PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
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

# Registers one runner invocation in both trees. The build entry names the configure
# host's CMake and its target files; the installed entry names a bare "cmake" that the
# runner's own host resolves, plus paths relative to ctest's installed working
# directory -- the same shape install_hipdnn_ctest_files() emits for every other
# installed test, and what plugin_sdk/tests/CMakeLists.txt already does by hand.
function(_hipdnn_add_runtime_load_entry name)
    cmake_parse_arguments(ARG "PRELOAD;RUN_DISABLED" "FILTER;STAGE_LIBDIR" "" ${ARGN})

    set(_build_options
        "-DHIPDNN_TEST_EXECUTABLE=$<TARGET_FILE:hipdnn_frontend_dynamic_load_tests>"
        "-DHIPDNN_TEST_FILTER=${ARG_FILTER}")
    set(_installed_options
        "-DHIPDNN_TEST_EXECUTABLE=../hipdnn_frontend_dynamic_load_tests"
        "-DHIPDNN_TEST_FILTER=${ARG_FILTER}")
    if(ARG_PRELOAD)
        list(APPEND _build_options "-DHIPDNN_TEST_PRELOAD=ON")
        list(APPEND _installed_options "-DHIPDNN_TEST_PRELOAD=ON")
    endif()
    if(ARG_RUN_DISABLED)
        list(APPEND _build_options "-DHIPDNN_TEST_RUN_DISABLED=ON")
        list(APPEND _installed_options "-DHIPDNN_TEST_RUN_DISABLED=ON")
    endif()
    if(ARG_STAGE_LIBDIR)
        # The fixture is disposable output, so it is staged under the writable work
        # area ctest runs in, one root per entry so no two entries can collide.
        list(APPEND _build_options
             "-DHIPDNN_TEST_BACKEND=$<TARGET_FILE:hipdnn_backend>"
             "-DHIPDNN_TEST_STAGE_ROOT=${CMAKE_CURRENT_BINARY_DIR}/runtime_load_fixtures/${name}"
             "-DHIPDNN_TEST_STAGE_LIBDIR=${ARG_STAGE_LIBDIR}")
        list(APPEND _installed_options
             "-DHIPDNN_TEST_BACKEND=../../${CMAKE_INSTALL_LIBDIR}/${_hipdnn_backend_file_name}"
             "-DHIPDNN_TEST_STAGE_ROOT=runtime_load_fixtures/${name}"
             "-DHIPDNN_TEST_STAGE_LIBDIR=${ARG_STAGE_LIBDIR}")
    endif()

    add_test(NAME ${name}
             COMMAND ${CMAKE_COMMAND} ${_build_options} -P "${_runtime_load_runner}"
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # One call per test name: the helper takes a single explicit name, so a second one
    # here would reach the parser as a stray argument and leave every name unlabelled.
    _apply_hipdnn_test_category_labels(${name})

    set(_snippet_body "add_test(${name} \"cmake\"")
    foreach(_option IN LISTS _installed_options)
        string(APPEND _snippet_body "\n    \"${_option}\"")
    endforeach()
    string(APPEND _snippet_body "\n    -P \"RuntimeLoadAsanPreloadRunner.cmake\")\n")

    get_test_property(${name} LABELS _labels)
    if(_labels)
        string(APPEND _snippet_body
               "set_tests_properties(${name} PROPERTIES LABELS \"${_labels}\")\n")
    endif()
    file(APPEND "${_runtime_load_snippet}" "${_snippet_body}\n")
endfunction()

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

    set(_runtime_load_runner "${CMAKE_CURRENT_BINARY_DIR}/RuntimeLoadAsanPreloadRunner.cmake")
    configure_file(RuntimeLoadAsanPreloadRunner.cmake.in "${_runtime_load_runner}" @ONLY)
    set(_runtime_load_snippet "${CMAKE_CURRENT_BINARY_DIR}/RuntimeLoadAsanPreloadTest.cmake")
    file(WRITE "${_runtime_load_snippet}"
         "# Autogenerated: installed-tree copies of the runtime-load runner entries in\n"
         "# projects/hipdnn/frontend/tests/CMakeLists.txt -- do not hand-edit.\n\n")
    set(_hipdnn_backend_file_name
        "${CMAKE_SHARED_LIBRARY_PREFIX}hipdnn_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")

    set(_runtime_load_ordinary_filter "TestRuntimeLoadBackendFactory.*")
    set(_runtime_load_staged_filter
        "TestRuntimeLoadBackendFactory.DISABLED_ResolvesTheStagedSelfRelativeBackend")

    # Preloading ASan activates its dlopen interceptor, which takes the loader's
    # implicit search away: these entries pass only if the resolver computed the path.
    # Instrumenting the same binary with ThreadSanitizer makes that pairing impossible,
    # so those two configurations register no preload entry. A missing ASan runtime
    # anywhere else remains a run-time failure, never a quiet configure-time omission.
    set(_runtime_load_thread_sanitized OFF)
    if(BUILD_THREAD_SANITIZER OR THEROCK_SANITIZER STREQUAL "TSAN")
        set(_runtime_load_thread_sanitized ON)
    endif()

    if(NOT _runtime_load_thread_sanitized)
        _hipdnn_add_runtime_load_entry(
            hipdnn_frontend_dynamic_load_tests_asan_preload
            PRELOAD FILTER "${_runtime_load_ordinary_filter}"
        )
    endif()

    # The exact self-relative expectation lives here rather than in the ordinary cases,
    # which must stay valid for any CMAKE_INSTALL_LIBDIR. The runner copies the real
    # executable and the real backend into a fixture of its own and requires that exact
    # backend, so neither a custom install layout nor a RUNPATH fallback can satisfy it.
    # lib and lib64 are separate runs because they are separate resolver candidates.
    if(TARGET hipdnn_backend)
        _hipdnn_add_runtime_load_entry(
            hipdnn_frontend_dynamic_load_tests_staged_lib
            RUN_DISABLED FILTER "${_runtime_load_staged_filter}" STAGE_LIBDIR lib
        )
        _hipdnn_add_runtime_load_entry(
            hipdnn_frontend_dynamic_load_tests_staged_lib64
            RUN_DISABLED FILTER "${_runtime_load_staged_filter}" STAGE_LIBDIR lib64
        )
        if(NOT _runtime_load_thread_sanitized)
            _hipdnn_add_runtime_load_entry(
                hipdnn_frontend_dynamic_load_tests_staged_asan_preload
                PRELOAD RUN_DISABLED FILTER "${_runtime_load_staged_filter}" STAGE_LIBDIR lib
            )
        endif()
    else()
        message(STATUS "hipDNN: hipdnn_backend is not built here, so the staged "
                       "self-relative runtime-load entries are not registered")
    endif()

    install(FILES "${_runtime_load_snippet}" "${_runtime_load_runner}"
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
    # One call per test name: the helper takes a single explicit name, so passing both
    # at once reaches the parser as a stray argument and leaves neither name labelled.
    _apply_hipdnn_test_category_labels(${_native_override_test})
    _apply_hipdnn_test_category_labels(${_native_override_cleanup})

    set(_native_override_snippet "${CMAKE_CURRENT_BINARY_DIR}/RuntimeLoadNativeOverrideTest.cmake")
    file(WRITE "${_native_override_snippet}"
         "add_test(${_native_override_test}\n"
         "    \"../hipdnn_frontend_dynamic_load_tests${CMAKE_EXECUTABLE_SUFFIX}\"\n"
         "    \"${_native_override_filter}\" --gtest_also_run_disabled_tests)\n"
         "add_test(${_native_override_cleanup} \"cmake\" -E rm -rf\n"
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
