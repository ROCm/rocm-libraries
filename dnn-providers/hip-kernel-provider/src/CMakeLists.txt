# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# OBJECT library with all implementation
add_library(
    hip_kernel_provider_impl OBJECT
    core/Handle.cpp
    core/Container.cpp
    core/PluginPublic.cpp
    core/Utils.cpp
    compilation/Kernel.cpp
    compilation/Program.cpp
)

add_subdirectory(engines)

target_include_directories(hip_kernel_provider_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/engines/
)
dnn_provider_generate_version_header(hip_kernel_provider hip_kernel_provider_impl ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(hip_kernel_provider_impl PUBLIC hip::host hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hiprtc::hiprtc)

target_compile_options(hip_kernel_provider_impl PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(hip_kernel_provider_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
clang_tidy_check(hip_kernel_provider_impl)
add_clang_tidy_custom_target()

# Static library for unit tests to link against (provides access to internal implementation)
add_library(hip_kernel_provider_private STATIC $<TARGET_OBJECTS:hip_kernel_provider_impl>)
target_include_directories(hip_kernel_provider_private PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_link_libraries(hip_kernel_provider_private PUBLIC hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hiprtc::hiprtc hip_kernel_provider_impl)
if(ENABLE_ASM_SDPA_ENGINE)
    target_compile_definitions(hip_kernel_provider_private PUBLIC HIPDNN_ENGINE_ASM_SDPA=1)
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    target_compile_definitions(hip_kernel_provider_private PUBLIC HIPDNN_ENGINE_HIP_MLOPS=1)
endif()

if(ENABLE_HIP_FLASH2_ENGINE)
    # Precompiled-kernel engine: compile the host-side engine/plan sources into the
    # provider object library. HipFlash2FwdPlan.hip is intentionally excluded (it is
    # the offline source for the shipped .co files and needs rocWMMA headers).
    target_sources(hip_kernel_provider_impl PRIVATE
        engines/hip_flash2_engine/HipFlash2Engine.cpp
        engines/hip_flash2_engine/HipFlash2FwdPlan.cpp
        engines/hip_flash2_engine/HipFlash2FwdPlanBuilder.cpp)
    target_compile_definitions(hip_kernel_provider_impl PRIVATE
        HIPDNN_ENGINE_HIP_FLASH2
        HIP_FLASH2_KERNEL_DIR="${CMAKE_INSTALL_PREFIX}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/hip_flash2_kernels")
    target_compile_definitions(hip_kernel_provider_private PUBLIC HIPDNN_ENGINE_HIP_FLASH2=1)
    # Defect E fix: copy .co files to build tree so build-tree test runs work without
    # manual HIP_FLASH2_KERNEL_DIR export (mirrors asm_sdpa_engine pattern)
    # Copy every .co present, not just the legacy object: naming only the
    # legacy one meant a build-tree ctest could never exercise a variant even
    # when the install was correct (S. Reeder had to stage objects by hand to
    # test the dispatcher at all).
    file(GLOB _f2_build_cos
         ${CMAKE_CURRENT_SOURCE_DIR}/engines/hip_flash2_engine/kernels/*.co)
    file(COPY ${_f2_build_cos}
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/hip_flash2_kernels)
    # Override HIP_FLASH2_KERNEL_DIR at test time to point to build tree
    set(HIP_FLASH2_KERNEL_DIR_BUILD "${CMAKE_CURRENT_BINARY_DIR}/hip_flash2_kernels"
        CACHE STRING "Build-tree path for Flash2 .co kernels (for test runs)")
endif()
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    target_compile_definitions(hip_kernel_provider_private PUBLIC HIPDNN_ENABLE_KERNEL_INGESTOR=1)
    # Only the install path is compiled in. Baking the build tree in as well would ship an
    # absolute path from this machine inside the plugin and prefer it on any host where it
    # happened to exist, which also means nothing would ever exercise the installed copy.
    # Build-tree runs set HIPDNN_DESCRIPTOR_DIR instead; the test targets do it for ctest.
    target_compile_definitions(hip_kernel_provider_impl PRIVATE
        HIPDNN_DESCRIPTOR_INSTALL_DIR="${CMAKE_INSTALL_PREFIX}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPDNN_DESCRIPTOR_SUBDIR}"
        # Container.cpp resolves this relative to the loaded .so first, so a
        # DESTDIR-staged, relocated, or repackaged install still finds its own
        # descriptors; HIPDNN_DESCRIPTOR_INSTALL_DIR above is only its fallback.
        HIPDNN_DESCRIPTOR_SUBDIR="${HIPDNN_DESCRIPTOR_SUBDIR}"
    )

    # The kpack kernel source adapter. Listed here rather than beside Kernel.cpp above
    # because it does not compile without the ingestor: KpackArchive.cpp is the only
    # translation unit in the provider that includes <rocm_kpack/kpack.h>.
    target_sources(hip_kernel_provider_impl PRIVATE
        compilation/KpackArchive.cpp
        compilation/KpackKernelLoader.cpp
        compilation/KpackProgram.cpp
    )

    # The object library gets rocm::rocm_kpack for its include directories; the two
    # targets that actually link have to name it themselves, because they consume the
    # objects via $<TARGET_OBJECTS:...>, which carries no link interface.
    target_link_libraries(hip_kernel_provider_impl PRIVATE rocm::rocm_kpack)
    target_link_libraries(hip_kernel_provider_private PUBLIC rocm::rocm_kpack)
endif()

# hip_kernel_provider - SHARED library (the actual plugin)
add_library(hip_kernel_provider SHARED $<TARGET_OBJECTS:hip_kernel_provider_impl>)
target_link_libraries(hip_kernel_provider PRIVATE hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hiprtc::hiprtc hip::host)

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    # See the note above hip_kernel_provider_impl: $<TARGET_OBJECTS:...> carries no link
    # interface, so the plugin has to name the kpack reader itself.
    target_link_libraries(hip_kernel_provider PRIVATE rocm::rocm_kpack)
endif()

if(NOT WIN32)
    target_link_libraries(hip_kernel_provider PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

target_compile_options(hip_kernel_provider PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    hip_kernel_provider
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

# Set RPATH origin for TheRock integration
set_target_properties(hip_kernel_provider PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")

# ============================================================================
# Tests
# ============================================================================
if(HIPKERNELPROVIDER_ENABLE_TESTS)

    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    if(NOT TARGET hipdnn_integration_tests)
        find_package(hipdnn_integration_tests CONFIG QUIET)
    endif()

    set(DNN_PROVIDER_TEST_RUN_DEPENDS hip_kernel_provider_tests hip_kernel_provider_integration_tests)
    if(TARGET hipdnn_integration_tests)
        list(APPEND DNN_PROVIDER_TEST_RUN_DEPENDS hip_kernel_provider hipdnn_integration_tests)
    endif()

    # Keep this after all build folders have been added so they have a chance to register their tests
    # using add_*_test_target()
    if(HIPKERNELPROVIDER_ENABLE_COVERAGE)
        finalize_test_targets("${PROJECT_NAME}" ENABLE_COVERAGE)
    else()
        finalize_test_targets("${PROJECT_NAME}")
    endif()

    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
        enable_testing()
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
        # Install-time CTestTestfile uses relative paths so `ctest` can run from the
        # installed location (bin/hip_kernel_provider/) and reach ../hip_kernel_provider_tests etc.
        set(HIPKERNELPROVIDER_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
        file(WRITE "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
[=[
# This is a test file generated by hip-kernel-provider for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
        )
        # The GTest YAML applies to both test binaries; each call appends its category suites
        # (e.g. hip_kernel_provider_tests_quick_suite, hip_kernel_provider_integration_tests_quick_suite)
        # to the same install-time CTestTestfile.cmake. ENVIRONMENT forwards the CMake-side
        # TEST_ENVIRONMENT (ASAN symbolizer path, coverage LLVM_PROFILE_FILE, set by
        # dnn-providers/cmake/Sanitizers.cmake / _build_test_environment_list_internal) to
        # every generated suite; without it those suites run unsanitized/uninstrumented.
        # The per-target published variable is that same ambient set merged with anything
        # passed to add_unit_test_target() -- HIPDNN_DESCRIPTOR_DIR, without which the
        # descriptor-backed engines are absent and every test that reads one fails.
        apply_test_category_labels(
            hip_kernel_provider_tests
            "${HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML}"
            "${PROJECT_BINARY_DIR}/staging"
            "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
            ENVIRONMENT ${hip_kernel_provider_tests_TEST_ENVIRONMENT}
        )
        # hip_kernel_provider_integration_tests_TEST_ENVIRONMENT is TEST_ENVIRONMENT
        # merged with any per-target ENVIRONMENT passed to add_integration_test_target()
        # (e.g. HIPDNN_AITER_ASM_DIR for the SDPA engine), published by
        # dnn-providers/cmake/Tests.cmake since the raw test name below is never
        # itself registered as a CTest test in this YAML-categorized mode.
        apply_test_category_labels(
            hip_kernel_provider_integration_tests
            "${HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML}"
            "${PROJECT_BINARY_DIR}/staging"
            "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
            ENVIRONMENT ${hip_kernel_provider_integration_tests_TEST_ENVIRONMENT}
        )
    endif()

    # The two calls above are the only writers of install-time CTest entries for this
    # provider's own test binaries, and they degrade to silence: apply_test_category_labels()
    # shells out to parse_test_categories.py, and a Python3 without PyYAML -- or any other
    # parser failure -- makes it message(WARNING) and return, leaving
    # HIPKERNELPROVIDER_CTEST_INSTALL_FILE as the bare header comment written above. The
    # file is installed regardless, so the install tree ships a CTestTestfile.cmake that
    # registers nothing: measured 0 add_test() against projects/hipdnn's 11, which reaches
    # its install through the unlabeled install_provider_ctest_files() fallback instead.
    #
    # Recorded here, where only those two calls can have contributed, and consumed at the
    # install() decision at the end of this block. Checked after the fact rather than by
    # predicting the parser's health because the failure is the parser's, not a property of
    # the inputs this file can see.
    set(_hkp_install_ctest_has_suites FALSE)
    if(HIPKERNELPROVIDER_CTEST_INSTALL_FILE)
        file(READ "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}" _hkp_install_ctest_text)
        string(FIND "${_hkp_install_ctest_text}" "add_test(" _hkp_install_ctest_pos)
        if(NOT _hkp_install_ctest_pos EQUAL -1)
            set(_hkp_install_ctest_has_suites TRUE)
        endif()
    endif()

    # ------------------------------------------------------------------------
    # The kernel-ingestor engines as engines-under-test
    # ------------------------------------------------------------------------
    # These registrations precede the HIP_MLOPS block below because that block folds
    # EXTERNAL_TEST_INSTALL_STAGING_hip_kernel_provider into the install-time CTest file
    # exactly once. An add_external_integration_test_target() call that lands after the fold
    # stages an add_test() no one reads.

    # The drop-in descriptor tree, generated rather than hand-assembled, so the registered
    # target exercises the supported authoring path end to end: config -> generator ->
    # descriptors + staged bundle -> hipRTC at prepare(). Wiped before each regeneration for
    # the reason the shipped descriptor staging gives -- a kernel dropped from the config
    # must not leave its KDP behind, which in a drop-in root means a withdrawn variant still
    # serving -- and that wipe is also why --force is not needed.
    set(_hkp_dropin_engine "hipkernel:PointwiseHiprtcDropin")
    set(_hkp_dropin_generator "${ROCM_LIBRARIES_ROOT}/projects/hipdnn/tools/IngestorGenerator")
    set(_hkp_dropin_config
        "${CMAKE_CURRENT_SOURCE_DIR}/../config/pointwise_hiprtc_dropin/pointwise_dropin.yaml")
    set(_hkp_dropin_out "${CMAKE_CURRENT_BINARY_DIR}/pointwise_hiprtc_dropin")
    # HIPDNN_DESCRIPTOR_RUNTIME_DIR names the root that CONTAINS descriptor sets, and the
    # generator emits its set under <output>/descriptors/<engine slug>/.
    set(_hkp_dropin_runtime_dir "${_hkp_dropin_out}/descriptors")
    set(_hkp_dropin_install_subdir
        "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/pointwise_hiprtc_dropin")

    set(_hkp_dropin_ready FALSE)
    if(NOT Python3_EXECUTABLE)
        find_package(Python3 COMPONENTS Interpreter QUIET)
    endif()
    if(HIPDNN_ENABLE_KERNEL_INGESTOR AND Python3_EXECUTABLE
       AND EXISTS "${_hkp_dropin_generator}/generate.py" AND EXISTS "${_hkp_dropin_config}")
        # Guarded rather than assumed, the way the kpack fixture guards on hipcc: the
        # generator needs PyYAML and Jinja2, and Python3_EXECUTABLE is whatever the build
        # pinned. A hard dependency would turn a bare interpreter into a configure error for
        # every consumer of this provider.
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import yaml, jinja2"
            RESULT_VARIABLE _hkp_dropin_python_rc
            OUTPUT_QUIET ERROR_QUIET
        )
        if(_hkp_dropin_python_rc EQUAL 0)
            set(_hkp_dropin_ready TRUE)
        else()
            message(STATUS
                "hip-kernel-provider: ${Python3_EXECUTABLE} lacks PyYAML/Jinja2 - "
                "skipping the ${_hkp_dropin_engine} smoke gate")
        endif()
    endif()

    if(_hkp_dropin_ready)
        set(_hkp_dropin_stamp "${CMAKE_CURRENT_BINARY_DIR}/pointwise_hiprtc_dropin.stamp")
        file(GLOB _hkp_dropin_bundle
             "${CMAKE_CURRENT_SOURCE_DIR}/../config/pointwise_hiprtc_dropin/pointwise_dropin_sources/*")
        add_custom_command(
            OUTPUT "${_hkp_dropin_stamp}"
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${_hkp_dropin_out}"
            COMMAND "${Python3_EXECUTABLE}" generate.py
                    --config "${_hkp_dropin_config}"
                    --output-dir "${_hkp_dropin_out}"
            COMMAND ${CMAKE_COMMAND} -E touch "${_hkp_dropin_stamp}"
            DEPENDS "${_hkp_dropin_config}" ${_hkp_dropin_bundle}
            WORKING_DIRECTORY "${_hkp_dropin_generator}"
            COMMENT "Generating the ${_hkp_dropin_engine} drop-in descriptor tree"
            VERBATIM
        )
        add_custom_target(hip_kernel_provider_dropin_descriptors ALL
            DEPENDS "${_hkp_dropin_stamp}"
            COMMENT "The ${_hkp_dropin_engine} drop-in descriptor tree"
        )

        # Installed so the install-tree CTest entry's INSTALL_ENVIRONMENT resolves: the
        # install tree is the one place a drop-in is exercised the way a customer would.
        install(DIRECTORY "${_hkp_dropin_runtime_dir}/"
                DESTINATION "${_hkp_dropin_install_subdir}"
        )
    endif()

    if(TARGET hipdnn_integration_tests)
        # hipkernel:ConvFwd and hipkernel:Pointwise: the shipped ingestor engines, run
        # through the shared graph suite by name. Independent of the drop-in path.
        #
        # REFERENCE_EXECUTOR governs the C++ parameterized tests; the bundle cases these
        # engines actually serve are governed by HIPDNN_TEST_VERIFICATION_MODE, and there is
        # no argument for it. None of these bundles carries golden data, so `auto` would
        # reach the GPU reference by falling through golden -> gpu; naming gpu says the
        # comparison is against a live device reference rather than leaving it to a
        # fallback chain whose next step is a silent skip. The value is path-free, so the
        # documented INSTALL_ENVIRONMENT-defaults-to-ENVIRONMENT fallback carries it into
        # the install tree correctly.
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-hipkernel-conv-fwd-external-integration-check
            PLUGIN_TARGET  hip_kernel_provider
            ENGINE_NAME    hipkernel:ConvFwd
            INSTALL_SUBDIR hip_kernel_provider
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/../config/hipkernel_ConvFwd.toml
            REFERENCE_EXECUTOR gpu
            ENVIRONMENT    "HIPDNN_TEST_VERIFICATION_MODE=gpu"
        )
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-hipkernel-pointwise-external-integration-check
            PLUGIN_TARGET  hip_kernel_provider
            ENGINE_NAME    hipkernel:Pointwise
            INSTALL_SUBDIR hip_kernel_provider
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/../config/hipkernel_Pointwise.toml
            REFERENCE_EXECUTOR gpu
            ENVIRONMENT    "HIPDNN_TEST_VERIFICATION_MODE=gpu"
        )
        # Not gpu, unlike the two above: hipDNN's GPU reference executor has no
        # BatchnormInference plan, so `--verification-mode gpu` reports
        # "GPU reference is not applicable for this graph" and every case skips (0/731
        # passed, measured). The capable independent reference for this operation is the
        # CPU one -- CpuFpReferenceBatchnorm, which is what the kernel's numerics were
        # proved against at batchnorm::getToleranceInference<float>()'s atol = rtol =
        # 2e-4. Naming cpu explicitly rather than leaving `auto` to walk
        # golden -> gpu -> cpu, whose next step after a miss is a silent skip.
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-hipkernel-batchnorm-inference-external-integration-check
            PLUGIN_TARGET  hip_kernel_provider
            ENGINE_NAME    hipkernel:BatchnormInference
            INSTALL_SUBDIR hip_kernel_provider
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/../config/hipkernel_BatchnormInference.toml
            REFERENCE_EXECUTOR cpu
            ENVIRONMENT    "HIPDNN_TEST_VERIFICATION_MODE=cpu"
        )
        # hipkernel:ConvPointwiseRtc, the fused ConvolutionFwd + unary Pointwise pack. Back
        # to gpu, unlike the batchnorm entry directly above, and checked rather than copied
        # from a sibling: the mode question is whether the reference executor implements
        # every op family in the graph, and this graph spans two. Both have a GPU plan
        # builder -- gpu-graph-executor/detail/GpuConvolutionFwdPlan.hpp and
        # GpuPointwisePlan.hpp -- and GpuPointwiseUnaryPlanBuilder::isApplicable admits
        # RELU_FWD, ABS and NEG, which is exactly this pack's activation set. The GPU
        # executor also materializes the virtual intermediate at its own declared type and
        # runs the nodes in order, which is the round trip this fusion elides and therefore
        # the comparison worth making. Named explicitly rather than left to `auto`, whose
        # golden -> gpu -> cpu chain ends in a silent skip.
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-hipkernel-conv-pointwise-rtc-external-integration-check
            PLUGIN_TARGET  hip_kernel_provider
            ENGINE_NAME    hipkernel:ConvPointwiseRtc
            INSTALL_SUBDIR hip_kernel_provider
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/../config/hipkernel_ConvPointwiseRtc.toml
            REFERENCE_EXECUTOR gpu
            ENVIRONMENT    "HIPDNN_TEST_VERIFICATION_MODE=gpu"
        )

        # THIS TARGET IS A SMOKE GATE, NOT A COVERAGE GATE. Green here means the
        # `hiprtc_file` path loaded a descriptor set from HIPDNN_DESCRIPTOR_RUNTIME_DIR,
        # compiled its bundle with hipRTC at prepare(), and served one single-element add
        # graph correctly. It says nothing about kernel quality: the kernel behind it is a
        # reference scaffold, and the suite holds exactly one graph this engine admits.
        #
        # Both ENVIRONMENT and INSTALL_ENVIRONMENT, not one: _stage_external_integration_
        # install_test() falls back to ARG_ENVIRONMENT when INSTALL_ENVIRONMENT is absent,
        # so the install-tree entry would otherwise carry this build tree's absolute path --
        # passing by accident on the build machine and broken for every consumer of the
        # install.
        #
        # INSTALL_ENVIRONMENT carries exactly one entry, and that is a constraint rather
        # than a preference. The staged install text escapes the separator
        # (`string(REPLACE ";" "\\;")` in that macro), and ctest's own script interpreter
        # reads `ENVIRONMENT "A=1\;B=2"` as the single variable `A` with the value
        # `1;B=2` -- measured: with three entries here, the installed entry set no
        # HIPDNN_DESCRIPTOR_RUNTIME_DIR at all and the run failed `Engine ... is not
        # loaded`. One entry is unaffected, and the runtime dir is the one that carries
        # the gate. Verification mode and log level stay on the build-tree ENVIRONMENT,
        # which is a real CMake list and takes all three.
        if(_hkp_dropin_ready)
            set(_hkp_synthetic_root "/__hipdnn_install_root__")
            file(RELATIVE_PATH _hkp_dropin_install_rel
                 "${_hkp_synthetic_root}/${CMAKE_INSTALL_BINDIR}/hip_kernel_provider"
                 "${_hkp_synthetic_root}/${_hkp_dropin_install_subdir}"
            )
            add_external_integration_test_target(
                TARGET_NAME    ${PROJECT_NAME}-hipkernel-pointwise-dropin-external-integration-check
                PLUGIN_TARGET  hip_kernel_provider
                ENGINE_NAME    ${_hkp_dropin_engine}
                INSTALL_SUBDIR hip_kernel_provider
                TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/../config/hipkernel_PointwiseHiprtcDropin.toml
                REFERENCE_EXECUTOR gpu
                ENVIRONMENT
                    "HIPDNN_DESCRIPTOR_RUNTIME_DIR=${_hkp_dropin_runtime_dir}"
                    "HIPDNN_TEST_VERIFICATION_MODE=gpu"
                    "HIPDNN_LOG_LEVEL=info"
                INSTALL_ENVIRONMENT
                    "HIPDNN_DESCRIPTOR_RUNTIME_DIR=${_hkp_dropin_install_rel}"
            )
            add_dependencies(
                ${PROJECT_NAME}-hipkernel-pointwise-dropin-external-integration-check
                hip_kernel_provider_dropin_descriptors
            )
        endif()
    endif()

    set(_external_category_args "")
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
        list(APPEND _external_category_args
            TEST_CATEGORIES_YAML "${HIPKERNELPROVIDER_INTEGRATION_CATEGORIES_YAML}"
            INSTALL_TEST_FILE "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
            TEST_NAME_PREFIX ${PROJECT_NAME}-external-integration
            ENVIRONMENT ${TEST_ENVIRONMENT}
        )
    endif()

    if(TARGET hipdnn_integration_tests)
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-external-integration-check
            PLUGIN_TARGET  hip_kernel_provider
            ENGINE_NAME    HIP_MLOPS_ENGINE
            INSTALL_SUBDIR hip_kernel_provider
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/../config/HIP_MLOPS_ENGINE.toml
            ${_external_category_args}
        )
        message(STATUS "Created ${PROJECT_NAME}-external-integration-check target")

        # This is a pre-registered CTest test (not a GTest binary), so it is
        # labeled via apply_ctest_category_labels()/HIP_MLOPS_ENGINE_test_categories_external.yaml.
        if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
            if(HIPKERNELPROVIDER_CTEST_INSTALL_FILE)
                # Fold the staged add_test() into the labeled file before labeling;
                # apply_ctest_category_labels() scans the file's add_test() names.
                get_property(_external_staging GLOBAL
                    PROPERTY "EXTERNAL_TEST_INSTALL_STAGING_hip_kernel_provider"
                )
                if(_external_staging)
                    file(APPEND "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                         "\n# External integration test (cross-provider suite)\n${_external_staging}"
                    )
                endif()
                # Install tree.
                apply_ctest_category_labels(
                    "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"
                    "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                )
                # Also apply rocKE's pre-registered CTest categories now that
                # the platform's staged add_test() entries (folded above) are
                # present in the file. This overwrites -- rather than merges
                # with -- the ad hoc "ckc;unit;host" / "ckc;python;..." labels
                # rocke/platform/CMakeLists.txt bakes into its install text:
                # ctest's install-tree script interpreter only honors the
                # last set_tests_properties() call per test, and
                # quick/standard/comprehensive/full is the RFC0010-standardized
                # replacement for those ad hoc tags (see ALMIOPEN-2278).
                apply_ctest_category_labels(
                    "${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"
                    "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                )
            endif()
            # Build tree.
            apply_ctest_category_labels(
                "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"
                EXPLICIT_TESTS ${PROJECT_NAME}-external-integration-check
            )
        endif()
    else()
        message(STATUS "hipdnn_integration_tests not found - skipping external-integration-check")
    endif()

    set(_asm_sdpa_external_category_args "")
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
        list(APPEND _asm_sdpa_external_category_args
            TEST_CATEGORIES_YAML "${HIPKERNELPROVIDER_ASM_SDPA_INTEGRATION_CATEGORIES_YAML}"
            INSTALL_TEST_FILE "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
            TEST_NAME_PREFIX ${PROJECT_NAME}-asm-sdpa-external-integration
            ENVIRONMENT ${TEST_ENVIRONMENT}
        )
    endif()

    if(TARGET hipdnn_integration_tests)
        add_external_integration_test_target(
            TARGET_NAME    ${PROJECT_NAME}-asm-sdpa-external-integration-check
            PLUGIN_TARGET  hip_kernel_provider
            ENGINE_NAME    ASM_SDPA_ENGINE
            INSTALL_SUBDIR hip_kernel_provider
            TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/../config/ASM_SDPA_ENGINE.toml
            ${_asm_sdpa_external_category_args}
        )
        message(STATUS "Created ${PROJECT_NAME}-asm-sdpa-external-integration-check target")

        # Unlike the HIP_MLOPS block above, this target does not re-fold
        # EXTERNAL_TEST_INSTALL_STAGING_hip_kernel_provider or re-apply
        # HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML (the provider-wide
        # _external.yaml): that staging/labeling is provider-wide and already
        # applied exactly once above. Re-running it here would re-append the
        # same rocke-staged add_test() entries into
        # HIPKERNELPROVIDER_CTEST_INSTALL_FILE a second time (duplicate CTest
        # test names). ASM_SDPA's own category suites are labeled via
        # TEST_CATEGORIES_YAML/INSTALL_TEST_FILE in
        # _asm_sdpa_external_category_args above.
    else()
        message(STATUS "hipdnn_integration_tests not found - skipping asm-sdpa-external-integration-check")
    endif()

    # Install the ctest files. When YAML-based categorization is active AND its parser ran,
    # install the generated install_CTestTestfile.cmake to bin/hip_kernel_provider/ so
    # TheRock CI's test_runner.py can drive `ctest -L quick` (etc.) from the artifact's
    # install tree. Otherwise fall back to the project's own install_provider_ctest_files(),
    # which emits an unlabeled CTestTestfile.cmake from the registered test targets and the
    # staged external-integration entries.
    #
    # _hkp_install_ctest_has_suites is the second half of that condition and it is the one
    # that matters: a categorization run whose parser failed leaves a file with no add_test()
    # at all, and installing it ships a test list that registers nothing. Unlabeled tests are
    # a tier-selection loss; no tests is the provider unverifiable from an install.
    if(HIPKERNELPROVIDER_CTEST_INSTALL_FILE AND _hkp_install_ctest_has_suites)
        install(FILES "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/hip_kernel_provider"
                RENAME "CTestTestfile.cmake"
        )
    else()
        if(HIPKERNELPROVIDER_CTEST_INSTALL_FILE)
            message(WARNING
                "hip-kernel-provider: the test-category parser registered no install-time "
                "tests (see the 'Failed to parse test categories YAML' warnings above); "
                "installing unlabeled CTest entries instead so the install tree is testable")
        endif()
        install_provider_ctest_files("hip_kernel_provider")
    endif()
endif()

# Process all kernels marked for inlining
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/KernelEmbedding.cmake")
# Generate embedded kernel sources and includes at configure time
embed_kernel_sources(
    TARGET hip_kernel_provider_impl
    OUTPUT_SRCS_CPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    OUTPUT_SRCS_HPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.hpp
    OUTPUT_INCS_CPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp
    OUTPUT_INCS_HPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.hpp
)

set(HIP_KERNEL_PROVIDER_GENERATED_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp)

set_source_files_properties(${HIP_KERNEL_PROVIDER_GENERATED_SOURCES} TARGET_DIRECTORY hip_kernel_provider_impl PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")

target_sources(hip_kernel_provider_impl PRIVATE "${HIP_KERNEL_PROVIDER_GENERATED_SOURCES}")
