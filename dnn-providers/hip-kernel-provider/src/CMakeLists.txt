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

        # The gfx950 dense-attention ingestor engine, registered HERE as well as
        # in src/integration_tests/CMakeLists.txt, because the two sites do
        # different jobs and only this one is driven by CI. The inner site's
        # target works when invoked by hand and never reaches the installed
        # CTestTestfile.cmake, so an engine registered only there sits
        # unexercised while the suite reports green.
        #
        # ORDERING IS LOAD-BEARING: this must run BEFORE the fold below.
        # add_external_integration_test_target() only APPENDS its install-tree
        # add_test() text to the global EXTERNAL_TEST_INSTALL_STAGING_<subdir>
        # property (HipdnnIntegrationTestHelpers.cmake:167, APPEND_STRING); the
        # block below is what writes that accumulated text into the installed
        # file. Registering after the fold creates the build-tree target, prints
        # a reassuring "Created ..." line, and silently omits the test from the
        # install tree -- measured exactly that way on a gfx950 device run:
        # target created, yet `ctest -N` in the install tree matched zero.
        #
        # Gated on the flags that PRODUCE the engine, not merely on the ingestor
        # being compiled: without the rocKE production pack there are no
        # descriptors, the engine never registers, and every case would FAIL
        # rather than skip.
        if(HIPDNN_ENABLE_KERNEL_INGESTOR
           AND HIPDNN_ENABLE_SDPA
           AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE)
            set(_gfx950_dense_external_category_args "")
            if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
                list(APPEND _gfx950_dense_external_category_args
                    INSTALL_TEST_FILE "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                    TEST_NAME_PREFIX ${PROJECT_NAME}-gfx950-attention-dense-external-integration
                    ENVIRONMENT ${TEST_ENVIRONMENT}
                )
            endif()
            add_external_integration_test_target(
                TARGET_NAME    ${PROJECT_NAME}-gfx950-attention-dense-external-integration-check
                PLUGIN_TARGET  hip_kernel_provider
                # Exactly as the UED spells it: --test-engine matches engineName,
                # which for a descriptor-registered engine comes from the UED's
                # own `name` field.
                ENGINE_NAME    hipkernel:Gfx950AttentionDense
                INSTALL_SUBDIR hip_kernel_provider
                REFERENCE_EXECUTOR gpu
                # bshd only, quick AND standard. Every other SdpaFwd bundle is
                # BHSD and this engine correctly declines it, so those are
                # EXPECTED to skip under this pin -- read the dispatched
                # FRACTION, never the exit code.
                GTEST_FILTER   *_SdpaFwd_bshd_*.*
                ${_gfx950_dense_external_category_args}
            )
            message(STATUS "Created ${PROJECT_NAME}-gfx950-attention-dense-external-integration-check target")
        else()
            message(STATUS "gfx950 attention-dense engine not built - skipping its external-integration-check")
        endif()

        # The gfx950 TILED-attention ingestor engine. Registered at this site for the
        # same reason the dense block above is, and subject to the same load-bearing
        # ORDERING: it must precede the fold below, which is what writes the staged
        # add_test() text into the installed CTestTestfile. A block placed after the
        # fold creates the build-tree target, prints a reassuring "Created ..." line,
        # and is silently absent from the install tree CI drives.
        #
        # THIS TARGET IS EXPECTED TO BE INERT ON MERGE, AND THAT IS DELIBERATE
        # (decision D2). It pins the right engine by the right name, and there is
        # currently no bundle it can serve: this kernel is structurally paged and
        # varlen, and BOTH shared reference executors decline paged KV
        # (GpuSdpaFwdPlan.hpp and test_sdk's SdpaFwdPlan.hpp), so a paged bundle has
        # nothing to be checked against and every case reports not-applicable.
        #
        # An inert CORRECT target beats a green MISLEADING one. The alternative --
        # dropping ENGINE_NAME, or pinning a different engine so something runs --
        # would report a fraction about somebody else's engine. What makes this
        # target live is the GpuSdpaFwdPlan block-table repair (D1), which is a
        # bounded gather-then-delegate adapter over the existing dense reference; the
        # moment it lands, these cases dispatch without touching this block.
        #
        # Until then the numeric evidence for this engine comes from rocKE's own
        # ref_paged_attn (pure torch, 79 scenarios) at stage 8d, reported as a
        # fraction rather than a colour.
        if(HIPDNN_ENABLE_KERNEL_INGESTOR
           AND HIPDNN_ENABLE_SDPA
           AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE)
            set(_gfx950_tiled_external_category_args "")
            if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED)
                list(APPEND _gfx950_tiled_external_category_args
                    INSTALL_TEST_FILE "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                    TEST_NAME_PREFIX ${PROJECT_NAME}-gfx950-attention-tiled-external-integration
                    ENVIRONMENT ${TEST_ENVIRONMENT}
                )
            endif()
            add_external_integration_test_target(
                TARGET_NAME    ${PROJECT_NAME}-gfx950-attention-tiled-external-integration-check
                PLUGIN_TARGET  hip_kernel_provider
                # Exactly as the UED spells it: --test-engine matches engineName,
                # which for a descriptor-registered engine comes from the UED's own
                # `name` field. A mismatch here fails before any case runs.
                ENGINE_NAME    hipkernel:Gfx950AttentionTiled
                INSTALL_SUBDIR hip_kernel_provider
                # gpu, matching the dense sibling, so this target goes live unchanged
                # when D1's repair teaches that executor block-table indirection.
                REFERENCE_EXECUTOR gpu
                # The paged bundles this integration authors. Every dense SdpaFwd
                # bundle is correctly declined by this engine -- it requires page
                # tables -- so a wider filter would report skips that say nothing.
                GTEST_FILTER   *_SdpaFwd_paged_*.*
                ${_gfx950_tiled_external_category_args}
            )
            message(STATUS "Created ${PROJECT_NAME}-gfx950-attention-tiled-external-integration-check target")
        else()
            message(STATUS "gfx950 attention-tiled engine not built - skipping its external-integration-check")
        endif()

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

    # Install the ctest files. When YAML-based categorization is active, install the
    # generated install_CTestTestfile.cmake to bin/hip_kernel_provider/ so TheRock CI's
    # test_runner.py can drive `ctest -L quick` (etc.) from the artifact's install tree.
    # Otherwise, fall back to the project's own install_provider_ctest_files() which
    # emits an unlabeled CTestTestfile.cmake.
    if(HIPKERNELPROVIDER_CTEST_INSTALL_FILE)
        install(FILES "${HIPKERNELPROVIDER_CTEST_INSTALL_FILE}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/hip_kernel_provider"
                RENAME "CTestTestfile.cmake"
        )
    else()
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
