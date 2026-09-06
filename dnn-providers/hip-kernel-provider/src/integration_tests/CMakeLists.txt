# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

# hipdnn SDK dependencies - use targets directly if available (superbuild),
# otherwise find installed packages (true standalone)
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    hip_kernel_provider_integration_tests
    main.cpp
)

target_compile_options(hip_kernel_provider_integration_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# To use the build tree plugin, we use the path relative to the test binary instead of the installed
# libhipdnn_backend.so HIPDNN_PLUGIN_ENGINE_SUBDIR is provided by the sdk
target_compile_definitions(
    hip_kernel_provider_integration_tests
    PRIVATE
        PLUGIN_PATH="../$<IF:$<PLATFORM_ID:Windows>,bin,lib>/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/hip_kernel_provider"
)

target_include_directories(hip_kernel_provider_integration_tests PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)

target_link_libraries(
    hip_kernel_provider_integration_tests PUBLIC GTest::gtest hip::host hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# Ensure that the hip_kernel_plugin is built before the integration tests
add_dependencies(hip_kernel_provider_integration_tests hip_kernel_provider)

set(HIP_KERNEL_PROVIDER_ASM_KERNEL_ENV "")
if(ENABLE_ASM_SDPA_ENGINE)
    if(NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "ENABLE_ASM_SDPA_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
                "Skipping SDPA integration tests. Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable them.")
    else()
        target_sources(hip_kernel_provider_integration_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/engines/asm_sdpa_engine/AsmSdpaConfigHelpers.cpp
            asm_sdpa_engine/IntegrationGpuSdpaForward.cpp
        )
        # Add include path for codegen-generated headers (e.g., asm_fmha_v3_fwd_configs.hpp)
        # and the shared test helpers (AsmSdpaConfigHelpers.hpp)
        target_include_directories(hip_kernel_provider_integration_tests PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/../engines/asm_sdpa_engine/generated
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/engines/asm_sdpa_engine
        )
        # Ensure codegen runs before compiling the integration tests
        add_dependencies(hip_kernel_provider_integration_tests asm_kernel_codegen)
        target_compile_definitions(hip_kernel_provider_integration_tests PRIVATE
            AITER_ASM_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../engines/asm_sdpa_engine/asm/asm_kernels"
        )
        set(HIP_KERNEL_PROVIDER_ASM_KERNEL_ENV
            "HIPDNN_AITER_ASM_DIR=${CMAKE_BINARY_DIR}/hip_kernel_provider/asm_kernels")
    endif()
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    target_sources(hip_kernel_provider_integration_tests PRIVATE
        hip_mlops_engine/IntegrationGpuPassByValue.cpp
    )
endif()

if(ENABLE_HIP_FLASH2_ENGINE)
    if(NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "ENABLE_HIP_FLASH2_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
                "Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable Flash2 integration tests.")
    else()
        add_subdirectory(hip_flash2_engine)
    endif()
endif()
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    target_sources(hip_kernel_provider_integration_tests PRIVATE
        kernel_ingestor_engine/IntegrationGpuKernelIngestor.cpp
        kernel_ingestor_engine/IntegrationGpuKernelIngestorKpack.cpp
    )
    # This target links only the SDKs, not hip_kernel_provider_private, so it does not
    # inherit the provider's gate and needs the definition set directly.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPDNN_ENABLE_KERNEL_INGESTOR=1
    )
    # The relative layout only, never an absolute build path: the kpack suite locates the
    # staged tree by walking from the loaded plugin exactly as the loader does, and needs
    # the same spelling of the subdirectory the provider compiled in. An absolute path
    # here would be baked into the installed test binary as well, naming a build tree it
    # cannot see.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPDNN_PACKAGED_FIXTURE_SUBDIR="${HIPDNN_DESCRIPTOR_SUBDIR}"
    )
    # Guarded rather than assumed: a root skips with a warning when hipcc is absent, and
    # the suite is written to GTEST_SKIP in exactly that case. A hard dependency would
    # turn a missing compiler into a configure error. hkp_descriptor_staging is the one
    # target that owns the staged tree, whichever roots contributed to it.
    if(TARGET hkp_descriptor_staging)
        add_dependencies(hip_kernel_provider_integration_tests hkp_descriptor_staging)
    endif()
endif()

clang_tidy_check(hip_kernel_provider_integration_tests)

# ENVIRONMENT is forwarded to add_integration_test_target() rather than set
# afterward via set_tests_properties(): when YAML-driven CTest
# categorization is active (DNN_PROVIDER_CTEST_CATEGORIES_YAML), the raw
# hip_kernel_provider_integration_tests CTest test is never registered --
# only the tiered suites created later by apply_test_category_labels() in
# ../CMakeLists.txt are, and set_tests_properties() on the unregistered
# name is a hard CMake configure error. add_integration_test_target()
# publishes the merged environment as
# hip_kernel_provider_integration_tests_TEST_ENVIRONMENT so that later call
# can forward it to whichever suites actually exist.
#
# No descriptor entry: this suite loads the real plugin, and the engine resolves its
# descriptor tree relative to the loaded module -- the staged copy in a build tree, the
# installed copy in an install tree. Naming one explicitly would pin the build path into
# the install-time CTest file and stop the installed descriptors from ever being read.
add_integration_test_target(hip_kernel_provider_integration_tests ${CMAKE_CURRENT_BINARY_DIR}
    LABELS slow
    ENVIRONMENT ${HIP_KERNEL_PROVIDER_ASM_KERNEL_ENV})

# Forward the merged raw-test environment (published by add_integration_test_target)
# to the parent scope so the tiered CTest suites created in ../CMakeLists.txt can
# pick it up. Setting ENVIRONMENT directly on the raw test name here is a hard
# configure error when YAML categorization is active, hence the forwarding.
if(DEFINED hip_kernel_provider_integration_tests_TEST_ENVIRONMENT)
    set(hip_kernel_provider_integration_tests_TEST_ENVIRONMENT
        "${hip_kernel_provider_integration_tests_TEST_ENVIRONMENT}" PARENT_SCOPE)
endif()

# Register the cross-provider ASM SDPA gpu-reference integration test target. In
# standalone provider builds the harness may not already be in scope, so try to
# import it first.
if(NOT COMMAND add_external_integration_test_target)
    find_package(hipdnn_integration_tests CONFIG QUIET)
endif()

if(ENABLE_ASM_SDPA_ENGINE AND HIPDNN_ENABLE_SDPA
   AND TARGET hipdnn_integration_tests AND COMMAND add_external_integration_test_target)
    set(_synthetic_root "/__hipdnn_install_root__")
    set(_install_cwd_abs "${_synthetic_root}/${CMAKE_INSTALL_BINDIR}/hip_kernel_provider")
    set(_install_asm_abs
        "${_synthetic_root}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/asm_kernels"
    )
    file(RELATIVE_PATH _install_asm_dir "${_install_cwd_abs}" "${_install_asm_abs}")

    add_external_integration_test_target(
        TARGET_NAME hip_kernel_provider_asm_sdpa_gpu_ref_integration_tests
        PLUGIN_TARGET hip_kernel_provider
        ENGINE_NAME ASM_SDPA_ENGINE
        INSTALL_SUBDIR hip_kernel_provider
        REFERENCE_EXECUTOR gpu
        GTEST_FILTER
            Smoke/IntegrationGpuSdpaFwdBfp16.Correctness/*
        ENVIRONMENT
            ${HIP_KERNEL_PROVIDER_ASM_KERNEL_ENV}
        INSTALL_ENVIRONMENT
            "HIPDNN_AITER_ASM_DIR=${_install_asm_dir}"
    )
endif()

# Register the gfx942 dense-attention ingestor engine's own gpu-reference target.
#
# Bundles are discovered once and run against whichever engine WINS the graph, so
# without an engine-pinned target this engine can sit entirely unexercised while the
# suite reports green -- it competes with ASM_SDPA_ENGINE for the same SDPA_FWD node.
# The BSHD bundles this integration adds are the ones no other engine can serve, but
# nothing in the suite guarantees THIS engine is the one that served them.
#
# Gated on the flags that actually PRODUCE the engine, not merely on the op being
# enabled: the descriptors exist only when the ingestor is on AND the rocKE producer
# ran over a production source root. Without that conjunction the engine never
# registers and every pinned case FAILS rather than skipping.
#
# --test-engine matches against the loaded engines' `engineName`. The provider's
# Container::getEngineName now answers for descriptor-registered engines too (#11336),
# filling the name from the UED's own `name` field -- so this pin resolves and the target
# runs the cases against THIS engine rather than whichever one wins the graph.
#
# Do NOT "fix" a failure here by dropping ENGINE_NAME or pinning a different engine: the
# whole point is that an unpinned run can report green while a different engine served
# every case.
if(HIPDNN_ENABLE_KERNEL_INGESTOR AND HIPDNN_ENABLE_SDPA
   AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE
   AND TARGET hipdnn_integration_tests AND COMMAND add_external_integration_test_target)
    add_external_integration_test_target(
        TARGET_NAME hip_kernel_provider_gfx942_attention_dense_gpu_ref_integration_tests
        PLUGIN_TARGET hip_kernel_provider
        # Exactly as the UED spells it.
        ENGINE_NAME hipkernel:Gfx942AttentionDense
        INSTALL_SUBDIR hip_kernel_provider
        # Computes the reference live, which is what makes the golden-free BSHD bundles
        # checkable. The GPU reference accepts these graphs: they are dense and
        # stride-based, with no paging, varlen, ragged, stats, dropout or fp8 descale.
        REFERENCE_EXECUTOR gpu
        # The bshd bundles only, quick AND standard. The suite name is the bundle's
        # own path (quick_SdpaFwd_bshd_bf16_hd128_causal_batch_Small.Small), NOT the
        # Smoke/<Fixture>.Correctness/* shape a C++ INSTANTIATE_TEST_SUITE_P produces --
        # verified against --gtest_list_tests. Every other SdpaFwd case is BHSD and this
        # engine correctly declines it, so those are EXPECTED TO SKIP under this pin:
        # --test-engine pins exactly one engine and a case belonging to another pack
        # reports not-applicable.
        GTEST_FILTER
            *_SdpaFwd_bshd_*.*
    )
endif()

# Register the gfx950 dense-attention ingestor engine's own gpu-reference target.
#
# Same rationale as the gfx942 block above -- bundles run against whichever engine
# WINS the graph, so without a pin this engine can sit unexercised while the suite
# reports green -- and the same gating, for the same reason: the descriptors exist
# only when the ingestor is on AND the rocKE producer ran, so without that
# conjunction every pinned case FAILS rather than skips.
#
# The two ingestor targets do NOT compete despite sharing the bundle filter. Each
# pack declares a single, different `arch` and packs arch-prune BEFORE the matcher
# runs, so at most one of them can match on a given device: on a gfx942 device the
# gfx950 pack is pruned and every case here reports not-applicable, and vice versa.
# That is also why this target is registered unconditionally rather than behind an
# arch check -- the pruning is the arch check, and doing it twice in two places is
# how the two answers drift apart.
#
# READ THE FRACTION, NOT THE COLOUR. This engine's shipped set is dispatcher-derived
# from rocKE's own prefill traces, and it matches only a handful of the SdpaFwd
# bundles that ship today -- most are BHSD, which this kernel bakes against and
# correctly declines. A green run of this target therefore proves very little on its
# own: it must be read as "N of M dispatched THIS engine, K skipped for a named
# reason, 0 unexplained". A pass where N is small is a coverage finding, not a
# success, and the fix is more bundles rather than a wider filter.
if(HIPDNN_ENABLE_KERNEL_INGESTOR AND HIPDNN_ENABLE_SDPA
   AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE
   AND TARGET hipdnn_integration_tests AND COMMAND add_external_integration_test_target)
    add_external_integration_test_target(
        TARGET_NAME hip_kernel_provider_gfx950_attention_dense_gpu_ref_integration_tests
        PLUGIN_TARGET hip_kernel_provider
        # Exactly as the UED spells it; --test-engine matches the engine's engineName,
        # which for a descriptor-registered engine comes from the UED's own `name`.
        ENGINE_NAME hipkernel:Gfx950AttentionDense
        INSTALL_SUBDIR hip_kernel_provider
        # Computes the reference live. The GPU reference accepts these graphs -- dense,
        # stride-based, no paging/varlen/ragged-tensor/stats/dropout/fp8. Note it also
        # implements sliding-window bounds, but this engine declines windowed graphs,
        # so no windowed case reaches it.
        REFERENCE_EXECUTOR gpu
        # The bshd bundles only, quick AND standard: the suite name is the bundle's own
        # path, not the Smoke/<Fixture>.Correctness/* shape a C++
        # INSTANTIATE_TEST_SUITE_P produces. Every other SdpaFwd case is BHSD and this
        # engine correctly declines it, so those are EXPECTED TO SKIP under this pin.
        GTEST_FILTER
            *_SdpaFwd_bshd_*.*
    )
endif()

# The gfx950 TILED-attention engine, at the hand-invoke site.
#
# THIS SITE IS NOT THE ONE CI DRIVES. It registers a build-tree target only; the
# installed CTestTestfile that TheRock's test_runner.py drives is written by the fold
# in ../CMakeLists.txt, where this engine also has a block. Both exist for the same
# reason the dense sibling has both: this one is convenient to invoke by hand while
# developing, that one is the one that defends the engine in CI. Deleting either
# leaves a real gap.
#
# EXPECTED TO BE INERT ON MERGE (decision D2), for the same reason as the sibling
# block at the CI site: this kernel is structurally paged, and both shared reference
# executors decline paged KV, so there is nothing for these cases to be checked
# against until the GpuSdpaFwdPlan block-table repair (D1) lands. The target is
# correctly pinned now so that repair makes it live without touching this file.
#
# Read the result as "N of M dispatched THIS engine, K skipped for a named reason, 0
# unexplained" -- never as a colour. A pass with N == 0 is a coverage finding, not a
# success.
if(HIPDNN_ENABLE_KERNEL_INGESTOR AND HIPDNN_ENABLE_SDPA
   AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE
   AND TARGET hipdnn_integration_tests AND COMMAND add_external_integration_test_target)
    add_external_integration_test_target(
        TARGET_NAME hip_kernel_provider_gfx950_attention_tiled_gpu_ref_integration_tests
        PLUGIN_TARGET hip_kernel_provider
        # Exactly as the UED spells it; --test-engine matches the engine's engineName,
        # which for a descriptor-registered engine comes from the UED's own `name`.
        ENGINE_NAME hipkernel:Gfx950AttentionTiled
        INSTALL_SUBDIR hip_kernel_provider
        # gpu, matching the dense sibling, so this target goes live unchanged once D1's
        # repair teaches that executor block-table indirection. Pinning the CPU
        # reference instead would not help: test_sdk's SdpaFwdPlan.hpp declines paged
        # KV too.
        REFERENCE_EXECUTOR gpu
        # The paged bundles this integration authors. Every dense SdpaFwd bundle is
        # correctly declined -- this engine REQUIRES page tables, the inverse of the
        # dense engine's rule -- so a wider filter would only add skips that say
        # nothing about this engine.
        GTEST_FILTER
            *SdpaFwd_paged*
    )
endif()
