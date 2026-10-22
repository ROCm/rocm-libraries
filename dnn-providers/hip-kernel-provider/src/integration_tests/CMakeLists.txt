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
    IntegrationGraphVerificationReference.cpp
)

target_compile_options(hip_kernel_provider_integration_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# To use the build tree plugin, we use the path relative to the test binary instead of the installed
# libhipdnn_backend.so. HIPKERNELPROVIDER_PLUGIN_ENGINE_RELDIR is the offset from the binary
# directory to the plugin engine directory, derived from the configured install layout.
target_compile_definitions(
    hip_kernel_provider_integration_tests
    PRIVATE
        PLUGIN_PATH="${HIPKERNELPROVIDER_PLUGIN_ENGINE_RELDIR}/hip_kernel_provider"
)

target_include_directories(hip_kernel_provider_integration_tests PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..
)

target_link_libraries(
    hip_kernel_provider_integration_tests PUBLIC GTest::gtest hip::host hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# Ensure that the hip_kernel_plugin is built before the integration tests
add_dependencies(hip_kernel_provider_integration_tests hip_kernel_provider)

if(ENABLE_ASM_SDPA_ENGINE)
    if(NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "ENABLE_ASM_SDPA_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
                "Skipping SDPA integration tests. Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable them.")
    else()
        target_sources(hip_kernel_provider_integration_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/engines/asm_sdpa_engine/AsmSdpaConfigHelpers.cpp
            asm_sdpa_engine/IntegrationGpuSdpaForward.cpp
            asm_sdpa_engine/IntegrationGpuSdpaBackward.cpp
        )
        # Add include path for codegen-generated headers (e.g., asm_fmha_v3_fwd_configs.hpp)
        # and the shared test helpers (AsmSdpaConfigHelpers.hpp)
        target_include_directories(hip_kernel_provider_integration_tests PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/../engines/asm_sdpa_engine/generated
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/engines/asm_sdpa_engine
        )
        # Ensure codegen runs before compiling the integration tests
        add_dependencies(hip_kernel_provider_integration_tests asm_kernel_codegen)
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
    # The attention knob suites: each a source and the comma-separated engines it pins. A
    # suite is compiled whenever the product root carries all of them, on any build arch,
    # so a binary built for one arch still runs it against another build's content for the
    # engines' arch, and it GTEST_SKIPs on other devices. The predicate is arch-agnostic and
    # lives in HkpPackaging.cmake, which only an ingestor build includes -- hence this guard.
    set(_knob_suites
        IntegrationGpuGfx950AttentionDenseKnobs.cpp Gfx950AttentionDense
        IntegrationGpuGfx950AttentionUnifiedKnobs.cpp
            Gfx950AttentionUnified2D,Gfx950AttentionUnified3D
        IntegrationGpuGfx942AttentionUnified3dKnobs.cpp Gfx942AttentionUnified3d
        IntegrationGpuGfx942AttentionDenseKnobs.cpp Gfx942AttentionDense
    )
    while(_knob_suites)
        list(POP_FRONT _knob_suites _source _engines)
        string(REPLACE "," ";" _engines "${_engines}")
        set(_offered TRUE)
        foreach(_engine IN LISTS _engines)
            hkp_product_offers_engine(_offers "hipkernel:${_engine}")
            if(NOT _offers)
                set(_offered FALSE)
            endif()
        endforeach()
        if(_offered AND NOT HIPDNN_ENABLE_SDPA)
            message(WARNING
                    "HIPDNN_ENABLE_KERNEL_INGESTOR is ON but hipDNN was built with "
                    "HIPDNN_ENABLE_SDPA=OFF. Skipping ${_source}. Rebuild hipDNN with "
                    "-DHIPDNN_ENABLE_SDPA=ON to enable it.")
        elseif(_offered)
            target_sources(hip_kernel_provider_integration_tests PRIVATE
                kernel_ingestor_engine/${_source}
            )
        endif()
    endwhile()
    # The dgrad AOT-instance engine is a product bundle of its own, gated like the knob
    # suites; it needs no SDPA.
    hkp_product_offers_engine(_hkp_conv_dgrad_aot_bundle "hipkernel:Gfx950ConvDgradAot")
    if(_hkp_conv_dgrad_aot_bundle)
        target_sources(hip_kernel_provider_integration_tests PRIVATE
            kernel_ingestor_engine/IntegrationGpuGfx950ConvDgrad.cpp
        )
        target_compile_definitions(hip_kernel_provider_integration_tests
            PRIVATE HKP_TEST_GFX950_CONV_DGRAD_AOT=1
        )
    endif()
    # The paged-K/V suite of the RDNA prefill engine, gated like the knob suites; it
    # GTEST_SKIPs on devices the engine's objects are not built for.
    hkp_product_offers_engine(_hkp_rdna_attention_bundle "hipkernel:Gfx1151WmmaAttention")
    if(_hkp_rdna_attention_bundle AND NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "HIPDNN_ENABLE_KERNEL_INGESTOR is ON but hipDNN was built with "
                "HIPDNN_ENABLE_SDPA=OFF. Skipping IntegrationGpuGfx1151WmmaAttentionPaged.cpp. "
                "Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable it.")
    elseif(_hkp_rdna_attention_bundle)
        target_sources(hip_kernel_provider_integration_tests PRIVATE
            kernel_ingestor_engine/IntegrationGpuGfx1151WmmaAttentionPaged.cpp
        )
    endif()
    target_sources(hip_kernel_provider_integration_tests PRIVATE
        kernel_ingestor_engine/IntegrationGpuKernelIngestor.cpp
        kernel_ingestor_engine/IntegrationGpuKernelIngestorKpack.cpp
        kernel_ingestor_engine/IntegrationGraphVerificationContext.cpp
    )
    # This target links only the SDKs, not hip_kernel_provider_private, so it does not
    # inherit the provider's gate and needs the definition set directly.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPDNN_ENABLE_KERNEL_INGESTOR=1
    )
    # The one descriptor set this binary's cases resolve, as the offset from the binary to
    # the staged tree. main() applies it when the environment names no set, so the binary
    # runs standalone, and the offset holds in the install tree too, so no machine-specific
    # path is compiled in or reaches the CTest files.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPKERNELPROVIDER_TEST_SET_INTEGRATION_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_INTEGRATION}"
    )
    # The production descriptor tree, loaded beside the integration set so the shipped
    # engines load in this binary too; the attention knob suites above need their
    # engines. A fixed binary-relative offset like the set above, so it holds in the build
    # and the install tree alike, and main() uses it only when the directory exists at run
    # time. The definition must not depend on this build's GPU targets: a multi-arch
    # install pairs this binary with another build's device content, so a build that packs
    # no production tree still names where one would be. The dependency applies only when
    # this configuration packs one (the top-level install rule keys on the same target).
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR="${HIPKERNELPROVIDER_PLUGIN_ENGINE_RELDIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}"
    )
    if(TARGET hkp_packaging_product)
        add_dependencies(hip_kernel_provider_integration_tests hkp_packaging_product)
    endif()
    # The shared descriptor-root resolver lives with the unit tests
    target_include_directories(hip_kernel_provider_integration_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests
    )
    # The relative layout only, never an absolute build path: the kpack suite locates the
    # staged tree by walking from the loaded plugin exactly as the loader does, and needs
    # the same spelling of the subdirectory the provider compiled in. An absolute path
    # here would be baked into the installed test binary as well, naming a build tree it
    # cannot see.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPKERNELPROVIDER_PACKAGED_FIXTURE_SUBDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_SUBDIR}/${HIPKERNELPROVIDER_TEST_SET_INTEGRATION}/${HIPKERNELPROVIDER_TEST_SET_ARCHIVE_FIXTURE}"
    )
    # The integration root is packed by three roots, and each stages under its own
    # packaging target. Name each one individually.
    foreach(_hkp_set hkp_packaging_integration hkp_packaging_integration_shared
                     hkp_packaging_archive_fixture)
        if(TARGET ${_hkp_set})
            add_dependencies(hip_kernel_provider_integration_tests ${_hkp_set})
        endif()
    endforeach()

    # The parent of the integration descriptor sets, which holds all three. A
    # walk of a parent reaches every subtree staged under it, so a further pack
    # there comes under the check with no edit here.
    hkp_verify_embedded_sources(
        TARGET hip_kernel_provider_integration_tests
        STAGED_DESCRIPTOR_ROOTS ${HIPKERNELPROVIDER_INTEGRATION_BUILD_DIR}
        PACK_NAMES integration integration_shared archive_fixture
    )
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
    LABELS slow)

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

# One gpu_ref entry per HKP_PINNED_ATTENTION_ENGINES engine (src/CMakeLists.txt), serving
# the bundles its filter variable names (the unified 3D engines serve only their decode,
# seqlen_q 1, cases).
#
# Gated on the same predicate as the census and the external integration target:
# ENGINE_NAME pins this target, so a build that does not ship the engine registers one
# whose engine never appears. The predicate lives in HkpPackaging.cmake, which only an
# ingestor build includes.
#
# Routed into the engine's architecture shard, not the common arch-neutral directory: the
# handler prunes by the first gfx directory in the path, so the entry goes when the engine
# does. Left in the common file it survives into a package for another architecture,
# where the engine ranks nothing, every case declines and the entry reports success
# having run nothing.
#
# hkp_finalize_census_install() folds the accumulator into that shard's CTest file,
# in both YAML modes: src/ hands it whichever common entrypoint that configuration
# produced, and the discovery stub goes there, so a sharded entry is reachable
# whether or not categorization is on.
#
# An unset filter variable expands to "", and add_external_integration_test_target() then
# passes no --gtest_filter: the engine-pinned target runs every bundle.
if(NOT HKP_PINNED_ATTENTION_ENGINES)
    message(FATAL_ERROR
        "HKP_PINNED_ATTENTION_ENGINES must be set in src/CMakeLists.txt before "
        "add_subdirectory(integration_tests)")
endif()
set(_engines ${HKP_PINNED_ATTENTION_ENGINES})
while(_engines)
    list(POP_FRONT _engines _arch _snake _local _filter_var _toml _categories_var)
    if(NOT ${_filter_var})
        message(FATAL_ERROR
            "${_filter_var} must be set in src/CMakeLists.txt before "
            "add_subdirectory(integration_tests)")
    endif()

    set(_available FALSE)
    if(HIPDNN_ENABLE_KERNEL_INGESTOR)
        hkp_product_ships_engine_for_arch(_available "hipkernel:${_local}" ${_arch})
    endif()
    if(HIPDNN_ENABLE_KERNEL_INGESTOR AND HIPDNN_ENABLE_SDPA
       AND _available
       AND TARGET hipdnn_integration_tests AND COMMAND add_external_integration_test_target)
        hkp_reserve_census_shard(_shard_dest ${_arch})

        add_external_integration_test_target(
            TARGET_NAME hip_kernel_provider_${_arch}_attention_${_snake}_gpu_ref_integration_tests
            PLUGIN_TARGET hip_kernel_provider
            ENGINE_NAME   "hipkernel:${_local}"
            INSTALL_SUBDIR hip_kernel_provider
            INSTALL_DESTINATION "${_shard_dest}"
            INSTALL_STAGING_KEY hkp_census_${_arch}
            REFERENCE_EXECUTOR gpu
            GTEST_FILTER "${${_filter_var}}"
            ENVIRONMENT_MODIFICATION ${TEST_ENVIRONMENT_MODIFICATION}
        )
    endif()
endwhile()

if(ENABLE_ASM_SDPA_ENGINE AND HIPDNN_ENABLE_SDPA
   AND TARGET hipdnn_integration_tests AND COMMAND add_external_integration_test_target)
    add_external_integration_test_target(
        TARGET_NAME hip_kernel_provider_asm_sdpa_gpu_ref_integration_tests
        PLUGIN_TARGET hip_kernel_provider
        ENGINE_NAME ASM_SDPA_ENGINE
        INSTALL_SUBDIR hip_kernel_provider
        REFERENCE_EXECUTOR gpu
        GTEST_FILTER
            Smoke/IntegrationGpuSdpaFwdBfp16.Correctness/*
        ENVIRONMENT_MODIFICATION
            ${TEST_ENVIRONMENT_MODIFICATION}
    )
endif()
