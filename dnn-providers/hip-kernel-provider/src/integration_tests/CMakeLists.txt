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

# HIPKERNELPROVIDER_PLUGIN_ENGINE_RELDIR is the offset from the test binary to the plugin
# engine directory, so the build tree plugin is used rather than the installed one.
target_compile_definitions(
    hip_kernel_provider_integration_tests
    PRIVATE
        PLUGIN_PATH="${HIPKERNELPROVIDER_PLUGIN_ENGINE_RELDIR}/hip_kernel_provider"
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
        kernel_ingestor_engine/IntegrationGraphVerificationContext.cpp
    )
    # Links only the SDKs, not hip_kernel_provider_private, so the gate is not inherited.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPDNN_ENABLE_KERNEL_INGESTOR=1
    )
    # Descriptor set resolved as an offset from the binary to the staged tree, applied by
    # main() when the environment names no set. The offset holds in the install tree too.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPKERNELPROVIDER_TEST_SET_INTEGRATION_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_INTEGRATION}"
    )
    # The shared descriptor-root resolver lives with the unit tests.
    target_include_directories(hip_kernel_provider_integration_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests
    )
    # Relative layout only: the kpack suite walks from the loaded plugin as the loader
    # does and needs the same subdirectory spelling the provider compiled in.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPKERNELPROVIDER_PACKAGED_FIXTURE_SUBDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_SUBDIR}/${HIPKERNELPROVIDER_TEST_SET_INTEGRATION}/${HIPKERNELPROVIDER_TEST_SET_ARCHIVE_FIXTURE}"
    )
    # The integration root is staged by three packaging targets; name each one.
    foreach(_hkp_set hkp_packaging_integration hkp_packaging_integration_shared
                     hkp_packaging_archive_fixture)
        if(TARGET ${_hkp_set})
            add_dependencies(hip_kernel_provider_integration_tests ${_hkp_set})
        endif()
    endforeach()

    # Parent of the integration descriptor sets: the walk covers every subtree staged
    # under it, so additional packs there need no edit here.
    hkp_verify_embedded_sources(
        TARGET hip_kernel_provider_integration_tests
        STAGED_DESCRIPTOR_ROOTS ${HIPKERNELPROVIDER_INTEGRATION_BUILD_DIR}
        PACK_NAMES integration integration_shared archive_fixture
    )
endif()

clang_tidy_check(hip_kernel_provider_integration_tests)

# ENVIRONMENT is passed to add_integration_test_target() rather than applied afterward
# with set_tests_properties(): under YAML-driven CTest categorization
# (DNN_PROVIDER_CTEST_CATEGORIES_YAML) only the tiered suites from
# apply_test_category_labels() in ../CMakeLists.txt are registered, and
# set_tests_properties() on an unregistered name is a hard configure error.
# add_integration_test_target() publishes the merged environment as
# hip_kernel_provider_integration_tests_TEST_ENVIRONMENT for that later call.
#
# No descriptor entry: the engine resolves its descriptor tree relative to the loaded
# module, so naming one would pin the build path into the install-time CTest file.
add_integration_test_target(hip_kernel_provider_integration_tests ${CMAKE_CURRENT_BINARY_DIR}
    LABELS slow
    ENVIRONMENT ${HIP_KERNEL_PROVIDER_ASM_KERNEL_ENV})

# Forward the merged raw-test environment to the parent scope for the tiered CTest suites
# created in ../CMakeLists.txt.
if(DEFINED hip_kernel_provider_integration_tests_TEST_ENVIRONMENT)
    set(hip_kernel_provider_integration_tests_TEST_ENVIRONMENT
        "${hip_kernel_provider_integration_tests_TEST_ENVIRONMENT}" PARENT_SCOPE)
endif()

# Cross-provider ASM SDPA gpu-reference integration test target. Standalone provider
# builds may not have the harness in scope, so import it first.
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
