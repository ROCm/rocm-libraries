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
    Batchnorm/IntegrationGpuBatchnormForwardInference.cpp
    Batchnorm/IntegrationGpuBatchnormForwardInferenceWithVariance.cpp
    Batchnorm/IntegrationGpuBatchnormForwardInferenceAndActivation.cpp
    Batchnorm/IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivation.cpp
    Batchnorm/IntegrationGpuBatchnormForwardTraining.cpp
    RMSnorm/IntegrationGpuRMSnormForward.cpp
    Layernorm/IntegrationGpuLayernormForward.cpp
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

if(ENABLE_ASM_SDPA_ENGINE)
    if(NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "ENABLE_ASM_SDPA_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
                "Skipping SDPA integration tests. Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable them.")
    else()
        target_sources(hip_kernel_provider_integration_tests PRIVATE
            asm_sdpa_engine/IntegrationGpuSdpaForward.cpp
            asm_sdpa_engine/IntegrationGpuSdpaBackward.cpp
        )
        target_compile_definitions(hip_kernel_provider_integration_tests PRIVATE
            AITER_ASM_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../engines/asm_sdpa_engine/asm/asm_kernels"
        )
    endif()
endif()

clang_tidy_check(hip_kernel_provider_integration_tests)

add_integration_test_target(hip_kernel_provider_integration_tests ${CMAKE_CURRENT_BINARY_DIR}
    LABELS slow)

if(ENABLE_ASM_SDPA_ENGINE)
    if(NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "ENABLE_ASM_SDPA_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
                "Skipping SDPA integration test environment setup.")
    else()
        set(ASM_KERNEL_ENV "HIPDNN_AITER_ASM_DIR=${CMAKE_BINARY_DIR}/hip_kernel_provider/asm_kernels")
        if(COVERAGE_ENVIRONMENT)
            set_tests_properties(hip_kernel_provider_integration_tests PROPERTIES
                ENVIRONMENT "${COVERAGE_ENVIRONMENT};${ASM_KERNEL_ENV}"
            )
        else()
            set_tests_properties(hip_kernel_provider_integration_tests PROPERTIES
                ENVIRONMENT "${ASM_KERNEL_ENV}"
            )
        endif()
    endif()
endif()
