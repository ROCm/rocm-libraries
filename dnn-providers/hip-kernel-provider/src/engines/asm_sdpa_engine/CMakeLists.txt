# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/AsmSdpaEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/SdpaBwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/SdpaFwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/SdpaFwdPlan.cpp
)

target_include_directories(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENGINE_ASM_SDPA=1)
target_compile_definitions(hip_kernel_provider_impl PRIVATE
    AITER_ASM_DIR="${CMAKE_INSTALL_PREFIX}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/asm_kernels"
)

# Copy ASM kernel binaries to build directory for local testing
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels/
     DESTINATION ${CMAKE_BINARY_DIR}/hip_kernel_provider/asm_kernels
     FILES_MATCHING PATTERN "*.co"
)
