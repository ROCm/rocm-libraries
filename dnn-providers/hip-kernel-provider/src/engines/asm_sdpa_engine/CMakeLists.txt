# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ============================================================================
# ASM Kernel Code Generation
# ============================================================================
# Find Python interpreter for code generation
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Define paths for code generation
set(ASM_CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels/codegen.py")
set(ASM_CODEGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ASM_FMHA_FWD_CONFIG_HEADER "${ASM_CODEGEN_OUTPUT_DIR}/asm_fmha_v3_fwd_configs.hpp")

# Define supported GPU architectures (used for both glob patterns and AITER_GPU_ARCHS)
set(ASM_SDPA_GPU_ARCHS gfx942 gfx950)

# CSV files that the codegen script depends on (iterate over all architectures)
set(ASM_KERNEL_CSV_FILES "")
foreach(ARCH ${ASM_SDPA_GPU_ARCHS})
    file(GLOB_RECURSE ARCH_CSV_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels/${ARCH}/**/*.csv"
    )
    list(APPEND ASM_KERNEL_CSV_FILES ${ARCH_CSV_FILES})
endforeach()

# Create the output directory
file(MAKE_DIRECTORY ${ASM_CODEGEN_OUTPUT_DIR})

# Custom command to generate ASM kernel config headers
add_custom_command(
    OUTPUT ${ASM_FMHA_FWD_CONFIG_HEADER}
    COMMAND ${CMAKE_COMMAND} -E env "AITER_GPU_ARCHS=${ASM_SDPA_GPU_ARCHS}" --
            ${Python3_EXECUTABLE} ${ASM_CODEGEN_SCRIPT}
            -m fmha_v3_fwd
            -o ${ASM_CODEGEN_OUTPUT_DIR}
    DEPENDS ${ASM_CODEGEN_SCRIPT} ${ASM_KERNEL_CSV_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels
    COMMENT "Generating ASM kernel configuration headers for ${ASM_SDPA_GPU_ARCHS}"
    VERBATIM
)

# Create a custom target for the generated headers
add_custom_target(asm_kernel_codegen
    DEPENDS ${ASM_FMHA_FWD_CONFIG_HEADER}
    COMMENT "Generating headers for asm sdpa kernels..."
)

# Ensure codegen runs before compiling the implementation
add_dependencies(hip_kernel_provider_impl asm_kernel_codegen)

# Add generated headers to include directories
target_include_directories(hip_kernel_provider_impl PRIVATE
    ${ASM_CODEGEN_OUTPUT_DIR}
)

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/AsmSdpaEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/SdpaBwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/SdpaFwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/SdpaBwdPlan.cpp
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
