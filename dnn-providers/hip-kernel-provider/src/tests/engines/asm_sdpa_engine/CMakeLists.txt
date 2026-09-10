# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ConfigHelpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmSdpaEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaBwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaBwdPlanGridMath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdPlanGridMath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdKernelArgs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaBwdKernelArgs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmKernelPath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmKpackArchive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmKpackLoading.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaModuleCache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdCase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AsmSdpaConfigHelpers.cpp
)

# Add include path for codegen-generated headers (e.g., asm_fmha_v3_fwd_configs.hpp)
target_include_directories(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/../../../engines/asm_sdpa_engine/generated
)

# Path to the build-tree kpack archives for real loading tests
target_compile_definitions(hip_kernel_provider_tests PRIVATE
    ASM_KPACK_TEST_DIR="${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/arch_content/asm_sdpa"
)

# Ensure codegen runs before compiling the tests
add_dependencies(hip_kernel_provider_tests asm_kernel_codegen)

# Ensure kpack archives are built before tests that load them
if(TARGET asm_kernel_kpack)
    add_dependencies(hip_kernel_provider_tests asm_kernel_kpack)
endif()
