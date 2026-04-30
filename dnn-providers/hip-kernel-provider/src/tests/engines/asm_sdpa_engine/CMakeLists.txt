# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmSdpaEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaBwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdKernelArgs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaBwdKernelArgs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmKernelPath.cpp
)

target_compile_definitions(hip_kernel_provider_tests PRIVATE
    AITER_ASM_DIR="/test/default/asm/dir"
)
