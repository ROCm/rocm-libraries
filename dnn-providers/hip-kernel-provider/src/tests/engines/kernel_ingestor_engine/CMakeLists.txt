# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHandleDeviceResolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestKernelIngestorEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddDispatchHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwisePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvFwdPack.cpp
)

# The code-object existence check resolves paths against the checkout rather than an
# install prefix, so it needs the in-tree kernel root the ASM engine ships.
if(ENABLE_ASM_SDPA_ENGINE)
    target_sources(hip_kernel_provider_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestAsmSdpaCodeObjects.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestAsmSdpaForwardMatchers.cpp
    )
    # This file is compiled into a target owned by a parent directory, where a
    # source-file property set here does not reach it.
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        ASM_KERNEL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../engines/asm_sdpa_engine/asm/asm_kernels"
    )
endif()
