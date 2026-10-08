# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The kernel archive tests exercise the engine's kpack layer, which is built whether or
# not hipDNN has SDPA, so they build either way. TheRock builds hipDNN with SDPA off, and
# there these are what check that the device's archive shipped (ALMIOPEN-2767).
target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmKernelPath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmKpackArchive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmKpackLoading.cpp
)

# Paths to the kpack archives for real loading tests: the build-tree copies, and the
# install-relative layout used when the build tree is absent, as in TheRock CI.
target_compile_definitions(hip_kernel_provider_tests PRIVATE
    ASM_KPACK_TEST_DIR="${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/arch_content/asm_sdpa"
    ASM_KPACK_INSTALL_DIR="${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/arch_content/asm_sdpa"
)

# Ensure kpack archives are built before tests that load them
if(TARGET asm_kernel_kpack)
    add_dependencies(hip_kernel_provider_tests asm_kernel_kpack)
endif()

if(NOT HIPDNN_ENABLE_SDPA)
    message(WARNING
            "ENABLE_ASM_SDPA_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
            "Skipping SDPA unit tests. Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable them.")
    return()
endif()

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ConfigHelpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAsmSdpaEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaBwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaBwdPlanGridMath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdPlanGridMath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdKernelArgs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaBwdKernelArgs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaModuleCache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSdpaFwdCase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AsmSdpaConfigHelpers.cpp
)

# Add include path for codegen-generated headers (e.g., asm_fmha_v3_fwd_configs.hpp)
target_include_directories(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/../../../engines/asm_sdpa_engine/generated
)

# Ensure codegen runs before compiling the tests
add_dependencies(hip_kernel_provider_tests asm_kernel_codegen)
