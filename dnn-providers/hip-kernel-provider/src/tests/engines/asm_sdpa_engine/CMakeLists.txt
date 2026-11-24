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
    ${CMAKE_CURRENT_SOURCE_DIR}/AsmSdpaFwdLaunchLimitRows.cpp
)

# Add include path for codegen-generated headers (e.g., asm_fmha_v3_fwd_configs.hpp)
target_include_directories(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/../../../engines/asm_sdpa_engine/generated
)

# Path to the build-tree kpack archives for real loading tests.
# These are unit tests (registered via add_unit_test_target) that run only in the
# build tree. Installed-layout / TheRock coverage is provided by the integration
# test suite, which resolves paths relative to the installed plugin.
target_compile_definitions(hip_kernel_provider_tests PRIVATE
    ASM_KPACK_TEST_DIR="${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/arch_content/asm_sdpa"
    ASM_KPACK_INSTALL_DIR="${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/arch_content/asm_sdpa"
)

# Ensure codegen runs before compiling the tests
add_dependencies(hip_kernel_provider_tests asm_kernel_codegen)

# Ensure kpack archives are built before tests that load them
if(TARGET asm_kernel_kpack)
    add_dependencies(hip_kernel_provider_tests asm_kernel_kpack)
endif()

# ============================================================================
# Selector revision fixtures
# ============================================================================
# The selector revision is a configure-time digest, so its recipe can only be run against
# other trees here. TestAsmSdpaSelectorRevision asserts that LF and CRLF copies of the
# engine tree agree with the built revision, and that each forward or backward input moves
# it while doc and engine-class changes do not. The fixtures copy the tree and run the
# recipe once per probe, so they are a build step that reruns only when an engine file, the
# recipe or the script changes, not work every configure repeats.
set(_hkp_asm_engine_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../engines/asm_sdpa_engine")
cmake_path(NORMAL_PATH _hkp_asm_engine_dir)
set(_hkp_asm_fixtures_script "${CMAKE_CURRENT_SOURCE_DIR}/AsmSdpaSelectorRevisionFixtures.cmake")
set(_hkp_asm_fixtures_header "${CMAKE_CURRENT_BINARY_DIR}/AsmSdpaSelectorRevisionFixtures.hpp")
set(_hkp_asm_fixtures_stamp "${CMAKE_CURRENT_BINARY_DIR}/AsmSdpaSelectorRevisionFixtures.stamp")
file(GLOB_RECURSE _hkp_asm_fixture_inputs CONFIGURE_DEPENDS "${_hkp_asm_engine_dir}/*")
list(FILTER _hkp_asm_fixture_inputs EXCLUDE REGEX "/descriptors/")
add_custom_command(
    OUTPUT "${_hkp_asm_fixtures_stamp}"
    BYPRODUCTS "${_hkp_asm_fixtures_header}"
    COMMAND "${CMAKE_COMMAND}" "-DENGINE_DIR=${_hkp_asm_engine_dir}"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/selector_revision"
            "-DOUTPUT=${_hkp_asm_fixtures_header}" -P "${_hkp_asm_fixtures_script}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_hkp_asm_fixtures_stamp}"
    DEPENDS "${_hkp_asm_fixtures_script}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/HkpSourceDigest.cmake"
            ${_hkp_asm_fixture_inputs}
    COMMENT "Computing ASM SDPA selector revision fixtures"
    VERBATIM)
add_custom_target(hkp_asm_selector_revision_fixtures
    DEPENDS "${_hkp_asm_fixtures_stamp}"
    COMMENT "ASM SDPA selector revision fixtures")
add_dependencies(hip_kernel_provider_tests hkp_asm_selector_revision_fixtures)
target_include_directories(hip_kernel_provider_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# ============================================================================
# Shipped L1 models
# ============================================================================
# The gate in TestAsmSdpaEngine.cpp checks the models this source tree ships. They are read
# here, at configure time, because the binary may run where no checkout exists (an installed
# or repackaged layout).
if(asm_sdpa IN_LIST HIPKERNELPROVIDER_DECLARED_MODEL_ENGINES)
    include(HkpShippedModels)
    hkp_write_shipped_models_header(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/AsmSdpaShippedModels.hpp"
        NAMESPACE asm_sdpa_engine::shipped_models
        SOURCE_DIR "${HIPKERNELPROVIDER_asm_sdpa_DESCRIPTOR_SOURCE_DIR}"
        FILES ${HIPKERNELPROVIDER_asm_sdpa_DESCRIPTOR_FILES})
endif()
