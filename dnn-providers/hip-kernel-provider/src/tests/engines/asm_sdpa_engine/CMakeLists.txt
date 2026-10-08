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

target_compile_definitions(hip_kernel_provider_tests PRIVATE
    AITER_ASM_DIR="/test/default/asm/dir"
)

# ============================================================================
# Selector revision fixtures
# ============================================================================
# The selector revision is a configure-time digest, so its recipe can only be run against
# other trees here. TestAsmSdpaSelectorRevision asserts that LF and CRLF copies of the
# engine tree agree with the built revision, and that each forward input moves it while
# backward-only and doc changes do not. Backward kernels are left out of the copied tree.
set(_hkp_asm_engine_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../engines/asm_sdpa_engine")
cmake_path(NORMAL_PATH _hkp_asm_engine_dir)
include("${_hkp_asm_engine_dir}/AsmSdpaSelectorRevision.cmake")

# cmake-lint: disable=E1126
# cmake-lint has no form spec for file(COPY_FILE) or file(CONFIGURE); both are valid CMake.
set(_hkp_asm_lf "${CMAKE_CURRENT_BINARY_DIR}/selector_revision/lf")
set(_hkp_asm_crlf "${CMAKE_CURRENT_BINARY_DIR}/selector_revision/crlf")
file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/selector_revision")
set(_hkp_at "@")
file(GLOB_RECURSE _hkp_asm_files RELATIVE "${_hkp_asm_engine_dir}" "${_hkp_asm_engine_dir}/*")
foreach(_rel IN LISTS _hkp_asm_files)
    if(_rel MATCHES "^asm/asm_kernels/[^/]+/fmha_v3_bwd/" OR _rel MATCHES "^descriptors/")
        continue()
    endif()
    if(_rel MATCHES "\\.co$")
        foreach(_tree IN ITEMS "${_hkp_asm_lf}" "${_hkp_asm_crlf}")
            cmake_path(GET _rel PARENT_PATH _rel_dir)
            file(MAKE_DIRECTORY "${_tree}/${_rel_dir}")
            file(COPY_FILE "${_hkp_asm_engine_dir}/${_rel}" "${_tree}/${_rel}")
        endforeach()
    else()
        # file(WRITE) writes CRLF on Windows; file(CONFIGURE) honors NEWLINE_STYLE. Every
        # `@` is routed through @_hkp_at@ so no source text is substituted.
        file(READ "${_hkp_asm_engine_dir}/${_rel}" _text)
        string(REPLACE "@" "@_hkp_at@" _text "${_text}")
        file(CONFIGURE OUTPUT "${_hkp_asm_lf}/${_rel}" CONTENT "${_text}" @ONLY NEWLINE_STYLE LF)
        file(CONFIGURE OUTPUT "${_hkp_asm_crlf}/${_rel}" CONTENT "${_text}" @ONLY
             NEWLINE_STYLE CRLF)
    endif()
endforeach()
hkp_asm_sdpa_fwd_revision(_hkp_asm_rev_lf "${_hkp_asm_lf}")
hkp_asm_sdpa_fwd_revision(_hkp_asm_rev_crlf "${_hkp_asm_crlf}")

set(_hkp_asm_probe_rows "")
# Applies <change> to <rel> in the LF tree, records the revision, and restores the file.
# EDIT appends a line; ADD creates <rel>. Restored from a copy because file(WRITE) would
# rewrite line endings on Windows.
function(_hkp_asm_revision_probe _change _rel _must_expire)
    set(_path "${_hkp_asm_lf}/${_rel}")
    set(_saved "${CMAKE_CURRENT_BINARY_DIR}/selector_revision/probe_original")
    if(_change STREQUAL "EDIT")
        file(COPY_FILE "${_path}" "${_saved}")
        file(APPEND "${_path}" "probe\n")
    else()
        cmake_path(GET _path PARENT_PATH _dir)
        file(MAKE_DIRECTORY "${_dir}")
        file(WRITE "${_path}" "probe\n")
    endif()
    hkp_asm_sdpa_fwd_revision(_probed "${_hkp_asm_lf}")
    if(_change STREQUAL "EDIT")
        file(RENAME "${_saved}" "${_path}")
    else()
        file(REMOVE "${_path}")
    endif()
    string(TOLOWER "${_change}" _label)
    string(APPEND _label " ${_rel}")
    set(_hkp_asm_probe_rows
        "${_hkp_asm_probe_rows}    Probe{\"${_label}\", \"${_probed}\", ${_must_expire}},\n"
        PARENT_SCOPE)
endfunction()
# Forward selection inputs.
_hkp_asm_revision_probe(EDIT asm/asm_kernels/codegen.py true)
_hkp_asm_revision_probe(EDIT asm/asm_kernels/gfx942/fmha_v3_fwd/fmha_fwd.csv true)
_hkp_asm_revision_probe(EDIT plans/SdpaPlanUtils.hpp true)
_hkp_asm_revision_probe(EDIT plans/SdpaFwdPlanBuilder.cpp true)
_hkp_asm_revision_probe(EDIT plans/SdpaFwdPlan.cpp true)
_hkp_asm_revision_probe(EDIT plans/SdpaFwdArgsBuilder.hpp true)
_hkp_asm_revision_probe(EDIT asm/SdpaFwdKernelArgs.hpp true)
# A new forward kernel or arch directory is an inventory change.
_hkp_asm_revision_probe(ADD asm/asm_kernels/gfx950/fmha_v3_fwd/fwd_probe.co true)
_hkp_asm_revision_probe(ADD asm/asm_kernels/gfx_probe/fmha_v3_fwd/fmha_fwd.csv true)
# Backward-only and provenance changes cannot move a forward throughput number.
_hkp_asm_revision_probe(EDIT plans/SdpaBwdPlanBuilder.cpp false)
_hkp_asm_revision_probe(EDIT plans/SdpaBwdPlan.cpp false)
_hkp_asm_revision_probe(ADD asm/asm_kernels/gfx942/fmha_v3_bwd/bwd_probe.co false)
_hkp_asm_revision_probe(ADD asm/asm_kernels/gfx942/fmha_v3_bwd/fmha_bwd.csv false)
_hkp_asm_revision_probe(EDIT asm/asm_kernels/gfx942/fmha_v3_fwd/SOURCE.md false)
_hkp_asm_revision_probe(EDIT asm/asm_kernels/README.md false)

string(CONCAT _hkp_asm_fixture_text
    "// Generated by ${CMAKE_CURRENT_LIST_FILE}. Do not edit.\n"
    "#pragma once\n\n"
    "#include <array>\n\n"
    "namespace asm_sdpa_engine::selector_revision_fixtures\n{\n\n"
    "inline constexpr const char* REVISION_LF = \"${_hkp_asm_rev_lf}\";\n"
    "inline constexpr const char* REVISION_CRLF = \"${_hkp_asm_rev_crlf}\";\n\n"
    "struct Probe\n{\n    const char* change;\n    const char* revision;\n    bool mustExpire;\n};\n\n"
    "inline constexpr std::array PROBES{\n${_hkp_asm_probe_rows}};\n\n"
    "} // namespace asm_sdpa_engine::selector_revision_fixtures\n")
# Rewritten only when content changes, so a no-op reconfigure recompiles nothing.
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/AsmSdpaSelectorRevisionFixtures.hpp"
     CONTENT "${_hkp_asm_fixture_text}" @ONLY)
target_include_directories(hip_kernel_provider_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
