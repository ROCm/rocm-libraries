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
# The forward selector revision is a configure-time digest (AsmSdpaSelectorRevision.cmake),
# so the only place its recipe can be exercised against trees other than this checkout is
# configure time. Two properties no single build can show are computed here and asserted by
# TestAsmSdpaSelectorRevision:
#
#   * line endings: the engine tree written once with LF and once with CRLF must report one
#     revision, and it must be the one this build compiled in -- a Windows and a Linux
#     checkout of one commit that disagree refuse each other's models;
#   * coverage: each forward selection input, changed, must move it; a backward-only or
#     documentation change must not.
#
# The tree is this engine's sources plus its forward kernels. Backward kernels are left
# out (their bytes cannot matter if the recipe is right, and they are most of the size);
# a stand-in is written for the probe that says one cannot move the value.
set(_hkp_asm_engine_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../engines/asm_sdpa_engine")
cmake_path(NORMAL_PATH _hkp_asm_engine_dir)
include("${_hkp_asm_engine_dir}/AsmSdpaSelectorRevision.cmake")

# cmake-lint: disable=E1126
# cmake-lint carries no form spec for file(COPY_FILE) or file(CONFIGURE) and reports them
# as invalid discriminators. Both are valid CMake (3.21 and 3.18); the floor here is 3.25.
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
        # Each tree gets exactly one style, whatever this checkout has. file(WRITE) cannot
        # do that: on Windows it writes text mode, turning every \n into \r\n. file(CONFIGURE)
        # takes NEWLINE_STYLE; every `@` is routed through @_hkp_at@ so no source text is
        # substituted. (A `#cmakedefine` line would still be rewritten -- none exists, and
        # one would fail CrlfAndLfCheckoutsReportTheBuiltRevision, not pass silently.)
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
# One probe: apply <change> to <rel> in the LF tree, take the revision, undo it, and record
# a table row asserting whether the revision had to move. EDIT appends a line to an
# existing text file; ADD creates <rel>. The undo restores the exact bytes from a copy kept
# outside the tree (file(WRITE) would rewrite the line endings on Windows).
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
        "${_hkp_asm_probe_rows}    {\"${_label}\", \"${_probed}\", ${_must_expire}},\n"
        PARENT_SCOPE)
endfunction()
# Forward selection inputs: each decides which forward kernel runs or what it is given.
_hkp_asm_revision_probe(EDIT asm/asm_kernels/codegen.py true)
_hkp_asm_revision_probe(EDIT asm/asm_kernels/gfx942/fmha_v3_fwd/fmha_fwd.csv true)
_hkp_asm_revision_probe(EDIT plans/SdpaPlanUtils.hpp true)
_hkp_asm_revision_probe(EDIT plans/SdpaFwdPlanBuilder.cpp true)
_hkp_asm_revision_probe(EDIT plans/SdpaFwdPlan.cpp true)
_hkp_asm_revision_probe(EDIT plans/SdpaFwdArgsBuilder.hpp true)
_hkp_asm_revision_probe(EDIT asm/SdpaFwdKernelArgs.hpp true)
# A forward kernel added, and an arch directory codegen would read, are inventory changes.
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
    "namespace asm_sdpa_engine::selector_revision_fixtures\n{\n\n"
    "inline constexpr const char* REVISION_LF = \"${_hkp_asm_rev_lf}\";\n"
    "inline constexpr const char* REVISION_CRLF = \"${_hkp_asm_rev_crlf}\";\n\n"
    "struct Probe\n{\n    const char* change;\n    const char* revision;\n    bool mustExpire;\n};\n\n"
    "inline constexpr Probe PROBES[] = {\n${_hkp_asm_probe_rows}};\n\n"
    "} // namespace asm_sdpa_engine::selector_revision_fixtures\n")
# Rewritten only when its content changes, so a reconfigure that changes nothing
# recompiles nothing.
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/AsmSdpaSelectorRevisionFixtures.hpp"
     CONTENT "${_hkp_asm_fixture_text}" @ONLY)
target_include_directories(hip_kernel_provider_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
