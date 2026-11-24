# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ============================================================================
# ASM SDPA selector revision fixtures, run at build time
# ============================================================================
#   cmake -DENGINE_DIR=<engine source dir> -DWORK_DIR=<scratch dir> -DOUTPUT=<header>
#         -P AsmSdpaSelectorRevisionFixtures.cmake
#
# Writes the header TestAsmSdpaSelectorRevision reads: the revision of LF and CRLF copies of
# the engine tree, the revision after each probe edit, and the engine files the recipe
# neither hashes nor lists as left out. The probes are driven by the recipe's own lists
# (hkp_asm_sdpa_sources, hkp_asm_sdpa_not_inputs, hkp_asm_sdpa_kernel_files), so they cannot
# cover less than the recipe hashes. Run by a build step whose dependencies are the engine's
# files, not at configure time: it copies the tree twice and runs the recipe once per probe,
# each run hashing every kernel.

# A -P script starts with every policy unset; without this CMP0053 is OLD, the @ escape
# below does nothing, and every run warns.
cmake_minimum_required(VERSION 3.25.2)

foreach(_required ENGINE_DIR WORK_DIR OUTPUT)
    if(NOT ${_required})
        message(FATAL_ERROR "AsmSdpaSelectorRevisionFixtures: -D${_required}=... is required.")
    endif()
endforeach()
include("${ENGINE_DIR}/AsmSdpaSelectorRevision.cmake")

# cmake-lint: disable=E1126
# cmake-lint has no form spec for file(COPY_FILE) or file(CONFIGURE); both are valid CMake.
set(_lf "${WORK_DIR}/lf")
set(_crlf "${WORK_DIR}/crlf")
file(REMOVE_RECURSE "${WORK_DIR}")
set(_hkp_at "@")
file(GLOB_RECURSE _files RELATIVE "${ENGINE_DIR}" "${ENGINE_DIR}/*")
foreach(_rel IN LISTS _files)
    if(_rel MATCHES "^descriptors/")
        continue()
    endif()
    if(_rel MATCHES "\\.co$")
        foreach(_tree IN ITEMS "${_lf}" "${_crlf}")
            cmake_path(GET _rel PARENT_PATH _rel_dir)
            file(MAKE_DIRECTORY "${_tree}/${_rel_dir}")
            file(COPY_FILE "${ENGINE_DIR}/${_rel}" "${_tree}/${_rel}")
        endforeach()
    else()
        # file(WRITE) writes CRLF on Windows; file(CONFIGURE) honors NEWLINE_STYLE. Every
        # `@` is routed through @_hkp_at@ so no source text is substituted.
        file(READ "${ENGINE_DIR}/${_rel}" _text)
        string(REPLACE "@" "@_hkp_at@" _text "${_text}")
        file(CONFIGURE OUTPUT "${_lf}/${_rel}" CONTENT "${_text}" @ONLY NEWLINE_STYLE LF)
        file(CONFIGURE OUTPUT "${_crlf}/${_rel}" CONTENT "${_text}" @ONLY NEWLINE_STYLE CRLF)
    endif()
endforeach()
hkp_asm_sdpa_revision(_rev_lf "${_lf}")
hkp_asm_sdpa_revision(_rev_crlf "${_crlf}")
hkp_asm_sdpa_revision_prefix(_rev_prefix)
# Every file of the engine tree is a decided input or a decided non-input; a new file that
# is neither would otherwise be left out of the revision without anyone choosing it.
hkp_asm_sdpa_unclassified_files(_unclassified "${ENGINE_DIR}")
list(JOIN _unclassified ", " _unclassified)

set(_probe_rows "")
# Applies <change> to <rel> in the LF tree, records the revision, and restores the file.
# EDIT appends a line (a byte to a code object); ADD creates <rel>; MOVE renames <rel> to
# <moved-to>. Restored from a copy because file(WRITE) would rewrite line endings on Windows.
function(_probe _change _rel _must_expire)
    set(_path "${_lf}/${_rel}")
    set(_saved "${WORK_DIR}/probe_original")
    set(_label "${_rel}")
    if(_change STREQUAL "EDIT")
        file(COPY_FILE "${_path}" "${_saved}")
        if(_rel MATCHES "\\.co$")
            file(APPEND "${_path}" "x")
        else()
            file(APPEND "${_path}" "probe\n")
        endif()
    elseif(_change STREQUAL "MOVE")
        set(_moved "${_lf}/${ARGV3}")
        cmake_path(GET _moved PARENT_PATH _dir)
        file(MAKE_DIRECTORY "${_dir}")
        file(RENAME "${_path}" "${_moved}")
        set(_label "${_rel} to ${ARGV3}")
    else()
        cmake_path(GET _path PARENT_PATH _dir)
        file(MAKE_DIRECTORY "${_dir}")
        file(WRITE "${_path}" "probe\n")
    endif()
    hkp_asm_sdpa_revision(_probed "${_lf}")
    if(_change STREQUAL "EDIT")
        file(RENAME "${_saved}" "${_path}")
    elseif(_change STREQUAL "MOVE")
        file(RENAME "${_moved}" "${_path}")
    else()
        file(REMOVE "${_path}")
    endif()
    string(TOLOWER "${_change}" _change_label)
    set(_probe_rows "${_probe_rows}    Probe{\"${_change_label} ${_label}\", \"${_probed}\", ${_must_expire}},\n"
        PARENT_SCOPE)
endfunction()
# Every selection source of either direction.
hkp_asm_sdpa_sources(_sources)
foreach(_source IN LISTS _sources)
    _probe(EDIT ${_source} true)
endforeach()
# The first CSV and code object of each direction: the bytes of a kernel, or the table of
# them, and the arch directory they sit in, are selection inputs.
hkp_asm_sdpa_kernel_files(_kernel_files "${_lf}")
foreach(_direction IN ITEMS fwd bwd)
    foreach(_extension IN ITEMS csv co)
        set(_candidates "${_kernel_files}")
        list(FILTER _candidates INCLUDE REGEX "/fmha_v3_${_direction}/.*\\.${_extension}$")
        list(GET _candidates 0 _kernel)
        file(RELATIVE_PATH _kernel_rel "${_lf}" "${_kernel}")
        string(REGEX REPLACE "^asm/asm_kernels/[^/]+/" "asm/asm_kernels/gfx_probe/"
               _kernel_moved "${_kernel_rel}")
        _probe(EDIT ${_kernel_rel} true)
        _probe(MOVE ${_kernel_rel} true ${_kernel_moved})
    endforeach()
endforeach()
# A new kernel or arch directory, in either direction, is an inventory change.
_probe(ADD asm/asm_kernels/gfx950/fmha_v3_fwd/fwd_probe.co true)
_probe(ADD asm/asm_kernels/gfx_probe/fmha_v3_fwd/fmha_fwd.csv true)
_probe(ADD asm/asm_kernels/gfx942/fmha_v3_bwd/bwd_probe.co true)
_probe(ADD asm/asm_kernels/gfx_probe/fmha_v3_bwd/fmha_bwd.csv true)
# Provenance and docs cannot move a throughput number.
_probe(EDIT asm/asm_kernels/gfx942/fmha_v3_fwd/SOURCE.md false)
_probe(EDIT asm/asm_kernels/gfx942/fmha_v3_bwd/SOURCE.md false)
# Everything else in the engine tree is declared not to decide a kernel or its launch.
hkp_asm_sdpa_not_inputs(_not_inputs)
foreach(_not_input IN LISTS _not_inputs)
    _probe(EDIT ${_not_input} false)
endforeach()
file(REMOVE_RECURSE "${WORK_DIR}")

string(CONCAT _text
    "// Generated by AsmSdpaSelectorRevisionFixtures.cmake. Do not edit.\n"
    "#pragma once\n\n"
    "#include <array>\n\n"
    "namespace asm_sdpa_engine::selector_revision_fixtures\n{\n\n"
    "inline constexpr const char* REVISION_PREFIX = \"${_rev_prefix}\";\n"
    "inline constexpr const char* REVISION_LF = \"${_rev_lf}\";\n"
    "inline constexpr const char* REVISION_CRLF = \"${_rev_crlf}\";\n"
    "// Engine files the recipe neither hashes nor lists as left out.\n"
    "inline constexpr const char* UNCLASSIFIED_FILES = \"${_unclassified}\";\n\n"
    "struct Probe\n{\n    const char* change;\n    const char* revision;\n    bool mustExpire;\n};\n\n"
    "inline constexpr std::array PROBES{\n${_probe_rows}};\n\n"
    "} // namespace asm_sdpa_engine::selector_revision_fixtures\n")
# Rewritten only when content changes, so an engine edit that moves no revision recompiles
# nothing.
set(_previous "")
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" _previous)
endif()
if(NOT _previous STREQUAL _text)
    file(WRITE "${OUTPUT}" "${_text}")
endif()
