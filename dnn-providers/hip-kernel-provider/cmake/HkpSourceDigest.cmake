# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

# ============================================================================
# hkp_source_digest(<out-digest> <base-dir> [MANIFEST <file> ENGINE <name> PREFIX <prefix>]
#                   <file>...)
# ============================================================================
# 16 hex digits over <file>..., the selector revision of an engine that binds its L1
# models by declared id: a deployed model records it and the loader refuses a mismatch.
# Each file is keyed by its path relative to <base-dir> and hashed in sorted order, so the
# digest does not depend on GLOB order. Text is hashed with LF line endings, so CRLF and LF
# checkouts of one commit agree; a code object (.co) is bytes, which git never converts.
#
# MANIFEST writes the selector dump `uhd_gen restamp` reads: the engine's name, its full
# revision (<prefix><digest>), and each input's relative path and hash exactly as hashed
# here. Two dumps of one engine differ by the inputs that changed, which is what tells a
# restamp whether a recorded model can still be trusted.
function(hkp_source_digest _out_digest _base_dir)
    cmake_parse_arguments(PARSE_ARGV 2 _arg "" "MANIFEST;ENGINE;PREFIX" "")
    set(_inputs ${_arg_UNPARSED_ARGUMENTS})
    list(SORT _inputs)
    set(_digest "")
    set(_hashed "")
    foreach(_input IN LISTS _inputs)
        file(RELATIVE_PATH _name "${_base_dir}" "${_input}")
        if(_input MATCHES "\\.co$")
            file(SHA256 "${_input}" _one)
        else()
            # file(READ) drops CR only on Windows; the REPLACE covers CRLF files on Linux.
            file(READ "${_input}" _text)
            string(REPLACE "\r\n" "\n" _text "${_text}")
            string(SHA256 _one "${_text}")
        endif()
        string(APPEND _digest "${_name}:${_one}\n")
        list(APPEND _hashed "${_name}" "${_one}")
    endforeach()
    string(SHA256 _revision "${_digest}")
    string(SUBSTRING "${_revision}" 0 16 _revision)
    set(${_out_digest} "${_revision}" PARENT_SCOPE)

    if(_arg_MANIFEST)
        if(NOT _arg_ENGINE OR NOT _arg_PREFIX)
            message(FATAL_ERROR "hkp_source_digest: MANIFEST needs ENGINE and PREFIX.")
        endif()
        set(_inputs_json "{}")
        while(_hashed)
            list(POP_FRONT _hashed _name _one)
            string(JSON _inputs_json SET "${_inputs_json}" "${_name}" "\"${_one}\"")
        endwhile()
        set(_selector "{}")
        string(JSON _selector SET "${_selector}" selector "\"hkp-source-digest-v1\"")
        string(JSON _selector SET "${_selector}" inputs "${_inputs_json}")
        set(_dump "{}")
        string(JSON _dump SET "${_dump}" engine "\"${_arg_ENGINE}\"")
        string(JSON _dump SET "${_dump}" selector_revision "\"${_arg_PREFIX}${_revision}\"")
        string(JSON _dump SET "${_dump}" selector "${_selector}")
        file(WRITE "${_arg_MANIFEST}" "${_dump}\n")
    endif()
endfunction()
