# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

# ============================================================================
# hkp_source_digest(<out-digest> <base-dir> <file>...)
# ============================================================================
# 16 hex digits over <file>..., the selector revision of an engine that binds its L1
# models by declared id: a deployed model records it and the loader refuses a mismatch.
# Each file is keyed by its path relative to <base-dir> and hashed in sorted order, so the
# digest does not depend on GLOB order. Text is hashed with LF line endings, so CRLF and LF
# checkouts of one commit agree; a code object (.co) is bytes, which git never converts.
function(hkp_source_digest _out_digest _base_dir)
    set(_inputs ${ARGN})
    list(SORT _inputs)
    set(_digest "")
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
    endforeach()
    string(SHA256 _revision "${_digest}")
    string(SUBSTRING "${_revision}" 0 16 _revision)
    set(${_out_digest} "${_revision}" PARENT_SCOPE)
endfunction()
