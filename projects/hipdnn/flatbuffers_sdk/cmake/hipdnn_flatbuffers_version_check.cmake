# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared FlatBuffers version validation, used by both the in-tree
# flatbuffers_sdk/CMakeLists.txt at build time and by the installed imported
# Config (hipdnn_flatbuffers_sdkConfig_imported.cmake.in) at consumer
# find_package() time. Single-sourcing the comparison and core message keeps
# the two call sites from drifting.
#
# Usage:
#   hipdnn_check_flatbuffers_version(
#       EXPECTED  <version>           # the version hipDNN was/is built against
#       CONTEXT   <human-readable>    # "hipDNN" or "hipDNN Flatbuffers SDK"
#       [DETAIL   <suffix>]           # optional extra remediation text
#   )
#
# Reads the ambient flatbuffers_VERSION / flatbuffers_FOUND set by the most
# recent find_package(flatbuffers). Emits FATAL_ERROR on mismatch, a WARNING
# when the package config didn't populate flatbuffers_VERSION, and is silent
# otherwise (including the FetchContent path where flatbuffers_FOUND is unset).

if(DEFINED _HIPDNN_FLATBUFFERS_VERSION_CHECK_INCLUDED)
    return()
endif()
set(_HIPDNN_FLATBUFFERS_VERSION_CHECK_INCLUDED TRUE)

# Validate that the ambient flatbuffers package matches the expected
# version. Fatal on mismatch, warning when find_package succeeded but
# omitted the version, silent when no flatbuffers state is present
# (FetchContent path).
function(hipdnn_check_flatbuffers_version)
    set(_options "")
    set(_one_value_args EXPECTED CONTEXT DETAIL)
    set(_multi_value_args "")
    cmake_parse_arguments(_ARG "${_options}" "${_one_value_args}" "${_multi_value_args}" ${ARGN})

    if(NOT _ARG_EXPECTED OR NOT _ARG_CONTEXT)
        message(FATAL_ERROR
            "hipdnn_check_flatbuffers_version: EXPECTED and CONTEXT are required."
        )
    endif()

    if(DEFINED flatbuffers_VERSION)
        if(NOT flatbuffers_VERSION VERSION_EQUAL "${_ARG_EXPECTED}")
            set(_msg "${_ARG_CONTEXT}: FlatBuffers version mismatch — expected ${_ARG_EXPECTED} but found ${flatbuffers_VERSION}")
            if(DEFINED flatbuffers_DIR)
                string(APPEND _msg " (via ${flatbuffers_DIR})")
            endif()
            string(APPEND _msg ".")
            if(_ARG_DETAIL)
                string(APPEND _msg " ${_ARG_DETAIL}")
            endif()
            message(FATAL_ERROR "${_msg}")
        endif()
    elseif(flatbuffers_FOUND)
        # find_package succeeded but the config didn't populate
        # flatbuffers_VERSION (some vendored or repackaged distributions skip
        # it). Warn rather than fail-close — a real ABI mismatch would still
        # surface as a static_assert in the FlatBuffers headers later — but
        # flag it so users can investigate.
        message(WARNING
            "${_ARG_CONTEXT}: could not validate FlatBuffers version. "
            "find_package(flatbuffers) succeeded but did not set "
            "flatbuffers_VERSION; expected ${_ARG_EXPECTED}. A different "
            "version may fail later at compile time."
        )
    endif()
    # If flatbuffers_FOUND is also unset, FetchContent built FlatBuffers from
    # the pinned git tag — the version is guaranteed by construction.
endfunction()
