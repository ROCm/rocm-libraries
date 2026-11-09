# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Queries must also work in the parent directory of an add_subdirectory consumer.
set_property(GLOBAL PROPERTY ROCM_ARCH_PROCESSORS "${_ROCM_ARCH_PROCESSORS}")
set_property(GLOBAL PROPERTY ROCM_ARCH_GENERICS "${_ROCM_ARCH_GENERICS}")

# Return sorted registered names to output; KIND defaults to PROCESSOR.
function(rocm_arch_list output)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "KIND" "")
    if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "rocm_arch_list: expected KIND PROCESSOR or GENERIC")
    endif()
    if(NOT DEFINED ARG_KIND OR ARG_KIND STREQUAL "PROCESSOR")
        get_property(names GLOBAL PROPERTY ROCM_ARCH_PROCESSORS)
    elseif(ARG_KIND STREQUAL "GENERIC")
        get_property(names GLOBAL PROPERTY ROCM_ARCH_GENERICS)
    else()
        message(FATAL_ERROR "rocm_arch_list: unknown kind ${ARG_KIND}")
    endif()
    set(${output} "${names}" PARENT_SCOPE)
endfunction()

# Return whether ID exactly matches a registered processor or generic target.
function(rocm_arch_is_known output)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "ID" "")
    if(NOT DEFINED ARG_ID OR ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "rocm_arch_is_known: expected ID <target-name>")
    endif()
    get_property(processors GLOBAL PROPERTY ROCM_ARCH_PROCESSORS)
    get_property(generics GLOBAL PROPERTY ROCM_ARCH_GENERICS)
    if(ARG_ID IN_LIST processors OR ARG_ID IN_LIST generics)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()
