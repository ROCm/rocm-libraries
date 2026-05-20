# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Capture the directory of THIS file at include time. CMAKE_CURRENT_LIST_DIR
# inside the function below would resolve to the *caller's* directory, not
# this module's, so the shared flag file would not be found.
set(_HIPDNN_FLATBUFFERS_GENERATE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# FlatBuffers header generation.
#
# Provides hipdnn_generate_flatbuffer_headers() which generates C++ headers from .fbs schema
# files using the active FlatBuffers dependency's build_flatbuffers().
#
# Usage:
#   hipdnn_generate_flatbuffer_headers(
#       TARGET            hipdnn_flatbuffers_sdk
#       SCHEMAS           schemas/data_types.fbs schemas/graph.fbs ...
#       SCHEMAS_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/schemas
#       GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/generated
#       OUTPUT_NAMESPACE  hipdnn_flatbuffers_sdk   # optional, defaults to hipdnn_flatbuffers_sdk
#   )

# Generate FlatBuffer C++ headers from .fbs schema files.
function(hipdnn_generate_flatbuffer_headers)
    set(_options "")
    set(_one_value_args TARGET SCHEMAS_DIR GENERATED_INCLUDE_DIR OUTPUT_NAMESPACE)
    set(_multi_value_args SCHEMAS)
    cmake_parse_arguments(ARG "${_options}" "${_one_value_args}" "${_multi_value_args}" ${ARGN})

    # Validate required arguments
    foreach(_required TARGET SCHEMAS SCHEMAS_DIR GENERATED_INCLUDE_DIR)
        if(NOT ARG_${_required})
            message(FATAL_ERROR "hipdnn_generate_flatbuffer_headers: missing required argument ${_required}")
        endif()
    endforeach()

    if(NOT ARG_OUTPUT_NAMESPACE)
        set(ARG_OUTPUT_NAMESPACE "hipdnn_flatbuffers_sdk")
    endif()

    # Single source of truth — the same flags are consumed by run_flatc.py.
    # See cmake/flatc_flags.txt for the format and comment rules.
    set(_flatc_flags_file "${_HIPDNN_FLATBUFFERS_GENERATE_DIR}/flatc_flags.txt")
    file(STRINGS "${_flatc_flags_file}" _flatc_flag_lines)
    set(_flatc_extra_flags "")
    foreach(_line IN LISTS _flatc_flag_lines)
        string(STRIP "${_line}" _stripped)
        if(_stripped STREQUAL "" OR _stripped MATCHES "^#")
            continue()
        endif()
        list(APPEND _flatc_extra_flags "${_stripped}")
    endforeach()

    set(_output_dir "${ARG_GENERATED_INCLUDE_DIR}")

    set(_gen_target_name "generate_${ARG_OUTPUT_NAMESPACE}_headers")

    # Use build_flatbuffers() from the active FlatBuffers dependency.
    _save_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

    set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS "${_flatc_extra_flags}")
    build_flatbuffers(
        "${ARG_SCHEMAS}" # flatbuffers_schemas
        "" # schema_include_dirs
        ${_gen_target_name} # custom_target_name
        "" # additional_dependencies
        ${_output_dir} # generated_includes_dir
        "" # binary_schemas_dir
        "" # copy_text_schemas_dir
    )

    if(TARGET flatc)
        set_target_properties(flatc PROPERTIES COMPILE_FLAGS "-w")
    endif()
    _restore_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

    add_dependencies(${ARG_TARGET} ${_gen_target_name})
endfunction()
