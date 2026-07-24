# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Capture the directory of THIS file at include time. CMAKE_CURRENT_LIST_DIR
# inside the function below would resolve to the *caller's* directory, not
# this module's, so the shared flag file would not be found.
set(_HIPDNN_FLATBUFFERS_GENERATE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# FlatBuffers header generation.
#
# Provides hipdnn_generate_flatbuffer_headers() which generates C++ headers from .fbs schema
# files by invoking the active FlatBuffers dependency's flatc compiler directly. Some installed
# FlatBuffers packages do not export upstream's build_flatbuffers() helper.
# Usage:
#   hipdnn_generate_flatbuffer_headers(
#       TARGET            hipdnn_flatbuffers_sdk
#       SCHEMAS           schemas/data_types.fbs schemas/graph.fbs ...
#       SCHEMAS_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/schemas
#       GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/generated
#       OUTPUT_NAMESPACE  hipdnn_flatbuffers_sdk   # optional, defaults to hipdnn_flatbuffers_sdk
#   )

# Resolve the flatc compiler command and build dependency for header generation.
function(_hipdnn_resolve_flatc_command OUT_COMMAND OUT_DEPENDENCY)
    if(TARGET flatbuffers::flatc)
        set(${OUT_COMMAND} "$<TARGET_FILE:flatbuffers::flatc>" PARENT_SCOPE)
        set(${OUT_DEPENDENCY} flatbuffers::flatc PARENT_SCOPE)
        return()
    endif()

    if(TARGET flatc)
        set(${OUT_COMMAND} "$<TARGET_FILE:flatc>" PARENT_SCOPE)
        set(${OUT_DEPENDENCY} flatc PARENT_SCOPE)
        return()
    endif()

    foreach(_candidate_var
            IN ITEMS
               FLATBUFFERS_FLATC_EXECUTABLE
               FlatBuffers_FLATC_EXECUTABLE
               flatbuffers_FLATC_EXECUTABLE
    )
        if(DEFINED ${_candidate_var} AND NOT "${${_candidate_var}}" STREQUAL "")
            set(${OUT_COMMAND} "${${_candidate_var}}" PARENT_SCOPE)
            set(${OUT_DEPENDENCY} "" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(_flatc_hints "")
    if(DEFINED flatbuffers_DIR)
        get_filename_component(_flatbuffers_prefix "${flatbuffers_DIR}/../../.." ABSOLUTE)
        list(APPEND _flatc_hints "${_flatbuffers_prefix}/bin")
    endif()

    find_program(_hipdnn_flatc NAMES flatc HINTS ${_flatc_hints} NO_CACHE)
    if(NOT _hipdnn_flatc)
        message(FATAL_ERROR
            "hipdnn_generate_flatbuffer_headers: could not find the FlatBuffers compiler "
            "(flatc). Install flatc alongside the FlatBuffers package or set "
            "FLATBUFFERS_FLATC_EXECUTABLE."
        )
    endif()

    set(${OUT_COMMAND} "${_hipdnn_flatc}" PARENT_SCOPE)
    set(${OUT_DEPENDENCY} "" PARENT_SCOPE)
endfunction()

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

    _hipdnn_resolve_flatc_command(_flatc_command _flatc_dependency)

    if(TARGET flatc)
        set_target_properties(flatc PROPERTIES COMPILE_FLAGS "-w")
    endif()

    set(_generated_headers "")
    foreach(_schema IN LISTS ARG_SCHEMAS)
        # Match upstream build_flatbuffers(): relative schema paths are resolved against
        # the caller's source directory. SCHEMAS_DIR is only passed to flatc as an include
        # directory for schema imports.
        if(IS_ABSOLUTE "${_schema}")
            set(_schema_path "${_schema}")
        else()
            set(_schema_path "${CMAKE_CURRENT_SOURCE_DIR}/${_schema}")
        endif()

        get_filename_component(_schema_name "${_schema_path}" NAME_WE)
        set(_generated_header "${_output_dir}/${_schema_name}_generated.h")
        set(_flatc_depends "${_schema_path}" "${_flatc_flags_file}")
        if(_flatc_dependency)
            list(PREPEND _flatc_depends ${_flatc_dependency})
        endif()

        add_custom_command(
            OUTPUT "${_generated_header}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_output_dir}"
            COMMAND "${_flatc_command}" --cpp ${_flatc_extra_flags}
                    -o "${_output_dir}"
                    -I "${ARG_SCHEMAS_DIR}"
                    "${_schema_path}"
            DEPENDS ${_flatc_depends}
            COMMENT "Generating ${_schema_name}_generated.h"
            VERBATIM
        )
        list(APPEND _generated_headers "${_generated_header}")
    endforeach()

    add_custom_target(
        ${_gen_target_name}
        DEPENDS ${_generated_headers}
        COMMENT "Generating ${ARG_OUTPUT_NAMESPACE} FlatBuffers headers"
    )
    set_property(TARGET ${_gen_target_name} PROPERTY GENERATED_INCLUDES_DIR ${_output_dir})

    add_dependencies(${ARG_TARGET} ${_gen_target_name})
endfunction()

# Generate a FlatBuffers binary reflection schema (.bfbs) from one .fbs schema.
# Unlike the C++ header path (flatc --cpp), the reflection schema retains custom
# field attributes (umd_input_tensor / umd_output_tensor / umd_name), which the
# umd_registry_gen tool reads via the FlatBuffers reflection API to emit the UMD
# op-schema registry (RFC 0018 Appendix B). flatc writes <schema-name>.bfbs into
# the -o directory; OUTPUT MUST name that file.
# Usage:
#   hipdnn_generate_bfbs(
#       SCHEMA      schemas/graph.fbs
#       SCHEMAS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/schemas
#       OUTPUT      ${CMAKE_CURRENT_BINARY_DIR}/umd/graph.bfbs
#   )
function(hipdnn_generate_bfbs)
    set(_one_value_args SCHEMA SCHEMAS_DIR OUTPUT)
    cmake_parse_arguments(ARG "" "${_one_value_args}" "" ${ARGN})

    foreach(_required SCHEMA SCHEMAS_DIR OUTPUT)
        if(NOT ARG_${_required})
            message(FATAL_ERROR "hipdnn_generate_bfbs: missing required argument ${_required}")
        endif()
    endforeach()

    if(IS_ABSOLUTE "${ARG_SCHEMA}")
        set(_schema_path "${ARG_SCHEMA}")
    else()
        set(_schema_path "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SCHEMA}")
    endif()

    get_filename_component(_out_dir "${ARG_OUTPUT}" DIRECTORY)

    _hipdnn_resolve_flatc_command(_flatc_command _flatc_dependency)

    # graph.fbs pulls in the other schemas via `include`; flatc reads them all, so
    # the .bfbs must regenerate when ANY schema in the include dir changes (a
    # DEPENDS on graph.fbs alone would miss edits to included tables/attributes).
    # CONFIGURE_DEPENDS re-runs the glob when a schema is added or removed.
    file(GLOB _schema_deps CONFIGURE_DEPENDS "${ARG_SCHEMAS_DIR}/*.fbs")
    set(_depends "${_schema_path}" ${_schema_deps})
    if(_flatc_dependency)
        list(PREPEND _depends ${_flatc_dependency})
    endif()

    add_custom_command(
        OUTPUT "${ARG_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_out_dir}"
        COMMAND "${_flatc_command}" -b --schema
                -o "${_out_dir}"
                -I "${ARG_SCHEMAS_DIR}"
                "${_schema_path}"
        DEPENDS ${_depends}
        COMMENT "Generating reflection schema ${ARG_OUTPUT}"
        VERBATIM
    )
endfunction()
