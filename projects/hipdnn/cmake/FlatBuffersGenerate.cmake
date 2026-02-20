# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Multi-version FlatBuffers header generation.
#
# Provides hipdnn_generate_flatbuffer_headers() which generates C++ headers from .fbs schema
# files for every supported FlatBuffers version. The primary version (matching the active
# FlatBuffers dependency) uses build_flatbuffers(). Secondary versions are handled by fetching
# and building their own flatc via ExternalProject_Add, then invoking it with add_custom_command.
#
# Usage:
#   hipdnn_generate_flatbuffer_headers(
#       TARGET            hipdnn_data_sdk
#       SCHEMAS           schemas/data_types.fbs schemas/graph.fbs ...
#       SCHEMAS_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/schemas
#       PRIMARY_VERSION   ${HIPDNN_FLATBUFFERS_VERSION}
#       SUPPORTED_VERSIONS "24.12.23" "25.9.23"
#       GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/generated
#   )

include(ExternalProject)

# Generate FlatBuffer C++ headers for all supported versions from .fbs schema files.
function(hipdnn_generate_flatbuffer_headers)
    set(_options "")
    set(_one_value_args TARGET SCHEMAS_DIR PRIMARY_VERSION GENERATED_INCLUDE_DIR)
    set(_multi_value_args SCHEMAS SUPPORTED_VERSIONS)
    cmake_parse_arguments(ARG "${_options}" "${_one_value_args}" "${_multi_value_args}" ${ARGN})

    # Validate required arguments
    foreach(_required TARGET SCHEMAS SCHEMAS_DIR PRIMARY_VERSION SUPPORTED_VERSIONS GENERATED_INCLUDE_DIR)
        if(NOT ARG_${_required})
            message(FATAL_ERROR "hipdnn_generate_flatbuffer_headers: missing required argument ${_required}")
        endif()
    endforeach()

    # Common flatc code generation flags
    set(_flatc_flags --cpp --gen-object-api --gen-mutable --gen-compare --defaults-json --scoped-enums)

    # Compute primary version directory
    string(REPLACE "." "_" _primary_ver_tag "${ARG_PRIMARY_VERSION}")
    set(_primary_ver_dir "v${_primary_ver_tag}")
    set(_primary_output_dir
        "${ARG_GENERATED_INCLUDE_DIR}/${_primary_ver_dir}/hipdnn_data_sdk/data_objects"
    )

    # --- Primary version: use build_flatbuffers() from the active FlatBuffers dependency ---
    _save_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

    set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS
        "--gen-object-api;--gen-mutable;--gen-compare;--defaults-json;--scoped-enums"
    )
    build_flatbuffers(
        "${ARG_SCHEMAS}" # flatbuffers_schemas
        "" # schema_include_dirs
        generate_hipdnn_data_sdk_headers # custom_target_name
        "" # additional_dependencies
        ${_primary_output_dir} # generated_includes_dir
        "" # binary_schemas_dir
        "" # copy_text_schemas_dir
    )

    if(TARGET flatc)
        set_target_properties(flatc PROPERTIES COMPILE_FLAGS "-w")
    endif()
    _restore_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

    add_dependencies(${ARG_TARGET} generate_hipdnn_data_sdk_headers)

    # --- Secondary versions: ExternalProject + custom commands ---
    # For each supported version other than the primary, download and build that version's flatc
    # compiler in isolation (ExternalProject avoids target name collisions), then invoke it on
    # every schema file via add_custom_command.
    foreach(_version IN LISTS ARG_SUPPORTED_VERSIONS)
        if(_version STREQUAL ARG_PRIMARY_VERSION)
            continue()
        endif()

        string(REPLACE "." "_" _ver_tag "${_version}")
        set(_ver_dir "v${_ver_tag}")
        set(_ep_name "flatc_${_ver_tag}")
        set(_flatc_build_dir "${CMAKE_CURRENT_BINARY_DIR}/_flatc_builds/${_ver_dir}")
        set(_flatc_binary "${_flatc_build_dir}/flatc")
        set(_output_dir
            "${ARG_GENERATED_INCLUDE_DIR}/${_ver_dir}/hipdnn_data_sdk/data_objects"
        )

        ExternalProject_Add(${_ep_name}
            GIT_REPOSITORY https://github.com/google/flatbuffers.git
            GIT_TAG v${_version}
            BINARY_DIR ${_flatc_build_dir}
            CMAKE_ARGS
                -DFLATBUFFERS_BUILD_FLATC=ON
                -DFLATBUFFERS_BUILD_FLATLIB=OFF
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_FLATHASH=OFF
                -DFLATBUFFERS_ENABLE_PCH=ON
                -DCMAKE_BUILD_TYPE=Release
            BUILD_COMMAND ${CMAKE_COMMAND} --build ${_flatc_build_dir} --target flatc
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS ${_flatc_binary}
        )

        set(_output_files)
        foreach(_schema IN LISTS ARG_SCHEMAS)
            get_filename_component(_schema_name ${_schema} NAME_WE)
            set(_output_file "${_output_dir}/${_schema_name}_generated.h")
            list(APPEND _output_files ${_output_file})

            add_custom_command(
                OUTPUT ${_output_file}
                COMMAND ${_flatc_binary}
                    -I ${ARG_SCHEMAS_DIR}
                    ${_flatc_flags}
                    -o ${_output_dir}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${_schema}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_schema}
                COMMENT "flatc ${_version}: generating ${_schema_name}_generated.h"
            )
        endforeach()

        set(_gen_target "generate_hipdnn_data_sdk_headers_${_ver_dir}")
        add_custom_target(${_gen_target} DEPENDS ${_output_files}
            COMMENT "Generating FlatBuffer headers for version ${_version}"
        )
        add_dependencies(${_gen_target} ${_ep_name})
        add_dependencies(${ARG_TARGET} ${_gen_target})
    endforeach()
endfunction()
