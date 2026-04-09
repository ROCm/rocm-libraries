# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(FetchContent)
include(ExternalProject)

# Multi-version FlatBuffers header generation.
#
# Provides hipdnn_generate_flatbuffer_headers() which generates C++ headers from .fbs schema
# files for every supported FlatBuffers version. The primary version (matching the active
# FlatBuffers dependency) uses build_flatbuffers(). Secondary versions are handled by
# downloading and building their own flatc at configure time, then invoking it with
# add_custom_command during the build.
#
# Usage:
#   hipdnn_generate_flatbuffer_headers(
#       TARGET            hipdnn_data_sdk
#       SCHEMAS           schemas/data_types.fbs schemas/graph.fbs ...
#       SCHEMAS_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/schemas
#       PRIMARY_VERSION   ${HIPDNN_FLATBUFFERS_VERSION}
#       SUPPORTED_VERSIONS "24.12.23" "25.9.23"
#       GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/generated
#       OUTPUT_NAMESPACE  hipdnn_data_sdk   # optional, defaults to hipdnn_data_sdk
#   )

# Helper function to generate headers for a secondary FlatBuffers version
# Uses parent scope variables: ARG_SCHEMAS, ARG_SCHEMAS_DIR, ARG_GENERATED_INCLUDE_DIR, ARG_TARGET
function(_hipdnn_generate_secondary_version _version _flatc_flags)
    string(REPLACE "." "_" _ver_tag "${_version}")
    set(_ver_dir "v${_ver_tag}")
    set(_fc_name "flatbuffers_${_ver_tag}")
    set(_ep_name "flatc_${_ver_tag}")
    set(_flatc_build_dir "${CMAKE_CURRENT_BINARY_DIR}/_flatc_builds/${_ver_dir}")
    set(_flatc_binary "${_flatc_build_dir}/flatc")
    set(_output_dir "${ARG_GENERATED_INCLUDE_DIR}/${_ver_dir}/${ARG_OUTPUT_NAMESPACE}/data_objects")

    # Download source at configure time via FetchContent (skipped if already populated)
    FetchContent_Declare(${_fc_name}
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG "v${_version}"
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(${_fc_name})
    if(NOT ${_fc_name}_POPULATED)
        message(STATUS "Downloading FlatBuffers v${_version} source...")
        FetchContent_Populate(${_fc_name}
            GIT_REPOSITORY https://github.com/google/flatbuffers.git
            GIT_TAG "v${_version}"
            GIT_SHALLOW TRUE
        )
    endif()

    # Convert compiler paths to CMake-style (forward slashes) for ExternalProject
    file(TO_CMAKE_PATH "${CMAKE_C_COMPILER}" _c_compiler)
    file(TO_CMAKE_PATH "${CMAKE_CXX_COMPILER}" _cxx_compiler)

    # Build flatc at build time via ExternalProject (separate CMake instance avoids
    # target name collisions). Guard prevents duplicate target when multiple SDKs
    # generate headers for the same FlatBuffers version.
    if(NOT TARGET ${_ep_name})
        ExternalProject_Add(${_ep_name}
            SOURCE_DIR "${${_fc_name}_SOURCE_DIR}"
            DOWNLOAD_COMMAND ""
            UPDATE_COMMAND ""
            CONFIGURE_HANDLED_BY_BUILD TRUE
            BINARY_DIR ${_flatc_build_dir}
            CMAKE_ARGS
                -DFLATBUFFERS_BUILD_FLATC=ON
                -DFLATBUFFERS_BUILD_FLATLIB=OFF
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DFLATBUFFERS_BUILD_FLATHASH=OFF
                -DFLATBUFFERS_ENABLE_PCH=ON
                -DCMAKE_C_COMPILER=${_c_compiler}
                -DCMAKE_CXX_COMPILER=${_cxx_compiler}
                -DCMAKE_RC_COMPILER=CMAKE_RC_COMPILER-NOTREQUIRED
                -DCMAKE_BUILD_TYPE=Release
            BUILD_COMMAND ${CMAKE_COMMAND} --build ${_flatc_build_dir} --target flatc
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS ${_flatc_binary}
            LOG_CONFIGURE ON
            LOG_BUILD ON
            LOG_OUTPUT_ON_FAILURE ON
        )
    else()
        # Target already created by another SDK — resolve the existing build directory
        ExternalProject_Get_Property(${_ep_name} BINARY_DIR)
        set(_flatc_build_dir "${BINARY_DIR}")
        set(_flatc_binary "${_flatc_build_dir}/flatc")
    endif()

    # Generate headers at build time using the built flatc
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
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_schema} ${_flatc_binary}
            COMMENT "flatc ${_version}: generating ${_schema_name}_generated.h"
        )
    endforeach()

    set(_gen_target "generate_${ARG_OUTPUT_NAMESPACE}_headers_${_ver_dir}")
    add_custom_target(${_gen_target} DEPENDS ${_output_files}
        COMMENT "Generating FlatBuffer headers for version ${_version}"
    )
    add_dependencies(${_gen_target} ${_ep_name})
    add_dependencies(${ARG_TARGET} ${_gen_target})
endfunction()

# Generate FlatBuffer C++ headers for all supported versions from .fbs schema files.
function(hipdnn_generate_flatbuffer_headers)
    set(_options "")
    set(_one_value_args TARGET SCHEMAS_DIR PRIMARY_VERSION GENERATED_INCLUDE_DIR OUTPUT_NAMESPACE)
    set(_multi_value_args SCHEMAS SUPPORTED_VERSIONS)
    cmake_parse_arguments(ARG "${_options}" "${_one_value_args}" "${_multi_value_args}" ${ARGN})

    # Validate required arguments
    foreach(_required TARGET SCHEMAS SCHEMAS_DIR PRIMARY_VERSION SUPPORTED_VERSIONS GENERATED_INCLUDE_DIR)
        if(NOT ARG_${_required})
            message(FATAL_ERROR "hipdnn_generate_flatbuffer_headers: missing required argument ${_required}")
        endif()
    endforeach()

    # Default OUTPUT_NAMESPACE to hipdnn_data_sdk for backward compatibility
    if(NOT ARG_OUTPUT_NAMESPACE)
        set(ARG_OUTPUT_NAMESPACE "hipdnn_data_sdk")
    endif()

    # Common extra flags (shared between primary and secondary generation)
    set(_flatc_extra_flags --gen-object-api --gen-mutable --gen-compare --defaults-json --scoped-enums)
    # Full flags for direct flatc invocation (add_custom_command includes --cpp explicitly)
    set(_flatc_flags --cpp ${_flatc_extra_flags})

    # Compute primary version directory
    string(REPLACE "." "_" _primary_ver_tag "${ARG_PRIMARY_VERSION}")
    set(_primary_ver_dir "v${_primary_ver_tag}")
    set(_primary_output_dir
        "${ARG_GENERATED_INCLUDE_DIR}/${_primary_ver_dir}/${ARG_OUTPUT_NAMESPACE}/data_objects"
    )

    set(_gen_target_name "generate_${ARG_OUTPUT_NAMESPACE}_headers")

    # --- Primary version: use build_flatbuffers() from the active FlatBuffers dependency ---
    _save_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

    set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS "${_flatc_extra_flags}")
    build_flatbuffers(
        "${ARG_SCHEMAS}" # flatbuffers_schemas
        "" # schema_include_dirs
        ${_gen_target_name} # custom_target_name
        "" # additional_dependencies
        ${_primary_output_dir} # generated_includes_dir
        "" # binary_schemas_dir
        "" # copy_text_schemas_dir
    )

    if(TARGET flatc)
        set_target_properties(flatc PROPERTIES COMPILE_FLAGS "-w")
    endif()
    _restore_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

    add_dependencies(${ARG_TARGET} ${_gen_target_name})

    # --- Secondary versions: configure-time download, build-time compilation + generation ---
    # For each supported version other than the primary, use FetchContent to download the
    # source at configure time, then build flatc at build time via ExternalProject_Add.
    # We can't use FetchContent_MakeAvailable / add_subdirectory because both FlatBuffers
    # versions define a "flatc" target and CMake targets are global (no scoping).
    # ExternalProject builds in its own CMake instance, avoiding target name collisions.

    foreach(_version IN LISTS ARG_SUPPORTED_VERSIONS)
        if(_version STREQUAL ARG_PRIMARY_VERSION)
            continue()
        endif()
        _hipdnn_generate_secondary_version(${_version} "${_flatc_flags}")
    endforeach()
endfunction()
