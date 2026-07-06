# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

foreach(_required_var CLANG_FORMAT_BINARY FORMAT_SOURCE_DIR FORMAT_MODE)
    if(NOT DEFINED ${_required_var} OR "${${_required_var}}" STREQUAL "")
        message(FATAL_ERROR "${_required_var} is required")
    endif()
endforeach()

if(NOT FORMAT_MODE STREQUAL "check" AND NOT FORMAT_MODE STREQUAL "format")
    message(FATAL_ERROR "FORMAT_MODE must be 'check' or 'format'")
endif()

# Finds a relative path given a source path and file path
function(get_relative_format_path OUTPUT_VAR SOURCE_DIR FILE_PATH)
    get_filename_component(_source_dir_abs "${SOURCE_DIR}" ABSOLUTE)
    string(REPLACE "\\" "/" _source_dir "${SOURCE_DIR}")
    string(REPLACE "\\" "/" _source_dir_abs "${_source_dir_abs}")
    string(REPLACE "\\" "/" _file_path "${FILE_PATH}")
    string(REGEX REPLACE "/$" "" _source_dir "${_source_dir}")
    string(REGEX REPLACE "/$" "" _source_dir_abs "${_source_dir_abs}")

    string(FIND "${_file_path}" "${_source_dir_abs}/" _absolute_prefix_position)
    if(_absolute_prefix_position EQUAL 0)
        string(LENGTH "${_source_dir_abs}/" _prefix_length)
        string(SUBSTRING "${_file_path}" ${_prefix_length} -1 _relative_path)
    else()
        string(FIND "${_file_path}" "${_source_dir}/" _prefix_position)
        if(_prefix_position EQUAL 0)
            string(LENGTH "${_source_dir}/" _prefix_length)
            string(SUBSTRING "${_file_path}" ${_prefix_length} -1 _relative_path)
        else()
            set(_relative_path "${_file_path}")
        endif()
    endif()

    set(${OUTPUT_VAR} "${_relative_path}" PARENT_SCOPE)
endfunction()

file(
    GLOB_RECURSE _format_files
    LIST_DIRECTORIES FALSE
    "${FORMAT_SOURCE_DIR}/*.cpp"
    "${FORMAT_SOURCE_DIR}/*.hpp"
    "${FORMAT_SOURCE_DIR}/*.c"
    "${FORMAT_SOURCE_DIR}/*.h"
)
list(SORT _format_files)

set(_excluded_relative_prefixes
    "build/"
    "flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/data_objects/"
)

set(_filtered_files)
foreach(_format_file IN LISTS _format_files)
    get_relative_format_path(_relative_path "${FORMAT_SOURCE_DIR}" "${_format_file}")

    set(_skip_file FALSE)
    foreach(_excluded_prefix IN LISTS _excluded_relative_prefixes)
        string(FIND "${_relative_path}" "${_excluded_prefix}" _prefix_position)
        if(_prefix_position EQUAL 0)
            set(_skip_file TRUE)
            break()
        endif()
    endforeach()

    if(NOT _skip_file)
        list(APPEND _filtered_files "${_format_file}")
    endif()
endforeach()

if(NOT _filtered_files)
    message(STATUS "No source files found for clang-format")
    return()
endif()

if(FORMAT_MODE STREQUAL "check")
    set(_clang_format_args --dry-run --Werror)
else()
    set(_clang_format_args --verbose -i)
endif()

set(_failed_files)
foreach(_format_file IN LISTS _filtered_files)
    get_relative_format_path(_relative_path "${FORMAT_SOURCE_DIR}" "${_format_file}")

    execute_process(
        COMMAND "${CLANG_FORMAT_BINARY}" ${_clang_format_args} "${_format_file}"
        WORKING_DIRECTORY "${FORMAT_SOURCE_DIR}"
        RESULT_VARIABLE _clang_format_result
    )

    if(NOT _clang_format_result EQUAL 0)
        list(APPEND _failed_files "${_relative_path}")
    endif()
endforeach()

if(_failed_files)
    string(REPLACE ";" "\n  " _failed_files_message "${_failed_files}")
    message(FATAL_ERROR "clang-format ${FORMAT_MODE} failed for:\n  ${_failed_files_message}")
endif()
