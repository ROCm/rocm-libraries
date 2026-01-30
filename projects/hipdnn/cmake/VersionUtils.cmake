# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Macro to setup versioning for a component
# Reads version.json, gets git hash, sets version variables, and calls project()
macro(hipdnn_setup_version COMPONENT_NAME)
    string(TOUPPER ${COMPONENT_NAME} COMPONENT_NAME_UPPER)

    # Read version from version.json
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.json" _version_json)
    string(JSON ${COMPONENT_NAME_UPPER}_VERSION GET ${_version_json} "${COMPONENT_NAME}_version")

    # Parse version components
    string(REPLACE "." ";" _version_list ${${COMPONENT_NAME_UPPER}_VERSION})
    list(GET _version_list 0 ${COMPONENT_NAME_UPPER}_VERSION_MAJOR)
    list(GET _version_list 1 ${COMPONENT_NAME_UPPER}_VERSION_MINOR)
    list(GET _version_list 2 ${COMPONENT_NAME_UPPER}_VERSION_PATCH)

    # Get git commit hash for tweak version
    execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ${COMPONENT_NAME_UPPER}_VERSION_TWEAK
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT ${COMPONENT_NAME_UPPER}_VERSION_TWEAK)
        set(${COMPONENT_NAME_UPPER}_VERSION_TWEAK "unknown")
    endif()

    # Full version string
    set(${COMPONENT_NAME_UPPER}_VERSION_STRING "${${COMPONENT_NAME_UPPER}_VERSION}.${${COMPONENT_NAME_UPPER}_VERSION_TWEAK}")

    # Set project with version
    project(${COMPONENT_NAME} VERSION ${${COMPONENT_NAME_UPPER}_VERSION} LANGUAGES CXX)

    message(STATUS "${COMPONENT_NAME} version: ${${COMPONENT_NAME_UPPER}_VERSION_STRING}")
endmacro()

# Function to generate the version header
macro(hipdnn_generate_version_header COMPONENT_NAME)
    set(_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")
    configure_file(
        "${_include_dir}/${COMPONENT_NAME}/version.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/${COMPONENT_NAME}/version.h"
        @ONLY
    )
endmacro()

# Capture the directory where this file resides
set(HIPDNN_CMAKE_UTILS_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Macro to read data_sdk version for minimum requirement
macro(hipdnn_get_data_sdk_version OUTPUT_VAR)
    # Determine path to data_sdk/version.json relative to this file location
    # This makes it resilient to where the macro is called from
    set(_data_sdk_version_file "${HIPDNN_CMAKE_UTILS_DIR}/../data_sdk/version.json")

    if(EXISTS "${_data_sdk_version_file}")
        file(READ "${_data_sdk_version_file}" _data_sdk_version_json)
        string(JSON ${OUTPUT_VAR} GET ${_data_sdk_version_json} "hipdnn_data_sdk_version")
    else()
        message(FATAL_ERROR "Could not find data_sdk version file at ${_data_sdk_version_file}")
    endif()
endmacro()
