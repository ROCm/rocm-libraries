# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TEMPLATE ADAPTATION: Rename the three functions in this file from
# example_provider_* to your_name_provider_* (snake_case brand + _provider).

# Function to find the version file path given the component name
function(example_provider_version_file_dir COMPONENT_NAME OUTPUT_PATH)
    set(${OUTPUT_PATH} "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../" PARENT_SCOPE)
endfunction()

# Function to setup versioning for a component
# Reads version.json, gets git hash, sets version variables
function(example_provider_setup_version COMPONENT_NAME)
    string(TOUPPER ${COMPONENT_NAME} COMPONENT_NAME_UPPER)

    # Read version from version.json
    example_provider_version_file_dir(${COMPONENT_NAME} _version_dir)
    file(READ "${_version_dir}/version.json" _version_json)
    string(JSON ${COMPONENT_NAME_UPPER}_VERSION GET ${_version_json} "${COMPONENT_NAME}_version")

    # Parse version components
    string(REPLACE "." ";" _version_list ${${COMPONENT_NAME_UPPER}_VERSION})
    list(GET _version_list 0 ${COMPONENT_NAME_UPPER}_VERSION_MAJOR)
    list(GET _version_list 1 ${COMPONENT_NAME_UPPER}_VERSION_MINOR)
    list(GET _version_list 2 ${COMPONENT_NAME_UPPER}_VERSION_PATCH)

    # Get git commit hash for tweak version
    execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE ${COMPONENT_NAME_UPPER}_VERSION_TWEAK
        RESULT_VARIABLE GIT_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(NOT GIT_RESULT EQUAL 0 OR ${COMPONENT_NAME_UPPER}_VERSION_TWEAK STREQUAL "")
        set(${COMPONENT_NAME_UPPER}_VERSION_TWEAK "unknown")
    endif()

    # Full version string
    set(${COMPONENT_NAME_UPPER}_VERSION_STRING "${${COMPONENT_NAME_UPPER}_VERSION}.${${COMPONENT_NAME_UPPER}_VERSION_TWEAK}")

    message(STATUS "${COMPONENT_NAME} version: ${${COMPONENT_NAME_UPPER}_VERSION_STRING}")

    # Propagate version variables to parent scope
    set(${COMPONENT_NAME_UPPER}_VERSION_MAJOR ${${COMPONENT_NAME_UPPER}_VERSION_MAJOR} PARENT_SCOPE)
    set(${COMPONENT_NAME_UPPER}_VERSION_MINOR ${${COMPONENT_NAME_UPPER}_VERSION_MINOR} PARENT_SCOPE)
    set(${COMPONENT_NAME_UPPER}_VERSION_PATCH ${${COMPONENT_NAME_UPPER}_VERSION_PATCH} PARENT_SCOPE)
    set(${COMPONENT_NAME_UPPER}_VERSION_TWEAK ${${COMPONENT_NAME_UPPER}_VERSION_TWEAK} PARENT_SCOPE)
    set(${COMPONENT_NAME_UPPER}_VERSION_STRING ${${COMPONENT_NAME_UPPER}_VERSION_STRING} PARENT_SCOPE)
    set(${COMPONENT_NAME_UPPER}_VERSION ${${COMPONENT_NAME_UPPER}_VERSION} PARENT_SCOPE)
endfunction()

# Function to generate the version header
function(example_provider_generate_version_header COMPONENT_NAME TARGET_NAME)
    example_provider_version_file_dir(${COMPONENT_NAME} _version_dir)
    configure_file(
        "${_version_dir}/templates/version.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/version.h"
        @ONLY
    )

    # Add generated include directory
    target_include_directories(
        ${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include
    )
endfunction()
