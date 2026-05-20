# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared helpers for finding and version-checking LLVM/Clang tools.
# Generalized from dnn-providers/cmake/CheckToolVersion.cmake.

set(EXPECTED_CLANG_FORMAT_VERSION "18")
set(EXPECTED_CLANG_TIDY_VERSION "20")
set(EXPECTED_LLVM_VERSION "20")

# Build a list of versioned tool search paths from a base directory.
function(get_versioned_search_paths OUTPUT_VAR BASE_PATH VERSION)
    set(PATHS_LIST
        "${BASE_PATH}${VERSION}/bin" "${BASE_PATH}${VERSION}/lib/llvm/bin"
        "${BASE_PATH}/${VERSION}/bin" "${BASE_PATH}/${VERSION}/lib/llvm/bin" "${BASE_PATH}/bin"
        "${BASE_PATH}/lib/llvm/bin"
    )
    set(${OUTPUT_VAR} ${PATHS_LIST} PARENT_SCOPE)
endfunction()

if(NOT WIN32)
    set(LLVM_TOOL_PATHS /usr/bin /usr/local/bin /opt/rocm/llvm/bin)
    if(DEFINED ROCM_PATH)
        list(APPEND LLVM_TOOL_PATHS ${ROCM_PATH}/llvm/bin)
    endif()
endif()

if(DEFINED LLVM_TOOLS_SEARCH_PREFIX)
    set(LLVM_TOOL_HINTS "${LLVM_TOOLS_SEARCH_PREFIX}")
elseif(DEFINED ENV{LLVM_TOOLS_SEARCH_PREFIX})
    set(LLVM_TOOL_HINTS "$ENV{LLVM_TOOLS_SEARCH_PREFIX}")
endif()

# Verify that a tool binary matches the expected major version.
function(checkToolVersion TOOL_BINARY TOOL_NAME EXPECTED_VERSION VERSION_REGEX
         SUCCESS_MESSAGE_FORMAT
)
    execute_process(
        COMMAND ${TOOL_BINARY} --version OUTPUT_VARIABLE VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(VERSION_OUTPUT MATCHES "${VERSION_REGEX}")
        set(TOOL_MAJOR_VERSION "${CMAKE_MATCH_1}")
        if(NOT TOOL_MAJOR_VERSION STREQUAL EXPECTED_VERSION)
            message(
                WARNING
                    "${TOOL_NAME} version mismatch! Expected: ${EXPECTED_VERSION}, Found: ${TOOL_MAJOR_VERSION}, Full version: ${VERSION_OUTPUT}"
            )
        else()
            string(REPLACE "{VERSION}" "${TOOL_MAJOR_VERSION}" SUCCESS_MSG
                           "${SUCCESS_MESSAGE_FORMAT}"
            )
            string(REPLACE "{PATH}" "${TOOL_BINARY}" SUCCESS_MSG "${SUCCESS_MSG}")
            message(STATUS "${SUCCESS_MSG}")
        endif()
        set(${TOOL_NAME}_MAJOR_VERSION ${TOOL_MAJOR_VERSION} PARENT_SCOPE)
    else()
        message(WARNING "Could not determine ${TOOL_NAME} version from: ${VERSION_OUTPUT}")
        set(${TOOL_NAME}_MAJOR_VERSION "unknown" PARENT_SCOPE)
    endif()
endfunction()

# Find a tool by name and check its version.
function(findAndCheckTool OUTPUT_VAR TOOL_NAME EXPECTED_VERSION VERSION_REGEX ERROR_LEVEL)
    set(SEARCH_HINTS)
    if(DEFINED LLVM_TOOL_HINTS)
        foreach(HINT ${LLVM_TOOL_HINTS})
            get_versioned_search_paths(VERSIONED_HINTS "${HINT}" "${EXPECTED_VERSION}")
            list(APPEND SEARCH_HINTS ${VERSIONED_HINTS})
        endforeach()
    endif()

    find_program(
        ${OUTPUT_VAR} NAMES ${TOOL_NAME}-${EXPECTED_VERSION} ${TOOL_NAME} HINTS ${SEARCH_HINTS}
        PATHS ${LLVM_TOOL_PATHS}
    )

    if(NOT ${OUTPUT_VAR})
        string(REPLACE ";" "\n  " FORMATTED_HINTS "${SEARCH_HINTS}")
        if(ERROR_LEVEL STREQUAL "FATAL_ERROR")
            message(
                FATAL_ERROR
                    "\n${TOOL_NAME} not found in searching PATH, CMake PROGRAM and PREFIX paths, and these LLVM_TOOLS_SEARCH_PREFIX derived paths:\n  ${FORMATTED_HINTS}\n"
            )
        else()
            message(
                ${ERROR_LEVEL}
                "${TOOL_NAME} not found in searching PATH, CMake PROGRAM and PREFIX paths, and LLVM_TOOLS_SEARCH_PREFIX derived paths"
            )
        endif()
        return()
    endif()

    if(VERSION_REGEX STREQUAL "SKIP_VERSION_CHECK")
        message(STATUS "Found ${TOOL_NAME} at ${${OUTPUT_VAR}}")
    else()
        checktoolversion(
            ${${OUTPUT_VAR}} "${TOOL_NAME}" ${EXPECTED_VERSION} "${VERSION_REGEX}"
            "Found ${TOOL_NAME} version {VERSION} at {PATH}"
        )
    endif()

    set(${OUTPUT_VAR} ${${OUTPUT_VAR}} PARENT_SCOPE)
endfunction()

# Find clang-format and verify its version.
function(findAndCheckClangFormat)
    findandchecktool(
        CLANG_FORMAT_BINARY "clang-format" ${EXPECTED_CLANG_FORMAT_VERSION}
        "clang-format version ([0-9]+)\\." FATAL_ERROR
    )
    set(CLANG_FORMAT_BINARY ${CLANG_FORMAT_BINARY} PARENT_SCOPE)
endfunction()

# Find clang-tidy and run-clang-tidy, verify versions.
function(findAndCheckClangTidy)
    if(ENABLE_CLANG_TIDY)
        set(_not_found_log_level FATAL_ERROR)
    else()
        set(_not_found_log_level STATUS)
    endif()

    findandchecktool(
        CLANG_TIDY_EXE "clang-tidy" ${EXPECTED_CLANG_TIDY_VERSION} "version ([0-9]+)\\."
        ${_not_found_log_level}
    )
    set(CLANG_TIDY_EXE ${CLANG_TIDY_EXE} PARENT_SCOPE)

    findandchecktool(
        RUN_CLANG_TIDY_EXE "run-clang-tidy" ${EXPECTED_CLANG_TIDY_VERSION} "SKIP_VERSION_CHECK"
        VERBOSE
    )
    if(RUN_CLANG_TIDY_EXE)
        set(RUN_CLANG_TIDY_EXE ${RUN_CLANG_TIDY_EXE} PARENT_SCOPE)
    endif()
endfunction()
