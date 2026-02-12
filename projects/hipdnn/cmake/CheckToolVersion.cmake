# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Expected tool versions
set(EXPECTED_CLANG_FORMAT_VERSION "18")
set(EXPECTED_CLANG_TIDY_VERSION "20")
set(EXPECTED_LLVM_VERSION "20")

# Helper function to generate version-specific search paths hints by concatenating the base path
# with a list of versioned path names.
function(get_versioned_search_paths OUTPUT_VAR BASE_PATH VERSION)
    set(PATHS_LIST
        "${BASE_PATH}${VERSION}/bin" "${BASE_PATH}${VERSION}/lib/llvm/bin"
        "${BASE_PATH}/${VERSION}/bin" "${BASE_PATH}/${VERSION}/lib/llvm/bin" "${BASE_PATH}/bin"
        "${BASE_PATH}/lib/llvm/bin"
    )
    set(${OUTPUT_VAR} ${PATHS_LIST} PARENT_SCOPE)
endfunction()

# CMake find_program() search order: CMAKE_PREFIX_PATH, CMAKE_PROGRAM_PATH, find_program(HINTS)
# which is set to LLVM_TOOL_HINTS below when LLVM_TOOLS_SEARCH_PREFIX is provided by the user,
# CMAKE_*_COMPILER_PATH, system PATH, CMake built-in common locations, and finally
# find_program(PATHS) which is set to LLVM_TOOL_PATHS in this file. All folders are searched first
# for the first program name, and this then repeats for each name provided in find_program(NAMES).
if(NOT WIN32)
    # Common search paths
    set(LLVM_TOOL_PATHS /usr/bin /usr/local/bin /opt/rocm/llvm/bin)
    if(DEFINED ROCM_PATH)
        list(APPEND LLVM_TOOL_PATHS ${ROCM_PATH}/llvm/bin)
    endif()
endif()

# Set up LLVM_TOOL_HINTS if LLVM_TOOLS_SEARCH_PREFIX is defined
# First check if it was set via cmake -D, otherwise fall back to environment variable
if(DEFINED LLVM_TOOLS_SEARCH_PREFIX)
    set(LLVM_TOOL_HINTS "${LLVM_TOOLS_SEARCH_PREFIX}")
    message(VERBOSE "Using LLVM_TOOLS_SEARCH_PREFIX hint: ${LLVM_TOOLS_SEARCH_PREFIX}")
elseif(DEFINED ENV{LLVM_TOOLS_SEARCH_PREFIX})
    set(LLVM_TOOL_HINTS "$ENV{LLVM_TOOLS_SEARCH_PREFIX}")
    message(VERBOSE "Using LLVM_TOOLS_SEARCH_PREFIX hint from environment: $ENV{LLVM_TOOLS_SEARCH_PREFIX}")
endif()

# Checks the version of a tool
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
        # Set the major version in parent scope for potential use
        set(${TOOL_NAME}_MAJOR_VERSION ${TOOL_MAJOR_VERSION} PARENT_SCOPE)
    else()
        message(WARNING "Could not determine ${TOOL_NAME} version from: ${VERSION_OUTPUT}")
        set(${TOOL_NAME}_MAJOR_VERSION "unknown" PARENT_SCOPE)
    endif()
endfunction()

# Common helper function to find and check a tool. This function will search first for a file named
# ${TOOL_NAME}-${EXPECTED_VERSION} and then for the bare ${TOOL_NAME}. Folders based on a provided
# LLVM_TOOLS_SEARCH_PREFIX will be searched as well as the standard CMake paths used by
# find_program().
# ~~~
# Parameters:
#   OUTPUT_VAR - Variable name to store the found tool path
#   TOOL_NAME - The single name of the single tool to search for (e.g., "clang-format")
#   EXPECTED_VERSION - Expected version to search for (typically assumed to be the tool's major version number)
#   VERSION_REGEX - Regex to extract the relevant portion to match EXPECTED_VERSION frmo the tool's --version output
#   ERROR_LEVEL - FATAL_ERROR, WARNING, STATUS, or VERBOSE for the not found message
# ~~~
function(findAndCheckTool OUTPUT_VAR TOOL_NAME EXPECTED_VERSION VERSION_REGEX ERROR_LEVEL)
    # Build version-specific paths if LLVM_TOOL_HINTS is set
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
        # Format SEARCH_HINTS with one path per line for output.
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
            if(SEARCH_HINTS)
                message(
                    VERBOSE
                    "Search included the following LLVM_TOOLS_SEARCH_PREFIX derived paths:\n  ${FORMATTED_HINTS}"
                )
            endif()
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

    # Export to parent scope
    set(${OUTPUT_VAR} ${${OUTPUT_VAR}} PARENT_SCOPE)
endfunction()

# Finds and checks clang-format
function(findAndCheckClangFormat)
    findandchecktool(
        CLANG_FORMAT_BINARY "clang-format" ${EXPECTED_CLANG_FORMAT_VERSION}
        "clang-format version ([0-9]+)\\." FATAL_ERROR
    )

    # Export to parent scope
    set(CLANG_FORMAT_BINARY ${CLANG_FORMAT_BINARY} PARENT_SCOPE)
endfunction()

# Finds and checks clang-tidy
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

    # Export to parent scope
    set(CLANG_TIDY_EXE ${CLANG_TIDY_EXE} PARENT_SCOPE)

    findandchecktool(
        RUN_CLANG_TIDY_EXE "run-clang-tidy" ${EXPECTED_CLANG_TIDY_VERSION} "SKIP_VERSION_CHECK"
        VERBOSE
    )

    if(RUN_CLANG_TIDY_EXE)
        set(RUN_CLANG_TIDY_EXE ${RUN_CLANG_TIDY_EXE} PARENT_SCOPE)
    endif()
endfunction()

# Finds and checks LLVM tools
function(findAndCheckLlvmTools)
    # Define the tools we need
    set(LLVM_TOOLS llvm-profdata llvm-cov llvm-cxxfilt)

    foreach(TOOL ${LLVM_TOOLS})
        string(TOUPPER ${TOOL} TOOL_UPPER)
        string(REPLACE "-" "_" TOOL_VAR ${TOOL_UPPER})

        findandchecktool(
            ${TOOL_VAR}_BINARY "${TOOL}" ${EXPECTED_LLVM_VERSION} "LLVM version ([0-9]+)\\."
            FATAL_ERROR
        )

        # Export to parent scope
        set(${TOOL_VAR}_BINARY ${${TOOL_VAR}_BINARY} PARENT_SCOPE)
    endforeach()
endfunction()

# Finds and checks llvm-symbolizer
function(findAndCheckLlvmSymbolizer)
    findandchecktool(
        LLVM_SYMBOLIZER_EXE "llvm-symbolizer" ${EXPECTED_LLVM_VERSION} "LLVM version ([0-9]+)\\."
        WARNING
    )

    if(LLVM_SYMBOLIZER_EXE)
        set(CMAKE_SYMBOLIZER ${LLVM_SYMBOLIZER_EXE} PARENT_SCOPE)
        # Export to parent scope
        set(LLVM_SYMBOLIZER_EXE ${LLVM_SYMBOLIZER_EXE} PARENT_SCOPE)
    else()
        message(WARNING "ASAN tests will be missing symbolized stack traces.")
    endif()
endfunction()
