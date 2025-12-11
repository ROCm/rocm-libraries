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
        "${BASE_PATH}${VERSION}/bin"
        "${BASE_PATH}${VERSION}/lib/llvm/bin"
        "${BASE_PATH}-${VERSION}/bin"
        "${BASE_PATH}-${VERSION}/lib/llvm/bin"
        "${BASE_PATH}_${VERSION}/bin"
        "${BASE_PATH}_${VERSION}/lib/llvm/bin"
        "${BASE_PATH}/${VERSION}/bin"
        "${BASE_PATH}/${VERSION}/lib/llvm/bin"
        "${BASE_PATH}/bin"
        "${BASE_PATH}/lib/llvm/bin"
    )
    set(${OUTPUT_VAR} ${PATHS_LIST} PARENT_SCOPE)
endfunction()

# CMake find_program() search order: CMAKE_PREFIX_PATH, CMAKE_PROGRAM_PATH, find_program(HINTS)
# which is set to LLVM_TOOL_HINTS below when LLVM_TOOLS_SEARCH_PREFIX is provided by the user,
# CMAKE_*_COMPILER_PATH, system PATH, CMake built-in common locations, and finally
# find_program(PATHS) which is set to LLVM_TOOL_PATHS in this file. All folders are searched first
# for the first program name, and this then repeats for each name provided in find_program(NAMES).
set(LLVM_TOOL_PATHS /usr/bin)
get_filename_component(COMPILER_PATH "${CMAKE_CXX_COMPILER}" PATH)
list(APPEND LLVM_TOOL_PATHS ${COMPILER_PATH})

# Set up LLVM_TOOL_HINTS if LLVM_TOOLS_SEARCH_PREFIX is defined
if(DEFINED LLVM_TOOLS_SEARCH_PREFIX)
    set(LLVM_TOOL_HINTS "${LLVM_TOOLS_SEARCH_PREFIX}")
    message(VERBOSE "Using LLVM_TOOLS_SEARCH_PREFIX as hint: ${LLVM_TOOLS_SEARCH_PREFIX}")
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

# Finds and checks clang-format
function(findAndCheckClangFormat)
    # Build version-specific paths if LLVM_TOOL_HINTS is set
    set(SEARCH_HINTS)
    if(DEFINED LLVM_TOOL_HINTS)
        foreach(HINT ${LLVM_TOOL_HINTS})
            get_versioned_search_paths(VERSIONED_HINTS "${HINT}" "${EXPECTED_CLANG_FORMAT_VERSION}")
            list(APPEND SEARCH_HINTS ${VERSIONED_HINTS})
        endforeach()
    endif()

    find_program(
        CLANG_FORMAT_BINARY NAMES clang-format-${EXPECTED_CLANG_FORMAT_VERSION} clang-format
        HINTS ${SEARCH_HINTS} PATHS ${LLVM_TOOL_PATHS}
    )

    if(NOT CLANG_FORMAT_BINARY)
        message(
            FATAL_ERROR "clang-format not found in PATH, /opt/rocm/llvm/bin, or compiler directory"
        )
        return()
    endif()

    checktoolversion(
        ${CLANG_FORMAT_BINARY} "clang-format" ${EXPECTED_CLANG_FORMAT_VERSION}
        "clang-format version ([0-9]+)\\." "Found clang-format version {VERSION} at {PATH}"
    )

    # Export to parent scope
    set(CLANG_FORMAT_BINARY ${CLANG_FORMAT_BINARY} PARENT_SCOPE)
endfunction()

# Finds and checks clang-tidy
function(findAndCheckClangTidy)
    # Build version-specific paths if LLVM_TOOL_HINTS is set
    set(SEARCH_HINTS)
    if(DEFINED LLVM_TOOL_HINTS)
        foreach(HINT ${LLVM_TOOL_HINTS})
            get_versioned_search_paths(VERSIONED_HINTS "${HINT}" "${EXPECTED_CLANG_TIDY_VERSION}")
            list(APPEND SEARCH_HINTS ${VERSIONED_HINTS})
        endforeach()
    endif()

    find_program(
        CLANG_TIDY_EXE NAMES clang-tidy-${EXPECTED_CLANG_TIDY_VERSION} clang-tidy
        HINTS ${SEARCH_HINTS} PATHS ${LLVM_TOOL_PATHS}
    )

    if(NOT CLANG_TIDY_EXE)
        message(
            FATAL_ERROR "clang-tidy not found in PATH, /opt/rocm/llvm/bin, or compiler directory"
        )
        return()
    endif()

    checktoolversion(
        ${CLANG_TIDY_EXE} "clang-tidy" ${EXPECTED_CLANG_TIDY_VERSION} "version ([0-9]+)\\."
        "Found clang-tidy version {VERSION} at {PATH}"
    )

    # Export to parent scope
    set(CLANG_TIDY_EXE ${CLANG_TIDY_EXE} PARENT_SCOPE)
endfunction()

# Finds and checks LLVM tools
function(findAndCheckLlvmTools)
    # Build version-specific paths if LLVM_TOOL_HINTS is set
    set(SEARCH_HINTS)
    if(DEFINED LLVM_TOOL_HINTS)
        foreach(HINT ${LLVM_TOOL_HINTS})
            get_versioned_search_paths(VERSIONED_HINTS "${HINT}" "${EXPECTED_LLVM_VERSION}")
            list(APPEND SEARCH_HINTS ${VERSIONED_HINTS})
        endforeach()
    endif()

    # Define the tools we need
    set(LLVM_TOOLS llvm-profdata llvm-cov llvm-cxxfilt)

    foreach(TOOL ${LLVM_TOOLS})
        string(TOUPPER ${TOOL} TOOL_UPPER)
        string(REPLACE "-" "_" TOOL_VAR ${TOOL_UPPER})

        find_program(
            ${TOOL_VAR}_BINARY NAMES ${TOOL}-${EXPECTED_LLVM_VERSION} ${TOOL} HINTS ${SEARCH_HINTS}
            PATHS ${LLVM_TOOL_PATHS}
        )

        if(NOT ${TOOL_VAR}_BINARY)
            message(
                FATAL_ERROR "${TOOL} not found in PATH, /opt/rocm/llvm/bin, or compiler directory"
            )
            return()
        endif()

        checktoolversion(
            ${${TOOL_VAR}_BINARY} ${TOOL} ${EXPECTED_LLVM_VERSION} "LLVM version ([0-9]+)\\."
            "Found ${TOOL} version {VERSION} at {PATH}"
        )

        # Export to parent scope
        set(${TOOL_VAR}_BINARY ${${TOOL_VAR}_BINARY} PARENT_SCOPE)
    endforeach()
endfunction()

# Finds and checks llvm-symbolizer
function(findAndCheckLlvmSymbolizer)
    # Build version-specific paths if LLVM_TOOL_HINTS is set
    set(SEARCH_HINTS)
    if(DEFINED LLVM_TOOL_HINTS)
        foreach(HINT ${LLVM_TOOL_HINTS})
            get_versioned_search_paths(VERSIONED_HINTS "${HINT}" "${EXPECTED_LLVM_VERSION}")
            list(APPEND SEARCH_HINTS ${VERSIONED_HINTS})
        endforeach()
    endif()

    find_program(
        LLVM_SYMBOLIZER_EXE NAMES llvm-symbolizer-${EXPECTED_LLVM_VERSION} llvm-symbolizer
        HINTS ${SEARCH_HINTS} PATHS ${LLVM_TOOL_PATHS}
    )

    if(NOT LLVM_SYMBOLIZER_EXE)
        message(
            WARNING
                "llvm-symbolizer not found in PATH, /opt/rocm/llvm/bin, or compiler directory.  ASAN tests will be missing symbolized stack traces."
        )
        return()
    endif()

    checktoolversion(
        ${LLVM_SYMBOLIZER_EXE} "llvm-symbolizer" ${EXPECTED_LLVM_VERSION}
        "LLVM version ([0-9]+)\\." "Found llvm-symbolizer version {VERSION} at {PATH}"
    )

    set(CMAKE_SYMBOLIZER ${LLVM_SYMBOLIZER_EXE} PARENT_SCOPE)
    # Export to parent scope
    set(LLVM_SYMBOLIZER_EXE ${LLVM_SYMBOLIZER_EXE} PARENT_SCOPE)
endfunction()
