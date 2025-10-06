# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Expected tool versions
set(EXPECTED_CLANG_FORMAT_VERSION "18")
set(EXPECTED_CLANG_TIDY_VERSION "20")
set(EXPECTED_LLVM_VERSION "20")

# Common search paths
set(LLVM_TOOL_PATHS /usr/bin /opt/rocm/llvm/bin)
get_filename_component(COMPILER_PATH "${CMAKE_CXX_COMPILER}" PATH)
list(APPEND LLVM_TOOL_PATHS ${COMPILER_PATH})

function(checkToolVersion TOOL_BINARY TOOL_NAME EXPECTED_VERSION VERSION_REGEX SUCCESS_MESSAGE_FORMAT)
    execute_process(
        COMMAND ${TOOL_BINARY} --version
        OUTPUT_VARIABLE VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(VERSION_OUTPUT MATCHES "${VERSION_REGEX}")
        set(TOOL_MAJOR_VERSION "${CMAKE_MATCH_1}")
        if(NOT TOOL_MAJOR_VERSION STREQUAL EXPECTED_VERSION)
            message(WARNING "${TOOL_NAME} version mismatch! Expected: ${EXPECTED_VERSION}, Found: ${TOOL_MAJOR_VERSION}, Full version: ${VERSION_OUTPUT}")
        else()
            string(REPLACE "{VERSION}" "${TOOL_MAJOR_VERSION}" SUCCESS_MSG "${SUCCESS_MESSAGE_FORMAT}")
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

function(findAndCheckClangFormat)
    find_program(CLANG_FORMAT_BINARY
        NAMES
            clang-format-${EXPECTED_CLANG_FORMAT_VERSION}
            clang-format
        PATHS
            ${LLVM_TOOL_PATHS}
    )
    
    if(NOT CLANG_FORMAT_BINARY)
        message(FATAL_ERROR "clang-format not found in PATH, /opt/rocm/llvm/bin, or compiler directory")
        return()
    endif()
    
    checkToolVersion(
        ${CLANG_FORMAT_BINARY}
        "clang-format"
        ${EXPECTED_CLANG_FORMAT_VERSION}
        "clang-format version ([0-9]+)\\."
        "Found clang-format version {VERSION} at {PATH}"
    )
    
    # Export to parent scope
    set(CLANG_FORMAT_BINARY ${CLANG_FORMAT_BINARY} PARENT_SCOPE)
endfunction()

function(findAndCheckClangTidy)
    find_program(CLANG_TIDY_EXE
        NAMES
            clang-tidy-${EXPECTED_CLANG_TIDY_VERSION}
            clang-tidy
        PATHS
            ${LLVM_TOOL_PATHS}
    )
    
    if(NOT CLANG_TIDY_EXE)
        message(FATAL_ERROR "clang-tidy not found in PATH, /opt/rocm/llvm/bin, or compiler directory")
        return()
    endif()
    
    checkToolVersion(
        ${CLANG_TIDY_EXE}
        "clang-tidy"
        ${EXPECTED_CLANG_TIDY_VERSION}
        "version ([0-9]+)\\."
        "Found clang-tidy version {VERSION} at {PATH}"
    )
    
    # Export to parent scope
    set(CLANG_TIDY_EXE ${CLANG_TIDY_EXE} PARENT_SCOPE)
endfunction()

function(findAndCheckLlvmTools)
    # Define the tools we need
    set(LLVM_TOOLS llvm-profdata llvm-cov llvm-cxxfilt)
    
    foreach(TOOL ${LLVM_TOOLS})
        string(TOUPPER ${TOOL} TOOL_UPPER)
        string(REPLACE "-" "_" TOOL_VAR ${TOOL_UPPER})
        
        find_program(${TOOL_VAR}_BINARY
            NAMES
                ${TOOL}-${EXPECTED_LLVM_VERSION}
                ${TOOL}
            PATHS
                ${LLVM_TOOL_PATHS}
        )
        
        if(NOT ${TOOL_VAR}_BINARY)
            message(FATAL_ERROR "${TOOL} not found in PATH, /opt/rocm/llvm/bin, or compiler directory")
            return()
        endif()
        
        checkToolVersion(
            ${${TOOL_VAR}_BINARY}
            ${TOOL}
            ${EXPECTED_LLVM_VERSION}
            "LLVM version ([0-9]+)\\."
            "Found ${TOOL} version {VERSION} at {PATH}"
        )
        
        # Export to parent scope
        set(${TOOL_VAR}_BINARY ${${TOOL_VAR}_BINARY} PARENT_SCOPE)
    endforeach()
endfunction()

function(findAndCheckLlvmSymbolizer)
    find_program(LLVM_SYMBOLIZER_EXE
        NAMES
            llvm-symbolizer-${EXPECTED_LLVM_VERSION}
            llvm-symbolizer
        PATHS
            ${LLVM_TOOL_PATHS}
    )
    
    if(NOT LLVM_SYMBOLIZER_EXE)
        message(FATAL_ERROR "llvm-symbolizer not found in PATH, /opt/rocm/llvm/bin, or compiler directory")
        return()
    endif()
    
    checkToolVersion(
        ${LLVM_SYMBOLIZER_EXE}
        "llvm-symbolizer"
        ${EXPECTED_LLVM_VERSION}
        "LLVM version ([0-9]+)\\."
        "Found llvm-symbolizer version {VERSION} at {PATH}"
    )
    
    set(CMAKE_SYMBOLIZER ${LLVM_SYMBOLIZER_EXE} PARENT_SCOPE)
    # Export to parent scope
    set(LLVM_SYMBOLIZER_EXE ${LLVM_SYMBOLIZER_EXE} PARENT_SCOPE)
endfunction()