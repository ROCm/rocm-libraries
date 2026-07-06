# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Helper function to verify a compiler is AMD/ROCm clang
function(verifyAmdRocmCompiler COMPILER_PATH COMPILER_NAME)
    execute_process(
        COMMAND ${COMPILER_PATH} --version OUTPUT_VARIABLE VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )

    if(VERSION_OUTPUT MATCHES "clang version"
       AND (VERSION_OUTPUT MATCHES "ROCm" OR VERSION_OUTPUT MATCHES "AMD" OR COMPILER_PATH MATCHES
                                                                             "rocm")
    )
        message(STATUS "✓ Confirmed AMD/ROCm type ${COMPILER_NAME} compiler")
    else()
        string(REGEX REPLACE "[\r\n]+" "\n  " VERSION_OUTPUT "${VERSION_OUTPUT}")
        message(
            WARNING "\n"
                    "Unable to confirm AMD/ROCm type ${COMPILER_NAME} compiler: ${COMPILER_PATH}\n"
                    "Expected to find \"AMD\" or \"ROCm\" in the compiler version.\n"
                    "Actual compiler version reported:\n  ${VERSION_OUTPUT}\n"
        )
    endif()
endfunction()

# Verify that we're using AMD/ROCm compiler.
verifyamdrocmcompiler(${CMAKE_CXX_COMPILER} "C++")

if(ENABLE_CLANG_FORMAT)
    include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)

    # Find and check clang-format version using unified function
    findandcheckclangformat()
    set(_CLANG_FORMAT_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/RunClangFormat.cmake")

    # Use prefixed target names in superbuild to avoid collisions
    if(ROCM_LIBS_SUPERBUILD)
        set(_CHECK_FORMAT_TARGET ${PROJECT_NAME}_check_format)
        set(_FORMAT_TARGET ${PROJECT_NAME}_format)
    else()
        set(_CHECK_FORMAT_TARGET check_format)
        set(_FORMAT_TARGET format)
    endif()

    add_custom_target(
        ${_CHECK_FORMAT_TARGET}
        COMMAND
            "${CMAKE_COMMAND}" "-DCLANG_FORMAT_BINARY=${CLANG_FORMAT_BINARY}"
            "-DFORMAT_SOURCE_DIR=${PROJECT_SOURCE_DIR}" "-DFORMAT_MODE=check" -P
            "${_CLANG_FORMAT_SCRIPT}"
        VERBATIM
        COMMENT "Checking code format (${PROJECT_NAME})"
    )

    add_custom_target(
        ${_FORMAT_TARGET}
        COMMAND
            "${CMAKE_COMMAND}" "-DCLANG_FORMAT_BINARY=${CLANG_FORMAT_BINARY}"
            "-DFORMAT_SOURCE_DIR=${PROJECT_SOURCE_DIR}" "-DFORMAT_MODE=format" -P
            "${_CLANG_FORMAT_SCRIPT}"
        VERBATIM
        COMMENT "Formatting code (${PROJECT_NAME})"
    )

    # Alias targets with consistent hyphenated naming
    add_custom_target(
        ${PROJECT_NAME}-check-format
        DEPENDS ${_CHECK_FORMAT_TARGET}
        COMMENT "Alias for ${_CHECK_FORMAT_TARGET}"
    )
    add_custom_target(
        ${PROJECT_NAME}-format
        DEPENDS ${_FORMAT_TARGET}
        COMMENT "Alias for ${_FORMAT_TARGET}"
    )
endif() # ENABLE_CLANG_FORMAT
