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

    set(HIPDNN_CLANG_FORMAT_FILES_PER_INVOCATION
        32
        CACHE STRING "Maximum source files passed to each clang-format invocation"
    )
    if(HIPDNN_CLANG_FORMAT_FILES_PER_INVOCATION LESS 1)
        message(FATAL_ERROR "HIPDNN_CLANG_FORMAT_FILES_PER_INVOCATION must be greater than 0")
    endif()

    set(HIPDNN_CLANG_FORMAT_JOBS
        0
        CACHE STRING "Maximum parallel clang-format invocations; 0 uses the host processor count"
    )
    if(HIPDNN_CLANG_FORMAT_JOBS LESS 0)
        message(FATAL_ERROR "HIPDNN_CLANG_FORMAT_JOBS must be greater than or equal to 0")
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # Adds a format and check-format target
    function(add_clang_format_target TARGET_NAME FORMAT_MODE)
        if(NOT FORMAT_MODE STREQUAL "check" AND NOT FORMAT_MODE STREQUAL "format")
            message(FATAL_ERROR "FORMAT_MODE must be 'check' or 'format'")
        endif()

        add_custom_target(
            ${TARGET_NAME}
            COMMAND
                "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/RunClangFormat.py"
                --clang-format "${CLANG_FORMAT_BINARY}" --source-dir "${PROJECT_SOURCE_DIR}"
                --mode "${FORMAT_MODE}" --files-per-invocation
                "${HIPDNN_CLANG_FORMAT_FILES_PER_INVOCATION}" --jobs
                "${HIPDNN_CLANG_FORMAT_JOBS}"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            VERBATIM
            COMMENT "Running clang-format ${FORMAT_MODE} (${PROJECT_NAME})"
        )
    endfunction()

    # Find and check clang-format version using unified function
    findandcheckclangformat()

    # Use prefixed target names in superbuild to avoid collisions
    if(ROCM_LIBS_SUPERBUILD)
        set(_CHECK_FORMAT_TARGET ${PROJECT_NAME}_check_format)
        set(_FORMAT_TARGET ${PROJECT_NAME}_format)
    else()
        set(_CHECK_FORMAT_TARGET check_format)
        set(_FORMAT_TARGET format)
    endif()

    add_clang_format_target(${_CHECK_FORMAT_TARGET} check)
    add_clang_format_target(${_FORMAT_TARGET} format)

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
