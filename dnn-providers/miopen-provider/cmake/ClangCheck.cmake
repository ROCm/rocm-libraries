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
