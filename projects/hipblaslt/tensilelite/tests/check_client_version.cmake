# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT CLIENT OR NOT EXPECTED_VERSION)
    message(FATAL_ERROR "CLIENT and EXPECTED_VERSION are required")
endif()

execute_process(
    COMMAND "${CLIENT}" --version
    RESULT_VARIABLE result
    OUTPUT_VARIABLE actual_version
    ERROR_VARIABLE error_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "${CLIENT} --version failed (${result}): ${error_output}")
endif()
if(NOT actual_version STREQUAL EXPECTED_VERSION)
    message(FATAL_ERROR
        "Client version mismatch: expected '${EXPECTED_VERSION}', got '${actual_version}'")
endif()
