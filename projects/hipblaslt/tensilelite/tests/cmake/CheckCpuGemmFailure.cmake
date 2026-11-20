# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

execute_process(
    COMMAND "${DRIVER}" ${ARGS}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    TIMEOUT 120
)
if(NOT "${result}" STREQUAL "1")
    message(FATAL_ERROR "Expected exit code 1, got ${result}:\n${output}${error}")
endif()
string(FIND "${output}${error}" "${EXPECTED_ERROR}" diagnostic_position)
if(diagnostic_position EQUAL -1)
    message(FATAL_ERROR "Missing diagnostic '${EXPECTED_ERROR}':\n${output}${error}")
endif()
