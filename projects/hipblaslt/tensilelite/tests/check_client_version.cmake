# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Runs one tensilelite-client identity query and compares its single line of
# output exactly with the value CMake computed at configure time.
#   -DCLIENT=<path> -DQUERY=<--version|--source-revision|--build-rocm> -DEXPECTED=<value>

if(NOT CLIENT OR NOT QUERY OR NOT DEFINED EXPECTED OR EXPECTED STREQUAL "")
    message(FATAL_ERROR "CLIENT, QUERY and EXPECTED are required")
endif()

execute_process(
    COMMAND "${CLIENT}" "${QUERY}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE actual
    ERROR_VARIABLE error_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    TIMEOUT 5
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "${CLIENT} ${QUERY} failed (${result}): ${error_output}")
endif()
if(NOT actual STREQUAL EXPECTED)
    message(FATAL_ERROR
        "Client ${QUERY} mismatch: expected '${EXPECTED}', got '${actual}'")
endif()
