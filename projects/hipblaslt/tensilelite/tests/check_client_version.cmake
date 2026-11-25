# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Runs one tensilelite-client identity query and compares its single line of
# output exactly with the saved final build metadata.
#   -DCLIENT=<path> -DQUERY=<--version|--source-revision|--build-rocm> -DMETADATA=<metadata.json>
# Without METADATA, -DEXPECTED=<value> supplies the expected line directly.

if(METADATA)
    file(READ "${METADATA}" _metadata)
    if(QUERY STREQUAL "--version")
        set(_key distribution_version)
    elseif(QUERY STREQUAL "--source-revision")
        set(_key source_revision)
    elseif(QUERY STREQUAL "--build-rocm")
        set(_key build_rocm)
    else()
        message(FATAL_ERROR "Unsupported identity query: ${QUERY}")
    endif()
    string(JSON EXPECTED GET "${_metadata}" "${_key}")
endif()
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
