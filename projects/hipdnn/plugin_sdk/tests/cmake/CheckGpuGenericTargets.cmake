# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Runs hipdnn_validate_gpu_generic_targets via `cmake -P`, optionally on a mutated copy:
#   -DGENERIC_TARGETS_JSON=<table.json> [-DMUTATION_OP=<op> -DMUTATION_PATH=<a.b.0> -DMUTATION_VALUE=<v>]
# MUTATION_OP: SET_STRING, SET_JSON, or RAW (replace the whole document).
# Prints "gpu_generic_targets: valid" only if every rule passed.

if(NOT GENERIC_TARGETS_JSON)
    message(FATAL_ERROR "CheckGpuGenericTargets: pass -DGENERIC_TARGETS_JSON=<table.json>")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/ValidateGpuGenericTargets.cmake")

set(table_path "${GENERIC_TARGETS_JSON}")
if(MUTATION_OP)
    if(NOT MUTATED_JSON)
        message(FATAL_ERROR "CheckGpuGenericTargets: MUTATION_OP needs -DMUTATED_JSON=<temp file>")
    endif()
    file(READ "${GENERIC_TARGETS_JSON}" json)
    string(REPLACE "." ";" path "${MUTATION_PATH}")
    if(MUTATION_OP STREQUAL "SET_STRING")
        string(JSON json SET "${json}" ${path} "\"${MUTATION_VALUE}\"")
    elseif(MUTATION_OP STREQUAL "SET_JSON")
        string(JSON json SET "${json}" ${path} "${MUTATION_VALUE}")
    elseif(MUTATION_OP STREQUAL "RAW")
        set(json "${MUTATION_VALUE}")
    else()
        message(FATAL_ERROR "CheckGpuGenericTargets: unknown MUTATION_OP '${MUTATION_OP}'")
    endif()
    file(WRITE "${MUTATED_JSON}" "${json}")
    set(table_path "${MUTATED_JSON}")
endif()

hipdnn_validate_gpu_generic_targets("${table_path}")
message(STATUS "gpu_generic_targets: valid")
