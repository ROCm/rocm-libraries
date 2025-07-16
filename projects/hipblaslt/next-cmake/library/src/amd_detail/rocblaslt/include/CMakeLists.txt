# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(_CMAKE_CURRENT_SOURCE_DIR1 "${ROCBLASLT_LIB_DIR}/include")
set(_CMAKE_CURRENT_SOURCE_DIR2 "${ROCBLASLT_LIB_DIR}/src/include")

target_sources(hipblaslt
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR1}/Debug.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR1}/rocblaslt-auxiliary.h"
        "${_CMAKE_CURRENT_SOURCE_DIR1}/rocblaslt-functions.h"
        "${_CMAKE_CURRENT_SOURCE_DIR1}/rocblaslt-types.h"
        "${_CMAKE_CURRENT_SOURCE_DIR1}/rocblaslt.h"

        "${_CMAKE_CURRENT_SOURCE_DIR2}/debug.h"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/definitions.h"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/handle.h"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/logging.h"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/rocroller_host.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/rocroller_host_internal.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/rocblaslt_mat_utils.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/status.h"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/tuple_helper.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/tensile_host.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/UserDrivenTuningParser.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/utility.hpp"
)
