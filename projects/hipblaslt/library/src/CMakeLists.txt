# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(amd_detail)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/auxiliary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_ostream.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipblaslt_internal.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipblaslt-ext-op-internal.hpp"
        
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/Debug.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/rocblaslt-auxiliary.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/rocblaslt-functions.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/rocblaslt-types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/rocblaslt.h"

        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/debug.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/definitions.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/handle.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/logging.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocroller_host.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocroller_host_internal.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt_mat_utils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/status.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/tuple_helper.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/tensile_host.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/UserDrivenTuningParser.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/utility.hpp"
)
