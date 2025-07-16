# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(_CMAKE_CURRENT_SOURCE_DIR "${HIPBLASLT_LIB_DIR}/src/include")

target_sources(hipblaslt
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/auxiliary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/exceptions.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_ostream.hpp"
)
