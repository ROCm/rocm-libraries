# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocblaslt)

set(_CMAKE_CURRENT_SOURCE_DIR "${HIPBLASLT_LIB_DIR}/src/amd_detail")

target_sources(hipblaslt
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/../hipblaslt_ostream.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext-op.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt.cpp"
)
