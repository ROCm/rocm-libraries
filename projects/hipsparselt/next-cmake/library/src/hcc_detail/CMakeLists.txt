# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../library/src/hcc_detail")

target_sources(hipsparselt PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/hipsparselt.cpp")

add_subdirectory(rocsparselt)
