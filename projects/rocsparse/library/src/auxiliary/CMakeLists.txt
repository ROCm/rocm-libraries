# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocsparse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_auxiliary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_error.cpp
)
