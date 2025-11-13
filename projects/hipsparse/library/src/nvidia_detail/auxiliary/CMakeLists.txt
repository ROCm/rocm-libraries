# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Auxiliary function sources
target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_auxiliary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_generic_auxiliary.cpp"
)
