# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Extra function sources
target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_csrgeam.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_csrgemm.cpp"
)
