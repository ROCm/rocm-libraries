# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bsrmm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bsrsm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_csrmm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_csrsm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gemmi.cpp"
)
