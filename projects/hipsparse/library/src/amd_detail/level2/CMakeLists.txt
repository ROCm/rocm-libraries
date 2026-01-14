# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bsrmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bsrsv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bsrxmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_csrmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_csrsv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gemvi.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_hybmv.cpp"
)
