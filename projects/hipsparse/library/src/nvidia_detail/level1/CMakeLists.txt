# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_axpyi.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_dotci.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_doti.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gthr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gthrz.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_roti.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_sctr.cpp"
)
