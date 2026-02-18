# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bsric0.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bsrilu0.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_csric0.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_csrilu0.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gpsv_interleaved_batch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gtsv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gtsv_interleaved_batch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gtsv_nopivot.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gtsv_strided_batch.cpp"
)
