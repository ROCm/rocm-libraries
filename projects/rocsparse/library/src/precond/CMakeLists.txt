# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocsparse PRIVATE
    # bsric0
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_clear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_kernel_17_32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_kernel_2_8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_kernel_2_8_unrolled.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_kernel_9_16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_kernel_general.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_kernel_launch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_bsric0_zero_pivot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_xbsric0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_xbsric0_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsric0/rocsparse_xbsric0_buffer_size.cpp

    # bsrilu0
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0_clear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0_kernel_2_8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0_kernel_33_64.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0_kernel_9_32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0_kernel_general.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0_kernel_launch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_bsrilu0_zero_pivot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_xbsrilu0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_xbsrilu0_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_xbsrilu0_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsrilu0/rocsparse_xbsrilu0_numeric_boost.cpp

    # csric0
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_clear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_get_tolerance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_kernel_binsearch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_kernel_hash.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_kernel_launch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_set_tolerance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_singular_pivot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_csric0_zero_pivot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_xcsric0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_xcsric0_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csric0/rocsparse_xcsric0_buffer_size.cpp

    # csrilu0
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_clear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_get_tolerance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_kernel_binsearch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_kernel_hash.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_kernel_launch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_set_tolerance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_singular_pivot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_csrilu0_zero_pivot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_xcsrilu0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_xcsrilu0_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_xcsrilu0_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrilu0/rocsparse_xcsrilu0_numeric_boost.cpp

    # gpsv
    ${CMAKE_CURRENT_SOURCE_DIR}/gpsv/rocsparse_gpsv_interleaved_batch.cpp

    # gtsv
    ${CMAKE_CURRENT_SOURCE_DIR}/gtsv/rocsparse_gtsv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gtsv/rocsparse_gtsv_interleaved_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gtsv/rocsparse_gtsv_no_pivot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gtsv/rocsparse_gtsv_no_pivot_strided_batch.cpp

    # itilu0
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0_async_inplace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0_async_split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0_compute_ex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0_history.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0_sync_split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0_sync_split_fusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0x_async.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0x_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0x_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0x_history.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0x_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0x_sync.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itilu0/rocsparse_csritilu0x_sync_fusion.cpp
)
