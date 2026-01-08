# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocsparse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_exclusive_scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_find_max.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_find_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_inclusive_scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_radix_sort_keys.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_radix_sort_pairs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_run_length_encode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_segmented_radix_sort_keys.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_segmented_radix_sort_pairs.cpp
)
