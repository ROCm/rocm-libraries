# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_bsrmv.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_bsrsv.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_coomv.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_csrmv.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_csrsv.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_ellmv.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gebsrmv.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gemvi.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_hybmv.cpp)

