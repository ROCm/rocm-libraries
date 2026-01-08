# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_bsrmm.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_bsrsm.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_csrmm.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_csrsm.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gebsrmm.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gemmi.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_sddmm.cpp)

