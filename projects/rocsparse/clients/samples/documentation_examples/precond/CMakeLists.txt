# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_bsric0.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_bsrilu0.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_csric0.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_csrilu0.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_csritilu0.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gpsv.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gtsv.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gtsv_interleaved_batch.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gtsv_no_pivot.cpp)
add_rocsparse_example(${CMAKE_CURRENT_SOURCE_DIR}/example_rocsparse_gtsv_no_pivot_strided_batch.cpp)

