# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_check_numerics_multi
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_check_numerics_multi.cpp")
target_link_libraries(sample_check_numerics_multi PRIVATE sample-helper)
