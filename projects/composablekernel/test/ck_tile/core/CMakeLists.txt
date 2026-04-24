# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_subdirectory(arch)
add_subdirectory(container)

add_gtest_executable(test_ck_tile_cyclic_shift_transform test_cyclic_shift_transform.cpp)
