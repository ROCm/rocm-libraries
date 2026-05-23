# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12|gfx13")
    add_gtest_executable(test_ck_tile_grouped_conv_bwd_weight test_ck_tile_grouped_conv_bwd_weight.cpp)
endif()

if(GPU_TARGETS MATCHES "gfx13")
    add_gtest_executable(test_ck_tile_wcnn_forward test_ck_tile_wcnn_forward.cpp)
endif()