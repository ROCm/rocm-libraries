# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_pooling test_pooling.cpp)
    set(_ck_tile_pooling_quick_cases
        "TestCkTilePooling/0.Pool2D_2x2"
        "TestCkTilePooling/1.Pool2D_3x3_WithPadding"
        "TestCkTilePooling/2.Pool2D_2x2"
        "TestCkTilePooling/0.Pool3D_2x2x2"
        "TestCkTilePooling/1.Pool3D_3x3x3"
    )
    list(JOIN _ck_tile_pooling_quick_cases ":" _ck_tile_pooling_quick_filter)
    add_ck_tile_category_test(
        test_ck_tile_pooling_quick
        test_ck_tile_pooling
        "--gtest_filter=${_ck_tile_pooling_quick_filter}"
    )
endif()
