# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_grouped_conv_bwd_weight test_ck_tile_grouped_conv_bwd_weight.cpp)
    add_gtest_executable(test_ck_tile_grouped_conv_fwd test_ck_tile_grouped_conv_fwd.cpp)
endif()

# StreamK requires cross-CU coherence via StreamKCoherency, which only has
# specializations for CDNA architectures (gfx90a/gfx942/gfx950).
if(GPU_TARGETS MATCHES "gfx90a|gfx942|gfx950")
    add_gtest_executable(test_ck_tile_grouped_conv_bwd_weight_streamk test_ck_tile_grouped_conv_bwd_weight_streamk.cpp)
    # Cover both reduction strategies, multiple shapes/groups, stride 2,
    # pure data-parallel work, and persistent scheduling.
    set(_ck_tile_grouped_conv_quick_cases
        "StreamKConvBwdWeight.Linear_EndToEnd_SmallShape"
        "StreamKConvBwdWeight.Tree_EndToEnd_MediumShape"
        "StreamKConvBwdWeight.Linear_EndToEnd_MultiGroup"
        "StreamKConvBwdWeight.Tree_EndToEnd_Stride2"
        "StreamKConvBwdWeight.Linear_EndToEnd_PureDP"
        "StreamKConvBwdWeight.TreePersistent_EndToEnd_SmallShape"
    )
    list(JOIN
        _ck_tile_grouped_conv_quick_cases ":" _ck_tile_grouped_conv_quick_filter
    )
    add_ck_tile_category_test(
        test_ck_tile_grouped_conv_quick
        test_ck_tile_grouped_conv_bwd_weight_streamk
        "--gtest_filter=${_ck_tile_grouped_conv_quick_filter}"
    )
endif()
