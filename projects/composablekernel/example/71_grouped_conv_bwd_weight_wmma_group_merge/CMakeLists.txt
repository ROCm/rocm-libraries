# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Task 4 (docs_gfx1250/GFX1250_WRW_REMAINING_TASKS.md): validates extending the existing
# NumGroupsToMerge block-diagonal group-packing mechanism (already implemented in
# TransformConvBwdWeightToGemmV2 / DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3) to
# depthwise grouped_conv2d_bwd_weight shapes.
if(GPU_TARGETS MATCHES "gfx12")
    add_example_executable(example_grouped_conv_bwd_weight_wmma_group_merge_bf16
                            grouped_conv_bwd_weight_wmma_group_merge_bf16.cpp)
endif()
