# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# T2-01: block-diagonal WMMA packing (GroupsPerWmma > 1) for small-channels-per-group
# grouped_conv2d_fwd. See docs_gfx1250/T2-01_BLOCK_DIAGONAL_WMMA_PACKING_DESIGN.md.
if(GPU_TARGETS MATCHES "gfx12")
    add_example_executable(example_grouped_conv2d_fwd_wmma_block_diagonal_fp16
                            grouped_conv2d_fwd_wmma_block_diagonal_fp16.cpp)
endif()
