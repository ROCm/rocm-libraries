# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Task 6 (docs_gfx1250/GFX1250_WRW_REMAINING_TASKS.md): DEVICE vs SYSTEM atomic
# scope stress test for wrw split-K buffer atomic-add.
if(GPU_TARGETS MATCHES "gfx12")
    add_example_executable(example_grouped_conv_bwd_weight_wmma_atomic_stress_bf16
                            grouped_conv_bwd_weight_wmma_atomic_stress_bf16.cpp)
endif()
