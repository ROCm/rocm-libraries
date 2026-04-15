# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(test_warp_decode test_warp_decode.cpp)

if(CK_USE_OCP_FP8)
    target_compile_options(test_warp_decode PRIVATE -DCK_TILE_USE_OCP_FP8)
endif()
