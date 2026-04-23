# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message("-- Adding: test/ck_tile/direct_conv/")

# Unit tests (CPU-only, compile with HIP for type support)
add_gtest_executable(test_direct_conv_mathutil test_mathutil.cpp)
add_gtest_executable(test_direct_conv_params test_conv_params.cpp)
add_gtest_executable(test_direct_conv_matrix_layout test_matrix_layout.cpp)
add_gtest_executable(test_direct_conv_swizzle test_swizzle.cpp)
add_gtest_executable(test_direct_conv_transpose_lds test_transpose_lds_layout.cpp)

# Integration test (gfx950 / CDNA4 only)
if(GPU_TARGETS MATCHES "gfx950")
    add_gtest_executable(test_direct_conv_grouped_4c_fp16 test_grouped_4c_fp16.cpp)
    # The direct conv kernels are ported from hipconv and use C-style casts, have unused
    # parameters, and other patterns that trigger -Weverything. Suppress these warnings for now.
    # TODO: Fix the code such that warnign can be re-enabled.
    if(TARGET test_direct_conv_grouped_4c_fp16)
        target_compile_options(test_direct_conv_grouped_4c_fp16 PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    add_gtest_executable(test_direct_conv_grouped_4c_fp16_tile_conv test_grouped_4c_fp16_tile_conv.cpp)
    if(TARGET test_direct_conv_grouped_4c_fp16_tile_conv)
        target_compile_options(test_direct_conv_grouped_4c_fp16_tile_conv PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    add_gtest_executable(test_direct_conv_grouped_16c_fp16 test_grouped_16c_fp16.cpp)
    if(TARGET test_direct_conv_grouped_16c_fp16)
        target_compile_options(test_direct_conv_grouped_16c_fp16 PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

endif()
