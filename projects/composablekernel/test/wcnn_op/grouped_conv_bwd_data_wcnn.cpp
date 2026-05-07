// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_conv_bwd_data_wcnn_impl.h"
#include <gtest/gtest.h>

#if 0
template <typename SrcType, typename GPUAccType, int LdsMode, int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;

#ifdef ENABLE_WAVEGROUP
    constexpr bool WaveGroup = true;
#else
    constexpr bool WaveGroup = false;
#endif
    // clang-format off
    {
    //                                                           |ShapeType  |FilterType |ShuffleOnLoad |Lds |WaveGroup |TestMask
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, 0,       WaveGroup, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, true,  0,       WaveGroup, TestMask | 0x40000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, LdsMode, WaveGroup, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, true,  LdsMode, WaveGroup, TestMask | 0x80000>();
    }
    else
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, 0,       WaveGroup, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_2X2, false, 0,       WaveGroup, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_2X2, false, 0,       WaveGroup, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, true,  0,       WaveGroup, TestMask | 0x400000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_2X2, true,  0,       WaveGroup, TestMask | 0x800000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_2X2, true,  0,       WaveGroup, TestMask | 0x1000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_2X2, false, LdsMode, WaveGroup, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, LdsMode, WaveGroup, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_2X2, false, LdsMode, WaveGroup, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, true,  LdsMode, WaveGroup, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_2X2, true,  LdsMode, WaveGroup, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_2X2, true,  LdsMode, WaveGroup, TestMask | 0x8000000>();
    }

    }
    // clang-format on
    return pass;
}

int main(int argc, char* argv[])
{
    bool pass = true;

    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    //                |SrcType |GPUAccType|LdsMode |TestMask
    pass &= run_test_fmt<half_t,  float,   0x7, 0x1>();
    pass &= run_test_fmt<bf8_t,   float,   0x6, 0x8>();
    pass &= run_test_fmt<int8_t,  int32_t, 0xe, 0x20>();
    pass &= run_test_fmt<half_t,  half_t,  0xf, 0x40>();
    pass &= run_test_fmt<f8_t,    half_t,  0x3, 0x100>();
    pass &= run_test_fmt<int8_t,  half_t,  0x9, 0x400>();
    // clang-format on

    std::cout << "grouped_conv_bwd_data_wcnn: ..... " << (pass ? "SUCCESS" : "FAILURE")
              << std::endl;
    return pass ? 0 : 1;
}
#endif

// clang-format off
// ===================================================================================================================================
// Test Suites:
// grouped_conv_bwd_data_wcnn    (48 cases)
// grouped_conv_bwd_data_wcnn_wg (48 cases)
// ===================================================================================================================================

// ===================================================================================================================================
// Test Suite: grouped_conv_bwd_data_wcnn
// ENABLE_WAVEGROUP is NOT defined
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn, half_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_float_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_float_s4x2_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0x7, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_float_s4x2_f2x2_sol_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, true, 0x7, false>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x6
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn, bf8_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, bf8_float_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, bf8_float_s4x2_f2x2_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, false, 0x6, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, bf8_float_s4x2_f2x2_sol_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, true, 0x6, false>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = int32_t | LdsMode = 0xe
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn, int8_int32_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_int32_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_int32_s4x2_f2x2_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, false, 0xe, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_int32_s4x2_f2x2_sol_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, true, 0xe, false>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0xf
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn, half_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s4x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s8x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s8x4_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0xf, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s4x2_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0xf, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s4x4_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0xf, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s4x2_f2x2_sol_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, true, 0xf, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s4x4_f2x2_sol_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, true, 0xf, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, half_half_s8x4_f2x2_sol_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, true, 0xf, false>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x3
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn, f8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s4x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s8x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s8x4_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, false, 0x3, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s4x2_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, false, 0x3, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s4x4_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, false, 0x3, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s4x2_f2x2_sol_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, true, 0x3, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s4x4_f2x2_sol_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, true, 0x3, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, f8_half_s8x4_f2x2_sol_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, true, 0x3, false>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x9
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn, int8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, false, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s4x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s8x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, true, 0, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s8x4_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, false, 0x9, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s4x2_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, false, 0x9, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s4x4_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, false, 0x9, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s4x2_f2x2_sol_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, true, 0x9, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s4x4_f2x2_sol_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, true, 0x9, false>()));
}
TEST(grouped_conv_bwd_data_wcnn, int8_half_s8x4_f2x2_sol_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, true, 0x9, false>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_bwd_data_wcnn_wg
// ENABLE_WAVEGROUP is defined
// WaveGroup = true
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn_wg, half_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_float_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_float_s4x2_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0x7, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_float_s4x2_f2x2_sol_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, true, 0x7, true>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x6
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn_wg, bf8_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, bf8_float_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, bf8_float_s4x2_f2x2_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, false, 0x6, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, bf8_float_s4x2_f2x2_sol_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, true, 0x6, true>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = int32_t | LdsMode = 0xe
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn_wg, int8_int32_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_int32_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_int32_s4x2_f2x2_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, false, 0xe, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_int32_s4x2_f2x2_sol_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, true, 0xe, true>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0xf
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s4x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s8x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s8x4_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0xf, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s4x2_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0xf, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s4x4_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0xf, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s4x2_f2x2_sol_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, true, 0xf, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s4x4_f2x2_sol_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, true, 0xf, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, half_half_s8x4_f2x2_sol_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, true, 0xf, true>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x3
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s4x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s8x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s8x4_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, false, 0x3, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s4x2_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, false, 0x3, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s4x4_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, false, 0x3, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s4x2_f2x2_sol_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, true, 0x3, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s4x4_f2x2_sol_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, true, 0x3, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, f8_half_s8x4_f2x2_sol_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, true, 0x3, true>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x9
// ==================================================================================
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, false, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s4x2_f2x2_sol)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s4x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s8x4_f2x2_sol)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, true, 0, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s8x4_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, false, 0x9, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s4x2_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, false, 0x9, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s4x4_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, false, 0x9, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s4x2_f2x2_sol_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, true, 0x9, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s4x4_f2x2_sol_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, true, 0x9, true>()));
}
TEST(grouped_conv_bwd_data_wcnn_wg, int8_half_s8x4_f2x2_sol_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, true, 0x9, true>()));
}

// clang-format on

// ===================================================================================================================================
// The main function
// ===================================================================================================================================
int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    if(parse_cmd_args_gtest(argc, argv, config) == false)
    {
        return -1;
    }

    return RUN_ALL_TESTS();
}
