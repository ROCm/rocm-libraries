// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_conv_fwd_wcnn_impl.h"
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

#if(defined(ENABLE_SPATIAL_CLUSTER) && defined(ENABLE_MULTI_CHAIN))
    constexpr int ClusterSize = 2;
#elif defined(ENABLE_SPATIAL_CLUSTER)
    constexpr int ClusterSize = 4;
#endif

    // clang-format off
    {
    //                                                           |ShapeType  |FilterType |Dilation |Lds |WaveGroup | EnableSpatialCluster | ClusterSize | TestMask
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
#if defined(ENABLE_SPATIAL_CLUSTER)
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup,  true, ClusterSize, TestMask | 0x20000>();
#else
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, false, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, false, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, false, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, 0,       WaveGroup, false, 0, TestMask | 0x10000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, false, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, false, 0, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, LdsMode, WaveGroup, false, 0, TestMask | 0x80000>();
#endif
    }
    else
    {
#if defined(ENABLE_SPATIAL_CLUSTER)
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, true, ClusterSize, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, true, ClusterSize, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, true, ClusterSize, TestMask | 0x100000>();
#else
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, false, 0, TestMask | 0x10000>();
        // TODO: fix it.
        bool fail_case = WaveGroup && (TestMask == 0x40) && (config.c == 0x40);
        if (fail_case == false)
        {
            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, false, 0, TestMask | 0x20000>();
            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, false, 0, TestMask | 0x40000>();
        }
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, false, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, false, 0, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, false, 0, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, false, 0, TestMask | 0x400000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, false, 0, TestMask | 0x800000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, false, 0, TestMask | 0x1000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, 0,       WaveGroup, false, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_2X2, false, 0,       WaveGroup, false, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_2X2, false, 0,       WaveGroup, false, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, false, 0, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, false, 0, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, false, 0, TestMask | 0x8000000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, false, 0, TestMask | 0x10000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, false, 0, TestMask | 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, false, 0, TestMask | 0x40000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_2X2, false, LdsMode, WaveGroup, false, 0, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_2X2, false, LdsMode, WaveGroup, false, 0, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_2X2, false, LdsMode, WaveGroup, false, 0, TestMask | 0x8000000>();
#endif
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
    //                |SrcType |GPUAccType | LdsMode |TestMask
    pass &= run_test_fmt<half_t,  float,   0x7, 0x1  >();
    pass &= run_test_fmt<bhalf_t, float,   0xf, 0x2  >();
    pass &= run_test_fmt<f8_t,    float,   0x5, 0x4  >();
    pass &= run_test_fmt<bf8_t,   float,   0x6, 0x8  >();
    pass &= run_test_fmt<int8_t,  float,   0xd, 0x10 >();
    pass &= run_test_fmt<int8_t,  int32_t, 0xe, 0x20 >();
    pass &= run_test_fmt<half_t,  half_t,  0xf, 0x40 >();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x7, 0x80 >();
    pass &= run_test_fmt<f8_t,    half_t,  0x3, 0x100>();
    pass &= run_test_fmt<bf8_t,   half_t,  0xb, 0x200>();
    pass &= run_test_fmt<int8_t,  half_t,  0x9, 0x400>();

    //FOR TILE_LOAD test
    // This tests can be passed only when https://github.amd.com/GFX-Modeling/shader_complex_ffm/pull/1284 merged.
    pass &= run_test_fmt<half_t,  half_t,  0x30, 0x800 >();
    pass &= run_test_fmt<half_t,  float,   0x30, 0x800 >();

    // clang-format on
    std::cout << "grouped_conv_fwd_wcnn: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
#endif

// clang-format off
// ===================================================================================================================================
// Test Suites:
// grouped_conv_fwd_wcnn          (175 tests)
// grouped_conv_fwd_wcnn_wg       (175 tests, two of them depends on "config.c == 0x40")
// grouped_conv_fwd_wcnn_wg_sc    (25  tests)
// grouped_conv_fwd_wcnn_wg_sc_mc (25  tests)
// ===================================================================================================================================

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn
// ENABLE_WAVEGROUP, ENABLE_SPATIAL_CLUSTER, ENABLE_MULTI_CHAIN are NOT defined
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f1x1_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_1X1, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f3x3_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0x7, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = float | LdsMode = 0xf
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, bhalf_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_float_s4x2_f1x1_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_1X1, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_float_s4x2_f3x3_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_3X3, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_float_s4x2_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_2X2, false, 0xf, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = float | LdsMode = 0x5
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, f8_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_float_s4x2_f1x1_l_in_acc)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_1X1, false, 0x5, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_float_s4x2_f3x3_l_in_acc)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_3X3, false, 0x5, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_float_s4x2_f2x2_l_in_acc)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_2X2, false, 0x5, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x6
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, bf8_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_float_s4x2_f1x1_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_1X1, false, 0x6, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_float_s4x2_f3x3_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_3X3, false, 0x6, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_float_s4x2_f2x2_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, false, 0x6, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0xd
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, i8_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_float_s4x2_f1x1_l_in_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_1X1, false, 0xd, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_float_s4x2_f3x3_l_in_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_3X3, false, 0xd, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_float_s4x2_f2x2_l_in_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_2X2, false, 0xd, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = int32_t | LdsMode = 0xe
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, i8_i32_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_i32_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_i32_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_i32_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_i32_s4x2_f1x1_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_1X1, false, 0xe, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_i32_s4x2_f3x3_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_3X3, false, 0xe, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_i32_s4x2_f2x2_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, false, 0xe, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0xf
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f1x1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f1x1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f1x1_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f1x1_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f1x1_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f3x3_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f3x3_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f3x3_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0xf, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0xf, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x4_f1x1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s8x4_f1x1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc)
{
    EXPECT_TRUE((
        run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc)
{
    EXPECT_TRUE((
        run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc)
{
    EXPECT_TRUE((
        run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc)
{
    EXPECT_TRUE((
        run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc)
{
    EXPECT_TRUE((
        run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc)
{
    EXPECT_TRUE((
        run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x2_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((
        run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_2X2, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s4x4_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((
        run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_2X2, false, 0x7, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bhalf_bhalf_s8x4_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((
        run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_2X2, false, 0x7, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x3
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, f8_half_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x4_f1x1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s8x4_f1x1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x2_f1x1_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_1X1, false, 0x3, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x4_f1x1_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_1X1, false, 0x3, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s8x4_f1x1_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_1X1, false, 0x3, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x2_f3x3_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_3X3, false, 0x3, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x4_f3x3_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_3X3, false, 0x3, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s8x4_f3x3_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_3X3, false, 0x3, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x2_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, false, 0x3, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s4x4_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, false, 0x3, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, f8_half_s8x4_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, false, 0x3, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = half_t | LdsMode = 0xb
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x4_f1x1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s8x4_f1x1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x2_f1x1_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_1X1, false, 0xb, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x4_f1x1_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_1X1, false, 0xb, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s8x4_f1x1_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_1X1, false, 0xb, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x2_f3x3_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_3X3, false, 0xb, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x4_f3x3_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_3X3, false, 0xb, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s8x4_f3x3_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_3X3, false, 0xb, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x2_f2x2_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_2X2, false, 0xb, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s4x4_f2x2_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_2X2, false, 0xb, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, bf8_half_s8x4_f2x2_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_2X2, false, 0xb, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x9
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, i8_half_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x4_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s8x4_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x2_f1x1_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_1X1, false, 0x9, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x4_f1x1_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_1X1, false, 0x9, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s8x4_f1x1_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_1X1, false, 0x9, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x2_f3x3_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_3X3, false, 0x9, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x4_f3x3_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_3X3, false, 0x9, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s8x4_f3x3_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_3X3, false, 0x9, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x2_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, false, 0x9, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s4x4_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, false, 0x9, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, i8_half_s8x4_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, false, 0x9, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x30 (TILE_LOAD test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f1x1_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f1x1_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f1x1_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f3x3_d_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f3x3_d_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f3x3_d_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f2x2_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f2x2_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f2x2_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f1x1_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f1x1_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f1x1_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f3x3_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f3x3_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f3x3_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x2_f2x2_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s4x4_f2x2_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_half_s8x4_f2x2_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0x30, false, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x30 (TILE_LOAD test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f1x1_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_1X1, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f3x3_d_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, true, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f2x2_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f1x1_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_1X1, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f3x3_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0x30, false, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn, half_float_s4x2_f2x2_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0x30, false, false, 0>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_wg
// ENABLE_WAVEGROUP is DEFINED, ENABLE_SPATIAL_CLUSTER and ENABLE_MULTI_CHAIN are NOT defined
// WaveGroup = true
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f1x1_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_1X1, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f3x3_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0x7, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = float | LdsMode = 0xf
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, bhalf_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_float_s4x2_f1x1_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_1X1, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_float_s4x2_f3x3_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_3X3, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_float_s4x2_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_2X2, false, 0xf, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = float | LdsMode = 0x5
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, f8_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_float_s4x2_f1x1_l_in_acc)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_1X1, false, 0x5, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_float_s4x2_f3x3_l_in_acc)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_3X3, false, 0x5, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_float_s4x2_f2x2_l_in_acc)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_2X2, false, 0x5, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x6
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, bf8_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_float_s4x2_f1x1_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_1X1, false, 0x6, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_float_s4x2_f3x3_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_3X3, false, 0x6, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_float_s4x2_f2x2_l_wei_acc)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_2X2, false, 0x6, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0xd
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, i8_float_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_float_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_float_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_float_s4x2_f1x1_l_in_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_1X1, false, 0xd, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_float_s4x2_f3x3_l_in_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_3X3, false, 0xd, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_float_s4x2_f2x2_l_in_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_2X2, false, 0xd, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = int32_t | LdsMode = 0xe
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, i8_i32_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_i32_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_i32_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_i32_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_i32_s4x2_f1x1_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_1X1, false, 0xe, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_i32_s4x2_f3x3_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_3X3, false, 0xe, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_i32_s4x2_f2x2_l_wei_acc_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_2X2, false, 0xe, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0xf
// NOTE: fail_case = WaveGroup && (TestMask == 0x40) && (config.c == 0x40)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f1x1)
{
    if(config.c == 0x40)
    {
        GTEST_SKIP() << "This test is masked out by config.c as it is a failed one.";
    }
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f1x1)
{
    if(config.c == 0x40)
    {
        GTEST_SKIP() << "This test is masked out by config.c as it is a failed one.";
    }
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f1x1_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f1x1_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f1x1_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f3x3_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f3x3_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f3x3_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0xf, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f2x2_l_in_wei_acc_async)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0xf, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x4_f1x1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s8x4_f1x1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x2_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_2X2, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s4x4_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_2X2, false, 0x7, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bhalf_bhalf_s8x4_f2x2_l_in_wei_acc)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_2X2, false, 0x7, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x3
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x4_f1x1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s8x4_f1x1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x2_f1x1_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_1X1, false, 0x3, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x4_f1x1_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_1X1, false, 0x3, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s8x4_f1x1_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_1X1, false, 0x3, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x2_f3x3_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_3X3, false, 0x3, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x4_f3x3_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_3X3, false, 0x3, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s8x4_f3x3_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_3X3, false, 0x3, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x2_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_2X2, false, 0x3, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s4x4_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_2X2, false, 0x3, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, f8_half_s8x4_f2x2_l_in_wei)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_2X2, false, 0x3, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = half_t | LdsMode = 0xb
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x4_f1x1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s8x4_f1x1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x2_f1x1_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_1X1, false, 0xb, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x4_f1x1_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_1X1, false, 0xb, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s8x4_f1x1_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_1X1, false, 0xb, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x2_f3x3_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_3X3, false, 0xb, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x4_f3x3_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_3X3, false, 0xb, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s8x4_f3x3_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_3X3, false, 0xb, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x2_f2x2_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_2X2, false, 0xb, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s4x4_f2x2_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_2X2, false, 0xb, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, bf8_half_s8x4_f2x2_l_in_wei_async)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_2X2, false, 0xb, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x9
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x2_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x4_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s8x4_f1x1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x2_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x4_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s8x4_f3x3_d)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x2_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x4_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s8x4_f2x2)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x2_f1x1_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_1X1, false, 0x9, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x4_f1x1_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_1X1, false, 0x9, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s8x4_f1x1_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_1X1, false, 0x9, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x2_f3x3_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_3X3, false, 0x9, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x4_f3x3_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_3X3, false, 0x9, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s8x4_f3x3_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_3X3, false, 0x9, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x2_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_2X2, false, 0x9, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s4x4_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_2X2, false, 0x9, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, i8_half_s8x4_f2x2_l_in_async)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_2X2, false, 0x9, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x30 (TILE_LOAD test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f1x1_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f1x1_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f1x1_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f3x3_d_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f3x3_d_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f3x3_d_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f2x2_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f2x2_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f2x2_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f1x1_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f1x1_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f1x1_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f3x3_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f3x3_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f3x3_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x2_f2x2_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s4x4_f2x2_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_2X2, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_half_s8x4_f2x2_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_2X2, false, 0x30, true, false, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x30 (TILE_LOAD test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f1x1_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_1X1, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f3x3_d_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, true, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f2x2_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f1x1_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_1X1, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f3x3_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0x30, true, false, 0>()));
}
TEST(grouped_conv_fwd_wcnn_wg, half_float_s4x2_f2x2_l_in_wei_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_2X2, false, 0x30, true, false, 0>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_wg_sc
// ENABLE_WAVEGROUP and ENABLE_SPATIAL_CLUSTER are DEFINED, ENABLE_MULTI_CHAIN is NOT defined
// WaveGroup = true, ClusterSize = 4
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, half_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = float | LdsMode = 0xf
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, bhalf_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = float | LdsMode = 0x5
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, f8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x6
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, bf8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0xd
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, i8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = int32_t | LdsMode = 0xe
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, i8_i32_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0xf
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, half_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, half_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, half_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, bhalf_bhalf_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, bhalf_bhalf_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, bhalf_bhalf_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x3
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, f8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, f8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, f8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = half_t | LdsMode = 0xb
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, bf8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, bf8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, bf8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x9
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, i8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, i8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, i8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x30 (TILE_LOAD test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, half_half_s4x2_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, half_half_s4x4_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 4>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc, half_half_s8x4_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 4>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x30 (TILE_LOAD test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc, half_float_s4x2_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 4>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_wg_sc_mc
// ENABLE_WAVEGROUP, ENABLE_SPATIAL_CLUSTER, and ENABLE_MULTI_CHAIN are all DEFINED
// WaveGroup = true, ClusterSize = 2
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, half_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = float | LdsMode = 0xf
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, bhalf_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = float | LdsMode = 0x5
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, f8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x6
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, bf8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0xd
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, i8_float_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = int32_t | LdsMode = 0xe
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, i8_i32_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int32_t, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0xf
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, half_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, half_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, half_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x7
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, bhalf_bhalf_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, bhalf_bhalf_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, bhalf_bhalf_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x3
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, f8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, f8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, f8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = half_t | LdsMode = 0xb
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, bf8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, bf8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, bf8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x9
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, i8_half_s4x2_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, i8_half_s4x4_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, i8_half_s8x4_f3x3)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x30 (TILE_LOAD test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, half_half_s4x2_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, half_half_s4x4_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, true, 2>()));
}
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, half_half_s8x4_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, true, 2>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x30 (TILE_LOAD test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_wg_sc_mc, half_float_s4x2_f3x3_tl)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, Shape_4X2, Filter_3X3, false, 0, true, true, 2>()));
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
