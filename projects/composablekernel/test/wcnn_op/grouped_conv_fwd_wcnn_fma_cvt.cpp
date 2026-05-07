// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_conv_fwd_wcnn_fma_cvt_impl.h"
#include <gtest/gtest.h>

#if 0
template <typename SrcType, typename GPUAccType, int LdsMode, uint32_t TestMask>
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
    //                                                          |ShapeType |Lds |WaveGroup |FmaMode | ActiveFunc | CvtToTensor | TestMask
    if constexpr(std::is_same<GPUAccType, float>::value)
    {
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 1, 0, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 0, 0, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 0, 1, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 1, 1, 0, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, LdsMode, WaveGroup, 1, 0, 0, TestMask | 0x20000>();
    }
    else
    {
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 1, 0, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, 0,       WaveGroup, 1, 0, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, 0,       WaveGroup, 1, 0, 0, TestMask | 0x40000>();

        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, LdsMode, WaveGroup, 1, 0, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, LdsMode, WaveGroup, 1, 0, 0, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, LdsMode, WaveGroup, 1, 0, 0, TestMask | 0x200000>();
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
    // Ds keep same with acc currently
    //                |SrcType |GPUAccType| LdsMode TestMask
    pass &= run_test_fmt<half_t,  float,   0x17, 0x1  >();
    pass &= run_test_fmt<bhalf_t, float,   0x1f, 0x2  >();
    pass &= run_test_fmt<f8_t,    float,   0x17, 0x4  >();
    pass &= run_test_fmt<bf8_t,   float,   0x1f, 0x8  >();
    pass &= run_test_fmt<int8_t,  float,   0x17, 0x10 >();

    pass &= run_test_fmt<half_t,  half_t,  0x1f, 0x40 >();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x17, 0x80 >();
    pass &= run_test_fmt<f8_t,    half_t,  0x1f, 0x100>();
    pass &= run_test_fmt<bf8_t,   half_t,  0x17, 0x200>();
    pass &= run_test_fmt<int8_t,  half_t,  0x1f, 0x400>();
    //
    // clang-format on
    std::cout << "grouped_conv_fwd_wcnn_fma: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}

template <typename SrcType, typename GPUAccType, int LdsMode, uint32_t TestMask>
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
    //                                                                 |ShapeType |Lds |WaveGroup | FmaMode | ActiveFunc | CvtToTensor | TestMask
    if constexpr(std::is_same<GPUAccType, float>::value)
    {
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 1, 0, 1, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 0, 0, 1, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 0, 1, 1, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 1, 1, 1, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, LdsMode, WaveGroup, 1, 0, 1, TestMask | 0x20000>();
    }
    else
    {
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 1, 0, 1, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, 0,       WaveGroup, 1, 0, 1, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, 0,       WaveGroup, 1, 0, 1, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, LdsMode, WaveGroup, 1, 0, 1, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, LdsMode, WaveGroup, 1, 0, 1, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, LdsMode, WaveGroup, 1, 0, 1, TestMask | 0x200000>();
    }
    // clang-format on
    return pass;
}

using half_t  = ck::half_t;
using bhalf_t = ck::bhalf_t;
using f8_t    = ck::f8_t;
using bf8_t   = ck::bf8_t;

int main(int argc, char* argv[])
{
    bool pass = true;
    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    // Ds keep same with acc currently
    //                |SrcType |GPUAccType| LdsMode TestMask
    pass &= run_test_fmt<half_t,  float,   0x1f, 0x1  >();
    pass &= run_test_fmt<bhalf_t, float,   0x17, 0x2  >();
    pass &= run_test_fmt<f8_t,    float,   0x1f, 0x4  >();
    pass &= run_test_fmt<bf8_t,   float,   0x17, 0x8  >();
    pass &= run_test_fmt<int8_t,  float,   0x1f, 0x10 >();
    pass &= run_test_fmt<half_t,  half_t,  0x17, 0x40 >();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x1f, 0x80 >();
    pass &= run_test_fmt<f8_t,    half_t,  0x17, 0x100>();
    pass &= run_test_fmt<bf8_t,   half_t,  0x1f, 0x200>();
    pass &= run_test_fmt<int8_t,  half_t,  0x17, 0x400>();
    //
    // clang-format on
    std::cout << "grouped_conv_fwd_wcnn_fma_cvt: ..... " << (pass ? "SUCCESS" : "FAILURE")
              << std::endl;
    return pass ? 0 : 1;
}
#endif

// clang-format off
// ===================================================================================================================================
// Test Suites:
// grouped_conv_fwd_wcnn_fma        (55 cases)
// grouped_conv_fwd_wcnn_fma_wg     (55 cases)
// grouped_conv_fwd_wcnn_fma_cvt    (55 cases)
// grouped_conv_fwd_wcnn_fma_cvt_wg (55 cases)
// ===================================================================================================================================

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_fma
// ENABLE_WAVEGROUP is NOT defined
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, half_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, half_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, half_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, half_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, half_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0x17, false, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, bhalf_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bhalf_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bhalf_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bhalf_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bhalf_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0x1f, false, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, f8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, f8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, f8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, f8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, f8_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0x17, false, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, bf8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bf8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bf8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bf8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bf8_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0x1f, false, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, i8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, i8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, i8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, i8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, i8_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0x17, false, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, half_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, half_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, half_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_8X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, half_half_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X2, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, half_half_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X4, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, half_half_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_8X4, 0x1f, false, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, bhalf_bhalf_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bhalf_bhalf_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bhalf_bhalf_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bhalf_bhalf_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bhalf_bhalf_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bhalf_bhalf_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, 0x17, false, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, f8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, f8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, f8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_8X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, f8_half_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X2, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, f8_half_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X4, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, f8_half_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_8X4, 0x1f, false, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = half_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, bf8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bf8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bf8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bf8_half_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bf8_half_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, bf8_half_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, 0x17, false, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma, i8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X2, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, i8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, i8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_8X4, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, i8_half_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X2, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, i8_half_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X4, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma, i8_half_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_8X4, 0x1f, false, 1, 0, 0>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_fma_wg
// ENABLE_WAVEGROUP is DEFINED
// WaveGroup = true
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, half_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, half_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, true, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, half_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, half_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, half_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0x17, true, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, true, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0x1f, true, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, true, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0x17, true, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, true, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0x1f, true, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, true, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0x17, true, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, half_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, half_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, half_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_8X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, half_half_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X2, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, half_half_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X4, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, half_half_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_8X4, 0x1f, true, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_bhalf_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_bhalf_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_bhalf_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_bhalf_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, 0x17, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_bhalf_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, 0x17, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bhalf_bhalf_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, 0x17, true, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_8X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_half_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X2, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_half_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X4, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, f8_half_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_8X4, 0x1f, true, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = half_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_half_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, 0x17, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_half_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, 0x17, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, bf8_half_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, 0x17, true, 1, 0, 0>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X2, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_8X4, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_half_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X2, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_half_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X4, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_fma_wg, i8_half_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_8X4, 0x1f, true, 1, 0, 0>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_fma_cvt
// ENABLE_WAVEGROUP is NOT defined
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0x1f, false, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0x17, false, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0x1f, false, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0x17, false, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0x1f, false, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_8X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_half_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X2, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_half_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X4, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, half_half_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_8X4, 0x17, false, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_bhalf_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_bhalf_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_bhalf_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_bhalf_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_bhalf_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bhalf_bhalf_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, 0x1f, false, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_8X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_half_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X2, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_half_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X4, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, f8_half_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_8X4, 0x17, false, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = half_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_half_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_half_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, bf8_half_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, 0x1f, false, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X2, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_8X4, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_half_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X2, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_half_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X4, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt, i8_half_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_8X4, 0x17, false, 1, 0, 1>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_fma_cvt_wg
// ENABLE_WAVEGROUP is DEFINED
// WaveGroup = true
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, true, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, true, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, float, half_t, Shape_4X2, 0x1f, true, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, true, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, true, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, 0x17, true, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, true, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, true, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, float, f8_t, Shape_4X2, 0x1f, true, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, true, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, true, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_float_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, float, bf8_t, Shape_4X2, 0x17, true, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_float_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_float_s4x2_fma0_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, true, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_float_s4x2_fma0_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, true, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_float_s4x2_fma1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_float_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, float, int8_t, Shape_4X2, 0x1f, true, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_8X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_half_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X2, 0x17, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_half_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_4X4, 0x17, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, half_half_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, half_t, Shape_8X4, 0x17, true, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_bhalf_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_bhalf_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_bhalf_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_bhalf_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_bhalf_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bhalf_bhalf_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, 0x1f, true, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_8X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_half_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X2, 0x17, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_half_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_4X4, 0x17, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, f8_half_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, f8_t, half_t, f8_t, Shape_8X4, 0x17, true, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = half_t | LdsMode = 0x1f
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_half_s4x2_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_half_s4x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, bf8_half_s8x4_l_in_wei_acc_ds_async_fma1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, 0x1f, true, 1, 0, 1>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x17
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_half_s4x2_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X2, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_half_s4x4_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_half_s8x4_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_8X4, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_half_s4x2_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X2, 0x17, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_half_s4x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_4X4, 0x17, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_fma_cvt_wg, i8_half_s8x4_l_in_wei_acc_ds_fma1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, int8_t, half_t, int8_t, Shape_8X4, 0x17, true, 1, 0, 1>()));
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
