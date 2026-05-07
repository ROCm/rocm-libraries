// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_conv_fwd_wcnn_sba_cvt_impl.h"
#include <gtest/gtest.h>

#if 0
template <typename SrcType, typename GPUAccType, int LdsMode, int SbaMode, uint32_t TestMask>
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
    //                                                     |ShapeType  |FilterType |Dilation |Lds |WaveGroup | SbaMode | ActiveFun | CvtToTensor | TestMask
    if constexpr(std::is_same<GPUAccType, float>::value)
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x1000>();

        //Tan (only for sba/uba)
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x20000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x4000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x80000>();
    }
    else
    {
        // 4X4 issue
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x400>();

        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x2000>();

        //ActivativeFun: 2
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x4000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x8000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x8000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x2000>();
// 8x4
        //ActivativeFun: 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x40000>();
        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x200000>();

        //ActivativeFun: 2
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x400000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x800000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x800000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x200000>();

// 4X2
        //NoneActFun
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x1000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_2X2, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x4000000>();

        //Relu
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x8000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x10000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x10000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x20000000>();

        //Tan (only for sba/uba)
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x40000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x80000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x80000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x20000000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x20000000>();
    }
    // clang-format on
    return pass;
}

int main(int argc, char* argv[])
{
    // To refactor on only Acc=32bit supported on 4x2 and Acc=16bit supported on 4x2/4x4/8x4
    bool pass = true;
    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    // Ds keep same with acc currently
    //            |SrcType |GPUAccType |LdsMode |SbaMode |TestMask
    pass &= run_test_fmt<half_t,  float,   0x17, 0, 0x1>();
#if !defined(ENABLE_WAVEGROUP)
    pass &= run_test_fmt<half_t,  float,   0x17, 1, 0x2>();
    pass &= run_test_fmt<half_t,  float,   0x17, 2, 0x4>();
    pass &= run_test_fmt<half_t,  half_t,  0x1f, 0, 0x8>();
#endif
    pass &= run_test_fmt<half_t,  half_t,  0x1f, 1, 0x10>();
#if !defined(ENABLE_WAVEGROUP)
    pass &= run_test_fmt<half_t,  half_t,  0x1f, 2, 0x20>();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x17, 0, 0x40>();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x17, 1, 0x80>();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x17, 2, 0x80>();
#endif
    // clang-format on
    std::cout << "conv_device_suba: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}

template <typename SrcType, typename GPUAccType, int LdsMode, int SbaMode, uint32_t TestMask>
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
    // TODO: verify cvtTensorScale != 0
    // clang-format off
    //                                                     |ShapeType  |FilterType |Dilation |Lds |WaveGroup | SbaMode | activeFun | CvtToTensor | TestMask
    if constexpr(std::is_same<GPUAccType, float>::value)
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x100>();
        // duplicate: pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x1000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x4000>();
    }
    else
    {
// 4X4
        //ActivativeFun: 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x400>();

        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x4000>();

// 8x4
        //ActivativeFun: 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x40000>();

        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x400000>();

// 4X2
        //ActivativeFun: 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x1000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x4000000>();

        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x10000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x40000000>();

        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x8000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x800000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x8000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x80000000>();
    }
    // clang-format on
    return pass;
}

using half_t  = ck::half_t;
using bhalf_t = ck::bhalf_t;
using f8_t    = ck::f8_t;
using bf8_t   = ck::bf8_t;
using int4_t  = ck::int4_t;
using uint4_t = ck::uint4_t;

int main(int argc, char* argv[])
{
    bool pass = true;
    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    // Ds keep same with acc currently
    //                |SrcType |GPUAccType |LdsMode | SbaMode |TestMask
    pass &= run_test_fmt<half_t,   float,   0x1f, 1, 0x1>();
#if !defined(ENABLE_WAVEGROUP)
    pass &= run_test_fmt<bhalf_t,  float,   0x17, 1, 0x2>();
    pass &= run_test_fmt<int8_t,   float,   0x17, 1, 0x4>();
    pass &= run_test_fmt<bf8_t,    float,   0x17, 1, 0x8>();
    pass &= run_test_fmt<f8_t,     float,   0x17, 1, 0x8>();
#endif
    pass &= run_test_fmt<f8_t,     half_t,  0x1f, 1, 0x10>();
#if !defined(ENABLE_WAVEGROUP)
    pass &= run_test_fmt<int8_t,   half_t,  0x1f, 1, 0x20>();
    pass &= run_test_fmt<half_t,   half_t,  0x1f, 1, 0x40>();
    pass &= run_test_fmt<bf8_t,    half_t,  0x1f, 1, 0x80>();
    pass &= run_test_fmt<bhalf_t,  bhalf_t, 0x1f, 1, 0x80>();

     // For tiled_store
     pass &= run_test_fmt<int8_t,   float,   0x80, 0, 0x4>();
     pass &= run_test_fmt<half_t,   float,   0x80, 0, 0x1>();
     pass &= run_test_fmt<half_t,   half_t,  0x80, 0, 0x40>();
     pass &= run_test_fmt<int8_t,   half_t,  0x80, 0, 0x20>();
#endif
    // clang-format on
    std::cout << "conv_device_suba_cvt: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
#endif

// clang-format off
// ===================================================================================================================================
// Test Suites:
// grouped_conv_fwd_wcnn_sba        (318 tests)
// grouped_conv_fwd_wcnn_sba_wg     (60  tests)
// grouped_conv_fwd_wcnn_sba_cvt    (273 tests)
// grouped_conv_fwd_wcnn_sba_cvt_wg (39  tests)
// ===================================================================================================================================

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_sba
// ENABLE_WAVEGROUP is NOT defined
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x17 | SbaMode = 0
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_d_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_d_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_d_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x17, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x17, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, false, 0, 2, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x17 | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x17, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_l_in_wei_acc_ds_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x17, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 2, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x17 | SbaMode = 2
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_d_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_d_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_d_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_l_in_wei_acc_ds_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x17, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f1x1_l_in_wei_acc_ds_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x17, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, false, 2, 2, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x1f | SbaMode = 0
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_d_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_d_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_d_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_d_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_d_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_d_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_d_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f2x2_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_d_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_d_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 0, 2, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f2x2_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 2, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x1f | SbaMode = 2
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_d_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_d_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_d_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_d_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_d_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_d_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_d_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f2x2_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_d_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_d_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba2_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba2_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba2_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 2, 2, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x17 | SbaMode = 0
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_d_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x17, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_d_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x17, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_d_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x17, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_d_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x17, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_d_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x17, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_d_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x17, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_d_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f2x2_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_2X2, false, 0, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_d_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_d_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x17, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x17, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x17, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x17, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x17, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x17, false, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_sba0_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 0, 2, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x17 | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x17, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x17, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x17, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x17, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f2x2_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_2X2, false, 0, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x17, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x17, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x17, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x17, false, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_sba1_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 2, 0>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x17 | SbaMode = 2
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_d_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x17, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_d_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x17, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_d_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x17, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_d_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x17, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_d_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x17, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_d_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x17, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_d_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f2x2_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_2X2, false, 0, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_d_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_d_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x17, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x17, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x17, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x17, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x17, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x17, false, 2, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_sba2_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 2, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_sba2_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 2, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_sba2_af2)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 2, 2, 0>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_sba_wg
// ENABLE_WAVEGROUP is DEFINED
// WaveGroup = true
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x17 | SbaMode = 0
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f1x1_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, true, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f3x3_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, true, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f3x3_d_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, true, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f3x3_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f3x3_d_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f1x1_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f1x1_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, true, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f3x3_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, true, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f3x3_d_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, true, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba0_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, true, 0, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f1x1_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x17, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba0_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, true, 0, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f1x1_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x17, true, 0, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_float_s4x2_f3x3_l_in_wei_acc_ds_sba0_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x17, true, 0, 2, 0>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f2x2_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_2X2, false, 0, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f1x1_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f3x3_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f3x3_d_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, true, 1, 2, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, true, 1, 0, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, true, 1, 1, 0>()));
}
TEST(grouped_conv_fwd_wcnn_sba_wg, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af2)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, true, 1, 2, 0>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_sba_cvt
// ENABLE_WAVEGROUP is NOT defined
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = float | LdsMode = 0x17 | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_float_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_float_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_float_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_float_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_float_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_float_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_float_s4x2_f1x1_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, Filter_1X1, false, 0x17, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, float, bhalf_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0x17 | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f1x1_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_1X1, false, 0x17, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = float | LdsMode = 0x17 | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_float_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, bf8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_float_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, bf8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_float_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, bf8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_float_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, bf8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_float_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, bf8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_float_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, bf8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, bf8_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_float_s4x2_f1x1_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, bf8_t, Shape_4X2, Filter_1X1, false, 0x17, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, float, bf8_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = float | LdsMode = 0x17 | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_float_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, f8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_float_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, f8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_float_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, f8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_float_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, f8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_float_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, f8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_float_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, f8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, f8_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_float_s4x2_f1x1_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, f8_t, Shape_4X2, Filter_1X1, false, 0x17, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_float_s4x2_f3x3_l_in_wei_acc_ds_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, float, f8_t, Shape_4X2, Filter_3X3, false, 0x17, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, f8_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bf8_t | GPUAccType = half_t | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bf8_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bf8_t, bf8_t, half_t, bf8_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = bhalf_t | GPUAccType = bhalf_t | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, true, 0, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_1X1, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_8X4, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, bhalf_bhalf_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<bhalf_t, bhalf_t, bhalf_t, bhalf_t, Shape_4X2, Filter_3X3, false, 0x1f, false, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = float | LdsMode = 0x80 | SbaMode = 0 (TILE STORE test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f1x1_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_d_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_d_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f1x1_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f1x1_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_1X1, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_float_s4x2_f3x3_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, float, int8_t, Shape_4X2, Filter_3X3, false, 0x80, false, 0, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x80 | SbaMode = 0 (TILE STORE test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f1x1_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_d_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_d_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f1x1_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f1x1_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_float_s4x2_f3x3_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x80, false, 0, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = half_t | LdsMode = 0x80 | SbaMode = 0 (TILE STORE test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f1x1_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_d_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f1x1_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f1x1_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_d_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f1x1_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_1X1, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f1x1_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_d_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f1x1_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f1x1_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_d_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f1x1_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_1X1, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f1x1_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_d_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f1x1_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f1x1_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_d_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f1x1_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_1X1, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x4_f3x3_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X4, Filter_3X3, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s8x4_f3x3_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_8X4, Filter_3X3, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, half_half_s4x2_f3x3_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<half_t, half_t, half_t, half_t, Shape_4X2, Filter_3X3, false, 0x80, false, 0, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = int8_t | GPUAccType = half_t | LdsMode = 0x80 | SbaMode = 0 (TILE STORE test)
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f1x1_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_d_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f1x1_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_1X1, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f1x1_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_1X1, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_d_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, true, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f1x1_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_1X1, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f1x1_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_d_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f1x1_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_1X1, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f1x1_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_1X1, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_d_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, true, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f1x1_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_1X1, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f1x1_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_d_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f1x1_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_1X1, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f1x1_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_1X1, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, false, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_d_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, true, 0, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f1x1_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_1X1, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x4_f3x3_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X4, Filter_3X3, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s8x4_f3x3_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_8X4, Filter_3X3, false, 0x80, false, 0, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_l_sba0_af0_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, false, 0x80, false, 0, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt, i8_half_s4x2_f3x3_l_sba0_af1_ts)
{
    EXPECT_TRUE((run_test<int8_t, int8_t, half_t, int8_t, Shape_4X2, Filter_3X3, false, 0x80, false, 0, 1, 1>()));
}

// ===================================================================================================================================
// Test Suite: grouped_conv_fwd_wcnn_sba_cvt_wg
// ENABLE_WAVEGROUP is DEFINED
// WaveGroup = true
// ===================================================================================================================================

// ==================================================================================
// Test Set: SrcType = half_t | GPUAccType = float | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, half_float_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, half_float_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, half_float_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, half_float_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, half_float_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, true, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, half_float_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, half_float_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, half_float_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_1X1, false, 0x1f, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, half_float_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<half_t, half_t, float, half_t, Shape_4X2, Filter_3X3, false, 0x1f, true, 1, 1, 1>()));
}

// ==================================================================================
// Test Set: SrcType = f8_t | GPUAccType = half_t | LdsMode = 0x1f | SbaMode = 1
// ==================================================================================
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_1X1, false, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, false, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, true, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_1X1, false, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_1X1, false, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, false, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, true, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_1X1, false, 0x1f, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_1X1, false, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, false, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, true, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_1X1, false, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_1X1, false, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, false, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, true, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_1X1, false, 0x1f, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f1x1_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_1X1, false, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f3x3_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, false, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f3x3_d_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, true, 0, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_1X1, false, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f1x1_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_1X1, false, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f3x3_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, false, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f3x3_d_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, true, 0, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f1x1_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_1X1, false, 0x1f, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, false, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X4, Filter_3X3, false, 0x1f, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, false, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s8x4_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_8X4, Filter_3X3, false, 0x1f, true, 1, 1, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af0)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, false, 0x1f, true, 1, 0, 1>()));
}
TEST(grouped_conv_fwd_wcnn_sba_cvt_wg, f8_half_s4x2_f3x3_l_in_wei_acc_ds_async_sba1_af1)
{
    EXPECT_TRUE((run_test<f8_t, f8_t, half_t, f8_t, Shape_4X2, Filter_3X3, false, 0x1f, true, 1, 1, 1>()));
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
