// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/ops/direct_convolution/utils/conv_params.hpp"

using namespace ck_tile::direct_conv;

TEST(ConvParams, ComputeOutputSize_3x3_Pad1)
{
    Conv2dParams par;
    par.n = 1; par.h = 8; par.w = 8;
    par.c_tot = 16; par.k_tot = 16;
    par.kh = 3; par.kw = 3;
    par.pad_h = 1; par.pad_w = 1;
    par.stride_h = 1; par.stride_w = 1;
    par.dilation_h = 1; par.dilation_w = 1;
    par.groups = 4;
    par.compute_output_size();
    EXPECT_EQ(par.p, 8);
    EXPECT_EQ(par.q, 8);
}

TEST(ConvParams, ComputeOutputSize_NoPad)
{
    Conv2dParams par;
    par.n = 1; par.h = 8; par.w = 8;
    par.c_tot = 16; par.k_tot = 16;
    par.kh = 3; par.kw = 3;
    par.pad_h = 0; par.pad_w = 0;
    par.stride_h = 1; par.stride_w = 1;
    par.dilation_h = 1; par.dilation_w = 1;
    par.groups = 4;
    par.compute_output_size();
    EXPECT_EQ(par.p, 6);
    EXPECT_EQ(par.q, 6);
}

TEST(ConvParams, ComputeOutputSize_Stride2)
{
    Conv2dParams par;
    par.n = 1; par.h = 8; par.w = 8;
    par.c_tot = 16; par.k_tot = 16;
    par.kh = 3; par.kw = 3;
    par.pad_h = 1; par.pad_w = 1;
    par.stride_h = 2; par.stride_w = 2;
    par.dilation_h = 1; par.dilation_w = 1;
    par.groups = 4;
    par.compute_output_size();
    EXPECT_EQ(par.p, 4);
    EXPECT_EQ(par.q, 4);
}

TEST(ConvParams, IsValid_Good)
{
    Conv2dParams par;
    par.n = 1; par.h = 8; par.w = 8;
    par.c_tot = 16; par.k_tot = 16;
    par.kh = 3; par.kw = 3;
    par.groups = 4;
    EXPECT_TRUE(par.is_valid());
}

TEST(ConvParams, IsValid_ZeroBatch)
{
    Conv2dParams par;
    par.n = 0; par.h = 8; par.w = 8;
    par.c_tot = 16; par.k_tot = 16;
    par.kh = 3; par.kw = 3;
    par.groups = 4;
    EXPECT_FALSE(par.is_valid());
}

TEST(ConvParams, IsValid_BadGroups)
{
    Conv2dParams par;
    par.n = 1; par.h = 8; par.w = 8;
    par.c_tot = 16; par.k_tot = 16;
    par.kh = 3; par.kw = 3;
    par.groups = 3;  // 16 % 3 != 0
    EXPECT_FALSE(par.is_valid());
}

TEST(ConvParams, IsValid_ZeroGroups)
{
    Conv2dParams par;
    par.n = 1; par.h = 8; par.w = 8;
    par.c_tot = 16; par.k_tot = 16;
    par.kh = 3; par.kw = 3;
    par.groups = 0;
    EXPECT_FALSE(par.is_valid());
}

TEST(ConvParams, ChannelsAndFiltersPerGroup)
{
    Conv2dParams par;
    par.c_tot = 16; par.k_tot = 32; par.groups = 4;
    EXPECT_EQ(par.channels_per_group(), 4);
    EXPECT_EQ(par.filters_per_group(), 8);
}

TEST(ConvParams, DirectionToString)
{
    EXPECT_STREQ(to_string(Direction::Fprop), "Fprop");
    EXPECT_STREQ(to_string(Direction::Dgrad), "Dgrad");
    EXPECT_STREQ(to_string(Direction::Wgrad), "Wgrad");
}

TEST(ConvParams, SizeViewFprop)
{
    Conv2dParams par;
    par.n = 1; par.h = 8; par.w = 10;
    par.c_tot = 16; par.k_tot = 16;
    par.kh = 3; par.kw = 3;
    par.pad_h = 1; par.pad_w = 1;
    par.stride_h = 1; par.stride_w = 1;
    par.dilation_h = 1; par.dilation_w = 1;
    par.groups = 4;
    par.compute_output_size();

    SizeView<Direction::Fprop> view(par);
    EXPECT_EQ(view.h(), par.h);
    EXPECT_EQ(view.w(), par.w);
    EXPECT_EQ(view.p(), par.p);
    EXPECT_EQ(view.q(), par.q);
    EXPECT_EQ(view.pad_h(), par.pad_h);
    EXPECT_EQ(view.pad_w(), par.pad_w);
    EXPECT_EQ(view.stride_h(), par.stride_h);
    EXPECT_EQ(view.stride_w(), par.stride_w);
    EXPECT_EQ(view.dilation_h(), par.dilation_h);
    EXPECT_EQ(view.dilation_w(), par.dilation_w);
}

TEST(ConvParams, SizeViewDgrad)
{
    Conv2dParams par;
    par.n = 1; par.h = 8; par.w = 10;
    par.c_tot = 16; par.k_tot = 16;
    par.kh = 3; par.kw = 3;
    par.pad_h = 1; par.pad_w = 1;
    par.stride_h = 1; par.stride_w = 1;
    par.dilation_h = 1; par.dilation_w = 1;
    par.groups = 4;
    par.compute_output_size();

    SizeView<Direction::Dgrad> view(par);
    // Dgrad swaps h/w with p/q
    EXPECT_EQ(view.h(), par.p);
    EXPECT_EQ(view.w(), par.q);
    EXPECT_EQ(view.p(), par.h);
    EXPECT_EQ(view.q(), par.w);
    // Dgrad padding: (kh-1)*dilation_h - pad_h
    EXPECT_EQ(view.pad_h(), (par.kh - 1) * par.dilation_h - par.pad_h);
    EXPECT_EQ(view.pad_w(), (par.kw - 1) * par.dilation_w - par.pad_w);
    // Dgrad swaps stride and dilation
    EXPECT_EQ(view.stride_h(), par.dilation_h);
    EXPECT_EQ(view.stride_w(), par.dilation_w);
    EXPECT_EQ(view.dilation_h(), par.stride_h);
    EXPECT_EQ(view.dilation_w(), par.stride_w);
}
