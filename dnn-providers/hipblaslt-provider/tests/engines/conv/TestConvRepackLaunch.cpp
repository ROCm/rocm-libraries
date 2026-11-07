// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>

#include <gtest/gtest.h>

#include "kernels/ConvRepackLaunch.hpp"

/**
 * @file TestConvRepackLaunch.cpp
 * @brief Which repack kernel each layout gets, and its launch shape: the rows, flat and tile
 *        im2col kernels, the rows and channels-first col2im kernels, the generic fallbacks
 *        past 32-bit offsets, and the tile shrinking until its halo fits LDS. No device is used.
 */
namespace
{

using namespace hipblaslt_plugin;

/// A 2-D convolution of n x c x h x w, 3x3 pad 1, in one layout, columns following it.
ConvColumnGeometry geometry(long long n, long long c, long long h, long long w, bool channelsLast)
{
    ConvColumnGeometry g{};
    g.n = n;
    g.groups = 1;
    g.cPerGroup = c;
    g.input[0] = g.output[0] = g.filter[0] = 1;
    g.input[1] = g.output[1] = h;
    g.input[2] = g.output[2] = w;
    g.filter[1] = g.filter[2] = 3;
    g.prePadding[1] = g.prePadding[2] = 1;
    for(int axis = 0; axis < 3; ++axis)
    {
        g.stride[axis] = g.dilation[axis] = 1;
    }
    if(channelsLast)
    {
        g.xStrides[1] = 1;
        g.xStrides[4] = c;
        g.xStrides[3] = c * w;
        g.xStrides[0] = c * w * h;
    }
    else
    {
        g.xStrides[4] = 1;
        g.xStrides[3] = w;
        g.xStrides[1] = h * w;
        g.xStrides[0] = c * h * w;
    }
    g.channelsInner = channelsLast ? 1 : 0;
    g.columnsFirst = channelsLast ? 0 : 1;
    return g;
}

alignas(16) unsigned char g_buffer[64];

Im2ColArgs im2col(const ConvColumnGeometry& g)
{
    Im2ColArgs args{};
    args.x = g_buffer;
    args.col = g_buffer;
    args.geometry = g;
    args.total = g.n * g.output[1] * g.output[2] * g.cPerGroup * 9;
    return args;
}

Col2ImArgs col2im(const ConvColumnGeometry& g)
{
    Col2ImArgs args{};
    args.col = reinterpret_cast<const float*>(g_buffer);
    args.x = g_buffer;
    args.geometry = g;
    args.total = g.n * g.cPerGroup * g.input[1] * g.input[2];
    return args;
}

TEST(TestConvRepackLaunch, ChannelsLastIm2colIsVectorizedRows)
{
    auto args = im2col(geometry(8, 256, 14, 14, true));
    const auto launch = im2colLaunch(args, 2);
    EXPECT_EQ(launch.entry, "Im2ColRows2V8");
    EXPECT_EQ(args.launch.lanes, 32); // 256 channels / 8 per vector
    EXPECT_EQ(args.launch.rows, 8);
    EXPECT_EQ(launch.grid, (8u * 14 * 14 + 7) / 8);
}

TEST(TestConvRepackLaunch, MisalignedChannelsFallBackToScalarRows)
{
    auto args = im2col(geometry(8, 20, 14, 14, true)); // 20 % 8 != 0
    EXPECT_EQ(im2colLaunch(args, 2).entry, "Im2ColRows2V1");
    args = im2col(geometry(8, 256, 14, 14, true));
    args.x = g_buffer + 2; // not 16-byte aligned
    EXPECT_EQ(im2colLaunch(args, 2).entry, "Im2ColRows2V1");
}

TEST(TestConvRepackLaunch, FewChannelsPerTapWalkTheRowFlat)
{
    auto args = im2col(geometry(8, 3, 112, 112, true));
    const auto launch = im2colLaunch(args, 2);
    EXPECT_EQ(launch.entry, "Im2ColRowsFlat2");
    EXPECT_EQ(args.launch.lanes, 32); // 27 columns per row
}

TEST(TestConvRepackLaunch, ChannelsFirstIm2colStagesATile)
{
    auto args = im2col(geometry(8, 64, 56, 56, false));
    const auto launch = im2colLaunch(args, 4);
    EXPECT_EQ(launch.entry, "Im2ColTile4");
    EXPECT_EQ(args.launch.tileH, 8);
    EXPECT_EQ(args.launch.tileW, 32);
    EXPECT_EQ(args.launch.haloH, 10);
    EXPECT_EQ(args.launch.haloW, 34);
    EXPECT_EQ(launch.ldsBytes, 10u * 34 * 4);
    EXPECT_EQ(launch.grid, 7u * 2 * 8 * 64); // tiles (7 x 2) x images x channels
}

TEST(TestConvRepackLaunch, ATileShrinksUntilItsHaloFits)
{
    auto g = geometry(1, 1, 200, 300, false);
    g.filter[1] = 41;
    g.filter[2] = 41;
    g.dilation[1] = g.dilation[2] = 4; // a 161 x 161 reach per tap
    auto args = im2col(g);
    // 2-byte elements: 8 x 32 needs a 168 x 192 halo (64.5 KB); 4 x 32 fits.
    const auto launch = im2colLaunch(args, 2);
    ASSERT_EQ(launch.entry, "Im2ColTile2");
    EXPECT_LE(launch.ldsBytes, 64u * 1024);
    EXPECT_EQ(args.launch.tileH, 4);
    EXPECT_EQ(args.launch.tileW, 32);
    // 4-byte elements: no tile's halo fits, so the generic kernel.
    auto wide = im2col(g);
    EXPECT_EQ(im2colLaunch(wide, 4).entry, "Im2Col4");
}

TEST(TestConvRepackLaunch, ThreeDChannelsFirstIsGeneric)
{
    auto g = geometry(1, 3, 16, 16, false);
    g.input[0] = g.output[0] = 8;
    g.filter[0] = 3;
    g.xStrides[2] = 256;
    auto args = im2col(g);
    EXPECT_EQ(im2colLaunch(args, 4).entry, "Im2Col4");
}

TEST(TestConvRepackLaunch, OffsetsPast32BitsAreGeneric)
{
    auto args = im2col(geometry(4096, 256, 64, 64, true)); // a column matrix of 38 G elements
    EXPECT_EQ(im2colLaunch(args, 2).entry, "Im2Col2");
    auto back = col2im(geometry(4096, 256, 64, 64, true));
    EXPECT_EQ(col2imLaunch(back, "Half", 2).entry, "Col2ImHalf");
}

TEST(TestConvRepackLaunch, Col2imPicksRowsOrChannelsFirst)
{
    auto last = col2im(geometry(8, 128, 28, 28, true));
    EXPECT_EQ(col2imLaunch(last, "Half", 2).entry, "Col2ImRowsHalfV4");
    EXPECT_EQ(last.launch.lanes, 32);
    auto odd = col2im(geometry(8, 6, 28, 28, true));
    EXPECT_EQ(col2imLaunch(odd, "Float", 4).entry, "Col2ImRowsFloatV1");
    auto first = col2im(geometry(8, 64, 56, 56, false));
    const auto launch = col2imLaunch(first, "Bfloat16", 2);
    EXPECT_EQ(launch.entry, "Col2ImChannelsFirstBfloat16");
    EXPECT_EQ(launch.grid, (8u * 64 * 56 * 56 + 255) / 256);
}

} // namespace
