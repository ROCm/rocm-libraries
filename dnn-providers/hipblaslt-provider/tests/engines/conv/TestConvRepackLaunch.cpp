// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include "kernels/ConvRepackLaunch.hpp"

/**
 * @file TestConvRepackLaunch.cpp
 * @brief Which repack kernel each layout gets, and its launch shape: the rows, per-tap and tile
 *        im2col kernels, the staged-groups, rows and channels-first col2im kernels, the generic
 *        fallbacks past 32-bit offsets, and the tile and group chunk shrinking until they fit LDS.
 *        No device is used.
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

TEST(TestConvRepackLaunch, FewChannelsPerTapTakeOneThreadPerTap)
{
    auto args = im2col(geometry(8, 3, 112, 112, true));
    const auto launch = im2colLaunch(args, 2);
    EXPECT_EQ(launch.entry, "Im2ColTapsStaged2");
    EXPECT_EQ(args.launch.lanes, 16); // 9 taps per row
    EXPECT_EQ(args.launch.rows, 64); // 64 rows of 27 fp16 columns: 3456 bytes, 16-byte aligned
    EXPECT_EQ(launch.ldsBytes, 64u * 27 * 2);

    // Unaligned column storage keeps the direct per-tap stores.
    auto misaligned = im2col(geometry(8, 3, 112, 112, true));
    misaligned.col = g_buffer + 2;
    EXPECT_EQ(im2colLaunch(misaligned, 2).entry, "Im2ColTaps2");
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

TEST(TestConvRepackLaunch, SmallPlanesShareABlockAcrossChannels)
{
    // 7 x 7 outputs: the plane is the tile and five channels fill a block.
    auto args = im2col(geometry(8, 512, 7, 7, false));
    auto launch = im2colLaunch(args, 4);
    EXPECT_EQ(launch.entry, "Im2ColTile4");
    EXPECT_EQ(args.launch.tileW, 7);
    EXPECT_EQ(args.launch.tileH, 7);
    EXPECT_EQ(args.launch.channelsPerBlock, 5);
    EXPECT_EQ(launch.grid, 8u * ((512 + 4) / 5));
    EXPECT_EQ(launch.ldsBytes, 5u * 9 * 9 * 4);

    // 14 wide: a 16 x 16 tile, one channel per block.
    args = im2col(geometry(8, 512, 20, 14, false));
    launch = im2colLaunch(args, 4);
    EXPECT_EQ(args.launch.tileW, 16);
    EXPECT_EQ(args.launch.tileH, 16);
    EXPECT_EQ(args.launch.channelsPerBlock, 1);
}

TEST(TestConvRepackLaunch, DisjointChannelsFirstIm2colRunsAlongOutputs)
{
    // 1x1 stride 2 (and 2x2 stride 2): no input element feeds two entries, so no tile; runs of a
    // column along its 14 x 14 outputs: 196 fp32 make 16-byte runs, 196 fp16 8-byte ones.
    auto g = geometry(8, 64, 28, 28, false);
    g.filter[1] = g.filter[2] = 1;
    g.prePadding[1] = g.prePadding[2] = 0;
    g.stride[1] = g.stride[2] = 2;
    g.output[1] = g.output[2] = 14;
    auto args = im2col(g);
    auto launch = im2colLaunch(args, 4, 0);
    EXPECT_EQ(launch.entry, "Im2ColDisjointOutputs4V4");
    EXPECT_EQ(launch.grid, (8u * 64 * 196 / 4 + 255) / 256);
    launch = im2colLaunch(args, 2, 0);
    EXPECT_EQ(launch.entry, "Im2ColDisjointOutputs2V4");
    auto patchify = g;
    patchify.filter[1] = patchify.filter[2] = 2;
    auto patchArgs = im2col(patchify);
    launch = im2colLaunch(patchArgs, 2, 0);
    EXPECT_EQ(launch.entry, "Im2ColDisjointOutputs2V4");
    EXPECT_EQ(launch.grid, (8u * 64 * 4 * 196 / 4 + 255) / 256);
    // Channels-last columns keep their rows kernels.
    auto last = geometry(8, 64, 28, 28, true);
    last.filter[1] = last.filter[2] = 1;
    last.prePadding[1] = last.prePadding[2] = 0;
    last.stride[1] = last.stride[2] = 2;
    auto lastArgs = im2col(last);
    EXPECT_EQ(im2colLaunch(lastArgs, 2).entry, "Im2ColRows2V8");

    // Runs narrow until the problem has 65536 threads: 8 x 64 columns of 196 outputs take
    // single elements (2-element runs would leave 8 x 64 x 98), 32 images 4-element runs.
    EXPECT_EQ(im2colLaunch(args, 2).entry, "Im2ColDisjointOutputs2V1");
    auto many = g;
    many.n = 32;
    auto manyArgs = im2col(many);
    EXPECT_EQ(im2colLaunch(manyArgs, 2).entry, "Im2ColDisjointOutputs2V4");
}

TEST(TestConvRepackLaunch, ATileShrinksUntilItsHaloFits)
{
    auto g = geometry(1, 1, 200, 300, false);
    g.filter[1] = 55;
    g.filter[2] = 55;
    g.dilation[1] = g.dilation[2] = 3; // a 163 x 163 reach per tap
    auto args = im2col(g);
    // 2-byte elements: the 8 x 32 tile needs a 170 x 194 halo (64.4 KiB); halving its longer
    // side, 8 x 16 (170 x 178, 59.1 KiB) fits.
    const auto launch = im2colLaunch(args, 2);
    ASSERT_EQ(launch.entry, "Im2ColTile2");
    EXPECT_LE(launch.ldsBytes, 64u * 1024);
    EXPECT_EQ(args.launch.tileH, 8);
    EXPECT_EQ(args.launch.tileW, 16);
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

/// @p groups groups of @p cPerGroup channels, channels-last.
ConvColumnGeometry groupedGeometry(long long groups, long long cPerGroup, long long size)
{
    auto g = geometry(8, groups * cPerGroup, size, size, true);
    g.groups = groups;
    g.cPerGroup = cPerGroup;
    return g;
}

TEST(TestConvRepackLaunch, NarrowGroupsCol2imStagesFilterRows)
{
    auto args = col2im(groupedGeometry(32, 4, 28));
    const auto launch = col2imLaunch(args, "Half", 2);
    EXPECT_EQ(launch.entry, "Col2ImGroupsHalfV4");
    EXPECT_EQ(args.launch.groupsPerBlock, 16);
    EXPECT_EQ(args.launch.lanes, 16);
    EXPECT_EQ(args.launch.rows, 16);
    EXPECT_EQ(args.launch.haloW, 19); // 16 positions plus the 3-wide filter's reach, plus one
    EXPECT_EQ(launch.ldsBytes, 19u * 16 * 3 * 4 * 4);
    EXPECT_EQ(launch.grid, 8u * 28 * 2 * 2);

    auto odd = col2im(groupedGeometry(4, 3, 28));
    EXPECT_EQ(col2imLaunch(odd, "Float", 4).entry, "Col2ImGroupsFloatV1");
}

TEST(TestConvRepackLaunch, StagedGroupsHalveUntilTheyFitLds)
{
    auto g = groupedGeometry(32, 8, 64);
    g.filter[1] = g.filter[2] = 11;
    auto args = col2im(g);
    const auto launch = col2imLaunch(args, "Float", 4);
    EXPECT_EQ(launch.entry, "Col2ImGroupsFloatV4");
    EXPECT_EQ(args.launch.groupsPerBlock, 4); // 8 needs 74 KiB
    EXPECT_LE(launch.ldsBytes, 64u * 1024);
}

TEST(TestConvRepackLaunch, WideGroupsCol2imReadsWholeVectors)
{
    auto args = col2im(groupedGeometry(2, 64, 28));
    EXPECT_EQ(col2imLaunch(args, "Half", 2).entry, "Col2ImRowsHalfV4");
}

TEST(TestConvRepackLaunch, DisjointTapsCol2imReadsColumnsInDxType)
{
    auto g = geometry(8, 64, 28, 28, false);
    g.filter[1] = g.filter[2] = 1;
    g.prePadding[1] = g.prePadding[2] = 0;
    g.stride[1] = g.stride[2] = 2;
    auto args = col2im(g);
    // Channels-first: runs along (height, width), which dx packs together: 16 bytes though rows
    // of 28 fp16 are only 8-byte aligned.
    auto launch = col2imLaunch(args, "Half", 2, 0);
    EXPECT_EQ(launch.entry, "Col2ImDisjointWidth2V8");
    EXPECT_EQ(args.launch.rows, 1);
    EXPECT_EQ(launch.grid, (8u * 64 * 28 * 28 / 8 + 255) / 256);
    launch = col2imLaunch(args, "Float", 4, 0);
    EXPECT_EQ(launch.entry, "Col2ImDisjointWidth4V4");
    EXPECT_EQ(launch.grid, (8u * 64 * 28 * 28 / 4 + 255) / 256);
    // Padded rows (a pitch of 30): runs within a row, as wide as every row start keeps aligned
    // (4 bytes).
    auto padded = g;
    padded.xStrides[3] = 30;
    padded.xStrides[1] = 28 * 30;
    padded.xStrides[0] = 64 * 28 * 30;
    auto paddedArgs = col2im(padded);
    launch = col2imLaunch(paddedArgs, "Half", 2, 0);
    EXPECT_EQ(launch.entry, "Col2ImDisjointWidth2V2");
    EXPECT_EQ(paddedArgs.launch.rows, 0);
    EXPECT_EQ(launch.grid, (8u * 64 * 28 * 14 + 255) / 256);

    // Channels-last: runs along (width, channels), which dx packs together.
    auto last = geometry(8, 64, 28, 28, true);
    last.filter[1] = last.filter[2] = 1;
    last.prePadding[1] = last.prePadding[2] = 0;
    last.stride[1] = last.stride[2] = 2;
    auto lastArgs = col2im(last);
    launch = col2imLaunch(lastArgs, "Half", 2, 0);
    EXPECT_EQ(launch.entry, "Col2ImDisjointChannels2V8");
    EXPECT_EQ(lastArgs.launch.rows, 1);
    EXPECT_EQ(launch.grid, (8u * 28 * 28 * 64 / 8 + 255) / 256);
    // Three channels (RGB patchify): the fused run still fills 16 bytes; padded positions do not
    // pack (width, channels), so runs stay within one position's channels, at most
    // as wide as they.
    auto rgb = last;
    rgb.cPerGroup = 3;
    rgb.xStrides[4] = 3;
    rgb.xStrides[3] = 3 * 32;
    rgb.input[2] = 32;
    auto rgbArgs = col2im(rgb);
    EXPECT_EQ(col2imLaunch(rgbArgs, "Half", 2, 0).entry, "Col2ImDisjointChannels2V8");
    rgb.xStrides[4] = 4;
    rgb.xStrides[3] = 4 * 32;
    rgbArgs = col2im(rgb);
    EXPECT_EQ(col2imLaunch(rgbArgs, "Half", 2, 0).entry, "Col2ImDisjointChannels2V2");
    EXPECT_EQ(rgbArgs.launch.rows, 0);

    // Offsets past 32 bits: the generic kernel.
    auto huge = geometry(4096, 256, 64, 64, true);
    huge.filter[1] = huge.filter[2] = 1;
    huge.prePadding[1] = huge.prePadding[2] = 0;
    huge.stride[1] = huge.stride[2] = 2;
    auto hugeArgs = col2im(huge);
    EXPECT_EQ(col2imLaunch(hugeArgs, "Half", 2, 0).entry, "Col2ImDisjoint2");
    // Neither width nor channels contiguous: the generic kernel.
    auto strided = g;
    strided.xStrides[4] = 2;
    strided.xStrides[3] = 56;
    strided.xStrides[1] = 56 * 28;
    strided.xStrides[0] = 64 * 56 * 28;
    auto stridedArgs = col2im(strided);
    EXPECT_EQ(col2imLaunch(stridedArgs, "Float", 4, 0).entry, "Col2ImDisjoint4");

    // A 4x4 patchify (stride 4): disjoint taps too; a 3x3 at stride 2 overlaps.
    auto patchify = geometry(8, 3, 224, 224, false);
    patchify.filter[1] = patchify.filter[2] = 4;
    patchify.stride[1] = patchify.stride[2] = 4;
    patchify.prePadding[1] = patchify.prePadding[2] = 0;
    auto patchArgs = col2im(patchify);
    EXPECT_EQ(col2imLaunch(patchArgs, "Half", 2, 0).entry, "Col2ImDisjointWidth2V8");
    auto overlap = geometry(8, 3, 224, 224, false);
    overlap.stride[1] = overlap.stride[2] = 2;
    auto overlapArgs = col2im(overlap);
    EXPECT_EQ(col2imLaunch(overlapArgs, "Half", 2, 0).entry.rfind("Col2ImDisjoint", 0),
              std::string::npos);

    // Runs narrow until the problem has 65536 threads: 8 x 64 planes of 784 fp16 take 8-byte
    // runs (8 x 64 x 196 threads).
    EXPECT_EQ(col2imLaunch(args, "Half", 2).entry, "Col2ImDisjointWidth2V4");
}

TEST(TestConvRepackLaunch, ChannelsFirstToLastCopyIsATiledTranspose)
{
    // NCHW to NHWC: h and w merge (contiguous in both), leaving n, c, pixels; src is unit-stride
    // along pixels, dst along c.
    StridedCopyArgs args{};
    args.rank = 4;
    const long long dims[] = {2, 8, 7, 7};
    const long long src[] = {392, 49, 7, 1};
    const long long dst[] = {392, 1, 56, 8};
    for(int axis = 0; axis < 4; ++axis)
    {
        args.dims[axis] = dims[axis];
        args.srcStrides[axis] = src[axis];
        args.dstStrides[axis] = dst[axis];
    }
    args.total = 2 * 8 * 49;
    auto copy = args;
    const auto launch = stridedCopyLaunch(copy, 2);
    // Offsets in 32 bits: the narrow transpose, 64 x 64 tiles of 16-bit pairs.
    EXPECT_EQ(launch.entry, "StridedTransposeNarrow2");
    EXPECT_EQ(copy.narrow, 1);
    EXPECT_EQ(copy.rank, 3);
    EXPECT_EQ(copy.dims[2], 49);
    EXPECT_EQ(copy.srcUnitAxis, 2);
    EXPECT_EQ(copy.dstUnitAxis, 1);
    EXPECT_EQ(launch.grid, 2u * 1 * 1); // n x ceil(8 / 64) x ceil(49 / 64)
    // 4-byte elements: 32 x 32 tiles.
    auto wide = args;
    EXPECT_EQ(stridedCopyLaunch(wide, 4).entry, "StridedTransposeNarrow4");

    // Offsets past 32 bits: the 64-bit transpose, 32 x 32 tiles.
    auto huge = args;
    huge.dims[0] = int64_t{1} << 23; // 2^23 images of 392
    auto hugeLaunch = stridedCopyLaunch(huge, 2);
    EXPECT_EQ(hugeLaunch.entry, "StridedTranspose2");
    EXPECT_EQ(huge.narrow, 0);
    EXPECT_EQ(hugeLaunch.grid, (1u << 23) * 1 * 2);
}

TEST(TestConvRepackLaunch, ContiguousCopyCollapsesToOneAxis)
{
    StridedCopyArgs args{};
    args.rank = 3;
    const long long dims[] = {4, 1, 10};
    const long long strides[] = {10, 10, 1};
    for(int axis = 0; axis < 3; ++axis)
    {
        args.dims[axis] = dims[axis];
        args.srcStrides[axis] = args.dstStrides[axis] = strides[axis];
    }
    args.total = 40;
    EXPECT_EQ(stridedCopyLaunch(args, 4).entry, "StridedCopy4");
    EXPECT_EQ(args.rank, 1);
    EXPECT_EQ(args.dims[0], 40);
}

} // namespace
