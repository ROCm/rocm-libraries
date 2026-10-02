// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdGeometry.hpp"

namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd;

constexpr Problem SMOKE{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1};

TEST(TestGfx950ConvFwdGeometry, RejectsAliasedAndPartiallyOverlappingStorage)
{
    // Adjacent storage is legal, independent of argument address order.
    EXPECT_TRUE(bufferRangesDoNotOverlap({0x1000, 0x1100, 0x1300}, {256, 512, 128}));
    EXPECT_TRUE(bufferRangesDoNotOverlap({0x1300, 0x1100, 0x1000}, {128, 512, 256}));
    EXPECT_FALSE(bufferRangesDoNotOverlap({0x1000, 0x1000, 0x2000}, {256, 512, 128}));
    EXPECT_FALSE(bufferRangesDoNotOverlap({0x1000, 0x10F0, 0x2000}, {256, 512, 128}));
    EXPECT_FALSE(bufferRangesDoNotOverlap({0x1000, 0x2000, 0x1010}, {256, 512, 128}));
    EXPECT_FALSE(bufferRangesDoNotOverlap({0x2000, 0x1000, 0x1010}, {256, 512, 128}));
    EXPECT_FALSE(bufferRangesDoNotOverlap({0x1000, 0x2000, 0x3000}, {0, 512, 128}));
    EXPECT_FALSE(bufferRangesDoNotOverlap(
        {0x1000, 0x2000, std::numeric_limits<uintptr_t>::max() - 63}, {256, 512, 128}));
}

TEST(TestGfx950ConvFwdGeometry, PaddedSmokeUsesByteCountsAndPartialMTile)
{
    const auto geometry = deriveGeometry(SMOKE);
    ASSERT_TRUE(geometry);
    EXPECT_EQ(geometry->ho, 14);
    EXPECT_EQ(geometry->wo, 14);
    EXPECT_EQ(geometry->gemmM, 392);
    EXPECT_EQ(geometry->tensorBytes[0], 25088);
    EXPECT_EQ(geometry->tensorBytes[1], 18432);
    EXPECT_EQ(geometry->tensorBytes[2], 25088);

    const auto launch = launchGeometry(SMOKE, *geometry, 64, 64, 2, 2, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 1U);
    EXPECT_EQ(launch->gridY, 7U);
    EXPECT_EQ(launch->gridZ, 1U);
    EXPECT_EQ(launch->blockX, 256U);
}

TEST(TestGfx950ConvFwdGeometry, NonSquareStrideAndDilationUseFloorOutputShape)
{
    const Problem problem{2, 32, 96, 15, 19, 3, 2, 2, 3, 1, 0, 2, 1};
    const auto geometry = deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    EXPECT_EQ(geometry->ho, 7);
    EXPECT_EQ(geometry->wo, 6);
    EXPECT_EQ(geometry->gemmM, 84);
    EXPECT_EQ(geometry->tensorBytes[0], 36480);
    EXPECT_EQ(geometry->tensorBytes[1], 36864);
    EXPECT_EQ(geometry->tensorBytes[2], 16128);

    const auto launch = launchGeometry(problem, *geometry, 64, 64, 2, 2, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 2U);
    EXPECT_EQ(launch->gridY, 2U);
}

TEST(TestGfx950ConvFwdGeometry, RefusesNonpositiveExtentAndEmptyOutput)
{
    auto problem = SMOKE;
    problem.c = 0;
    EXPECT_FALSE(deriveGeometry(problem));
    problem = SMOKE;
    problem.n = -1;
    EXPECT_FALSE(deriveGeometry(problem));
    problem = SMOKE;
    problem.y = 20;
    EXPECT_FALSE(deriveGeometry(problem));
    problem = SMOKE;
    problem.strideH = 0;
    EXPECT_FALSE(deriveGeometry(problem));
    problem = SMOKE;
    problem.dilationW = -1;
    EXPECT_FALSE(deriveGeometry(problem));
    problem = SMOKE;
    problem.padH = -1;
    EXPECT_FALSE(deriveGeometry(problem));
}

TEST(TestGfx950ConvFwdGeometry, RefusesSignedByteCountOverflowForEveryOperand)
{
    EXPECT_EQ(tensorByteSize({1, 1, 1, 1073741823}), 2147483646);
    EXPECT_FALSE(tensorByteSize({1, 1, 1, 1073741824}));
    EXPECT_FALSE(tensorByteSize({std::numeric_limits<int64_t>::max(), 1, 1, 1}));

    auto problem = SMOKE;
    problem.n = 100000000;
    EXPECT_FALSE(deriveGeometry(problem)); // A bytes.
    problem = SMOKE;
    problem.y = 100000000;
    EXPECT_FALSE(deriveGeometry(problem)); // B bytes.
    problem = Problem{1, 1, 32768, 256, 256, 1, 1, 1, 1, 0, 0, 1, 1};
    EXPECT_FALSE(deriveGeometry(problem)); // D bytes; A and B each fit.
}

TEST(TestGfx950ConvFwdGeometry, RefusesOverflowInPaddedAndDilatedCoordinates)
{
    auto problem = SMOKE;
    problem.padH = std::numeric_limits<int64_t>::max();
    EXPECT_FALSE(deriveGeometry(problem));
    problem = SMOKE;
    problem.dilationW = std::numeric_limits<int32_t>::max();
    EXPECT_FALSE(deriveGeometry(problem));
}

TEST(TestGfx950ConvFwdGeometry, EnforcesTheBuildersGridAndWorkgroupLimits)
{
    const auto geometry = deriveGeometry(SMOKE);
    ASSERT_TRUE(geometry);
    EXPECT_FALSE(launchGeometry(SMOKE, *geometry, 0, 64, 2, 2, 64));
    EXPECT_FALSE(launchGeometry(SMOKE, *geometry, 64, -1, 2, 2, 64));
    EXPECT_FALSE(launchGeometry(SMOKE, *geometry, 64, 64, 0, 2, 64));
    EXPECT_FALSE(launchGeometry(SMOKE, *geometry, 64, 64, 2, 2, 32));
    EXPECT_FALSE(launchGeometry(SMOKE, *geometry, 64, 64, 16, 2, 64));
    EXPECT_FALSE(
        launchGeometry(SMOKE, *geometry, 64, 64, std::numeric_limits<int64_t>::max(), 2, 64));

    const Problem tooManyTiles{1, 1, 1, 2048, 2048, 1, 1, 1, 1, 0, 0, 1, 1};
    const auto largeGeometry = deriveGeometry(tooManyTiles);
    ASSERT_TRUE(largeGeometry);
    EXPECT_FALSE(launchGeometry(tooManyTiles, *largeGeometry, 64, 64, 2, 2, 64));
}

TEST(TestGfx950ConvFwdGeometry, GroupedFilterBytesAndGridUsePerGroupChannels)
{
    // G=4: B is [K, Y, X, C / G], and the N-tiles cover K / G per group on grid z.
    const Problem grouped{2, 32, 96, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 4};
    const auto geometry = deriveGeometry(grouped);
    ASSERT_TRUE(geometry);
    EXPECT_EQ(geometry->gemmM, 162);
    EXPECT_EQ(geometry->tensorBytes[0], 10368);
    EXPECT_EQ(geometry->tensorBytes[1], 13824);
    EXPECT_EQ(geometry->tensorBytes[2], 31104);

    const auto launch = launchGeometry(grouped, *geometry, 64, 16, 2, 2, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 2U);
    EXPECT_EQ(launch->gridY, 3U);
    EXPECT_EQ(launch->gridZ, 4U);
    EXPECT_EQ(launch->blockX, 256U);
}

TEST(TestGfx950ConvFwdGeometry, DepthwiseUsesOneChannelPerFilterAndOneGroupPerChannel)
{
    const Problem depthwise{2, 16, 32, 11, 11, 3, 3, 2, 2, 1, 1, 1, 1, 16};
    const auto geometry = deriveGeometry(depthwise);
    ASSERT_TRUE(geometry);
    EXPECT_EQ(geometry->ho, 6);
    EXPECT_EQ(geometry->wo, 6);
    EXPECT_EQ(geometry->tensorBytes[1], 576);

    // A tile wider than K / G still launches one N-tile; the epilogue masks the rest.
    const auto launch = launchGeometry(depthwise, *geometry, 64, 64, 2, 2, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 1U);
    EXPECT_EQ(launch->gridY, 2U);
    EXPECT_EQ(launch->gridZ, 16U);
}

TEST(TestGfx950ConvFwdGeometry, RefusesInvalidGroupCounts)
{
    for(const auto groups : {int64_t{0}, int64_t{-2}, int64_t{3}, int64_t{64}})
    {
        SCOPED_TRACE(groups);
        auto problem = SMOKE;
        problem.groups = groups;
        EXPECT_FALSE(deriveGeometry(problem));
    }
    // C divisible by G but K not.
    auto problem = SMOKE;
    problem.k = 34;
    problem.groups = 4;
    EXPECT_FALSE(deriveGeometry(problem));

    // A geometry derived for another group count must not launch one that is invalid.
    const auto geometry = deriveGeometry(SMOKE);
    ASSERT_TRUE(geometry);
    problem = SMOKE;
    problem.groups = 0;
    EXPECT_FALSE(launchGeometry(problem, *geometry, 64, 64, 2, 2, 64));
    problem.groups = 3;
    EXPECT_FALSE(launchGeometry(problem, *geometry, 64, 64, 2, 2, 64));
}

TEST(TestGfx950ConvFwdGeometry, RefusesMoreGroupsThanGridZCanHold)
{
    const Problem atLimit{1, 65535, 65535, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 65535};
    const auto geometry = deriveGeometry(atLimit);
    ASSERT_TRUE(geometry);
    const auto launch = launchGeometry(atLimit, *geometry, 64, 64, 2, 2, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridZ, 65535U);

    const Problem overLimit{1, 65536, 65536, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 65536};
    const auto overGeometry = deriveGeometry(overLimit);
    ASSERT_TRUE(overGeometry);
    EXPECT_FALSE(launchGeometry(overLimit, *overGeometry, 64, 64, 2, 2, 64));
}

/// Depthwise N=2, C=K=groups, square filter, one stride and padding.
Problem depthwise(int64_t groups, int64_t h, int64_t w, int64_t filter, int64_t stride, int64_t pad)
{
    return Problem{2, groups, groups, h, w, filter, filter, stride, stride, pad, pad, 1, 1, groups};
}

TEST(TestGfx950ConvFwdGeometry, FamilyAndVariantValueSetsAreClosed)
{
    EXPECT_EQ(parseKernelFamily(0), KernelFamily::IMPLICIT_GEMM);
    EXPECT_EQ(parseKernelFamily(1), KernelFamily::DIRECT_DEPTHWISE);
    for(const auto value : {int64_t{-1}, int64_t{2}, std::numeric_limits<int64_t>::max()})
    {
        SCOPED_TRACE(value);
        EXPECT_FALSE(parseKernelFamily(value));
    }
    EXPECT_EQ(parseDirectVariant("none"), DirectVariant::NONE);
    EXPECT_EQ(parseDirectVariant("std"), DirectVariant::STD);
    EXPECT_EQ(parseDirectVariant("spatial"), DirectVariant::SPATIAL);
    for(const auto* value : {"", "Std", "SPATIAL", "direct_depthwise", "std "})
    {
        SCOPED_TRACE(value);
        EXPECT_FALSE(parseDirectVariant(value));
    }
}

TEST(TestGfx950ConvFwdGeometry,
     DirectRowCoverageAcceptsOnlyShapesWhoseLastInputRowFeedsTheLastOutput)
{
    // {extent, pad, filter, stride, covered}. The direct kernels write output row
    // floor(p / stride) for input rows p < extent: accepted iff floor((extent - 1) / s)
    // equals Ho - 1.
    const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, bool>> cases{
        {14, 1, 3, 1, true}, // "same" padding at stride 1.
        {17, 1, 3, 2, true}, // Stride 2, odd extent.
        {16, 1, 3, 2, true}, // Stride 2, even extent.
        {18, 1, 3, 2, true},
        {9, 3, 7, 1, true},
        {56, 8, 17, 1, true},
        {8, 1, 4, 2, true}, // 2P != KH - 1 still covers every row.
        {8, 0, 2, 2, true},
        {9, 0, 3, 3, true},
        {14, 0, 1, 1, true},
        // Wrong-answer path 1: fewer outputs than streamed rows, so the runtime H loop
        // writes past Ho into the next image.
        {14, 0, 3, 1, false},
        {16, 0, 3, 2, false},
        {70, 0, 3, 1, false},
        // Wrong-answer path 2: more outputs than streamed rows leave the last ones unwritten.
        {14, 2, 3, 1, false},
        {16, 2, 3, 2, false},
        {9, 4, 7, 1, false},
        // No valid output at all, and invalid operands.
        {2, 0, 3, 1, false},
        {14, 1, 3, 0, false},
        {14, -1, 3, 1, false},
        {0, 1, 3, 1, false}};
    for(const auto& [extent, pad, filter, stride, covered] : cases)
    {
        SCOPED_TRACE(::testing::Message()
                     << extent << " pad " << pad << " filter " << filter << " stride " << stride);
        EXPECT_EQ(directRowsCovered(extent, pad, filter, stride), covered);
    }

    // Both axes must be covered; a rectangular filter checks each against its own extent.
    EXPECT_TRUE(directRowsCovered(depthwise(32, 14, 14, 3, 1, 1)));
    auto problem = depthwise(32, 14, 14, 3, 1, 1);
    problem.x = 1;
    problem.padW = 0;
    EXPECT_TRUE(directRowsCovered(problem));
    problem.padW = 1; // W: Wo = 16 but only 14 columns are streamed.
    EXPECT_FALSE(directRowsCovered(problem));
    problem = depthwise(32, 14, 14, 3, 1, 1);
    problem.padH = 0; // H spills while W is covered.
    EXPECT_FALSE(directRowsCovered(problem));
}

TEST(TestGfx950ConvFwdGeometry, DirectProblemGuardMatchesTheAdapter)
{
    // Expected values are _direct_error(spec) == "" in convolution_forward.py for the
    // same problems; the tuning-side checks live in directLaunchGeometry.
    const std::vector<std::pair<Problem, bool>> cases{
        {{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, true},
        {{2, 96, 96, 17, 17, 3, 3, 2, 2, 1, 1, 1, 1, 96}, true},
        {{2, 96, 96, 16, 18, 3, 3, 2, 2, 1, 1, 1, 1, 96}, true},
        {{2, 5, 5, 9, 9, 7, 7, 1, 1, 3, 3, 1, 1, 5}, true},
        {{2, 5, 5, 56, 24, 17, 17, 1, 1, 8, 8, 1, 1, 5}, true},
        {{2, 24, 24, 8, 8, 4, 4, 2, 2, 1, 1, 1, 1, 24}, true},
        {{2, 1, 1, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true},
        // pH != pW, even where each axis alone would be covered.
        {{2, 24, 24, 14, 14, 3, 1, 1, 1, 1, 0, 1, 1, 24}, false},
        {{2, 24, 24, 13, 13, 3, 3, 1, 1, 1, 0, 1, 1, 24}, false},
        // sH != sW.
        {{2, 24, 24, 13, 13, 3, 3, 2, 1, 1, 1, 1, 1, 24}, false},
        // Dilation on either axis.
        {{2, 24, 24, 13, 13, 3, 3, 1, 1, 2, 2, 2, 2, 24}, false},
        {{2, 24, 24, 13, 13, 3, 3, 1, 1, 1, 1, 1, 2, 24}, false},
        // Not pure depthwise: K = 2C, C = 2K, groups < C.
        {{2, 16, 32, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}, false},
        {{2, 32, 16, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}, false},
        {{2, 16, 16, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 8}, false},
        // Rows or columns not covered by the input stream.
        {{2, 24, 24, 12, 12, 3, 3, 1, 1, 0, 0, 1, 1, 24}, false},
        {{2, 24, 24, 12, 12, 3, 3, 1, 1, 2, 2, 1, 1, 24}, false},
        {{2, 24, 24, 14, 14, 3, 1, 1, 1, 1, 1, 1, 1, 24}, false},
        {{2, 24, 24, 9, 9, 7, 7, 1, 1, 4, 4, 1, 1, 24}, false},
        {{2, 24, 24, 16, 16, 3, 3, 2, 2, 0, 0, 1, 1, 24}, false}};
    for(size_t i = 0; i < cases.size(); ++i)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(directProblemSupported(cases[i].first), cases[i].second);
    }
}

TEST(TestGfx950ConvFwdGeometry, DirectStdGridTilesOutputColumnsChannelsAndImages)
{
    // ceil(Wo / block_w) x ceil(G / (64 * block_waves)) x N, 64 * block_waves threads.
    const auto problem = depthwise(96, 14, 14, 3, 1, 1);
    const auto geometry = deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    auto launch = directLaunchGeometry(problem, *geometry, DirectVariant::STD, 4, 1, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 4U);
    EXPECT_EQ(launch->gridY, 2U); // Partial channel tile: 96 channels over 64 lanes.
    EXPECT_EQ(launch->gridZ, 2U);
    EXPECT_EQ(launch->blockX, 64U);

    launch = directLaunchGeometry(problem, *geometry, DirectVariant::STD, 3, 2, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 5U);
    EXPECT_EQ(launch->gridY, 1U);
    EXPECT_EQ(launch->blockX, 128U);

    // Stride 2 on a non-square input: Wo = 9.
    const auto strided = depthwise(96, 16, 18, 3, 2, 1);
    const auto stridedGeometry = deriveGeometry(strided);
    ASSERT_TRUE(stridedGeometry);
    EXPECT_EQ(stridedGeometry->ho, 8);
    EXPECT_EQ(stridedGeometry->wo, 9);
    launch = directLaunchGeometry(strided, *stridedGeometry, DirectVariant::STD, 4, 1, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 3U);
    EXPECT_EQ(launch->gridY, 2U);

    launch = directLaunchGeometry(problem, *geometry, DirectVariant::STD, 32, 16, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 1U);
    EXPECT_EQ(launch->gridY, 1U);
    EXPECT_EQ(launch->blockX, 1024U);
}

TEST(TestGfx950ConvFwdGeometry, DirectSpatialGridDerivesItsBlockWidthFromTheGroupCount)
{
    // block_w = block_waves * (64 / G); no channel tile.
    const auto problem = depthwise(32, 14, 14, 3, 1, 1);
    const auto geometry = deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    auto launch = directLaunchGeometry(problem, *geometry, DirectVariant::SPATIAL, 0, 1, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 7U); // Two columns per wave.
    EXPECT_EQ(launch->gridY, 1U);
    EXPECT_EQ(launch->gridZ, 2U);
    EXPECT_EQ(launch->blockX, 64U);

    launch = directLaunchGeometry(problem, *geometry, DirectVariant::SPATIAL, 0, 3, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 3U);
    EXPECT_EQ(launch->blockX, 192U);

    // G=3: 21 columns per wave; G=5: 12, so a 9-column output is one block.
    const auto three = depthwise(3, 40, 24, 3, 1, 1);
    const auto threeGeometry = deriveGeometry(three);
    ASSERT_TRUE(threeGeometry);
    launch = directLaunchGeometry(three, *threeGeometry, DirectVariant::SPATIAL, 0, 1, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 2U);
    const auto five = depthwise(5, 9, 9, 7, 1, 3);
    const auto fiveGeometry = deriveGeometry(five);
    ASSERT_TRUE(fiveGeometry);
    launch = directLaunchGeometry(five, *fiveGeometry, DirectVariant::SPATIAL, 0, 1, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 1U);

    // G=63 still has one column per wave; G=64 is left to the std variant.
    const auto below = depthwise(63, 14, 14, 3, 1, 1);
    const auto belowGeometry = deriveGeometry(below);
    ASSERT_TRUE(belowGeometry);
    launch = directLaunchGeometry(below, *belowGeometry, DirectVariant::SPATIAL, 0, 1, 64);
    ASSERT_TRUE(launch);
    EXPECT_EQ(launch->gridX, 14U);
    const auto at = depthwise(64, 14, 14, 3, 1, 1);
    const auto atGeometry = deriveGeometry(at);
    ASSERT_TRUE(atGeometry);
    EXPECT_FALSE(directLaunchGeometry(at, *atGeometry, DirectVariant::SPATIAL, 0, 1, 64));
    EXPECT_TRUE(directLaunchGeometry(at, *atGeometry, DirectVariant::STD, 4, 1, 64));
}

TEST(TestGfx950ConvFwdGeometry, DirectLaunchRefusesOutOfContractTuning)
{
    const auto problem = depthwise(32, 14, 14, 3, 1, 1);
    const auto geometry = deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    for(const auto variant : {DirectVariant::STD, DirectVariant::SPATIAL})
    {
        const int64_t blockW = variant == DirectVariant::STD ? 4 : 0;
        for(const auto blockWaves : {int64_t{0}, int64_t{-1}, int64_t{17}})
        {
            SCOPED_TRACE(blockWaves);
            EXPECT_FALSE(directLaunchGeometry(problem, *geometry, variant, blockW, blockWaves, 64));
        }
        EXPECT_TRUE(directLaunchGeometry(problem, *geometry, variant, blockW, 16, 64));
        EXPECT_FALSE(directLaunchGeometry(problem, *geometry, variant, blockW, 1, 32));
    }
    EXPECT_FALSE(directLaunchGeometry(problem, *geometry, DirectVariant::NONE, 0, 1, 64));
    EXPECT_FALSE(directLaunchGeometry(problem, *geometry, DirectVariant::NONE, 4, 1, 64));
    EXPECT_FALSE(directLaunchGeometry(problem, *geometry, DirectVariant::STD, 0, 1, 64));
    EXPECT_FALSE(directLaunchGeometry(problem, *geometry, DirectVariant::STD, -4, 1, 64));
    EXPECT_FALSE(directLaunchGeometry(problem,
                                      *geometry,
                                      DirectVariant::STD,
                                      int64_t{std::numeric_limits<int32_t>::max()} + 1,
                                      1,
                                      64));
    EXPECT_TRUE(directLaunchGeometry(
        problem, *geometry, DirectVariant::STD, std::numeric_limits<int32_t>::max(), 1, 64));
    EXPECT_FALSE(directLaunchGeometry(problem, *geometry, DirectVariant::SPATIAL, 4, 1, 64));
}

TEST(TestGfx950ConvFwdGeometry, DirectLaunchRefusesEveryGridAxisAbove65535)
{
    // x: Wo columns at block_w 1.
    for(const auto wi : {int64_t{65535}, int64_t{65536}})
    {
        SCOPED_TRACE(wi);
        const Problem wide{1, 1, 1, 1, wi, 1, 1, 1, 1, 0, 0, 1, 1, 1};
        const auto geometry = deriveGeometry(wide);
        ASSERT_TRUE(geometry);
        EXPECT_EQ(directLaunchGeometry(wide, *geometry, DirectVariant::STD, 1, 1, 64).has_value(),
                  wi == 65535);
        // The same output fits once more columns share a block.
        EXPECT_TRUE(directLaunchGeometry(wide, *geometry, DirectVariant::STD, 2, 1, 64));
    }
    // y: channel tiles of 64 lanes.
    for(const auto groups : {int64_t{65535} * 64, int64_t{65535} * 64 + 1})
    {
        SCOPED_TRACE(groups);
        const Problem channels{1, groups, groups, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, groups};
        const auto geometry = deriveGeometry(channels);
        ASSERT_TRUE(geometry);
        EXPECT_EQ(
            directLaunchGeometry(channels, *geometry, DirectVariant::STD, 1, 1, 64).has_value(),
            groups == int64_t{65535} * 64);
        EXPECT_TRUE(directLaunchGeometry(channels, *geometry, DirectVariant::STD, 1, 2, 64));
    }
    // z: one slice per image, for both variants.
    for(const auto n : {int64_t{65535}, int64_t{65536}})
    {
        SCOPED_TRACE(n);
        const Problem images{n, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 2};
        const auto geometry = deriveGeometry(images);
        ASSERT_TRUE(geometry);
        EXPECT_EQ(directLaunchGeometry(images, *geometry, DirectVariant::STD, 4, 1, 64).has_value(),
                  n == 65535);
        EXPECT_EQ(
            directLaunchGeometry(images, *geometry, DirectVariant::SPATIAL, 0, 1, 64).has_value(),
            n == 65535);
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
