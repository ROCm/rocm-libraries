// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <string>
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
    EXPECT_EQ(parseDirectVariant("tiled"), DirectVariant::TILED);
    EXPECT_EQ(parseDirectVariant("col"), DirectVariant::COL);
    EXPECT_EQ(parseDirectVariant("grouped"), DirectVariant::GROUPED);
    EXPECT_EQ(parseDirectVariant("nongrouped"), DirectVariant::NONGROUPED);
    for(const auto* value :
        {"", "Std", "SPATIAL", "direct_depthwise", "std ", "direct_depthwise_tiled", "Col"})
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
    // Expected values are supports_gfx950_conv_fwd's problem-side verdict in
    // convolution_forward.py for the same problems: _direct_error plus rocKE's own
    // forward_padding_reason. The tuning-side checks live in directLaunchGeometry.
    const std::vector<std::pair<Problem, bool>> cases{
        {{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, true},
        {{2, 96, 96, 17, 17, 3, 3, 2, 2, 1, 1, 1, 1, 96}, true},
        {{2, 96, 96, 16, 18, 3, 3, 2, 2, 1, 1, 1, 1, 96}, true},
        {{2, 5, 5, 9, 9, 7, 7, 1, 1, 3, 3, 1, 1, 5}, true},
        {{2, 5, 5, 56, 24, 17, 17, 1, 1, 8, 8, 1, 1, 5}, true},
        {{2, 1, 1, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true},
        // Every row and column covered, but rocKE refuses an even filter or padding other
        // than (KH - 1) / 2.
        {{2, 24, 24, 8, 8, 4, 4, 2, 2, 1, 1, 1, 1, 24}, false},
        {{2, 24, 24, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, 24}, false},
        {{2, 24, 24, 9, 9, 3, 3, 3, 3, 0, 0, 1, 1, 24}, false},
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

TEST(TestGfx950ConvFwdGeometry, MagicDivisionMatchesRockesCalculateMagicNumbers)
{
    // (divisor, multiplier as i32, shift) from calculate_magic_numbers and _magic_as_i32.
    const std::vector<std::tuple<int64_t, int32_t, int32_t>> cases{
        {1, 1, 0},
        {2, 1, 1},
        {3, 1431655766, 2},
        {7, 613566757, 3},
        {8, 1, 3},
        {13, 991146300, 4},
        {14, 613566757, 4},
        {15, 286331154, 4},
        {5, -1717986918, 3},
        {1000, 103079216, 10},
        {65535, 65538, 16},
        {int64_t{1} << 30, 1, 30},
        {std::numeric_limits<int32_t>::max(), 3, 31}};
    for(const auto& [divisor, multiplier, shift] : cases)
    {
        SCOPED_TRACE(divisor);
        const auto magic = magicDivision(divisor);
        ASSERT_TRUE(magic);
        EXPECT_EQ(magic->multiplier, multiplier);
        EXPECT_EQ(magic->shift, shift);
    }
    EXPECT_FALSE(magicDivision(0));
    EXPECT_FALSE(magicDivision(int64_t{1} << 31));

    // The kernel's mul-hi sequence reproduces every quotient in the 31-bit range.
    for(const int64_t divisor : {3, 7, 13, 15, 1000, 65535})
    {
        const auto magic = magicDivision(divisor);
        ASSERT_TRUE(magic);
        for(const uint32_t dividend : {0U, 1U, 999U, 65536U, 2147483647U})
        {
            const auto multiplier = static_cast<uint32_t>(magic->multiplier);
            const auto high = static_cast<uint32_t>((uint64_t{dividend} * multiplier) >> 32);
            EXPECT_EQ((uint64_t{high} + dividend) >> magic->shift, dividend / divisor);
        }
    }
}

TEST(TestGfx950ConvFwdGeometry, ImplicitGemmKernargsMatchRockesConvArgs)
{
    // Grouped, strided, padded and dilated, with distinct extents on every axis. Expected
    // values are ConvArgs.from_problem(ConvProblem(...), tile_m=64, tile_n=16)
    // .to_launch_values() in rocke/library/kernels/common/conv_args.py.
    const Problem problem{2, 8, 12, 15, 19, 3, 5, 2, 1, 1, 2, 1, 2, 4};
    const auto geometry = deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    const auto launch = launchGeometry(problem, *geometry, 64, 16, 2, 2, 64);
    ASSERT_TRUE(launch);
    const auto kernargs = implicitGemmKernargs(problem, *geometry, *launch);
    ASSERT_TRUE(kernargs);
    const ImplicitGemmKernargs expected{
        2,    15,  19, 8, 12, 3,         5,   2,           1,   1, 2,  1,  2,
        4,    8,   15, 2, 3,  30,        240, 2280,        152, 8, 30, 10, 2,
        1440, 180, 12, 1, 3,  286331154, 4,   -1717986918, 3,   1, 1,  4,  1};
    for(size_t i = 0; i < expected.size(); ++i)
    {
        SCOPED_TRACE(IMPLICIT_GEMM_KERNARG_NAMES[i]);
        EXPECT_EQ((*kernargs)[i], expected[i]);
    }

    // The tile counts the kernel decodes its workgroup id against are the grid's.
    EXPECT_EQ(kernargs->at(37), static_cast<int32_t>(launch->gridY));
    EXPECT_EQ(kernargs->at(38), static_cast<int32_t>(launch->gridX));
    EXPECT_EQ(IMPLICIT_GEMM_KERNARG_NAMES[37], "p_num_pid_m");
    EXPECT_EQ(IMPLICIT_GEMM_KERNARG_NAMES[38], "p_num_pid_n");
}

TEST(TestGfx950ConvFwdGeometry, ImplicitGemmKernargsRefuseA24BitReductionOverflow)
{
    // Y * X * C / groups must stay below 2^23 for the kernel's 24-bit address products.
    Problem problem{1, 1 << 17, 8, 8, 8, 8, 8, 1, 1, 0, 0, 1, 1};
    auto geometry = deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    auto launch = launchGeometry(problem, *geometry, 64, 64, 2, 2, 64);
    ASSERT_TRUE(launch);
    EXPECT_FALSE(implicitGemmKernargs(problem, *geometry, *launch));

    problem.c = (1 << 17) - 1;
    geometry = deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    launch = launchGeometry(problem, *geometry, 64, 64, 2, 2, 64);
    ASSERT_TRUE(launch);
    EXPECT_TRUE(implicitGemmKernargs(problem, *geometry, *launch));
}

TEST(TestGfx950ConvFwdGeometry, DirectKernargsMatchRockesConvArgs)
{
    // ConvArgs.from_problem(DirectConvProblem(N=2, H=14, W=13, groups=32, cpg=1, kpg=1,
    // KH=3, KW=3, PAD=1, stride=2)).to_launch_values() in conv_args.py.
    const Problem problem{2, 32, 32, 14, 13, 3, 3, 2, 2, 1, 1, 1, 1, 32};
    const auto geometry = deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    const DirectKernargs expected{2, 14, 13, 7, 7, 32, 32, 32, 5824, 416, 32, 1568, 224, 32};
    EXPECT_EQ(directKernargs(problem, *geometry), expected);
}

TEST(TestGfx950ConvFwdGeometry, AotInstanceLaunchTilesTheRuntimeProblemWithItsOwnTile)
{
    // One instance, two problems: the grid and the AOT tile counts follow the graph.
    const AotInstance instance{64, 32, 2, 1, 8, 8, 8, false, 0, 0, true};
    for(const auto& problem : {Problem{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
                               Problem{4, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 2}})
    {
        const auto geometry = deriveGeometry(problem);
        ASSERT_TRUE(geometry);
        const auto launch = aotInstanceLaunch(problem, *geometry, instance);
        ASSERT_TRUE(launch);
        const auto kpg = problem.k / problem.groups;
        EXPECT_EQ(launch->gridX, static_cast<unsigned int>((kpg + 31) / 32));
        EXPECT_EQ(launch->gridY, static_cast<unsigned int>((geometry->gemmM + 63) / 64));
        EXPECT_EQ(launch->gridZ, static_cast<unsigned int>(problem.groups));
        EXPECT_EQ(launch->blockX, 128U);
        const auto kernargs = implicitGemmKernargs(problem, *geometry, *launch);
        ASSERT_TRUE(kernargs);
        EXPECT_EQ(kernargs->at(37), static_cast<int32_t>(launch->gridY)); // p_num_pid_m
        EXPECT_EQ(kernargs->at(38), static_cast<int32_t>(launch->gridX)); // p_num_pid_n
    }
}

TEST(TestGfx950ConvFwdGeometry, AotInstanceLaunchRefusesWhatTheBinaryCannotCompute)
{
    const Problem grouped{2, 24, 40, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 2}; // cpg 12, kpg 20
    const auto geometry = deriveGeometry(grouped);
    ASSERT_TRUE(geometry);
    const auto launches = [&](AotInstance instance) {
        return aotInstanceLaunch(grouped, *geometry, instance).has_value();
    };
    const AotInstance base{64, 64, 2, 2, 4, 4, 4, false, 0, 0, true};
    EXPECT_TRUE(launches(base));
    auto changed = base;
    changed.grouped = false;
    EXPECT_FALSE(launches(changed)); // Ungrouped code path on a grouped problem.
    changed = base;
    changed.vectorA = 8;
    EXPECT_FALSE(launches(changed)); // 8 does not divide C / groups = 12.
    changed = base;
    changed.vectorC = 8;
    EXPECT_FALSE(launches(changed)); // 8 does not divide K / groups = 20.
    changed = base;
    changed.asyncDma = true;
    changed.asyncChunkA = changed.asyncChunkB = 4;
    EXPECT_TRUE(launches(changed));
    changed.asyncChunkB = 8;
    EXPECT_FALSE(launches(changed)); // A chunk would straddle a filter position.
    changed.asyncChunkA = changed.asyncChunkB = 0;
    EXPECT_FALSE(launches(changed)); // An async binary must record its chunks.

    // gridDim.y: 16-row tiles over N * Ho * Wo = 2^21 outputs need 131072 workgroups.
    const Problem tall{2048, 8, 8, 32, 32, 1, 1, 1, 1, 0, 0, 1, 1};
    const auto tallGeometry = deriveGeometry(tall);
    ASSERT_TRUE(tallGeometry);
    EXPECT_FALSE(
        aotInstanceLaunch(tall, *tallGeometry, {16, 16, 1, 1, 8, 8, 8, false, 0, 0, true}));
    EXPECT_TRUE(
        aotInstanceLaunch(tall, *tallGeometry, {256, 16, 4, 1, 8, 8, 8, false, 0, 0, true}));
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

TEST(TestGfx950ConvFwdGeometry, DirectInstanceServesOnlyItsCompiledFilterStrideAndPadding)
{
    // N, Hi, Wi and the group count are kernel arguments, so one compiled 3x3 stride-1
    // instance serves every guarded depthwise problem with those values.
    const DirectInstance instance{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::STD, 4, 1};
    for(const auto& problem : {depthwise(32, 14, 14, 3, 1, 1),
                               depthwise(96, 7, 21, 3, 1, 1),
                               depthwise(1, 64, 64, 3, 1, 1)})
    {
        const auto geometry = deriveGeometry(problem);
        ASSERT_TRUE(geometry);
        EXPECT_TRUE(directServes(instance.compiled, problem));
        const auto launch = directInstanceLaunch(problem, *geometry, instance);
        ASSERT_TRUE(launch);
        // The exact-shape engine's grid for the same tuning.
        const auto expected
            = directLaunchGeometry(problem, *geometry, DirectVariant::STD, 4, 1, DIRECT_WAVE_SIZE);
        ASSERT_TRUE(expected);
        EXPECT_EQ(std::tie(launch->gridX, launch->gridY, launch->gridZ, launch->blockX),
                  std::tie(expected->gridX, expected->gridY, expected->gridZ, expected->blockX));
    }
    // A build problem's capabilities are its filter, stride, padding and dilation only.
    auto build = depthwise(8, 64, 64, 3, 1, 1);
    build.n = 1;
    const auto fromBuild = directCapabilities(build);
    EXPECT_TRUE(directServes(fromBuild, depthwise(32, 14, 14, 3, 1, 1)));
    for(const auto& other : {depthwise(32, 14, 14, 5, 1, 2), depthwise(32, 17, 17, 3, 2, 1)})
    {
        EXPECT_FALSE(directServes(fromBuild, other));
        const auto geometry = deriveGeometry(other);
        ASSERT_TRUE(geometry);
        EXPECT_FALSE(directInstanceLaunch(other, *geometry, instance));
    }
    auto dilated = instance;
    dilated.compiled.dilationH = dilated.compiled.dilationW = 2;
    const auto problem = depthwise(32, 14, 14, 3, 1, 1);
    const auto geometry = deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    EXPECT_FALSE(directInstanceLaunch(problem, *geometry, dilated));
}

TEST(TestGfx950ConvFwdGeometry, DirectInstanceLaunchAppliesTheGuardAndTheVariantRules)
{
    const DirectInstance spatial{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::SPATIAL, 0, 1};
    const DirectInstance std3{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::STD, 4, 1};
    const auto launches = [](const Problem& problem, const DirectInstance& instance) {
        const auto geometry = deriveGeometry(problem);
        return geometry && directInstanceLaunch(problem, *geometry, instance).has_value();
    };
    // Spatial serves fewer groups than a wave; std serves either side.
    EXPECT_TRUE(launches(depthwise(63, 14, 14, 3, 1, 1), spatial));
    EXPECT_FALSE(launches(depthwise(64, 14, 14, 3, 1, 1), spatial));
    EXPECT_TRUE(launches(depthwise(64, 14, 14, 3, 1, 1), std3));
    // The direct guard: channel multiplier, ungrouped, and rocKE's padding rule, each with
    // the instance's own compiled values.
    EXPECT_FALSE(launches(Problem{2, 16, 32, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}, std3));
    EXPECT_FALSE(launches(Problem{2, 16, 16, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1}, std3));
    const DirectInstance unpadded{{3, 3, 1, 1, 0, 0, 1, 1}, DirectVariant::STD, 4, 1};
    EXPECT_FALSE(launches(depthwise(24, 12, 12, 3, 1, 0), unpadded));
    // Malformed tuning is refused by the launch geometry.
    EXPECT_FALSE(launches(depthwise(32, 14, 14, 3, 1, 1),
                          {{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::SPATIAL, 4, 1}));
    EXPECT_FALSE(launches(depthwise(32, 14, 14, 3, 1, 1),
                          {{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONE, 4, 1}));
    EXPECT_FALSE(launches(depthwise(32, 14, 14, 3, 1, 1),
                          {{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::STD, 4, 17}));
}

/// One (instance, problem) case of the AOT direct variants and what
/// gfx950_conv_fwd_aot_serves, Gfx950ConvFwdAotSpec.grid() and .block() in
/// convolution_forward.py give for it: whether the instance serves the problem, and then
/// its grid and block. test_make_conv_aot_config.py recomputes every row from the adapter,
/// which test_convolution_forward_aot_packaging.py holds to rocKE's own plan_for.
struct DirectLaunchCase
{
    DirectInstance instance;
    Problem problem;
    bool served;
    unsigned int gridX;
    unsigned int gridY;
    unsigned int gridZ;
    unsigned int blockX;
};

// Instance: {filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w},
// variant, block_w, block_waves, block_h, block_q, block_groups, cpg, kpg, tile_h, tile_w,
// tile_k_out, ck, waves_m, waves_n, atom tile. Problem: Problem's field order.
// clang-format off
const std::vector<DirectLaunchCase>& directLaunchCases()
{
    static const std::vector<DirectLaunchCase> s_cases{
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 2, 2, 8, 33, 3, 3, 1, 1, 1, 1, 1, 1, 2}, true, 9, 1, 16, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 32, 32, 31, 7, 3, 3, 1, 1, 1, 1, 1, 1, 32}, true, 2, 1, 62, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 64, 64, 33, 28, 3, 3, 1, 1, 1, 1, 1, 1, 64}, true, 7, 1, 33, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 2, 2, 3, 31, 3, 3, 1, 1, 1, 1, 1, 1, 2}, true, 8, 1, 3, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 16, 16, 56, 2, 3, 3, 1, 1, 1, 1, 1, 1, 16}, true, 1, 1, 112, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 65, 65, 56, 7, 3, 3, 1, 1, 0, 0, 1, 1, 65}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 130, 130, 56, 7, 3, 3, 1, 2, 1, 1, 1, 1, 65}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 16, 16, 31, 3, 5, 5, 1, 1, 1, 1, 1, 1, 16}, false, 0, 0, 0, 0},
        {{{7, 7, 2, 2, 3, 3, 1, 1}, DirectVariant::TILED, 4, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 65, 65, 28, 31, 7, 7, 2, 2, 3, 3, 1, 1, 65}, true, 4, 1, 6, 128},
        {{{7, 7, 2, 2, 3, 3, 1, 1}, DirectVariant::TILED, 4, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 300, 300, 5, 28, 7, 7, 2, 2, 3, 3, 1, 1, 300}, true, 4, 3, 1, 128},
        {{{7, 7, 2, 2, 3, 3, 1, 1}, DirectVariant::TILED, 4, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 5, 5, 3, 1, 7, 7, 2, 2, 3, 3, 1, 1, 5}, true, 1, 1, 1, 128},
        {{{7, 7, 2, 2, 3, 3, 1, 1}, DirectVariant::TILED, 4, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 96, 96, 8, 28, 7, 7, 2, 2, 3, 3, 1, 1, 96}, true, 4, 1, 1, 128},
        {{{7, 7, 2, 2, 3, 3, 1, 1}, DirectVariant::TILED, 4, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 65, 65, 3, 8, 7, 7, 2, 2, 3, 3, 1, 1, 65}, true, 1, 1, 3, 128},
        {{{7, 7, 2, 2, 3, 3, 1, 1}, DirectVariant::TILED, 4, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 64, 64, 1, 7, 7, 7, 2, 2, 4, 4, 1, 1, 64}, false, 0, 0, 0, 0},
        {{{7, 7, 2, 2, 3, 3, 1, 1}, DirectVariant::TILED, 4, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 600, 600, 3, 8, 7, 7, 2, 2, 3, 3, 1, 1, 300}, false, 0, 0, 0, 0},
        {{{7, 7, 2, 2, 3, 3, 1, 1}, DirectVariant::TILED, 4, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 128, 128, 28, 33, 7, 7, 2, 3, 3, 3, 1, 1, 128}, false, 0, 0, 0, 0},
        {{{13, 13, 1, 1, 6, 6, 1, 1}, DirectVariant::TILED, 16, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 96, 96, 31, 2, 13, 13, 1, 1, 6, 6, 1, 1, 96}, true, 1, 2, 16, 64},
        {{{13, 13, 1, 1, 6, 6, 1, 1}, DirectVariant::TILED, 16, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 5, 5, 28, 12, 13, 13, 1, 1, 6, 6, 1, 1, 5}, true, 1, 1, 42, 64},
        {{{13, 13, 1, 1, 6, 6, 1, 1}, DirectVariant::TILED, 16, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 48, 48, 5, 3, 13, 13, 1, 1, 6, 6, 1, 1, 48}, true, 1, 1, 3, 64},
        {{{13, 13, 1, 1, 6, 6, 1, 1}, DirectVariant::TILED, 16, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 96, 96, 2, 5, 13, 13, 1, 1, 6, 6, 1, 1, 96}, true, 1, 2, 2, 64},
        {{{13, 13, 1, 1, 6, 6, 1, 1}, DirectVariant::TILED, 16, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 16, 16, 8, 2, 13, 13, 1, 1, 6, 6, 1, 1, 16}, true, 1, 1, 8, 64},
        {{{13, 13, 1, 1, 6, 6, 1, 1}, DirectVariant::TILED, 16, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 32, 32, 17, 31, 13, 13, 1, 1, 6, 6, 2, 2, 32}, false, 0, 0, 0, 0},
        {{{13, 13, 1, 1, 6, 6, 1, 1}, DirectVariant::TILED, 16, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 128, 128, 8, 5, 13, 13, 2, 2, 6, 6, 1, 1, 128}, false, 0, 0, 0, 0},
        {{{13, 13, 1, 1, 6, 6, 1, 1}, DirectVariant::TILED, 16, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 8, 8, 17, 3, 13, 13, 1, 2, 6, 6, 1, 1, 8}, false, 0, 0, 0, 0},
        {{{21, 21, 1, 1, 10, 10, 1, 1}, DirectVariant::TILED, 14, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 1, 1, 28, 31, 21, 21, 1, 1, 10, 10, 1, 1, 1}, true, 3, 1, 56, 128},
        {{{21, 21, 1, 1, 10, 10, 1, 1}, DirectVariant::TILED, 14, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 8, 8, 33, 31, 21, 21, 1, 1, 10, 10, 1, 1, 8}, true, 3, 1, 33, 128},
        {{{21, 21, 1, 1, 10, 10, 1, 1}, DirectVariant::TILED, 14, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 16, 16, 5, 7, 21, 21, 1, 1, 10, 10, 1, 1, 16}, true, 1, 1, 10, 128},
        {{{21, 21, 1, 1, 10, 10, 1, 1}, DirectVariant::TILED, 14, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 300, 300, 3, 33, 21, 21, 1, 1, 10, 10, 1, 1, 300}, true, 3, 3, 9, 128},
        {{{21, 21, 1, 1, 10, 10, 1, 1}, DirectVariant::TILED, 14, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 2, 2, 5, 12, 21, 21, 1, 1, 10, 10, 1, 1, 2}, true, 1, 1, 10, 128},
        {{{21, 21, 1, 1, 10, 10, 1, 1}, DirectVariant::TILED, 14, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 4, 4, 3, 12, 21, 21, 1, 1, 10, 10, 1, 1, 2}, false, 0, 0, 0, 0},
        {{{21, 21, 1, 1, 10, 10, 1, 1}, DirectVariant::TILED, 14, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 63, 63, 33, 28, 21, 21, 1, 1, 2, 2, 1, 1, 63}, false, 0, 0, 0, 0},
        {{{21, 21, 1, 1, 10, 10, 1, 1}, DirectVariant::TILED, 14, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 192, 192, 1, 56, 21, 21, 1, 1, 10, 10, 1, 1, 96}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 0, 0, 1, 1}, DirectVariant::COL, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 32, 32, 28, 28, 3, 3, 1, 1, 0, 0, 1, 1, 32}, true, 26, 1, 4, 64},
        {{{3, 3, 1, 1, 0, 0, 1, 1}, DirectVariant::COL, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 16, 16, 3, 7, 3, 3, 1, 1, 0, 0, 1, 1, 16}, true, 5, 1, 1, 64},
        {{{3, 3, 1, 1, 0, 0, 1, 1}, DirectVariant::COL, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 16, 16, 7, 8, 3, 3, 1, 1, 0, 0, 1, 1, 16}, true, 6, 1, 1, 64},
        {{{3, 3, 1, 1, 0, 0, 1, 1}, DirectVariant::COL, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 16, 16, 56, 8, 3, 3, 1, 1, 0, 0, 1, 1, 16}, true, 6, 1, 7, 64},
        {{{3, 3, 1, 1, 0, 0, 1, 1}, DirectVariant::COL, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 5, 5, 7, 5, 3, 3, 1, 1, 0, 0, 1, 1, 5}, true, 3, 1, 1, 64},
        {{{3, 3, 1, 1, 0, 0, 1, 1}, DirectVariant::COL, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 300, 300, 1, 7, 3, 3, 1, 1, 1, 1, 1, 1, 300}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 0, 0, 1, 1}, DirectVariant::COL, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 16, 16, 3, 12, 3, 3, 1, 2, 0, 0, 1, 1, 16}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 0, 0, 1, 1}, DirectVariant::COL, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 8, 16, 3, 12, 3, 3, 1, 1, 0, 0, 1, 1, 8}, false, 0, 0, 0, 0},
        {{{5, 5, 1, 1, 2, 2, 1, 1}, DirectVariant::COL, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 16, 16, 7, 56, 5, 5, 1, 1, 2, 2, 1, 1, 16}, true, 14, 1, 2, 128},
        {{{5, 5, 1, 1, 2, 2, 1, 1}, DirectVariant::COL, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 64, 64, 28, 8, 5, 5, 1, 1, 2, 2, 1, 1, 64}, true, 2, 1, 12, 128},
        {{{5, 5, 1, 1, 2, 2, 1, 1}, DirectVariant::COL, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 300, 300, 33, 33, 5, 5, 1, 1, 2, 2, 1, 1, 300}, true, 9, 3, 10, 128},
        {{{5, 5, 1, 1, 2, 2, 1, 1}, DirectVariant::COL, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 63, 63, 5, 33, 5, 5, 1, 1, 2, 2, 1, 1, 63}, true, 9, 1, 1, 128},
        {{{5, 5, 1, 1, 2, 2, 1, 1}, DirectVariant::COL, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 1, 1, 31, 31, 5, 5, 1, 1, 2, 2, 1, 1, 1}, true, 8, 1, 12, 128},
        {{{5, 5, 1, 1, 2, 2, 1, 1}, DirectVariant::COL, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 8, 16, 1, 17, 5, 5, 1, 1, 2, 2, 1, 1, 8}, false, 0, 0, 0, 0},
        {{{5, 5, 1, 1, 2, 2, 1, 1}, DirectVariant::COL, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 8, 8, 12, 2, 5, 5, 1, 2, 2, 2, 1, 1, 8}, false, 0, 0, 0, 0},
        {{{5, 5, 1, 1, 2, 2, 1, 1}, DirectVariant::COL, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 65, 65, 12, 12, 3, 3, 1, 1, 2, 2, 1, 1, 65}, false, 0, 0, 0, 0},
        {{{7, 7, 2, 2, 0, 0, 1, 1}, DirectVariant::COL, 2, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 8, 8, 28, 33, 7, 7, 2, 2, 0, 0, 1, 1, 8}, true, 7, 1, 2, 256},
        {{{7, 7, 2, 2, 0, 0, 1, 1}, DirectVariant::COL, 2, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 16, 16, 56, 17, 7, 7, 2, 2, 0, 0, 1, 1, 16}, true, 3, 1, 4, 256},
        {{{7, 7, 2, 2, 0, 0, 1, 1}, DirectVariant::COL, 2, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 96, 96, 7, 33, 7, 7, 2, 2, 0, 0, 1, 1, 96}, true, 7, 1, 1, 256},
        {{{7, 7, 2, 2, 0, 0, 1, 1}, DirectVariant::COL, 2, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 63, 63, 33, 31, 7, 7, 2, 2, 0, 0, 1, 1, 63}, true, 7, 1, 2, 256},
        {{{7, 7, 2, 2, 0, 0, 1, 1}, DirectVariant::COL, 2, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 96, 96, 33, 12, 7, 7, 2, 2, 0, 0, 1, 1, 96}, true, 2, 1, 1, 256},
        {{{7, 7, 2, 2, 0, 0, 1, 1}, DirectVariant::COL, 2, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 300, 300, 28, 33, 1, 1, 2, 2, 0, 0, 1, 1, 300}, false, 0, 0, 0, 0},
        {{{7, 7, 2, 2, 0, 0, 1, 1}, DirectVariant::COL, 2, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 2, 2, 31, 12, 7, 7, 2, 3, 0, 0, 1, 1, 2}, false, 0, 0, 0, 0},
        {{{7, 7, 2, 2, 0, 0, 1, 1}, DirectVariant::COL, 2, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 2, 2, 17, 56, 7, 7, 2, 3, 0, 0, 1, 1, 2}, false, 0, 0, 0, 0},
        {{{9, 9, 2, 2, 4, 4, 1, 1}, DirectVariant::COL, 2, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 2, 2, 56, 28, 9, 9, 2, 2, 4, 4, 1, 1, 2}, true, 7, 1, 2, 64},
        {{{9, 9, 2, 2, 4, 4, 1, 1}, DirectVariant::COL, 2, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 96, 96, 28, 17, 9, 9, 2, 2, 4, 4, 1, 1, 96}, true, 5, 2, 3, 64},
        {{{9, 9, 2, 2, 4, 4, 1, 1}, DirectVariant::COL, 2, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 7, 31, 9, 9, 2, 2, 4, 4, 1, 1, 1}, true, 8, 1, 1, 64},
        {{{9, 9, 2, 2, 4, 4, 1, 1}, DirectVariant::COL, 2, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 5, 5, 12, 56, 9, 9, 2, 2, 4, 4, 1, 1, 5}, true, 14, 1, 3, 64},
        {{{9, 9, 2, 2, 4, 4, 1, 1}, DirectVariant::COL, 2, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 2, 2, 33, 56, 9, 9, 2, 2, 4, 4, 1, 1, 2}, true, 14, 1, 1, 64},
        {{{9, 9, 2, 2, 4, 4, 1, 1}, DirectVariant::COL, 2, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 2, 2, 33, 56, 9, 9, 2, 2, 5, 5, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{9, 9, 2, 2, 4, 4, 1, 1}, DirectVariant::COL, 2, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 300, 300, 28, 5, 9, 9, 1, 1, 4, 4, 1, 1, 300}, false, 0, 0, 0, 0},
        {{{9, 9, 2, 2, 4, 4, 1, 1}, DirectVariant::COL, 2, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 16, 16, 1, 28, 9, 9, 1, 1, 4, 4, 1, 1, 16}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {2, 256, 256, 7, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64}, true, 4, 64, 2, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {2, 12, 12, 31, 33, 3, 3, 1, 1, 1, 1, 1, 1, 3}, true, 3, 3, 4, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {3, 24, 24, 33, 5, 3, 3, 1, 1, 1, 1, 1, 1, 6}, true, 1, 6, 9, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {1, 8, 8, 31, 8, 3, 3, 1, 1, 1, 1, 1, 1, 2}, true, 1, 2, 2, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {2, 4, 4, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 2, 1, 4, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {2, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 2, 2, 32}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {1, 24, 24, 7, 5, 3, 3, 2, 2, 1, 1, 1, 1, 6}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0}, {1, 512, 512, 12, 8, 3, 3, 1, 1, 2, 2, 1, 1, 128}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0}, {2, 384, 384, 12, 28, 3, 3, 1, 1, 1, 1, 1, 1, 48}, true, 2, 6, 2, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0}, {3, 128, 128, 28, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}, true, 1, 2, 6, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0}, {3, 512, 512, 31, 33, 3, 3, 1, 1, 1, 1, 1, 1, 64}, true, 3, 8, 6, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0}, {1, 512, 512, 8, 31, 3, 3, 1, 1, 1, 1, 1, 1, 64}, true, 2, 8, 1, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0}, {2, 1024, 1024, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1, 128}, true, 1, 16, 2, 512},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0}, {2, 48, 48, 3, 12, 3, 3, 1, 1, 1, 1, 1, 1, 6}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0}, {1, 32, 32, 12, 33, 3, 3, 1, 1, 2, 2, 1, 1, 4}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0}, {1, 384, 384, 33, 17, 3, 3, 2, 2, 1, 1, 1, 1, 48}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 4, 16, 16, 0, 0, 0, 0, 0, 0, 0}, {2, 384, 384, 3, 12, 3, 3, 1, 1, 1, 1, 1, 1, 24}, true, 1, 6, 2, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 4, 16, 16, 0, 0, 0, 0, 0, 0, 0}, {1, 768, 768, 17, 28, 3, 3, 1, 1, 1, 1, 1, 1, 48}, true, 1, 12, 2, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 4, 16, 16, 0, 0, 0, 0, 0, 0, 0}, {2, 64, 64, 28, 33, 3, 3, 1, 1, 1, 1, 1, 1, 4}, true, 2, 1, 4, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 4, 16, 16, 0, 0, 0, 0, 0, 0, 0}, {2, 64, 64, 2, 17, 3, 3, 1, 1, 1, 1, 1, 1, 4}, true, 1, 1, 2, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 4, 16, 16, 0, 0, 0, 0, 0, 0, 0}, {1, 768, 768, 1, 28, 3, 3, 1, 1, 1, 1, 1, 1, 48}, true, 1, 12, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 4, 16, 16, 0, 0, 0, 0, 0, 0, 0}, {1, 48, 48, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 3}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 4, 16, 16, 0, 0, 0, 0, 0, 0, 0}, {3, 384, 384, 31, 17, 3, 3, 2, 2, 1, 1, 1, 1, 24}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 32, 4, 16, 16, 0, 0, 0, 0, 0, 0, 0}, {1, 16, 16, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {3, 1536, 1536, 31, 17, 3, 3, 1, 1, 1, 1, 1, 1, 48}, true, 2, 48, 6, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {1, 512, 512, 28, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}, true, 1, 16, 2, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {1, 192, 192, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 6}, true, 2, 6, 2, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {2, 64, 64, 3, 5, 3, 3, 1, 1, 1, 1, 1, 1, 2}, true, 1, 2, 2, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {1, 64, 64, 2, 17, 3, 3, 1, 1, 1, 1, 1, 1, 2}, true, 2, 2, 1, 64},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {3, 192, 384, 8, 8, 3, 3, 1, 2, 1, 1, 1, 1, 6}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {2, 1024, 1024, 3, 7, 3, 3, 1, 1, 0, 0, 1, 1, 32}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0, 16, 16, 1, 32, 32, 0, 0, 0, 0, 0, 0, 0}, {1, 1536, 1536, 31, 8, 3, 3, 1, 2, 1, 1, 1, 1, 48}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {1, 64, 64, 17, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 6, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {2, 96, 32, 33, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 10, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {2, 96, 64, 33, 30, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 20, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {3, 96, 256, 28, 30, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 96, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {2, 16, 128, 28, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1}, true, 64, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {1, 48, 16, 8, 30, 3, 3, 1, 1, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {2, 256, 512, 3, 30, 3, 3, 1, 1, 1, 1, 1, 1, 2}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 8, 32, 32, 16, 1, 4, 32}, {1, 96, 24, 17, 96, 3, 3, 2, 2, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 128, 32, 1, 4, 16}, {1, 256, 16, 3, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1}, true, 2, 1, 1, 256},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 128, 32, 1, 4, 16}, {2, 64, 48, 8, 56, 3, 3, 2, 2, 1, 1, 1, 1, 1}, true, 4, 1, 1, 256},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 128, 32, 1, 4, 16}, {2, 96, 256, 33, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1}, true, 8, 1, 1, 256},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 128, 32, 1, 4, 16}, {2, 32, 96, 33, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1}, true, 4, 1, 1, 256},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 128, 32, 1, 4, 16}, {1, 128, 96, 13, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1}, true, 1, 1, 1, 256},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 128, 32, 1, 4, 16}, {3, 16, 48, 3, 100, 3, 3, 1, 1, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 128, 32, 1, 4, 16}, {2, 24, 48, 8, 96, 3, 3, 2, 2, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 2, 2, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 128, 32, 1, 4, 16}, {2, 96, 64, 3, 56, 5, 5, 2, 2, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 3, 3, 3, 3}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 32, 32, 2, 4, 16}, {1, 32, 160, 28, 30, 3, 3, 1, 1, 3, 3, 3, 3, 1}, true, 20, 1, 1, 512},
        {{{3, 3, 1, 1, 3, 3, 3, 3}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 32, 32, 2, 4, 16}, {1, 128, 48, 33, 192, 3, 3, 1, 1, 3, 3, 3, 3, 1}, true, 72, 1, 1, 512},
        {{{3, 3, 1, 1, 3, 3, 3, 3}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 32, 32, 2, 4, 16}, {2, 64, 160, 17, 32, 3, 3, 1, 1, 3, 3, 3, 3, 1}, true, 40, 1, 1, 512},
        {{{3, 3, 1, 1, 3, 3, 3, 3}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 32, 32, 2, 4, 16}, {1, 32, 96, 13, 192, 3, 3, 1, 1, 3, 3, 3, 3, 1}, true, 36, 1, 1, 512},
        {{{3, 3, 1, 1, 3, 3, 3, 3}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 32, 32, 2, 4, 16}, {1, 96, 32, 33, 48, 3, 3, 1, 1, 3, 3, 3, 3, 1}, true, 9, 1, 1, 512},
        {{{3, 3, 1, 1, 3, 3, 3, 3}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 32, 32, 2, 4, 16}, {2, 96, 256, 17, 48, 3, 3, 2, 2, 3, 3, 3, 3, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 3, 3, 3, 3}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 32, 32, 2, 4, 16}, {3, 128, 16, 13, 48, 3, 3, 1, 1, 0, 0, 3, 3, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 3, 3, 3, 3}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 32, 32, 2, 4, 16}, {1, 24, 256, 17, 64, 3, 3, 1, 1, 3, 3, 3, 3, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 3, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 2, 16}, {1, 64, 160, 17, 96, 3, 3, 1, 1, 1, 1, 3, 1, 1}, true, 18, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 3, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 2, 16}, {1, 32, 256, 28, 28, 3, 3, 1, 1, 1, 1, 3, 1, 1}, true, 16, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 3, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 2, 16}, {2, 128, 16, 8, 16, 3, 3, 1, 1, 1, 1, 3, 1, 1}, true, 2, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 3, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 2, 16}, {2, 64, 48, 28, 32, 3, 3, 1, 1, 1, 1, 3, 1, 1}, true, 8, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 3, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 2, 16}, {2, 128, 48, 17, 30, 3, 3, 1, 1, 1, 1, 3, 1, 1}, true, 4, 1, 1, 256},
        {{{3, 3, 1, 1, 1, 1, 3, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 2, 16}, {2, 24, 96, 8, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 3, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 2, 16}, {2, 16, 320, 8, 64, 3, 3, 2, 2, 1, 1, 3, 1, 2}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 1, 1, 3, 1}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 16, 64, 32, 2, 2, 16}, {1, 32, 160, 33, 30, 3, 3, 1, 2, 1, 1, 3, 1, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 2, 2, 1, 2}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 32, 64, 1, 8, 16}, {3, 128, 64, 33, 64, 3, 3, 1, 1, 2, 2, 1, 2, 1}, true, 36, 1, 1, 512},
        {{{3, 3, 1, 1, 2, 2, 1, 2}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 32, 64, 1, 8, 16}, {2, 64, 16, 13, 192, 3, 3, 1, 1, 2, 2, 1, 2, 1}, true, 12, 1, 1, 512},
        {{{3, 3, 1, 1, 2, 2, 1, 2}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 32, 64, 1, 8, 16}, {1, 256, 96, 3, 64, 3, 3, 1, 1, 2, 2, 1, 2, 1}, true, 6, 1, 1, 512},
        {{{3, 3, 1, 1, 2, 2, 1, 2}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 32, 64, 1, 8, 16}, {1, 128, 96, 13, 96, 3, 3, 1, 1, 2, 2, 1, 2, 1}, true, 9, 1, 1, 512},
        {{{3, 3, 1, 1, 2, 2, 1, 2}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 32, 64, 1, 8, 16}, {1, 96, 16, 33, 30, 5, 5, 1, 1, 2, 2, 1, 2, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 2, 2, 1, 2}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 32, 64, 1, 8, 16}, {2, 8, 32, 3, 48, 3, 3, 1, 1, 2, 2, 2, 2, 1}, false, 0, 0, 0, 0},
        {{{3, 3, 1, 1, 2, 2, 1, 2}, DirectVariant::NONGROUPED, 0, 0, 0, 0, 0, 0, 0, 16, 32, 32, 64, 1, 8, 16}, {2, 8, 96, 8, 56, 3, 3, 1, 1, 2, 2, 1, 2, 1}, false, 0, 0, 0, 0},
    };
    return s_cases;
}
// clang-format on

TEST(TestGfx950ConvFwdGeometry, AotDirectVariantsLaunchWhereAndHowTheAdapterDoes)
{
    size_t served = 0;
    for(size_t i = 0; i < directLaunchCases().size(); ++i)
    {
        SCOPED_TRACE(i);
        const auto& entry = directLaunchCases()[i];
        const auto geometry = deriveGeometry(entry.problem);
        ASSERT_TRUE(geometry);
        const auto launch = directInstanceLaunch(entry.problem, *geometry, entry.instance);
        ASSERT_EQ(launch.has_value(), entry.served);
        if(launch)
        {
            ++served;
            EXPECT_EQ(std::tie(launch->gridX, launch->gridY, launch->gridZ, launch->blockX),
                      std::tie(entry.gridX, entry.gridY, entry.gridZ, entry.blockX));
        }
    }
    // Both verdicts for every variant.
    EXPECT_GT(served, size_t{0});
    EXPECT_LT(served, directLaunchCases().size());
}

TEST(TestGfx950ConvFwdGeometry, NongroupedTileWidthsAreTheOnesASweepOfWoTries)
{
    // tile_w_candidates in conv_direct_nongrouped.py: the exact divisors of Wo among one
    // to six atoms, else the first width with the least waste.
    for(const auto [tileW, wo, atom, fits] : std::vector<std::tuple<int, int, int, bool>>{
            {32, 96, 32, true},
            {96, 96, 32, true},
            {64, 96, 32, false},
            {48, 96, 16, true},
            {80, 96, 16, false},
            {32, 93, 32, true},
            {64, 93, 32, false},
            {16, 7, 16, true},
            {32, 7, 16, false},
            {16, 48, 16, true},
            {32, 48, 16, false},
            {32, 48, 32, true},
            {192, 192, 32, true},
            {224, 224, 32, false},
        })
    {
        SCOPED_TRACE(std::to_string(tileW) + " " + std::to_string(wo));
        EXPECT_EQ(nongroupedTileWidthFits(tileW, wo, atom), fits);
    }
    EXPECT_EQ(nongroupedAtomTile("32x32x16"), 32);
    EXPECT_EQ(nongroupedAtomTile("32x32x8"), 32);
    EXPECT_EQ(nongroupedAtomTile("16x16x32"), 16);
    EXPECT_EQ(nongroupedAtomTile("16x16x16"), 16);
    EXPECT_FALSE(nongroupedAtomTile("none"));
    EXPECT_FALSE(nongroupedAtomTile("8x8x8"));
}

TEST(TestGfx950ConvFwdGeometry, AotDirectVariantsRefuseWhatRockesValidatorsRefuse)
{
    const auto launches = [](const Problem& problem, const DirectInstance& instance) {
        const auto geometry = deriveGeometry(problem);
        return geometry && directInstanceLaunch(problem, *geometry, instance).has_value();
    };
    DirectInstance tiled{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::TILED, 4, 1};
    tiled.blockH = 2;
    EXPECT_TRUE(launches(depthwise(32, 14, 14, 3, 1, 1), tiled));
    // Pure depthwise only, and one stride and padding for both axes.
    EXPECT_FALSE(launches(Problem{2, 16, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 16}, tiled));
    EXPECT_FALSE(launches(Problem{2, 16, 16, 14, 14, 3, 3, 1, 2, 1, 1, 1, 1, 16}, tiled));
    auto noRows = tiled;
    noRows.blockH = 0;
    EXPECT_FALSE(launches(depthwise(32, 14, 14, 3, 1, 1), noRows));

    // col: "valid" padding, rows bounded by the output; a block no wider than Wo.
    DirectInstance col{{5, 5, 2, 2, 0, 0, 1, 1}, DirectVariant::COL, 4, 1};
    col.blockH = 8;
    EXPECT_TRUE(launches(Problem{2, 24, 24, 13, 17, 5, 5, 2, 2, 0, 0, 1, 1, 24}, col));
    EXPECT_FALSE(launches(Problem{2, 24, 24, 13, 6, 5, 5, 2, 2, 0, 0, 1, 1, 24}, col));
    EXPECT_FALSE(launches(Problem{2, 24, 24, 13, 17, 5, 5, 2, 2, 2, 2, 1, 1, 24}, col));

    // grouped: its compiled channel counts and a group count block_groups divides.
    DirectInstance grouped{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::GROUPED, 0, 0};
    grouped.blockH = 16;
    grouped.blockQ = 32;
    grouped.blockGroups = 4;
    grouped.cpg = grouped.kpg = 8;
    EXPECT_TRUE(launches(Problem{2, 64, 64, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 8}, grouped));
    EXPECT_FALSE(launches(Problem{2, 48, 48, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 6}, grouped));
    EXPECT_FALSE(launches(Problem{2, 64, 128, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 8}, grouped));
    EXPECT_FALSE(launches(Problem{2, 128, 128, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 8}, grouped));

    // nongrouped: groups == 1, C a multiple of ck and 8, K of the atom tile.
    DirectInstance nongrouped{{3, 3, 1, 1, 1, 1, 1, 1}, DirectVariant::NONGROUPED, 0, 0};
    nongrouped.tileH = 8;
    nongrouped.tileW = 32;
    nongrouped.tileKOut = 64;
    nongrouped.ck = 16;
    nongrouped.wavesM = 2;
    nongrouped.wavesN = 2;
    nongrouped.atomTile = 32;
    const Problem dense{1, 64, 96, 10, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1};
    const auto geometry = deriveGeometry(dense);
    ASSERT_TRUE(geometry);
    const auto launch = directInstanceLaunch(dense, *geometry, nongrouped);
    ASSERT_TRUE(launch);
    // 3 column tiles x 2 row tiles x 1 image x 2 channel tiles, flat.
    EXPECT_EQ(std::tie(launch->gridX, launch->gridY, launch->gridZ, launch->blockX),
              std::make_tuple(12U, 1U, 1U, 256U));
    for(const auto& other : {Problem{1, 128, 192, 10, 96, 3, 3, 1, 1, 1, 1, 1, 1, 2},
                             Problem{1, 56, 96, 10, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1},
                             Problem{1, 64, 80, 10, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1},
                             Problem{1, 64, 96, 10, 96, 3, 3, 1, 1, 1, 1, 2, 2, 1}})
    {
        EXPECT_FALSE(launches(other, nongrouped));
    }
    // Every tensor must end below the kernel's masked-access offset.
    const Problem huge{680, 64, 32, 256, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1};
    const auto hugeGeometry = deriveGeometry(huge);
    ASSERT_TRUE(hugeGeometry);
    EXPECT_GT(hugeGeometry->tensorBytes[0], NONGROUPED_OOB_BASE);
    EXPECT_FALSE(directInstanceLaunch(huge, *hugeGeometry, nongrouped));
}

TEST(TestGfx950ConvFwdGeometry, DefaultDirectArmFollowsTheGroupCount)
{
    // gfx950_conv_fwd_direct_spec_for_request with no overrides.
    EXPECT_TRUE(isDefaultDirectArm(63, DirectVariant::SPATIAL, 0, 1));
    EXPECT_FALSE(isDefaultDirectArm(63, DirectVariant::SPATIAL, 0, 2));
    EXPECT_FALSE(isDefaultDirectArm(63, DirectVariant::STD, 4, 1));
    EXPECT_TRUE(isDefaultDirectArm(64, DirectVariant::STD, 4, 1));
    EXPECT_FALSE(isDefaultDirectArm(64, DirectVariant::STD, 16, 1));
    EXPECT_FALSE(isDefaultDirectArm(64, DirectVariant::SPATIAL, 0, 1));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
