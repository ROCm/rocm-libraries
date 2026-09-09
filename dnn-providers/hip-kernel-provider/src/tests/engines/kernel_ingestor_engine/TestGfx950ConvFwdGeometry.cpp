// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <limits>

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

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
