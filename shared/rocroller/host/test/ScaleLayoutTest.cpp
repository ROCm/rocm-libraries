// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <rocRoller/hostnumerics/ScaleLayout.hpp>
#include <stdexcept>
#include <vector>

using namespace roc::hostnumerics;
using namespace rocRoller::hostnumerics;

namespace
{
    // The factors of a natural element index, fastest first, as the kernel's VGPRs,
    // SIMDs and lanes index a swizzle tile: [VGPR index, VGPR block, SIMD block,
    // reduction swizzle tile, (reduction tile), lane, SIMD index index, SIMD index block,
    // non-reduction swizzle tile, (non-reduction tile)]. The tile factors are present only with a tile.
    std::vector<size_t> SwizzleFactorSizes(size_t                      reduction,
                                           size_t                      nonReduction,
                                           const ScaleSwizzle&         swizzle,
                                           const std::optional<Shape>& tile)
    {
        const size_t lanesPerSimd = 16, simdsPerWave = 4;
        const size_t simdIndex      = swizzle.tileNonReduction / lanesPerSimd;
        const size_t simdBlock      = simdsPerWave / simdIndex;
        const size_t vgprIndex      = std::min(simdIndex, swizzle.subTileReduction);
        const size_t vgprBlock      = swizzle.tileReduction / simdBlock / vgprIndex;
        const size_t simdIndexBlock = vgprIndex;
        const size_t simdIndexIndex = simdIndex / simdIndexBlock;
        if(!tile)
            return {vgprIndex,
                    vgprBlock,
                    simdBlock,
                    reduction / swizzle.tileReduction,
                    lanesPerSimd,
                    simdIndexIndex,
                    simdIndexBlock,
                    nonReduction / swizzle.tileNonReduction};
        return {vgprIndex,
                vgprBlock,
                simdBlock,
                (*tile)[1] / swizzle.tileReduction,
                reduction / (*tile)[1],
                lanesPerSimd,
                simdIndexIndex,
                simdIndexBlock,
                (*tile)[0] / swizzle.tileNonReduction,
                nonReduction / (*tile)[0]};
    }

    // Stores each element at the offset whose index factors are ordered by
    // `physicalOrder`, fastest first.
    template <typename T>
    std::vector<T> ReorderFactors(const std::vector<T>&      input,
                                  const std::vector<size_t>& sizes,
                                  const std::vector<size_t>& physicalOrder)
    {
        std::vector<size_t> strides(sizes.size());
        size_t              stride = 1;
        for(const size_t dimension : physicalOrder)
        {
            strides[dimension] = stride;
            stride *= sizes[dimension];
        }
        std::vector<T>      output(input.size());
        std::vector<size_t> coordinate(sizes.size(), 0);
        for(size_t source = 0; source < input.size(); ++source)
        {
            size_t destination = 0;
            for(size_t dimension = 0; dimension < sizes.size(); ++dimension)
                destination += coordinate[dimension] * strides[dimension];
            output[destination] = input[source];
            for(size_t dimension = 0;
                dimension < sizes.size() && ++coordinate[dimension] == sizes[dimension];
                ++dimension)
                coordinate[dimension] = 0;
        }
        return output;
    }

    struct SwizzleCase
    {
        size_t               reduction;
        size_t               nonReduction;
        ScaleSwizzle         swizzle;
        std::optional<Shape> tile;
        std::vector<size_t>  physicalOrder;
    };

    template <typename T>
    void ExpectKernelOrder(const SwizzleCase& test)
    {
        std::vector<T> input(test.reduction * test.nonReduction);
        std::iota(input.begin(), input.end(), T{0});
        const std::vector<T> expected = ReorderFactors(
            input,
            SwizzleFactorSizes(test.reduction, test.nonReduction, test.swizzle, test.tile),
            test.physicalOrder);
        const auto transform
            = ScaleStoragePermutation({test.nonReduction, test.reduction}, test.swizzle, test.tile);
        std::vector<T> output(transform.physicalElementCount());
        copyWithReshapePermutation<T>(input, output, transform);
        EXPECT_EQ(output, expected)
            << "tileReduction " << test.swizzle.tileReduction << " tileNonReduction "
            << test.swizzle.tileNonReduction << " subTileReduction "
            << test.swizzle.subTileReduction << (test.tile ? " with tile" : "");
    }

    // The kernel's load order for each swizzle: tileNonReduction 64 and tileNonReduction 32 with
    // subTileReduction 4 swap the SIMD index block with the VGPR index; tileNonReduction 32 with
    // subTileReduction 4 also swaps the SIMD and VGPR blocks; tileNonReduction 32 with subTileReduction
    // 2 rotates the VGPR index, VGPR block and SIMD block. A tile moves its own factors (reduction tile
    // and non-reduction tile) to the slowest positions.
    const std::vector<SwizzleCase>& SwizzleCases()
    {
        static const std::vector<SwizzleCase> cases{
            {256, 128, ScaleSwizzle{256, 64, 4}, std::nullopt, {6, 1, 2, 3, 4, 5, 0, 7}},
            {128, 64, ScaleSwizzle{128, 32, 4}, std::nullopt, {6, 2, 1, 3, 4, 5, 0, 7}},
            {128, 64, ScaleSwizzle{128, 32, 2}, std::nullopt, {1, 2, 0, 3, 4, 5, 6, 7}},
            {256, 128, ScaleSwizzle{256, 64, 4}, Shape{64, 256}, {7, 1, 2, 3, 5, 6, 0, 8, 4, 9}},
            {512, 128, ScaleSwizzle{128, 32, 4}, Shape{32, 128}, {7, 2, 1, 3, 5, 6, 0, 8, 4, 9}},
            {256, 64, ScaleSwizzle{128, 32, 2}, Shape{64, 128}, {1, 2, 0, 3, 5, 6, 7, 8, 4, 9}},
        };
        return cases;
    }

    TEST(ScaleLayoutTest, SwizzlesIntoKernelLoadOrder)
    {
        for(const SwizzleCase& test : SwizzleCases())
            ExpectKernelOrder<float>(test);
    }

    TEST(ScaleLayoutTest, SwizzlesDoubleIntegerAndByteElements)
    {
        for(const SwizzleCase& test : SwizzleCases())
        {
            ExpectKernelOrder<double>(test);
            ExpectKernelOrder<int>(test);
            ExpectKernelOrder<uint8_t>(test);
        }
    }

    TEST(ScaleLayoutTest, NoSwizzleUsesOrdinaryTilingOrIdentity)
    {
        const std::vector<int> input{0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<int>       output(input.size());
        copyWithReshapePermutation<int>(
            input, output, ScaleStoragePermutation({2, 4}, std::nullopt, std::nullopt));
        EXPECT_EQ(output, input);
        copyWithReshapePermutation<int>(
            input, output, ScaleStoragePermutation({2, 4}, std::nullopt, Shape{2, 2}));
        EXPECT_EQ(output, (std::vector<int>{0, 1, 4, 5, 2, 3, 6, 7}));
    }

    TEST(ScaleLayoutTest, RejectsUnsupportedSwizzles)
    {
        for(const ScaleSwizzle& swizzle : {ScaleSwizzle{128, 48, 4},
                                           ScaleSwizzle{0, 32, 4},
                                           ScaleSwizzle{126, 32, 4},
                                           ScaleSwizzle{128, 32, 0},
                                           ScaleSwizzle{128, 32, 1},
                                           ScaleSwizzle{128, 32, 3},
                                           ScaleSwizzle{128, 64, 3}})
            EXPECT_THROW(ScaleStoragePermutation({64, 128}, swizzle, std::nullopt),
                         std::runtime_error);
    }

    TEST(ScaleLayoutTest, RejectsIndivisibleExtentsAndTiles)
    {
        const ScaleSwizzle swizzle{128, 32, 4};
        for(const Shape& tile :
            {Shape{16, 0}, Shape{0, 16}, Shape{16, 17}, Shape{32, 96}, Shape{48, 128}})
            EXPECT_THROW(ScaleStoragePermutation({64, 128}, swizzle, tile), std::runtime_error);
        EXPECT_THROW(ScaleStoragePermutation({64, 128}, ScaleSwizzle{256, 32, 4}, Shape{64, 128}),
                     std::runtime_error);
        EXPECT_THROW(ScaleStoragePermutation({64, 128}, ScaleSwizzle{128, 64, 4}, Shape{32, 128}),
                     std::runtime_error);
        EXPECT_THROW(ScaleStoragePermutation({64, 129}, swizzle, std::nullopt), std::runtime_error);
        EXPECT_THROW(ScaleStoragePermutation({65, 128}, swizzle, std::nullopt), std::runtime_error);
        EXPECT_THROW(ScaleStoragePermutation({64, 128, 1}, swizzle, std::nullopt),
                     std::runtime_error);
        EXPECT_THROW(ScaleStoragePermutation({64, 128}, swizzle, Shape{128}), std::runtime_error);
    }

    TEST(ScaleLayoutTest, EmptyTensorsStillRequireDivisibleExtents)
    {
        const ScaleSwizzle swizzle{128, 32, 4};
        EXPECT_EQ(ScaleStoragePermutation({64, 0}, swizzle, std::nullopt).physicalElementCount(),
                  0u);
        EXPECT_EQ(
            ScaleStoragePermutation({0, 128}, std::nullopt, Shape{16, 16}).physicalElementCount(),
            0u);
        EXPECT_THROW(ScaleStoragePermutation({0, 4}, swizzle, std::nullopt), std::runtime_error);
        EXPECT_THROW(ScaleStoragePermutation({4, 0}, std::nullopt, Shape{16, 16}),
                     std::runtime_error);
    }
} // namespace
