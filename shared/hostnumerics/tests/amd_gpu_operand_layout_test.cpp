// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <roc/hostnumerics/amd_gpu_layout/operand.hpp>
#include <stdexcept>
#include <vector>

using namespace roc::hostnumerics::amd_gpu_layout;

namespace {
// ============================================================================
// Tutorial: preparing an operand for an AMD GPU kernel
// ============================================================================

// An operand tile stores each tile contiguously. reduction is the contiguous extent, in
// the input and within each output tile. Here a 4 x 4 operand is split into
// 2 x 2 tiles; no swizzle is requested.
TEST(AmdGpuOperandLayoutTutorial, TileOperand) {
    // Element (reduction, nonReduction) is stored at reduction + 4 * nonReduction and holds 10 *
    // nonReduction + reduction.
    std::vector<int> operand(16);
    for (int nonReduction = 0; nonReduction < 4; ++nonReduction)
        for (int reduction = 0; reduction < 4; ++reduction)
            operand[reduction + 4 * nonReduction] = 10 * nonReduction + reduction;

    const std::vector<int> tiled =
        copyOperandToPhysicalLayout(operand, OperandExtents{/*reduction=*/4, /*nonReduction=*/4},
                                    std::nullopt, OperandTile{/*reduction=*/2, /*nonReduction=*/2});
    // Tiles follow the same order as elements: the tile holding reduction 0-1, nonReduction 0-1,
    // then reduction 2-3, nonReduction 0-1, then reduction 0-1, nonReduction 2-3, and so on.
    EXPECT_EQ(tiled,
              (std::vector<int>{0, 1, 10, 11, 2, 3, 12, 13, 20, 21, 30, 31, 22, 23, 32, 33}));
}

// ============================================================================
// Swizzles and tiles
// ============================================================================

// The factors of a natural element index, fastest first, as the kernel's VGPRs,
// SIMDs and lanes index a swizzle tile: [VGPR index, VGPR block, SIMD block,
// reduction swizzle tile, (reduction tile), lane, SIMD index index, SIMD index block,
// non-reduction swizzle tile, (non-reduction tile)]. The tile factors are present only with a tile.
std::vector<size_t> swizzleFactorSizes(size_t reduction, size_t nonReduction,
                                       const OperandSwizzle& swizzle,
                                       const std::optional<OperandTile>& tile) {
    const size_t lanesPerSimd = 16, simdsPerWave = 4;
    const size_t simdIndex = swizzle.tileNonReduction / lanesPerSimd;
    const size_t simdBlock = simdsPerWave / simdIndex;
    const size_t vgprIndex = std::min(simdIndex, swizzle.subTileReduction);
    const size_t vgprBlock = swizzle.tileReduction / simdBlock / vgprIndex;
    const size_t simdIndexBlock = vgprIndex;
    const size_t simdIndexIndex = simdIndex / simdIndexBlock;
    if (!tile)
        return {
            vgprIndex,    vgprBlock,      simdBlock,      reduction / swizzle.tileReduction,
            lanesPerSimd, simdIndexIndex, simdIndexBlock, nonReduction / swizzle.tileNonReduction};
    return {vgprIndex,
            vgprBlock,
            simdBlock,
            tile->reduction / swizzle.tileReduction,
            reduction / tile->reduction,
            lanesPerSimd,
            simdIndexIndex,
            simdIndexBlock,
            tile->nonReduction / swizzle.tileNonReduction,
            nonReduction / tile->nonReduction};
}

// Stores each element at the offset whose index factors are ordered by
// `physicalOrder`, fastest first.
template <typename T>
std::vector<T> reorderFactors(const std::vector<T>& input, const std::vector<size_t>& sizes,
                              const std::vector<size_t>& physicalOrder) {
    std::vector<size_t> strides(sizes.size());
    size_t stride = 1;
    for (const size_t dimension : physicalOrder) {
        strides[dimension] = stride;
        stride *= sizes[dimension];
    }
    std::vector<T> output(input.size());
    std::vector<size_t> coordinate(sizes.size(), 0);
    for (size_t source = 0; source < input.size(); ++source) {
        size_t destination = 0;
        for (size_t dimension = 0; dimension < sizes.size(); ++dimension)
            destination += coordinate[dimension] * strides[dimension];
        output[destination] = input[source];
        for (size_t dimension = 0;
             dimension < sizes.size() && ++coordinate[dimension] == sizes[dimension]; ++dimension)
            coordinate[dimension] = 0;
    }
    return output;
}

struct SwizzleCase {
    size_t reduction;
    size_t nonReduction;
    OperandSwizzle swizzle;
    std::optional<OperandTile> tile;
    std::vector<size_t> physicalOrder;
};

template <typename T>
void expectKernelOrder(const SwizzleCase& test) {
    std::vector<T> input(test.reduction * test.nonReduction);
    std::iota(input.begin(), input.end(), T{0});
    const std::vector<T> expected = reorderFactors(
        input, swizzleFactorSizes(test.reduction, test.nonReduction, test.swizzle, test.tile),
        test.physicalOrder);
    EXPECT_EQ(copyOperandToPhysicalLayout(input, OperandExtents{test.reduction, test.nonReduction},
                                          test.swizzle, test.tile),
              expected)
        << "tileReduction " << test.swizzle.tileReduction << " tileNonReduction "
        << test.swizzle.tileNonReduction << " subTileReduction " << test.swizzle.subTileReduction
        << (test.tile ? " with tile" : "");
}

// The kernel's load order for each swizzle: tileNonReduction 64 and tileNonReduction 32 with
// subTileReduction 4 swap the SIMD index block with the VGPR index; tileNonReduction 32 with
// subTileReduction 4 also swaps the SIMD and VGPR blocks; tileNonReduction 32 with subTileReduction
// 2 rotates the VGPR index, VGPR block and SIMD block. A tile moves its own factors (reduction tile
// and non-reduction tile) to the slowest positions.
const std::vector<SwizzleCase>& swizzleCases() {
    static const std::vector<SwizzleCase> cases{
        {256, 128, OperandSwizzle{256, 64, 4}, std::nullopt, {6, 1, 2, 3, 4, 5, 0, 7}},
        {128, 64, OperandSwizzle{128, 32, 4}, std::nullopt, {6, 2, 1, 3, 4, 5, 0, 7}},
        {128, 64, OperandSwizzle{128, 32, 2}, std::nullopt, {1, 2, 0, 3, 4, 5, 6, 7}},
        {256,
         128,
         OperandSwizzle{256, 64, 4},
         OperandTile{256, 64},
         {7, 1, 2, 3, 5, 6, 0, 8, 4, 9}},
        {512,
         128,
         OperandSwizzle{128, 32, 4},
         OperandTile{128, 32},
         {7, 2, 1, 3, 5, 6, 0, 8, 4, 9}},
        {256, 64, OperandSwizzle{128, 32, 2}, OperandTile{128, 64}, {1, 2, 0, 3, 5, 6, 7, 8, 4, 9}},
    };
    return cases;
}

TEST(OperandLayoutTest, SwizzlesIntoKernelLoadOrder) {
    for (const SwizzleCase& test : swizzleCases()) expectKernelOrder<float>(test);
}

TEST(OperandLayoutTest, SwizzlesDoubleAndIntegerElements) {
    for (const SwizzleCase& test : swizzleCases()) {
        expectKernelOrder<double>(test);
        expectKernelOrder<int>(test);
    }
}

TEST(OperandLayoutTest, TileOnlyStoresEachTileContiguously) {
    constexpr size_t reduction = 128, nonReduction = 64, tileReduction = 16, tileNonReduction = 8;
    std::vector<float> input(reduction * nonReduction);
    std::iota(input.begin(), input.end(), 0.0f);

    const auto output =
        copyOperandToPhysicalLayout(input, OperandExtents{reduction, nonReduction}, std::nullopt,
                                    OperandTile{tileReduction, tileNonReduction});
    ASSERT_EQ(output.size(), input.size());
    for (size_t reductionIndex = 0; reductionIndex < reduction; ++reductionIndex)
        for (size_t nonReductionIndex = 0; nonReductionIndex < nonReduction; ++nonReductionIndex) {
            const size_t tileIndex =
                reductionIndex / tileReduction +
                (reduction / tileReduction) * (nonReductionIndex / tileNonReduction);
            const size_t destination = tileIndex * tileReduction * tileNonReduction +
                                       reductionIndex % tileReduction +
                                       tileReduction * (nonReductionIndex % tileNonReduction);
            EXPECT_EQ(output[destination], input[reductionIndex + reduction * nonReductionIndex])
                << "reduction " << reductionIndex << " nonReduction " << nonReductionIndex;
        }
}

TEST(OperandLayoutTest, NoSwizzleOrTileCopiesTheOperand) {
    const std::vector<int> input{1, 2, 3, 4, 5, 6};
    EXPECT_EQ(copyOperandToPhysicalLayout(input, OperandExtents{2, 3}, std::nullopt, std::nullopt),
              input);
    // A single tile covering the whole operand leaves its order unchanged.
    const std::vector<int> eight{1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(
        copyOperandToPhysicalLayout(eight, OperandExtents{2, 4}, std::nullopt, OperandTile{2, 4}),
        eight);
}

// ============================================================================
// Rejected inputs
// ============================================================================

TEST(OperandLayoutTest, RejectsElementCountMismatch) {
    const std::vector<float> input(100);
    EXPECT_THROW(
        copyOperandToPhysicalLayout(
            input, OperandExtents{128, 64},
            OperandSwizzle{/*tileReduction=*/128, /*tileNonReduction=*/32, /*subTileReduction=*/4},
            std::nullopt),
        std::runtime_error);
    EXPECT_THROW(
        copyOperandToPhysicalLayout(input, OperandExtents{10, 9}, std::nullopt, std::nullopt),
        std::runtime_error);
}

TEST(OperandLayoutTest, RejectsUnsupportedSwizzles) {
    const std::vector<float> input(size_t{128} * 64);
    const OperandExtents extents{128, 64};
    for (const OperandSwizzle& swizzle : {
             OperandSwizzle{/*tileReduction=*/128, /*tileNonReduction=*/48, /*subTileReduction=*/4},
             OperandSwizzle{/*tileReduction=*/0, /*tileNonReduction=*/32, /*subTileReduction=*/4},
             OperandSwizzle{/*tileReduction=*/126, /*tileNonReduction=*/32, /*subTileReduction=*/4},
             OperandSwizzle{/*tileReduction=*/128, /*tileNonReduction=*/32, /*subTileReduction=*/0},
             OperandSwizzle{/*tileReduction=*/128, /*tileNonReduction=*/32, /*subTileReduction=*/1},
             OperandSwizzle{/*tileReduction=*/128, /*tileNonReduction=*/32, /*subTileReduction=*/3},
         })
        EXPECT_THROW(copyOperandToPhysicalLayout(input, extents, swizzle, std::nullopt),
                     std::runtime_error)
            << swizzle.tileReduction << " " << swizzle.tileNonReduction << " "
            << swizzle.subTileReduction;
}

TEST(OperandLayoutTest, RejectsIndivisibleExtentsAndTiles) {
    const std::vector<float> input(size_t{128} * 64);
    const OperandExtents extents{128, 64};
    const OperandSwizzle swizzle{/*tileReduction=*/128, /*tileNonReduction=*/32,
                                 /*subTileReduction=*/4};
    for (const OperandTile& tile : {OperandTile{0, 16}, OperandTile{16, 0}, OperandTile{17, 16}})
        EXPECT_THROW(copyOperandToPhysicalLayout(input, extents, std::nullopt, tile),
                     std::runtime_error)
            << tile.reduction << " " << tile.nonReduction;
    // A tile must divide into swizzle tiles, and the extents into tiles.
    EXPECT_THROW(copyOperandToPhysicalLayout(input, extents, swizzle, OperandTile{96, 32}),
                 std::runtime_error);
    EXPECT_THROW(copyOperandToPhysicalLayout(input, extents, swizzle, OperandTile{128, 48}),
                 std::runtime_error);
    EXPECT_THROW(
        copyOperandToPhysicalLayout(
            input, extents,
            OperandSwizzle{/*tileReduction=*/256, /*tileNonReduction=*/32, /*subTileReduction=*/4},
            OperandTile{128, 64}),
        std::runtime_error);
    EXPECT_THROW(
        copyOperandToPhysicalLayout(
            input, extents,
            OperandSwizzle{/*tileReduction=*/128, /*tileNonReduction=*/64, /*subTileReduction=*/4},
            OperandTile{128, 32}),
        std::runtime_error);
}

TEST(OperandLayoutTest, EmptyOperandRequiresDivisibleExtents) {
    const std::vector<float> empty;
    const OperandSwizzle swizzle{/*tileReduction=*/128, /*tileNonReduction=*/32,
                                 /*subTileReduction=*/4};
    EXPECT_TRUE(
        copyOperandToPhysicalLayout(empty, OperandExtents{0, 64}, swizzle, std::nullopt).empty());
    EXPECT_TRUE(copyOperandToPhysicalLayout(empty, OperandExtents{128, 0}, std::nullopt,
                                            OperandTile{16, 16})
                    .empty());
    // Each extent must divide by its tile even when there are no elements.
    EXPECT_THROW(copyOperandToPhysicalLayout(empty, OperandExtents{4, 0}, swizzle, std::nullopt),
                 std::runtime_error);
    EXPECT_THROW(
        copyOperandToPhysicalLayout(empty, OperandExtents{0, 4}, std::nullopt, OperandTile{16, 16}),
        std::runtime_error);
}
}  // namespace
