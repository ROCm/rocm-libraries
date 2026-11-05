// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <roc/hostnumerics/amd_gpu_layout/operand.hpp>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#error "AMD GPU layout consumers must not inherit OpenMP compile flags."
#endif

using namespace roc::hostnumerics::amd_gpu_layout;

namespace {
// ============================================================================
// Tutorial: preparing an operand for an AMD GPU kernel
// ============================================================================

// An operand tile stores each tile contiguously. K is the contiguous extent, in
// the input and within each output tile. Here a 4 x 4 operand is split into
// 2 x 2 tiles; no swizzle is requested.
TEST(AmdGpuOperandLayoutTutorial, TileOperand) {
    // Element (k, mn) is stored at k + 4 * mn and holds 10 * mn + k.
    std::vector<int> operand(16);
    for (int mn = 0; mn < 4; ++mn)
        for (int k = 0; k < 4; ++k) operand[k + 4 * mn] = 10 * mn + k;

    const std::vector<int> tiled = copyOperandToPhysicalLayout(
        operand, OperandExtents{/*k=*/4, /*mn=*/4}, std::nullopt, OperandTile{/*k=*/2, /*mn=*/2});
    // Tiles follow the same order as elements: the tile holding k 0-1, mn 0-1,
    // then k 2-3, mn 0-1, then k 0-1, mn 2-3, and so on.
    EXPECT_EQ(tiled,
              (std::vector<int>{0, 1, 10, 11, 2, 3, 12, 13, 20, 21, 30, 31, 22, 23, 32, 33}));
}

// ============================================================================
// Swizzles and tiles
// ============================================================================

// The factors of a natural element index, fastest first, as the kernel's VGPRs,
// SIMDs and lanes index a swizzle tile: [VGPR index, VGPR block, SIMD block,
// K swizzle tile, (K tile), lane, SIMD index index, SIMD index block,
// MN swizzle tile, (MN tile)]. The tile factors are present only with a tile.
std::vector<size_t> swizzleFactorSizes(size_t k, size_t mn, const OperandSwizzle& swizzle,
                                       const std::optional<OperandTile>& tile) {
    const size_t lanesPerSimd = 16, simdsPerWave = 4;
    const size_t simdIndex = swizzle.tileMN / lanesPerSimd;
    const size_t simdBlock = simdsPerWave / simdIndex;
    const size_t vgprIndex = std::min(simdIndex, swizzle.subTileK);
    const size_t vgprBlock = swizzle.tileK / simdBlock / vgprIndex;
    const size_t simdIndexBlock = vgprIndex;
    const size_t simdIndexIndex = simdIndex / simdIndexBlock;
    if (!tile)
        return {vgprIndex,    vgprBlock,      simdBlock,      k / swizzle.tileK,
                lanesPerSimd, simdIndexIndex, simdIndexBlock, mn / swizzle.tileMN};
    return {vgprIndex,    vgprBlock,      simdBlock,      tile->k / swizzle.tileK,   k / tile->k,
            lanesPerSimd, simdIndexIndex, simdIndexBlock, tile->mn / swizzle.tileMN, mn / tile->mn};
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
    size_t k;
    size_t mn;
    OperandSwizzle swizzle;
    std::optional<OperandTile> tile;
    std::vector<size_t> physicalOrder;
};

template <typename T>
void expectKernelOrder(const SwizzleCase& test) {
    std::vector<T> input(test.k * test.mn);
    std::iota(input.begin(), input.end(), T{0});
    const std::vector<T> expected = reorderFactors(
        input, swizzleFactorSizes(test.k, test.mn, test.swizzle, test.tile), test.physicalOrder);
    EXPECT_EQ(copyOperandToPhysicalLayout(input, OperandExtents{test.k, test.mn}, test.swizzle,
                                          test.tile),
              expected)
        << "tileK " << test.swizzle.tileK << " tileMN " << test.swizzle.tileMN << " subTileK "
        << test.swizzle.subTileK << (test.tile ? " with tile" : "");
}

// The kernel's load order for each swizzle: tileMN 64 and tileMN 32 with
// subTileK 4 swap the SIMD index block with the VGPR index; tileMN 32 with
// subTileK 4 also swaps the SIMD and VGPR blocks; tileMN 32 with subTileK 2
// rotates the VGPR index, VGPR block and SIMD block. A tile moves its own
// factors (K tile and MN tile) to the slowest positions.
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
    constexpr size_t k = 128, mn = 64, tileK = 16, tileMN = 8;
    std::vector<float> input(k * mn);
    std::iota(input.begin(), input.end(), 0.0f);

    const auto output = copyOperandToPhysicalLayout(input, OperandExtents{k, mn}, std::nullopt,
                                                    OperandTile{tileK, tileMN});
    ASSERT_EQ(output.size(), input.size());
    for (size_t kIndex = 0; kIndex < k; ++kIndex)
        for (size_t mnIndex = 0; mnIndex < mn; ++mnIndex) {
            const size_t tileIndex = kIndex / tileK + (k / tileK) * (mnIndex / tileMN);
            const size_t destination =
                tileIndex * tileK * tileMN + kIndex % tileK + tileK * (mnIndex % tileMN);
            EXPECT_EQ(output[destination], input[kIndex + k * mnIndex])
                << "k " << kIndex << " mn " << mnIndex;
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
    EXPECT_THROW(copyOperandToPhysicalLayout(
                     input, OperandExtents{128, 64},
                     OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/4}, std::nullopt),
                 std::runtime_error);
    EXPECT_THROW(
        copyOperandToPhysicalLayout(input, OperandExtents{10, 9}, std::nullopt, std::nullopt),
        std::runtime_error);
}

TEST(OperandLayoutTest, RejectsUnsupportedSwizzles) {
    const std::vector<float> input(size_t{128} * 64);
    const OperandExtents extents{128, 64};
    for (const OperandSwizzle& swizzle : {
             OperandSwizzle{/*tileK=*/128, /*tileMN=*/48, /*subTileK=*/4},
             OperandSwizzle{/*tileK=*/0, /*tileMN=*/32, /*subTileK=*/4},
             OperandSwizzle{/*tileK=*/126, /*tileMN=*/32, /*subTileK=*/4},
             OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/0},
             OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/1},
             OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/3},
         })
        EXPECT_THROW(copyOperandToPhysicalLayout(input, extents, swizzle, std::nullopt),
                     std::runtime_error)
            << swizzle.tileK << " " << swizzle.tileMN << " " << swizzle.subTileK;
}

TEST(OperandLayoutTest, RejectsIndivisibleExtentsAndTiles) {
    const std::vector<float> input(size_t{128} * 64);
    const OperandExtents extents{128, 64};
    const OperandSwizzle swizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/4};
    for (const OperandTile& tile : {OperandTile{0, 16}, OperandTile{16, 0}, OperandTile{17, 16}})
        EXPECT_THROW(copyOperandToPhysicalLayout(input, extents, std::nullopt, tile),
                     std::runtime_error)
            << tile.k << " " << tile.mn;
    // A tile must divide into swizzle tiles, and the extents into tiles.
    EXPECT_THROW(copyOperandToPhysicalLayout(input, extents, swizzle, OperandTile{96, 32}),
                 std::runtime_error);
    EXPECT_THROW(copyOperandToPhysicalLayout(input, extents, swizzle, OperandTile{128, 48}),
                 std::runtime_error);
    EXPECT_THROW(copyOperandToPhysicalLayout(
                     input, extents, OperandSwizzle{/*tileK=*/256, /*tileMN=*/32, /*subTileK=*/4},
                     OperandTile{128, 64}),
                 std::runtime_error);
    EXPECT_THROW(copyOperandToPhysicalLayout(
                     input, extents, OperandSwizzle{/*tileK=*/128, /*tileMN=*/64, /*subTileK=*/4},
                     OperandTile{128, 32}),
                 std::runtime_error);
}

TEST(OperandLayoutTest, EmptyOperandRequiresDivisibleExtents) {
    const std::vector<float> empty;
    const OperandSwizzle swizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/4};
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
