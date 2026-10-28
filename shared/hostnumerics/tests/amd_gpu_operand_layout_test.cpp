// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_amd_gpu_layout_test_support.hpp"

namespace amd_gpu_layout_test {
namespace {
// The expected output of a swizzle-only layout, built by an independent index
// walk over the kernel's lane and VGPR factors.
template <typename T>
void expectSwizzled(size_t k, size_t mn, const OperandSwizzle& swizzle) {
    std::vector<float> input(k * mn);
    FillSwizzle(input, k, mn, swizzle);
    std::vector<float> expected(k * mn);
    FillPreSwizzle(expected, k, mn, swizzle);

    const std::vector<T> typedInput(input.begin(), input.end());
    const std::vector<T> output =
        copyOperandToPhysicalLayout(typedInput, OperandExtents{k, mn}, swizzle, std::nullopt);
    EXPECT_EQ(output, std::vector<T>(expected.begin(), expected.end()));
}

void expectSwizzledAndTiled(size_t k, size_t mn, const OperandSwizzle& swizzle,
                            const OperandTile& tile) {
    std::vector<float> input(k * mn);
    FillSwizzleAndTile(input, k, mn, swizzle, tile);
    std::vector<float> expected(k * mn);
    FillPreSwizzleAndTile(expected, k, mn, swizzle, tile);
    EXPECT_EQ(copyOperandToPhysicalLayout(input, OperandExtents{k, mn}, swizzle, tile), expected);
}
}  // namespace

TEST(OperandLayoutTest, SwizzleTileMN64) {
    expectSwizzled<float>(256, 128, OperandSwizzle{/*tileK=*/256, /*tileMN=*/64, /*subTileK=*/4});
}

TEST(OperandLayoutTest, SwizzleTileMN32SubTileK4) {
    expectSwizzled<float>(128, 64, OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/4});
}

TEST(OperandLayoutTest, SwizzleTileMN32SubTileK2) {
    expectSwizzled<float>(128, 64, OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/2});
}

TEST(OperandLayoutTest, SwizzlesDoubleAndIntegerElements) {
    expectSwizzled<double>(128, 64, OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/4});
    expectSwizzled<int>(128, 64, OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/4});
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

TEST(OperandLayoutTest, SwizzleAndTileTileMN64) {
    expectSwizzledAndTiled(256, 128, OperandSwizzle{/*tileK=*/256, /*tileMN=*/64, /*subTileK=*/4},
                           OperandTile{/*k=*/256, /*mn=*/64});
}

TEST(OperandLayoutTest, SwizzleAndTileTileMN32SubTileK4) {
    expectSwizzledAndTiled(512, 128, OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/4},
                           OperandTile{/*k=*/128, /*mn=*/32});
}

TEST(OperandLayoutTest, SwizzleAndTileTileMN32SubTileK2) {
    expectSwizzledAndTiled(256, 64, OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/2},
                           OperandTile{/*k=*/128, /*mn=*/64});
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

TEST(OperandLayoutTest, RejectsElementCountMismatch) {
    const std::vector<float> input(100);
    EXPECT_THROW(copyOperandToPhysicalLayout(
                     input, OperandExtents{128, 64},
                     OperandSwizzle{/*tileK=*/128, /*tileMN=*/32, /*subTileK=*/4}, std::nullopt),
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

}  // namespace amd_gpu_layout_test
