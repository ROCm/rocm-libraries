// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>
#include <span>
#include <stdexcept>
#include <vector>

using namespace roc::hostnumerics::amd_gpu_layout;

namespace {
std::vector<std::byte> copyScales(const std::vector<uint8_t>& natural,
                                  const MxScaleStoragePlan& plan) {
    const auto bytes = std::as_bytes(std::span(natural));
    return copyMxScaleStorageToPhysicalLayout(bytes.data(), bytes.size(), plan);
}

// Nonzero scales, so that zero padding is distinguishable.
std::vector<uint8_t> numberedScales(size_t count) {
    std::vector<uint8_t> scales(count);
    for (size_t index = 0; index < count; ++index)
        scales[index] = static_cast<uint8_t>(index % 251 + 1);
    return scales;
}

std::vector<uint8_t> integers(const std::vector<std::byte>& bytes) {
    std::vector<uint8_t> result;
    for (const std::byte byte : bytes) result.push_back(std::to_integer<uint8_t>(byte));
    return result;
}

// ============================================================================
// Tutorial: preparing MX scales for an AMD GPU kernel
// ============================================================================

// A gfx950 MX kernel reads its scales in a tiled byte order. Describe the
// natural scale tensor as [slow, fast] = [MN rows, K blocks], plan it to learn
// the device buffer size, then copy the natural bytes into that layout.
TEST(AmdGpuMxLayoutTutorial, PlanAndCopyGfx950Scales) {
    // Two MN rows of three K blocks; scale (row, kBlock) holds 10 * row + kBlock + 1.
    const std::array<std::byte, 6> natural{std::byte{1},  std::byte{2},  std::byte{3},
                                           std::byte{11}, std::byte{12}, std::byte{13}};
    const auto plan = MxScaleStoragePlan::tile32x8({2, 3});
    // gfx950 pads MN to a multiple of 32 and K blocks to a multiple of 8.
    EXPECT_EQ(plan.naturalByteCount(), 6u);
    EXPECT_EQ(plan.physicalByteCount(), 32u * 8u);

    const std::vector<std::byte> physical =
        copyMxScaleStorageToPhysicalLayout(natural.data(), natural.size(), plan);
    // Within the first 32 x 8 tile, consecutive MN rows are 4 bytes apart and
    // consecutive K blocks 64 bytes apart. Every other byte is zero padding.
    std::vector<std::byte> expected(plan.physicalByteCount(), std::byte{0});
    for (size_t row = 0; row < 2; ++row)
        for (size_t kBlock = 0; kBlock < 3; ++kBlock)
            expected[4 * row + 64 * kBlock] = natural[3 * row + kBlock];
    EXPECT_EQ(physical, expected);
}

// ============================================================================
// Plans
// ============================================================================

TEST(MxScaleStoragePlanTest, PlansExactPhysicalByteCounts) {
    const auto natural = MxScaleStoragePlan::natural({8, 17});
    EXPECT_EQ(natural.naturalShape(), (std::array<size_t, 2>{8, 17}));
    EXPECT_EQ(natural.naturalByteCount(), 136u);
    EXPECT_EQ(natural.physicalByteCount(), 136u);

    const auto gfx950 = MxScaleStoragePlan::tile32x8({17, 8});
    EXPECT_EQ(gfx950.naturalByteCount(), 136u);
    EXPECT_EQ(gfx950.physicalByteCount(), 256u);
    EXPECT_EQ(MxScaleStoragePlan::tile32x8({50, 13}).physicalByteCount(), 64u * 16u);

    // Current gfx1250 kernels use tiles of four or eight scales.
    struct Gfx1250Case {
        std::array<size_t, 2> shape;
        size_t scalesPerTile;
        size_t physicalByteCount;
    };
    for (const Gfx1250Case& test :
         {Gfx1250Case{{17, 8}, 4, 136}, Gfx1250Case{{8, 4}, 4, 32}, Gfx1250Case{{16, 8}, 8, 128},
          Gfx1250Case{{8, 5}, 4, 64}, Gfx1250Case{{8, 7}, 8, 64}})
        EXPECT_EQ(MxScaleStoragePlan::fastDimensionTiles(test.shape, test.scalesPerTile)
                      .physicalByteCount(),
                  test.physicalByteCount)
            << test.shape[0] << "x" << test.shape[1] << " scales per tile " << test.scalesPerTile;
}

TEST(MxScaleStoragePlanTest, RejectsZeroTileWidthAndOverflow) {
    for (const auto shape : {std::array<size_t, 2>{1, 4}, {0, 0}})
        EXPECT_THROW(MxScaleStoragePlan::fastDimensionTiles(shape, 0), std::runtime_error);

    constexpr size_t maximum = std::numeric_limits<size_t>::max();
    EXPECT_THROW(MxScaleStoragePlan::natural({maximum, 2}), std::runtime_error);
    // The natural byte counts fit, but padding overflows.
    EXPECT_THROW(MxScaleStoragePlan::tile32x8({maximum, 1}), std::runtime_error);
    EXPECT_THROW(MxScaleStoragePlan::fastDimensionTiles({maximum / 2, 1}, 4), std::runtime_error);
}

// ============================================================================
// Copies
// ============================================================================

TEST(MxScaleStorageCopyTest, CopiesNaturalStorage) {
    const auto emptyPlan = MxScaleStoragePlan::natural({0, 0});
    EXPECT_TRUE(copyMxScaleStorageToPhysicalLayout(nullptr, 0, emptyPlan).empty());

    const std::vector<uint8_t> natural{1, 2, 3, 0, 4, 5, 6, 0};
    EXPECT_EQ(integers(copyScales(natural, MxScaleStoragePlan::natural({2, 4}))), natural);
}

TEST(MxScaleStorageCopyTest, RejectsStorageThatDoesNotMatchThePlan) {
    EXPECT_THROW(copyScales(std::vector<uint8_t>(7), MxScaleStoragePlan::natural({2, 4})),
                 std::runtime_error);
    EXPECT_THROW(copyScales(std::vector<uint8_t>(100), MxScaleStoragePlan::tile32x8({64, 16})),
                 std::runtime_error);
    EXPECT_THROW(
        copyScales(std::vector<uint8_t>(7), MxScaleStoragePlan::fastDimensionTiles({2, 4}, 4)),
        std::runtime_error);
    const auto plan = MxScaleStoragePlan::natural({2, 4});
    EXPECT_THROW(copyMxScaleStorageToPhysicalLayout(nullptr, 8, plan), std::runtime_error);
}

TEST(MxScaleStorageCopyTest, Gfx950MatchesAiterShuffleWithZeroPadding) {
    // AITER's e8m0_shuffle views the zero-padded [rows, cols] scales as
    // [rows / 32, 2, 16, cols / 8, 2, 4], permutes them to (0, 3, 5, 2, 4, 1)
    // and stores the result contiguously.
    for (const std::array<size_t, 2> shape : {std::array<size_t, 2>{50, 13}, {64, 16}, {32, 8}}) {
        const size_t rows = shape[0], cols = shape[1];
        const size_t paddedRows = (rows + 31) / 32 * 32, paddedCols = (cols + 7) / 8 * 8;
        const std::vector<uint8_t> input = numberedScales(rows * cols);
        const auto output = copyScales(input, MxScaleStoragePlan::tile32x8(shape));
        ASSERT_EQ(output.size(), paddedRows * paddedCols);
        for (size_t row = 0; row < paddedRows; ++row)
            for (size_t col = 0; col < paddedCols; ++col) {
                const size_t rowTile = row / 32, rowHalf = row / 16 % 2, rowLane = row % 16;
                const size_t colTile = col / 8, colHalf = col / 4 % 2, colLane = col % 4;
                const size_t destination =
                    ((((rowTile * (paddedCols / 8) + colTile) * 4 + colLane) * 16 + rowLane) * 2 +
                     colHalf) *
                        2 +
                    rowHalf;
                const uint8_t expected = row < rows && col < cols ? input[row * cols + col] : 0;
                EXPECT_EQ(std::to_integer<uint8_t>(output[destination]), expected)
                    << rows << "x" << cols << " row " << row << " col " << col;
            }
    }
}

TEST(MxScaleStorageCopyTest, FastDimensionTilesKeepScaleGroupsContiguous) {
    // With 32-element blocks, tiles hold 128 / 32 = 4 consecutive fast scales.
    // The physical order is [fast tile, slow, within tile], with zero padding.
    const std::vector<uint8_t> natural{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ(integers(copyScales(natural, MxScaleStoragePlan::fastDimensionTiles({2, 5}, 4))),
              (std::vector<uint8_t>{1, 2, 3, 4, 6, 7, 8, 9, 5, 0, 0, 0, 10, 0, 0, 0}));

    struct Case {
        size_t slow, fast, scalesPerTile;
    };
    for (const Case& test : {Case{2, 8, 4}, Case{3, 10, 8}, Case{4, 16, 4}, Case{2, 5, 3}}) {
        const size_t tileSize = test.scalesPerTile;
        const size_t paddedFast = (test.fast + tileSize - 1) / tileSize * tileSize;
        const std::vector<uint8_t> input = numberedScales(test.slow * test.fast);
        const std::vector<uint8_t> output =
            integers(copyScales(input, MxScaleStoragePlan::fastDimensionTiles(
                                           {test.slow, test.fast}, test.scalesPerTile)));
        ASSERT_EQ(output.size(), test.slow * paddedFast);
        for (size_t tile = 0; tile < paddedFast / tileSize; ++tile)
            for (size_t slow = 0; slow < test.slow; ++slow)
                for (size_t inTile = 0; inTile < tileSize; ++inTile) {
                    const size_t fast = tile * tileSize + inTile;
                    const uint8_t expected = fast < test.fast ? input[slow * test.fast + fast] : 0;
                    EXPECT_EQ(output[(tile * test.slow + slow) * tileSize + inTile], expected)
                        << test.slow << "x" << test.fast << " slow " << slow << " fast " << fast;
                }
    }
}
}  // namespace
