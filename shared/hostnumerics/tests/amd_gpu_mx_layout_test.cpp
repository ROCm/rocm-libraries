// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace roc::hostnumerics::amd_gpu_layout;

namespace {
std::vector<std::byte> copyScales(const std::vector<uint8_t>& natural,
                                  std::array<size_t, 2> naturalShape, size_t blockSize,
                                  MxScaleStorageLayout layout) {
    return copyMxScaleStorageToPhysicalLayout(reinterpret_cast<const std::byte*>(natural.data()),
                                              natural.size(),
                                              MxScaleStoragePlan(naturalShape, blockSize, layout));
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
    // Each scale applies to a block of 32 data elements. Only gfx1250's layout
    // depends on the block size, but every plan records it.
    const MxScaleStoragePlan plan({2, 3}, 32, MxScaleStorageLayout::Gfx950);
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
    const MxScaleStoragePlan natural({8, 17}, 32, MxScaleStorageLayout::Natural);
    EXPECT_EQ(natural.layout(), MxScaleStorageLayout::Natural);
    EXPECT_EQ(natural.naturalShape(), (std::array<size_t, 2>{8, 17}));
    EXPECT_EQ(natural.blockSize(), 32u);
    EXPECT_EQ(natural.naturalByteCount(), 136u);
    EXPECT_EQ(natural.physicalByteCount(), 136u);

    const MxScaleStoragePlan gfx950({17, 8}, 32, MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(gfx950.naturalByteCount(), 136u);
    EXPECT_EQ(gfx950.physicalByteCount(), 256u);
    EXPECT_EQ(MxScaleStoragePlan({50, 13}, 32, MxScaleStorageLayout::Gfx950).physicalByteCount(),
              64u * 16u);

    // gfx1250 pads the fast dimension to a multiple of 128 / blockSize.
    struct Gfx1250Case {
        std::array<size_t, 2> shape;
        size_t blockSize;
        size_t physicalByteCount;
    };
    for (const Gfx1250Case& test :
         {Gfx1250Case{{17, 8}, 32, 136}, Gfx1250Case{{8, 4}, 32, 32}, Gfx1250Case{{16, 8}, 16, 128},
          Gfx1250Case{{8, 5}, 32, 64}, Gfx1250Case{{8, 7}, 16, 64}})
        EXPECT_EQ(MxScaleStoragePlan(test.shape, test.blockSize, MxScaleStorageLayout::Gfx1250)
                      .physicalByteCount(),
                  test.physicalByteCount)
            << test.shape[0] << "x" << test.shape[1] << " block " << test.blockSize;
}

TEST(MxScaleStoragePlanTest, BlockSizeIsRequired) {
    for (const MxScaleStorageLayout layout :
         {MxScaleStorageLayout::Natural, MxScaleStorageLayout::Gfx950,
          MxScaleStorageLayout::Gfx1250})
        EXPECT_THROW(MxScaleStoragePlan({1, 4}, 0, layout), std::runtime_error);
    for (const size_t blockSize : {1, 8, 64, 128})
        EXPECT_THROW(MxScaleStoragePlan({1, 4}, blockSize, MxScaleStorageLayout::Gfx1250),
                     std::runtime_error)
            << blockSize;
    // The natural and gfx950 byte layouts do not depend on the block size.
    for (const size_t blockSize : {8, 16, 32, 64}) {
        EXPECT_EQ(MxScaleStoragePlan({17, 8}, blockSize, MxScaleStorageLayout::Natural)
                      .physicalByteCount(),
                  136u);
        EXPECT_EQ(MxScaleStoragePlan({17, 8}, blockSize, MxScaleStorageLayout::Gfx950)
                      .physicalByteCount(),
                  256u);
    }
}

TEST(MxScaleStoragePlanTest, RejectsInvalidLayoutsAndOverflow) {
    // Deliberately construct an unnamed enum value to test its rejection.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    const auto invalidLayout = static_cast<MxScaleStorageLayout>(3);
    EXPECT_THROW(MxScaleStoragePlan({1, 4}, 32, invalidLayout), std::runtime_error);

    constexpr size_t maximum = std::numeric_limits<size_t>::max();
    EXPECT_THROW(MxScaleStoragePlan({maximum, 2}, 32, MxScaleStorageLayout::Natural),
                 std::runtime_error);
    // The natural byte counts fit, but padding overflows.
    EXPECT_THROW(MxScaleStoragePlan({maximum, 1}, 32, MxScaleStorageLayout::Gfx950),
                 std::runtime_error);
    EXPECT_THROW(MxScaleStoragePlan({maximum / 2, 1}, 32, MxScaleStorageLayout::Gfx1250),
                 std::runtime_error);
}

TEST(MxScaleStoragePlanTest, MapsArchitectureNames) {
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx950"), MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx950:sramecc+:xnack-"),
              MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx1250"), MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx1250:xnack+"),
              MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx1250-strict"),
              MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx1250-strict:xnack-"),
              MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx942"), MxScaleStorageLayout::Natural);

    for (const std::string_view architectureName :
         {"notgfx950", "gfx9500", "gfx950-sramecc+", "notgfx1250", "gfx12500", "gfx1250suffix",
          "gfx1250-strictsuffix"})
        EXPECT_EQ(mxScaleStorageLayoutForArchitectureName(architectureName),
                  MxScaleStorageLayout::Natural)
            << architectureName;
}

// ============================================================================
// Copies
// ============================================================================

TEST(MxScaleStorageCopyTest, CopiesNaturalStorage) {
    const MxScaleStoragePlan emptyPlan({0, 0}, 32, MxScaleStorageLayout::Natural);
    EXPECT_TRUE(copyMxScaleStorageToPhysicalLayout(nullptr, 0, emptyPlan).empty());

    const std::vector<uint8_t> natural{1, 2, 3, 0, 4, 5, 6, 0};
    EXPECT_EQ(integers(copyScales(natural, {2, 4}, 32, MxScaleStorageLayout::Natural)), natural);
}

TEST(MxScaleStorageCopyTest, RejectsStorageThatDoesNotMatchThePlan) {
    EXPECT_THROW(copyScales(std::vector<uint8_t>(7), {2, 4}, 32, MxScaleStorageLayout::Natural),
                 std::runtime_error);
    EXPECT_THROW(copyScales(std::vector<uint8_t>(100), {64, 16}, 32, MxScaleStorageLayout::Gfx950),
                 std::runtime_error);
    EXPECT_THROW(copyScales(std::vector<uint8_t>(7), {2, 4}, 32, MxScaleStorageLayout::Gfx1250),
                 std::runtime_error);
    const MxScaleStoragePlan plan({2, 4}, 32, MxScaleStorageLayout::Natural);
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
        const auto output = copyScales(input, shape, 32, MxScaleStorageLayout::Gfx950);
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

TEST(MxScaleStorageCopyTest, Gfx1250TilesTheFastDimension) {
    // With 32-element blocks, tiles hold 128 / 32 = 4 consecutive fast scales.
    // The physical order is [fast tile, slow, within tile], with zero padding.
    const std::vector<uint8_t> natural{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ(integers(copyScales(natural, {2, 5}, 32, MxScaleStorageLayout::Gfx1250)),
              (std::vector<uint8_t>{1, 2, 3, 4, 6, 7, 8, 9, 5, 0, 0, 0, 10, 0, 0, 0}));

    struct Case {
        size_t slow, fast, blockSize;
    };
    for (const Case& test : {Case{2, 8, 32}, Case{3, 10, 16}, Case{4, 16, 32}}) {
        const size_t tileSize = 128 / test.blockSize;
        const size_t paddedFast = (test.fast + tileSize - 1) / tileSize * tileSize;
        const std::vector<uint8_t> input = numberedScales(test.slow * test.fast);
        const std::vector<uint8_t> output = integers(copyScales(
            input, {test.slow, test.fast}, test.blockSize, MxScaleStorageLayout::Gfx1250));
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
