// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_amd_gpu_layout_test_support.hpp"

namespace amd_gpu_layout_test {
// ============================================================================
// Tutorial: preparing MX scales and operands for an AMD GPU kernel
// ============================================================================

// A gfx950 MX kernel reads its scales in a tiled byte order. Describe the
// natural scale tensor as [slow, fast] = [MN rows, K blocks], plan it to learn
// the device buffer size, then copy the natural bytes into that layout.
TEST(AmdGpuLayoutTutorial, PlanAndCopyGfx950Scales) {
    // Two MN rows of three K blocks; scale (row, kBlock) holds 10 * row + kBlock + 1.
    const std::array<std::byte, 6> natural{std::byte{1},  std::byte{2},  std::byte{3},
                                           std::byte{11}, std::byte{12}, std::byte{13}};
    const MxScaleStoragePlan plan = planMxScaleStorage({2, 3}, 32, MxScaleStorageLayout::Gfx950);
    // gfx950 pads MN to a multiple of 32 and K blocks to a multiple of 8.
    EXPECT_EQ(plan.naturalByteCount, 6u);
    EXPECT_EQ(plan.physicalByteCount, 32u * 8u);

    const std::vector<std::byte> physical =
        copyMxScaleStorageToPhysicalLayout(natural.data(), natural.size(), plan);
    // Within the first 32 x 8 tile, consecutive MN rows are 4 bytes apart and
    // consecutive K blocks 64 bytes apart. Every other byte is zero padding.
    std::vector<std::byte> expected(plan.physicalByteCount, std::byte{0});
    for (size_t row = 0; row < 2; ++row)
        for (size_t kBlock = 0; kBlock < 3; ++kBlock)
            expected[4 * row + 64 * kBlock] = natural[3 * row + kBlock];
    EXPECT_EQ(physical, expected);
}

// preSwizzle with only a pre-tile configuration makes each tile contiguous.
// Dimension 0 of `sizes` is the contiguous one, in the input and within each
// output tile. Here a 4 x 4 operand with K fastest is split into 2 x 2 tiles.
TEST(AmdGpuLayoutTutorial, PreTileOperand) {
    // Element (k, mn) is stored at k + 4 * mn and holds 10 * mn + k.
    std::vector<int> operand(16);
    for (int mn = 0; mn < 4; ++mn)
        for (int k = 0; k < 4; ++k) operand[k + 4 * mn] = 10 * mn + k;

    const std::vector<int> tiled = preSwizzle(operand, {4, 4}, {}, {2, 2});
    // Tiles follow the same order as elements: the tile holding k 0-1, mn 0-1,
    // then k 2-3, mn 0-1, then k 0-1, mn 2-3, and so on.
    EXPECT_EQ(tiled,
              (std::vector<int>{0, 1, 10, 11, 2, 3, 12, 13, 20, 21, 30, 31, 22, 23, 32, 33}));
}

TEST(MxScaleStorageLayoutTest, CopiesNaturalStorage) {
    const MxScaleStoragePlan emptyPlan =
        planMxScaleStorage({0, 0}, 32, MxScaleStorageLayout::Natural);
    EXPECT_TRUE(copyMxScaleStorageToPhysicalLayout(nullptr, 0, emptyPlan).empty());

    const std::array<std::byte, 8> natural{
        std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0},
        std::byte{4}, std::byte{5}, std::byte{6}, std::byte{0},
    };
    const MxScaleStoragePlan plan = planMxScaleStorage({2, 4}, 32, MxScaleStorageLayout::Natural);
    const std::vector<std::byte> physical =
        copyMxScaleStorageToPhysicalLayout(natural.data(), natural.size(), plan);
    const std::array<uint8_t, 8> expected{1, 2, 3, 0, 4, 5, 6, 0};

    ASSERT_EQ(physical.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index)
        EXPECT_EQ(std::to_integer<uint8_t>(physical[index]), expected[index]);
}

TEST(MxScaleStorageLayoutTest, RejectsInvalidNaturalStorageSize) {
    const std::array<std::byte, 7> natural{};
    const MxScaleStoragePlan plan = planMxScaleStorage({2, 4}, 32, MxScaleStorageLayout::Natural);
    EXPECT_THROW(copyMxScaleStorageToPhysicalLayout(natural.data(), natural.size(), plan),
                 std::runtime_error);
}

TEST(MxScaleStorageLayoutTest, RejectsOverflowingNaturalStorageDimensions) {
    EXPECT_THROW(
        planMxScaleStorage({runtimeSize(std::numeric_limits<size_t>::max()), runtimeSize(2)}, 32,
                           MxScaleStorageLayout::Natural),
        std::runtime_error);
}

TEST(MxScaleStorageLayoutTest, PlansExactPhysicalByteCounts) {
    const MxScaleStoragePlan natural =
        planMxScaleStorage({8, 17}, 32, MxScaleStorageLayout::Natural);
    EXPECT_EQ(natural.naturalByteCount, 136);
    EXPECT_EQ(natural.physicalByteCount, 136);

    const MxScaleStoragePlan gfx950 = planMxScaleStorage({17, 8}, 32, MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(gfx950.naturalByteCount, 136);
    EXPECT_EQ(gfx950.physicalByteCount, 256);

    const MxScaleStoragePlan gfx1250 =
        planMxScaleStorage({17, 8}, 32, MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(gfx1250.naturalByteCount, 136);
    EXPECT_EQ(gfx1250.physicalByteCount, 136);

    // gfx1250 pads the fast dimension to a multiple of 128 / blockSize.
    struct Gfx1250Case {
        std::array<size_t, 2> shape;
        size_t blockSize;
        size_t physicalByteCount;
    };
    for (const Gfx1250Case& test : {Gfx1250Case{{8, 4}, 32, 32}, Gfx1250Case{{16, 8}, 16, 128},
                                    Gfx1250Case{{8, 5}, 32, 64}, Gfx1250Case{{8, 7}, 16, 64}})
        EXPECT_EQ(planMxScaleStorage(test.shape, test.blockSize, MxScaleStorageLayout::Gfx1250)
                      .physicalByteCount,
                  test.physicalByteCount)
            << test.shape[0] << "x" << test.shape[1] << " block " << test.blockSize;
}

TEST(MxScaleStorageLayoutTest, OnlyGfx1250ConstrainsBlockSize) {
    for (const size_t blockSize : {0, 1, 8, 64, 128})
        EXPECT_THROW(planMxScaleStorage({1, 4}, blockSize, MxScaleStorageLayout::Gfx1250),
                     std::runtime_error)
            << blockSize;
    EXPECT_EQ(planMxScaleStorage({17, 8}, 0, MxScaleStorageLayout::Natural).physicalByteCount, 136);
    EXPECT_EQ(planMxScaleStorage({17, 8}, 0, MxScaleStorageLayout::Gfx950).physicalByteCount, 256);
}

TEST(MxScaleStorageLayoutTest, MapsArchitectureNames) {
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
}

TEST(MxScaleStorageLayoutTest, RejectsMisleadingArchitectureSubstrings) {
    for (const std::string_view architectureName :
         {"notgfx950", "gfx9500", "gfx950-sramecc+", "notgfx1250", "gfx12500", "gfx1250suffix",
          "gfx1250-strictsuffix"})
        EXPECT_EQ(mxScaleStorageLayoutForArchitectureName(architectureName),
                  MxScaleStorageLayout::Natural)
            << architectureName;
}
}  // namespace amd_gpu_layout_test
