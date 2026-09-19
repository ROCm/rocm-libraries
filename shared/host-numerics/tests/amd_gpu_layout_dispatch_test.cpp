// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_amd_gpu_layout_test_support.hpp"

namespace amd_gpu_layout_test {
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
                 std::invalid_argument);
}

TEST(MxScaleStorageLayoutTest, RejectsOverflowingNaturalStorageDimensions) {
    EXPECT_THROW(
        planMxScaleStorage({runtimeSize(std::numeric_limits<size_t>::max()), runtimeSize(2)}, 32,
                           MxScaleStorageLayout::Natural),
        std::overflow_error);
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
}

TEST(MxScaleStorageLayoutTest, MapsArchitectureNames) {
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx950"), MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx950:sramecc+:xnack-"),
              MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx1250"), MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx1250:xnack+"),
              MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(mxScaleStorageLayoutForArchitectureName("gfx942"), MxScaleStorageLayout::Natural);
}

TEST(MxScaleStorageLayoutTest, RejectsMisleadingArchitectureSubstrings) {
    for (const std::string_view architectureName :
         {"notgfx950", "gfx9500", "gfx950-sramecc+", "notgfx1250", "gfx12500", "gfx1250suffix"})
        EXPECT_EQ(mxScaleStorageLayoutForArchitectureName(architectureName),
                  MxScaleStorageLayout::Natural)
            << architectureName;
}
}  // namespace amd_gpu_layout_test
