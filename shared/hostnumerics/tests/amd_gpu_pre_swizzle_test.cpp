// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_amd_gpu_layout_test_support.hpp"

namespace amd_gpu_layout_test {
TEST(PreSwizzleTest, PreSwizzleOnlySwizzle64) {
    size_t k = 256;
    size_t mn = 128;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {64, 256, 4};  // tileMN=64, tileK=256, subTileK=4
    std::vector<size_t> preTileSize;                    // empty
    FillSwizzle(input, k, mn, preSwizzleSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzle(expected, k, mn, preSwizzleSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleOnlySwizzle32SubTile4) {
    size_t k = 128;
    size_t mn = 64;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};  // tileMN=32, tileK=128, subTileK=4
    std::vector<size_t> preTileSize;                    // empty
    FillSwizzle(input, k, mn, preSwizzleSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    std::vector<float> expected(k * mn);
    FillPreSwizzle(expected, k, mn, preSwizzleSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleOnlySwizzle32SubTile2) {
    size_t k = 128;
    size_t mn = 64;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 2};  // tileMN=32, tileK=128, subTileK=2
    std::vector<size_t> preTileSize;                    // empty
    FillSwizzle(input, k, mn, preSwizzleSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzle(expected, k, mn, preSwizzleSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleOnlyPreTile) {
    size_t k = 128;
    size_t mn = 64;
    std::vector<float> input(k * mn);
    std::iota(input.begin(), input.end(), 0.0f);

    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize;  // empty
    std::vector<size_t> preTileSize = {16, 16};

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleBothSwizzleAndTile64) {
    size_t k = 256;
    size_t mn = 128;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {64, 256, 4};
    std::vector<size_t> preTileSize = {256, 64};
    FillSwizzleAndTile(input, k, mn, preSwizzleSize, preTileSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzleAndTile(expected, k, mn, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleBothSwizzleAndTile32SubTile4) {
    size_t k = 512;
    size_t mn = 128;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize = {128, 32};
    FillSwizzleAndTile(input, k, mn, preSwizzleSize, preTileSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzleAndTile(expected, k, mn, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleBothSwizzleAndTile32SubTile2) {
    size_t k = 256;
    size_t mn = 64;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 2};
    std::vector<size_t> preTileSize = {128, 64};
    FillSwizzleAndTile(input, k, mn, preSwizzleSize, preTileSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzleAndTile(expected, k, mn, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleInvalidTileMN) {
    std::vector<float> input(128 * 64);
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {48, 128, 4};  // tileMN not 32 or 64
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleTileKNotMultipleOf4) {
    std::vector<float> input(127 * 64);
    std::vector<size_t> sizes = {127, 64};
    std::vector<size_t> preSwizzleSize = {32, 127, 4};  // tileK not multiple of 4
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleSizeMismatch) {
    std::vector<float> input(100);  // Wrong size
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleBatchDimensionNotSupported) {
    std::vector<float> input(2 * 128 * 64);
    std::vector<size_t> sizes = {2, 128, 64};  // 3D tensor (batch dimension)
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleInvalidPreSwizzleSize) {
    std::vector<float> input(128 * 64);
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128};  // Only 2 elements, need 3
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleEmptyConfigurationIsIdentity) {
    const std::vector<int> input{1, 2, 3, 4, 5, 6};
    EXPECT_EQ(preSwizzle(input, {2, 3}, {}, {}), input);
}

TEST(PreSwizzleTest, PreSwizzleRejectsInvalidPreTileConfiguration) {
    const std::vector<float> input(128 * 64);
    EXPECT_THROW(preSwizzle(input, {128, 64}, {}, {16}), std::runtime_error);
    EXPECT_THROW(preSwizzle(input, {128, 64}, {}, {0, 16}), std::runtime_error);
    EXPECT_THROW(preSwizzle(input, {128, 64}, {}, {16, 0}), std::runtime_error);
    EXPECT_THROW(preSwizzle(input, {128, 64}, {}, {17, 16}), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleRejectsInvalidSwizzleDivisors) {
    const std::vector<float> input(128 * 64);
    EXPECT_THROW(preSwizzle(input, {128, 64}, {32, 0, 4}, {}), std::runtime_error);
    EXPECT_THROW(preSwizzle(input, {128, 64}, {32, 128, 0}, {}), std::runtime_error);
    EXPECT_THROW(preSwizzle(input, {128, 64}, {32, 128, 1}, {}), std::runtime_error);
    EXPECT_THROW(preSwizzle(input, {128, 64}, {32, 64, 4}, {96, 32}), std::runtime_error);
    EXPECT_THROW(preSwizzle(input, {128, 64}, {32, 128, 4}, {128, 48}), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzlePtTileSizeKZero) {
    std::vector<float> input(128 * 64);
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 256, 4};  // tileK > sizes[0]
    std::vector<size_t> preTileSize = {128, 64};

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzlePtTileSizeMNZero) {
    std::vector<float> input(128 * 64);
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {128, 128, 4};  // tileMN > sizes[1]
    std::vector<size_t> preTileSize = {128, 64};

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

// ============================================================================
// Edge Cases and Integration Tests
// ============================================================================

TEST(PreSwizzleTest, PreSwizzleIdentityOnSmallData) {
    // Test that swizzling preserves all data
    std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<size_t> sizes = {2, 4};
    std::vector<size_t> preSwizzleSize;
    std::vector<size_t> preTileSize = {2, 4};  // Same as sizes, should be identity

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());

    // Verify all elements are present
    std::vector<int> sortedOutput = output;
    std::vector<int> sortedInput = input;
    std::sort(sortedOutput.begin(), sortedOutput.end());
    std::sort(sortedInput.begin(), sortedInput.end());
    EXPECT_EQ(sortedOutput, sortedInput);
}

TEST(PreSwizzleTest, PreSwizzleDoubleType) {
    // Test that templates work with different types
    std::vector<double> input(128 * 64);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<double>(i) * 0.1;

    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleIntegerType) {
    // Test that templates work with integer types
    size_t k = 128;
    size_t mn = 64;
    std::vector<int> input(k * mn);

    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;

    // Create float versions for helper functions, then convert
    std::vector<float> inputFloat(k * mn);
    FillSwizzle(inputFloat, k, mn, preSwizzleSize);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<int>(inputFloat[i]);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expectedFloat(k * mn);
    FillPreSwizzle(expectedFloat, k, mn, preSwizzleSize);
    std::vector<int> expected(k * mn);
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<int>(expectedFloat[i]);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

// ============================================================================
// Tests for the gfx950 MX physical scale layout.
// ============================================================================

}  // namespace amd_gpu_layout_test
