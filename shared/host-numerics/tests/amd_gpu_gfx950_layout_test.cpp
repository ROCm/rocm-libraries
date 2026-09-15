// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_amd_gpu_layout_test_support.hpp"

namespace amd_gpu_layout_test {
TEST(MxScaleStorageGFX950Test, AlignedSizes) {
    // Basic test with aligned sizes: 64 rows, 16 cols
    size_t numRows = 64;
    size_t numCols = 16;
    std::vector<uint8_t> input(numRows * numCols);
    std::iota(input.begin(), input.end(), uint8_t(0));

    auto output = copyScaleStorage(input, {numRows, numCols}, 32, MxScaleStorageLayout::Gfx950);

    // Output size should equal input size (no padding needed)
    ASSERT_EQ(output.size(), numRows * numCols);

    const std::array<std::byte, 8> expectedPrefix{
        std::byte{0},  std::byte{0},  std::byte{4},  std::byte{4},
        std::byte{16}, std::byte{16}, std::byte{20}, std::byte{20},
    };
    for (size_t index = 0; index < expectedPrefix.size(); ++index)
        EXPECT_EQ(output[index], expectedPrefix[index]) << "prefix index " << index;

    // All elements should be present (permutation preserves data)
    std::vector<std::byte> sortedOutput = output;
    std::vector<std::byte> sortedInput = encodedBytes(input);
    std::sort(sortedOutput.begin(), sortedOutput.end());
    std::sort(sortedInput.begin(), sortedInput.end());
    EXPECT_EQ(sortedOutput, sortedInput);
}

TEST(MxScaleStorageGFX950Test, UnalignedRows) {
    // numRows = 50 (not divisible by 32), numCols = 16 (divisible by 8)
    size_t numRows = 50;
    size_t numCols = 16;
    std::vector<uint8_t> input(numRows * numCols);
    std::iota(input.begin(), input.end(), uint8_t(1));

    auto output = copyScaleStorage(input, {numRows, numCols}, 32, MxScaleStorageLayout::Gfx950);

    ASSERT_EQ(output.size(), 64 * 16);
}

TEST(MxScaleStorageGFX950Test, UnalignedCols) {
    // numRows = 64 (divisible by 32), numCols = 13 (not divisible by 8)
    size_t numRows = 64;
    size_t numCols = 13;
    std::vector<uint8_t> input(numRows * numCols);
    std::iota(input.begin(), input.end(), uint8_t(1));

    auto output = copyScaleStorage(input, {numRows, numCols}, 32, MxScaleStorageLayout::Gfx950);

    ASSERT_EQ(output.size(), 64 * 16);
}

TEST(MxScaleStorageGFX950Test, BothUnaligned) {
    // numRows = 50 (not divisible by 32), numCols = 13 (not divisible by 8)
    size_t numRows = 50;
    size_t numCols = 13;
    std::vector<uint8_t> input(numRows * numCols);
    std::iota(input.begin(), input.end(), uint8_t(1));

    auto output = copyScaleStorage(input, {numRows, numCols}, 32, MxScaleStorageLayout::Gfx950);

    ASSERT_EQ(output.size(), 64 * 16);
}

TEST(MxScaleStorageGFX950Test, PaddedMatchesManualPad) {
    // Verify that converting unaligned data gives the same result as manually
    // padding the data and then converting with aligned sizes.
    size_t numRows = 50;
    size_t numCols = 13;
    constexpr size_t paddedRows = 64;
    constexpr size_t paddedCols = 16;

    std::vector<uint8_t> input(numRows * numCols);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<uint8_t>(i);

    auto output1 = copyScaleStorage(input, {numRows, numCols}, 32, MxScaleStorageLayout::Gfx950);

    // Method 2: Manually pad and call with aligned sizes
    std::vector<uint8_t> manualPadded(paddedRows * paddedCols, 0);
    for (size_t r = 0; r < numRows; ++r) {
        std::copy(input.begin() + r * numCols, input.begin() + r * numCols + numCols,
                  manualPadded.begin() + r * paddedCols);
    }
    auto output2 =
        copyScaleStorage(manualPadded, {paddedRows, paddedCols}, 32, MxScaleStorageLayout::Gfx950);

    ASSERT_EQ(output1.size(), output2.size());
    EXPECT_EQ(output1, output2);
}

TEST(MxScaleStorageGFX950Test, AlignedNoExtraPadding) {
    // When sizes are already aligned, output should be same size as input
    size_t numRows = 32;
    size_t numCols = 8;
    std::vector<uint8_t> input(numRows * numCols);
    std::iota(input.begin(), input.end(), uint8_t{0});

    auto output = copyScaleStorage(input, {numRows, numCols}, 32, MxScaleStorageLayout::Gfx950);

    ASSERT_EQ(output.size(), input.size());

    // All elements should be preserved
    std::vector<std::byte> sortedOutput = output;
    std::vector<std::byte> sortedInput = encodedBytes(input);
    std::sort(sortedOutput.begin(), sortedOutput.end());
    std::sort(sortedInput.begin(), sortedInput.end());
    EXPECT_EQ(sortedOutput, sortedInput);
}

TEST(MxScaleStorageGFX950Test, InputSizeMismatch) {
    std::vector<uint8_t> input(100);
    EXPECT_THROW(copyScaleStorage(input, {64, 16}, 32, MxScaleStorageLayout::Gfx950),
                 std::invalid_argument);
}

}  // namespace amd_gpu_layout_test
