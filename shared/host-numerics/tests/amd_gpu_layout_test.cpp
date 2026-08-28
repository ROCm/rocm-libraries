// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <roc/host_numerics/amd_gpu_layout/mx.hpp>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#error "AMDGPULayout consumers must not inherit OpenMP compile flags."
#endif

using namespace roc::host_numerics::amd_gpu_layout;

namespace {
size_t runtimeSize(size_t value) {
    volatile size_t runtimeValue = value;
    return runtimeValue;
}

std::vector<std::byte> copyScaleStorage(const std::vector<uint8_t>& input,
                                        std::array<size_t, 2> dimensions, size_t blockSize,
                                        MxScaleStorageLayout layout) {
    return copyMxScaleStorageToPhysicalLayout(reinterpret_cast<const std::byte*>(input.data()),
                                              input.size(), dimensions, blockSize, layout);
}

std::vector<std::byte> encodedBytes(const std::vector<uint8_t>& input) {
    std::vector<std::byte> result(input.size());
    std::transform(input.begin(), input.end(), result.begin(),
                   [](uint8_t value) { return static_cast<std::byte>(value); });
    return result;
}

std::vector<size_t> testStrides(const std::vector<size_t>& sizes,
                                const std::vector<size_t>& dimensionOrder = {}) {
    std::vector<size_t> strides(sizes.size());
    size_t stride = 1;
    if (dimensionOrder.empty()) {
        for (size_t dimension = 0; dimension < sizes.size(); ++dimension) {
            strides[dimension] = stride;
            stride *= sizes[dimension];
        }
    } else {
        for (const size_t dimension : dimensionOrder) {
            strides[dimension] = stride;
            stride *= sizes[dimension];
        }
    }
    return strides;
}
}  // namespace

class MultiIndex {
   public:
    std::vector<size_t> sizes;
    std::vector<size_t> indexes;
    std::vector<size_t> strides;

    explicit MultiIndex(const std::vector<size_t>& sizes) : sizes(sizes) {
        indexes.resize(sizes.size(), 0);
        strides = testStrides(this->sizes);
    }

    MultiIndex(const std::vector<size_t>& sizes, const std::vector<size_t>& dimensionOrder)
        : sizes(sizes) {
        indexes.resize(sizes.size(), 0);
        strides = testStrides(sizes, dimensionOrder);
    }

    size_t index() const {
        size_t result = 0;
        for (size_t dimension = 0; dimension < indexes.size(); ++dimension)
            result += indexes[dimension] * strides[dimension];
        return result;
    }

    size_t operator*() const {
        return index();
    }

    MultiIndex& operator++() {
        for (size_t dimension = 0; dimension < indexes.size(); ++dimension) {
            ++indexes[dimension];
            if (indexes[dimension] < sizes[dimension]) break;
            if (dimension == indexes.size() - 1) break;
            indexes[dimension] = 0;
        }
        return *this;
    }

    bool isEnd() const {
        return !indexes.empty() && indexes.back() >= sizes.back();
    }
};

void FillSwizzle(std::vector<float>& scales, size_t k, size_t mn,
                 std::vector<size_t> const& preSwizzleSize) {
    auto tileMN = preSwizzleSize[0];
    auto tileK = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t nLanesPerSIMD = 16;
    size_t nSIMDsPerWave = 4;
    size_t nSIMDIndex = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    auto numTilesK = k / tileK;
    auto numTilesMN = mn / tileMN;

    auto sizes = {nVGPRIndex,    nVGPRBlock,      nSIMDBlock,      numTilesK,
                  nLanesPerSIMD, nSIMDIndexIndex, nSIMDIndexBlock, numTilesMN};

    auto mi = MultiIndex(sizes);

    for (; !mi.isEnd(); ++mi) {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for (size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

void FillPreSwizzle(std::vector<float>& scales, size_t k, size_t mn,
                    std::vector<size_t> const& preSwizzleSize) {
    auto tileMN = preSwizzleSize[0];
    auto tileK = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t nLanesPerSIMD = 16;
    size_t nSIMDsPerWave = 4;
    size_t nSIMDIndex = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    auto numTilesK = k / tileK;
    auto numTilesMN = mn / tileMN;

    std::vector<size_t> sizes = {nVGPRIndex,    nVGPRBlock,      nSIMDBlock,      numTilesK,
                                 nLanesPerSIMD, nSIMDIndexIndex, nSIMDIndexBlock, numTilesMN};

    std::vector<size_t> dimOrder;
    if (tileMN == 64) {
        // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
        dimOrder = {6, 1, 2, 3, 4, 5, 0, 7};
    } else if (tileMN == 32 && subTileK == 4) {
        // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
        //              swap nSIMDBlock (2) and nVGPRBlock (1)
        dimOrder = {6, 2, 1, 3, 4, 5, 0, 7};
    } else if (tileMN == 32 && subTileK == 2) {
        // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
        dimOrder = {1, 2, 0, 3, 4, 5, 6, 7};
    }

    auto mi = MultiIndex(sizes, dimOrder);

    for (; !mi.isEnd(); ++mi) {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for (size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

void FillSwizzleAndTile(std::vector<float>& scales, size_t k, size_t mn,
                        std::vector<size_t> const& preSwizzleSize,
                        std::vector<size_t> const& preTileSize) {
    auto tileMN = preSwizzleSize[0];
    auto tileK = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t ptTileSizeK = preTileSize[0];
    size_t ptTileSizeMN = preTileSize[1];

    size_t nLanesPerSIMD = 16;
    size_t nSIMDsPerWave = 4;
    size_t nSIMDIndex = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    auto sizes = {
        nVGPRIndex,    nVGPRBlock,      nSIMDBlock,      ptTileSizeK / tileK,   k / ptTileSizeK,
        nLanesPerSIMD, nSIMDIndexIndex, nSIMDIndexBlock, ptTileSizeMN / tileMN, mn / ptTileSizeMN};

    auto mi = MultiIndex(sizes);

    for (; !mi.isEnd(); ++mi) {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for (size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

void FillPreSwizzleAndTile(std::vector<float>& scales, size_t k, size_t mn,
                           std::vector<size_t> const& preSwizzleSize,
                           std::vector<size_t> const& preTileSize) {
    auto tileMN = preSwizzleSize[0];
    auto tileK = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t ptTileSizeK = preTileSize[0];
    size_t ptTileSizeMN = preTileSize[1];

    size_t nLanesPerSIMD = 16;
    size_t nSIMDsPerWave = 4;
    size_t nSIMDIndex = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    std::vector<size_t> sizes = {
        nVGPRIndex,    nVGPRBlock,      nSIMDBlock,      ptTileSizeK / tileK,   k / ptTileSizeK,
        nLanesPerSIMD, nSIMDIndexIndex, nSIMDIndexBlock, ptTileSizeMN / tileMN, mn / ptTileSizeMN};

    std::vector<size_t> dimOrder;
    if (tileMN == 64) {
        // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
        // Pre tile: push workgroup tiles (4 and 9) to the end
        dimOrder = {7, 1, 2, 3, 5, 6, 0, 8, 4, 9};
    } else if (tileMN == 32 && subTileK == 4) {
        // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
        //              swap nSIMDBlock (2) and nVGPRBlock (1)
        // Pre tile: push workgroup tiles (4 and 9) to the end
        dimOrder = {7, 2, 1, 3, 5, 6, 0, 8, 4, 9};
    } else if (tileMN == 32 && subTileK == 2) {
        // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
        // Pre tile: push workgroup tiles (4 and 9) to the end
        dimOrder = {1, 2, 0, 3, 5, 6, 7, 8, 4, 9};
    }

    auto mi = MultiIndex(sizes, dimOrder);

    for (; !mi.isEnd(); ++mi) {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for (size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

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
                 std::runtime_error);
}

TEST(MxScaleStorageGFX1250Test, ThrowsOnZeroBlock) {
    std::vector<uint8_t> in(4);
    EXPECT_THROW(copyScaleStorage(in, {1, 4}, 0, MxScaleStorageLayout::Gfx1250),
                 std::runtime_error);
}

TEST(MxScaleStorageGFX1250Test, RejectsUnsupportedBlockAndOverflow) {
    std::vector<uint8_t> input;
    EXPECT_THROW(copyScaleStorage(input, {0, 0}, 64, MxScaleStorageLayout::Gfx1250),
                 std::runtime_error);
    const size_t maximumSize = runtimeSize(std::numeric_limits<size_t>::max());
    EXPECT_THROW(copyScaleStorage(input, {maximumSize, 2}, 16, MxScaleStorageLayout::Gfx1250),
                 std::overflow_error);
}

TEST(MxScaleStorageGFX1250Test, ThrowsOnSizeMismatch) {
    std::vector<uint8_t> in(7);  // slow*fast = 8 expected
    EXPECT_THROW(copyScaleStorage(in, {runtimeSize(2), runtimeSize(4)}, runtimeSize(32),
                                  MxScaleStorageLayout::Gfx1250),
                 std::runtime_error);
}

TEST(MxScaleStorageGFX1250Test, MapsAlignedFastDim) {
    // mxBlock=32 -> dimk=4. slow=2, fast=8 (=2 tiles). Use values v[s*fast+f] = s*10+f.
    constexpr size_t slow = 2, fast = 8, mxBlock = 32;
    constexpr size_t dimk = 128 / mxBlock;  // = 4
    std::vector<uint8_t> in(slow * fast);
    for (size_t s = 0; s < slow; ++s)
        for (size_t f = 0; f < fast; ++f) in[s * fast + f] = static_cast<uint8_t>(s * 10 + f);

    auto out = copyScaleStorage(in, {slow, fast}, mxBlock, MxScaleStorageLayout::Gfx1250);
    ASSERT_EQ(out.size(), slow * fast);  // no padding

    // Output layout: {numTiles, slow, dimk}
    // out[tile, s, j] should equal in[s, tile*dimk + j]
    size_t const numTiles = fast / dimk;
    for (size_t tile = 0; tile < numTiles; ++tile)
        for (size_t s = 0; s < slow; ++s)
            for (size_t j = 0; j < dimk; ++j) {
                size_t const outIdx = tile * (slow * dimk) + s * dimk + j;
                size_t const inIdx = s * fast + (tile * dimk + j);
                EXPECT_EQ(std::to_integer<uint8_t>(out[outIdx]), in[inIdx])
                    << "tile=" << tile << " s=" << s << " j=" << j;
            }
}

TEST(MxScaleStorageGFX1250Test, PadsFastDimWithZeros) {
    // mxBlock=16 -> dimk=8. slow=3, fast=10 -> paddedFast=16, two tiles, second
    // tile has 6 padded zero scales.
    constexpr size_t slow = 3, fast = 10, mxBlock = 16;
    constexpr size_t dimk = 128 / mxBlock;  // = 8
    std::vector<uint8_t> in(slow * fast);
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = static_cast<uint8_t>(i + 1);  // non-zero so we can spot pads

    auto out = copyScaleStorage(in, {slow, fast}, mxBlock, MxScaleStorageLayout::Gfx1250);
    ASSERT_EQ(out.size(), slow * 16);

    size_t const numTiles = 16 / dimk;
    size_t seenZeros = 0;
    for (size_t tile = 0; tile < numTiles; ++tile)
        for (size_t s = 0; s < slow; ++s)
            for (size_t j = 0; j < dimk; ++j) {
                size_t const outIdx = tile * (slow * dimk) + s * dimk + j;
                size_t const srcFast = tile * dimk + j;
                if (srcFast < fast)
                    EXPECT_EQ(std::to_integer<uint8_t>(out[outIdx]), in[s * fast + srcFast]);
                else {
                    EXPECT_EQ(out[outIdx], std::byte{0});
                    ++seenZeros;
                }
            }
    EXPECT_EQ(seenZeros, slow * (16 - fast));
}

TEST(MxScaleStorageGFX1250Test, MultiplesPreservePayload) {
    // For perfectly-aligned fastDim, the swizzle is a pure permutation: every
    // input byte appears exactly once in the output (unsorted equality).
    constexpr size_t slow = 4, fast = 16, mxBlock = 32;
    std::vector<uint8_t> in(slow * fast);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 3 + 7);

    auto out = copyScaleStorage(in, {slow, fast}, mxBlock, MxScaleStorageLayout::Gfx1250);
    ASSERT_EQ(out.size(), in.size());

    std::vector<std::byte> sortedIn = encodedBytes(in);
    auto sortedOut = out;
    std::sort(sortedIn.begin(), sortedIn.end());
    std::sort(sortedOut.begin(), sortedOut.end());
    EXPECT_EQ(sortedIn, sortedOut);
}

TEST(MxScaleStorageLayoutTest, CopiesNaturalStorage) {
    EXPECT_TRUE(
        copyMxScaleStorageToPhysicalLayout(nullptr, 0, {0, 0}, 32, MxScaleStorageLayout::Natural)
            .empty());

    const std::array<std::byte, 8> natural{
        std::byte{1}, std::byte{2}, std::byte{3}, std::byte{0},
        std::byte{4}, std::byte{5}, std::byte{6}, std::byte{0},
    };
    const std::vector<std::byte> physical = copyMxScaleStorageToPhysicalLayout(
        natural.data(), natural.size(), {2, 4}, 32, MxScaleStorageLayout::Natural);
    const std::array<uint8_t, 8> expected{1, 2, 3, 0, 4, 5, 6, 0};

    ASSERT_EQ(physical.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index)
        EXPECT_EQ(std::to_integer<uint8_t>(physical[index]), expected[index]);
}

TEST(MxScaleStorageLayoutTest, RejectsInvalidNaturalStorageSize) {
    const std::array<std::byte, 7> natural{};
    EXPECT_THROW(copyMxScaleStorageToPhysicalLayout(natural.data(), natural.size(), {2, 4}, 32,
                                                    MxScaleStorageLayout::Natural),
                 std::invalid_argument);
}

TEST(MxScaleStorageLayoutTest, RejectsOverflowingNaturalStorageDimensions) {
    EXPECT_THROW(copyMxScaleStorageToPhysicalLayout(
                     nullptr, 0, {runtimeSize(std::numeric_limits<size_t>::max()), runtimeSize(2)},
                     32, MxScaleStorageLayout::Natural),
                 std::overflow_error);
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
