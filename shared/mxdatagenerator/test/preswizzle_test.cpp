// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <mxDataGenerator/PreSwizzle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace DGen;

// ============================================================================
// Tests for product() helper function
// ============================================================================

TEST(PreSwizzleTest, ProductEmptyVector)
{
    std::vector<size_t> empty;
    EXPECT_EQ(product(empty), 1);
}

TEST(PreSwizzleTest, ProductSingleElement)
{
    std::vector<size_t> vec = {5};
    EXPECT_EQ(product(vec), 5);
}

TEST(PreSwizzleTest, ProductMultipleElements)
{
    std::vector<size_t> vec = {2, 3, 4};
    EXPECT_EQ(product(vec), 24);
}

TEST(PreSwizzleTest, ProductWithZero)
{
    std::vector<size_t> vec = {2, 0, 4};
    EXPECT_EQ(product(vec), 0);
}

TEST(PreSwizzleTest, ProductWithOne)
{
    std::vector<size_t> vec = {1, 5, 1, 3};
    EXPECT_EQ(product(vec), 15);
}

// ============================================================================
// Tests for computeStrides()
// ============================================================================

TEST(PreSwizzleTest, ComputeStridesEmpty)
{
    std::vector<size_t> sizes;
    auto                strides = computeStrides(sizes);
    EXPECT_TRUE(strides.empty());
}

TEST(PreSwizzleTest, ComputeStridesSingleElement)
{
    std::vector<size_t> sizes   = {10};
    auto                strides = computeStrides(sizes);
    ASSERT_EQ(strides.size(), 1);
    EXPECT_EQ(strides[0], 1);
}

TEST(PreSwizzleTest, ComputeStridesColMajor)
{
    std::vector<size_t> sizes   = {10, 20, 30};
    auto                strides = computeStrides(sizes);
    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 1);
    EXPECT_EQ(strides[1], 10);
    EXPECT_EQ(strides[2], 200);
}

// ============================================================================
// Tests for computeShuffledStrides()
// ============================================================================

TEST(PreSwizzleTest, ComputeShuffledStridesIdentity)
{
    std::vector<size_t> sizes         = {2, 3, 4};
    std::vector<size_t> dimOrder      = {0, 1, 2};
    auto                strides       = computeShuffledStrides(sizes, dimOrder);
    auto                normalStrides = computeStrides(sizes);
    EXPECT_EQ(strides, normalStrides);
}

TEST(PreSwizzleTest, ComputeShuffledStridesReverse)
{
    std::vector<size_t> sizes    = {2, 3, 4};
    std::vector<size_t> dimOrder = {2, 1, 0};
    auto                strides  = computeShuffledStrides(sizes, dimOrder);
    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12); // 3 * 4
    EXPECT_EQ(strides[1], 4); // 4
    EXPECT_EQ(strides[2], 1); // 1
}

TEST(PreSwizzleTest, ComputeShuffledStridesCustomOrder)
{
    std::vector<size_t> sizes    = {2, 3, 4};
    std::vector<size_t> dimOrder = {1, 0, 2};
    auto                strides  = computeShuffledStrides(sizes, dimOrder);
    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 3); // sizes[1] = 3
    EXPECT_EQ(strides[1], 1); // first in order
    EXPECT_EQ(strides[2], 6); // 2 * 3
}

// ============================================================================
// Tests for shuffleDims()
// ============================================================================

TEST(PreSwizzleTest, ShuffleDimsIdentity)
{
    std::vector<int>    input      = {0, 1, 2, 3, 4, 5};
    std::vector<size_t> sizes      = {2, 3};
    auto                srcStrides = computeStrides(sizes);
    auto                output     = shuffleDims(input, sizes, srcStrides, srcStrides);
    EXPECT_EQ(input, output);
}

TEST(PreSwizzleTest, ShuffleDimsTranspose2D)
{
    // Input is 2x3 matrix in col-major order: [[0,1,2], [3,4,5]]
    std::vector<int>    input = {0, 3, 1, 4, 2, 5};
    std::vector<size_t> sizes = {2, 3};

    auto                srcStrides = computeStrides(sizes);
    std::vector<size_t> dimOrder   = {1, 0}; // transpose
    auto                dstStrides = computeShuffledStrides(sizes, dimOrder);

    auto output = shuffleDims(input, sizes, dstStrides, srcStrides);

    // After shuffle: 3x2 matrix [[0,3], [1,4], [2,5]]
    std::vector<int> expected = {0, 1, 2, 3, 4, 5};
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, ShuffleDims3D)
{
    // 2x2x2 cube: input(i,j,k) = 4k + 2j + i (col-major storage)
    std::vector<int>    input = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<size_t> sizes = {2, 2, 2};

    auto                srcStrides = computeStrides(sizes);
    std::vector<size_t> dimOrder   = {2, 1, 0}; // reverse dimensions
    auto                dstStrides = computeShuffledStrides(sizes, dimOrder);

    auto output = shuffleDims(input, sizes, dstStrides, srcStrides);

    // After shuffle: ouput(i,j,k) = input(k,j,i) = 4i + 2j + k
    std::vector<int> expected = {0, 4, 2, 6, 1, 5, 3, 7};
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, ShuffleDimsSizeMismatch)
{
    std::vector<int>    input      = {0, 1, 2, 3};
    std::vector<size_t> sizes      = {2, 3}; // 6 elements expected
    auto                srcStrides = computeStrides(sizes);
    auto                dstStrides = computeStrides(sizes);

    EXPECT_THROW(shuffleDims(input, sizes, dstStrides, srcStrides), std::runtime_error);
}

TEST(PreSwizzleTest, ShuffleDimsTooFewDimensions)
{
    std::vector<int>    input      = {0, 1};
    std::vector<size_t> sizes      = {2};
    auto                srcStrides = computeStrides(sizes);
    auto                dstStrides = computeStrides(sizes);

    EXPECT_THROW(shuffleDims(input, sizes, dstStrides, srcStrides), std::runtime_error);
}

TEST(PreSwizzleTest, ShuffleDimsDimensionMismatch)
{
    std::vector<int>    input      = {0, 1, 2, 3};
    std::vector<size_t> sizes      = {2, 2};
    std::vector<size_t> srcStrides = {1, 2};
    std::vector<size_t> dstStrides = {1, 2, 4}; // wrong size

    EXPECT_THROW(shuffleDims(input, sizes, dstStrides, srcStrides), std::runtime_error);
}

// ============================================================================
// Tests for preSwizzle()
// ============================================================================

// Helper class: Multi-dimensional index iterator
class MultiIndex
{
public:
    // Iterator traits
    using iterator_category = std::forward_iterator_tag;
    using value_type        = size_t;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const size_t*;
    using reference         = size_t;

    std::vector<size_t> sizes;
    std::vector<size_t> indexes;
    std::vector<size_t> strides;
    size_t              totalElements;

    MultiIndex(std::vector<size_t> const& sizes)
        : sizes(sizes)
        , totalElements(product(sizes))
    {
        indexes.resize(sizes.size(), 0);
        strides = computeStrides(this->sizes);
    }

    MultiIndex(std::vector<size_t> const& sizes, std::vector<size_t> const& dimOrder)
        : sizes(sizes)
        , totalElements(product(sizes))
    {
        indexes.resize(sizes.size(), 0);
        strides = computeShuffledStrides(sizes, dimOrder);
    }

    // Create end iterator
    static MultiIndex end(std::vector<size_t> const& sizes)
    {
        MultiIndex it(sizes);
        if(!it.sizes.empty())
            it.indexes.back() = it.sizes.back(); // Last dimension past the end
        return it;
    }

    static MultiIndex end(std::vector<size_t> const& sizes, std::vector<size_t> const& dimOrder)
    {
        MultiIndex it(sizes, dimOrder);
        if(!it.sizes.empty())
            it.indexes.back() = it.sizes.back(); // Last dimension past the end
        return it;
    }

    // Get linear index from multi-dimensional coordinates
    size_t index() const
    {
        size_t rv = 0;
        for(size_t i = 0; i < indexes.size(); ++i)
            rv += indexes[i] * strides[i];
        return rv;
    }

    // Dereference operator returns the linear index
    size_t operator*() const
    {
        return index();
    }

    // Pre-increment
    MultiIndex& operator++()
    {
        for(size_t i = 0; i < indexes.size(); ++i)
        {
            indexes[i]++;
            if(indexes[i] < sizes[i])
                break;

            // If this is the last dimension and we've overflowed,
            // don't reset - leave it at sizes[i] to indicate end
            if(i == indexes.size() - 1)
                break;

            indexes[i] = 0;
        }
        return *this;
    }

    // Post-increment
    MultiIndex operator++(int)
    {
        MultiIndex tmp = *this;
        ++(*this);
        return tmp;
    }

    // Equality comparison
    bool operator==(const MultiIndex& other) const
    {
        return indexes == other.indexes && strides == other.strides;
    }

    // Inequality comparison
    bool operator!=(const MultiIndex& other) const
    {
        return !(*this == other);
    }

    // Check if we're at the end
    bool isEnd() const
    {
        return !indexes.empty() && indexes.back() >= sizes.back();
    }
};

TEST(PreSwizzleTest, MultiIndexIteratorInterface)
{
    std::vector<size_t> sizes = {2, 3, 4};
    MultiIndex          mi(sizes);
    MultiIndex          end = MultiIndex::end(sizes);

    // Test dereference operator
    EXPECT_EQ(*mi, 0); // First element should have index 0

    // Test pre-increment
    ++mi;
    EXPECT_EQ(*mi, 1);

    // Test post-increment
    MultiIndex mi2 = mi++;
    EXPECT_EQ(*mi2, 1); // Original value
    EXPECT_EQ(*mi, 2); // Incremented value

    // Test inequality
    EXPECT_NE(mi, end);

    // Iterate through all elements
    mi = MultiIndex(sizes);
    std::vector<size_t> collectedIndices;

    for(; mi != end; ++mi)
    {
        collectedIndices.push_back(*mi);
    }

    // Should have iterated through all 2*3*4 = 24 elements
    EXPECT_EQ(collectedIndices.size(), 24);
    EXPECT_EQ(mi, end);

    // Verify indices are sequential
    std::vector<size_t> expected
        = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    EXPECT_EQ(collectedIndices, expected);
}

TEST(PreSwizzleTest, MultiIndexWithDimOrder)
{
    std::vector<size_t> sizes    = {2, 3};
    std::vector<size_t> dimOrder = {1, 0};
    MultiIndex          mi(sizes, dimOrder);
    MultiIndex          end = MultiIndex::end(sizes, dimOrder);

    std::vector<size_t> collectedIndices;
    for(; mi != end; ++mi)
    {
        collectedIndices.push_back(*mi);
    }

    EXPECT_EQ(collectedIndices.size(), 6);

    std::vector<size_t> expected = {0, 3, 1, 4, 2, 5};
    EXPECT_EQ(collectedIndices, expected);
}

void FillSwizzle(std::vector<float>&        scales,
                 size_t                     k,
                 size_t                     mn,
                 std::vector<size_t> const& preSwizzleSize)
{
    auto tileMN   = preSwizzleSize[0];
    auto tileK    = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t nLanesPerSIMD   = 16;
    size_t nSIMDsPerWave   = 4;
    size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    auto numTilesK  = k / tileK;
    auto numTilesMN = mn / tileMN;

    auto sizes = {nVGPRIndex,
                  nVGPRBlock,
                  nSIMDBlock,
                  numTilesK,
                  nLanesPerSIMD,
                  nSIMDIndexIndex,
                  nSIMDIndexBlock,
                  numTilesMN};

    auto mi = MultiIndex(sizes);

    for(; !mi.isEnd(); ++mi)
    {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for(size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

void FillPreSwizzle(std::vector<float>&        scales,
                    size_t                     k,
                    size_t                     mn,
                    std::vector<size_t> const& preSwizzleSize)
{
    auto tileMN   = preSwizzleSize[0];
    auto tileK    = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t nLanesPerSIMD   = 16;
    size_t nSIMDsPerWave   = 4;
    size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    auto numTilesK  = k / tileK;
    auto numTilesMN = mn / tileMN;

    std::vector<size_t> sizes = {nVGPRIndex,
                                 nVGPRBlock,
                                 nSIMDBlock,
                                 numTilesK,
                                 nLanesPerSIMD,
                                 nSIMDIndexIndex,
                                 nSIMDIndexBlock,
                                 numTilesMN};

    std::vector<size_t> dimOrder;
    if(tileMN == 64)
    {
        // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
        dimOrder = {6, 1, 2, 3, 4, 5, 0, 7};
    }
    else if(tileMN == 32 && subTileK == 4)
    {
        // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
        //              swap nSIMDBlock (2) and nVGPRBlock (1)
        dimOrder = {6, 2, 1, 3, 4, 5, 0, 7};
    }
    else if(tileMN == 32 && subTileK == 2)
    {
        // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
        dimOrder = {1, 2, 0, 3, 4, 5, 6, 7};
    }

    auto mi = MultiIndex(sizes, dimOrder);

    for(; !mi.isEnd(); ++mi)
    {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for(size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

void FillSwizzleAndTile(std::vector<float>&        scales,
                        size_t                     k,
                        size_t                     mn,
                        std::vector<size_t> const& preSwizzleSize,
                        std::vector<size_t> const& preTileSize)
{
    auto tileMN   = preSwizzleSize[0];
    auto tileK    = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t ptTileSizeK  = preTileSize[0];
    size_t ptTileSizeMN = preTileSize[1];

    size_t nLanesPerSIMD   = 16;
    size_t nSIMDsPerWave   = 4;
    size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    auto sizes = {nVGPRIndex,
                  nVGPRBlock,
                  nSIMDBlock,
                  ptTileSizeK / tileK,
                  k / ptTileSizeK,
                  nLanesPerSIMD,
                  nSIMDIndexIndex,
                  nSIMDIndexBlock,
                  ptTileSizeMN / tileMN,
                  mn / ptTileSizeMN};

    auto mi = MultiIndex(sizes);

    for(; !mi.isEnd(); ++mi)
    {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for(size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

void FillPreSwizzleAndTile(std::vector<float>&        scales,
                           size_t                     k,
                           size_t                     mn,
                           std::vector<size_t> const& preSwizzleSize,
                           std::vector<size_t> const& preTileSize)
{
    auto tileMN   = preSwizzleSize[0];
    auto tileK    = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t ptTileSizeK  = preTileSize[0];
    size_t ptTileSizeMN = preTileSize[1];

    size_t nLanesPerSIMD   = 16;
    size_t nSIMDsPerWave   = 4;
    size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    std::vector<size_t> sizes = {nVGPRIndex,
                                 nVGPRBlock,
                                 nSIMDBlock,
                                 ptTileSizeK / tileK,
                                 k / ptTileSizeK,
                                 nLanesPerSIMD,
                                 nSIMDIndexIndex,
                                 nSIMDIndexBlock,
                                 ptTileSizeMN / tileMN,
                                 mn / ptTileSizeMN};

    std::vector<size_t> dimOrder;
    if(tileMN == 64)
    {
        // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
        // Pre tile: push workgroup tiles (4 and 9) to the end
        dimOrder = {7, 1, 2, 3, 5, 6, 0, 8, 4, 9};
    }
    else if(tileMN == 32 && subTileK == 4)
    {
        // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
        //              swap nSIMDBlock (2) and nVGPRBlock (1)
        // Pre tile: push workgroup tiles (4 and 9) to the end
        dimOrder = {7, 2, 1, 3, 5, 6, 0, 8, 4, 9};
    }
    else if(tileMN == 32 && subTileK == 2)
    {
        // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
        // Pre tile: push workgroup tiles (4 and 9) to the end
        dimOrder = {1, 2, 0, 3, 5, 6, 7, 8, 4, 9};
    }

    auto mi = MultiIndex(sizes, dimOrder);

    for(; !mi.isEnd(); ++mi)
    {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for(size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

TEST(PreSwizzleTest, PreSwizzleOnlySwizzle64)
{
    size_t             k  = 256;
    size_t             mn = 128;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes          = {k, mn};
    std::vector<size_t> preSwizzleSize = {64, 256, 4}; // tileMN=64, tileK=256, subTileK=4
    std::vector<size_t> preTileSize; // empty
    FillSwizzle(input, k, mn, preSwizzleSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzle(expected, k, mn, preSwizzleSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleOnlySwizzle32SubTile4)
{
    size_t             k  = 128;
    size_t             mn = 64;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes          = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 4}; // tileMN=32, tileK=128, subTileK=4
    std::vector<size_t> preTileSize; // empty
    FillSwizzle(input, k, mn, preSwizzleSize);

    auto               output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    std::vector<float> expected(k * mn);
    FillPreSwizzle(expected, k, mn, preSwizzleSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleOnlySwizzle32SubTile2)
{
    size_t             k  = 128;
    size_t             mn = 64;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes          = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 2}; // tileMN=32, tileK=128, subTileK=2
    std::vector<size_t> preTileSize; // empty
    FillSwizzle(input, k, mn, preSwizzleSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzle(expected, k, mn, preSwizzleSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleOnlyPreTile)
{
    size_t             k  = 128;
    size_t             mn = 64;
    std::vector<float> input(k * mn);
    std::iota(input.begin(), input.end(), 0.0f);

    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize; // empty
    std::vector<size_t> preTileSize = {16, 16};

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleBothSwizzleAndTile64)
{
    size_t             k  = 256;
    size_t             mn = 128;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes          = {k, mn};
    std::vector<size_t> preSwizzleSize = {64, 256, 4};
    std::vector<size_t> preTileSize    = {256, 64};
    FillSwizzleAndTile(input, k, mn, preSwizzleSize, preTileSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzleAndTile(expected, k, mn, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleBothSwizzleAndTile32SubTile4)
{
    size_t             k  = 512;
    size_t             mn = 128;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes          = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize    = {128, 32};
    FillSwizzleAndTile(input, k, mn, preSwizzleSize, preTileSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzleAndTile(expected, k, mn, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleBothSwizzleAndTile32SubTile2)
{
    size_t             k  = 256;
    size_t             mn = 64;
    std::vector<float> input(k * mn);

    std::vector<size_t> sizes          = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 2};
    std::vector<size_t> preTileSize    = {128, 64};
    FillSwizzleAndTile(input, k, mn, preSwizzleSize, preTileSize);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expected(k * mn);
    FillPreSwizzleAndTile(expected, k, mn, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, PreSwizzleInvalidTileMN)
{
    std::vector<float>  input(128 * 64);
    std::vector<size_t> sizes          = {128, 64};
    std::vector<size_t> preSwizzleSize = {48, 128, 4}; // tileMN not 32 or 64
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleTileKNotMultipleOf4)
{
    std::vector<float>  input(127 * 64);
    std::vector<size_t> sizes          = {127, 64};
    std::vector<size_t> preSwizzleSize = {32, 127, 4}; // tileK not multiple of 4
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleSizeMismatch)
{
    std::vector<float>  input(100); // Wrong size
    std::vector<size_t> sizes          = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleBatchDimensionNotSupported)
{
    std::vector<float>  input(2 * 128 * 64);
    std::vector<size_t> sizes          = {2, 128, 64}; // 3D tensor (batch dimension)
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzleInvalidPreSwizzleSize)
{
    std::vector<float>  input(128 * 64);
    std::vector<size_t> sizes          = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128}; // Only 2 elements, need 3
    std::vector<size_t> preTileSize;

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzlePtTileSizeKZero)
{
    std::vector<float>  input(128 * 64);
    std::vector<size_t> sizes          = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 256, 4}; // tileK > sizes[0]
    std::vector<size_t> preTileSize    = {128, 64};

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

TEST(PreSwizzleTest, PreSwizzlePtTileSizeMNZero)
{
    std::vector<float>  input(128 * 64);
    std::vector<size_t> sizes          = {128, 64};
    std::vector<size_t> preSwizzleSize = {128, 128, 4}; // tileMN > sizes[1]
    std::vector<size_t> preTileSize    = {128, 64};

    EXPECT_THROW(preSwizzle(input, sizes, preSwizzleSize, preTileSize), std::runtime_error);
}

// ============================================================================
// Edge Cases and Integration Tests
// ============================================================================

TEST(PreSwizzleTest, PreSwizzleIdentityOnSmallData)
{
    // Test that swizzling preserves all data
    std::vector<int>    input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<size_t> sizes = {2, 4};
    std::vector<size_t> preSwizzleSize;
    std::vector<size_t> preTileSize = {2, 4}; // Same as sizes, should be identity

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());

    // Verify all elements are present
    std::vector<int> sortedOutput = output;
    std::vector<int> sortedInput  = input;
    std::sort(sortedOutput.begin(), sortedOutput.end());
    std::sort(sortedInput.begin(), sortedInput.end());
    EXPECT_EQ(sortedOutput, sortedInput);
}

TEST(PreSwizzleTest, PreSwizzleDoubleType)
{
    // Test that templates work with different types
    std::vector<double> input(128 * 64);
    for(size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<double>(i) * 0.1;

    std::vector<size_t> sizes          = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleIntegerType)
{
    // Test that templates work with integer types
    size_t           k  = 128;
    size_t           mn = 64;
    std::vector<int> input(k * mn);

    std::vector<size_t> sizes          = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;

    // Create float versions for helper functions, then convert
    std::vector<float> inputFloat(k * mn);
    FillSwizzle(inputFloat, k, mn, preSwizzleSize);
    for(size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<int>(inputFloat[i]);

    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);

    std::vector<float> expectedFloat(k * mn);
    FillPreSwizzle(expectedFloat, k, mn, preSwizzleSize);
    std::vector<int> expected(k * mn);
    for(size_t i = 0; i < expected.size(); ++i)
        expected[i] = static_cast<int>(expectedFloat[i]);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_EQ(output, expected);
}

// ============================================================================
// Tests for roundUp() helper
// ============================================================================

TEST(PreSwizzleTest, RoundUpAlreadyAligned)
{
    EXPECT_EQ(roundUp(32, 32), 32);
    EXPECT_EQ(roundUp(64, 32), 64);
    EXPECT_EQ(roundUp(8, 8), 8);
    EXPECT_EQ(roundUp(16, 8), 16);
}

TEST(PreSwizzleTest, RoundUpNotAligned)
{
    EXPECT_EQ(roundUp(1, 32), 32);
    EXPECT_EQ(roundUp(31, 32), 32);
    EXPECT_EQ(roundUp(33, 32), 64);
    EXPECT_EQ(roundUp(50, 32), 64);
    EXPECT_EQ(roundUp(1, 8), 8);
    EXPECT_EQ(roundUp(7, 8), 8);
    EXPECT_EQ(roundUp(9, 8), 16);
    EXPECT_EQ(roundUp(13, 8), 16);
}

// ============================================================================
// Tests for preSwizzleScalesGFX950PaddedSize()
// ============================================================================

TEST(PreSwizzleTest, PaddedSizeAligned)
{
    // Already aligned: 64 rows (mult of 32), 16 cols (mult of 8)
    EXPECT_EQ(preSwizzleScalesGFX950PaddedSize(64, 16), 64 * 16);
    EXPECT_EQ(preSwizzleScalesGFX950PaddedSize(32, 8), 32 * 8);
}

TEST(PreSwizzleTest, PaddedSizeUnalignedRows)
{
    // 50 rows -> padded to 64, 16 cols stays
    EXPECT_EQ(preSwizzleScalesGFX950PaddedSize(50, 16), 64 * 16);
}

TEST(PreSwizzleTest, PaddedSizeUnalignedCols)
{
    // 64 rows stays, 13 cols -> padded to 16
    EXPECT_EQ(preSwizzleScalesGFX950PaddedSize(64, 13), 64 * 16);
}

TEST(PreSwizzleTest, PaddedSizeBothUnaligned)
{
    // 50 rows -> 64, 13 cols -> 16
    EXPECT_EQ(preSwizzleScalesGFX950PaddedSize(50, 13), 64 * 16);
}

// ============================================================================
// Tests for preSwizzleScalesGFX950()
// ============================================================================

TEST(PreSwizzleScalesGFX950Test, AlignedSizes)
{
    // Basic test with aligned sizes: 64 rows, 16 cols
    size_t numRows = 64;
    size_t numCols = 16;
    std::vector<uint8_t> input(numRows * numCols);
    std::iota(input.begin(), input.end(), uint8_t(0));

    auto output = preSwizzleScalesGFX950(input, {numRows, numCols});

    // Output size should equal input size (no padding needed)
    ASSERT_EQ(output.size(), numRows * numCols);

    // All elements should be present (permutation preserves data)
    std::vector<uint8_t> sortedOutput = output;
    std::vector<uint8_t> sortedInput  = input;
    std::sort(sortedOutput.begin(), sortedOutput.end());
    std::sort(sortedInput.begin(), sortedInput.end());
    EXPECT_EQ(sortedOutput, sortedInput);
}

TEST(PreSwizzleScalesGFX950Test, UnalignedRows)
{
    // numRows = 50 (not divisible by 32), numCols = 16 (divisible by 8)
    size_t numRows = 50;
    size_t numCols = 16;
    std::vector<uint8_t> input(numRows * numCols);
    std::iota(input.begin(), input.end(), uint8_t(1));

    auto output = preSwizzleScalesGFX950(input, {numRows, numCols});

    // Output should be padded: 64 * 16
    size_t expectedSize = preSwizzleScalesGFX950PaddedSize(numRows, numCols);
    ASSERT_EQ(expectedSize, 64 * 16);
    ASSERT_EQ(output.size(), expectedSize);
}

TEST(PreSwizzleScalesGFX950Test, UnalignedCols)
{
    // numRows = 64 (divisible by 32), numCols = 13 (not divisible by 8)
    size_t numRows = 64;
    size_t numCols = 13;
    std::vector<uint8_t> input(numRows * numCols);
    std::iota(input.begin(), input.end(), uint8_t(1));

    auto output = preSwizzleScalesGFX950(input, {numRows, numCols});

    // Output should be padded: 64 * 16
    size_t expectedSize = preSwizzleScalesGFX950PaddedSize(numRows, numCols);
    ASSERT_EQ(expectedSize, 64 * 16);
    ASSERT_EQ(output.size(), expectedSize);
}

TEST(PreSwizzleScalesGFX950Test, BothUnaligned)
{
    // numRows = 50 (not divisible by 32), numCols = 13 (not divisible by 8)
    size_t numRows = 50;
    size_t numCols = 13;
    std::vector<uint8_t> input(numRows * numCols);
    std::iota(input.begin(), input.end(), uint8_t(1));

    auto output = preSwizzleScalesGFX950(input, {numRows, numCols});

    // Output should be padded: 64 * 16
    size_t expectedSize = preSwizzleScalesGFX950PaddedSize(numRows, numCols);
    ASSERT_EQ(expectedSize, 64 * 16);
    ASSERT_EQ(output.size(), expectedSize);
}

TEST(PreSwizzleScalesGFX950Test, PaddedMatchesManualPad)
{
    // Verify that calling preSwizzleScalesGFX950 with unaligned data gives
    // the same result as manually padding the data and then calling with aligned sizes
    size_t numRows = 50;
    size_t numCols = 13;
    size_t paddedRows = roundUp(numRows, 32); // 64
    size_t paddedCols = roundUp(numCols, 8);  // 16

    std::vector<float> input(numRows * numCols);
    for(size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i) * 0.5f;

    // Method 1: Let preSwizzleScalesGFX950 handle padding internally
    auto output1 = preSwizzleScalesGFX950(input, {numRows, numCols});

    // Method 2: Manually pad and call with aligned sizes
    std::vector<float> manualPadded(paddedRows * paddedCols, 0.0f);
    for(size_t r = 0; r < numRows; ++r)
    {
        std::copy(input.begin() + r * numCols,
                  input.begin() + r * numCols + numCols,
                  manualPadded.begin() + r * paddedCols);
    }
    auto output2 = preSwizzleScalesGFX950(manualPadded, {paddedRows, paddedCols});

    ASSERT_EQ(output1.size(), output2.size());
    EXPECT_EQ(output1, output2);
}

TEST(PreSwizzleScalesGFX950Test, AlignedNoExtraPadding)
{
    // When sizes are already aligned, output should be same size as input
    size_t numRows = 32;
    size_t numCols = 8;
    std::vector<float> input(numRows * numCols);
    std::iota(input.begin(), input.end(), 0.0f);

    auto output = preSwizzleScalesGFX950(input, {numRows, numCols});

    ASSERT_EQ(output.size(), input.size());

    // All elements should be preserved
    std::vector<float> sortedOutput = output;
    std::vector<float> sortedInput  = input;
    std::sort(sortedOutput.begin(), sortedOutput.end());
    std::sort(sortedInput.begin(), sortedInput.end());
    EXPECT_EQ(sortedOutput, sortedInput);
}

TEST(PreSwizzleScalesGFX950Test, InvalidSizesDimension)
{
    std::vector<uint8_t> input(100);
    EXPECT_THROW(preSwizzleScalesGFX950(input, {10, 10, 1}), std::runtime_error);
}

TEST(PreSwizzleScalesGFX950Test, InputSizeMismatch)
{
    std::vector<uint8_t> input(100);
    EXPECT_THROW(preSwizzleScalesGFX950(input, {64, 16}), std::runtime_error);
}

// ============================================================================
// Tests for preSwizzleScalesGFX1250() (dimk-based swizzle for gfx1250 / non-
// rocroller WMMA path).
//
// The swizzle is "pad fast dim to multiple of dimk = 128 / mxBlock; view as
// {slow, fast/dimk, dimk}; permute (1,0,2)". This block tests:
//   * the natural-layout to swizzled-layout index mapping
//   * round-trip preservation of the original payload (after stripping the
//     padded zero scales)
//   * padding behaviour when fastDim % dimk != 0
//   * padded-size helper
//   * input validation
// ============================================================================

TEST(PreSwizzleScalesGFX1250Test, PaddedSizeNoPad)
{
    EXPECT_EQ(preSwizzleScalesGFX1250PaddedSize(/*slowDim=*/8, /*fastDim=*/4, /*mxBlock=*/32), 32);
    EXPECT_EQ(preSwizzleScalesGFX1250PaddedSize(/*slowDim=*/16, /*fastDim=*/8, /*mxBlock=*/16),
              128);
}

TEST(PreSwizzleScalesGFX1250Test, PaddedSizeWithPad)
{
    EXPECT_EQ(preSwizzleScalesGFX1250PaddedSize(8, 5, 32), 8 * 8);
    EXPECT_EQ(preSwizzleScalesGFX1250PaddedSize(8, 7, 16), 8 * 8);
}

TEST(PreSwizzleScalesGFX1250Test, ThrowsOnZeroBlock)
{
    std::vector<uint8_t> in(4);
    EXPECT_THROW(preSwizzleScalesGFX1250(in, 1, 4, 0), std::runtime_error);
    EXPECT_THROW(preSwizzleScalesGFX1250PaddedSize(1, 4, 0), std::runtime_error);
}

TEST(PreSwizzleScalesGFX1250Test, ThrowsOnSizeMismatch)
{
    std::vector<uint8_t> in(7); // slow*fast = 8 expected
    EXPECT_THROW(preSwizzleScalesGFX1250(in, 2, 4, 32), std::runtime_error);
}

TEST(PreSwizzleScalesGFX1250Test, MapsAlignedFastDim)
{
    // mxBlock=32 -> dimk=4. slow=2, fast=8 (=2 tiles). Use values v[s*fast+f] = s*10+f.
    constexpr size_t          slow = 2, fast = 8, mxBlock = 32;
    constexpr size_t          dimk = 128 / mxBlock; // = 4
    std::vector<unsigned int> in(slow * fast);
    for(size_t s = 0; s < slow; ++s)
        for(size_t f = 0; f < fast; ++f)
            in[s * fast + f] = static_cast<unsigned int>(s * 10 + f);

    auto out = preSwizzleScalesGFX1250(in, slow, fast, mxBlock);
    ASSERT_EQ(out.size(), slow * fast); // no padding

    // Output layout: {numTiles, slow, dimk}
    // out[tile, s, j] should equal in[s, tile*dimk + j]
    size_t const numTiles = fast / dimk;
    for(size_t tile = 0; tile < numTiles; ++tile)
        for(size_t s = 0; s < slow; ++s)
            for(size_t j = 0; j < dimk; ++j)
            {
                size_t const outIdx = tile * (slow * dimk) + s * dimk + j;
                size_t const inIdx  = s * fast + (tile * dimk + j);
                EXPECT_EQ(out[outIdx], in[inIdx])
                    << "tile=" << tile << " s=" << s << " j=" << j;
            }
}

TEST(PreSwizzleScalesGFX1250Test, PadsFastDimWithZeros)
{
    // mxBlock=16 -> dimk=8. slow=3, fast=10 -> paddedFast=16, two tiles, second
    // tile has 6 padded zero scales.
    constexpr size_t      slow = 3, fast = 10, mxBlock = 16;
    constexpr size_t      dimk = 128 / mxBlock; // = 8
    std::vector<uint8_t>  in(slow * fast);
    for(size_t i = 0; i < in.size(); ++i)
        in[i] = static_cast<uint8_t>(i + 1); // non-zero so we can spot pads

    auto out = preSwizzleScalesGFX1250(in, slow, fast, mxBlock);
    ASSERT_EQ(out.size(), slow * 16);

    size_t const numTiles = 16 / dimk;
    size_t       seenZeros = 0;
    for(size_t tile = 0; tile < numTiles; ++tile)
        for(size_t s = 0; s < slow; ++s)
            for(size_t j = 0; j < dimk; ++j)
            {
                size_t const outIdx  = tile * (slow * dimk) + s * dimk + j;
                size_t const srcFast = tile * dimk + j;
                if(srcFast < fast)
                    EXPECT_EQ(out[outIdx], in[s * fast + srcFast]);
                else
                {
                    EXPECT_EQ(out[outIdx], 0);
                    ++seenZeros;
                }
            }
    EXPECT_EQ(seenZeros, slow * (16 - fast));
}

TEST(PreSwizzleScalesGFX1250Test, MultiplesPreservePayload)
{
    // For perfectly-aligned fastDim, the swizzle is a pure permutation: every
    // input byte appears exactly once in the output (unsorted equality).
    constexpr size_t     slow = 4, fast = 16, mxBlock = 32;
    std::vector<uint8_t> in(slow * fast);
    for(size_t i = 0; i < in.size(); ++i)
        in[i] = static_cast<uint8_t>(i * 3 + 7);

    auto out = preSwizzleScalesGFX1250(in, slow, fast, mxBlock);
    ASSERT_EQ(out.size(), in.size());

    auto sortedIn  = in;
    auto sortedOut = out;
    std::sort(sortedIn.begin(), sortedIn.end());
    std::sort(sortedOut.begin(), sortedOut.end());
    EXPECT_EQ(sortedIn, sortedOut);
}
