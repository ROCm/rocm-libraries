/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <PreSwizzle.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <numeric>
#include <stdexcept>

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
    auto strides = computeStrides(sizes);
    EXPECT_TRUE(strides.empty());
}

TEST(PreSwizzleTest, ComputeStridesSingleElement)
{
    std::vector<size_t> sizes = {10};
    auto strides = computeStrides(sizes);
    ASSERT_EQ(strides.size(), 1);
    EXPECT_EQ(strides[0], 1);
}

TEST(PreSwizzleTest, ComputeStridesColMajor)
{
    std::vector<size_t> sizes = {10, 20, 30};
    auto strides = computeStrides(sizes);
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
    std::vector<size_t> sizes = {2, 3, 4};
    std::vector<size_t> dimOrder = {0, 1, 2};
    auto strides = computeShuffledStrides(sizes, dimOrder);
    auto normalStrides = computeStrides(sizes);
    EXPECT_EQ(strides, normalStrides);
}

TEST(PreSwizzleTest, ComputeShuffledStridesReverse)
{
    std::vector<size_t> sizes = {2, 3, 4};
    std::vector<size_t> dimOrder = {2, 1, 0};
    auto strides = computeShuffledStrides(sizes, dimOrder);
    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12);  // 3 * 4
    EXPECT_EQ(strides[1], 4);   // 4
    EXPECT_EQ(strides[2], 1);   // 1
}

TEST(PreSwizzleTest, ComputeShuffledStridesCustomOrder)
{
    std::vector<size_t> sizes = {2, 3, 4};
    std::vector<size_t> dimOrder = {1, 0, 2};
    auto strides = computeShuffledStrides(sizes, dimOrder);
    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 3);   // sizes[1] = 3
    EXPECT_EQ(strides[1], 1);   // first in order
    EXPECT_EQ(strides[2], 6);   // 2 * 3
}

// ============================================================================
// Tests for shuffleDims()
// ============================================================================

TEST(PreSwizzleTest, ShuffleDimsIdentity)
{
    std::vector<int> input = {0, 1, 2, 3, 4, 5};
    std::vector<size_t> sizes = {2, 3};
    auto srcStrides = computeStrides(sizes);
    auto output = shuffleDims(input, sizes, srcStrides, srcStrides);
    EXPECT_EQ(input, output);
}

TEST(PreSwizzleTest, ShuffleDimsTranspose2D)
{
    // Input is 2x3 matrix in row-major order: [[0,1,2], [3,4,5]]
    std::vector<int> input = {0, 1, 2, 3, 4, 5};
    std::vector<size_t> sizes = {2, 3};
    
    auto srcStrides = computeStrides(sizes);
    std::vector<size_t> dimOrder = {1, 0};  // transpose
    auto dstStrides = computeShuffledStrides(sizes, dimOrder);
    
    auto output = shuffleDims(input, sizes, dstStrides, srcStrides);
    
    // After transpose: 3x2 matrix [[0,3], [1,4], [2,5]]
    std::vector<int> expected = {0, 3, 1, 4, 2, 5};
    EXPECT_EQ(output, expected);
}

TEST(PreSwizzleTest, ShuffleDims3D)
{
    // 2x2x2 cube
    std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<size_t> sizes = {2, 2, 2};
    
    auto srcStrides = computeStrides(sizes);
    std::vector<size_t> dimOrder = {2, 1, 0};  // reverse dimensions
    auto dstStrides = computeShuffledStrides(sizes, dimOrder);
    
    auto output = shuffleDims(input, sizes, dstStrides, srcStrides);
    
    // Verify the shuffle was performed
    ASSERT_EQ(output.size(), input.size());
    // Element at position [0,0,0] should stay at [0,0,0]
    EXPECT_EQ(output[0], 0);
    // Element at position [1,1,1] should stay at [1,1,1]
    EXPECT_EQ(output[7], 7);
}

TEST(PreSwizzleTest, ShuffleDimsSizeMismatch)
{
    std::vector<int> input = {0, 1, 2, 3};
    std::vector<size_t> sizes = {2, 3};  // 6 elements expected
    auto srcStrides = computeStrides(sizes);
    auto dstStrides = computeStrides(sizes);
    
    EXPECT_THROW(
        shuffleDims(input, sizes, dstStrides, srcStrides),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, ShuffleDimsTooFewDimensions)
{
    std::vector<int> input = {0, 1};
    std::vector<size_t> sizes = {2};
    auto srcStrides = computeStrides(sizes);
    auto dstStrides = computeStrides(sizes);
    
    EXPECT_THROW(
        shuffleDims(input, sizes, dstStrides, srcStrides),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, ShuffleDimsDimensionMismatch)
{
    std::vector<int> input = {0, 1, 2, 3};
    std::vector<size_t> sizes = {2, 2};
    std::vector<size_t> srcStrides = {1, 2};
    std::vector<size_t> dstStrides = {1, 2, 4};  // wrong size
    
    EXPECT_THROW(
        shuffleDims(input, sizes, dstStrides, srcStrides),
        std::runtime_error
    );
}

// ============================================================================
// Tests for preSwizzleScale()
// ============================================================================

TEST(PreSwizzleTest, PreSwizzleScaleBasic)
{
    // Create a simple scale tensor
    size_t scaleRows = 128;  // K / blockSize
    size_t scaleCols = 64;   // M or N
    std::vector<float> input(scaleRows * scaleCols);
    
    // Fill with sequential values for testing
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<size_t> tile = {32, 128, 4};  // {tileMN, tileK, subTileK}
    
    auto output = preSwizzleScale(input, scaleRows, scaleCols, tile);
    
    ASSERT_EQ(output.size(), input.size());
    // The output should be a permutation of the input
    std::vector<float> sortedOutput = output;
    std::vector<float> sortedInput = input;
    std::sort(sortedOutput.begin(), sortedOutput.end());
    std::sort(sortedInput.begin(), sortedInput.end());
    EXPECT_EQ(sortedOutput, sortedInput);
}

TEST(PreSwizzleTest, PreSwizzleScaleInvalidTileSize)
{
    std::vector<float> input(128 * 64);
    std::vector<size_t> tile = {32, 128};  // Only 2 elements, need 3
    
    EXPECT_THROW(
        preSwizzleScale(input, 128, 64, tile),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, PreSwizzleScaleInvalidTileMN)
{
    std::vector<float> input(128 * 64);
    std::vector<size_t> tile = {64, 128, 4};  // tileMN must be 32
    
    EXPECT_THROW(
        preSwizzleScale(input, 128, 64, tile),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, PreSwizzleScaleTileKNotMultipleOf4)
{
    std::vector<float> input(127 * 64);
    std::vector<size_t> tile = {32, 127, 4};  // tileK not multiple of 4
    
    EXPECT_THROW(
        preSwizzleScale(input, 127, 64, tile),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, PreSwizzleScaleSizeMismatch)
{
    std::vector<float> input(100);  // Wrong size
    std::vector<size_t> tile = {32, 128, 4};
    
    EXPECT_THROW(
        preSwizzleScale(input, 128, 64, tile),
        std::runtime_error
    );
}

// ============================================================================
// Tests for preSwizzle()
// ============================================================================

TEST(PreSwizzleTest, PreSwizzleOnlySwizzle64)
{
    size_t k = 256;
    size_t mn = 128;
    std::vector<float> input(k * mn);
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {64, 256, 4};  // tileMN=64, tileK=256, subTileK=4
    std::vector<size_t> preTileSize;  // empty
    
    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    
    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleOnlySwizzle32SubTile4)
{
    size_t k = 128;
    size_t mn = 64;
    std::vector<float> input(k * mn);
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};  // tileMN=32, tileK=128, subTileK=4
    std::vector<size_t> preTileSize;  // empty
    
    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    
    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleOnlySwizzle32SubTile2)
{
    size_t k = 128;
    size_t mn = 64;
    std::vector<float> input(k * mn);
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 2};  // tileMN=32, tileK=128, subTileK=2
    std::vector<size_t> preTileSize;  // empty
    
    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    
    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleOnlyPreTile)
{
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

TEST(PreSwizzleTest, PreSwizzleBothSwizzleAndTile64)
{
    size_t k = 256;
    size_t mn = 128;
    std::vector<float> input(k * mn);
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {64, 256, 4};
    std::vector<size_t> preTileSize = {256, 64};
    
    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    
    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleBothSwizzleAndTile32SubTile4)
{
    size_t k = 512;
    size_t mn = 128;
    std::vector<float> input(k * mn);
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize = {128, 32};
    
    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    
    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleBothSwizzleAndTile32SubTile2)
{
    size_t k = 256;
    size_t mn = 64;
    std::vector<float> input(k * mn);
    std::iota(input.begin(), input.end(), 0.0f);
    
    std::vector<size_t> sizes = {k, mn};
    std::vector<size_t> preSwizzleSize = {32, 128, 2};
    std::vector<size_t> preTileSize = {128, 64};
    
    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    
    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleInvalidTileMN)
{
    std::vector<float> input(128 * 64);
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {48, 128, 4};  // tileMN not 32 or 64
    std::vector<size_t> preTileSize;
    
    EXPECT_THROW(
        preSwizzle(input, sizes, preSwizzleSize, preTileSize),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, PreSwizzleTileKNotMultipleOf4)
{
    std::vector<float> input(127 * 64);
    std::vector<size_t> sizes = {127, 64};
    std::vector<size_t> preSwizzleSize = {32, 127, 4};  // tileK not multiple of 4
    std::vector<size_t> preTileSize;
    
    EXPECT_THROW(
        preSwizzle(input, sizes, preSwizzleSize, preTileSize),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, PreSwizzleSizeMismatch)
{
    std::vector<float> input(100);  // Wrong size
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;
    
    EXPECT_THROW(
        preSwizzle(input, sizes, preSwizzleSize, preTileSize),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, PreSwizzleBatchDimensionNotSupported)
{
    std::vector<float> input(2 * 128 * 64);
    std::vector<size_t> sizes = {2, 128, 64};  // 3D tensor (batch dimension)
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;
    
    EXPECT_THROW(
        preSwizzle(input, sizes, preSwizzleSize, preTileSize),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, PreSwizzleInvalidPreSwizzleSize)
{
    std::vector<float> input(128 * 64);
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128};  // Only 2 elements, need 3
    std::vector<size_t> preTileSize;
    
    EXPECT_THROW(
        preSwizzle(input, sizes, preSwizzleSize, preTileSize),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, PreSwizzlePtTileSizeKZero)
{
    std::vector<float> input(128 * 64);
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 256, 4};  // tileK > sizes[0]
    std::vector<size_t> preTileSize = {128, 64};
    
    EXPECT_THROW(
        preSwizzle(input, sizes, preSwizzleSize, preTileSize),
        std::runtime_error
    );
}

TEST(PreSwizzleTest, PreSwizzlePtTileSizeMNZero)
{
    std::vector<float> input(128 * 64);
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {128, 128, 4};  // tileMN > sizes[1]
    std::vector<size_t> preTileSize = {128, 64};
    
    EXPECT_THROW(
        preSwizzle(input, sizes, preSwizzleSize, preTileSize),
        std::runtime_error
    );
}

// ============================================================================
// Edge Cases and Integration Tests
// ============================================================================

TEST(PreSwizzleTest, PreSwizzleScaleSmallTensor)
{
    size_t scaleRows = 32;  // minimum for tileK=128 would be larger, so this tests smaller case
    size_t scaleCols = 32;
    std::vector<float> input(scaleRows * scaleCols);
    std::iota(input.begin(), input.end(), 1.0f);
    
    std::vector<size_t> tile = {32, 32, 4};
    
    // This might fail validation, but let's see what happens
    // Actually tileK must be multiple of 4, and we have 32 which is valid
    auto output = preSwizzleScale(input, scaleRows, scaleCols, tile);
    
    EXPECT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleIdentityOnSmallData)
{
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

TEST(PreSwizzleTest, PreSwizzleDoubleType)
{
    // Test that templates work with different types
    std::vector<double> input(128 * 64);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<double>(i) * 0.1;
    
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;
    
    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    
    ASSERT_EQ(output.size(), input.size());
}

TEST(PreSwizzleTest, PreSwizzleIntegerType)
{
    // Test that templates work with integer types
    std::vector<int> input(128 * 64);
    std::iota(input.begin(), input.end(), 0);
    
    std::vector<size_t> sizes = {128, 64};
    std::vector<size_t> preSwizzleSize = {32, 128, 4};
    std::vector<size_t> preTileSize;
    
    auto output = preSwizzle(input, sizes, preSwizzleSize, preTileSize);
    
    ASSERT_EQ(output.size(), input.size());
}

