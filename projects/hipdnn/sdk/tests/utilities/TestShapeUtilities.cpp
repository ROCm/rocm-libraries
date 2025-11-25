// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

using namespace hipdnn_sdk::utilities;

TEST(TestShapeUtils, GenerateStridesNhwcValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> strideOrder = {3, 0, 2, 1}; // NHWC
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 1, 8, 2}));
}

TEST(TestShapeUtils, GenerateStridesNdhwcValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    std::vector<int64_t> strideOrder = {4, 0, 3, 2, 1}; // NDHWC
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 1, 40, 10, 2}));
}

TEST(TestShapeUtils, GenerateStridesNchwValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> strideOrder = {3, 2, 1, 0}; // NCHW
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 12, 4, 1}));
}

TEST(TestShapeUtils, GenerateStridesNcdhwValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    std::vector<int64_t> strideOrder = {4, 3, 2, 1, 0}; // NCDHW
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 60, 20, 5, 1}));
}

TEST(TestShapeUtils, GenerateStridesSingleDimension)
{
    std::vector<int64_t> dim = {5};
    std::vector<int64_t> strideOrder = {0};
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{1}));
}

TEST(TestShapeUtils, StrideOrderNhwcFiveDimensions)
{
    size_t numDims = 5;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{4, 0, 3, 2, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcFourDimensions)
{
    size_t numDims = 4;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{3, 0, 2, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcThreeDimensions)
{
    size_t numDims = 3;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{2, 0, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcWithTwoDimensions)
{
    size_t numDims = 2;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{1, 0}));
}

TEST(TestShapeUtils, StrideOrderNhwcWithSingleDimension)
{
    size_t numDims = 1;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{0}));
}

TEST(TestShapeUtils, GenerateStridesEmptyDimensions)
{
    std::vector<int64_t> dim = {};
    std::vector<int64_t> strideOrder = {};
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}

TEST(TestShapeUtils, BroadcastCompatibleExactMatch)
{
    std::vector<int64_t> inputDims = {2, 3, 4};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleInputNon1OutputIs1Invalid)
{
    // Input has non-1 value where output has 1 - invalid
    std::vector<int64_t> inputDims = {2, 3, 4};
    std::vector<int64_t> outputDims = {2, 1, 4};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleInputIs1OutputNon1Valid)
{
    // Input has 1 where output has non-1 - valid (broadcast)
    std::vector<int64_t> inputDims = {2, 1, 4};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleFewerInputDims)
{
    // Input has fewer dimensions - valid (implicit leading 1s)
    std::vector<int64_t> inputDims = {3, 4};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleMoreInputDimsInvalid)
{
    // Input has more dimensions than output - invalid
    std::vector<int64_t> inputDims = {2, 3, 4, 5};
    std::vector<int64_t> outputDims = {3, 4, 5};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleMismatchedDimsInvalid)
{
    // Input and output don't match (3 vs 5) - invalid
    std::vector<int64_t> inputDims = {2, 3, 4};
    std::vector<int64_t> outputDims = {2, 5, 4};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleEmptyInputValid)
{
    // Empty input is broadcastable to any output
    std::vector<int64_t> inputDims = {};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleEmptyOutputWithNonEmptyInputInvalid)
{
    // Non-empty input cannot broadcast to empty output
    std::vector<int64_t> inputDims = {2, 3};
    std::vector<int64_t> outputDims = {};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleBothEmpty)
{
    // Both empty is valid (exact match)
    std::vector<int64_t> inputDims = {};
    std::vector<int64_t> outputDims = {};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleScalarToTensor)
{
    // Scalar [1] broadcasts to any shape
    std::vector<int64_t> inputDims = {1};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleVectorToMatrix)
{
    // Vector [4] broadcasts to matrix [3, 4]
    std::vector<int64_t> inputDims = {4};
    std::vector<int64_t> outputDims = {3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleVectorMismatchInvalid)
{
    // Vector [5] cannot broadcast to matrix [3, 4]
    std::vector<int64_t> inputDims = {5};
    std::vector<int64_t> outputDims = {3, 4};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleComplexBroadcast)
{
    // Complex broadcast: [1, 3, 1] -> [2, 3, 4]
    std::vector<int64_t> inputDims = {1, 3, 1};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatiblePartialMatchWithOnes)
{
    // [2, 1, 4] -> [2, 3, 4]
    std::vector<int64_t> inputDims = {2, 1, 4};
    std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleHigherDimensionalBroadcast)
{
    // 5D broadcast: [1, 1, 3, 1, 5] -> [2, 4, 3, 6, 5]
    std::vector<int64_t> inputDims = {1, 1, 3, 1, 5};
    std::vector<int64_t> outputDims = {2, 4, 3, 6, 5};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleLeadingOnesImplicit)
{
    // [3, 4] broadcasts to [1, 2, 3, 4] (implicit leading 1s)
    std::vector<int64_t> inputDims = {3, 4};
    std::vector<int64_t> outputDims = {1, 2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleSingleDimToMultiDim)
{
    // Single dimension [5] to multi-dimensional [2, 3, 5]
    std::vector<int64_t> inputDims = {5};
    std::vector<int64_t> outputDims = {2, 3, 5};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleAllOnesInput)
{
    // All 1s input broadcasts to any matching rank output
    std::vector<int64_t> inputDims = {1, 1, 1};
    std::vector<int64_t> outputDims = {5, 7, 9};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleMixedOnesAndMatches)
{
    // Mix of exact matches and 1s: [2, 1, 4, 1] -> [2, 3, 4, 5]
    std::vector<int64_t> inputDims = {2, 1, 4, 1};
    std::vector<int64_t> outputDims = {2, 3, 4, 5};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, GenerateStridesMoreDimsThanStridesThrows)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> strideOrder = {2, 1, 0};

    EXPECT_THROW(generateStrides(dim, strideOrder), std::invalid_argument);
}

TEST(TestShapeUtils, GenerateStridesLessDimsThanStridesValid)
{
    std::vector<int64_t> dim = {1, 2, 3};
    std::vector<int64_t> strideOrder = {3, 2, 1, 0};

    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{6, 3, 1}));
}

TEST(TestShapeUtils, GenerateStridesFewerDimsNhwcOrdering)
{
    std::vector<int64_t> dim = {2, 3, 4};
    std::vector<int64_t> strideOrder = {2, 0, 1, 3};

    // Should use first 3 elements: {2, 0, 1} for NHW ordering
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{12, 1, 3}));
}

TEST(TestShapeUtils, GenerateStridesEmptyDimsWithNonEmptyStridesValid)
{
    std::vector<int64_t> dim = {};
    std::vector<int64_t> strideOrder = {3, 2, 1, 0};

    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesEmpty)
{
    std::vector<int64_t> dims = {};
    auto strides = generateStrides(dims);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides1D)
{
    std::vector<int64_t> dims = {10};
    auto strides = generateStrides(dims);

    EXPECT_EQ(strides, (std::vector<int64_t>{1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides2D)
{
    std::vector<int64_t> dims = {5, 8};
    auto strides = generateStrides(dims);

    // Row-major: {8, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{8, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides3D)
{
    std::vector<int64_t> dims = {4, 5, 6};
    auto strides = generateStrides(dims);

    // Row-major: {5*6, 6, 1} = {30, 6, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{30, 6, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides4D)
{
    std::vector<int64_t> dims = {2, 3, 4, 5};
    auto strides = generateStrides(dims);

    // Row-major: {3*4*5, 4*5, 5, 1} = {60, 20, 5, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{60, 20, 5, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides5D)
{
    std::vector<int64_t> dims = {2, 3, 4, 5, 6};
    auto strides = generateStrides(dims);

    // Row-major: {3*4*5*6, 4*5*6, 5*6, 6, 1} = {360, 120, 30, 6, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{360, 120, 30, 6, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides6D)
{
    std::vector<int64_t> dims = {1, 2, 3, 4, 5, 6};
    auto strides = generateStrides(dims);

    // Row-major: {2*3*4*5*6, 3*4*5*6, 4*5*6, 5*6, 6, 1} = {720, 360, 120, 30, 6, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{720, 360, 120, 30, 6, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides7D)
{
    std::vector<int64_t> dims = {2, 2, 2, 2, 2, 2, 2};
    auto strides = generateStrides(dims);

    // Row-major: {64, 32, 16, 8, 4, 2, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{64, 32, 16, 8, 4, 2, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesWithOnes)
{
    std::vector<int64_t> dims = {1, 1, 5, 1, 3};
    auto strides = generateStrides(dims);

    // Row-major: {1*5*1*3, 5*1*3, 1*3, 3, 1} = {15, 15, 3, 3, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{15, 15, 3, 3, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesLargeDims)
{
    std::vector<int64_t> dims = {100, 200};
    auto strides = generateStrides(dims);

    EXPECT_EQ(strides, (std::vector<int64_t>{200, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesMatchesNchwFor4D)
{
    std::vector<int64_t> dims = {2, 3, 4, 5};
    auto packedStrides = generateStrides(dims);
    auto nchwStrides = generateStrides(dims, {3, 2, 1, 0}); // NCHW stride order

    EXPECT_EQ(packedStrides, nchwStrides);
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesMatchesNcdhwFor5D)
{
    std::vector<int64_t> dims = {2, 3, 4, 5, 6};
    auto packedStrides = generateStrides(dims);
    auto ncdhwStrides = generateStrides(dims, {4, 3, 2, 1, 0}); // NCDHW stride order

    EXPECT_EQ(packedStrides, ncdhwStrides);
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesCalculation)
{
    // Test the actual calculation logic
    std::vector<int64_t> dims = {3, 4, 5};
    auto strides = generateStrides(dims);

    // Verify each stride is product of dimensions after it
    EXPECT_EQ(strides[0], dims[1] * dims[2]); // 4 * 5 = 20
    EXPECT_EQ(strides[1], dims[2]); // 5
    EXPECT_EQ(strides[2], 1); // 1 (always for last dimension)
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesAllOnes)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    auto strides = generateStrides(dims);

    EXPECT_EQ(strides, (std::vector<int64_t>{1, 1, 1, 1}));
}

TEST(TestShapeUtils, GetDerivedShape5DValid)
{
    std::vector<int64_t> shape = {2, 4, 8, 16, 32};
    auto derivedShape = getDerivedShape(shape);

    EXPECT_EQ(derivedShape, (std::vector<int64_t>{1, 4, 1, 1, 1}));
}

TEST(TestShapeUtils, GetDerivedShapeThrowsForSingleDimension)
{
    std::vector<int64_t> shape = {10};
    EXPECT_THROW(getDerivedShape(shape), std::runtime_error);
}

TEST(TestShapeUtils, CalculateGroupCountReturnsOneForStandardConvolution)
{
    std::vector<int64_t> inputDims = {1, 16, 32, 32};
    std::vector<int64_t> weightDims = {32, 16, 3, 3};

    auto groupCount = calculateGroupCount(inputDims, weightDims);
    EXPECT_EQ(groupCount, 1);
}

TEST(TestShapeUtils, CalculateGroupCountReturnsCorrectValueForGroupedConvolution)
{
    std::vector<int64_t> inputDims = {1, 16, 32, 32};
    std::vector<int64_t> weightDims = {32, 8, 3, 3};

    auto groupCount = calculateGroupCount(inputDims, weightDims);
    EXPECT_EQ(groupCount, 2);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForZeroWeightChannels)
{
    std::vector<int64_t> inputDims = {1, 16, 32, 32};
    std::vector<int64_t> weightDims = {32, 0, 3, 3};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForZeroInputChannels)
{
    std::vector<int64_t> inputDims = {1, 0, 32, 32};
    std::vector<int64_t> weightDims = {32, 8, 3, 3};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForInputDimsLessThanTwo)
{
    std::vector<int64_t> inputDims = {16};
    std::vector<int64_t> weightDims = {32, 16, 3, 3};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForWeightDimsLessThanTwo)
{
    std::vector<int64_t> inputDims = {1, 16, 32, 32};
    std::vector<int64_t> weightDims = {32};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForNonDivisibleChannels)
{
    std::vector<int64_t> inputDims = {1, 16, 32, 32};
    std::vector<int64_t> weightDims = {32, 7, 3, 3};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}
