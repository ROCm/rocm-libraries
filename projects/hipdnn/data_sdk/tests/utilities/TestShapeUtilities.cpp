// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

using namespace hipdnn_data_sdk::utilities;

struct ExtractStrideOrderTestCase
{
    std::vector<int64_t> strides;
    std::vector<int64_t> order;
    bool log;
    std::vector<int64_t> dim;
};

// Outside anonymous namespace so ADL finds it for gtest printing
// NOLINTNEXTLINE(misc-use-internal-linkage)
std::ostream& operator<<(std::ostream& os, const ExtractStrideOrderTestCase& tc)
{
    os << "ExtractStrideOrderTestCase(";
    os << " strides: ";
    os << vecToString(tc.strides);
    os << " order: ";
    os << vecToString(tc.order);
    os << " log: " << tc.log;
    os << " )";
    return os;
}

using TestShapeUtilsExtractStrideOrder = ::testing::TestWithParam<ExtractStrideOrderTestCase>;

TEST(TestShapeUtils, GenerateStridesNhwcValid)
{
    const std::vector<int64_t> dim = {1, 2, 3, 4};
    const std::vector<int64_t> strideOrder = {3, 0, 2, 1}; // NHWC
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 1, 8, 2}));
}

TEST(TestShapeUtils, GenerateStridesNdhwcValid)
{
    const std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    const std::vector<int64_t> strideOrder = {4, 0, 3, 2, 1}; // NDHWC
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 1, 40, 10, 2}));
}

TEST(TestShapeUtils, GenerateStridesNchwValid)
{
    const std::vector<int64_t> dim = {1, 2, 3, 4};
    const std::vector<int64_t> strideOrder = {3, 2, 1, 0}; // NCHW
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 12, 4, 1}));
}

TEST(TestShapeUtils, GenerateStridesNcdhwValid)
{
    const std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    const std::vector<int64_t> strideOrder = {4, 3, 2, 1, 0}; // NCDHW
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 60, 20, 5, 1}));
}

TEST(TestShapeUtils, GenerateStridesBhsdValid)
{
    // BHSD: dims [B=2, H=4, S=128, D=64], row-major (same stride order as NCHW)
    const std::vector<int64_t> dim = {2, 4, 128, 64};
    auto strides = generateStrides(dim, TensorLayout::BHSD.strideOrder);

    // D contiguous, then S, then H, then B
    EXPECT_EQ(strides, (std::vector<int64_t>{32768, 8192, 64, 1}));

    // BHSD should produce the same strides as NCHW
    auto nchwStrides = generateStrides(dim, TensorLayout::NCHW.strideOrder);
    EXPECT_EQ(strides, nchwStrides);
}

TEST(TestShapeUtils, GenerateStridesBshdValid)
{
    // BSHD: dims [B=2, H=4, S=128, D=64], sequence-major
    const std::vector<int64_t> dim = {2, 4, 128, 64};
    auto strides = generateStrides(dim, TensorLayout::BSHD.strideOrder);

    // D contiguous, then H, then S, then B
    EXPECT_EQ(strides, (std::vector<int64_t>{32768, 64, 256, 1}));

    // BSHD should NOT produce the same strides as NHWC
    auto nhwcStrides = generateStrides(dim, TensorLayout::NHWC.strideOrder);
    EXPECT_NE(strides, nhwcStrides);
}

TEST(TestShapeUtils, GenerateStridesSingleDimension)
{
    const std::vector<int64_t> dim = {5};
    const std::vector<int64_t> strideOrder = {0};
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{1}));
}

TEST(TestShapeUtils, StrideOrderNhwcFiveDimensions)
{
    const size_t numDims = 5;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{4, 0, 3, 2, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcFourDimensions)
{
    const size_t numDims = 4;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{3, 0, 2, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcThreeDimensions)
{
    const size_t numDims = 3;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{2, 0, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcWithTwoDimensions)
{
    const size_t numDims = 2;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{1, 0}));
}

TEST(TestShapeUtils, StrideOrderNhwcWithSingleDimension)
{
    const size_t numDims = 1;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{0}));
}

TEST(TestShapeUtils, GenerateStridesEmptyDimensions)
{
    const std::vector<int64_t> dim = {};
    const std::vector<int64_t> strideOrder = {};
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}

TEST(TestShapeUtils, BroadcastCompatibleExactMatch)
{
    const std::vector<int64_t> inputDims = {2, 3, 4};
    const std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleInputNon1OutputIs1Invalid)
{
    // Input has non-1 value where output has 1 - invalid
    const std::vector<int64_t> inputDims = {2, 3, 4};
    const std::vector<int64_t> outputDims = {2, 1, 4};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleInputIs1OutputNon1Valid)
{
    // Input has 1 where output has non-1 - valid (broadcast)
    const std::vector<int64_t> inputDims = {2, 1, 4};
    const std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleFewerInputDims)
{
    // Input has fewer dimensions - valid (implicit leading 1s)
    const std::vector<int64_t> inputDims = {3, 4};
    const std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleMoreInputDimsInvalid)
{
    // Input has more dimensions than output - invalid
    const std::vector<int64_t> inputDims = {2, 3, 4, 5};
    const std::vector<int64_t> outputDims = {3, 4, 5};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleMismatchedDimsInvalid)
{
    // Input and output don't match (3 vs 5) - invalid
    const std::vector<int64_t> inputDims = {2, 3, 4};
    const std::vector<int64_t> outputDims = {2, 5, 4};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleEmptyInputValid)
{
    // Empty input is broadcastable to any output
    const std::vector<int64_t> inputDims = {};
    const std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleEmptyOutputWithNonEmptyInputInvalid)
{
    // Non-empty input cannot broadcast to empty output
    const std::vector<int64_t> inputDims = {2, 3};
    const std::vector<int64_t> outputDims = {};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleBothEmpty)
{
    // Both empty is valid (exact match)
    const std::vector<int64_t> inputDims = {};
    const std::vector<int64_t> outputDims = {};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleScalarToTensor)
{
    // Scalar [1] broadcasts to any shape
    const std::vector<int64_t> inputDims = {1};
    const std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleVectorToMatrix)
{
    // Vector [4] broadcasts to matrix [3, 4]
    const std::vector<int64_t> inputDims = {4};
    const std::vector<int64_t> outputDims = {3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleVectorMismatchInvalid)
{
    // Vector [5] cannot broadcast to matrix [3, 4]
    const std::vector<int64_t> inputDims = {5};
    const std::vector<int64_t> outputDims = {3, 4};

    EXPECT_FALSE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleComplexBroadcast)
{
    // Complex broadcast: [1, 3, 1] -> [2, 3, 4]
    const std::vector<int64_t> inputDims = {1, 3, 1};
    const std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatiblePartialMatchWithOnes)
{
    // [2, 1, 4] -> [2, 3, 4]
    const std::vector<int64_t> inputDims = {2, 1, 4};
    const std::vector<int64_t> outputDims = {2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleHigherDimensionalBroadcast)
{
    // 5D broadcast: [1, 1, 3, 1, 5] -> [2, 4, 3, 6, 5]
    const std::vector<int64_t> inputDims = {1, 1, 3, 1, 5};
    const std::vector<int64_t> outputDims = {2, 4, 3, 6, 5};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleLeadingOnesImplicit)
{
    // [3, 4] broadcasts to [1, 2, 3, 4] (implicit leading 1s)
    const std::vector<int64_t> inputDims = {3, 4};
    const std::vector<int64_t> outputDims = {1, 2, 3, 4};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleSingleDimToMultiDim)
{
    // Single dimension [5] to multi-dimensional [2, 3, 5]
    const std::vector<int64_t> inputDims = {5};
    const std::vector<int64_t> outputDims = {2, 3, 5};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleAllOnesInput)
{
    // All 1s input broadcasts to any matching rank output
    const std::vector<int64_t> inputDims = {1, 1, 1};
    const std::vector<int64_t> outputDims = {5, 7, 9};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, BroadcastCompatibleMixedOnesAndMatches)
{
    // Mix of exact matches and 1s: [2, 1, 4, 1] -> [2, 3, 4, 5]
    const std::vector<int64_t> inputDims = {2, 1, 4, 1};
    const std::vector<int64_t> outputDims = {2, 3, 4, 5};

    EXPECT_TRUE(areDimensionsBroadcastCompatible(inputDims, outputDims));
}

TEST(TestShapeUtils, GenerateStridesMoreDimsThanStridesThrows)
{
    const std::vector<int64_t> dim = {1, 2, 3, 4};
    const std::vector<int64_t> strideOrder = {2, 1, 0};

    EXPECT_THROW(generateStrides(dim, strideOrder), std::invalid_argument);
}

TEST(TestShapeUtils, GenerateStridesLessDimsThanStridesValid)
{
    const std::vector<int64_t> dim = {1, 2, 3};
    const std::vector<int64_t> strideOrder = {3, 2, 1, 0};

    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{6, 3, 1}));
}

TEST(TestShapeUtils, GenerateStridesFewerDimsNhwcOrdering)
{
    const std::vector<int64_t> dim = {2, 3, 4};
    const std::vector<int64_t> strideOrder = {2, 0, 1, 3};

    // Should use first 3 elements: {2, 0, 1} for NHW ordering
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{12, 1, 3}));
}

TEST(TestShapeUtils, GenerateStridesEmptyDimsWithNonEmptyStridesValid)
{
    const std::vector<int64_t> dim = {};
    const std::vector<int64_t> strideOrder = {3, 2, 1, 0};

    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesEmpty)
{
    const std::vector<int64_t> dims = {};
    auto strides = generateStrides(dims);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides1D)
{
    const std::vector<int64_t> dims = {10};
    auto strides = generateStrides(dims);

    EXPECT_EQ(strides, (std::vector<int64_t>{1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides2D)
{
    const std::vector<int64_t> dims = {5, 8};
    auto strides = generateStrides(dims);

    // Row-major: {8, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{8, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides3D)
{
    const std::vector<int64_t> dims = {4, 5, 6};
    auto strides = generateStrides(dims);

    // Row-major: {5*6, 6, 1} = {30, 6, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{30, 6, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides4D)
{
    const std::vector<int64_t> dims = {2, 3, 4, 5};
    auto strides = generateStrides(dims);

    // Row-major: {3*4*5, 4*5, 5, 1} = {60, 20, 5, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{60, 20, 5, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides5D)
{
    const std::vector<int64_t> dims = {2, 3, 4, 5, 6};
    auto strides = generateStrides(dims);

    // Row-major: {3*4*5*6, 4*5*6, 5*6, 6, 1} = {360, 120, 30, 6, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{360, 120, 30, 6, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides6D)
{
    const std::vector<int64_t> dims = {1, 2, 3, 4, 5, 6};
    auto strides = generateStrides(dims);

    // Row-major: {2*3*4*5*6, 3*4*5*6, 4*5*6, 5*6, 6, 1} = {720, 360, 120, 30, 6, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{720, 360, 120, 30, 6, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStrides7D)
{
    const std::vector<int64_t> dims = {2, 2, 2, 2, 2, 2, 2};
    auto strides = generateStrides(dims);

    // Row-major: {64, 32, 16, 8, 4, 2, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{64, 32, 16, 8, 4, 2, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesWithOnes)
{
    const std::vector<int64_t> dims = {1, 1, 5, 1, 3};
    auto strides = generateStrides(dims);

    // Row-major: {1*5*1*3, 5*1*3, 1*3, 3, 1} = {15, 15, 3, 3, 1}
    EXPECT_EQ(strides, (std::vector<int64_t>{15, 15, 3, 3, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesLargeDims)
{
    const std::vector<int64_t> dims = {100, 200};
    auto strides = generateStrides(dims);

    EXPECT_EQ(strides, (std::vector<int64_t>{200, 1}));
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesMatchesNchwFor4D)
{
    const std::vector<int64_t> dims = {2, 3, 4, 5};
    auto packedStrides = generateStrides(dims);
    auto nchwStrides = generateStrides(dims, {3, 2, 1, 0}); // NCHW stride order

    EXPECT_EQ(packedStrides, nchwStrides);
}

TEST(TestShapeUtils, GenerateDefaultPackedStridesMatchesNcdhwFor5D)
{
    const std::vector<int64_t> dims = {2, 3, 4, 5, 6};
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
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    auto strides = generateStrides(dims);

    EXPECT_EQ(strides, (std::vector<int64_t>{1, 1, 1, 1}));
}

TEST_P(TestShapeUtilsExtractStrideOrder, VerifyExtractedStrideOrder)
{
    auto& testCase = GetParam();

    // Create RAII log recorder - captures all logs for this test
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    // Sanity-check test data
    EXPECT_EQ(generateStrides(testCase.dim, testCase.order), testCase.strides);

    const std::vector<int64_t> deducedStrideOrder = extractStrideOrder(testCase.strides);
    EXPECT_EQ(deducedStrideOrder, testCase.order);

    if(testCase.log)
    {
        // Verify we got exactly one warning log
        ASSERT_EQ(recorder.getRecordedLogCount(), 1) << "Expected 1 log, but captured:\n"
                                                     << recorder.getRecordedLogsAsString();

        // Build expected warning message
        const std::string expectedLogSuffix
            = "extractStrideOrder(): Stride lengths " + vecToString(testCase.strides)
              + " are not unique, the deduced stride order " + vecToString(deducedStrideOrder)
              + " may not be correct.";

        // Verify log contains expected message at WARN level
        EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, expectedLogSuffix))
            << "Expected log containing: \"" << expectedLogSuffix << "\"\n"
            << "But captured:\n"
            << recorder.getRecordedLogsAsString();
    }
    else
    {
        // Verify no logs were emitted
        EXPECT_EQ(recorder.getRecordedLogCount(), 0) << "Expected no logs, but captured:\n"
                                                     << recorder.getRecordedLogsAsString();
    }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestShapeUtilsExtractStrideOrder,
    ::testing::Values(
        // NHWC tests
        ExtractStrideOrderTestCase{{24, 1, 8, 2}, {3, 0, 2, 1}, false, {10, 2, 3, 4}},
        ExtractStrideOrderTestCase{{2, 1, 2, 1}, {3, 0, 2, 1}, true, {10, 1, 1, 2}},
        ExtractStrideOrderTestCase{{2, 1, 2, 2}, {3, 0, 2, 1}, true, {10, 2, 1, 1}},

        // NDHWC tests
        ExtractStrideOrderTestCase{{120, 1, 40, 10, 2}, {4, 0, 3, 2, 1}, false, {10, 2, 3, 4, 5}},
        ExtractStrideOrderTestCase{{4, 1, 4, 4, 4}, {4, 0, 3, 2, 1}, true, {10, 4, 1, 1, 1}},
        ExtractStrideOrderTestCase{{4, 1, 4, 1, 1}, {4, 0, 3, 2, 1}, true, {10, 1, 1, 4, 1}},
        ExtractStrideOrderTestCase{{2, 1, 2, 2, 1}, {4, 0, 3, 2, 1}, true, {10, 1, 1, 1, 2}},

        // NCHW tests
        ExtractStrideOrderTestCase{{24, 12, 4, 1}, {3, 2, 1, 0}, false, {10, 2, 3, 4}},
        ExtractStrideOrderTestCase{{4, 4, 1, 1}, {3, 2, 1, 0}, true, {10, 1, 4, 1}},
        ExtractStrideOrderTestCase{{4, 4, 4, 1}, {3, 2, 1, 0}, true, {10, 1, 1, 4}},

        // NCDHW tests
        ExtractStrideOrderTestCase{{120, 60, 20, 5, 1}, {4, 3, 2, 1, 0}, false, {1, 2, 3, 4, 5}},
        ExtractStrideOrderTestCase{{4, 4, 4, 1, 1}, {4, 3, 2, 1, 0}, true, {10, 1, 1, 4, 1}},
        ExtractStrideOrderTestCase{{4, 4, 1, 1, 1}, {4, 3, 2, 1, 0}, true, {10, 1, 4, 1, 1}},
        ExtractStrideOrderTestCase{{4, 4, 4, 4, 1}, {4, 3, 2, 1, 0}, true, {10, 1, 1, 1, 4}},

        // All ones --> NC...W
        ExtractStrideOrderTestCase{{1, 1, 1, 1}, {3, 2, 1, 0}, true, {1, 1, 1, 1}},
        ExtractStrideOrderTestCase{{1, 1, 1, 1, 1}, {4, 3, 2, 1, 0}, true, {1, 1, 1, 1, 1}},

        // Uncommon

        ExtractStrideOrderTestCase{{}, {}, false, {}},

        ExtractStrideOrderTestCase{{1}, {0}, false, {1}},

        ExtractStrideOrderTestCase{{1, 1}, {1, 0}, true, {1, 1}},
        ExtractStrideOrderTestCase{{2, 1}, {1, 0}, false, {1, 2}},
        ExtractStrideOrderTestCase{{1, 2}, {0, 1}, false, {2, 1}},

        ExtractStrideOrderTestCase{{1, 1, 1}, {2, 1, 0}, true, {1, 1, 1}},
        ExtractStrideOrderTestCase{{4, 2, 1}, {2, 1, 0}, false, {1, 2, 2}},
        ExtractStrideOrderTestCase{{4, 1, 2}, {2, 0, 1}, false, {1, 2, 2}},
        ExtractStrideOrderTestCase{{1, 2, 4}, {0, 1, 2}, false, {2, 2, 1}},
        ExtractStrideOrderTestCase{{1, 1, 2}, {1, 0, 2}, true, {2, 1, 1}},
        ExtractStrideOrderTestCase{{2, 1, 2}, {2, 0, 1}, true, {1, 2, 1}}, // NWC
        ExtractStrideOrderTestCase{{2, 1, 1}, {2, 1, 0}, true, {1, 2, 1}}, // NCW

        ExtractStrideOrderTestCase{{1, 2, 4, 8}, {0, 1, 2, 3}, false, {2, 2, 2, 1}},
        ExtractStrideOrderTestCase{{1, 4, 2, 2}, {0, 3, 2, 1}, true, {2, 1, 2, 1}},
        ExtractStrideOrderTestCase{{1, 1, 4, 8}, {1, 0, 2, 3}, true, {4, 1, 2, 1}},
        ExtractStrideOrderTestCase{{1, 4, 8, 1}, {1, 2, 3, 0}, true, {4, 2, 1, 1}},

        ExtractStrideOrderTestCase{{1, 4, 2, 16, 8}, {0, 2, 1, 4, 3}, false, {2, 2, 2, 1, 2}},
        ExtractStrideOrderTestCase{{1, 4, 2, 2, 2}, {0, 4, 3, 2, 1}, true, {2, 1, 2, 1, 1}},
        ExtractStrideOrderTestCase{{1, 1, 1, 4, 8}, {2, 1, 0, 3, 4}, true, {4, 1, 1, 2, 1}},
        ExtractStrideOrderTestCase{{1, 4, 8, 8, 1}, {1, 2, 4, 3, 0}, true, {4, 2, 1, 1, 1}},

        ExtractStrideOrderTestCase{
            {1, 4, 2, 32, 16, 8}, {0, 2, 1, 5, 4, 3}, false, {2, 2, 2, 1, 2, 2}},
        ExtractStrideOrderTestCase{
            {1, 4, 2, 2, 2, 2}, {0, 5, 4, 3, 2, 1}, true, {2, 1, 2, 1, 1, 1}},
        ExtractStrideOrderTestCase{
            {1, 1, 1, 4, 4, 8}, {2, 1, 0, 4, 3, 5}, true, {4, 1, 1, 2, 1, 1}},
        ExtractStrideOrderTestCase{
            {1, 4, 8, 8, 8, 1}, {1, 2, 5, 4, 3, 0}, true, {4, 2, 1, 1, 1, 1}}));

TEST(TestShapeUtils, GenerateStridesWithPackedAxisNchwAxis1)
{
    // NCHW input, make axis 1 (C) most packed
    auto strides
        = generateStridesWithPackedAxis({65536, 1024, 32, 1}, {2, 64, 32, 32}, {2, 64, 32, 32}, 1);

    // Sorted by stride ascending: [3,2,1,0], rotate axis=1 to front → [1,0,3,2]
    // strideOrder=[1,0,3,2]: dim1 gets stride 1
    EXPECT_EQ(strides, (std::vector<int64_t>{64, 1, 4096, 128}));
}

TEST(TestShapeUtils, GenerateStridesWithPackedAxisNhwcAxis3)
{
    // NHWC input, make axis 3 (W) most packed
    auto strides
        = generateStridesWithPackedAxis({65536, 1, 2048, 64}, {2, 64, 32, 32}, {2, 64, 32, 32}, 3);

    EXPECT_EQ(strides, (std::vector<int64_t>{1024, 2048, 32, 1}));
}

TEST(TestShapeUtils, GenerateStridesWithPackedAxisNoAxisPreservesOrder)
{
    // Without axis, stride ordering matches the reference layout
    auto strides
        = generateStridesWithPackedAxis({65536, 1024, 32, 1}, {2, 64, 32, 32}, {2, 64, 32, 32});

    EXPECT_EQ(strides, (std::vector<int64_t>{65536, 1024, 32, 1}));
}

TEST(TestShapeUtils, GenerateStridesWithPackedAxisDifferentTargetDims)
{
    // Target dims differ from reference (e.g., scale tensor with reduced axis dim)
    auto strides
        = generateStridesWithPackedAxis({65536, 1024, 32, 1}, {2, 64, 32, 32}, {2, 2, 32, 32}, 1);

    EXPECT_EQ(strides, (std::vector<int64_t>{2, 1, 128, 4}));
}

TEST(TestShapeUtils, GenerateStridesWithPackedAxisSingletonTiebreaker)
{
    // Singletons with equal strides should sort before non-singletons
    // dims {1, 4, 1, 4} with strides {4, 1, 4, 1}: dims 0 and 2 are singletons with stride 4
    auto strides = generateStridesWithPackedAxis({4, 1, 4, 1}, {1, 4, 1, 4}, {1, 4, 1, 4}, 1);

    // Sorted ascending: dim1(1), dim3(1), dim0(4,singleton), dim2(4,singleton)
    // Rotate axis=1 to front: [1, 3, 0, 2]
    // strideOrder: [2, 0, 3, 1] → dim1 gets stride 1
    EXPECT_EQ(strides[1], 1); // axis dim is most packed
}

TEST(TestShapeUtils, GenerateStridesWithPackedAxis3D)
{
    // 3D tensor, axis=0
    auto strides = generateStridesWithPackedAxis({12, 4, 1}, {3, 3, 4}, {3, 3, 4}, 0);

    // Sorted ascending: [2,1,0], rotate axis=0 to front: [0,2,1]
    // strideOrder: [0,2,1] → dim0 gets stride 1
    EXPECT_EQ(strides[0], 1);
    EXPECT_EQ(strides, (std::vector<int64_t>{1, 12, 3}));
}

TEST(TestShapeUtils, GenerateStridesWithPackedAxisLastAxis)
{
    // Packing the last axis (already most packed in NCHW) preserves original layout
    auto strides
        = generateStridesWithPackedAxis({65536, 1024, 32, 1}, {2, 64, 32, 32}, {2, 64, 32, 32}, 3);

    // Sorted ascending: [3,2,1,0], rotate axis=3 (at position 0) to front → no change
    EXPECT_EQ(strides, (std::vector<int64_t>{65536, 1024, 32, 1}));
}

TEST(TestShapeUtils, IsTensorPackedTrueForPackedTensor)
{
    const std::vector<int64_t> shape3D = {2, 3, 4};
    auto strides3D = generateStrides(shape3D);
    EXPECT_TRUE(isTensorPacked(shape3D, strides3D));

    const std::vector<int64_t> shape4D = {2, 3, 4, 5};
    auto stridesNchw = generateStrides(shape4D);
    EXPECT_TRUE(isTensorPacked(shape4D, stridesNchw));
    auto stridesNhwc = generateStrides(shape4D, {3, 0, 2, 1});
    EXPECT_TRUE(isTensorPacked(shape4D, stridesNhwc));

    const std::vector<int64_t> shape5D = {2, 3, 4, 5, 6};
    auto stridesNcdhw = generateStrides(shape5D);
    EXPECT_TRUE(isTensorPacked(shape5D, stridesNcdhw));
    auto stridesNdhwc = generateStrides(shape5D, {4, 0, 3, 2, 1});
    EXPECT_TRUE(isTensorPacked(shape5D, stridesNdhwc));
}

TEST(TestShapeUtils, IsTensorPackedFalseForNonPackedTensor)
{
    const std::vector<int64_t> shape3D = {2, 3, 4};
    const std::vector<int64_t> nonPackedStrides3D = {13, 4, 1};
    EXPECT_FALSE(isTensorPacked(shape3D, nonPackedStrides3D));

    const std::vector<int64_t> shape4D = {2, 3, 4, 5};
    const std::vector<int64_t> nonPackedStrides4D = {50, 20, 5, 1};
    EXPECT_FALSE(isTensorPacked(shape4D, nonPackedStrides4D));

    const std::vector<int64_t> shape5D = {2, 3, 4, 5, 6};
    const std::vector<int64_t> nonPackedStrides5D = {400, 120, 30, 6, 1};
    EXPECT_FALSE(isTensorPacked(shape5D, nonPackedStrides5D));
}

TEST(TestShapeUtils, IsTensorPackedEmptyTensor)
{
    // Empty tensors are considered packed
    const std::vector<int64_t> emptyDims = {};
    const std::vector<int64_t> emptyStrides = {};
    EXPECT_TRUE(isTensorPacked(emptyDims, emptyStrides));
}

TEST(TestShapeUtils, IsTensorPackedSingleDimension)
{
    // Single dimension tensors are packed if stride is 1
    const std::vector<int64_t> dims = {10};
    const std::vector<int64_t> strides = {1};
    EXPECT_TRUE(isTensorPacked(dims, strides));
}

TEST(TestShapeUtils, IsTensorPackedWithDimensionSizeOne)
{
    // Dimensions with size 1 - stride is irrelevant for that dimension
    const std::vector<int64_t> dims1 = {1, 3, 4};
    const std::vector<int64_t> strides1 = {12, 4, 1}; // Standard packed
    EXPECT_TRUE(isTensorPacked(dims1, strides1));

    const std::vector<int64_t> dims2 = {1, 3, 4};
    const std::vector<int64_t> strides2 = {100, 4, 1}; // Large stride for size-1 dim, still packed
    EXPECT_TRUE(isTensorPacked(dims2, strides2));

    const std::vector<int64_t> dims3 = {2, 1, 4};
    const std::vector<int64_t> strides3 = {4, 100, 1}; // Large stride for size-1 dim, still packed
    EXPECT_TRUE(isTensorPacked(dims3, strides3));
}

TEST(TestShapeUtils, IsTensorPackedRowMajorVsColumnMajor)
{
    // Both row-major and column-major 2x3 matrices should be packed
    const std::vector<int64_t> dims = {2, 3};

    // Row-major: strides = [3, 1]
    const std::vector<int64_t> rowMajorStrides = {3, 1};
    EXPECT_TRUE(isTensorPacked(dims, rowMajorStrides));

    // Column-major: strides = [1, 2]
    const std::vector<int64_t> colMajorStrides = {1, 2};
    EXPECT_TRUE(isTensorPacked(dims, colMajorStrides));
}

TEST(TestShapeUtils, IsTensorPackedWithNegativeStrides)
{
    // Negative strides are valid for reversed tensor dimensions
    // A 2x3 tensor with reversed first dimension: strides = [-3, 1]
    // The math still works: count = 6, space = (2-1)*(-3) + (3-1)*1 = -3 + 2 = -1
    // So count != space + 1 (6 != 0), which means it's not packed
    const std::vector<int64_t> dims = {2, 3};
    const std::vector<int64_t> negativeStrides = {-3, 1};
    // With negative strides, the tensor is typically not packed in the standard sense
    EXPECT_FALSE(isTensorPacked(dims, negativeStrides));
}

TEST(TestShapeUtils, IsTensorPackedMismatchedSizes)
{
    // Mismatched dims and strides should throw
    const std::vector<int64_t> dims = {2, 3, 4};
    const std::vector<int64_t> strides = {12, 4}; // Only 2 strides for 3 dims
    EXPECT_THROW(isTensorPacked(dims, strides), std::invalid_argument);
}

TEST(TestShapeUtils, IsTensorPackedOverlappingStrides)
{
    // Strides that cause overlap (e.g., both dimensions have stride 1)
    const std::vector<int64_t> dims = {2, 3};
    const std::vector<int64_t> overlappingStrides = {1, 1};
    // count = 6, space = (2-1)*1 + (3-1)*1 = 1 + 2 = 3, count != space+1 (6 != 4)
    EXPECT_FALSE(isTensorPacked(dims, overlappingStrides));
}

TEST(TestShapeUtils, IsTensorPackedAllOnesShape)
{
    // Shape with all 1s should be packed
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const std::vector<int64_t> strides = {1, 1, 1, 1};
    EXPECT_TRUE(isTensorPacked(dims, strides));
}

TEST(TestShapeUtils, IsTensorPackedLargeTensor)
{
    // Test with a larger tensor
    const std::vector<int64_t> dims = {4, 16, 32, 64};
    auto strides = generateStrides(dims);
    EXPECT_TRUE(isTensorPacked(dims, strides));
}

TEST(TestShapeUtils, IsTensorPackedCustomPackedLayout)
{
    // Custom packed layout (not standard NCHW/NHWC but still contiguous)
    const std::vector<int64_t> dims = {2, 3, 4};
    // Order: [2, 0, 1] means dimension 2 is innermost, then 0, then 1
    const std::vector<int64_t> customOrder = {2, 0, 1};
    auto customStrides = generateStrides(dims, customOrder);
    EXPECT_TRUE(isTensorPacked(dims, customStrides));
}

TEST(TestShapeUtils, GetDerivedShape5DValid)
{
    const std::vector<int64_t> shape = {2, 4, 8, 16, 32};
    auto derivedShape = getDerivedShape(shape);

    EXPECT_EQ(derivedShape, (std::vector<int64_t>{1, 4, 1, 1, 1}));
}

TEST(TestShapeUtils, GetDerivedShapeThrowsForSingleDimension)
{
    const std::vector<int64_t> shape = {10};
    EXPECT_THROW(getDerivedShape(shape), std::runtime_error);
}

TEST(TestShapeUtils, CalculateGroupCountReturnsOneForStandardConvolution)
{
    const std::vector<int64_t> inputDims = {1, 16, 32, 32};
    const std::vector<int64_t> weightDims = {32, 16, 3, 3};

    auto groupCount = calculateGroupCount(inputDims, weightDims);
    EXPECT_EQ(groupCount, 1);
}

TEST(TestShapeUtils, CalculateGroupCountReturnsCorrectValueForGroupedConvolution)
{
    const std::vector<int64_t> inputDims = {1, 16, 32, 32};
    const std::vector<int64_t> weightDims = {32, 8, 3, 3};

    auto groupCount = calculateGroupCount(inputDims, weightDims);
    EXPECT_EQ(groupCount, 2);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForZeroWeightChannels)
{
    const std::vector<int64_t> inputDims = {1, 16, 32, 32};
    const std::vector<int64_t> weightDims = {32, 0, 3, 3};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForZeroInputChannels)
{
    const std::vector<int64_t> inputDims = {1, 0, 32, 32};
    const std::vector<int64_t> weightDims = {32, 8, 3, 3};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForInputDimsLessThanTwo)
{
    const std::vector<int64_t> inputDims = {16};
    const std::vector<int64_t> weightDims = {32, 16, 3, 3};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForWeightDimsLessThanTwo)
{
    const std::vector<int64_t> inputDims = {1, 16, 32, 32};
    const std::vector<int64_t> weightDims = {32};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}

TEST(TestShapeUtils, CalculateGroupCountThrowsForNonDivisibleChannels)
{
    const std::vector<int64_t> inputDims = {1, 16, 32, 32};
    const std::vector<int64_t> weightDims = {32, 7, 3, 3};

    EXPECT_THROW(calculateGroupCount(inputDims, weightDims), std::invalid_argument);
}

TEST(TestShapeUtils, IsLayoutAgnosticTrueForAllOnes)
{
    // Degenerate tensors (all dims=1) are layout-agnostic
    EXPECT_TRUE(isLayoutAgnostic({1, 1, 1, 1}));
    EXPECT_TRUE(isLayoutAgnostic({1, 1, 1, 1, 1}));
    EXPECT_TRUE(isLayoutAgnostic({1}));
}

TEST(TestShapeUtils, IsLayoutAgnosticTrueForSingleNonTrivialDim)
{
    // Channel-only/derived tensors (one non-trivial dimension)
    EXPECT_TRUE(isLayoutAgnostic({1, 64, 1, 1}));
    EXPECT_TRUE(isLayoutAgnostic({1, 128, 1, 1, 1}));
    EXPECT_TRUE(isLayoutAgnostic({1, 1, 1, 32}));
    EXPECT_TRUE(isLayoutAgnostic({10, 1, 1, 1}));
}

TEST(TestShapeUtils, IsLayoutAgnosticTrueForEmpty)
{
    EXPECT_TRUE(isLayoutAgnostic({}));
}

TEST(TestShapeUtils, IsLayoutAgnosticFalseForMultipleNonTrivialDims)
{
    // Standard input tensors with multiple non-trivial dimensions
    EXPECT_FALSE(isLayoutAgnostic({2, 3, 4, 5}));
    EXPECT_FALSE(isLayoutAgnostic({1, 64, 32, 32}));
    EXPECT_FALSE(isLayoutAgnostic({2, 128, 1, 1}));
    EXPECT_FALSE(isLayoutAgnostic({1, 1, 2, 3}));
}

TEST(TestShapeUtils, IsLayoutAgnosticBoundaryCase)
{
    // Exactly two non-trivial dims should be false
    EXPECT_FALSE(isLayoutAgnostic({2, 2, 1, 1}));
    EXPECT_FALSE(isLayoutAgnostic({1, 2, 2, 1}));
}
