// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include "harness/golden/GoldenTensorComparator.hpp"

using namespace hipdnn_integration_tests::golden;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

std::unique_ptr<hipdnn_data_sdk::utilities::ITensor> makeTensor(const std::vector<int64_t>& dims,
                                                                const std::vector<float>& values)
{
    std::vector<int64_t> strides(dims.size());
    int64_t stride = 1;
    for(auto i = static_cast<int64_t>(dims.size()) - 1; i >= 0; --i)
    {
        strides[static_cast<size_t>(i)] = stride;
        stride *= dims[static_cast<size_t>(i)];
    }

    auto tensor = std::make_unique<hipdnn_data_sdk::utilities::Tensor<float>>(dims, strides);
    auto* data = static_cast<float*>(tensor->rawHostData());
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = values[i];
    }
    return tensor;
}

} // namespace

TEST(TestGoldenTensorComparator, IdenticalTensorsPass)
{
    auto expected = makeTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto actual = makeTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    auto result = compareTensors<float>(*expected, *actual, 1e-6f, 1e-6f);
    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.mismatchCount, 0u);
    EXPECT_EQ(result.totalElements, 6u);
}

TEST(TestGoldenTensorComparator, MismatchDetected)
{
    auto expected = makeTensor({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto actual = makeTensor({4}, {1.0f, 2.0f, 3.5f, 4.0f});

    auto result = compareTensors<float>(*expected, *actual, 0.01f, 0.01f);
    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.mismatchCount, 1u);
    EXPECT_EQ(result.worstFlatIndex, 2);
    EXPECT_NEAR(result.worstExpected, 3.0, 1e-5);
    EXPECT_NEAR(result.worstActual, 3.5, 1e-5);
}

TEST(TestGoldenTensorComparator, AbsToleranceAllowsSmallDiff)
{
    auto expected = makeTensor({3}, {1.0f, 2.0f, 3.0f});
    auto actual = makeTensor({3}, {1.001f, 2.001f, 3.001f});

    auto result = compareTensors<float>(*expected, *actual, 0.01f, 0.0f);
    EXPECT_TRUE(result.passed);
}

TEST(TestGoldenTensorComparator, RelToleranceAllowsProportionalDiff)
{
    auto expected = makeTensor({2}, {100.0f, 200.0f});
    auto actual = makeTensor({2}, {100.5f, 201.0f});

    auto result = compareTensors<float>(*expected, *actual, 0.0f, 0.01f);
    EXPECT_TRUE(result.passed);
}

TEST(TestGoldenTensorComparator, MultipleMismatchesReportsWorst)
{
    auto expected = makeTensor({3}, {1.0f, 2.0f, 3.0f});
    auto actual = makeTensor({3}, {1.1f, 5.0f, 3.2f});

    auto result = compareTensors<float>(*expected, *actual, 0.01f, 0.01f);
    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.mismatchCount, 3u);
    EXPECT_EQ(result.worstFlatIndex, 1);
    EXPECT_NEAR(result.maxAbsError, 3.0, 1e-5);
}

TEST(TestGoldenTensorComparator, NegativeZeroEqualsPositiveZero)
{
    auto expected = makeTensor({2}, {0.0f, -0.0f});
    auto actual = makeTensor({2}, {-0.0f, 0.0f});

    auto result = compareTensors<float>(*expected, *actual, 0.0f, 0.0f);
    EXPECT_TRUE(result.passed);
}

TEST(TestGoldenTensorComparator, FlatIndexToMultiDimCorrect)
{
    auto indices = flatIndexToMultiDim(5, {2, 3});
    ASSERT_EQ(indices.size(), 2u);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 2);
}

TEST(TestGoldenTensorComparator, FlatIndexToMultiDim4D)
{
    // shape [2, 3, 4, 5], flat index 47
    // 47 = 0*60 + 2*20 + 1*5 + 2  => (0, 2, 1, 2)
    auto indices = flatIndexToMultiDim(47, {2, 3, 4, 5});
    ASSERT_EQ(indices.size(), 4u);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 2);
    EXPECT_EQ(indices[2], 1);
    EXPECT_EQ(indices[3], 2);
}

TEST(TestGoldenTensorComparator, FormatComparisonFailureContainsAllFields)
{
    ComparisonResult result;
    result.passed = false;
    result.totalElements = 100;
    result.mismatchCount = 5;
    result.maxAbsError = 0.5;
    result.maxRelError = 0.1;
    result.worstFlatIndex = 42;
    result.worstExpected = 1.0;
    result.worstActual = 1.5;
    result.usedAtol = 0.01f;
    result.usedRtol = 0.01f;

    auto msg = formatComparisonFailure(
        "/path/to/bundle.json", 5, "output_tensor", {10, 10}, "fp32", result);

    EXPECT_NE(msg.find("bundle.json"), std::string::npos);
    EXPECT_NE(msg.find('5'), std::string::npos);
    EXPECT_NE(msg.find("output_tensor"), std::string::npos);
    EXPECT_NE(msg.find("[10, 10]"), std::string::npos);
    EXPECT_NE(msg.find("fp32"), std::string::npos);
    EXPECT_NE(msg.find("42"), std::string::npos);
    EXPECT_NE(msg.find("5 / 100"), std::string::npos);
}

// NOLINTEND(readability-identifier-naming)
