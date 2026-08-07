// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <sstream>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;

// =================================================================================================
// formatComparisonHeader
// =================================================================================================

TEST(TestFormatComparisonHeader, ContainsAllFields)
{
    const ComparisonContext ctx{
        "Bundle: /path/to/bundle", "output_tensor (UID 42, output)", "FLOAT", 1e-5f, 1e-4f};

    const Tensor<float> tensor({2, 3, 4});
    const std::string header = formatComparisonHeader(ctx, tensor);

    EXPECT_NE(header.find("Comparison FAILED"), std::string::npos);
    EXPECT_NE(header.find("Bundle: /path/to/bundle"), std::string::npos);
    EXPECT_NE(header.find("output_tensor (UID 42, output)"), std::string::npos);
    EXPECT_NE(header.find("FLOAT"), std::string::npos);
    EXPECT_NE(header.find("atol="), std::string::npos);
    EXPECT_NE(header.find("rtol="), std::string::npos);
}

TEST(TestFormatComparisonHeader, IncludesShape)
{
    const ComparisonContext ctx{"Test: MyTest.Case", "x", "HALF", 0.0f, 0.0f};

    const Tensor<float> tensor({8, 16});
    const std::string header = formatComparisonHeader(ctx, tensor);

    EXPECT_NE(header.find("Shape:"), std::string::npos);
    EXPECT_NE(header.find("8, 16"), std::string::npos);
}

// =================================================================================================
// appendComparisonDiffByDataType
// =================================================================================================

TEST(TestAppendComparisonDiffByDataType, FloatProducesSummary)
{
    Tensor<float> ref({4});
    Tensor<float> actual({4});

    for(int64_t i = 0; i < 4; ++i)
    {
        ref.setHostValue(0.0f, std::vector<int64_t>{i});
        actual.setHostValue(0.0f, std::vector<int64_t>{i});
    }
    ref.setHostValue(1.0f, std::vector<int64_t>{2});
    actual.setHostValue(2.0f, std::vector<int64_t>{2});

    std::ostringstream oss;
    appendComparisonDiffByDataType(oss, DT::FLOAT, "y", ref, actual, 0.0f, 0.0f);
    const std::string output = oss.str();

    EXPECT_NE(output.find("Total elements:"), std::string::npos);
    EXPECT_NE(output.find("Mismatched:"), std::string::npos);
}

TEST(TestAppendComparisonDiffByDataType, DoubleProducesSummary)
{
    Tensor<double> ref({3});
    Tensor<double> actual({3});

    for(int64_t i = 0; i < 3; ++i)
    {
        ref.setHostValue(1.0, std::vector<int64_t>{i});
        actual.setHostValue(1.0, std::vector<int64_t>{i});
    }
    actual.setHostValue(99.0, std::vector<int64_t>{0});

    std::ostringstream oss;
    appendComparisonDiffByDataType(oss, DT::DOUBLE, "z", ref, actual, 0.0f, 0.0f);

    EXPECT_NE(oss.str().find("Total elements:"), std::string::npos);
}

TEST(TestAppendComparisonDiffByDataType, UnsupportedTypeShowsMessage)
{
    Tensor<float> ref({2});
    Tensor<float> actual({2});

    std::ostringstream oss;
    appendComparisonDiffByDataType(oss, DT::INT8, "w", ref, actual, 0.0f, 0.0f);

    EXPECT_NE(oss.str().find("no element-wise diff available for data type: INT8"),
              std::string::npos);
}

TEST(TestAppendComparisonDiffByDataType, MatchingTensorsShowZeroMismatches)
{
    Tensor<float> ref({3});
    Tensor<float> actual({3});

    for(int64_t i = 0; i < 3; ++i)
    {
        ref.setHostValue(static_cast<float>(i), std::vector<int64_t>{i});
        actual.setHostValue(static_cast<float>(i), std::vector<int64_t>{i});
    }

    std::ostringstream oss;
    appendComparisonDiffByDataType(oss, DT::FLOAT, "clean", ref, actual, 1e-5f, 1e-5f);
    const std::string output = oss.str();

    EXPECT_NE(output.find("Mismatched:     0"), std::string::npos);
    EXPECT_EQ(output.find("Worst mismatches:"), std::string::npos);
}

// =================================================================================================
// appendComparisonDiff (direct template call)
// =================================================================================================

TEST(TestAppendComparisonDiff, ProducesDiffOutput)
{
    Tensor<float> ref({2, 2});
    Tensor<float> actual({2, 2});

    iterateAlongDimensions(ref.dims(), [&](const std::vector<int64_t>& idx) {
        ref.setHostValue(1.0f, idx);
        actual.setHostValue(1.0f, idx);
    });
    actual.setHostValue(5.0f, std::vector<int64_t>{0, 1});

    std::ostringstream oss;
    appendComparisonDiff<float>(oss, "t", ref, actual, 0.0f, 0.0f);
    const std::string output = oss.str();

    EXPECT_NE(output.find("Mismatched:     1"), std::string::npos);
    EXPECT_NE(output.find("Worst mismatches:"), std::string::npos);
}
