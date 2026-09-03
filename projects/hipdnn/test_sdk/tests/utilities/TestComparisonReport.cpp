// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <regex>
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
// appendComparisonDiffByDataType — typed dispatch
// =================================================================================================

namespace
{

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr DT DATA_TYPE = DT::FLOAT;
};

template <>
struct DataTypeTraits<double>
{
    static constexpr DT DATA_TYPE = DT::DOUBLE;
};

template <>
struct DataTypeTraits<hipdnn_data_sdk::types::half>
{
    static constexpr DT DATA_TYPE = DT::HALF;
};

template <>
struct DataTypeTraits<hipdnn_data_sdk::types::bfloat16>
{
    static constexpr DT DATA_TYPE = DT::BFLOAT16;
};

} // namespace

using SupportedDataTypes = ::testing::
    Types<float, double, hipdnn_data_sdk::types::half, hipdnn_data_sdk::types::bfloat16>;

template <typename T>
class ComparisonDiffByDataTypeDispatch : public ::testing::Test
{
};

TYPED_TEST_SUITE(ComparisonDiffByDataTypeDispatch, SupportedDataTypes, );

TYPED_TEST(ComparisonDiffByDataTypeDispatch, MismatchProducesSummary)
{
    using T = TypeParam;
    constexpr auto DATA_TYPE = DataTypeTraits<T>::DATA_TYPE;

    Tensor<T> ref({4});
    Tensor<T> actual({4});

    for(int64_t i = 0; i < 4; ++i)
    {
        ref.setHostValue(static_cast<T>(0.0f), std::vector<int64_t>{i});
        actual.setHostValue(static_cast<T>(0.0f), std::vector<int64_t>{i});
    }
    ref.setHostValue(static_cast<T>(1.0f), std::vector<int64_t>{2});
    actual.setHostValue(static_cast<T>(2.0f), std::vector<int64_t>{2});

    std::ostringstream oss;
    appendComparisonDiffByDataType(oss, DATA_TYPE, "y", ref, actual, 0.0f, 0.0f);
    const std::string output = oss.str();

    EXPECT_NE(output.find("Total elements:"), std::string::npos);
    EXPECT_NE(output.find("Mismatched:"), std::string::npos);
    EXPECT_TRUE(std::regex_search(output, std::regex(R"(Mismatched:\s+1\b)")));
}

TYPED_TEST(ComparisonDiffByDataTypeDispatch, MatchingTensorsShowZeroMismatches)
{
    using T = TypeParam;
    constexpr auto DATA_TYPE = DataTypeTraits<T>::DATA_TYPE;

    Tensor<T> ref({3});
    Tensor<T> actual({3});

    for(int64_t i = 0; i < 3; ++i)
    {
        ref.setHostValue(static_cast<T>(static_cast<float>(i)), std::vector<int64_t>{i});
        actual.setHostValue(static_cast<T>(static_cast<float>(i)), std::vector<int64_t>{i});
    }

    std::ostringstream oss;
    appendComparisonDiffByDataType(oss, DATA_TYPE, "clean", ref, actual, 1e-5f, 1e-5f);
    const std::string output = oss.str();

    EXPECT_TRUE(std::regex_search(output, std::regex(R"(Mismatched:\s+0\b)")));
    EXPECT_EQ(output.find("Worst mismatches:"), std::string::npos);
}

// =================================================================================================
// appendComparisonDiffByDataType — unsupported type
// =================================================================================================

TEST(TestComparisonDiffByDataTypeDispatch, UnsupportedTypeShowsMessage)
{
    Tensor<float> ref({2});
    Tensor<float> actual({2});

    std::ostringstream oss;
    appendComparisonDiffByDataType(oss, DT::INT8, "w", ref, actual, 0.0f, 0.0f);

    EXPECT_NE(oss.str().find("no element-wise diff available for data type: INT8"),
              std::string::npos);
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

    EXPECT_TRUE(std::regex_search(output, std::regex(R"(Mismatched:\s+1\b)")));
    EXPECT_NE(output.find("Worst mismatches:"), std::string::npos);
}
