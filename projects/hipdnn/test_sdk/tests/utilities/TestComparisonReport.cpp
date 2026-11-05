// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Helpers.hpp"

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_data_sdk::helpers::createRaggedSdpaTensor;

// =================================================================================================
// formatComparisonHeader
// =================================================================================================

TEST(TestFormatComparisonHeader, ContainsAllFields)
{
    const ComparisonContext ctx{"Bundle: /path/to/bundle",
                                "output_tensor (UID 42, output)",
                                "FLOAT",
                                1e-5f,
                                1e-4f,
                                std::nullopt};

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
    const ComparisonContext ctx{"Test: MyTest.Case", "x", "HALF", 0.0f, 0.0f, std::nullopt};

    const Tensor<float> tensor({8, 16});
    const std::string header = formatComparisonHeader(ctx, tensor);

    EXPECT_NE(header.find("Shape:"), std::string::npos);
    EXPECT_NE(header.find("8, 16"), std::string::npos);
}

// A check that atol/rtol do not describe says so, instead of printing two numbers that
// had nothing to do with the verdict. The header owns the wording: callers hand over the
// threshold, never a rendered sentence.
TEST(TestFormatComparisonHeader, RmsThresholdReplacesAtolRtol)
{
    const ComparisonContext ctx{"Test: MyTest.Case", "dscale", "FLOAT", 1e-5f, 1e-4f, 1e-4f};

    const Tensor<float> tensor({8, 16});
    const std::string header = formatComparisonHeader(ctx, tensor);

    EXPECT_NE(header.find("relative RMS <= 0.0001"), std::string::npos);
    EXPECT_EQ(header.find("atol="), std::string::npos);
    EXPECT_EQ(header.find("rtol="), std::string::npos);
    // The element counts printed below an aggregate verdict are elements that differ at
    // all, not elements that failed. Saying so is the header's job, not the caller's.
    EXPECT_NE(header.find("aggregate check"), std::string::npos);
    EXPECT_NE(header.find("not elements that failed"), std::string::npos);
}

// The threshold is optional, and absent means an ordinary per-element verdict.
TEST(TestFormatComparisonHeader, NoRmsThresholdPrintsAtolRtol)
{
    const ComparisonContext ctx{"Test: MyTest.Case", "dx", "FLOAT", 1e-5f, 1e-4f, std::nullopt};

    const Tensor<float> tensor({8, 16});
    const std::string header = formatComparisonHeader(ctx, tensor);

    EXPECT_NE(header.find("atol=1e-05"), std::string::npos);
    EXPECT_NE(header.find("rtol=0.0001"), std::string::npos);
    EXPECT_EQ(header.find("relative RMS"), std::string::npos);
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

TEST(TestAppendComparisonDiff, RaggedLayoutMismatchIsReported)
{
    auto expected = createRaggedSdpaTensor<float>(1.0f);
    auto actual = createRaggedSdpaTensor<float>(1.0f, {0, 3, 5});

    std::ostringstream oss;
    appendComparisonDiff<float>(oss, "o", expected, actual, 0.0f, 0.0f);
    const std::string output = oss.str();

    EXPECT_NE(output.find("ragged layout mismatch"), std::string::npos) << output;
    EXPECT_EQ(output.find("Total elements:"), std::string::npos) << output;
}

// =================================================================================================
// Non-finite elements the validator rejected
// =================================================================================================

namespace
{

template <typename T = float>
Tensor<T> makeValues(const std::vector<float>& values)
{
    Tensor<T> tensor({static_cast<int64_t>(values.size())});
    for(size_t i = 0; i < values.size(); ++i)
    {
        tensor.setHostValue(static_cast<T>(values[i]),
                            std::vector<int64_t>{static_cast<int64_t>(i)});
    }
    return tensor;
}

constexpr float QUIET_NAN = std::numeric_limits<float>::quiet_NaN();
constexpr float POSITIVE_INF = std::numeric_limits<float>::infinity();

} // namespace

// The RMS check fails on a NaN, and the report lists elements with atol = rtol = 0. A NaN used
// to show as `Mismatched: 0`, which read as a failure with nothing wrong.
TEST(TestAppendComparisonDiff, RmsNaNFailureShowsNonZeroCount)
{
    auto expected = makeValues({1.0f, 2.0f, 3.0f, 4.0f});
    auto actual = makeValues({1.0f, 2.0f, QUIET_NAN, 4.0f});

    const auto validator = createRmsValidator(DT::FLOAT, 1e-3f);
    ASSERT_FALSE(validator->allClose(expected, actual));

    std::ostringstream oss;
    appendComparisonDiffByDataType(oss, DT::FLOAT, "dscale", expected, actual, 0.0f, 0.0f);
    const std::string output = oss.str();

    EXPECT_TRUE(std::regex_search(output, std::regex(R"(Mismatched:\s+1\b)"))) << output;
    EXPECT_NE(output.find("[2]: ref=3.000000e+00, impl=nan"), std::string::npos) << output;
}

TEST(TestAppendComparisonDiff, MaskedReportCountsUnwrittenElementsAndSkipsMatchedNonFinite)
{
    using hipdnn_data_sdk::types::half;
    const auto sentinel = payloadNanSentinelBits<half>();
    ASSERT_TRUE(sentinel.has_value());

    // Position 0: NaN matches NaN. 1: +Inf matches +Inf. 2: unwritten. 3: NaN against finite.
    auto expected = makeValues<half>({QUIET_NAN, POSITIVE_INF, QUIET_NAN, 1.0f, 2.0f});
    auto actual = makeValues<half>({QUIET_NAN, POSITIVE_INF, 0.0f, QUIET_NAN, 2.0f});
    actual.setHostValue(payloadNanSentinelValue<half>(), std::vector<int64_t>{2});

    const auto validator = createAllCloseMatchingNonFiniteValidator(DT::HALF, 0.0f, 0.0f, sentinel);
    ASSERT_FALSE(validator->allClose(expected, actual));

    std::ostringstream oss;
    appendComparisonDiffByDataType(oss,
                                   DT::HALF,
                                   "o",
                                   expected,
                                   actual,
                                   0.0f,
                                   0.0f,
                                   NonFiniteMask::matching(sentinel),
                                   MatchingInfinities::REJECTED);
    const std::string output = oss.str();

    EXPECT_TRUE(std::regex_search(output, std::regex(R"(Mismatched:\s+2\b)"))) << output;
    EXPECT_NE(output.find("[2]: ref=nan, impl=nan (unwritten-output sentinel)"), std::string::npos)
        << output;
    EXPECT_NE(output.find("[3]: ref=1.000000e+00, impl=nan"), std::string::npos) << output;
    EXPECT_EQ(output.find("[0]:"), std::string::npos) << output;
    EXPECT_EQ(output.find("[1]:"), std::string::npos) << output;
}

TEST(TestAppendComparisonDiff, MatchingInfinitiesAreNotListedWhenAccepted)
{
    auto expected = makeValues({-POSITIVE_INF, 1.0f});
    auto actual = makeValues({-POSITIVE_INF, 1.0f});

    std::ostringstream rejected;
    appendComparisonDiff<float>(rejected, "stats", expected, actual, 0.0f, 0.0f);
    EXPECT_TRUE(std::regex_search(rejected.str(), std::regex(R"(Mismatched:\s+1\b)")))
        << rejected.str();

    std::ostringstream accepted;
    appendComparisonDiff<float>(accepted,
                                "stats",
                                expected,
                                actual,
                                0.0f,
                                0.0f,
                                NonFiniteMask{},
                                MatchingInfinities::ACCEPTED);
    EXPECT_TRUE(std::regex_search(accepted.str(), std::regex(R"(Mismatched:\s+0\b)")))
        << accepted.str();
}

TEST(TestAppendComparisonDiff, AllFiniteReportKeepsItsShape)
{
    auto expected = makeValues({1.0f, 1.0f});
    auto actual = makeValues({1.0f, 3.0f});

    std::ostringstream oss;
    appendComparisonDiff<float>(oss, "t", expected, actual, 0.0f, 0.0f);

    EXPECT_EQ(oss.str(),
              "  Tensor diff for \"t\":\n"
              "    Total elements: 2\n"
              "    Mismatched:     1 (50.00%)\n"
              "    Max abs diff:   2.000000e+00 at [1]\n"
              "    Mean abs diff:  2.000000e+00\n"
              "    Worst mismatches:\n"
              "      [1]: ref=1.000000e+00, impl=3.000000e+00, diff=2.000000e+00\n");
}
