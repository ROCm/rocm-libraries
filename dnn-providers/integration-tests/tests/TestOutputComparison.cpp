// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The output comparison, driven directly. It used to sit inside the harness with an
// EXPECT_TRUE in the middle of it, so the only way to reach it was to run a whole
// TestBody() and read the failure back out of a fake part-result reporter. It now
// returns its mismatches, so these assert on them.

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/OutputComparison.hpp"

using hipdnn_integration_tests::bundle::compareOutputs;
using hipdnn_integration_tests::bundle::compareTensor;
using hipdnn_integration_tests::bundle::ComparisonTolerance;
using hipdnn_integration_tests::bundle::OutputTensors;
using hipdnn_integration_tests::bundle::tensorLabel;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t K_UID_A = 5; // named "y_out"
constexpr int64_t K_UID_B = 4; // unnamed

// The batchnorm graph the other bundle tests use, with uid 5 given a name so both
// label paths are reachable: uid 5 reports its name, uid 4 falls back to its uid.
const std::string K_GRAPH
    = R"({"nodes": [{"inputs": {"x_tensor_uid": 0, "mean_tensor_uid": 1, )"
      R"("inv_variance_tensor_uid": 2, "scale_tensor_uid": 3, "bias_tensor_uid": 4}, )"
      R"("outputs": {"y_tensor_uid": 5}, "type": "BatchnormInferenceAttributes", )"
      R"("compute_data_type": "float", "name": ""}], "tensors": [)"
      R"({"name": "", "uid": 0, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "", "uid": 1, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "", "uid": 2, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "", "uid": 3, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "", "uid": 4, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "y_out", "uid": 5, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
      R"("data_type": "float", "virtual": false}], "io_data_type": "float", )"
      R"("compute_data_type": "float", "intermediate_data_type": "float", "name": ""})";
// Built the same way the bundle loader builds one, so these tests walk a real
// flatbuffer graph rather than a hand-rolled attribute map.
flatbuffers::DetachedBuffer makeGraphBuffer()
{
    flatbuffers::DetachedBuffer buffer;
    EXPECT_TRUE(hipdnn_integration_tests::bundle::detail::buildGraphBuffer(
        nlohmann::json::parse(K_GRAPH), buffer));
    return buffer;
}

std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    floatTensor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs, float value)
{
    auto tensor = hipdnn_test_sdk::detail::createTensorFromAttribute(attrs);
    tensor->fillTensorWithValue(value);
    return tensor;
}

ComparisonTolerance exact()
{
    return ComparisonTolerance{0.0f, 0.0f};
}

} // namespace

// ---------------------------------------------------------------------------
// tensorLabel: named tensors report their name, unnamed ones their uid.
// ---------------------------------------------------------------------------

TEST(TestOutputComparison, LabelPrefersTheTensorName)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};

    EXPECT_EQ(tensorLabel(K_UID_A, *wrapper.getTensorMap().at(K_UID_A)), "y_out");
}

TEST(TestOutputComparison, LabelFallsBackToTheUid)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};

    EXPECT_EQ(tensorLabel(K_UID_B, *wrapper.getTensorMap().at(K_UID_B)), "uid=4");
}

// ---------------------------------------------------------------------------
// compareTensor: nullopt on a match, a formatted report on a mismatch.
// ---------------------------------------------------------------------------

TEST(TestOutputComparison, MatchingTensorReportsNothing)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    auto expected = floatTensor(attrs, 3.5f);
    auto actual = floatTensor(attrs, 3.5f);

    EXPECT_FALSE(
        compareTensor(K_UID_A, attrs, *expected, *actual, exact(), "Bundle: b").has_value());
}

TEST(TestOutputComparison, MismatchCarriesTheUidLabelAndDiff)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    auto expected = floatTensor(attrs, 3.5f);
    auto actual = floatTensor(attrs, 9.25f);

    const auto mismatch
        = compareTensor(K_UID_A, attrs, *expected, *actual, exact(), "Bundle: my-bundle");

    ASSERT_TRUE(mismatch.has_value());
    EXPECT_EQ(mismatch->uid, K_UID_A);
    EXPECT_EQ(mismatch->label, "y_out");
    // The report is what a reader gets instead of the tensors, so it has to name the
    // bundle, the tensor and the values that disagreed.
    EXPECT_NE(mismatch->report.find("my-bundle"), std::string::npos);
    EXPECT_NE(mismatch->report.find("y_out"), std::string::npos);
    EXPECT_NE(mismatch->report.find("9.25"), std::string::npos);
}

// Tolerance is the caller's to decide, and it decides the answer.
TEST(TestOutputComparison, ToleranceDecidesWhetherADifferenceMatters)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    auto expected = floatTensor(attrs, 1.0f);
    auto actual = floatTensor(attrs, 1.01f);

    EXPECT_TRUE(compareTensor(K_UID_A, attrs, *expected, *actual, exact(), "b").has_value());
    EXPECT_FALSE(
        compareTensor(K_UID_A, attrs, *expected, *actual, ComparisonTolerance{0.1f, 0.1f}, "b")
            .has_value());
}

// ---------------------------------------------------------------------------
// compareOutputs: every uid is compared, and the walk does not stop at the first
// mismatch — one failing test should name every tensor that drifted.
// ---------------------------------------------------------------------------

TEST(TestOutputComparison, AllOutputsMatchingYieldsNoMismatches)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& map = wrapper.getTensorMap();

    OutputTensors actual;
    actual[K_UID_A] = floatTensor(*map.at(K_UID_A), 1.0f);
    actual[K_UID_B] = floatTensor(*map.at(K_UID_B), 2.0f);

    OutputTensors expected;
    expected[K_UID_A] = floatTensor(*map.at(K_UID_A), 1.0f);
    expected[K_UID_B] = floatTensor(*map.at(K_UID_B), 2.0f);

    const auto mismatches = compareOutputs(
        wrapper,
        {K_UID_A, K_UID_B},
        actual,
        [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& { return *expected.at(uid); },
        [](auto) { return exact(); },
        "Bundle: b");

    EXPECT_TRUE(mismatches.empty());
}

TEST(TestOutputComparison, EveryDriftedTensorIsReportedNotJustTheFirst)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& map = wrapper.getTensorMap();

    OutputTensors actual;
    actual[K_UID_A] = floatTensor(*map.at(K_UID_A), 1.0f);
    actual[K_UID_B] = floatTensor(*map.at(K_UID_B), 2.0f);

    OutputTensors expected;
    expected[K_UID_A] = floatTensor(*map.at(K_UID_A), 99.0f);
    expected[K_UID_B] = floatTensor(*map.at(K_UID_B), 99.0f);

    const auto mismatches = compareOutputs(
        wrapper,
        {K_UID_A, K_UID_B},
        actual,
        [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& { return *expected.at(uid); },
        [](auto) { return exact(); },
        "Bundle: b");

    ASSERT_EQ(mismatches.size(), 2u);
    EXPECT_EQ(mismatches[0].uid, K_UID_A);
    EXPECT_EQ(mismatches[1].uid, K_UID_B);
}

// Only the uids asked for are compared: an output the bundle does not list is not
// this comparison's business.
TEST(TestOutputComparison, OnlyTheRequestedUidsAreCompared)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& map = wrapper.getTensorMap();

    OutputTensors actual;
    actual[K_UID_A] = floatTensor(*map.at(K_UID_A), 1.0f);
    actual[K_UID_B] = floatTensor(*map.at(K_UID_B), 2.0f);

    OutputTensors expected;
    expected[K_UID_A] = floatTensor(*map.at(K_UID_A), 1.0f);
    expected[K_UID_B] = floatTensor(*map.at(K_UID_B), 99.0f);

    const auto mismatches = compareOutputs(
        wrapper,
        {K_UID_A},
        actual,
        [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& { return *expected.at(uid); },
        [](auto) { return exact(); },
        "Bundle: b");

    EXPECT_TRUE(mismatches.empty()) << "uid 4 drifted but was not in the list";
}

// NOLINTEND(readability-identifier-naming)
