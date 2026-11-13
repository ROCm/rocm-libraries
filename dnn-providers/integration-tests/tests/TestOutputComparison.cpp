// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The output comparison, driven directly. It returns its mismatches, so these assert on
// them.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn-gpu-ref/GpuReferenceValidationFactory.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/NonFiniteMask.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/LiveRows.hpp"
#include "harness/bundle/OutputComparison.hpp"

using hipdnn_integration_tests::ValidationSite;
using hipdnn_integration_tests::bundle::compareOutputs;
using hipdnn_integration_tests::bundle::compareTensor;
using hipdnn_integration_tests::bundle::ComparisonTolerance;
using hipdnn_integration_tests::bundle::makeValidator;
using hipdnn_integration_tests::bundle::OutputTensors;
using hipdnn_integration_tests::bundle::tensorLabel;
using hipdnn_integration_tests::bundle::ValidatorKind;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t K_UID_A = 5; // named "y_out"
constexpr int64_t K_UID_B = 4; // unnamed
// Not wired to a node: it exists so a test can ask what an integer output does when a
// 'tensors' glob is wide enough to select RMS for it.
constexpr int64_t K_UID_INT = 6; // named "counts", INT32
// Not wired to a node: a dense [B, H, S, 1] row statistic for the live-row tests.
constexpr int64_t K_UID_STATS = 7; // named "stats"

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
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "counts", "uid": 6, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
      R"("data_type": "int32", "virtual": false}, )"
      R"({"name": "stats", "uid": 7, "strides": [12, 4, 1, 1], "dims": [2, 3, 4, 1], )"
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

std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    floatTensor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs, float value)
{
    auto tensor = hipdnn_test_sdk::detail::createTensorFromAttribute(attrs);
    tensor->fillTensorWithValue(value);
    return tensor;
}

// A [1, 3, 1, 1] tensor with the three values written individually, so a test can build
// the magnitude spread that separates allclose from RMS.
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    floatTensor3(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                 float v0,
                 float v1,
                 float v2)
{
    auto tensor = hipdnn_test_sdk::detail::createTensorFromAttribute(attrs);
    hipdnn_data_sdk::utilities::TensorView<float> view(*tensor);
    view.getHostValue({0, 0, 0, 0}) = v0;
    view.getHostValue({0, 1, 0, 0}) = v1;
    view.getHostValue({0, 2, 0, 0}) = v2;
    tensor->markHostModified();
    return tensor;
}

// An integer tensor, filled with one value. Integer outputs exist in this suite, and
// the RMS validator has no implementation for them.
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    intTensor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs, int32_t value)
{
    auto tensor = hipdnn_test_sdk::detail::createTensorFromAttribute(attrs);
    tensor->fillTensorWithValue(static_cast<float>(value));
    return tensor;
}

ComparisonTolerance exact()
{
    return ComparisonTolerance::allClose(0.0f, 0.0f);
}

// Leaves new values on the device only, the way an engine or a GPU reference does:
// the host copy still holds the old fill until something reads it back.
void overwriteOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor, float value)
{
    const std::vector<float> values(tensor.elementCount(), value);
    ASSERT_EQ(hipMemcpy(tensor.rawDeviceData(),
                        values.data(),
                        values.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              hipSuccess);
    tensor.markDeviceModified();
}

constexpr float K_RMS_THRESHOLD = 1e-4f;

constexpr float K_INF = std::numeric_limits<float>::infinity();
constexpr float K_NAN = std::numeric_limits<float>::quiet_NaN();

ComparisonTolerance exactMatchingInfinities()
{
    return ComparisonTolerance::allCloseMatchingInfinities(0.0f, 0.0f);
}

// A ragged [B, H, S, D] = [2, 3, 4, 5] tensor in the BSHD physical layout, holding 2 and 3
// sequence rows, filled with one value.
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor> raggedFloatTensor(float value)
{
    const std::vector<int32_t> offsets = {0, 30, 75};
    auto offsetTensor = std::make_shared<hipdnn_data_sdk::utilities::Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(offsets.size()), 1, 1, 1});
    for(size_t i = 0; i < offsets.size(); ++i)
    {
        offsetTensor->setHostValue(offsets[i], static_cast<int64_t>(i), 0, 0, 0);
    }
    auto tensor = std::make_shared<hipdnn_data_sdk::utilities::RaggedTensor<float>>(
        std::vector<int64_t>{2, 3, 4, 5},
        std::vector<int64_t>{60, 5, 15, 1},
        hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS,
        offsetTensor);
    tensor->fillWithValue(value);
    return tensor;
}

// The Stats (or head-major primary) tensor of a ragged SDPA case, holding the logical value
// 100*b + 10*h + s at every in-block coordinate (d = 0), whatever the storage layout.
// `perturb` adds 1 to the value at one coordinate.
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    raggedLogicalTensor(const hipdnn_integration_tests::bundle::fixtures::RaggedLayout& layout,
                        bool perturb = false)
{
    using hipdnn_data_sdk::utilities::Tensor;
    const auto int32Aux = [](const std::vector<int32_t>& values) {
        auto tensor = std::make_shared<Tensor<int32_t>>(
            std::vector<int64_t>{static_cast<int64_t>(values.size()), 1, 1, 1});
        for(size_t i = 0; i < values.size(); ++i)
        {
            tensor->setHostValue(values[i], static_cast<int64_t>(i), 0, 0, 0);
        }
        return tensor;
    };

    auto tensor = std::make_shared<hipdnn_data_sdk::utilities::RaggedTensor<float>>(
        layout.dims,
        layout.strides,
        hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS,
        int32Aux(layout.offsets),
        std::nullopt,
        layout.offsetMultiplier);
    tensor->fillWithValue(-1.0f);

    const auto info = tensor->raggedIterationInfo();
    const auto& lengths = info->lengths;
    for(int64_t b = 0; b < layout.dims[0]; ++b)
    {
        for(int64_t h = 0; h < layout.dims[1]; ++h)
        {
            for(int64_t s = 0; s < lengths[static_cast<size_t>(b)]; ++s)
            {
                const bool bump = perturb && b == 1 && h == 1 && s == 1;
                tensor->setHostValue(
                    static_cast<float>(100 * b + 10 * h + s) + (bump ? 1.0f : 0.0f), b, h, s, 0);
            }
        }
    }
    return tensor;
}

// A dense [2, 3, 4, 1] float tensor with `strides`. Every position holds NaN except the
// live rows s < lengths[b], which hold 10*b + h + s.
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    liveRowStats(const std::vector<int64_t>& strides, const std::vector<int64_t>& lengths)
{
    auto tensor = std::make_shared<hipdnn_data_sdk::utilities::Tensor<float>>(
        std::vector<int64_t>{2, 3, 4, 1}, strides);
    tensor->fillWithValue(K_NAN);
    for(int64_t b = 0; b < 2; ++b)
    {
        for(int64_t h = 0; h < 3; ++h)
        {
            for(int64_t s = 0; s < lengths[static_cast<size_t>(b)]; ++s)
            {
                tensor->setHostValue(static_cast<float>(10 * b + h + s), b, h, s, 0);
            }
        }
    }
    return tensor;
}

const std::vector<int64_t> K_STATS_STRIDES = {12, 4, 1, 1};
// Token-major: the head index changes fastest.
const std::vector<int64_t> K_STATS_TOKEN_MAJOR_STRIDES = {12, 1, 3, 1};

std::vector<hipdnn_integration_tests::bundle::TensorMismatch>
    compareStats(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                 const std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>& expected,
                 const std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>& actual,
                 const hipdnn_integration_tests::bundle::LiveRowMap& liveRows,
                 ValidationSite site = ValidationSite::HOST,
                 std::optional<ValidationSite> liveRowSite = std::nullopt)
{
    OutputTensors actualOutputs;
    actualOutputs[K_UID_STATS] = actual;
    return compareOutputs(
        wrapper,
        {K_UID_STATS},
        actualOutputs,
        [&](int64_t) -> hipdnn_data_sdk::utilities::ITensor& { return *expected; },
        [](const std::string&, auto) { return exact(); },
        site,
        "Bundle: b",
        liveRows,
        liveRowSite);
}

// liveRowStats with `pastValue` in place of NaN at every position past the lengths.
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    liveRowStatsPast(const std::vector<int64_t>& lengths, float pastValue)
{
    auto tensor = liveRowStats(K_STATS_STRIDES, lengths);
    hipdnn_data_sdk::utilities::TensorView<float> view(*tensor);
    for(int64_t b = 0; b < 2; ++b)
    {
        for(int64_t h = 0; h < 3; ++h)
        {
            for(int64_t s = lengths[static_cast<size_t>(b)]; s < 4; ++s)
            {
                view.getHostValue({b, h, s, 0}) = pastValue;
            }
        }
    }
    return tensor;
}

hipdnn_integration_tests::bundle::LiveRowMap statsLiveRows(std::vector<int64_t> lengths)
{
    hipdnn_integration_tests::bundle::LiveRowMap map;
    map[K_UID_STATS] = hipdnn_integration_tests::bundle::LiveRows{2, std::move(lengths)};
    return map;
}

// Same shape as liveRowStats(), templated on the element type so the "stats" output
// can be built as something other than float.
template <class T>
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    liveRowStatsT(const std::vector<int64_t>& strides, const std::vector<int64_t>& lengths)
{
    auto tensor = std::make_shared<hipdnn_data_sdk::utilities::Tensor<T>>(
        std::vector<int64_t>{2, 3, 4, 1}, strides);
    tensor->fillWithValue(static_cast<T>(K_NAN));
    for(int64_t b = 0; b < 2; ++b)
    {
        for(int64_t h = 0; h < 3; ++h)
        {
            for(int64_t s = 0; s < lengths[static_cast<size_t>(b)]; ++s)
            {
                tensor->setHostValue(static_cast<T>(10 * b + h + s), b, h, s, 0);
            }
        }
    }
    return tensor;
}

// makeGraphBuffer() with the "stats" tensor's data_type replaced, so a test can drive
// makeLiveRowView() through a dtype other than FLOAT.
flatbuffers::DetachedBuffer makeGraphBufferWithStatsType(const std::string& dataType)
{
    auto json = nlohmann::json::parse(K_GRAPH);
    for(auto& tensor : json.at("tensors"))
    {
        if(tensor.at("uid").get<int64_t>() == K_UID_STATS)
        {
            tensor["data_type"] = dataType;
        }
    }
    flatbuffers::DetachedBuffer buffer;
    EXPECT_TRUE(hipdnn_integration_tests::bundle::detail::buildGraphBuffer(json, buffer));
    return buffer;
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

    EXPECT_FALSE(compareTensor(
                     K_UID_A, attrs, *expected, *actual, exact(), ValidationSite::HOST, "Bundle: b")
                     .has_value());
}

TEST(TestOutputComparison, MismatchCarriesTheUidLabelAndDiff)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    auto expected = floatTensor(attrs, 3.5f);
    auto actual = floatTensor(attrs, 9.25f);

    const auto mismatch = compareTensor(
        K_UID_A, attrs, *expected, *actual, exact(), ValidationSite::HOST, "Bundle: my-bundle");

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

    EXPECT_TRUE(
        compareTensor(K_UID_A, attrs, *expected, *actual, exact(), ValidationSite::HOST, "b")
            .has_value());
    EXPECT_FALSE(compareTensor(K_UID_A,
                               attrs,
                               *expected,
                               *actual,
                               ComparisonTolerance::allClose(0.1f, 0.1f),
                               ValidationSite::HOST,
                               "b")
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
        [](const std::string&, auto) { return exact(); },
        ValidationSite::HOST,
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
        [](const std::string&, auto) { return exact(); },
        ValidationSite::HOST,
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
        [](const std::string&, auto) { return exact(); },
        ValidationSite::HOST,
        "Bundle: b");

    EXPECT_TRUE(mismatches.empty()) << "uid 4 drifted but was not in the list";
}

// ---------------------------------------------------------------------------
// Validator kind. ALLCLOSE grades each element against its own magnitude; RMS grades
// the tensor against its largest. The two disagree exactly where reduction outputs
// live: an element near zero, on a tensor whose scale is large.
//
// These run at both sites. A GPU reference's output is compared on the device by
// default and golden data on the host, and the two must reach the same verdict on the
// same data. The tensors are filled on the host; the device validators read them after
// migration, as they read a golden tensor under --validator gpu.
// ---------------------------------------------------------------------------

class TestOutputComparisonSite : public ::testing::TestWithParam<ValidationSite>
{
protected:
    void SetUp() override
    {
        if(GetParam() == ValidationSite::DEVICE)
        {
            SKIP_IF_NO_DEVICES();
        }
    }
};

INSTANTIATE_TEST_SUITE_P(BothSites,
                         TestOutputComparisonSite,
                         ::testing::Values(ValidationSite::HOST, ValidationSite::DEVICE),
                         [](const ::testing::TestParamInfo<ValidationSite>& info) {
                             return std::string(info.param == ValidationSite::HOST ? "Host"
                                                                                   : "Device");
                         });

TEST_P(TestOutputComparisonSite, RmsAcceptsANearZeroElementThatAllcloseRejects)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    // Middle element is 5 orders of magnitude below the tensor scale and drifts by 1e-4:
    // negligible against the tensor, unbounded against itself.
    auto expected = floatTensor3(attrs, 1000.0f, 0.001f, -1000.0f);
    auto actual = floatTensor3(attrs, 1000.0f, 0.0011f, -1000.0f);

    EXPECT_TRUE(compareTensor(K_UID_B,
                              attrs,
                              *expected,
                              *actual,
                              ComparisonTolerance::allClose(0.0f, 1e-4f),
                              GetParam(),
                              "b")
                    .has_value())
        << "allclose should reject: rtol*|ref| is 1e-7 against a 1e-4 drift";

    EXPECT_FALSE(compareTensor(K_UID_B,
                               attrs,
                               *expected,
                               *actual,
                               ComparisonTolerance::rms(K_RMS_THRESHOLD),
                               GetParam(),
                               "b")
                     .has_value())
        << "relative RMS is ~6e-8 against a 1e-4 threshold";
}

TEST_P(TestOutputComparisonSite, RmsStillRejectsDriftLargeAgainstTheTensorScale)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, 1000.0f, 0.001f, -1000.0f);
    auto actual = floatTensor3(attrs, 1000.0f, 0.001f, -900.0f);

    EXPECT_TRUE(compareTensor(K_UID_B,
                              attrs,
                              *expected,
                              *actual,
                              ComparisonTolerance::rms(K_RMS_THRESHOLD),
                              GetParam(),
                              "b")
                    .has_value())
        << "RMS is a real check, not a pass-through";
}

TEST_P(TestOutputComparisonSite, RmsFailureReportsItsThresholdNotAtolRtol)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, 1000.0f, 0.001f, -1000.0f);
    auto actual = floatTensor3(attrs, 1000.0f, 0.001f, -900.0f);

    const auto mismatch = compareTensor(K_UID_B,
                                        attrs,
                                        *expected,
                                        *actual,
                                        ComparisonTolerance::rms(K_RMS_THRESHOLD),
                                        GetParam(),
                                        "b");

    ASSERT_TRUE(mismatch.has_value());
    EXPECT_NE(mismatch->report.find("relative RMS"), std::string::npos);
    EXPECT_EQ(mismatch->report.find("atol="), std::string::npos)
        << "atol/rtol did not decide this failure and must not be printed as if they had";
}

// Equal verdicts at both sites are the point, which is also why the verdict tests above
// cannot tell which validator ran. The site has to build its own site's validator: a
// DEVICE comparison that quietly built a host validator would pass every one of them.
TEST(TestOutputComparison, EachSiteBuildsItsOwnValidator)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto built = [](const ComparisonTolerance& tolerance, ValidationSite site) {
        auto selection = makeValidator(DataType::FLOAT, "y_out", tolerance, site);
        EXPECT_TRUE(selection.error.empty()) << selection.error;
        return std::move(selection.validator);
    };
    const auto allClose = ComparisonTolerance::allClose(0.0f, 0.0f);
    const auto rms = ComparisonTolerance::rms(K_RMS_THRESHOLD);

    const auto hostAllClose = built(allClose, ValidationSite::HOST);
    const auto deviceAllClose = built(allClose, ValidationSite::DEVICE);
    const auto hostRms = built(rms, ValidationSite::HOST);
    const auto deviceRms = built(rms, ValidationSite::DEVICE);

    EXPECT_NE(dynamic_cast<const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float>*>(
                  hostAllClose.get()),
              nullptr);
    EXPECT_NE(
        dynamic_cast<const hipdnn_gpu_ref::GpuFpReferenceValidation<float>*>(deviceAllClose.get()),
        nullptr);
    EXPECT_NE(
        dynamic_cast<const hipdnn_test_sdk::utilities::CpuFpReferenceMiopenRmsValidation<float>*>(
            hostRms.get()),
        nullptr);
    EXPECT_NE(
        dynamic_cast<const hipdnn_gpu_ref::GpuFpReferenceRmsValidation<float>*>(deviceRms.get()),
        nullptr);
}

// The point of the lookup taking a uid and a label: one graph, two outputs, two
// validators. This is what an engine TOML [[validator_overrides]] entry drives.
TEST(TestOutputComparison, ValidatorKindIsChosenPerTensor)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& map = wrapper.getTensorMap();

    // uid 4 is the near-zero reduction-like output; uid 5 ("y_out") drifts outright.
    OutputTensors expected;
    expected[K_UID_A] = floatTensor(*map.at(K_UID_A), 1.0f);
    expected[K_UID_B] = floatTensor3(*map.at(K_UID_B), 1000.0f, 0.001f, -1000.0f);

    OutputTensors actual;
    actual[K_UID_A] = floatTensor(*map.at(K_UID_A), 99.0f);
    actual[K_UID_B] = floatTensor3(*map.at(K_UID_B), 1000.0f, 0.0011f, -1000.0f);

    std::vector<std::string> labelsSeen;
    const auto mismatches = compareOutputs(
        wrapper,
        {K_UID_A, K_UID_B},
        actual,
        [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& { return *expected.at(uid); },
        [&](const std::string& label, auto) {
            labelsSeen.push_back(label);
            return label == "uid=4" ? ComparisonTolerance::rms(K_RMS_THRESHOLD)
                                    : ComparisonTolerance::allClose(0.0f, 1e-4f);
        },
        ValidationSite::HOST,
        "Bundle: b");

    EXPECT_EQ(labelsSeen, (std::vector<std::string>{"y_out", "uid=4"}))
        << "the lookup must be given the label a TOML glob would match on";
    ASSERT_EQ(mismatches.size(), 1u) << "uid 4 passes under RMS; uid 5 fails under allclose";
    EXPECT_EQ(mismatches[0].uid, K_UID_A);
}

// ---------------------------------------------------------------------------
// Both harnesses resolve a tensor's label through the same helper: it is what a TOML
// 'tensors' glob matches on and what the report prints, and a raw tensor name is
// neither when the graph left the tensor unnamed.
// ---------------------------------------------------------------------------

TEST(TestOutputComparison, LabelFromANameKeepsTheName)
{
    EXPECT_EQ(tensorLabel(7, "LayernormBackward_0::DSCALE"), "LayernormBackward_0::DSCALE");
}

TEST(TestOutputComparison, LabelFromAnEmptyNameFallsBackToTheUid)
{
    EXPECT_EQ(tensorLabel(7, ""), "uid=7");
}

// ---------------------------------------------------------------------------
// RMS is defined for float, half, bfloat16 and double only. A 'tensors' glob one
// wildcard too wide can still select it for an integer output, and the TOML parser
// cannot catch that: a tensor's dtype is not known until its graph is read. It has to
// come back as a comparison failure naming the tensor, not an exception out of the
// test body.
// ---------------------------------------------------------------------------

TEST_P(TestOutputComparisonSite, RmsOnAnUnsupportedDataTypeIsReportedNotThrown)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_INT);

    auto expected = intTensor(attrs, 7);
    auto actual = intTensor(attrs, 7);

    std::optional<hipdnn_integration_tests::bundle::TensorMismatch> mismatch;
    ASSERT_NO_THROW(mismatch = compareTensor(K_UID_INT,
                                             attrs,
                                             *expected,
                                             *actual,
                                             ComparisonTolerance::rms(K_RMS_THRESHOLD),
                                             GetParam(),
                                             "Bundle: b"));

    // Equal tensors, so this is not a numerical verdict: it reports that the override
    // could not be honoured at all.
    ASSERT_TRUE(mismatch.has_value());
    EXPECT_EQ(mismatch->label, "counts");
    EXPECT_NE(mismatch->report.find("counts"), std::string::npos);
    EXPECT_NE(mismatch->report.find("INT32"), std::string::npos);
    EXPECT_NE(mismatch->report.find("validator_overrides"), std::string::npos)
        << "the operator has to be told which config section over-matched";
}

// The guard is scoped to RMS: integer outputs still compare normally under the default.
TEST_P(TestOutputComparisonSite, AllcloseStillGradesIntegerOutputs)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_INT);

    auto expected = intTensor(attrs, 7);
    auto matching = intTensor(attrs, 7);
    auto drifted = intTensor(attrs, 9);

    EXPECT_FALSE(compareTensor(K_UID_INT, attrs, *expected, *matching, exact(), GetParam(), "b")
                     .has_value());
    EXPECT_TRUE(
        compareTensor(K_UID_INT, attrs, *expected, *drifted, exact(), GetParam(), "b").has_value());
}

// An integer output nobody wrote still holds its sentinel fill (the type's maximum). If
// neither the engine nor the reference wrote it, both sides hold the same value, and the
// comparison must fail anyway, at either site.
TEST_P(TestOutputComparisonSite, UnwrittenIntegerOutputFailsEvenWhenBothSidesMatch)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_INT);

    auto expected = hipdnn_test_sdk::detail::createTensorFromAttribute(attrs);
    auto actual = hipdnn_test_sdk::detail::createTensorFromAttribute(attrs);
    expected->fillWithSentinelValue();
    actual->fillWithSentinelValue();

    EXPECT_TRUE(
        compareTensor(K_UID_INT, attrs, *expected, *actual, exact(), GetParam(), "b").has_value());
}

// A ValidatorKind with no case in makeValidator must not be graded by whichever branch
// happens to fall through. Silently comparing a tensor with a validator nobody chose is
// the failure mode this whole mechanism exists to prevent, so the unhandled kind is
// refused outright rather than defaulted.
TEST(TestOutputComparison, UnhandledValidatorKindIsRefused)
{
    ComparisonTolerance bogus = ComparisonTolerance::allClose(0.0f, 0.0f);
    bogus.kind = static_cast<ValidatorKind>(99);

    EXPECT_THROW(makeValidator(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                               "y_out",
                               bogus,
                               ValidationSite::HOST),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// ValidationSite::DEVICE is where a GPU reference's output is compared. The engine and
// the reference both leave fresh values on the device with a stale host copy, so the
// device comparison has to read the device side, and the failure report — built on
// the host — has to read those same values back rather than the stale fill.
// ---------------------------------------------------------------------------

TEST(TestGpuOutputComparison, DeviceSiteComparesTheValuesLeftOnTheDevice)
{
    SKIP_IF_NO_DEVICES();

    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    // Host copies agree; device copies disagree.
    auto expected = floatTensor(attrs, 3.5f);
    auto actual = floatTensor(attrs, 3.5f);
    ASSERT_NO_FATAL_FAILURE(overwriteOnDevice(*expected, 3.5f));
    ASSERT_NO_FATAL_FAILURE(overwriteOnDevice(*actual, 9.25f));

    const auto mismatch = compareTensor(
        K_UID_A, attrs, *expected, *actual, exact(), ValidationSite::DEVICE, "Bundle: my-bundle");

    ASSERT_TRUE(mismatch.has_value()) << "the device values differ; the stale host fill does not";
    EXPECT_EQ(mismatch->label, "y_out");
    EXPECT_NE(mismatch->report.find("9.25"), std::string::npos)
        << "the report must show the device values, not the stale host fill";
}

TEST(TestGpuOutputComparison, DeviceSiteAcceptsMatchingDeviceValues)
{
    SKIP_IF_NO_DEVICES();

    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    // Host copies disagree; device copies agree.
    auto expected = floatTensor(attrs, 3.5f);
    auto actual = floatTensor(attrs, 9.25f);
    ASSERT_NO_FATAL_FAILURE(overwriteOnDevice(*expected, 1.0f));
    ASSERT_NO_FATAL_FAILURE(overwriteOnDevice(*actual, 1.0f));

    EXPECT_FALSE(
        compareTensor(
            K_UID_A, attrs, *expected, *actual, exact(), ValidationSite::DEVICE, "Bundle: b")
            .has_value());
}

// The device site grades a floating-point allclose tensor by the host rule: two infinities
// of the same sign match. Both kinds are checked, because an engine TOML can select either.
TEST(TestGpuOutputComparison, DeviceSiteMatchesSameSignedInfinities)
{
    SKIP_IF_NO_DEVICES();

    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    for(const auto& tolerance : {exact(), exactMatchingInfinities()})
    {
        for(const float inf : {-K_INF, K_INF})
        {
            auto expected = floatTensor3(attrs, inf, 1.0f, inf);
            auto actual = floatTensor3(attrs, inf, 1.0f, inf);
            EXPECT_FALSE(
                compareTensor(
                    K_UID_B, attrs, *expected, *actual, tolerance, ValidationSite::DEVICE, "b")
                    .has_value())
                << "both sides hold " << inf << " and both are right";
        }
    }

    const auto halfBuffer = makeGraphBufferWithStatsType("half");
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper halfWrapper{halfBuffer.data(),
                                                                                 halfBuffer.size()};
    const auto& halfAttrs = *halfWrapper.getTensorMap().at(K_UID_STATS);
    auto expected = floatTensor(halfAttrs, -K_INF);
    auto actual = floatTensor(halfAttrs, -K_INF);
    EXPECT_FALSE(
        compareTensor(
            K_UID_STATS, halfAttrs, *expected, *actual, exact(), ValidationSite::DEVICE, "b")
            .has_value())
        << "a HALF output follows the same rule";
}

TEST(TestGpuOutputComparison, DeviceSiteStillRejectsWhatTheHostRejects)
{
    SKIP_IF_NO_DEVICES();

    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    struct Case
    {
        const char* name;
        float expected0;
        float actual0;
        float actual1;
    };
    const std::vector<Case> cases = {
        {"opposite-signed infinities", -K_INF, K_INF, 1.0f},
        {"NaN on both sides", K_NAN, K_NAN, 1.0f},
        {"finite versus infinite", -K_INF, -1e30f, 1.0f},
        {"infinite versus finite", 1.0f, K_INF, 1.0f},
        {"a drifted finite element beside matched infinities", -K_INF, -K_INF, 1.1f},
    };
    for(const auto& c : cases)
    {
        for(const auto& tolerance : {exact(), exactMatchingInfinities()})
        {
            auto expected = floatTensor3(attrs, c.expected0, 1.0f, 2.0f);
            auto actual = floatTensor3(attrs, c.actual0, c.actual1, 2.0f);
            EXPECT_TRUE(
                compareTensor(
                    K_UID_B, attrs, *expected, *actual, tolerance, ValidationSite::DEVICE, "b")
                    .has_value())
                << c.name;
        }
    }

    auto expected = floatTensor3(attrs, -K_INF, 1.0f, 2.0f);
    auto actual = floatTensor3(attrs, -K_INF, 1.1f, 2.0f);
    EXPECT_FALSE(compareTensor(K_UID_B,
                               attrs,
                               *expected,
                               *actual,
                               ComparisonTolerance::allClose(0.2f, 0.0f),
                               ValidationSite::DEVICE,
                               "b")
                     .has_value())
        << "atol still decides the finite elements";
}

// The host grading at the device site reads the values the engine and the reference left
// on the device, not the stale host fill.
TEST(TestGpuOutputComparison, DeviceSiteInfinityMatchReadsTheDeviceValues)
{
    SKIP_IF_NO_DEVICES();

    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    auto differingOnDevice = floatTensor(attrs, -K_INF);
    auto differingOnDeviceToo = floatTensor(attrs, -K_INF);
    ASSERT_NO_FATAL_FAILURE(overwriteOnDevice(*differingOnDevice, 1.0f));
    ASSERT_NO_FATAL_FAILURE(overwriteOnDevice(*differingOnDeviceToo, 9.25f));
    EXPECT_TRUE(compareTensor(K_UID_A,
                              attrs,
                              *differingOnDevice,
                              *differingOnDeviceToo,
                              exact(),
                              ValidationSite::DEVICE,
                              "b")
                    .has_value())
        << "the device values differ; only the stale host fill holds matching infinities";

    auto expected = floatTensor(attrs, 3.5f);
    auto actual = floatTensor(attrs, 9.25f);
    ASSERT_NO_FATAL_FAILURE(overwriteOnDevice(*expected, -K_INF));
    ASSERT_NO_FATAL_FAILURE(overwriteOnDevice(*actual, -K_INF));
    EXPECT_FALSE(
        compareTensor(K_UID_A, attrs, *expected, *actual, exact(), ValidationSite::DEVICE, "b")
            .has_value())
        << "the device values are matching infinities; only the stale host fill differs";
}

// ---------------------------------------------------------------------------
// ALLCLOSE_MATCHING_INFINITIES. An output whose correct value is infinite on both
// sides — a fully masked SDPA forward log-sum-exp row — cannot be graded by
// |ref - impl|, which is NaN for two infinities and so fails a tensor that is right.
// The kind relaxes exactly that element and nothing else.
// ---------------------------------------------------------------------------

TEST(TestOutputComparison, AllcloseMatchingInfinitiesAcceptsSameSignedInfinities)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, -K_INF, 1.0f, -K_INF);
    auto actual = floatTensor3(attrs, -K_INF, 1.0f, -K_INF);

    EXPECT_FALSE(compareTensor(K_UID_B,
                               attrs,
                               *expected,
                               *actual,
                               exactMatchingInfinities(),
                               ValidationSite::HOST,
                               "b")
                     .has_value())
        << "both sides are -inf and both are right";

    EXPECT_FALSE(
        compareTensor(K_UID_B, attrs, *expected, *actual, exact(), ValidationSite::HOST, "b")
            .has_value())
        << "on the host, plain allclose of a floating-point tensor matches them too";
}

TEST(TestOutputComparison, AllcloseMatchingInfinitiesStillRejectsOppositeSignedInfinities)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, -K_INF, 1.0f, 1.0f);
    auto actual = floatTensor3(attrs, K_INF, 1.0f, 1.0f);

    EXPECT_TRUE(compareTensor(K_UID_B,
                              attrs,
                              *expected,
                              *actual,
                              exactMatchingInfinities(),
                              ValidationSite::HOST,
                              "b")
                    .has_value())
        << "the sign is part of the match; +inf where -inf belongs is a real disagreement";
}

TEST(TestOutputComparison, AllcloseMatchingInfinitiesStillRejectsNaN)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, K_NAN, 1.0f, 1.0f);
    auto actual = floatTensor3(attrs, K_NAN, 1.0f, 1.0f);

    EXPECT_TRUE(compareTensor(K_UID_B,
                              attrs,
                              *expected,
                              *actual,
                              exactMatchingInfinities(),
                              ValidationSite::HOST,
                              "b")
                    .has_value())
        << "a NaN is not an infinity: matching NaNs stay a failure on both sides";
}

TEST(TestOutputComparison, AllcloseMatchingInfinitiesStillRejectsFiniteVersusInfinite)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, -K_INF, 1.0f, 1.0f);
    auto actual = floatTensor3(attrs, -1e30f, 1.0f, 1.0f);

    EXPECT_TRUE(compareTensor(K_UID_B,
                              attrs,
                              *expected,
                              *actual,
                              exactMatchingInfinities(),
                              ValidationSite::HOST,
                              "b")
                    .has_value())
        << "a very large finite value is not an infinity, however large";
}

TEST(TestOutputComparison, AllcloseMatchingInfinitiesStillGradesFiniteElementsByAtolRtol)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, -K_INF, 1.0f, 2.0f);
    auto actual = floatTensor3(attrs, -K_INF, 1.1f, 2.0f);

    EXPECT_TRUE(compareTensor(K_UID_B,
                              attrs,
                              *expected,
                              *actual,
                              exactMatchingInfinities(),
                              ValidationSite::HOST,
                              "b")
                    .has_value())
        << "the matched infinity must not carry the drifted element through with it";

    EXPECT_FALSE(compareTensor(K_UID_B,
                               attrs,
                               *expected,
                               *actual,
                               ComparisonTolerance::allCloseMatchingInfinities(0.2f, 0.0f),
                               ValidationSite::HOST,
                               "b")
                     .has_value())
        << "atol still decides the finite elements";
}

// Both tensors hold -inf, so the kind alone decides which of them passes.
TEST(TestOutputComparison, AllcloseMatchingInfinitiesIsChosenPerTensor)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& map = wrapper.getTensorMap();

    OutputTensors expected;
    expected[K_UID_A] = floatTensor(*map.at(K_UID_A), -K_INF);
    expected[K_UID_INT] = intTensor(*map.at(K_UID_INT), 7);

    OutputTensors actual;
    actual[K_UID_A] = floatTensor(*map.at(K_UID_A), -K_INF);
    actual[K_UID_INT] = intTensor(*map.at(K_UID_INT), 7);

    std::vector<std::string> labelsSeen;
    const auto mismatches = compareOutputs(
        wrapper,
        {K_UID_A, K_UID_INT},
        actual,
        [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& { return *expected.at(uid); },
        [&](const std::string& label, auto) {
            labelsSeen.push_back(label);
            return label == "counts" ? exactMatchingInfinities() : exact();
        },
        ValidationSite::HOST,
        "Bundle: b");

    EXPECT_EQ(labelsSeen, (std::vector<std::string>{"y_out", "counts"}))
        << "the lookup must be given the label a TOML glob would match on";
    ASSERT_EQ(mismatches.size(), 1u)
        << "y_out is plain allclose and matches its infinities on the host; only counts "
           "gets the kind that an integer tensor cannot take";
    EXPECT_EQ(mismatches[0].uid, K_UID_INT);
    EXPECT_NE(mismatches[0].report.find("allclose_matching_infinities"), std::string::npos);
}

// Integers have no infinity, so this kind is undefined for them — the same shape of
// over-matched glob RMS already has.
TEST(TestOutputComparison, AllcloseMatchingInfinitiesOnAnUnsupportedDataTypeIsReportedNotThrown)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_INT);

    auto expected = intTensor(attrs, 7);
    auto actual = intTensor(attrs, 7);

    std::optional<hipdnn_integration_tests::bundle::TensorMismatch> mismatch;
    ASSERT_NO_THROW(mismatch = compareTensor(K_UID_INT,
                                             attrs,
                                             *expected,
                                             *actual,
                                             exactMatchingInfinities(),
                                             ValidationSite::HOST,
                                             "Bundle: b"));

    // Equal tensors, so this is not a numerical verdict: it reports that the override
    // could not be honoured at all.
    ASSERT_TRUE(mismatch.has_value());
    EXPECT_EQ(mismatch->label, "counts");
    EXPECT_NE(mismatch->report.find("counts"), std::string::npos);
    EXPECT_NE(mismatch->report.find("INT32"), std::string::npos);
    EXPECT_NE(mismatch->report.find("validator_overrides"), std::string::npos)
        << "the operator has to be told which config section over-matched";
    EXPECT_NE(mismatch->report.find("allclose_matching_infinities"), std::string::npos)
        << "and which of the three validators that section named";
    // The only sentence in the message that says what to do about it.
    EXPECT_NE(mismatch->report.find("Narrow that entry's 'tensors' glob"), std::string::npos);
}

// makeValidator() has no device validator for this kind. The C++ graph harness builds its
// validators through it, so a device request is refused with a report.
TEST(TestOutputComparison, AllcloseMatchingInfinitiesValidatorIsRefusedOnTheDevice)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    EXPECT_NE(
        makeValidator(DataType::FLOAT, "y_out", exactMatchingInfinities(), ValidationSite::HOST)
            .validator,
        nullptr);

    const auto selection = makeValidator(
        DataType::FLOAT, "y_out", exactMatchingInfinities(), ValidationSite::DEVICE);
    EXPECT_EQ(selection.validator, nullptr);
    EXPECT_NE(selection.error.find("allclose_matching_infinities"), std::string::npos)
        << "the operator has to be told which validator could not be honoured";
    EXPECT_NE(selection.error.find("--validator cpu"), std::string::npos)
        << "and the one flag that makes the run grade on the host instead";
    EXPECT_EQ(selection.error.find("does not support this data type"), std::string::npos)
        << "the report must not blame the data type for a site refusal";
}

// Unlike RMS, this kind does not replace what decided the verdict: atol and rtol are
// exactly what graded every finite element, so the report keeps printing them.
TEST(TestOutputComparison, AllcloseMatchingInfinitiesFailureStillReportsAtolRtol)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, 1.0f, 2.0f, 3.0f);
    auto actual = floatTensor3(attrs, 1.0f, 2.5f, 3.0f);

    const auto mismatch
        = compareTensor(K_UID_B,
                        attrs,
                        *expected,
                        *actual,
                        ComparisonTolerance::allCloseMatchingInfinities(1e-3f, 1e-3f),
                        ValidationSite::HOST,
                        "b");

    ASSERT_TRUE(mismatch.has_value());
    EXPECT_NE(mismatch->report.find("atol="), std::string::npos);
    EXPECT_EQ(mismatch->report.find("relative RMS"), std::string::npos)
        << "no threshold decided this failure; reporting one would name a check that did not run";
}

// ---------------------------------------------------------------------------
// The non-finite mask (`metadata.validator: "match_nonfinite"`), layered on the resolved
// tolerance by withNonFiniteMask(). Host only.
// ---------------------------------------------------------------------------

namespace
{

ComparisonTolerance maskedExact()
{
    return hipdnn_integration_tests::bundle::withNonFiniteMask(
        exact(), hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
}

// A [1, 3, 1, 1] float tensor holding the unwritten-output sentinel bits at `position` and
// `value` elsewhere. Stored bitwise: a float NaN converted through arithmetic could lose the
// payload that identifies it.
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor> tensorWithSentinelAt(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs, int position, float value)
{
    auto tensor = floatTensor(attrs, value);
    const auto sentinel = hipdnn_test_sdk::utilities::payloadNanSentinelValue<float>();
    std::memcpy(static_cast<float*>(tensor->rawHostData()) + position, &sentinel, sizeof(float));
    tensor->markHostModified();
    return tensor;
}

} // namespace

// Bug caught: the mask never being consulted (a NaN that is right on both sides still fails),
// or being on for everyone (the same pair passing without it).
TEST(TestOutputComparison, NonFiniteMaskMatchesNaNAndSameSignedInfinitiesByPosition)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    // One finite element each: a reference with none is not a verified output (see below).
    auto expected = floatTensor3(attrs, K_NAN, K_INF, 1.0f);
    auto actual = floatTensor3(attrs, K_NAN, K_INF, 1.0f);
    auto negExpected = floatTensor3(attrs, -K_INF, 1.0f, K_NAN);
    auto negActual = floatTensor3(attrs, -K_INF, 1.0f, K_NAN);

    for(const auto& pair : {std::make_pair(expected.get(), actual.get()),
                            std::make_pair(negExpected.get(), negActual.get())})
    {
        EXPECT_FALSE(
            compareTensor(
                K_UID_B, attrs, *pair.first, *pair.second, maskedExact(), ValidationSite::HOST, "b")
                .has_value())
            << "NaN, +inf and -inf are right on both sides";
        EXPECT_TRUE(
            compareTensor(
                K_UID_B, attrs, *pair.first, *pair.second, exact(), ValidationSite::HOST, "b")
                .has_value())
            << "without the mask the same pair fails: the default grading is unchanged";
    }
}

// Bug caught: the mask forgiving a disagreement about where the non-finite values are.
TEST(TestOutputComparison, NonFiniteMaskStillRejectsEveryNonFiniteDisagreement)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    struct Case
    {
        float expected;
        float actual;
        const char* what;
    };
    const std::vector<Case> cases = {{K_NAN, 1.0f, "NaN reference, finite output"},
                                     {1.0f, K_NAN, "finite reference, NaN output"},
                                     {K_INF, 1.0f, "Inf reference, finite output"},
                                     {1.0f, K_INF, "finite reference, Inf output"},
                                     {K_INF, -K_INF, "+Inf against -Inf"},
                                     {K_NAN, K_INF, "NaN against Inf"},
                                     {K_INF, K_NAN, "Inf against NaN"}};
    for(const auto& c : cases)
    {
        // Two more finite elements, so the finite part is never empty and only this position
        // can decide the verdict.
        auto expected = floatTensor3(attrs, c.expected, 1.0f, 2.0f);
        auto actual = floatTensor3(attrs, c.actual, 1.0f, 2.0f);
        EXPECT_TRUE(
            compareTensor(
                K_UID_B, attrs, *expected, *actual, maskedExact(), ValidationSite::HOST, "b")
                .has_value())
            << c.what;
    }
}

// Bug caught: matched non-finite positions carrying a finite neighbour's drift through, or
// finite elements escaping the tolerance because the tensor also holds a NaN.
TEST(TestOutputComparison, NonFiniteMaskStillGradesFiniteElementsByTheResolvedTolerance)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, K_NAN, 1.0f, 2.0f);
    auto actual = floatTensor3(attrs, K_NAN, 1.1f, 2.0f);

    EXPECT_TRUE(
        compareTensor(K_UID_B, attrs, *expected, *actual, maskedExact(), ValidationSite::HOST, "b")
            .has_value());
    const auto loose = hipdnn_integration_tests::bundle::withNonFiniteMask(
        ComparisonTolerance::allClose(0.2f, 0.0f),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    EXPECT_FALSE(compareTensor(K_UID_B, attrs, *expected, *actual, loose, ValidationSite::HOST, "b")
                     .has_value())
        << "atol of the resolved tolerance still decides the finite elements";
}

// Bug caught: an engine that never wrote an element passing because NaN matches NaN. The element
// still holds the payload sentinel the harness filled the output with, and fails; a NaN the
// engine really computed (the default quiet NaN) passes.
TEST(TestOutputComparison, NonFiniteMaskFailsAnElementThatStillHoldsTheSentinel)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, K_NAN, 1.0f, 1.0f);
    auto unwritten = tensorWithSentinelAt(attrs, 0, 1.0f);
    auto computed = floatTensor3(attrs, K_NAN, 1.0f, 1.0f);

    const auto tolerance = maskedExact();
    ASSERT_TRUE(tolerance.unwrittenSentinelBits.has_value());
    EXPECT_TRUE(
        compareTensor(K_UID_B, attrs, *expected, *unwritten, tolerance, ValidationSite::HOST, "b")
            .has_value())
        << "an element holding the sentinel was never written";
    EXPECT_FALSE(
        compareTensor(K_UID_B, attrs, *expected, *computed, tolerance, ValidationSite::HOST, "b")
            .has_value())
        << "an ordinary NaN the engine computed is not the sentinel";
}

// Bug caught: a reference that is nothing but non-finite values passing (NaN matches NaN
// everywhere), which hides a run that produced garbage on both sides.
TEST(TestOutputComparison, NonFiniteMaskFailsAReferenceWithNoFiniteElement)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    // The control: the same two non-finite positions pass beside one finite element.
    auto expected = floatTensor3(attrs, K_NAN, K_INF, 1.0f);
    auto actual = floatTensor3(attrs, K_NAN, K_INF, 1.0f);
    EXPECT_FALSE(
        compareTensor(K_UID_B, attrs, *expected, *actual, maskedExact(), ValidationSite::HOST, "b")
            .has_value());

    auto nothingFinite = floatTensor3(attrs, K_NAN, K_INF, -K_INF);
    auto nothingFiniteToo = floatTensor3(attrs, K_NAN, K_INF, -K_INF);
    EXPECT_TRUE(compareTensor(K_UID_B,
                              attrs,
                              *nothingFinite,
                              *nothingFiniteToo,
                              maskedExact(),
                              ValidationSite::HOST,
                              "b")
                    .has_value())
        << "NaN, +inf and -inf on both sides, and nothing finite to grade";

    auto allNaNExpected = floatTensor(attrs, K_NAN);
    auto allNaNActual = floatTensor(attrs, K_NAN);
    EXPECT_TRUE(compareTensor(K_UID_B,
                              attrs,
                              *allNaNExpected,
                              *allNaNActual,
                              maskedExact(),
                              ValidationSite::HOST,
                              "b")
                    .has_value())
        << "an all-NaN reference is not a verified output";
}

// Bug caught: the mask replacing a per-tensor RMS override with allclose, or the RMS grade
// being skipped, or its normalizer inflated by the non-finite positions so a real error hides.
TEST(TestOutputComparison, NonFiniteMaskKeepsRmsGradingAndItsNormalizer)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    const auto rmsMasked = hipdnn_integration_tests::bundle::withNonFiniteMask(
        ComparisonTolerance::rms(K_RMS_THRESHOLD),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(rmsMasked.kind, ValidatorKind::RMS) << "the resolved kind stays in force";
    EXPECT_TRUE(rmsMasked.matchNonFinite);

    // Finite elements 1 and 2 plus a matched NaN: identical outputs pass, a 10% error on the
    // large element fails however many non-finite positions sit beside it.
    auto expected = floatTensor3(attrs, K_NAN, 1.0f, 2.0f);
    auto same = floatTensor3(attrs, K_NAN, 1.0f, 2.0f);
    auto drifted = floatTensor3(attrs, K_NAN, 1.0f, 2.2f);
    EXPECT_FALSE(
        compareTensor(K_UID_B, attrs, *expected, *same, rmsMasked, ValidationSite::HOST, "b")
            .has_value());
    EXPECT_TRUE(
        compareTensor(K_UID_B, attrs, *expected, *drifted, rmsMasked, ValidationSite::HOST, "b")
            .has_value())
        << "RMS over the finite elements is still enforced";

    auto infExpected = floatTensor3(attrs, K_INF, 1.0f, 2.0f);
    auto infDrifted = floatTensor3(attrs, K_INF, 1.0f, 2.2f);
    EXPECT_TRUE(compareTensor(
                    K_UID_B, attrs, *infExpected, *infDrifted, rmsMasked, ValidationSite::HOST, "b")
                    .has_value())
        << "an Inf position does not inflate the normalizer either";
}

// Bug caught: the mask switched on for an output it cannot grade, or the sentinel set for a
// type with no payload NaN.
TEST(TestOutputComparison, NonFiniteMaskIsOnlyLayeredOnFloatingPointOutputs)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_integration_tests::bundle::canMatchNonFinite;
    using hipdnn_integration_tests::bundle::withNonFiniteMask;

    for(const auto type : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE})
    {
        const auto masked = withNonFiniteMask(exact(), type);
        EXPECT_TRUE(canMatchNonFinite(type)) << EnumNameDataType(type);
        EXPECT_TRUE(masked.matchNonFinite) << EnumNameDataType(type);
        EXPECT_TRUE(masked.unwrittenSentinelBits.has_value()) << EnumNameDataType(type);
    }
    for(const auto type : {DataType::INT32,
                           DataType::INT8,
                           DataType::UINT8,
                           DataType::FP8_E4M3,
                           DataType::FP8_E5M2,
                           DataType::BOOLEAN})
    {
        const auto masked = withNonFiniteMask(exact(), type);
        EXPECT_FALSE(canMatchNonFinite(type)) << EnumNameDataType(type);
        EXPECT_FALSE(masked.matchNonFinite) << EnumNameDataType(type);
        EXPECT_FALSE(masked.unwrittenSentinelBits.has_value()) << EnumNameDataType(type);
    }
}

// Bug caught: the harness's own "can this output carry the mask" list drifting from the SDK's
// payload-sentinel list. A type matchable here without a payload sentinel would let an unwritten
// output pass as NaN matching NaN; one with a sentinel but not matchable would never be graded.
// Walks every DataType enumerator, so a type added later is covered without editing the test.
TEST(TestOutputComparison, NonFiniteMaskMatchableTypesAreExactlyThoseWithAPayloadSentinel)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_integration_tests::bundle::canMatchNonFinite;
    using hipdnn_integration_tests::bundle::withNonFiniteMask;

    std::vector<DataType> matchable;
    for(const auto type : hipdnn_flatbuffers_sdk::data_objects::EnumValuesDataType())
    {
        const bool hasSentinel
            = hipdnn_test_sdk::utilities::payloadNanSentinelBits(type).has_value();
        EXPECT_EQ(canMatchNonFinite(type), hasSentinel) << EnumNameDataType(type);
        EXPECT_EQ(withNonFiniteMask(exact(), type).matchNonFinite, hasSentinel)
            << EnumNameDataType(type);
        if(canMatchNonFinite(type))
        {
            matchable.push_back(type);
        }
    }

    EXPECT_EQ(matchable,
              (std::vector<DataType>{
                  DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE}));
}

// Bug caught: a device-site request served by the host validator (reading device memory
// through host pointers) instead of refused, as every other host-only validator is.
TEST(TestOutputComparison, NonFiniteMaskOnTheDeviceIsRefusedNotHostGraded)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, K_NAN, 1.0f, 2.0f);
    auto actual = floatTensor3(attrs, K_NAN, 1.0f, 2.0f);

    std::optional<hipdnn_integration_tests::bundle::TensorMismatch> mismatch;
    ASSERT_NO_THROW(mismatch = compareTensor(K_UID_B,
                                             attrs,
                                             *expected,
                                             *actual,
                                             maskedExact(),
                                             ValidationSite::DEVICE,
                                             "Bundle: b"));
    ASSERT_TRUE(mismatch.has_value());
    EXPECT_NE(mismatch->report.find("match_nonfinite"), std::string::npos)
        << "the operator has to be told which validator could not be honoured";
}

// Bug caught: a mask forced onto an integer output (which has no NaN) throwing out of the
// comparison instead of naming the tensor.
TEST(TestOutputComparison, NonFiniteMaskOnAnIntegerOutputIsReportedNotThrown)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_INT);

    auto expected = intTensor(attrs, 7);
    auto actual = intTensor(attrs, 7);
    auto tolerance = exact();
    tolerance.matchNonFinite = true;

    std::optional<hipdnn_integration_tests::bundle::TensorMismatch> mismatch;
    ASSERT_NO_THROW(
        mismatch = compareTensor(
            K_UID_INT, attrs, *expected, *actual, tolerance, ValidationSite::HOST, "Bundle: b"));
    ASSERT_TRUE(mismatch.has_value());
    EXPECT_NE(mismatch->report.find("counts"), std::string::npos);
    EXPECT_NE(mismatch->report.find("match_nonfinite"), std::string::npos);
}

// Bug caught: a failure report that lists the matched NaN positions as drift without saying
// which rule decided, sending a reader after elements that were fine.
TEST(TestOutputComparison, NonFiniteMaskFailureSaysWhichRuleGradedIt)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    auto expected = floatTensor3(attrs, K_NAN, 1.0f, 2.0f);
    auto actual = floatTensor3(attrs, K_NAN, 1.5f, 2.0f);
    const auto mismatch = compareTensor(
        K_UID_B, attrs, *expected, *actual, maskedExact(), ValidationSite::HOST, "b");
    ASSERT_TRUE(mismatch.has_value());
    EXPECT_NE(mismatch->report.find("match_nonfinite"), std::string::npos);
}

// Ragged Stats compare within one storage layout: [H,T] (L7a) and [T,H] (L7b) each match the
// same logical data and report a single differing in-block element. A pair whose layouts
// differ (origins, lengths or strides) is reported as a ragged layout mismatch.
TEST(TestOutputComparison, RaggedStatsCompareWithinEachStorageLayout)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);
    using namespace hipdnn_integration_tests::bundle;

    for(const auto& layout : {fixtures::raggedSdpaHeadTokenStatsCase().stats,
                              fixtures::raggedSdpaTokenHeadStatsCase().stats})
    {
        auto expected = raggedLogicalTensor(layout);
        auto same = raggedLogicalTensor(layout);
        auto different = raggedLogicalTensor(layout, /*perturb=*/true);

        EXPECT_FALSE(
            compareTensor(K_UID_A, attrs, *expected, *same, exact(), ValidationSite::HOST, "b")
                .has_value());
        EXPECT_TRUE(
            compareTensor(K_UID_A, attrs, *expected, *different, exact(), ValidationSite::HOST, "b")
                .has_value());
    }

    auto headToken = raggedLogicalTensor(fixtures::raggedSdpaHeadTokenStatsCase().stats);
    auto tokenHead = raggedLogicalTensor(fixtures::raggedSdpaTokenHeadStatsCase().stats);
    const auto mismatch
        = compareTensor(K_UID_A, attrs, *headToken, *tokenHead, exact(), ValidationSite::HOST, "b");
    ASSERT_TRUE(mismatch.has_value());
    EXPECT_NE(mismatch->report.find("ragged layout mismatch"), std::string::npos);
}

// A head-major output (L6, head pitch S_max*D) compares against itself and reports a
// differing in-block element.
TEST(TestOutputComparison, RaggedHeadMajorOutputCompares)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);
    using namespace hipdnn_integration_tests::bundle;

    const auto layout = fixtures::raggedSdpaHeadMajorCase().qkvo;

    auto expected = raggedLogicalTensor(layout);
    auto same = raggedLogicalTensor(layout);
    auto different = raggedLogicalTensor(layout, /*perturb=*/true);

    EXPECT_FALSE(compareTensor(K_UID_A, attrs, *expected, *same, exact(), ValidationSite::HOST, "b")
                     .has_value());
    EXPECT_TRUE(
        compareTensor(K_UID_A, attrs, *expected, *different, exact(), ValidationSite::HOST, "b")
            .has_value());
}

// The device validators index every padded position, which for a ragged tensor reads other
// batches' rows, so a device-site ragged comparison is refused with a pointer to the host.
TEST(TestOutputComparison, RaggedOutputOnDeviceIsReportedNotThrown)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    auto expected = raggedFloatTensor(3.5f);
    auto actual = raggedFloatTensor(3.5f);

    EXPECT_FALSE(
        compareTensor(K_UID_A, attrs, *expected, *actual, exact(), ValidationSite::HOST, "b")
            .has_value())
        << "on the host a ragged pair compares normally";

    std::optional<hipdnn_integration_tests::bundle::TensorMismatch> mismatch;
    ASSERT_NO_THROW(mismatch = compareTensor(
                        K_UID_A, attrs, *expected, *actual, exact(), ValidationSite::DEVICE, "b"));

    ASSERT_TRUE(mismatch.has_value());
    EXPECT_EQ(mismatch->uid, K_UID_A);
    EXPECT_NE(mismatch->report.find("y_out"), std::string::npos);
    EXPECT_NE(mismatch->report.find("--validator cpu"), std::string::npos);
}

// A dense row statistic whose rows past each batch's live length are NaN in both
// references compares only its live rows when the bundle declares them.
TEST(TestOutputComparison, LiveRowsSkipPositionsPastEachBatchLength)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {1, 0};

    auto expected = liveRowStats(K_STATS_STRIDES, lengths);
    auto actual = liveRowStats(K_STATS_STRIDES, lengths);

    EXPECT_TRUE(compareStats(wrapper, expected, actual, statsLiveRows(lengths)).empty());
    EXPECT_EQ(compareStats(wrapper, expected, actual, {}).size(), 1u)
        << "without live rows the NaN positions fail the dense comparison";
}

// makeLiveRowView()'s dtype switch (LiveRows.hpp) is otherwise exercised only at
// FLOAT; SDPA, the feature that introduced it, commonly produces HALF output.
TEST(TestOutputComparison, LiveRowsSkipPositionsPastEachBatchLengthHalf)
{
    using hipdnn_data_sdk::types::half;

    const auto buffer = makeGraphBufferWithStatsType("half");
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {1, 0};

    auto expected = liveRowStatsT<half>(K_STATS_STRIDES, lengths);
    auto actual = liveRowStatsT<half>(K_STATS_STRIDES, lengths);

    EXPECT_TRUE(compareStats(wrapper, expected, actual, statsLiveRows(lengths)).empty());
    EXPECT_EQ(compareStats(wrapper, expected, actual, {}).size(), 1u)
        << "without live rows the NaN positions fail the dense comparison";
}

TEST(TestOutputComparison, LiveRowMismatchStillFails)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {1, 0};

    auto expected = liveRowStats(K_STATS_STRIDES, lengths);
    auto actual = liveRowStats(K_STATS_STRIDES, lengths);
    hipdnn_data_sdk::utilities::TensorView<float>(*actual).getHostValue({0, 1, 0, 0}) = 99.0f;

    const auto mismatches = compareStats(wrapper, expected, actual, statsLiveRows(lengths));
    ASSERT_EQ(mismatches.size(), 1u);
    EXPECT_EQ(mismatches[0].uid, K_UID_STATS);
    EXPECT_NE(mismatches[0].report.find("stats"), std::string::npos);
}

// A value past the live length may differ: it is not compared.
TEST(TestOutputComparison, LiveRowPositionPastTheLengthIsNotCompared)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {1, 0};

    auto expected = liveRowStats(K_STATS_STRIDES, lengths);
    auto actual = liveRowStats(K_STATS_STRIDES, lengths);
    hipdnn_data_sdk::utilities::TensorView<float>(*actual).getHostValue({0, 1, 2, 0}) = 99.0f;
    hipdnn_data_sdk::utilities::TensorView<float>(*actual).getHostValue({1, 0, 0, 0}) = -K_INF;

    EXPECT_TRUE(compareStats(wrapper, expected, actual, statsLiveRows(lengths)).empty());
}

// With every length 0 no element is compared, and the comparison fails rather than
// passing with nothing checked.
TEST(TestOutputComparison, LiveRowAllZeroLengthsFailsBecauseNoElementIsCompared)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {0, 0};

    auto expected = liveRowStats(K_STATS_STRIDES, lengths);
    auto actual = liveRowStats(K_STATS_STRIDES, lengths);

    EXPECT_EQ(compareStats(wrapper, expected, actual, statsLiveRows(lengths)).size(), 1u);
}

TEST(TestOutputComparison, LiveRowTokenMajorStridesCompare)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {3, 2};

    auto expected = liveRowStats(K_STATS_TOKEN_MAJOR_STRIDES, lengths);
    auto same = liveRowStats(K_STATS_TOKEN_MAJOR_STRIDES, lengths);
    auto different = liveRowStats(K_STATS_TOKEN_MAJOR_STRIDES, lengths);
    hipdnn_data_sdk::utilities::TensorView<float>(*different).getHostValue({1, 2, 1, 0}) = 99.0f;

    EXPECT_TRUE(compareStats(wrapper, expected, same, statsLiveRows(lengths)).empty());
    EXPECT_EQ(compareStats(wrapper, expected, same, {}).size(), 1u);
    EXPECT_EQ(compareStats(wrapper, expected, different, statsLiveRows(lengths)).size(), 1u);
}

TEST(TestOutputComparison, LiveRowOutputOnDeviceIsReportedNotThrown)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {1, 0};

    auto expected = liveRowStats(K_STATS_STRIDES, lengths);
    auto actual = liveRowStats(K_STATS_STRIDES, lengths);

    std::vector<hipdnn_integration_tests::bundle::TensorMismatch> mismatches;
    ASSERT_NO_THROW(mismatches = compareStats(
                        wrapper, expected, actual, statsLiveRows(lengths), ValidationSite::DEVICE));

    ASSERT_EQ(mismatches.size(), 1u);
    EXPECT_EQ(mismatches[0].uid, K_UID_STATS);
    EXPECT_NE(mismatches[0].report.find("stats"), std::string::npos);
    EXPECT_NE(mismatches[0].report.find("--validator cpu"), std::string::npos);
}

TEST(TestOutputComparison, LiveRowEntryThatDoesNotFitTheOutputThrows)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    auto expected = liveRowStats(K_STATS_STRIDES, {1, 0});
    auto actual = liveRowStats(K_STATS_STRIDES, {1, 0});

    EXPECT_THROW(compareStats(wrapper, expected, actual, statsLiveRows({1})),
                 std::invalid_argument);
    EXPECT_THROW(compareStats(wrapper, expected, actual, statsLiveRows({1, 5})),
                 std::invalid_argument);
    EXPECT_THROW(compareStats(wrapper, expected, actual, statsLiveRows({-1, 0})),
                 std::invalid_argument);
}

// Values that are not NaN past the lengths: only the lengths keep them out of the
// comparison. Full lengths compare them, and no map compares them too.
TEST(TestOutputComparison, LiveRowValuesPastTheLengthThatAreNotNaNAreNotCompared)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {1, 0};

    struct PastValues
    {
        const char* name;
        float expected;
        float actual;
    };
    for(const auto& past : {PastValues{"inf_against_finite", -K_INF, 7.0f},
                            PastValues{"inf_against_opposite_inf", -K_INF, K_INF},
                            PastValues{"finite_against_finite", 5.0f, 6.0f}})
    {
        SCOPED_TRACE(past.name);
        const auto expected = liveRowStatsPast(lengths, past.expected);
        const auto actual = liveRowStatsPast(lengths, past.actual);

        EXPECT_TRUE(compareStats(wrapper, expected, actual, statsLiveRows(lengths)).empty());
        EXPECT_EQ(compareStats(wrapper, expected, actual, {}).size(), 1u);
        EXPECT_EQ(compareStats(wrapper, expected, actual, statsLiveRows({4, 4})).size(), 1u);
    }
}

// The two sides may have different strides: the views report the same layout.
TEST(TestOutputComparison, LiveRowSidesWithDifferentStridesCompare)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {3, 2};

    const auto expected = liveRowStats(K_STATS_STRIDES, lengths);
    const auto same = liveRowStats(K_STATS_TOKEN_MAJOR_STRIDES, lengths);
    const auto different = liveRowStats(K_STATS_TOKEN_MAJOR_STRIDES, lengths);
    hipdnn_data_sdk::utilities::TensorView<float>(*different).getHostValue({0, 2, 2, 0}) = 99.0f;

    EXPECT_TRUE(compareStats(wrapper, expected, same, statsLiveRows(lengths)).empty());
    EXPECT_EQ(compareStats(wrapper, expected, different, statsLiveRows(lengths)).size(), 1u);
}

// A run that compares the other outputs on the device can still compare a live-row
// output on the host.
TEST(TestOutputComparison, LiveRowSiteOverridesTheDeviceSite)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {1, 0};

    const auto expected = liveRowStats(K_STATS_STRIDES, lengths);
    const auto same = liveRowStats(K_STATS_STRIDES, lengths);
    const auto different = liveRowStats(K_STATS_STRIDES, lengths);
    hipdnn_data_sdk::utilities::TensorView<float>(*different).getHostValue({0, 1, 0, 0}) = 99.0f;

    EXPECT_TRUE(compareStats(wrapper,
                             expected,
                             same,
                             statsLiveRows(lengths),
                             ValidationSite::DEVICE,
                             ValidationSite::HOST)
                    .empty());
    EXPECT_EQ(compareStats(wrapper,
                           expected,
                           different,
                           statsLiveRows(lengths),
                           ValidationSite::DEVICE,
                           ValidationSite::HOST)
                  .size(),
              1u);
    const auto refused = compareStats(wrapper,
                                      expected,
                                      same,
                                      statsLiveRows(lengths),
                                      ValidationSite::HOST,
                                      ValidationSite::DEVICE);
    ASSERT_EQ(refused.size(), 1u);
    EXPECT_NE(refused[0].report.find("NOT COMPARABLE ON DEVICE"), std::string::npos);
}

// An output whose values are on the device is read back to the host before the live-row
// view reads it. The host copy holds zeros, so a stale read fails the live rows.
TEST(TestGpuOutputComparison, LiveRowDeviceResidentOutputComparesOnTheHost)
{
    SKIP_IF_NO_DEVICES();
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {1, 0};

    const auto expected = liveRowStats(K_STATS_STRIDES, lengths);
    const auto source = liveRowStats(K_STATS_STRIDES, lengths);
    const std::shared_ptr<hipdnn_data_sdk::utilities::ITensor> actual
        = std::make_shared<hipdnn_data_sdk::utilities::Tensor<float>>(
            std::vector<int64_t>{2, 3, 4, 1}, K_STATS_STRIDES);
    actual->fillTensorWithValue(0.0f);
    ASSERT_EQ(hipMemcpy(actual->rawDeviceData(),
                        source->rawHostData(),
                        source->elementSpace() * sizeof(float),
                        hipMemcpyHostToDevice),
              hipSuccess);
    actual->markDeviceModified();

    EXPECT_TRUE(compareStats(wrapper,
                             expected,
                             actual,
                             statsLiveRows(lengths),
                             ValidationSite::DEVICE,
                             ValidationSite::HOST)
                    .empty());
}

TEST(TestOutputComparison, LiveRowViewRejectsATypeThatIsNotFloatingPoint)
{
    hipdnn_data_sdk::utilities::Tensor<int32_t> tensor(std::vector<int64_t>{2, 3, 4, 1});
    EXPECT_THROW(hipdnn_integration_tests::bundle::detail::makeLiveRowView(
                     tensor,
                     hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                     hipdnn_integration_tests::bundle::LiveRows{2, {1, 0}}),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// resolveLiveRows: reads the lengths of a run, or takes them from a ragged tensor.
// ---------------------------------------------------------------------------

namespace
{

std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    int32Lengths(const std::vector<int32_t>& values)
{
    auto tensor = std::make_shared<hipdnn_data_sdk::utilities::Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(values.size()), 1, 1, 1});
    for(size_t i = 0; i < values.size(); ++i)
    {
        tensor->setHostValue(values[i], static_cast<int64_t>(i), 0, 0, 0);
    }
    return tensor;
}

hipdnn_integration_tests::OutputLiveRows statsByLengths(int64_t lengthsUid)
{
    hipdnn_integration_tests::OutputLiveRows entry;
    entry.uid = K_UID_STATS;
    entry.axis = 2;
    entry.lengthsUid = lengthsUid;
    return entry;
}

} // namespace

TEST(TestOutputComparison, LiveRowResolveReadsTheLengthsTensor)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    auto lengths = int32Lengths({4, 0});

    const auto map = hipdnn_integration_tests::bundle::resolveLiveRows(
        {statsByLengths(20)}, wrapper, [&](int64_t uid) {
            return uid == 20 ? lengths.get() : nullptr;
        });

    ASSERT_EQ(map.count(K_UID_STATS), 1u);
    EXPECT_EQ(map.at(K_UID_STATS).axis, 2);
    EXPECT_EQ(map.at(K_UID_STATS).lengths, (std::vector<int64_t>{4, 0}));
}

TEST(TestOutputComparison, LiveRowResolveReadsAnInt64LengthsTensor)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto lengths = std::make_shared<hipdnn_data_sdk::utilities::Tensor<int64_t>>(
        std::vector<int64_t>{2, 1, 1, 1});
    lengths->setHostValue(int64_t{3}, 0, 0, 0, 0);
    lengths->setHostValue(int64_t{1}, 1, 0, 0, 0);

    const auto map = hipdnn_integration_tests::bundle::resolveLiveRows(
        {statsByLengths(20)}, wrapper, [&](int64_t) { return lengths.get(); });

    EXPECT_EQ(map.at(K_UID_STATS).lengths, (std::vector<int64_t>{3, 1}));
}

// The axis is checked as a 64-bit value: 2^32 + 2 does not pass as axis 2.
TEST(TestOutputComparison, LiveRowResolveRejectsAnAxisPastTheRank)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto lengths = int32Lengths({1, 0});
    auto entry = statsByLengths(20);
    entry.axis = (int64_t{1} << 32) + 2;

    try
    {
        hipdnn_integration_tests::bundle::resolveLiveRows(
            {entry}, wrapper, [&](int64_t) { return lengths.get(); });
        ADD_FAILURE() << "an axis past the rank resolved";
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("axis 4294967298 is not in the range 1 to 3"),
                  std::string::npos)
            << e.what();
    }
}

TEST(TestOutputComparison, LiveRowResolveTakesTheLayoutOfALikeTensor)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    auto ragged = raggedFloatTensor(0.0f);
    hipdnn_integration_tests::OutputLiveRows entry;
    entry.uid = K_UID_STATS;
    entry.likeUid = 21;

    const auto map = hipdnn_integration_tests::bundle::resolveLiveRows(
        {entry}, wrapper, [&](int64_t uid) { return uid == 21 ? ragged.get() : nullptr; });

    ASSERT_EQ(map.count(K_UID_STATS), 1u);
    EXPECT_EQ(map.at(K_UID_STATS).axis, hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS);
    EXPECT_EQ(map.at(K_UID_STATS).lengths, (std::vector<int64_t>{2, 3}));
}

TEST(TestOutputComparison, LiveRowResolveRejectsALengthOutsideTheExtent)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    for(const auto& values : {std::vector<int32_t>{5, 0}, std::vector<int32_t>{0, -1}})
    {
        auto lengths = int32Lengths(values);
        try
        {
            hipdnn_integration_tests::bundle::resolveLiveRows(
                {statsByLengths(20)}, wrapper, [&](int64_t) { return lengths.get(); });
            ADD_FAILURE() << "a length outside 0 to 4 resolved";
        }
        catch(const std::runtime_error& e)
        {
            EXPECT_NE(std::string(e.what()).find("not in the range 0 to 4"), std::string::npos)
                << e.what();
        }
    }
}

TEST(TestOutputComparison, LiveRowResolveRejectsATensorWithNoData)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto noTensor = [](int64_t) -> hipdnn_data_sdk::utilities::ITensor* { return nullptr; };

    EXPECT_THROW(
        hipdnn_integration_tests::bundle::resolveLiveRows({statsByLengths(20)}, wrapper, noTensor),
        std::runtime_error);

    auto dense = liveRowStats(K_STATS_STRIDES, {1, 0});
    hipdnn_integration_tests::OutputLiveRows entry;
    entry.uid = K_UID_STATS;
    entry.likeUid = 21;
    EXPECT_THROW(hipdnn_integration_tests::bundle::resolveLiveRows(
                     {entry}, wrapper, [&](int64_t) { return dense.get(); }),
                 std::runtime_error)
        << "a like tensor that is not ragged has no live lengths";
}

// ---------------------------------------------------------------------------
// Ragged outputs at the host-only site, and same-signed infinities on the host.
// ---------------------------------------------------------------------------

namespace
{

// makeGraphBuffer() with y_out marked ragged, so compareOutputs() routes it as a ragged
// output.
flatbuffers::DetachedBuffer makeGraphBufferWithRaggedOutput()
{
    auto json = nlohmann::json::parse(K_GRAPH);
    for(auto& tensor : json.at("tensors"))
    {
        if(tensor.at("uid").get<int64_t>() == K_UID_A)
        {
            tensor["ragged_offset_tensor_uid"] = K_UID_INT;
        }
    }
    flatbuffers::DetachedBuffer buffer;
    EXPECT_TRUE(hipdnn_integration_tests::bundle::detail::buildGraphBuffer(json, buffer));
    return buffer;
}

std::vector<hipdnn_integration_tests::bundle::TensorMismatch>
    compareRagged(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                  const std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>& expected,
                  const std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>& actual,
                  ValidationSite site,
                  std::optional<ValidationSite> hostOnlySite)
{
    OutputTensors actualOutputs;
    actualOutputs[K_UID_A] = actual;
    return compareOutputs(
        wrapper,
        {K_UID_A},
        actualOutputs,
        [&](int64_t) -> hipdnn_data_sdk::utilities::ITensor& { return *expected; },
        [](const std::string&, auto) { return exact(); },
        site,
        "Bundle: b",
        {},
        hostOnlySite);
}

struct InfinityPair
{
    const char* name;
    float expected;
    float actual;
    bool matches;
};

const std::vector<InfinityPair> K_INFINITY_PAIRS = {{"+inf/+inf", K_INF, K_INF, true},
                                                    {"-inf/-inf", -K_INF, -K_INF, true},
                                                    {"+inf/-inf", K_INF, -K_INF, false},
                                                    {"-inf/+inf", -K_INF, K_INF, false},
                                                    {"inf/finite", -K_INF, 1.0f, false},
                                                    {"finite/inf", 1.0f, K_INF, false}};

} // namespace

// A GPU reference leaves a ragged output on the device. With the host-only site set
// (what --validator auto and cpu give), the ragged output is compared on the host.
TEST(TestOutputComparison, RaggedOutputIsComparedAtTheHostOnlySite)
{
    const auto buffer = makeGraphBufferWithRaggedOutput();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};

    const auto expected = raggedFloatTensor(3.5f);
    const auto same = raggedFloatTensor(3.5f);
    const auto different = raggedFloatTensor(4.5f);

    EXPECT_TRUE(compareRagged(wrapper, expected, same, ValidationSite::DEVICE, ValidationSite::HOST)
                    .empty());
    EXPECT_EQ(
        compareRagged(wrapper, expected, different, ValidationSite::DEVICE, ValidationSite::HOST)
            .size(),
        1u);

    const auto refused
        = compareRagged(wrapper, expected, same, ValidationSite::DEVICE, ValidationSite::DEVICE);
    ASSERT_EQ(refused.size(), 1u);
    EXPECT_NE(refused[0].report.find("Ragged output NOT COMPARABLE ON DEVICE"), std::string::npos);

    const auto noHostOnlySite
        = compareRagged(wrapper, expected, same, ValidationSite::DEVICE, std::nullopt);
    ASSERT_EQ(noHostOnlySite.size(), 1u)
        << "without a host-only site the ragged output stays at the device site";
}

// The engine output is on the device only, and its host copy holds stale values. The
// comparison copies it to the host before the host validator reads it.
TEST(TestGpuOutputComparison, RaggedDeviceResidentOutputComparesOnTheHost)
{
    SKIP_IF_NO_DEVICES();
    const auto buffer = makeGraphBufferWithRaggedOutput();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};

    const auto expected = raggedFloatTensor(3.5f);
    const auto actual = raggedFloatTensor(0.0f);
    const std::vector<float> values(actual->elementSpace(), 3.5f);
    ASSERT_EQ(hipMemcpy(actual->rawDeviceData(),
                        values.data(),
                        values.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              hipSuccess);
    actual->markDeviceModified();

    EXPECT_TRUE(
        compareRagged(wrapper, expected, actual, ValidationSite::DEVICE, ValidationSite::HOST)
            .empty());
}

// On the host, plain allclose matches two infinities of the same sign. Other pairs that
// hold an infinity, and NaN against NaN, still fail.
TEST(TestOutputComparison, HostAllcloseMatchesOnlySameSignedInfinities)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_B);

    for(const auto& pair : K_INFINITY_PAIRS)
    {
        SCOPED_TRACE(pair.name);
        auto expected = floatTensor3(attrs, pair.expected, 1.0f, 2.0f);
        auto actual = floatTensor3(attrs, pair.actual, 1.0f, 2.0f);
        EXPECT_EQ(
            compareTensor(K_UID_B, attrs, *expected, *actual, exact(), ValidationSite::HOST, "b")
                .has_value(),
            !pair.matches);
    }

    auto nanExpected = floatTensor3(attrs, K_NAN, 1.0f, 2.0f);
    auto nanActual = floatTensor3(attrs, K_NAN, 1.0f, 2.0f);
    EXPECT_TRUE(
        compareTensor(K_UID_B, attrs, *nanExpected, *nanActual, exact(), ValidationSite::HOST, "b")
            .has_value())
        << "NaN never matches";
}

// The live-row comparison grades its live positions in the same way. An expected NaN at
// a live position is not compared, so a NaN actual value is the failing case there.
TEST(TestOutputComparison, LiveRowHostAllcloseMatchesOnlySameSignedInfinities)
{
    const auto buffer = makeGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const std::vector<int64_t> lengths = {2, 1};

    for(const auto& pair : K_INFINITY_PAIRS)
    {
        SCOPED_TRACE(pair.name);
        auto expected = liveRowStats(K_STATS_STRIDES, lengths);
        auto actual = liveRowStats(K_STATS_STRIDES, lengths);
        hipdnn_data_sdk::utilities::TensorView<float>(*expected).getHostValue({0, 1, 1, 0})
            = pair.expected;
        hipdnn_data_sdk::utilities::TensorView<float>(*actual).getHostValue({0, 1, 1, 0})
            = pair.actual;
        EXPECT_EQ(compareStats(wrapper, expected, actual, statsLiveRows(lengths)).empty(),
                  pair.matches);
    }

    auto expected = liveRowStats(K_STATS_STRIDES, lengths);
    auto actual = liveRowStats(K_STATS_STRIDES, lengths);
    hipdnn_data_sdk::utilities::TensorView<float>(*actual).getHostValue({0, 1, 1, 0}) = K_NAN;
    EXPECT_EQ(compareStats(wrapper, expected, actual, statsLiveRows(lengths)).size(), 1u)
        << "a NaN in a live position of the actual output fails";
}

// NOLINTEND(readability-identifier-naming)

// ---------------------------------------------------------------------------
// Padded leading stride. A tensor whose rows are padded has gap elements between the end of a
// row and the start of the next; they belong to no element, so whatever an engine or a reference
// left there must never decide a verdict, at either site.
// ---------------------------------------------------------------------------

namespace
{

// The batchnorm graph of K_GRAPH with y_out (uid 5) given a padded leading stride: [2, 3, 4, 5]
// elements whose rows are 8 apart instead of 5: 120 elements in an allocation of 189, with 3 gap
// elements after every row but the last.
const std::string K_PADDED_GRAPH
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
      R"({"name": "y_out", "uid": 5, "strides": [96, 32, 8, 1], "dims": [2, 3, 4, 5], )"
      R"("data_type": "float", "virtual": false}], "io_data_type": "float", )"
      R"("compute_data_type": "float", "intermediate_data_type": "float", "name": ""})";

flatbuffers::DetachedBuffer makePaddedGraphBuffer()
{
    flatbuffers::DetachedBuffer buffer;
    EXPECT_TRUE(hipdnn_integration_tests::bundle::detail::buildGraphBuffer(
        nlohmann::json::parse(K_PADDED_GRAPH), buffer));
    return buffer;
}

// A y_out whose every element is `value` and whose every gap byte is `gapByte`, the gaps set
// before the elements are written and left as they are.
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    paddedFloatTensor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                      float value,
                      unsigned char gapByte)
{
    auto tensor = hipdnn_test_sdk::detail::createTensorFromAttribute(attrs);
    std::memset(tensor->rawHostData(), gapByte, tensor->elementSpace() * tensor->elementSize());
    tensor->fillTensorWithValue(value);
    return tensor;
}

} // namespace

// Bug caught: a comparison that walks the allocation (a linear fast path taken for a tensor
// that is not packed, or a validator that reads elementSpace) instead of the logical elements:
// two tensors that hold the same elements but different bytes in the gaps (NaN against zero) would
// disagree. The layout is [96, 32, 8, 1] over [2, 3, 4, 5], so it is padded in every row.
TEST_P(TestOutputComparisonSite, GapBytesOfAPaddedOutputAreNeverCompared)
{
    const auto buffer = makePaddedGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    auto expected = paddedFloatTensor(attrs, 3.5f, 0xFF); // NaN bytes in the gaps
    auto actual = paddedFloatTensor(attrs, 3.5f, 0x00); // zero bytes in the gaps
    ASSERT_FALSE(expected->isPacked());
    ASSERT_GT(expected->elementSpace(), expected->elementCount());

    EXPECT_FALSE(
        compareTensor(K_UID_A, attrs, *expected, *actual, exact(), GetParam(), "b").has_value());
    EXPECT_FALSE(
        compareTensor(K_UID_A, attrs, *actual, *expected, exact(), GetParam(), "b").has_value());
}

// Bug caught: the comparison stopping short of the allocation's end: the element at the largest
// offset (the last element of the last row) is the one a short walk drops, and a first-row
// element is the one a walk that pairs elements by position in the allocation misreads.
TEST_P(TestOutputComparisonSite, ADifferenceInTheLastAndTheFirstElementOfAPaddedOutputIsFound)
{
    const auto buffer = makePaddedGraphBuffer();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper{buffer.data(),
                                                                             buffer.size()};
    const auto& attrs = *wrapper.getTensorMap().at(K_UID_A);

    const std::vector<std::vector<int64_t>> positions = {{1, 2, 3, 4}, {0, 0, 0, 0}, {0, 1, 3, 4}};
    for(const auto& position : positions)
    {
        auto expected = paddedFloatTensor(attrs, 3.5f, 0xFF);
        auto actual = paddedFloatTensor(attrs, 3.5f, 0x00);
        hipdnn_data_sdk::utilities::TensorView<float> view(*actual);
        view.getHostValue(position) = 9.25f;
        actual->markHostModified();

        const auto mismatch
            = compareTensor(K_UID_A, attrs, *expected, *actual, exact(), GetParam(), "b");
        ASSERT_TRUE(mismatch.has_value())
            << "element at offset " << actual->getIndex(position) << " differs";
        EXPECT_NE(mismatch->report.find("9.25"), std::string::npos);
    }
}
