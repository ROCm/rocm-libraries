// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The runner grades its outputs with bundle::compareTensor, directly on dense outputs and
// through raggedView on ragged ones. Two infinities of the same sign match; NaN, opposite
// signs, an infinity against a finite value and finite drift past the tolerance do not.

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "RaggedCompareView.hpp"
#include "harness/bundle/OutputComparison.hpp"

namespace
{

using hipdnn_data_sdk::utilities::ITensor;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using hipdnn_integration_tests::ValidationSite;
using hipdnn_integration_tests::bundle::compareTensor;
using hipdnn_integration_tests::bundle::ComparisonTolerance;

constexpr float K_INF = std::numeric_limits<float>::infinity();
constexpr float K_NAN = std::numeric_limits<float>::quiet_NaN();
constexpr int64_t K_UID = 3;

// Two sequences in 4-token slots of a [B 2, H 1, S 4, D 1] output; sequence 0 holds 3 live
// tokens and sequence 1 holds 1. Position (b 0, s 1) is live in both views.
const std::vector<int64_t> DIMS = {2, 1, 4, 1};
const std::vector<int64_t> STRIDES = {4, 4, 1, 1};
constexpr int64_t PROBE = 1;

struct Pair
{
    const char* name;
    float expected;
    float actual;
    bool matches;
};

const std::vector<Pair> PAIRS = {{"+inf/+inf", K_INF, K_INF, true},
                                 {"-inf/-inf", -K_INF, -K_INF, true},
                                 {"+inf/-inf", K_INF, -K_INF, false},
                                 {"-inf/+inf", -K_INF, K_INF, false},
                                 {"inf/finite", -K_INF, 1.0f, false},
                                 {"finite/inf", 1.0f, K_INF, false},
                                 {"NaN/NaN", K_NAN, K_NAN, false},
                                 {"finite drift", 1.0f, 1.1f, false}};

// The attributes of a FLOAT output named "o" with the dims above.
class OutputAttributes
{
public:
    OutputAttributes()
    {
        _builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
            _builder, K_UID, "o", DataType::FLOAT, &STRIDES, &DIMS));
    }

    const TensorAttributes& get() const
    {
        return *flatbuffers::GetRoot<TensorAttributes>(_builder.GetBufferPointer());
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
};

// Every element holds 0.5, except the probe position, which holds `probe`.
std::shared_ptr<Tensor<float>> output(float probe)
{
    auto tensor = std::make_shared<Tensor<float>>(DIMS, STRIDES);
    auto* data = static_cast<float*>(tensor->rawHostData());
    for(int64_t i = 0; i < DIMS[0] * STRIDES[0]; ++i)
    {
        data[i] = 0.5f;
    }
    data[PROBE] = probe;
    tensor->markHostModified();
    return tensor;
}

std::shared_ptr<ITensor> int32Tensor(const std::vector<int32_t>& values)
{
    auto tensor = std::make_shared<Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(values.size()), 1, 1, 1},
        std::vector<int64_t>{1, 1, 1, 1});
    auto* data = static_cast<int32_t*>(tensor->rawHostData());
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = values[i];
    }
    return tensor;
}

const std::vector<ComparisonTolerance> TOLERANCES
    = {ComparisonTolerance::allClose(1e-3f, 1e-3f),
       ComparisonTolerance::allCloseMatchingInfinities(1e-3f, 1e-3f)};

void expectPairsAtDenseSite(ValidationSite site)
{
    const OutputAttributes attrs;
    for(const auto& tolerance : TOLERANCES)
    {
        for(const auto& pair : PAIRS)
        {
            SCOPED_TRACE(pair.name);
            auto expected = output(pair.expected);
            auto actual = output(pair.actual);
            EXPECT_EQ(
                compareTensor(K_UID, attrs.get(), *expected, *actual, tolerance, site, "runner")
                    .has_value(),
                !pair.matches);
        }
    }
}

} // namespace

TEST(TestRunnerInfinityCompare, DenseOutputOnTheHost)
{
    expectPairsAtDenseSite(ValidationSite::HOST);
}

TEST(TestRunnerInfinityCompare, DenseOutputOnTheDevice)
{
    SKIP_IF_NO_DEVICES();
    expectPairsAtDenseSite(ValidationSite::DEVICE);
}

// A ragged output reads NaN in its expected slack as "not compared", so the NaN pair puts the
// NaN on the actual side only.
TEST(TestRunnerInfinityCompare, RaggedOutputThroughTheRaggedView)
{
    const OutputAttributes attrs;
    const auto offsets = int32Tensor({0, 4, 8});
    const auto lengths = int32Tensor({3, 1});
    for(const auto& tolerance : TOLERANCES)
    {
        for(auto pair : PAIRS)
        {
            SCOPED_TRACE(pair.name);
            if(std::isnan(pair.expected))
            {
                pair.expected = 1.0f;
            }
            auto expected = output(pair.expected);
            auto actual = output(pair.actual);
            auto expectedView
                = sdpa_case_flow::raggedView(*expected, DataType::FLOAT, offsets, 1, lengths);
            auto actualView
                = sdpa_case_flow::raggedView(*actual, DataType::FLOAT, offsets, 1, lengths);
            EXPECT_EQ(compareTensor(K_UID,
                                    attrs.get(),
                                    *expectedView,
                                    *actualView,
                                    tolerance,
                                    ValidationSite::HOST,
                                    "runner")
                          .has_value(),
                      !pair.matches);
        }
    }
}
