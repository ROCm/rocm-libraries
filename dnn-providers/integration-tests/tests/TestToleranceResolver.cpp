// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "harness/ValidationSite.hpp"
#include "harness/bundle/OutputComparison.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_flatbuffers_sdk::data_objects;
namespace tol = hipdnn_integration_tests::tolerance;
namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

// Tolerance functions only inspect attributes_type() per node — shapes and uids
// are irrelevant. This builder creates a minimal valid graph from just node types.
fb::GraphWrapper buildGraph(flatbuffers::FlatBufferBuilder& builder,
                            const std::vector<NodeAttributes>& nodeTypes)
{
    builder.Clear();

    const std::vector<int64_t> dims = {1};
    const std::vector<int64_t> strides = {1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "t", DataType::FLOAT, &strides, &dims));

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.reserve(nodeTypes.size());
    for(const auto attr : nodeTypes)
    {
        nodes.push_back(CreateNodeDirect(builder, "n", DataType::FLOAT, attr, 0));
    }

    auto graph = CreateGraphDirect(
        builder, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    builder.Finish(graph);
    return fb::GraphWrapper(builder.GetBufferPointer(), builder.GetSize());
}

} // namespace

// Known fp32 reference values from TestTolerances.hpp
constexpr float kConvFwdFp32 = 1e-5f;
constexpr float kBnInferenceFp32 = 2e-4f;
constexpr float kPointwiseFp32 = 1e-5f;
constexpr float kFallback = 1e-3f;

// ── Single non-Pointwise op: both policies agree ────────────────────────────

TEST(TestToleranceResolver, SingleConvFwdMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::ConvolutionFwdAttributes});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kConvFwdFp32);
}

TEST(TestToleranceResolver, SingleConvFwdOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::ConvolutionFwdAttributes});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kConvFwdFp32);
}

// ── Discriminating case: BN inference (2e-4) then conv fwd (1e-5) ───────────
// MAX picks the loose one (2e-4), OUTPUT_OP picks the last non-Pointwise (1e-5).

TEST(TestToleranceResolver, BnInferenceConvFwdMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(
        b,
        {NodeAttributes::BatchnormInferenceAttributes, NodeAttributes::ConvolutionFwdAttributes});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kBnInferenceFp32);
}

TEST(TestToleranceResolver, BnInferenceConvFwdOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(
        b,
        {NodeAttributes::BatchnormInferenceAttributes, NodeAttributes::ConvolutionFwdAttributes});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kConvFwdFp32);
}

// ── All-Pointwise: OUTPUT_OP falls back to MAX ──────────────────────────────

TEST(TestToleranceResolver, AllPointwiseMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::PointwiseAttributes});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kPointwiseFp32);
}

TEST(TestToleranceResolver, AllPointwiseOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::PointwiseAttributes});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kPointwiseFp32);
}

// ── Empty graph: both return 1e-3 floor ─────────────────────────────────────

TEST(TestToleranceResolver, EmptyGraphMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kFallback);
}

TEST(TestToleranceResolver, EmptyGraphOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kFallback);
}

// ── Unknown op: conservative 1e-3 fallback ──────────────────────────────────

TEST(TestToleranceResolver, UnknownOpMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::NONE});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kFallback);
}

TEST(TestToleranceResolver, UnknownOpOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::NONE});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kFallback);
}

// ── ResampleFwd / ResampleBwd: per-dtype table, not the 1e-3 fallback ──────
// fp32 1e-5, fp16 1e-3, bf16 1e-2 (resample::getTolerance<T>()). Unmapped, every
// dtype resolved to the 1e-3 fallback: fp32 100x too loose, bf16 10x too tight.

TEST(TestToleranceResolver, ResampleFwdUsesPerDtypeTolerance)
{
    EXPECT_FLOAT_EQ(tol::toleranceForNode(NodeAttributes::ResampleFwdAttributes, DataType::FLOAT),
                    1e-5f);
    EXPECT_FLOAT_EQ(tol::toleranceForNode(NodeAttributes::ResampleFwdAttributes, DataType::HALF),
                    1e-3f);
    EXPECT_FLOAT_EQ(
        tol::toleranceForNode(NodeAttributes::ResampleFwdAttributes, DataType::BFLOAT16), 1e-2f);
}

TEST(TestToleranceResolver, ResampleBwdUsesPerDtypeTolerance)
{
    EXPECT_FLOAT_EQ(tol::toleranceForNode(NodeAttributes::ResampleBwdAttributes, DataType::FLOAT),
                    1e-5f);
    EXPECT_FLOAT_EQ(tol::toleranceForNode(NodeAttributes::ResampleBwdAttributes, DataType::HALF),
                    1e-3f);
    EXPECT_FLOAT_EQ(
        tol::toleranceForNode(NodeAttributes::ResampleBwdAttributes, DataType::BFLOAT16), 1e-2f);
}

// ── fp8 outputs: one step of the output format, for every op ────────────────
// rtol is 2^-(mantissa bits), one step at the low end of a binade. atol is the
// subnormal step, 2^(1 - bias - mantissa bits).

template <typename T>
struct Fp8Format;

template <>
struct Fp8Format<hipdnn_data_sdk::types::fp8_e4m3>
{
    static constexpr DataType DATA_TYPE = DataType::FP8_E4M3;
    static constexpr float ATOL = 0x1p-9f;
    static constexpr float RTOL = 0x1p-3f;
};

template <>
struct Fp8Format<hipdnn_data_sdk::types::fp8_e4m3_fnuz>
{
    static constexpr DataType DATA_TYPE = DataType::FP8_E4M3_FNUZ;
    static constexpr float ATOL = 0x1p-10f;
    static constexpr float RTOL = 0x1p-3f;
};

template <>
struct Fp8Format<hipdnn_data_sdk::types::fp8_e5m2>
{
    static constexpr DataType DATA_TYPE = DataType::FP8_E5M2;
    static constexpr float ATOL = 0x1p-16f;
    static constexpr float RTOL = 0x1p-2f;
};

template <>
struct Fp8Format<hipdnn_data_sdk::types::fp8_e5m2_fnuz>
{
    static constexpr DataType DATA_TYPE = DataType::FP8_E5M2_FNUZ;
    static constexpr float ATOL = 0x1p-17f;
    static constexpr float RTOL = 0x1p-2f;
};

template <typename T>
class TestToleranceResolverFp8 : public ::testing::Test
{
protected:
    using Format = Fp8Format<T>;

    // The resolved tolerance, applied by the host allclose validator the harness uses.
    static bool passes(const tol::Tolerance& tolerance, T reference, T output)
    {
        hipdnn_data_sdk::utilities::Tensor<T> refTensor(std::vector<int64_t>{1});
        hipdnn_data_sdk::utilities::Tensor<T> outTensor(std::vector<int64_t>{1});
        refTensor.setHostValue(reference, 0);
        outTensor.setHostValue(output, 0);

        auto selection = hipdnn_integration_tests::bundle::makeValidator(
            Format::DATA_TYPE,
            "O",
            hipdnn_integration_tests::bundle::ComparisonTolerance::allClose(tolerance.atol,
                                                                            tolerance.rtol),
            hipdnn_integration_tests::ValidationSite::HOST);
        EXPECT_NE(selection.validator, nullptr) << selection.error;
        return selection.validator != nullptr
               && selection.validator->allClose(refTensor, outTensor);
    }

    // The value `steps` representable values away from zero from `value`, with its sign.
    // A negative `steps` moves toward zero.
    static T stepsFrom(T value, int steps)
    {
        const int magnitude = value.data & 0x7F;
        const int sign = value.data & 0x80;
        return T::from_bits(static_cast<uint8_t>(sign | (magnitude + steps)));
    }
};

using Fp8Types = ::testing::Types<hipdnn_data_sdk::types::fp8_e4m3,
                                  hipdnn_data_sdk::types::fp8_e4m3_fnuz,
                                  hipdnn_data_sdk::types::fp8_e5m2,
                                  hipdnn_data_sdk::types::fp8_e5m2_fnuz>;
TYPED_TEST_SUITE(TestToleranceResolverFp8, Fp8Types, );

TYPED_TEST(TestToleranceResolverFp8, ResolvesToOneStepOfTheFormat)
{
    using Format = typename TestFixture::Format;

    const auto direct = tol::fp8OutputTolerance(Format::DATA_TYPE);
    ASSERT_TRUE(direct.has_value());
    EXPECT_EQ(direct->atol, Format::ATOL);
    EXPECT_EQ(direct->rtol, Format::RTOL);

    flatbuffers::FlatBufferBuilder b;
    for(const auto& nodes : std::vector<std::vector<NodeAttributes>>{
            {NodeAttributes::SdpaAttributes},
            {NodeAttributes::ConvolutionWrwAttributes},
            {NodeAttributes::MatmulAttributes, NodeAttributes::PointwiseAttributes},
            {NodeAttributes::NONE},
            {}})
    {
        auto w = buildGraph(b, nodes);
        for(const auto policy :
            {tol::TolerancePolicy::MAX_ACROSS_NODES, tol::TolerancePolicy::OUTPUT_OP_TOLERANCE})
        {
            const auto value = tol::defaultTolerance(w, Format::DATA_TYPE, policy);
            EXPECT_EQ(value.atol, Format::ATOL);
            EXPECT_EQ(value.rtol, Format::RTOL);

            float atol = 0.0f;
            float rtol = 0.0f;
            tol::resolveTolerance(w, Format::DATA_TYPE, "", atol, rtol, policy);
            EXPECT_EQ(atol, Format::ATOL);
            EXPECT_EQ(rtol, Format::RTOL);
        }
    }
}

TYPED_TEST(TestToleranceResolverFp8, OneStepPassesAndTwoStepsFail)
{
    using T = TypeParam;
    using Format = typename TestFixture::Format;

    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::SdpaAttributes});
    const auto tolerance = tol::defaultTolerance(w, Format::DATA_TYPE);

    const T one(1.0f);
    const T topOfBinade = TestFixture::stepsFrom(one, static_cast<int>(1.0f / Format::RTOL) - 1);
    ASSERT_EQ(static_cast<float>(TestFixture::stepsFrom(topOfBinade, 1)), 2.0f);

    // Below the low end of a binade the step halves, so only references above it are
    // also checked toward zero.
    const std::vector<std::pair<T, bool>> references = {{T(0.0f), false},
                                                        {T::from_bits(2), true},
                                                        {one, false},
                                                        {T(1.5f), true},
                                                        {topOfBinade, true},
                                                        {T(64.0f), false},
                                                        {T(-1.5f), true}};
    for(const auto& [reference, checkTowardZero] : references)
    {
        const auto label = std::to_string(static_cast<float>(reference));
        const auto passesAt = [&tolerance, ref = reference](int steps) {
            return TestFixture::passes(tolerance, ref, TestFixture::stepsFrom(ref, steps));
        };

        EXPECT_TRUE(passesAt(0)) << label;
        EXPECT_TRUE(passesAt(1)) << label;
        EXPECT_FALSE(passesAt(2)) << label;
        if(checkTowardZero)
        {
            EXPECT_TRUE(passesAt(-1)) << label;
            EXPECT_FALSE(passesAt(-2)) << label;
        }
    }
}

// ── Non-fp8 outputs: the per-op value as both atol and rtol ─────────────────

TEST(TestToleranceResolver, NonFp8OutputsUsePerOpValueForAtolAndRtol)
{
    flatbuffers::FlatBufferBuilder b;
    auto sdpa = buildGraph(b, {NodeAttributes::SdpaAttributes});
    for(const auto& [dataType, expected] :
        std::vector<std::pair<DataType, float>>{{DataType::FLOAT, 1e-5f},
                                                {DataType::HALF, 1e-2f},
                                                {DataType::BFLOAT16, 1e-2f},
                                                {DataType::DOUBLE, 1e-5f},
                                                {DataType::INT32, kFallback},
                                                {DataType::FP8_E8M0, kFallback}})
    {
        EXPECT_FALSE(tol::fp8OutputTolerance(dataType).has_value());
        const auto value = tol::defaultTolerance(sdpa, dataType);
        EXPECT_FLOAT_EQ(value.atol, expected);
        EXPECT_FLOAT_EQ(value.rtol, expected);
    }

    flatbuffers::FlatBufferBuilder bnConvBuilder;
    auto bnConv = buildGraph(
        bnConvBuilder,
        {NodeAttributes::BatchnormInferenceAttributes, NodeAttributes::ConvolutionFwdAttributes});
    const auto maxValue
        = tol::defaultTolerance(bnConv, DataType::FLOAT, tol::TolerancePolicy::MAX_ACROSS_NODES);
    EXPECT_FLOAT_EQ(maxValue.atol, kBnInferenceFp32);
    EXPECT_FLOAT_EQ(maxValue.rtol, kBnInferenceFp32);
    const auto outputOpValue
        = tol::defaultTolerance(bnConv, DataType::FLOAT, tol::TolerancePolicy::OUTPUT_OP_TOLERANCE);
    EXPECT_FLOAT_EQ(outputOpValue.atol, kConvFwdFp32);
    EXPECT_FLOAT_EQ(outputOpValue.rtol, kConvFwdFp32);

    float atol = 0.0f;
    float rtol = 0.0f;
    tol::resolveTolerance(sdpa, DataType::HALF, "", atol, rtol);
    EXPECT_FLOAT_EQ(atol, 1e-2f);
    EXPECT_FLOAT_EQ(rtol, 1e-2f);
}

// NOLINTEND(readability-identifier-naming)
