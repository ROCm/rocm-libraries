// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENGINE_HIP_MLOPS_INGESTOR

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/HipMlopsTestSupport.hpp"

/**
 * @file TestHipMlopsBatchnormInferencePack.cpp
 * @brief The HIP MLOps engine on the kernel ingestor, batchnorm inference (ALMIOPEN-3073): the
 *        engine accepts what the hand-written builder accepts -- inference with an inverse
 *        variance or with a variance and epsilon, alone or followed by a ReLU -- and computes
 *        each through the engine as a CPU reference does.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing::hip_mlops;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr int64_t X_UID = 1;
constexpr int64_t Y_UID = 2;
constexpr int64_t MEAN_UID = 3;
constexpr int64_t VARIANCE_UID = 4; // the inverse variance, or the variance
constexpr int64_t SCALE_UID = 5;
constexpr int64_t BIAS_UID = 6;
constexpr int64_t EPSILON_UID = 7;
constexpr int64_t ACTIVATION_OUT_UID = 8; // fused: the activation's output

constexpr float CLIP = 0.75f;

enum class Activation
{
    NONE,
    RELU,
    CLIPPED_RELU,
    SIGMOID,
    LEAKY_RELU,
};

struct BatchnormGraph
{
    // A variance and epsilon (BatchnormInferenceAttributesVarianceExt) rather than the inverse
    // variance (BatchnormInferenceAttributes).
    bool withVariance = false;
    // N, C, then the spatial dimensions.
    std::vector<int64_t> dims = {2, 6, 5, 7};
    float epsilon = 1e-2f;
    Activation activation = Activation::NONE;
    bool overrideShape = false;
    bool xVirtual = false;
    bool meanVirtual = false;
    // A fused graph whose y is a graph output.
    bool intermediateNotVirtual = false;
    bool swapNodes = false;
    bool twoBatchnormNodes = false;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    data_objects::DataType yType = data_objects::DataType::FLOAT;
};

bool fused(const BatchnormGraph& spec)
{
    return spec.activation != Activation::NONE;
}

/// [1, C, 1, ...]: one value per channel.
std::vector<int64_t> channelDims(const BatchnormGraph& spec)
{
    std::vector<int64_t> dims(spec.dims.size(), 1);
    dims[1] = spec.dims[1];
    return dims;
}

flatbuffers::FlatBufferBuilder buildBatchnormGraph(const BatchnormGraph& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto ioStrides = packedStrides(spec.dims);
    const auto perChannelDims = channelDims(spec);
    const auto perChannelStrides = packedStrides(perChannelDims);
    const bool isFused = fused(spec);

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    const auto addTensor = [&](int64_t uid,
                               const std::vector<int64_t>& dims,
                               const std::vector<int64_t>& strides,
                               bool isVirtual = false,
                               data_objects::DataType type = data_objects::DataType::FLOAT) {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, uid, nullptr, type, &strides, &dims, isVirtual));
    };
    addTensor(X_UID, spec.dims, ioStrides, spec.xVirtual);
    addTensor(Y_UID, spec.dims, ioStrides, isFused && !spec.intermediateNotVirtual, spec.yType);
    addTensor(MEAN_UID, perChannelDims, perChannelStrides, spec.meanVirtual);
    addTensor(VARIANCE_UID, perChannelDims, perChannelStrides);
    addTensor(SCALE_UID, perChannelDims, perChannelStrides);
    addTensor(BIAS_UID, perChannelDims, perChannelStrides);
    if(isFused)
    {
        addTensor(ACTIVATION_OUT_UID, spec.dims, ioStrides);
    }
    if(spec.withVariance)
    {
        const std::vector<int64_t> scalarDims = {1};
        const data_objects::Float32Value epsilon(spec.epsilon);
        tensors.push_back(
            data_objects::CreateTensorAttributesDirect(builder,
                                                       EPSILON_UID,
                                                       nullptr,
                                                       data_objects::DataType::FLOAT,
                                                       &scalarDims,
                                                       &scalarDims,
                                                       false,
                                                       data_objects::TensorValue::Float32Value,
                                                       builder.CreateStruct(epsilon).Union()));
    }

    flatbuffers::Offset<void> batchnorm;
    data_objects::NodeAttributes type{};
    if(spec.withVariance)
    {
        batchnorm
            = data_objects::CreateBatchnormInferenceAttributesVarianceExt(
                  builder, X_UID, MEAN_UID, VARIANCE_UID, SCALE_UID, BIAS_UID, Y_UID, EPSILON_UID)
                  .Union();
        type = data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt;
    }
    else
    {
        batchnorm = data_objects::CreateBatchnormInferenceAttributes(
                        builder, X_UID, MEAN_UID, VARIANCE_UID, SCALE_UID, BIAS_UID, Y_UID)
                        .Union();
        type = data_objects::NodeAttributes::BatchnormInferenceAttributes;
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder, "batchnorm", spec.computeType, type, batchnorm));
    if(isFused)
    {
        data_objects::PointwiseAttributesBuilder activation(builder);
        activation.add_operation(spec.activation == Activation::SIGMOID
                                     ? data_objects::PointwiseMode::SIGMOID_FWD
                                     : data_objects::PointwiseMode::RELU_FWD);
        activation.add_in_0_tensor_uid(Y_UID);
        activation.add_out_0_tensor_uid(ACTIVATION_OUT_UID);
        if(spec.activation == Activation::CLIPPED_RELU)
        {
            activation.add_relu_upper_clip(CLIP);
        }
        if(spec.activation == Activation::LEAKY_RELU)
        {
            activation.add_relu_lower_clip_slope(0.1f);
        }
        const auto activationAttributes = activation.Finish().Union();
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "activation",
                                           spec.computeType,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           activationAttributes));
    }
    if(spec.twoBatchnormNodes)
    {
        nodes.push_back(data_objects::CreateNodeDirect(
            builder, "batchnorm2", spec.computeType, type, batchnorm));
    }
    if(spec.swapNodes)
    {
        std::reverse(nodes.begin(), nodes.end());
    }

    auto name = builder.CreateString("hip_mlops_batchnorm_inference");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    graphBuilder.add_is_override_shape_enabled(spec.overrideShape);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

auto match(const BatchnormGraph& spec)
{
    return matchGraph(buildBatchnormGraph(spec));
}

BatchnormGraph edited(void (*edit)(BatchnormGraph&))
{
    BatchnormGraph spec;
    edit(spec);
    return spec;
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

TEST(TestHipMlopsBatchnormInferenceGraphMatch, RoutesEachFormToItsPackFusedOrNot)
{
    EXPECT_EQ(boundOperation(match({})), "BATCHNORM_INFERENCE");
    EXPECT_EQ(boundOperation(match(edited([](auto& s) { s.activation = Activation::RELU; }))),
              "BATCHNORM_INFERENCE");
    EXPECT_EQ(boundOperation(match(edited([](auto& s) { s.withVariance = true; }))),
              "BATCHNORM_INFERENCE_VARIANCE");
    EXPECT_EQ(boundOperation(match(edited([](auto& s) {
                  s.withVariance = true;
                  s.activation = Activation::CLIPPED_RELU;
              }))),
              "BATCHNORM_INFERENCE_VARIANCE");
    // 3D and 5D as well as 4D.
    EXPECT_EQ(boundOperation(match(edited([](auto& s) { s.dims = {3, 8, 40}; }))),
              "BATCHNORM_INFERENCE");
    EXPECT_EQ(boundOperation(match(edited([](auto& s) { s.dims = {2, 4, 3, 5, 6}; }))),
              "BATCHNORM_INFERENCE");
}

struct RefusalCase
{
    const char* name;
    BatchnormGraph spec;
};

class TestHipMlopsBatchnormInferenceRefusal : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestHipMlopsBatchnormInferenceRefusal, Declines)
{
    EXPECT_FALSE(match(GetParam().spec).has_value());
}

// The hand-written builder's refusals.
INSTANTIATE_TEST_SUITE_P(
    ,
    TestHipMlopsBatchnormInferenceRefusal,
    ::testing::Values(
        RefusalCase{"OverrideShapes", edited([](auto& s) { s.overrideShape = true; })},
        RefusalCase{"HalfCompute",
                    edited([](auto& s) { s.computeType = data_objects::DataType::HALF; })},
        RefusalCase{"VarianceHalfCompute", edited([](auto& s) {
                        s.withVariance = true;
                        s.computeType = data_objects::DataType::HALF;
                    })},
        RefusalCase{"FusedHalfCompute", edited([](auto& s) {
                        s.activation = Activation::RELU;
                        s.computeType = data_objects::DataType::HALF;
                    })},
        RefusalCase{"VirtualInput", edited([](auto& s) { s.xVirtual = true; })},
        RefusalCase{"VirtualMean", edited([](auto& s) { s.meanVirtual = true; })},
        RefusalCase{"VarianceVirtualMean", edited([](auto& s) {
                        s.withVariance = true;
                        s.meanVirtual = true;
                    })},
        // y may be wider than x but not narrower.
        RefusalCase{"HalfOutputOfFloatInput",
                    edited([](auto& s) { s.yType = data_objects::DataType::HALF; })},
        RefusalCase{"TwoBatchnormNodes", edited([](auto& s) { s.twoBatchnormNodes = true; })},
        RefusalCase{"FusedNodesSwapped", edited([](auto& s) {
                        s.activation = Activation::RELU;
                        s.swapNodes = true;
                    })},
        RefusalCase{"FusedIntermediateNotVirtual", edited([](auto& s) {
                        s.activation = Activation::RELU;
                        s.intermediateNotVirtual = true;
                    })},
        RefusalCase{"VarianceFusedIntermediateNotVirtual", edited([](auto& s) {
                        s.withVariance = true;
                        s.activation = Activation::RELU;
                        s.intermediateNotVirtual = true;
                    })},
        RefusalCase{"SigmoidActivation",
                    edited([](auto& s) { s.activation = Activation::SIGMOID; })},
        RefusalCase{"LeakyRelu", edited([](auto& s) { s.activation = Activation::LEAKY_RELU; })}),
    [](const ::testing::TestParamInfo<RefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Numerics through the engine
// ---------------------------------------------------------------------------

double activate(Activation activation, double value)
{
    switch(activation)
    {
    case Activation::RELU:
        return std::max(value, 0.0);
    case Activation::CLIPPED_RELU:
        return std::min(static_cast<double>(CLIP), std::max(value, 0.0));
    default:
        return value;
    }
}

/// y = scale * (x - mean) * r + bias per channel, r the inverse variance or
/// 1 / sqrt(variance + epsilon), then the activation.
std::vector<float> reference(const BatchnormGraph& spec,
                             const std::vector<float>& x,
                             const std::vector<float>& mean,
                             const std::vector<float>& variance,
                             const std::vector<float>& scale,
                             const std::vector<float>& bias)
{
    const auto batches = static_cast<size_t>(spec.dims[0]);
    const auto channels = static_cast<size_t>(spec.dims[1]);
    const size_t spatial = x.size() / (batches * channels);
    std::vector<float> y(x.size());
    for(size_t n = 0; n < batches; ++n)
    {
        for(size_t c = 0; c < channels; ++c)
        {
            const double r = spec.withVariance
                                 ? 1.0
                                       / std::sqrt(static_cast<double>(variance[c])
                                                   + static_cast<double>(spec.epsilon))
                                 : static_cast<double>(variance[c]);
            for(size_t i = 0; i < spatial; ++i)
            {
                const size_t at = (n * channels + c) * spatial + i;
                const double value
                    = static_cast<double>(scale[c])
                          * (static_cast<double>(x[at]) - static_cast<double>(mean[c])) * r
                      + static_cast<double>(bias[c]);
                y[at] = static_cast<float>(activate(spec.activation, value));
            }
        }
    }
    return y;
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    size_t failures = 0;
    for(size_t i = 0; i < actual.size() && failures < 10; ++i)
    {
        const float tolerance = 1e-5f * (1.0f + std::fabs(expected[i]));
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "element " << i;
        failures += std::fabs(actual[i] - expected[i]) > tolerance ? 1 : 0;
    }
}

void checkInference(const BatchnormGraph& spec)
{
    const size_t count = elementCount(spec.dims);
    const auto channels = static_cast<size_t>(spec.dims[1]);
    const auto x = deterministicValues(count, 0.4f);
    // Per channel: a mean near zero, a positive variance (or inverse), scale near one.
    auto mean = deterministicValues(channels, 1.3f);
    auto variance = deterministicValues(channels, 2.1f);
    auto scale = deterministicValues(channels, 0.7f);
    const auto bias = deterministicValues(channels, 2.9f);
    for(size_t c = 0; c < channels; ++c)
    {
        mean[c] *= 0.25f;
        variance[c] = 1.0f + 0.4f * variance[c];
        scale[c] = 1.0f + 0.25f * scale[c];
    }
    const auto expected = reference(spec, x, mean, variance, scale, bias);

    const DeviceBuffer<float> xBuffer(count, x);
    const DeviceBuffer<float> yBuffer(count);
    const DeviceBuffer<float> meanBuffer(channels, mean);
    const DeviceBuffer<float> varianceBuffer(channels, variance);
    const DeviceBuffer<float> scaleBuffer(channels, scale);
    const DeviceBuffer<float> biasBuffer(channels, bias);
    const std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{X_UID, xBuffer.get()},
           {fused(spec) ? ACTIVATION_OUT_UID : Y_UID, yBuffer.get()},
           {MEAN_UID, meanBuffer.get()},
           {VARIANCE_UID, varianceBuffer.get()},
           {SCALE_UID, scaleBuffer.get()},
           {BIAS_UID, biasBuffer.get()}};

    EngineRun run;
    ASSERT_NO_FATAL_FAILURE(executeThroughTheEngine(buildBatchnormGraph(spec), buffers, &run));
    EXPECT_EQ(run.workspaceBytes, 0U);
    EXPECT_EQ(run.maxWorkspaceBytes, 0U);
    expectNear(yBuffer.read(), expected);
}

TEST(TestHipMlopsBatchnormInferenceDevice, EachFormMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    for(const bool withVariance : {false, true})
    {
        SCOPED_TRACE(withVariance ? "variance and epsilon" : "inverse variance");
        BatchnormGraph spec;
        spec.withVariance = withVariance;
        checkInference(spec);
    }
}

TEST(TestHipMlopsBatchnormInferenceDevice, FusedFormsApplyTheActivation)
{
    SKIP_IF_NO_DEVICES();

    // The clipped ReLU's clip is a launch argument rather than a compile option.
    for(const bool withVariance : {false, true})
    {
        for(const auto activation : {Activation::RELU, Activation::CLIPPED_RELU})
        {
            SCOPED_TRACE(withVariance ? "variance and epsilon" : "inverse variance");
            SCOPED_TRACE(activation == Activation::RELU ? "relu" : "clipped relu");
            BatchnormGraph spec;
            spec.withVariance = withVariance;
            spec.activation = activation;
            checkInference(spec);
        }
    }
}

TEST(TestHipMlopsBatchnormInferenceDevice, ThreeAndFiveDimensionalInputsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& dims : {std::vector<int64_t>{3, 8, 40}, std::vector<int64_t>{2, 4, 3, 5, 6}})
    {
        SCOPED_TRACE(dims.size());
        BatchnormGraph spec;
        spec.dims = dims;
        spec.withVariance = true;
        checkInference(spec);
    }
}

} // namespace

#endif // HIPDNN_ENGINE_HIP_MLOPS_INGESTOR
