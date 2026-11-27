// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/hip_mlops/HipMlopsTestSupport.hpp"

/**
 * @file TestHipMlopsBatchnormInferencePack.cpp
 * @brief The HIP MLOps batchnorm inference engine on the kernel ingestor (ALMIOPEN-3073):
 *        hipkernel:BatchnormInference accepts what the batchnorm applicability checks accept --
 *        inference with an inverse variance or with a variance and epsilon, one pack each,
 *        alone or followed by a ReLU -- and computes each as a CPU reference does.
 *
 *        Each pack ships a kernel per vector width (ALMIOPEN-3066), addressed by the knob
 *        `vector_size`: it offers the widths dividing the vectorized dimension, the inference
 *        plans' choice ranks first, and every width, pinned through the knob, computes as the
 *        reference does.
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
    // x and y channel-last (NLC, NHWC, NDHWC) rather than channel-first.
    bool channelLast = false;
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
    const auto ioStrides
        = spec.channelLast ? channelLastStrides(spec.dims) : packedStrides(spec.dims);
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

/// The engine that serves @p spec, in either form.
const MlopsEngine& engineFor(const BatchnormGraph& /*spec*/)
{
    return BATCHNORM_INFERENCE_ENGINE;
}

auto match(const BatchnormGraph& spec)
{
    return matchGraph(engineFor(spec), buildBatchnormGraph(spec));
}

/// @p spec's graph reaches its engine alone, which binds @p operation.
void expectServedAlone(const BatchnormGraph& spec, const char* operation)
{
    EXPECT_THAT(acceptingEngines(buildBatchnormGraph(spec)),
                ::testing::ElementsAre(engineFor(spec).name));
    EXPECT_EQ(boundOperation(match(spec)), operation);
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

TEST(TestHipMlopsBatchnormInferencePack, IsOfferedUnderItsOwnNameAndIdWithAPackPerForm)
{
    expectOfferedUnderItsOwnNameAndId(BATCHNORM_INFERENCE_ENGINE);
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

TEST(TestHipMlopsBatchnormInferenceGraphMatch, RoutesEachFormToItsPackOfItsEngineAloneFusedOrNot)
{
    expectServedAlone({}, "BATCHNORM_INFERENCE");
    expectServedAlone(edited([](auto& s) { s.activation = Activation::RELU; }),
                      "BATCHNORM_INFERENCE");
    expectServedAlone(edited([](auto& s) { s.withVariance = true; }),
                      "BATCHNORM_INFERENCE_VARIANCE");
    expectServedAlone(edited([](auto& s) {
                          s.withVariance = true;
                          s.activation = Activation::CLIPPED_RELU;
                      }),
                      "BATCHNORM_INFERENCE_VARIANCE");
    // 3D and 5D as well as 4D.
    expectServedAlone(edited([](auto& s) { s.dims = {3, 8, 40}; }), "BATCHNORM_INFERENCE");
    expectServedAlone(edited([](auto& s) { s.dims = {2, 4, 3, 5, 6}; }), "BATCHNORM_INFERENCE");
}

struct RefusalCase
{
    const char* name;
    BatchnormGraph spec;
};

class TestHipMlopsBatchnormInferenceRefusal : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestHipMlopsBatchnormInferenceRefusal, EveryEngineDeclines)
{
    EXPECT_THAT(acceptingEngines(buildBatchnormGraph(GetParam().spec)), ::testing::IsEmpty());
}

// The applicability checks' refusals.
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
// Knob: one kernel per vector width
// ---------------------------------------------------------------------------

const std::vector<std::string> KNOBS = {"vector_size"};

BatchnormGraph shape(bool withVariance, std::vector<int64_t> dims, bool channelLast = false)
{
    BatchnormGraph spec;
    spec.withVariance = withVariance;
    spec.dims = std::move(dims);
    spec.channelLast = channelLast;
    return spec;
}

auto ranked(const BatchnormGraph& spec)
{
    return rankedKernels(engineFor(spec), buildBatchnormGraph(spec));
}

TEST(TestHipMlopsBatchnormInferenceKnobs, EachPackShipsAKernelPerVectorWidth)
{
    EXPECT_EQ(engineKnobs(BATCHNORM_INFERENCE_ENGINE), KNOBS);

    const auto kernels = loadedKernels(BATCHNORM_INFERENCE_ENGINE);
    EXPECT_EQ(kernels.size(), 6U);
    // The operation tells the packs' kernels apart, so no two share a KMD tuple.
    std::set<std::pair<std::string, int64_t>> tuples;
    for(const auto& kernel : kernels)
    {
        const auto& operation = kernel.getStringMetadata("operation");
        EXPECT_TRUE(tuples.emplace(operation, kernel.getIntMetadata("vector_size")).second)
            << kernel.name;
        EXPECT_EQ(kernel.source.entryPoint,
                  operation == "BATCHNORM_INFERENCE" ? "BatchNormFwdInferSpatialEstInvVar"
                                                     : "BatchNormFwdInferSpatialEst");
    }
    const std::set<std::pair<std::string, int64_t>> expected = {
        {"BATCHNORM_INFERENCE", 1},
        {"BATCHNORM_INFERENCE", 2},
        {"BATCHNORM_INFERENCE", 4},
        {"BATCHNORM_INFERENCE_VARIANCE", 1},
        {"BATCHNORM_INFERENCE_VARIANCE", 2},
        {"BATCHNORM_INFERENCE_VARIANCE", 4},
    };
    EXPECT_EQ(tuples, expected);
}

struct WidthCase
{
    const char* name;
    std::vector<int64_t> dims;
    bool channelLast;
    // The widths the knob offers, the inference plans' choice (the widest) first.
    std::vector<int64_t> widths;
};

class TestHipMlopsBatchnormInferenceWidths : public ::testing::TestWithParam<WidthCase>
{
};

// computeVectorSize()'s branches: C for a channel-last layout, the spatial size otherwise,
// divisible by 4, by 2 only, or by neither.
TEST_P(TestHipMlopsBatchnormInferenceWidths, TheKnobOffersTheWidthsDividingTheVectorizedDimension)
{
    for(const bool withVariance : {false, true})
    {
        SCOPED_TRACE(withVariance ? "variance and epsilon" : "inverse variance");
        const auto kernels = ranked(shape(withVariance, GetParam().dims, GetParam().channelLast));
        expectUniqueTuples(kernels, KNOBS);
        // Ranked widest first: the inference plans' choice, then the narrower.
        EXPECT_EQ(knobValues(kernels, "vector_size"), GetParam().widths);
        for(const auto& kernel : kernels)
        {
            EXPECT_EQ(kernel.getStringMetadata("operation"),
                      withVariance ? "BATCHNORM_INFERENCE_VARIANCE" : "BATCHNORM_INFERENCE");
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestHipMlopsBatchnormInferenceWidths,
    ::testing::Values(WidthCase{"NchwSpatialBy4", {2, 6, 2, 4}, false, {4, 2, 1}},
                      WidthCase{"NchwSpatialBy2", {2, 6, 2, 3}, false, {2, 1}},
                      WidthCase{"NchwSpatialOdd", {2, 6, 5, 7}, false, {1}},
                      WidthCase{"NhwcChannelsBy4", {2, 8, 5, 7}, true, {4, 2, 1}},
                      WidthCase{"NhwcChannelsBy2", {2, 6, 5, 7}, true, {2, 1}},
                      WidthCase{"NhwcChannelsOdd", {2, 3, 4, 4}, true, {1}},
                      WidthCase{"NclLengthBy4", {3, 8, 40}, false, {4, 2, 1}},
                      WidthCase{"NcdhwSpatialBy2", {2, 4, 3, 5, 6}, false, {2, 1}}),
    [](const ::testing::TestParamInfo<WidthCase>& info) { return info.param.name; });

TEST(TestHipMlopsBatchnormInferenceKnobs, PinningAWidthBuildsItWhereItDivides)
{
    for(const bool withVariance : {false, true})
    {
        SCOPED_TRACE(withVariance ? "variance and epsilon" : "inverse variance");
        const auto by4 = ranked(shape(withVariance, {2, 6, 2, 4}));
        for(const int64_t width : {1, 2, 4})
        {
            EXPECT_EQ(pinnedTuple(by4, KNOBS, {{"vector_size", width}}),
                      (KnobSettings{{"vector_size", width}}));
        }
        // A spatial size of 6 does not split into vectors of 4.
        const auto by2 = ranked(shape(withVariance, {2, 6, 2, 3}));
        EXPECT_FALSE(pinnedTuple(by2, KNOBS, {{"vector_size", 4}}).has_value());
    }
}

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
                const size_t at = spec.channelLast ? (n * spatial + i) * channels + c
                                                   : (n * channels + c) * spatial + i;
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

/// Runs @p spec through its engine, with @p knobs pinned, against the reference.
void checkInference(const BatchnormGraph& spec, const KnobSettings& knobs = {})
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
    ASSERT_NO_FATAL_FAILURE(
        executeThroughTheEngine(engineFor(spec), buildBatchnormGraph(spec), buffers, &run, knobs));
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

TEST(TestHipMlopsBatchnormInferenceDevice, TheProviderAdvertisesTheRankedKnob)
{
    SKIP_IF_NO_DEVICES();

    for(const bool withVariance : {false, true})
    {
        SCOPED_TRACE(withVariance ? "variance and epsilon" : "inverse variance");
        const auto spec = shape(withVariance, {2, 6, 2, 3});
        const auto knobs = advertisedKnobs(engineFor(spec), buildBatchnormGraph(spec));
        ASSERT_EQ(knobs.size(), 1U);
        EXPECT_EQ(knobs[0].name, "vector_size");
        EXPECT_EQ(knobs[0].defaultValue, 2);
        EXPECT_THAT(knobs[0].validValues, ::testing::UnorderedElementsAre(1, 2));
    }
}

TEST(TestHipMlopsBatchnormInferenceDevice, EveryVectorWidthMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    // Vectorized along the spatial dimension (NCHW) and along C (NHWC).
    for(const bool withVariance : {false, true})
    {
        for(const bool channelLast : {false, true})
        {
            SCOPED_TRACE(withVariance ? "variance and epsilon" : "inverse variance");
            SCOPED_TRACE(channelLast ? "NHWC" : "NCHW");
            const auto spec = shape(withVariance,
                                    channelLast ? std::vector<int64_t>{2, 8, 5, 7}
                                                : std::vector<int64_t>{2, 6, 2, 4},
                                    channelLast);
            std::set<int64_t> exercised;
            for(const auto& kernel :
                rankedKernelsOnThisDevice(engineFor(spec), buildBatchnormGraph(spec)))
            {
                SCOPED_TRACE(kernel.name);
                const auto pins = knobTuple(kernel, KNOBS);
                checkInference(spec, pins);
                exercised.insert(pins.front().second);
            }
            EXPECT_EQ(exercised, (std::set<int64_t>{1, 2, 4}));
        }
    }
}

} // namespace
