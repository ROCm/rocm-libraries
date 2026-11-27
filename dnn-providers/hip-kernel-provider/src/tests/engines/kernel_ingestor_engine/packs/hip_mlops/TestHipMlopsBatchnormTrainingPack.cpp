// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/hip_mlops/HipMlopsTestSupport.hpp"

/**
 * @file TestHipMlopsBatchnormTrainingPack.cpp
 * @brief The HIP MLOps batchnorm training and backward engines on the kernel ingestor
 *        (ALMIOPEN-3074): hipkernel:BatchnormTraining and hipkernel:BatchnormBackward each
 *        accept what the batchnorm applicability checks accept, and compute training (with
 *        and without saved and running statistics, alone or followed by a ReLU) and the
 *        backward (with saved statistics or recomputing them) as a CPU reference does. The
 *        three-node backward fusion is covered by its integration bundles.
 *
 *        Each engine ships a kernel per performance configuration (ALMIOPEN-3066), addressed
 *        by the knobs `variant`, `block_size` and `vector_size`: the knobs offer what can run
 *        the graph, the launch heuristics' choice ranks first, and every configuration, pinned
 *        through the knobs, computes as the reference does.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing::hip_mlops;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr int64_t X_UID = 1;
constexpr int64_t Y_UID = 2;
constexpr int64_t SCALE_UID = 3;
constexpr int64_t BIAS_UID = 4;
constexpr int64_t EPSILON_UID = 5;
constexpr int64_t MEAN_UID = 6;
constexpr int64_t INV_VARIANCE_UID = 7;
constexpr int64_t PREV_MEAN_UID = 8;
constexpr int64_t PREV_VARIANCE_UID = 9;
constexpr int64_t MOMENTUM_UID = 10;
constexpr int64_t NEXT_MEAN_UID = 11;
constexpr int64_t NEXT_VARIANCE_UID = 12;
constexpr int64_t ACTIVATION_OUT_UID = 13;
constexpr int64_t DY_UID = 14;
constexpr int64_t DX_UID = 15;
constexpr int64_t DSCALE_UID = 16;
constexpr int64_t DBIAS_UID = 17;

constexpr float EPSILON = 1e-3f;
constexpr float MOMENTUM = 0.1f;
constexpr float CLIP = 0.75f;
constexpr double DEFAULT_EPSILON = 1e-5;

enum class Activation
{
    NONE,
    RELU,
    CLIPPED_RELU,
    LEAKY_RELU,
};

struct BatchnormGraph
{
    bool backward = false;
    // N, C, then the spatial dimensions.
    std::vector<int64_t> dims = {2, 6, 5, 7};
    // x, y and their gradients channel-last (NLC, NHWC, NDHWC) rather than channel-first.
    bool channelLast = false;
    // Training: the saved mean and inverse variance outputs. Backward: those inputs.
    bool withSaved = true;
    // Training only: previous and next running statistics and the momentum.
    bool withRunning = true;
    // Training only: only the previous running mean, which the checks refuse.
    bool partialRunning = false;
    Activation activation = Activation::NONE;
    bool overrideShape = false;
    bool xVirtual = false;
    bool intermediateNotVirtual = false;
    bool swapNodes = false;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
};

bool fused(const BatchnormGraph& spec)
{
    return spec.activation != Activation::NONE;
}

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
    const auto addTensor = [&](int64_t uid, bool perChannel, bool isVirtual = false) {
        const auto& dims = perChannel ? perChannelDims : spec.dims;
        const auto& strides = perChannel ? perChannelStrides : ioStrides;
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, uid, nullptr, data_objects::DataType::FLOAT, &strides, &dims, isVirtual));
    };
    const auto addScalar = [&](int64_t uid, float value) {
        const std::vector<int64_t> scalarDims = {1};
        const data_objects::Float32Value scalar(value);
        tensors.push_back(
            data_objects::CreateTensorAttributesDirect(builder,
                                                       uid,
                                                       nullptr,
                                                       data_objects::DataType::FLOAT,
                                                       &scalarDims,
                                                       &scalarDims,
                                                       false,
                                                       data_objects::TensorValue::Float32Value,
                                                       builder.CreateStruct(scalar).Union()));
    };

    addTensor(X_UID, false, spec.xVirtual);
    addTensor(SCALE_UID, true);
    if(spec.withSaved)
    {
        addTensor(MEAN_UID, true);
        addTensor(INV_VARIANCE_UID, true);
    }

    flatbuffers::Offset<void> batchnorm;
    data_objects::NodeAttributes type{};
    if(spec.backward)
    {
        addTensor(DY_UID, false);
        addTensor(DX_UID, false);
        addTensor(DSCALE_UID, true);
        addTensor(DBIAS_UID, true);
        data_objects::BatchnormBackwardAttributesBuilder attributes(builder);
        attributes.add_dy_tensor_uid(DY_UID);
        attributes.add_x_tensor_uid(X_UID);
        if(spec.withSaved)
        {
            attributes.add_mean_tensor_uid(MEAN_UID);
            attributes.add_inv_variance_tensor_uid(INV_VARIANCE_UID);
        }
        attributes.add_scale_tensor_uid(SCALE_UID);
        attributes.add_dx_tensor_uid(DX_UID);
        attributes.add_dscale_tensor_uid(DSCALE_UID);
        attributes.add_dbias_tensor_uid(DBIAS_UID);
        batchnorm = attributes.Finish().Union();
        type = data_objects::NodeAttributes::BatchnormBackwardAttributes;
    }
    else
    {
        addTensor(BIAS_UID, true);
        addTensor(Y_UID, false, isFused && !spec.intermediateNotVirtual);
        addScalar(EPSILON_UID, EPSILON);
        if(spec.withRunning || spec.partialRunning)
        {
            addTensor(PREV_MEAN_UID, true);
        }
        if(spec.withRunning)
        {
            addTensor(PREV_VARIANCE_UID, true);
            addTensor(NEXT_MEAN_UID, true);
            addTensor(NEXT_VARIANCE_UID, true);
            addScalar(MOMENTUM_UID, MOMENTUM);
        }
        if(isFused)
        {
            addTensor(ACTIVATION_OUT_UID, false);
        }
        data_objects::BatchnormAttributesBuilder attributes(builder);
        attributes.add_x_tensor_uid(X_UID);
        attributes.add_scale_tensor_uid(SCALE_UID);
        attributes.add_bias_tensor_uid(BIAS_UID);
        attributes.add_epsilon_tensor_uid(EPSILON_UID);
        attributes.add_y_tensor_uid(Y_UID);
        if(spec.withSaved)
        {
            attributes.add_mean_tensor_uid(MEAN_UID);
            attributes.add_inv_variance_tensor_uid(INV_VARIANCE_UID);
        }
        if(spec.withRunning || spec.partialRunning)
        {
            attributes.add_prev_running_mean_tensor_uid(PREV_MEAN_UID);
        }
        if(spec.withRunning)
        {
            attributes.add_prev_running_variance_tensor_uid(PREV_VARIANCE_UID);
            attributes.add_momentum_tensor_uid(MOMENTUM_UID);
            attributes.add_next_running_mean_tensor_uid(NEXT_MEAN_UID);
            attributes.add_next_running_variance_tensor_uid(NEXT_VARIANCE_UID);
        }
        batchnorm = attributes.Finish().Union();
        type = data_objects::NodeAttributes::BatchnormAttributes;
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder, "batchnorm", spec.computeType, type, batchnorm));
    if(isFused)
    {
        data_objects::PointwiseAttributesBuilder activation(builder);
        activation.add_operation(data_objects::PointwiseMode::RELU_FWD);
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
    if(spec.swapNodes)
    {
        std::reverse(nodes.begin(), nodes.end());
    }

    auto name = builder.CreateString("hip_mlops_batchnorm_training");
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

/// The engine that serves @p spec: training, or the backward.
const MlopsEngine& engineFor(const BatchnormGraph& spec)
{
    return spec.backward ? BATCHNORM_BACKWARD_ENGINE : BATCHNORM_TRAINING_ENGINE;
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

TEST(TestHipMlopsBatchnormTrainingPack, TrainingAndBackwardAreEachOfferedUnderTheirOwnNameAndId)
{
    expectOfferedUnderItsOwnNameAndId(BATCHNORM_TRAINING_ENGINE);
    expectOfferedUnderItsOwnNameAndId(BATCHNORM_BACKWARD_ENGINE);
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

TEST(TestHipMlopsBatchnormTrainingGraphMatch, RoutesTrainingAndBackwardToTheirEnginesAlone)
{
    expectServedAlone({}, "BATCHNORM_TRAINING");
    expectServedAlone(edited([](auto& s) {
                          s.withSaved = false;
                          s.withRunning = false;
                      }),
                      "BATCHNORM_TRAINING");
    expectServedAlone(edited([](auto& s) { s.activation = Activation::RELU; }),
                      "BATCHNORM_TRAINING");
    expectServedAlone(edited([](auto& s) { s.backward = true; }), "BATCHNORM_BACKWARD");
    expectServedAlone(edited([](auto& s) {
                          s.backward = true;
                          s.withSaved = false;
                      }),
                      "BATCHNORM_BACKWARD");
}

struct RefusalCase
{
    const char* name;
    BatchnormGraph spec;
};

class TestHipMlopsBatchnormTrainingRefusal : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestHipMlopsBatchnormTrainingRefusal, EveryEngineDeclines)
{
    EXPECT_THAT(acceptingEngines(buildBatchnormGraph(GetParam().spec)), ::testing::IsEmpty());
}

// The applicability checks' refusals.
INSTANTIATE_TEST_SUITE_P(
    ,
    TestHipMlopsBatchnormTrainingRefusal,
    ::testing::Values(
        RefusalCase{"OverrideShapes", edited([](auto& s) { s.overrideShape = true; })},
        RefusalCase{"HalfCompute",
                    edited([](auto& s) { s.computeType = data_objects::DataType::HALF; })},
        RefusalCase{"BackwardHalfCompute", edited([](auto& s) {
                        s.backward = true;
                        s.computeType = data_objects::DataType::HALF;
                    })},
        RefusalCase{"VirtualInput", edited([](auto& s) { s.xVirtual = true; })},
        RefusalCase{"BackwardVirtualInput", edited([](auto& s) {
                        s.backward = true;
                        s.xVirtual = true;
                    })},
        RefusalCase{"PartialRunningStatistics", edited([](auto& s) {
                        s.withRunning = false;
                        s.partialRunning = true;
                    })},
        RefusalCase{"FusedNodesSwapped", edited([](auto& s) {
                        s.activation = Activation::RELU;
                        s.swapNodes = true;
                    })},
        RefusalCase{"FusedIntermediateNotVirtual", edited([](auto& s) {
                        s.activation = Activation::RELU;
                        s.intermediateNotVirtual = true;
                    })},
        RefusalCase{"LeakyRelu", edited([](auto& s) { s.activation = Activation::LEAKY_RELU; })}),
    [](const ::testing::TestParamInfo<RefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Knobs: one kernel per configuration
// ---------------------------------------------------------------------------

const std::vector<std::string> KNOBS = {"variant", "block_size", "vector_size"};

/// A configuration as its kernel's knob tuple: variant, block_size, vector_size.
KnobSettings config(int64_t variant, int64_t blockSize, int64_t vectorSize)
{
    return {{"variant", variant}, {"block_size", blockSize}, {"vector_size", vectorSize}};
}

/// The configurations each engine ships: variants 0, 1 and 3 with 256 or 1024 threads, and
/// variant 2 (no workgroup size of its own) with vector size 1, 2 or 4.
std::set<KnobSettings> allConfigs()
{
    std::set<KnobSettings> configs;
    for(const int64_t variant : {0, 1, 3})
    {
        for(const int64_t blockSize : {256, 1024})
        {
            configs.insert(config(variant, blockSize, 1));
        }
    }
    for(const int64_t vectorSize : {1, 2, 4})
    {
        configs.insert(config(2, 0, vectorSize));
    }
    return configs;
}

auto ranked(const BatchnormGraph& spec)
{
    return rankedKernels(engineFor(spec), buildBatchnormGraph(spec));
}

/// The knob tuple of the kernel @p spec's unpinned plan builds.
std::optional<KnobSettings> unpinned(const BatchnormGraph& spec)
{
    return pinnedTuple(ranked(spec), KNOBS, {});
}

TEST(TestHipMlopsBatchnormTrainingKnobs, EachEngineShipsAKernelPerConfiguration)
{
    for(const auto* engine : {&BATCHNORM_TRAINING_ENGINE, &BATCHNORM_BACKWARD_ENGINE})
    {
        SCOPED_TRACE(std::string(engine->name));
        EXPECT_EQ(engineKnobs(*engine), KNOBS);

        const auto kernels = loadedKernels(*engine);
        std::set<KnobSettings> tuples;
        for(const auto& kernel : kernels)
        {
            EXPECT_TRUE(tuples.insert(knobTuple(kernel, KNOBS)).second) << kernel.name;
            // One program for every configuration, compiled with its own options.
            EXPECT_EQ(kernel.source.entryPoint,
                      engine == &BATCHNORM_TRAINING_ENGINE ? "BatchNormFwdTrainSpatial"
                                                           : "BatchNormBwdSpatial");
        }
        EXPECT_EQ(kernels.size(), 9U);
        EXPECT_EQ(tuples, allConfigs());
    }
}

/// Expects each knob to offer @p values, @p defaults first: what the kernels admitted for
/// @p spec's graph carry, the best ranked kernel's value first.
void expectKnobs(const BatchnormGraph& spec,
                 const std::map<std::string, std::vector<int64_t>>& values,
                 const KnobSettings& defaults)
{
    const auto kernels = ranked(spec);
    ASSERT_FALSE(kernels.empty());
    expectUniqueTuples(kernels, KNOBS);
    for(const auto& [knob, defaultValue] : defaults)
    {
        SCOPED_TRACE(knob);
        const auto offered = knobValues(kernels, knob);
        ASSERT_FALSE(offered.empty());
        EXPECT_EQ(offered.front(), defaultValue);
        EXPECT_THAT(offered, ::testing::UnorderedElementsAreArray(values.at(knob)));
    }
}

TEST(TestHipMlopsBatchnormTrainingKnobs, KnobsOfferTheConfigurationsThatRunTheGraph)
{
    for(const bool backward : {false, true})
    {
        for(const bool withSaved : {true, false})
        {
            SCOPED_TRACE(backward ? "backward" : "training");
            SCOPED_TRACE(withSaved ? "with saved statistics" : "without saved statistics");
            // NCHW with 35 spatial positions: variants 0, 1 and 3 with either workgroup, and
            // variant 2 only unvectorized; variant 0 of 256 threads is the heuristics' choice.
            BatchnormGraph nchw;
            nchw.backward = backward;
            nchw.withSaved = withSaved;
            expectKnobs(
                nchw,
                {{"variant", {0, 1, 2, 3}}, {"block_size", {0, 256, 1024}}, {"vector_size", {1}}},
                config(0, 256, 1));

            // NHWC with 8 channels: no variant 0 or 3, variant 2 with every vector size, and
            // spatial multiple by 4 is the heuristics' choice.
            BatchnormGraph nhwc = nchw;
            nhwc.dims = {2, 8, 4, 4};
            nhwc.channelLast = true;
            expectKnobs(
                nhwc,
                {{"variant", {1, 2}}, {"block_size", {0, 256, 1024}}, {"vector_size", {1, 2, 4}}},
                config(2, 0, 4));
        }
    }
}

struct DefaultCase
{
    const char* name;
    BatchnormGraph spec;
    KnobSettings expected;
};

BatchnormGraph
    shape(bool backward, std::vector<int64_t> dims, bool channelLast = false, bool withSaved = true)
{
    BatchnormGraph spec;
    spec.backward = backward;
    spec.dims = std::move(dims);
    spec.channelLast = channelLast;
    spec.withSaved = withSaved;
    spec.withRunning = !backward;
    return spec;
}

class TestHipMlopsBatchnormTrainingDefault : public ::testing::TestWithParam<DefaultCase>
{
};

// Unpinned, the engine builds what the launch heuristics choose
// (defaultBatchnormFwdTrainingConfig(), defaultBatchnormBwdConfig(); TestDefaultSpatialConfig's
// branches that an fp32 graph reaches).
TEST_P(TestHipMlopsBatchnormTrainingDefault, TheHeuristicsChoiceRanksFirst)
{
    EXPECT_EQ(unpinned(GetParam().spec), GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestHipMlopsBatchnormTrainingDefault,
    ::testing::Values(
        // Forward, NCHW, spatial single
        DefaultCase{"FwdVariant0Small", shape(false, {2, 6, 5, 7}), config(0, 256, 1)},
        DefaultCase{"FwdVariant0Spatial256", shape(false, {2, 2, 16, 16}), config(0, 1024, 1)},
        DefaultCase{"FwdVariant3", shape(false, {2, 2, 20, 30}), config(3, 1024, 1)},
        DefaultCase{"FwdVariant1LargeSpatial", shape(false, {2, 3, 40, 40}), config(1, 1024, 1)},
        // A channel of 32M values with a spatial size beyond the workgroup of variants 0
        // and 3
        DefaultCase{"FwdVariant1LargeSpatialLargeChannel",
                    shape(false, {2, 2, 4096, 4096}),
                    config(1, 1024, 1)},
        // Forward, NCHW, spatial multiple keeps the smallest applicable vector size
        DefaultCase{"FwdNchwMultiple", shape(false, {3, 2, 24, 24}), config(2, 0, 1)},
        // Channel last tries spatial multiple by 4, then by 1
        DefaultCase{"FwdNhwcMultipleVector4", shape(false, {2, 8, 4, 4}, true), config(2, 0, 4)},
        DefaultCase{"FwdNhwcMultipleVector1", shape(false, {2, 3, 4, 4}, true), config(2, 0, 1)},
        // Backward, NCHW, spatial single
        DefaultCase{"BwdVariant0Small", shape(true, {2, 6, 5, 7}), config(0, 256, 1)},
        DefaultCase{"BwdVariant0Spatial256", shape(true, {2, 2, 16, 16}), config(0, 1024, 1)},
        DefaultCase{
            "BwdVariant3SmallSpatialLargeBatch", shape(true, {128, 2, 10, 20}), config(3, 256, 1)},
        DefaultCase{
            "BwdVariant3MidSpatialSmallBatch", shape(true, {2, 2, 20, 30}), config(3, 1024, 1)},
        DefaultCase{
            "BwdVariant1MidSpatialLargeBatch", shape(true, {32, 2, 20, 30}), config(1, 1024, 1)},
        DefaultCase{"BwdVariant1LargeSpatial", shape(true, {2, 3, 40, 40}), config(1, 1024, 1)},
        // Backward, NCHW, spatial multiple once a channel holds 32M values
        DefaultCase{
            "BwdNchwMultiple", shape(true, {64, 2, 1024, 1024}, false, false), config(2, 0, 1)},
        // Backward, channel last
        DefaultCase{"BwdNhwcMultipleVector4", shape(true, {2, 8, 4, 4}, true), config(2, 0, 4)},
        DefaultCase{"BwdNhwcMultipleVector4Recomputing",
                    shape(true, {2, 8, 4, 4}, true, false),
                    config(2, 0, 4)}),
    [](const ::testing::TestParamInfo<DefaultCase>& info) { return info.param.name; });

TEST(TestHipMlopsBatchnormTrainingKnobs, PinningAKnobValueBuildsTheBestRankedConfigurationWithIt)
{
    for(const bool backward : {false, true})
    {
        SCOPED_TRACE(backward ? "backward" : "training");
        const auto nchw = ranked(shape(backward, {2, 6, 5, 7}));
        EXPECT_EQ(pinnedTuple(nchw, KNOBS, {{"variant", 0}}), config(0, 256, 1));
        EXPECT_EQ(pinnedTuple(nchw, KNOBS, {{"variant", 1}}), config(1, 1024, 1));
        EXPECT_EQ(pinnedTuple(nchw, KNOBS, {{"variant", 2}}), config(2, 0, 1));
        EXPECT_EQ(pinnedTuple(nchw, KNOBS, {{"variant", 3}}), config(3, 1024, 1));
        EXPECT_EQ(pinnedTuple(nchw, KNOBS, {{"block_size", 0}}), config(2, 0, 1));
        EXPECT_EQ(pinnedTuple(nchw, KNOBS, {{"block_size", 256}}), config(0, 256, 1));
        EXPECT_EQ(pinnedTuple(nchw, KNOBS, {{"block_size", 1024}}), config(1, 1024, 1));
        EXPECT_EQ(pinnedTuple(nchw, KNOBS, {{"variant", 3}, {"block_size", 256}}),
                  config(3, 256, 1));
        // 35 spatial positions do not vectorize.
        EXPECT_FALSE(pinnedTuple(nchw, KNOBS, {{"vector_size", 2}}).has_value());

        const auto nhwc = ranked(shape(backward, {2, 8, 4, 4}, true));
        EXPECT_EQ(pinnedTuple(nhwc, KNOBS, {{"vector_size", 1}}), config(1, 1024, 1));
        EXPECT_EQ(pinnedTuple(nhwc, KNOBS, {{"vector_size", 2}}), config(2, 0, 2));
        EXPECT_EQ(pinnedTuple(nhwc, KNOBS, {{"vector_size", 4}}), config(2, 0, 4));
        EXPECT_EQ(pinnedTuple(nhwc, KNOBS, {{"variant", 1}}), config(1, 1024, 1));
        EXPECT_EQ(pinnedTuple(nhwc, KNOBS, {{"block_size", 256}}), config(1, 256, 1));
        // Variants 0 and 3 index channel-first tensors.
        EXPECT_FALSE(pinnedTuple(nhwc, KNOBS, {{"variant", 0}}).has_value());
        EXPECT_FALSE(pinnedTuple(nhwc, KNOBS, {{"variant", 3}}).has_value());
    }
}

// ---------------------------------------------------------------------------
// Numerics through the engine
// ---------------------------------------------------------------------------

/// Per-channel statistics over the batch and spatial dimensions, in double.
struct Statistics
{
    std::vector<double> mean;
    std::vector<double> variance; // population
    size_t perChannel;
};

struct Layout
{
    size_t batches;
    size_t channels;
    size_t spatial;
    bool channelLast;

    size_t at(size_t n, size_t c, size_t i) const
    {
        return channelLast ? (n * spatial + i) * channels + c : (n * channels + c) * spatial + i;
    }
};

Layout layoutOf(const BatchnormGraph& spec)
{
    const auto batches = static_cast<size_t>(spec.dims[0]);
    const auto channels = static_cast<size_t>(spec.dims[1]);
    return {batches, channels, elementCount(spec.dims) / (batches * channels), spec.channelLast};
}

Statistics statisticsOf(const Layout& layout, const std::vector<float>& x)
{
    Statistics stats{std::vector<double>(layout.channels),
                     std::vector<double>(layout.channels),
                     layout.batches * layout.spatial};
    for(size_t c = 0; c < layout.channels; ++c)
    {
        double sum = 0.0;
        for(size_t n = 0; n < layout.batches; ++n)
        {
            for(size_t i = 0; i < layout.spatial; ++i)
            {
                sum += static_cast<double>(x[layout.at(n, c, i)]);
            }
        }
        const double mean = sum / static_cast<double>(stats.perChannel);
        double squares = 0.0;
        for(size_t n = 0; n < layout.batches; ++n)
        {
            for(size_t i = 0; i < layout.spatial; ++i)
            {
                const double centred = static_cast<double>(x[layout.at(n, c, i)]) - mean;
                squares += centred * centred;
            }
        }
        stats.mean[c] = mean;
        stats.variance[c] = squares / static_cast<double>(stats.perChannel);
    }
    return stats;
}

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

void expectNear(const std::vector<float>& actual,
                const std::vector<double>& expected,
                const char* what,
                float relative = 1e-4f)
{
    SCOPED_TRACE(what);
    ASSERT_EQ(actual.size(), expected.size());
    size_t failures = 0;
    for(size_t i = 0; i < actual.size() && failures < 10; ++i)
    {
        const auto want = static_cast<float>(expected[i]);
        const float tolerance = relative * (1.0f + std::fabs(want));
        EXPECT_NEAR(actual[i], want, tolerance) << "element " << i;
        failures += std::fabs(actual[i] - want) > tolerance ? 1 : 0;
    }
}

struct Inputs
{
    Layout layout;
    std::vector<float> x;
    std::vector<float> scale;
    std::vector<float> bias;
};

Inputs inputsFor(const BatchnormGraph& spec)
{
    const auto layout = layoutOf(spec);
    auto x = deterministicValues(elementCount(spec.dims), 0.4f);
    auto scale = deterministicValues(layout.channels, 0.7f);
    auto bias = deterministicValues(layout.channels, 2.9f);
    for(size_t c = 0; c < layout.channels; ++c)
    {
        scale[c] = 1.0f + 0.25f * scale[c];
        bias[c] *= 0.25f;
    }
    // A per-channel offset, so the means differ.
    for(size_t n = 0; n < layout.batches; ++n)
    {
        for(size_t c = 0; c < layout.channels; ++c)
        {
            for(size_t i = 0; i < layout.spatial; ++i)
            {
                x[layout.at(n, c, i)] += 0.3f * static_cast<float>(c);
            }
        }
    }
    return {layout, std::move(x), std::move(scale), std::move(bias)};
}

/// Runs @p spec's training through its engine, with @p knobs pinned, against the reference.
void checkTraining(const BatchnormGraph& spec, const KnobSettings& knobs = {})
{
    const auto in = inputsFor(spec);
    const auto& layout = in.layout;
    const size_t count = in.x.size();
    const size_t channels = layout.channels;
    const auto stats = statisticsOf(layout, in.x);

    std::vector<double> y(count);
    std::vector<double> invVariance(channels);
    for(size_t c = 0; c < channels; ++c)
    {
        invVariance[c] = 1.0 / std::sqrt(stats.variance[c] + static_cast<double>(EPSILON));
        for(size_t n = 0; n < layout.batches; ++n)
        {
            for(size_t i = 0; i < layout.spatial; ++i)
            {
                const size_t at = layout.at(n, c, i);
                const double value = static_cast<double>(in.scale[c])
                                         * (static_cast<double>(in.x[at]) - stats.mean[c])
                                         * invVariance[c]
                                     + static_cast<double>(in.bias[c]);
                y[at] = activate(spec.activation, value);
            }
        }
    }
    const auto prevMean = deterministicValues(channels, 1.3f);
    auto prevVariance = deterministicValues(channels, 2.1f);
    for(auto& value : prevVariance)
    {
        value = 1.0f + 0.4f * value;
    }
    std::vector<double> nextMean(channels);
    std::vector<double> nextVariance(channels);
    const auto momentum = static_cast<double>(MOMENTUM);
    const auto unbiased
        = static_cast<double>(stats.perChannel) / static_cast<double>(stats.perChannel - 1);
    for(size_t c = 0; c < channels; ++c)
    {
        nextMean[c]
            = (1.0 - momentum) * static_cast<double>(prevMean[c]) + momentum * stats.mean[c];
        nextVariance[c] = (1.0 - momentum) * static_cast<double>(prevVariance[c])
                          + momentum * stats.variance[c] * unbiased;
    }

    const DeviceBuffer<float> xBuffer(count, in.x);
    const DeviceBuffer<float> yBuffer(count);
    const DeviceBuffer<float> scaleBuffer(channels, in.scale);
    const DeviceBuffer<float> biasBuffer(channels, in.bias);
    const DeviceBuffer<float> meanBuffer(channels);
    const DeviceBuffer<float> invVarianceBuffer(channels);
    const DeviceBuffer<float> prevMeanBuffer(channels, prevMean);
    const DeviceBuffer<float> prevVarianceBuffer(channels, prevVariance);
    const DeviceBuffer<float> nextMeanBuffer(channels);
    const DeviceBuffer<float> nextVarianceBuffer(channels);
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{X_UID, xBuffer.get()},
           {fused(spec) ? ACTIVATION_OUT_UID : Y_UID, yBuffer.get()},
           {SCALE_UID, scaleBuffer.get()},
           {BIAS_UID, biasBuffer.get()}};
    if(spec.withSaved)
    {
        buffers.push_back({MEAN_UID, meanBuffer.get()});
        buffers.push_back({INV_VARIANCE_UID, invVarianceBuffer.get()});
    }
    if(spec.withRunning)
    {
        buffers.push_back({PREV_MEAN_UID, prevMeanBuffer.get()});
        buffers.push_back({PREV_VARIANCE_UID, prevVarianceBuffer.get()});
        buffers.push_back({NEXT_MEAN_UID, nextMeanBuffer.get()});
        buffers.push_back({NEXT_VARIANCE_UID, nextVarianceBuffer.get()});
    }

    EngineRun run;
    ASSERT_NO_FATAL_FAILURE(
        executeThroughTheEngine(engineFor(spec), buildBatchnormGraph(spec), buffers, &run, knobs));
    EXPECT_EQ(run.workspaceBytes, 0U);
    EXPECT_EQ(run.maxWorkspaceBytes, 0U);

    expectNear(yBuffer.read(), y, "y");
    if(spec.withSaved)
    {
        expectNear(meanBuffer.read(), stats.mean, "saved mean");
        expectNear(invVarianceBuffer.read(), invVariance, "saved inverse variance");
    }
    if(spec.withRunning)
    {
        expectNear(nextMeanBuffer.read(), nextMean, "next running mean");
        expectNear(nextVarianceBuffer.read(), nextVariance, "next running variance");
    }
}

/// Runs @p spec's backward through its engine, with @p knobs pinned, against the reference.
void checkBackward(const BatchnormGraph& spec, const KnobSettings& knobs = {})
{
    const auto in = inputsFor(spec);
    const auto& layout = in.layout;
    const size_t count = in.x.size();
    const size_t channels = layout.channels;
    const auto stats = statisticsOf(layout, in.x);
    const auto dy = deterministicValues(count, 1.9f);
    // With saved statistics they are the forward's (epsilon EPSILON); without, the kernel
    // recomputes them with the default epsilon.
    const double epsilon = spec.withSaved ? static_cast<double>(EPSILON) : DEFAULT_EPSILON;

    std::vector<float> mean(channels);
    std::vector<float> invVariance(channels);
    std::vector<double> dx(count);
    std::vector<double> dscale(channels);
    std::vector<double> dbias(channels);
    const auto m = static_cast<double>(stats.perChannel);
    for(size_t c = 0; c < channels; ++c)
    {
        mean[c] = static_cast<float>(stats.mean[c]);
        invVariance[c] = static_cast<float>(1.0 / std::sqrt(stats.variance[c] + epsilon));
        const auto r = static_cast<double>(invVariance[c]);
        const auto mu = static_cast<double>(mean[c]);
        double sumDy = 0.0;
        double sumDyXhat = 0.0;
        for(size_t n = 0; n < layout.batches; ++n)
        {
            for(size_t i = 0; i < layout.spatial; ++i)
            {
                const size_t at = layout.at(n, c, i);
                const double xhat = (static_cast<double>(in.x[at]) - mu) * r;
                sumDy += static_cast<double>(dy[at]);
                sumDyXhat += static_cast<double>(dy[at]) * xhat;
            }
        }
        dbias[c] = sumDy;
        dscale[c] = sumDyXhat;
        for(size_t n = 0; n < layout.batches; ++n)
        {
            for(size_t i = 0; i < layout.spatial; ++i)
            {
                const size_t at = layout.at(n, c, i);
                const double xhat = (static_cast<double>(in.x[at]) - mu) * r;
                dx[at] = static_cast<double>(in.scale[c]) * r / m
                         * (m * static_cast<double>(dy[at]) - sumDy - xhat * sumDyXhat);
            }
        }
    }

    const DeviceBuffer<float> xBuffer(count, in.x);
    const DeviceBuffer<float> dyBuffer(count, dy);
    const DeviceBuffer<float> dxBuffer(count);
    const DeviceBuffer<float> scaleBuffer(channels, in.scale);
    const DeviceBuffer<float> dscaleBuffer(channels);
    const DeviceBuffer<float> dbiasBuffer(channels);
    const DeviceBuffer<float> meanBuffer(channels, mean);
    const DeviceBuffer<float> invVarianceBuffer(channels, invVariance);
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{X_UID, xBuffer.get()},
                                                       {DY_UID, dyBuffer.get()},
                                                       {DX_UID, dxBuffer.get()},
                                                       {SCALE_UID, scaleBuffer.get()},
                                                       {DSCALE_UID, dscaleBuffer.get()},
                                                       {DBIAS_UID, dbiasBuffer.get()}};
    if(spec.withSaved)
    {
        buffers.push_back({MEAN_UID, meanBuffer.get()});
        buffers.push_back({INV_VARIANCE_UID, invVarianceBuffer.get()});
    }

    EngineRun run;
    ASSERT_NO_FATAL_FAILURE(
        executeThroughTheEngine(engineFor(spec), buildBatchnormGraph(spec), buffers, &run, knobs));
    EXPECT_EQ(run.workspaceBytes, 0U);

    expectNear(dxBuffer.read(), dx, "dx", 1e-3f);
    expectNear(dscaleBuffer.read(), dscale, "dscale", 1e-3f);
    expectNear(dbiasBuffer.read(), dbias, "dbias", 1e-3f);
}

TEST(TestHipMlopsBatchnormTrainingDevice, TrainingMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    for(const bool withSaved : {true, false})
    {
        for(const bool withRunning : {true, false})
        {
            SCOPED_TRACE(withSaved ? "with saved statistics" : "without saved statistics");
            SCOPED_TRACE(withRunning ? "with running statistics" : "without running statistics");
            BatchnormGraph spec;
            spec.withSaved = withSaved;
            spec.withRunning = withRunning;
            checkTraining(spec);
        }
    }
}

TEST(TestHipMlopsBatchnormTrainingDevice, FusedTrainingAppliesTheActivation)
{
    SKIP_IF_NO_DEVICES();

    for(const auto activation : {Activation::RELU, Activation::CLIPPED_RELU})
    {
        SCOPED_TRACE(activation == Activation::RELU ? "relu" : "clipped relu");
        BatchnormGraph spec;
        spec.activation = activation;
        checkTraining(spec);
    }
}

TEST(TestHipMlopsBatchnormTrainingDevice, BackwardMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    for(const bool withSaved : {true, false})
    {
        SCOPED_TRACE(withSaved ? "with saved statistics" : "recomputing statistics");
        BatchnormGraph spec;
        spec.backward = true;
        spec.withSaved = withSaved;
        spec.withRunning = false;
        checkBackward(spec);
    }
}

TEST(TestHipMlopsBatchnormTrainingDevice, TheProviderAdvertisesTheRankedKnobs)
{
    SKIP_IF_NO_DEVICES();

    for(const bool backward : {false, true})
    {
        for(const bool channelLast : {false, true})
        {
            SCOPED_TRACE(backward ? "backward" : "training");
            SCOPED_TRACE(channelLast ? "NHWC" : "NCHW");
            const auto spec = shape(backward,
                                    channelLast ? std::vector<int64_t>{2, 8, 4, 4}
                                                : std::vector<int64_t>{2, 6, 5, 7},
                                    channelLast);
            const auto kernels
                = rankedKernelsOnThisDevice(engineFor(spec), buildBatchnormGraph(spec));
            const auto knobs = advertisedKnobs(engineFor(spec), buildBatchnormGraph(spec));
            ASSERT_EQ(knobs.size(), KNOBS.size());
            for(const auto& knob : knobs)
            {
                SCOPED_TRACE(knob.name);
                const auto expected = knobValues(kernels, knob.name);
                ASSERT_FALSE(expected.empty());
                EXPECT_EQ(knob.defaultValue, expected.front());
                EXPECT_THAT(knob.validValues, ::testing::UnorderedElementsAreArray(expected));
            }
        }
    }
}

/// Every knob value, and so every configuration, the device sweeps below must pin at least once.
const std::map<std::string, std::set<int64_t>> EVERY_KNOB_VALUE
    = {{"variant", {0, 1, 2, 3}}, {"block_size", {0, 256, 1024}}, {"vector_size", {1, 2, 4}}};

/// Pins each configuration @p spec's engine admits on this device in turn and runs @p check
/// with it, adding the values pinned to @p exercised.
template <typename Check>
void sweepConfigurations(const BatchnormGraph& spec,
                         Check check,
                         std::map<std::string, std::set<int64_t>>& exercised)
{
    for(const auto& kernel : rankedKernelsOnThisDevice(engineFor(spec), buildBatchnormGraph(spec)))
    {
        SCOPED_TRACE(kernel.name);
        const auto pins = knobTuple(kernel, KNOBS);
        check(spec, pins);
        for(const auto& [knob, value] : pins)
        {
            exercised[knob].insert(value);
        }
    }
}

TEST(TestHipMlopsBatchnormTrainingDevice, EveryTrainingConfigurationMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    // NCHW reaches variants 0, 1 and 3 and unvectorized variant 2; NHWC every vector size.
    std::map<std::string, std::set<int64_t>> exercised;
    for(const bool channelLast : {false, true})
    {
        SCOPED_TRACE(channelLast ? "NHWC" : "NCHW");
        sweepConfigurations(
            shape(false,
                  channelLast ? std::vector<int64_t>{2, 8, 4, 4} : std::vector<int64_t>{2, 6, 5, 7},
                  channelLast),
            [](const BatchnormGraph& spec, const KnobSettings& pins) { checkTraining(spec, pins); },
            exercised);
    }
    EXPECT_EQ(exercised, EVERY_KNOB_VALUE);
}

TEST(TestHipMlopsBatchnormTrainingDevice, EveryBackwardConfigurationMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    // Variant 2 launches three kernels with saved statistics and five recomputing them.
    std::map<std::string, std::set<int64_t>> exercised;
    for(const bool withSaved : {true, false})
    {
        for(const bool channelLast : {false, true})
        {
            SCOPED_TRACE(withSaved ? "with saved statistics" : "recomputing statistics");
            SCOPED_TRACE(channelLast ? "NHWC" : "NCHW");
            sweepConfigurations(
                shape(true,
                      channelLast ? std::vector<int64_t>{2, 8, 4, 4}
                                  : std::vector<int64_t>{2, 6, 5, 7},
                      channelLast,
                      withSaved),
                [](const BatchnormGraph& spec, const KnobSettings& pins) {
                    checkBackward(spec, pins);
                },
                exercised);
        }
    }
    EXPECT_EQ(exercised, EVERY_KNOB_VALUE);
}

} // namespace
