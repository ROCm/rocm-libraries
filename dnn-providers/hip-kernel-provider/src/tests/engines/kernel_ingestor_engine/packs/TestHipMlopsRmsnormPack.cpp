// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENGINE_HIP_MLOPS_INGESTOR

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/HipMlopsTestSupport.hpp"

/**
 * @file TestHipMlopsRmsnormPack.cpp
 * @brief The HIP MLOps engine on the kernel ingestor, RMSnorm family (ALMIOPEN-3072): the
 *        engine accepts what the hand-written builders accept -- RMSnorm alone, and fused with
 *        a ReLU after the forward or before the backward -- and computes all four through the
 *        engine as CPU references do.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing::hip_mlops;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr int64_t X_UID = 1;
constexpr int64_t Y_UID = 2;
constexpr int64_t SCALE_UID = 3;
constexpr int64_t BIAS_UID = 4;
constexpr int64_t INV_RMS_UID = 5;
constexpr int64_t EPSILON_UID = 6;
constexpr int64_t ACTIVATION_OUT_UID = 7; // fused forward: the activation's output
constexpr int64_t DY_UID = 8;
constexpr int64_t DX_UID = 9;
constexpr int64_t DSCALE_UID = 10;
constexpr int64_t DBIAS_UID = 11;
constexpr int64_t GRADIENT_UID = 12; // fused backward: the gradient the activation reads
constexpr int64_t PRE_ACTIVATION_UID = 13; // fused backward: the forward activation's input

constexpr float CLIP = 0.75f;

enum class Activation
{
    NONE,
    RELU,
    CLIPPED_RELU,
    IDENTITY,
    SIGMOID,
    LEAKY_RELU,
};

/// An RMSnorm over every dimension after the first: one inverse RMS per row of `dims[0]`.
struct RmsnormGraph
{
    bool backward = false;
    // Inner size 800, more than one workgroup's 256 lanes.
    std::vector<int64_t> dims = {3, 4, 10, 20};
    // Forward: bias. Backward: dbias.
    bool withBias = true;
    // Forward only; the backward reads it.
    bool withInvRms = true;
    float epsilon = 1e-2f;
    // Fused after the forward, or before the backward.
    Activation activation = Activation::NONE;
    bool overrideShape = false;
    bool xVirtual = false;
    // A fused graph whose intermediate (y, or the backward's dy) is a graph output.
    bool intermediateNotVirtual = false;
    bool swapNodes = false;
    bool twoRmsnormNodes = false;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
};

bool fused(const RmsnormGraph& spec)
{
    return spec.activation != Activation::NONE;
}

std::vector<int64_t> scaleDims(const RmsnormGraph& spec)
{
    auto dims = spec.dims;
    dims[0] = 1;
    return dims;
}

std::vector<int64_t> invRmsDims(const RmsnormGraph& spec)
{
    std::vector<int64_t> dims(spec.dims.size(), 1);
    dims[0] = spec.dims[0];
    return dims;
}

flatbuffers::Offset<data_objects::PointwiseAttributes>
    activationAttributes(flatbuffers::FlatBufferBuilder& builder,
                         const RmsnormGraph& spec,
                         int64_t in0,
                         std::optional<int64_t> in1,
                         int64_t out0)
{
    const bool backward = spec.backward;
    auto mode
        = backward ? data_objects::PointwiseMode::RELU_BWD : data_objects::PointwiseMode::RELU_FWD;
    if(spec.activation == Activation::IDENTITY)
    {
        mode = data_objects::PointwiseMode::IDENTITY;
    }
    else if(spec.activation == Activation::SIGMOID)
    {
        mode = backward ? data_objects::PointwiseMode::SIGMOID_BWD
                        : data_objects::PointwiseMode::SIGMOID_FWD;
    }

    data_objects::PointwiseAttributesBuilder attributes(builder);
    attributes.add_operation(mode);
    attributes.add_in_0_tensor_uid(in0);
    if(in1.has_value())
    {
        attributes.add_in_1_tensor_uid(*in1);
    }
    attributes.add_out_0_tensor_uid(out0);
    if(spec.activation == Activation::CLIPPED_RELU)
    {
        attributes.add_relu_upper_clip(CLIP);
    }
    if(spec.activation == Activation::LEAKY_RELU)
    {
        attributes.add_relu_lower_clip_slope(0.1f);
    }
    return attributes.Finish();
}

flatbuffers::FlatBufferBuilder buildRmsnormGraph(const RmsnormGraph& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto ioDims = spec.dims;
    const auto ioStrides = packedStrides(ioDims);
    const auto affineDims = scaleDims(spec);
    const auto affineStrides = packedStrides(affineDims);
    const auto statDims = invRmsDims(spec);
    const auto statStrides = packedStrides(statDims);
    const bool isFused = fused(spec);

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    const auto addTensor = [&](int64_t uid,
                               const std::vector<int64_t>& dims,
                               const std::vector<int64_t>& strides,
                               bool isVirtual = false) {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, uid, nullptr, data_objects::DataType::FLOAT, &strides, &dims, isVirtual));
    };
    addTensor(X_UID, ioDims, ioStrides, spec.xVirtual);
    addTensor(SCALE_UID, affineDims, affineStrides);
    const bool intermediateVirtual = isFused && !spec.intermediateNotVirtual;
    if(spec.backward)
    {
        addTensor(DY_UID, ioDims, ioStrides, intermediateVirtual);
        addTensor(INV_RMS_UID, statDims, statStrides);
        addTensor(DX_UID, ioDims, ioStrides);
        addTensor(DSCALE_UID, affineDims, affineStrides);
        if(spec.withBias)
        {
            addTensor(DBIAS_UID, affineDims, affineStrides);
        }
        if(isFused)
        {
            addTensor(GRADIENT_UID, ioDims, ioStrides);
            if(spec.activation != Activation::IDENTITY)
            {
                addTensor(PRE_ACTIVATION_UID, ioDims, ioStrides);
            }
        }
    }
    else
    {
        addTensor(Y_UID, ioDims, ioStrides, intermediateVirtual);
        if(spec.withBias)
        {
            addTensor(BIAS_UID, affineDims, affineStrides);
        }
        if(spec.withInvRms)
        {
            addTensor(INV_RMS_UID, statDims, statStrides);
        }
        if(isFused)
        {
            addTensor(ACTIVATION_OUT_UID, ioDims, ioStrides);
        }
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

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    const auto addNode = [&](const char* name,
                             data_objects::NodeAttributes type,
                             flatbuffers::Offset<void> a) {
        nodes.push_back(data_objects::CreateNodeDirect(builder, name, spec.computeType, type, a));
    };
    if(spec.backward)
    {
        data_objects::RMSNormBackwardAttributesBuilder attributes(builder);
        attributes.add_dy_tensor_uid(DY_UID);
        attributes.add_x_tensor_uid(X_UID);
        attributes.add_scale_tensor_uid(SCALE_UID);
        attributes.add_inv_rms_tensor_uid(INV_RMS_UID);
        attributes.add_dx_tensor_uid(DX_UID);
        attributes.add_dscale_tensor_uid(DSCALE_UID);
        if(spec.withBias)
        {
            attributes.add_dbias_tensor_uid(DBIAS_UID);
        }
        const auto rmsnorm = attributes.Finish().Union();
        if(isFused)
        {
            const auto in1 = spec.activation == Activation::IDENTITY
                                 ? std::nullopt
                                 : std::optional<int64_t>(PRE_ACTIVATION_UID);
            const auto activation
                = activationAttributes(builder, spec, GRADIENT_UID, in1, DY_UID).Union();
            addNode(
                "activation_bwd", data_objects::NodeAttributes::PointwiseAttributes, activation);
        }
        addNode("rmsnorm_bwd", data_objects::NodeAttributes::RMSNormBackwardAttributes, rmsnorm);
    }
    else
    {
        data_objects::RMSNormAttributesBuilder attributes(builder);
        attributes.add_x_tensor_uid(X_UID);
        attributes.add_scale_tensor_uid(SCALE_UID);
        attributes.add_epsilon_tensor_uid(EPSILON_UID);
        attributes.add_y_tensor_uid(Y_UID);
        if(spec.withBias)
        {
            attributes.add_bias_tensor_uid(BIAS_UID);
        }
        if(spec.withInvRms)
        {
            attributes.add_inv_rms_tensor_uid(INV_RMS_UID);
        }
        const auto rmsnorm = attributes.Finish().Union();
        addNode("rmsnorm", data_objects::NodeAttributes::RMSNormAttributes, rmsnorm);
        if(isFused)
        {
            const auto activation
                = activationAttributes(builder, spec, Y_UID, std::nullopt, ACTIVATION_OUT_UID)
                      .Union();
            addNode("activation", data_objects::NodeAttributes::PointwiseAttributes, activation);
        }
        if(spec.twoRmsnormNodes)
        {
            addNode("rmsnorm2", data_objects::NodeAttributes::RMSNormAttributes, rmsnorm);
        }
    }
    if(spec.swapNodes)
    {
        std::reverse(nodes.begin(), nodes.end());
    }

    auto name = builder.CreateString("hip_mlops_rmsnorm");
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

auto match(const RmsnormGraph& spec)
{
    return matchGraph(buildRmsnormGraph(spec));
}

RmsnormGraph edited(void (*edit)(RmsnormGraph&))
{
    RmsnormGraph spec;
    edit(spec);
    return spec;
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

TEST(TestHipMlopsRmsnormGraphMatch, RoutesEachDirectionToItsPackFusedOrNot)
{
    EXPECT_EQ(boundOperation(match({})), "RMSNORM_FWD");
    EXPECT_EQ(boundOperation(match(edited([](auto& s) { s.activation = Activation::RELU; }))),
              "RMSNORM_FWD");
    EXPECT_EQ(boundOperation(match(edited([](auto& s) { s.backward = true; }))), "RMSNORM_BWD");
    EXPECT_EQ(boundOperation(match(edited([](auto& s) {
                  s.backward = true;
                  s.activation = Activation::RELU;
              }))),
              "RMSNORM_BWD");
    EXPECT_EQ(boundOperation(match(edited([](auto& s) {
                  s.backward = true;
                  s.activation = Activation::IDENTITY;
              }))),
              "RMSNORM_BWD");
}

struct RefusalCase
{
    const char* name;
    RmsnormGraph spec;
};

class TestHipMlopsRmsnormRefusal : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestHipMlopsRmsnormRefusal, Declines)
{
    EXPECT_FALSE(match(GetParam().spec).has_value());
}

// The hand-written builders' refusals.
INSTANTIATE_TEST_SUITE_P(
    ,
    TestHipMlopsRmsnormRefusal,
    ::testing::Values(
        RefusalCase{"OverrideShapes", edited([](auto& s) { s.overrideShape = true; })},
        RefusalCase{"HalfCompute",
                    edited([](auto& s) { s.computeType = data_objects::DataType::HALF; })},
        RefusalCase{"BackwardHalfCompute", edited([](auto& s) {
                        s.backward = true;
                        s.computeType = data_objects::DataType::HALF;
                    })},
        RefusalCase{"FusedHalfCompute", edited([](auto& s) {
                        s.activation = Activation::RELU;
                        s.computeType = data_objects::DataType::HALF;
                    })},
        RefusalCase{"VirtualInput", edited([](auto& s) { s.xVirtual = true; })},
        RefusalCase{"BackwardVirtualInput", edited([](auto& s) {
                        s.backward = true;
                        s.xVirtual = true;
                    })},
        RefusalCase{"TwoRmsnormNodes", edited([](auto& s) { s.twoRmsnormNodes = true; })},
        RefusalCase{"FusedNodesSwapped", edited([](auto& s) {
                        s.activation = Activation::RELU;
                        s.swapNodes = true;
                    })},
        RefusalCase{"BackwardFusedNodesSwapped", edited([](auto& s) {
                        s.backward = true;
                        s.activation = Activation::RELU;
                        s.swapNodes = true;
                    })},
        RefusalCase{"FusedIntermediateNotVirtual", edited([](auto& s) {
                        s.activation = Activation::RELU;
                        s.intermediateNotVirtual = true;
                    })},
        RefusalCase{"BackwardFusedIntermediateNotVirtual", edited([](auto& s) {
                        s.backward = true;
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

std::vector<float> invRmsReference(const std::vector<float>& x, size_t rows, double epsilon)
{
    const size_t inner = x.size() / rows;
    std::vector<float> invRms(rows);
    for(size_t r = 0; r < rows; ++r)
    {
        double squares = 0.0;
        for(size_t i = 0; i < inner; ++i)
        {
            const auto value = static_cast<double>(x[r * inner + i]);
            squares += value * value;
        }
        invRms[r]
            = static_cast<float>(1.0 / std::sqrt(squares / static_cast<double>(inner) + epsilon));
    }
    return invRms;
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

/// The gradient through the activation, given the activation's input.
double activationGradient(Activation activation, double gradient, double input)
{
    switch(activation)
    {
    case Activation::RELU:
        return input > 0.0 ? gradient : 0.0;
    case Activation::CLIPPED_RELU:
        return input > 0.0 && input <= static_cast<double>(CLIP) ? gradient : 0.0;
    default:
        return gradient;
    }
}

std::vector<float> forwardReference(const std::vector<float>& x,
                                    const std::vector<float>& scale,
                                    const std::vector<float>& bias,
                                    const std::vector<float>& invRms,
                                    Activation activation)
{
    const size_t inner = scale.size();
    std::vector<float> y(x.size());
    for(size_t r = 0; r < invRms.size(); ++r)
    {
        for(size_t i = 0; i < inner; ++i)
        {
            const size_t at = r * inner + i;
            double value = static_cast<double>(x[at]) * static_cast<double>(invRms[r])
                           * static_cast<double>(scale[i]);
            value += bias.empty() ? 0.0 : static_cast<double>(bias[i]);
            y[at] = static_cast<float>(activate(activation, value));
        }
    }
    return y;
}

struct BackwardReference
{
    std::vector<float> dx;
    std::vector<float> dscale;
    std::vector<float> dbias;
};

/// With g the gradient through the activation: dx = g * scale * r - mean(g * scale * x) * x *
/// r^3 per row, dscale = sum over rows of g * x * r, dbias = sum over rows of g.
BackwardReference backwardReference(const std::vector<float>& gradient,
                                    const std::vector<float>& preActivation,
                                    const std::vector<float>& x,
                                    const std::vector<float>& scale,
                                    const std::vector<float>& invRms,
                                    Activation activation)
{
    const size_t inner = scale.size();
    std::vector<double> dscale(inner, 0.0);
    std::vector<double> dbias(inner, 0.0);
    BackwardReference result{std::vector<float>(x.size()), {}, {}};
    const auto g = [&](size_t at) {
        return activationGradient(activation,
                                  static_cast<double>(gradient[at]),
                                  preActivation.empty() ? 0.0
                                                        : static_cast<double>(preActivation[at]));
    };
    for(size_t r = 0; r < invRms.size(); ++r)
    {
        const auto rstd = static_cast<double>(invRms[r]);
        double sum = 0.0;
        for(size_t i = 0; i < inner; ++i)
        {
            const size_t at = r * inner + i;
            sum += g(at) * static_cast<double>(scale[i]) * static_cast<double>(x[at]);
            dscale[i] += g(at) * static_cast<double>(x[at]) * rstd;
            dbias[i] += g(at);
        }
        const double mean = sum / static_cast<double>(inner);
        for(size_t i = 0; i < inner; ++i)
        {
            const size_t at = r * inner + i;
            result.dx[at]
                = static_cast<float>(g(at) * static_cast<double>(scale[i]) * rstd
                                     - mean * static_cast<double>(x[at]) * rstd * rstd * rstd);
        }
    }
    result.dscale.assign(dscale.begin(), dscale.end());
    result.dbias.assign(dbias.begin(), dbias.end());
    return result;
}

/// Close relative to the value: fp32 reductions over up to a few thousand terms.
void expectNear(const std::vector<float>& actual,
                const std::vector<float>& expected,
                const char* what)
{
    SCOPED_TRACE(what);
    ASSERT_EQ(actual.size(), expected.size());
    size_t failures = 0;
    for(size_t i = 0; i < actual.size() && failures < 10; ++i)
    {
        const float tolerance = 1e-4f * (1.0f + std::fabs(expected[i]));
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "element " << i;
        failures += std::fabs(actual[i] - expected[i]) > tolerance ? 1 : 0;
    }
}

struct Inputs
{
    size_t rows;
    std::vector<float> x;
    std::vector<float> scale;
    std::vector<float> bias;
};

Inputs inputsFor(const RmsnormGraph& spec)
{
    const size_t count = elementCount(spec.dims);
    const size_t inner = count / static_cast<size_t>(spec.dims[0]);
    // Scale around 1 and bias around 0 rather than straddling zero.
    auto scale = deterministicValues(inner, 1.1f);
    auto bias = deterministicValues(inner, 2.9f);
    for(auto& value : scale)
    {
        value = 1.0f + 0.25f * value;
    }
    for(auto& value : bias)
    {
        value *= 0.25f;
    }
    return {static_cast<size_t>(spec.dims[0]),
            deterministicValues(count, 0.4f),
            std::move(scale),
            std::move(bias)};
}

void checkForward(const RmsnormGraph& spec)
{
    const auto in = inputsFor(spec);
    const size_t count = in.x.size();
    // The graph's epsilon reaches the kernel.
    const auto invRms = invRmsReference(in.x, in.rows, static_cast<double>(spec.epsilon));
    const auto expected = forwardReference(
        in.x, in.scale, spec.withBias ? in.bias : std::vector<float>{}, invRms, spec.activation);

    const DeviceBuffer<float> x(count, in.x);
    const DeviceBuffer<float> scale(in.scale.size(), in.scale);
    const DeviceBuffer<float> bias(in.bias.size(), in.bias);
    const DeviceBuffer<float> invRmsBuffer(in.rows);
    const DeviceBuffer<float> y(count);
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{X_UID, x.get()},
           {SCALE_UID, scale.get()},
           {fused(spec) ? ACTIVATION_OUT_UID : Y_UID, y.get()}};
    if(spec.withBias)
    {
        buffers.push_back({BIAS_UID, bias.get()});
    }
    if(spec.withInvRms)
    {
        buffers.push_back({INV_RMS_UID, invRmsBuffer.get()});
    }

    EngineRun run;
    ASSERT_NO_FATAL_FAILURE(executeThroughTheEngine(buildRmsnormGraph(spec), buffers, &run));
    EXPECT_EQ(run.workspaceBytes, 0U);
    EXPECT_EQ(run.maxWorkspaceBytes, 0U);

    expectNear(y.read(), expected, "y");
    if(spec.withInvRms)
    {
        expectNear(invRmsBuffer.read(), invRms, "invRms");
    }
}

void checkBackward(const RmsnormGraph& spec)
{
    const auto in = inputsFor(spec);
    const size_t count = in.x.size();
    const auto invRms = invRmsReference(in.x, in.rows, 1e-5);
    const auto gradient = deterministicValues(count, 1.9f);
    // Straddles zero and the clip, so the activation's gradient masks some elements.
    const auto preActivation = fused(spec) && spec.activation != Activation::IDENTITY
                                   ? deterministicValues(count, 2.6f)
                                   : std::vector<float>{};
    const auto expected
        = backwardReference(gradient, preActivation, in.x, in.scale, invRms, spec.activation);

    const DeviceBuffer<float> dy(count, gradient);
    const DeviceBuffer<float> preActivationBuffer(count, preActivation);
    const DeviceBuffer<float> x(count, in.x);
    const DeviceBuffer<float> scale(in.scale.size(), in.scale);
    const DeviceBuffer<float> invRmsBuffer(in.rows, invRms);
    const DeviceBuffer<float> dx(count);
    const DeviceBuffer<float> dscale(in.scale.size());
    const DeviceBuffer<float> dbias(in.scale.size());
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{X_UID, x.get()},
           {SCALE_UID, scale.get()},
           {INV_RMS_UID, invRmsBuffer.get()},
           {DX_UID, dx.get()},
           {DSCALE_UID, dscale.get()},
           {fused(spec) ? GRADIENT_UID : DY_UID, dy.get()}};
    if(!preActivation.empty())
    {
        buffers.push_back({PRE_ACTIVATION_UID, preActivationBuffer.get()});
    }
    if(spec.withBias)
    {
        buffers.push_back({DBIAS_UID, dbias.get()});
    }

    EngineRun run;
    ASSERT_NO_FATAL_FAILURE(executeThroughTheEngine(buildRmsnormGraph(spec), buffers, &run));
    EXPECT_EQ(run.workspaceBytes, 0U);
    EXPECT_EQ(run.maxWorkspaceBytes, 0U);

    expectNear(dx.read(), expected.dx, "dx");
    expectNear(dscale.read(), expected.dscale, "dscale");
    if(spec.withBias)
    {
        expectNear(dbias.read(), expected.dbias, "dbias");
    }
}

TEST(TestHipMlopsRmsnormDevice, ForwardMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    for(const bool withOptionals : {true, false})
    {
        SCOPED_TRACE(withOptionals ? "with bias and inverse RMS" : "without");
        RmsnormGraph spec;
        spec.withBias = withOptionals;
        spec.withInvRms = withOptionals;
        checkForward(spec);
    }
}

TEST(TestHipMlopsRmsnormDevice, FusedForwardAppliesTheActivation)
{
    SKIP_IF_NO_DEVICES();

    // The clipped ReLU's clip is a launch argument rather than a compile option.
    for(const auto activation : {Activation::RELU, Activation::CLIPPED_RELU})
    {
        SCOPED_TRACE(activation == Activation::RELU ? "relu" : "clipped relu");
        RmsnormGraph spec;
        spec.activation = activation;
        checkForward(spec);
    }
}

TEST(TestHipMlopsRmsnormDevice, BackwardMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    for(const bool withBias : {true, false})
    {
        SCOPED_TRACE(withBias ? "with dbias" : "without dbias");
        RmsnormGraph spec;
        spec.backward = true;
        spec.withBias = withBias;
        checkBackward(spec);
    }
}

TEST(TestHipMlopsRmsnormDevice, FusedBackwardAppliesTheActivationGradient)
{
    SKIP_IF_NO_DEVICES();

    for(const auto activation : {Activation::RELU, Activation::CLIPPED_RELU, Activation::IDENTITY})
    {
        SCOPED_TRACE(static_cast<int>(activation));
        RmsnormGraph spec;
        spec.backward = true;
        spec.activation = activation;
        checkBackward(spec);
    }
}

} // namespace

#endif // HIPDNN_ENGINE_HIP_MLOPS_INGESTOR
