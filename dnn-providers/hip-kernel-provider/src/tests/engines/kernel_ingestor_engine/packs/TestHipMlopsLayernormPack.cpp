// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENGINE_HIP_MLOPS_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/HipMlopsTestSupport.hpp"

/**
 * @file TestHipMlopsLayernormPack.cpp
 * @brief The HIP MLOps engine on the kernel ingestor, Layernorm family (ALMIOPEN-3071): the
 *        engine accepts what the hand-written builder accepts, and computes forward and
 *        backward layernorm through the engine as CPU references do -- the backward on both
 *        of its launch sequences, with the graph's epsilon and with the default.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing::hip_mlops;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr int64_t X_UID = 1;
constexpr int64_t Y_UID = 2;
constexpr int64_t DY_UID = 3;
constexpr int64_t DX_UID = 4;
constexpr int64_t SCALE_UID = 5;
constexpr int64_t BIAS_UID = 6;
constexpr int64_t DSCALE_UID = 7;
constexpr int64_t DBIAS_UID = 8;
constexpr int64_t MEAN_UID = 9;
constexpr int64_t INV_VARIANCE_UID = 10;
constexpr int64_t EPSILON_UID = 11;

constexpr double DEFAULT_EPSILON = 1e-5;

/// A layernorm over every dimension after the first: statistics per row of `dims[0]`.
struct LayernormGraph
{
    bool backward = false;
    // Inner size 2400, more than one workgroup's 1024 lanes.
    std::vector<int64_t> dims = {3, 5, 16, 30};
    // Forward: the mean and inverse variance outputs. Backward: those inputs.
    bool withStatistics = true;
    // Backward only; the forward always has one.
    bool withEpsilon = true;
    float epsilon = 1e-2f;
    bool overrideShape = false;
    bool xVirtual = false;
    bool xUnpacked = false;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    bool twoNodes = false;
};

std::vector<int64_t> normalizedDims(const LayernormGraph& spec)
{
    return {spec.dims.begin() + 1, spec.dims.end()};
}

std::vector<int64_t> statisticDims(const LayernormGraph& spec)
{
    std::vector<int64_t> dims(spec.dims.size(), 1);
    dims[0] = spec.dims[0];
    return dims;
}

flatbuffers::FlatBufferBuilder buildLayernormGraph(const LayernormGraph& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto ioStrides = packedStrides(spec.dims);
    const auto affineDims = normalizedDims(spec);
    const auto affineStrides = packedStrides(affineDims);
    const auto statDims = statisticDims(spec);
    const auto statStrides = packedStrides(statDims);
    // Each row padded by one element.
    auto paddedDims = spec.dims;
    paddedDims.back() += 1;
    const auto xStrides = spec.xUnpacked ? packedStrides(paddedDims) : ioStrides;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    const auto addTensor = [&](int64_t uid,
                               const std::vector<int64_t>& dims,
                               const std::vector<int64_t>& strides,
                               bool isVirtual = false) {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, uid, nullptr, data_objects::DataType::FLOAT, &strides, &dims, isVirtual));
    };
    addTensor(X_UID, spec.dims, xStrides, spec.xVirtual);
    addTensor(SCALE_UID, affineDims, affineStrides);
    if(spec.backward)
    {
        addTensor(DY_UID, spec.dims, ioStrides);
        addTensor(DX_UID, spec.dims, ioStrides);
        addTensor(DSCALE_UID, affineDims, affineStrides);
        addTensor(DBIAS_UID, affineDims, affineStrides);
    }
    else
    {
        addTensor(Y_UID, spec.dims, ioStrides);
        addTensor(BIAS_UID, affineDims, affineStrides);
    }
    if(spec.withStatistics)
    {
        addTensor(MEAN_UID, statDims, statStrides);
        addTensor(INV_VARIANCE_UID, statDims, statStrides);
    }
    const bool withEpsilon = !spec.backward || spec.withEpsilon;
    if(withEpsilon)
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

    const auto optionalUid = [](bool present, int64_t uid) {
        return present ? flatbuffers::Optional<int64_t>(uid) : flatbuffers::nullopt;
    };
    const auto normalizedDimCount = static_cast<int64_t>(spec.dims.size() - 1);
    flatbuffers::Offset<void> attributes;
    data_objects::NodeAttributes type{};
    if(spec.backward)
    {
        attributes = data_objects::CreateLayernormBackwardAttributes(
                         builder,
                         DY_UID,
                         X_UID,
                         SCALE_UID,
                         optionalUid(spec.withStatistics, MEAN_UID),
                         optionalUid(spec.withStatistics, INV_VARIANCE_UID),
                         optionalUid(withEpsilon, EPSILON_UID),
                         DX_UID,
                         DSCALE_UID,
                         DBIAS_UID,
                         normalizedDimCount)
                         .Union();
        type = data_objects::NodeAttributes::LayernormBackwardAttributes;
    }
    else
    {
        attributes = data_objects::CreateLayernormAttributes(
                         builder,
                         X_UID,
                         SCALE_UID,
                         BIAS_UID,
                         EPSILON_UID,
                         Y_UID,
                         normalizedDimCount,
                         optionalUid(spec.withStatistics, MEAN_UID),
                         optionalUid(spec.withStatistics, INV_VARIANCE_UID))
                         .Union();
        type = data_objects::NodeAttributes::LayernormAttributes;
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder, "layernorm", spec.computeType, type, attributes));
    if(spec.twoNodes)
    {
        nodes.push_back(data_objects::CreateNodeDirect(
            builder, "layernorm2", spec.computeType, type, attributes));
    }

    auto name = builder.CreateString("hip_mlops_layernorm");
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

auto match(const LayernormGraph& spec)
{
    return matchGraph(buildLayernormGraph(spec));
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

TEST(TestHipMlopsLayernormGraphMatch, RoutesEachDirectionToItsPack)
{
    EXPECT_EQ(boundOperation(match({})), "LAYERNORM_FWD");

    LayernormGraph withoutStatistics;
    withoutStatistics.withStatistics = false;
    EXPECT_EQ(boundOperation(match(withoutStatistics)), "LAYERNORM_FWD");

    LayernormGraph backward;
    backward.backward = true;
    EXPECT_EQ(boundOperation(match(backward)), "LAYERNORM_BWD");

    // Every optional backward operand absent.
    backward.withStatistics = false;
    backward.withEpsilon = false;
    EXPECT_EQ(boundOperation(match(backward)), "LAYERNORM_BWD");
}

struct RefusalCase
{
    const char* name;
    LayernormGraph spec;
};

LayernormGraph edited(void (*edit)(LayernormGraph&))
{
    LayernormGraph spec;
    edit(spec);
    return spec;
}

class TestHipMlopsLayernormRefusal : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestHipMlopsLayernormRefusal, Declines)
{
    EXPECT_FALSE(match(GetParam().spec).has_value());
}

// The hand-written builder's refusals, each in both directions where it applies.
INSTANTIATE_TEST_SUITE_P(
    ,
    TestHipMlopsLayernormRefusal,
    ::testing::Values(RefusalCase{"OverrideShapes",
                                  edited([](auto& s) { s.overrideShape = true; })},
                      RefusalCase{"TwoNodes", edited([](auto& s) { s.twoNodes = true; })},
                      RefusalCase{"HalfCompute", edited([](auto& s) {
                                      s.computeType = data_objects::DataType::HALF;
                                  })},
                      RefusalCase{"BackwardHalfCompute", edited([](auto& s) {
                                      s.backward = true;
                                      s.computeType = data_objects::DataType::HALF;
                                  })},
                      RefusalCase{"VirtualInput", edited([](auto& s) { s.xVirtual = true; })},
                      RefusalCase{"BackwardVirtualInput", edited([](auto& s) {
                                      s.backward = true;
                                      s.xVirtual = true;
                                  })},
                      RefusalCase{"UnpackedInput", edited([](auto& s) { s.xUnpacked = true; })},
                      RefusalCase{"BackwardUnpackedInput", edited([](auto& s) {
                                      s.backward = true;
                                      s.xUnpacked = true;
                                  })}),
    [](const ::testing::TestParamInfo<RefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Numerics through the engine
// ---------------------------------------------------------------------------

/// Per-row statistics and normalized values, in double.
struct Statistics
{
    std::vector<double> mean;
    std::vector<double> invVariance;
};

Statistics statistics(const std::vector<float>& x, size_t rows, size_t inner, double epsilon)
{
    Statistics result{std::vector<double>(rows), std::vector<double>(rows)};
    for(size_t r = 0; r < rows; ++r)
    {
        double sum = 0.0;
        for(size_t i = 0; i < inner; ++i)
        {
            sum += static_cast<double>(x[r * inner + i]);
        }
        const double mean = sum / static_cast<double>(inner);
        double squares = 0.0;
        for(size_t i = 0; i < inner; ++i)
        {
            const double centred = static_cast<double>(x[r * inner + i]) - mean;
            squares += centred * centred;
        }
        result.mean[r] = mean;
        result.invVariance[r] = 1.0 / std::sqrt(squares / static_cast<double>(inner) + epsilon);
    }
    return result;
}

struct ForwardReference
{
    std::vector<float> y;
    std::vector<float> mean;
    std::vector<float> invVariance;
};

ForwardReference forwardReference(const std::vector<float>& x,
                                  const std::vector<float>& scale,
                                  const std::vector<float>& bias,
                                  size_t rows,
                                  double epsilon)
{
    const size_t inner = scale.size();
    const auto stats = statistics(x, rows, inner, epsilon);
    ForwardReference result{std::vector<float>(x.size()), {}, {}};
    for(size_t r = 0; r < rows; ++r)
    {
        for(size_t i = 0; i < inner; ++i)
        {
            const double normalized
                = (static_cast<double>(x[r * inner + i]) - stats.mean[r]) * stats.invVariance[r];
            result.y[r * inner + i] = static_cast<float>(normalized * static_cast<double>(scale[i])
                                                         + static_cast<double>(bias[i]));
        }
        result.mean.push_back(static_cast<float>(stats.mean[r]));
        result.invVariance.push_back(static_cast<float>(stats.invVariance[r]));
    }
    return result;
}

struct BackwardReference
{
    std::vector<float> dx;
    std::vector<float> dscale;
    std::vector<float> dbias;
};

/// dx = invVariance * (g - mean(g) - xhat * mean(g * xhat)) with g = dy * scale, and the
/// affine gradients summed over rows.
BackwardReference backwardReference(const std::vector<float>& dy,
                                    const std::vector<float>& x,
                                    const std::vector<float>& scale,
                                    const Statistics& stats)
{
    const size_t inner = scale.size();
    const size_t rows = stats.mean.size();
    std::vector<double> dscale(inner, 0.0);
    std::vector<double> dbias(inner, 0.0);
    BackwardReference result{std::vector<float>(x.size()), {}, {}};
    for(size_t r = 0; r < rows; ++r)
    {
        double sumG = 0.0;
        double sumGXhat = 0.0;
        for(size_t i = 0; i < inner; ++i)
        {
            const size_t at = r * inner + i;
            const double xhat = (static_cast<double>(x[at]) - stats.mean[r]) * stats.invVariance[r];
            const double g = static_cast<double>(dy[at]) * static_cast<double>(scale[i]);
            sumG += g;
            sumGXhat += g * xhat;
            dscale[i] += static_cast<double>(dy[at]) * xhat;
            dbias[i] += static_cast<double>(dy[at]);
        }
        const double meanG = sumG / static_cast<double>(inner);
        const double meanGXhat = sumGXhat / static_cast<double>(inner);
        for(size_t i = 0; i < inner; ++i)
        {
            const size_t at = r * inner + i;
            const double xhat = (static_cast<double>(x[at]) - stats.mean[r]) * stats.invVariance[r];
            const double g = static_cast<double>(dy[at]) * static_cast<double>(scale[i]);
            result.dx[at]
                = static_cast<float>(stats.invVariance[r] * (g - meanG - xhat * meanGXhat));
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
    std::vector<float> dy;
};

Inputs inputsFor(const LayernormGraph& spec)
{
    const size_t count = elementCount(spec.dims);
    const size_t inner = elementCount(normalizedDims(spec));
    // Scale and bias around 1 and 0 rather than straddling zero.
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
            std::move(bias),
            deterministicValues(count, 1.9f)};
}

/// Runs the backward of @p spec through the engine against the reference. With
/// statistics in the graph, they are the reference forward's; without, the kernel
/// recomputes them with @p expectedEpsilon.
EngineRun checkBackward(const LayernormGraph& spec, double expectedEpsilon)
{
    const auto in = inputsFor(spec);
    const size_t count = in.x.size();
    const auto stats = statistics(in.x, in.rows, in.scale.size(), expectedEpsilon);
    const auto expected = backwardReference(in.dy, in.x, in.scale, stats);

    const std::vector<float> mean(stats.mean.begin(), stats.mean.end());
    const std::vector<float> invVariance(stats.invVariance.begin(), stats.invVariance.end());
    const DeviceBuffer<float> dy(count, in.dy);
    const DeviceBuffer<float> x(count, in.x);
    const DeviceBuffer<float> scale(in.scale.size(), in.scale);
    const DeviceBuffer<float> meanBuffer(in.rows, mean);
    const DeviceBuffer<float> invVarianceBuffer(in.rows, invVariance);
    const DeviceBuffer<float> dx(count);
    const DeviceBuffer<float> dscale(in.scale.size());
    const DeviceBuffer<float> dbias(in.scale.size());

    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{DY_UID, dy.get()},
                                                       {X_UID, x.get()},
                                                       {SCALE_UID, scale.get()},
                                                       {DX_UID, dx.get()},
                                                       {DSCALE_UID, dscale.get()},
                                                       {DBIAS_UID, dbias.get()}};
    if(spec.withStatistics)
    {
        buffers.push_back({MEAN_UID, meanBuffer.get()});
        buffers.push_back({INV_VARIANCE_UID, invVarianceBuffer.get()});
    }

    EngineRun run;
    EXPECT_NO_FATAL_FAILURE(executeThroughTheEngine(buildLayernormGraph(spec), buffers, &run));
    EXPECT_EQ(run.workspaceBytes, run.maxWorkspaceBytes);

    expectNear(dx.read(), expected.dx, "dx");
    expectNear(dscale.read(), expected.dscale, "dscale");
    expectNear(dbias.read(), expected.dbias, "dbias");
    return run;
}

TEST(TestHipMlopsLayernormDevice, ForwardMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    for(const bool withStatistics : {true, false})
    {
        SCOPED_TRACE(withStatistics ? "with statistics" : "without statistics");
        LayernormGraph spec;
        spec.withStatistics = withStatistics;
        const auto in = inputsFor(spec);
        const size_t count = in.x.size();
        // The graph's epsilon, not the default, reaches the kernel.
        const auto expected
            = forwardReference(in.x, in.scale, in.bias, in.rows, static_cast<double>(spec.epsilon));

        const DeviceBuffer<float> x(count, in.x);
        const DeviceBuffer<float> y(count);
        const DeviceBuffer<float> scale(in.scale.size(), in.scale);
        const DeviceBuffer<float> bias(in.bias.size(), in.bias);
        const DeviceBuffer<float> mean(in.rows);
        const DeviceBuffer<float> invVariance(in.rows);
        std::vector<hipdnnPluginDeviceBuffer_t> buffers = {
            {X_UID, x.get()}, {Y_UID, y.get()}, {SCALE_UID, scale.get()}, {BIAS_UID, bias.get()}};
        if(withStatistics)
        {
            buffers.push_back({MEAN_UID, mean.get()});
            buffers.push_back({INV_VARIANCE_UID, invVariance.get()});
        }

        EngineRun run;
        ASSERT_NO_FATAL_FAILURE(executeThroughTheEngine(buildLayernormGraph(spec), buffers, &run));
        EXPECT_EQ(run.workspaceBytes, 0U);
        EXPECT_EQ(run.maxWorkspaceBytes, 0U);

        expectNear(y.read(), expected.y, "y");
        if(withStatistics)
        {
            expectNear(mean.read(), expected.mean, "mean");
            expectNear(invVariance.read(), expected.invVariance, "invVariance");
        }
    }
}

TEST(TestHipMlopsLayernormDevice, BackwardWithStatisticsMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    LayernormGraph spec;
    spec.backward = true;
    const auto run = checkBackward(spec, static_cast<double>(spec.epsilon));
    // Given statistics and few rows: one pass for dscale and dbias, no workspace.
    EXPECT_EQ(run.workspaceBytes, 0U);
}

TEST(TestHipMlopsLayernormDevice, BackwardRecomputesStatisticsWithTheGraphsEpsilon)
{
    SKIP_IF_NO_DEVICES();

    LayernormGraph spec;
    spec.backward = true;
    spec.withStatistics = false;
    const auto run = checkBackward(spec, static_cast<double>(spec.epsilon));
    EXPECT_GT(run.workspaceBytes, 0U);
}

TEST(TestHipMlopsLayernormDevice, BackwardWithoutEpsilonUsesTheDefault)
{
    SKIP_IF_NO_DEVICES();

    LayernormGraph spec;
    spec.backward = true;
    spec.withStatistics = false;
    spec.withEpsilon = false;
    checkBackward(spec, DEFAULT_EPSILON);
}

TEST(TestHipMlopsLayernormDevice, BackwardOnTheParallelPathMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    // The backward reduces dscale and dbias in parallel partial sums once the rows hold
    // more than 4 * 1024 work items per compute unit (LayernormBwdPlan.cpp); size the batch
    // just past that on this device.
    int device = 0;
    int computeUnits = 0;
    ASSERT_EQ(hipSuccess, hipGetDevice(&device));
    ASSERT_EQ(hipSuccess,
              hipDeviceGetAttribute(&computeUnits, hipDeviceAttributeMultiprocessorCount, device));

    LayernormGraph spec;
    spec.backward = true;
    spec.dims = {4 * static_cast<int64_t>(computeUnits) + 1, 4, 16, 16};
    const auto run = checkBackward(spec, static_cast<double>(spec.epsilon));
    // Given statistics, only the parallel partial sums need a workspace.
    EXPECT_GT(run.workspaceBytes, 0U);
}

} // namespace

#endif // HIPDNN_ENGINE_HIP_MLOPS_INGESTOR
