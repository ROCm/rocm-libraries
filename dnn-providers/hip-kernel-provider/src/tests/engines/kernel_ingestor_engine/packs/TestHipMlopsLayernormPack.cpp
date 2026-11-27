// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENGINE_HIP_MLOPS_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/HipMlopsTestSupport.hpp"

/**
 * @file TestHipMlopsLayernormPack.cpp
 * @brief The HIP MLOps Layernorm engines on the kernel ingestor (ALMIOPEN-3071):
 *        hipkernel:LayernormFwd and hipkernel:LayernormBwd each accept what the hand-written
 *        builder accepts in their direction (up to the same launch thread limit), and compute
 *        forward and backward layernorm as CPU references do -- the backward on both of its
 *        launch sequences, with the graph's epsilon and with the default.
 *
 *        hipkernel:LayernormFwd ships a kernel per workgroup size (ALMIOPEN-3066), addressed
 *        by the knob `local_size`, and hipkernel:LayernormBwd a kernel per scale/bias
 *        reduction and workgroup size, addressed by `parallel` and `local_size`: the
 *        hand-written plans' configuration ranks first, a reduction whose launches do not fit
 *        the device is not admitted, the workspace follows the reduction selected and not the
 *        workgroup, and each configuration, pinned through the knobs, computes as the
 *        reference does.
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

/// The engine that serves @p spec's direction.
const MlopsEngine& engineFor(const LayernormGraph& spec)
{
    return spec.backward ? LAYERNORM_BWD_ENGINE : LAYERNORM_FWD_ENGINE;
}

auto match(const LayernormGraph& spec)
{
    return matchGraph(engineFor(spec), buildLayernormGraph(spec));
}

/// @p spec's graph reaches its engine alone, which binds @p operation.
void expectServedAlone(const LayernormGraph& spec, const char* operation)
{
    EXPECT_THAT(acceptingEngines(buildLayernormGraph(spec)),
                ::testing::ElementsAre(engineFor(spec).name));
    EXPECT_EQ(boundOperation(match(spec)), operation);
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

TEST(TestHipMlopsLayernormPack, EachDirectionIsOfferedUnderItsOwnNameAndId)
{
    expectOfferedUnderItsOwnNameAndId(LAYERNORM_FWD_ENGINE);
    expectOfferedUnderItsOwnNameAndId(LAYERNORM_BWD_ENGINE);
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

TEST(TestHipMlopsLayernormGraphMatch, RoutesEachDirectionToItsEngineAlone)
{
    expectServedAlone({}, "LAYERNORM_FWD");

    LayernormGraph withoutStatistics;
    withoutStatistics.withStatistics = false;
    expectServedAlone(withoutStatistics, "LAYERNORM_FWD");

    LayernormGraph backward;
    backward.backward = true;
    expectServedAlone(backward, "LAYERNORM_BWD");

    // Every optional backward operand absent.
    backward.withStatistics = false;
    backward.withEpsilon = false;
    expectServedAlone(backward, "LAYERNORM_BWD");
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

TEST_P(TestHipMlopsLayernormRefusal, EveryEngineDeclines)
{
    EXPECT_THAT(acceptingEngines(buildLayernormGraph(GetParam().spec)), ::testing::IsEmpty());
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

// The hand-written builder's boundary, on its graphs (TestLayernormPlanBuilder): both
// directions launch outerSize * stride blocks of 1024 threads, so 2^22 - 1 rows fit and 2^22
// rows reach 2^32 threads.
TEST(TestHipMlopsLayernormGraphMatch, ServesRowCountsBelowTheThreadLimit)
{
    EXPECT_THAT(acceptingEngines(hipdnn_test_sdk::utilities::createValidLayernormFpropGraph(
                    {4, 1, 1, 1}, {4194303, 4, 1, 1})),
                ::testing::ElementsAre(LAYERNORM_FWD_ENGINE.name));
    EXPECT_THAT(acceptingEngines(hipdnn_test_sdk::utilities::createValidLayernormBwdGraph(
                    {4, 1, 1, 1}, {4194303, 4, 1, 1})),
                ::testing::ElementsAre(LAYERNORM_BWD_ENGINE.name));
}

// Declined by the graph match, as by the hand-written builder, rather than failing at prepare.
TEST(TestHipMlopsLayernormGraphMatch, DeclinesRowCountsAtTheThreadLimit)
{
    EXPECT_THAT(acceptingEngines(hipdnn_test_sdk::utilities::createValidLayernormFpropGraph(
                    {4, 1, 1, 1}, {4194304, 4, 1, 1})),
                ::testing::IsEmpty());
    EXPECT_THAT(acceptingEngines(hipdnn_test_sdk::utilities::createValidLayernormBwdGraph(
                    {4, 1, 1, 1}, {4194304, 4, 1, 1})),
                ::testing::IsEmpty());
}

// ---------------------------------------------------------------------------
// Knob: one forward kernel per workgroup size
// ---------------------------------------------------------------------------

/// The workgroup sizes both directions ship a kernel for; the hand-written plans' is 1024.
const std::set<int64_t> LOCAL_SIZES = {256, 512, 1024};
constexpr int64_t DEFAULT_LOCAL_SIZE = 1024;

const std::vector<std::string> FWD_KNOBS = {"local_size"};

TEST(TestHipMlopsLayernormFwdKnob, TheForwardShipsAKernelPerWorkgroupSize)
{
    EXPECT_EQ(engineKnobs(LAYERNORM_FWD_ENGINE), FWD_KNOBS);
    std::set<int64_t> localSizes;
    for(const auto& kernel : loadedKernels(LAYERNORM_FWD_ENGINE))
    {
        EXPECT_TRUE(localSizes.insert(kernel.getIntMetadata("local_size")).second) << kernel.name;
        EXPECT_EQ(kernel.source.entryPoint, "LayernormFwd");
    }
    EXPECT_EQ(localSizes, LOCAL_SIZES);
}

// Every workgroup runs any graph the engine accepts: the default, the widest, launches the
// most threads, and the graph match has checked its launch. The hand-written plan's ranks
// first, on a row wider than every workgroup and on one narrower than every workgroup.
TEST(TestHipMlopsLayernormFwdKnob, TheHandWrittenWorkgroupRanksFirst)
{
    for(const auto& dims : {std::vector<int64_t>{3, 5, 16, 30}, std::vector<int64_t>{8, 2, 3, 5}})
    {
        SCOPED_TRACE(elementCount(dims));
        LayernormGraph spec;
        spec.dims = dims;
        const auto kernels = rankedKernels(engineFor(spec), buildLayernormGraph(spec));
        expectUniqueTuples(kernels, FWD_KNOBS);
        const auto offered = knobValues(kernels, "local_size");
        ASSERT_EQ(offered.size(), LOCAL_SIZES.size());
        EXPECT_EQ(offered.front(), DEFAULT_LOCAL_SIZE);
        EXPECT_EQ(std::set<int64_t>(offered.begin(), offered.end()), LOCAL_SIZES);
    }
}

TEST(TestHipMlopsLayernormFwdKnob, PinningAWorkgroupBuildsIt)
{
    const LayernormGraph spec{};
    const auto kernels = rankedKernels(engineFor(spec), buildLayernormGraph(spec));
    for(const int64_t localSize : LOCAL_SIZES)
    {
        EXPECT_EQ(pinnedTuple(kernels, FWD_KNOBS, {{"local_size", localSize}}),
                  (KnobSettings{{"local_size", localSize}}));
    }
}

// ---------------------------------------------------------------------------
// Knob: one backward kernel per scale/bias reduction and workgroup size
// ---------------------------------------------------------------------------

const std::vector<std::string> KNOBS = {"parallel", "local_size"};

/// A device of 4 CUs: a parallel pass keeps 4 * 1024 * 4 = 16384 work items busy.
constexpr int COMPUTE_UNITS = 4;

LayernormGraph backwardOf(std::vector<int64_t> dims, bool withStatistics = true)
{
    LayernormGraph spec;
    spec.backward = true;
    spec.dims = std::move(dims);
    spec.withStatistics = withStatistics;
    return spec;
}

TEST(TestHipMlopsLayernormBwdKnob, TheBackwardShipsAKernelPerReductionAndWorkgroupSize)
{
    EXPECT_EQ(engineKnobs(LAYERNORM_BWD_ENGINE), KNOBS);
    std::set<KnobSettings> tuples;
    for(const auto& kernel : loadedKernels(LAYERNORM_BWD_ENGINE))
    {
        EXPECT_TRUE(tuples.insert(knobTuple(kernel, KNOBS)).second) << kernel.name;
        EXPECT_EQ(kernel.source.entryPoint, "LayernormBwd");
    }
    std::set<KnobSettings> expected;
    for(const int64_t parallel : {0, 1})
    {
        for(const int64_t localSize : LOCAL_SIZES)
        {
            expected.insert(KnobSettings{{"parallel", parallel}, {"local_size", localSize}});
        }
    }
    EXPECT_EQ(tuples, expected);
}

struct ReductionCase
{
    const char* name;
    std::vector<int64_t> dims;
    int64_t expected;
};

class TestHipMlopsLayernormBwdDefault : public ::testing::TestWithParam<ReductionCase>
{
};

// Both reductions run any graph at every workgroup, so on these devices the knobs offer them
// all; the launch heuristics' choice (defaultLayernormBwdConfig()) ranks first, at the
// hand-written plan's workgroup.
TEST_P(TestHipMlopsLayernormBwdDefault, TheHeuristicsReductionRanksFirst)
{
    for(const bool withStatistics : {true, false})
    {
        SCOPED_TRACE(withStatistics ? "with statistics" : "without statistics");
        const auto spec = backwardOf(GetParam().dims, withStatistics);
        const auto kernels = rankedKernels(engineFor(spec),
                                           buildLayernormGraph(spec),
                                           devicePropertiesWithComputeUnits(COMPUTE_UNITS));
        expectUniqueTuples(kernels, KNOBS);
        const auto offered = knobValues(kernels, "parallel");
        ASSERT_EQ(offered.size(), 2U);
        EXPECT_EQ(offered.front(), GetParam().expected);
        EXPECT_THAT(offered, ::testing::UnorderedElementsAre(0, 1));
        const auto localSizes = knobValues(kernels, "local_size");
        ASSERT_FALSE(localSizes.empty());
        EXPECT_EQ(localSizes.front(), DEFAULT_LOCAL_SIZE);
        EXPECT_EQ(std::set<int64_t>(localSizes.begin(), localSizes.end()), LOCAL_SIZES);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestHipMlopsLayernormBwdDefault,
    ::testing::Values(
        // 3 rows of 2400: the whole tensor fits one parallel pass.
        ReductionCase{"FewRowsOnePass", {3, 5, 16, 30}, 0},
        // 8 rows of 2400 exceed it while a row does not: split across workgroups.
        ReductionCase{"ManyRowsParallel", {8, 5, 16, 30}, 1},
        // A row of 20480 alone exceeds it.
        ReductionCase{"WideRowsOnePass", {2, 5, 64, 64}, 0}),
    [](const ::testing::TestParamInfo<ReductionCase>& info) { return info.param.name; });

// The graph match checks the launches every reduction makes alike, as the hand-written builder
// does; the parallel reduction's grid grows with the device. The layernorm validator caps a
// tensor below 2^32 elements, so the widest reachable case is 2 rows of 2^31 - 1: on 4 CUs a row
// exceeds one parallel pass and the parallel reduction launches 2^31 - 1 threads (rounded up to
// whole blocks); on 2^22 CUs its parallel size grows to 2 and its grid rounds up to 2^32 threads,
// so only the one-pass kernel is admitted.
TEST(TestHipMlopsLayernormBwdKnob, AReductionReachingTheThreadLimitIsNotAdmitted)
{
    const auto graph = [] {
        return hipdnn_test_sdk::utilities::createValidLayernormBwdGraph({2147483647, 1, 1, 1},
                                                                        {2, 2147483647, 1, 1});
    };
    EXPECT_THAT(knobValues(rankedKernels(LAYERNORM_BWD_ENGINE,
                                         graph(),
                                         devicePropertiesWithComputeUnits(COMPUTE_UNITS)),
                           "parallel"),
                ::testing::UnorderedElementsAre(0, 1));
    EXPECT_THAT(
        knobValues(
            rankedKernels(LAYERNORM_BWD_ENGINE, graph(), devicePropertiesWithComputeUnits(1 << 22)),
            "parallel"),
        ::testing::ElementsAre(0));
}

/// The workspace the backward's handler reports for its kernel of reduction @p parallel and
/// workgroup @p localSize on @p spec's graph, on a device of COMPUTE_UNITS CUs: what a plan of
/// that kernel allocates.
size_t workspaceOf(const LayernormGraph& spec,
                   int64_t parallel,
                   int64_t localSize = DEFAULT_LOCAL_SIZE)
{
    for(const auto& kernel : loadedKernels(LAYERNORM_BWD_ENGINE))
    {
        if(kernel.getIntMetadata("parallel") == parallel
           && kernel.getIntMetadata("local_size") == localSize)
        {
            return workspaceBytesOf(LAYERNORM_BWD_ENGINE,
                                    buildLayernormGraph(spec),
                                    kernel,
                                    devicePropertiesWithComputeUnits(COMPUTE_UNITS));
        }
    }
    ADD_FAILURE() << "no kernel of reduction " << parallel << " and workgroup " << localSize;
    return 0;
}

TEST(TestHipMlopsLayernormBwdKnob, TheWorkspaceFollowsTheSelectedReduction)
{
    // 3 rows of 2400 on 4 CUs: the parallel reduction keeps 2 partial sums of dscale and dbias
    // per column, 4 * 2 * 2400 * 2 bytes; recomputing the statistics adds a mean and an
    // inverse variance per row, 4 * 2 * 3 bytes, to either.
    EXPECT_EQ(workspaceOf(backwardOf({3, 5, 16, 30}), 0), 0U);
    EXPECT_EQ(workspaceOf(backwardOf({3, 5, 16, 30}), 1), 38400U);
    EXPECT_EQ(workspaceOf(backwardOf({3, 5, 16, 30}, false), 0), 24U);
    EXPECT_EQ(workspaceOf(backwardOf({3, 5, 16, 30}, false), 1), 38424U);
    // The parallel size is sized for the widest workgroup at every workgroup, so the
    // workspace does not follow the workgroup.
    for(const int64_t localSize : LOCAL_SIZES)
    {
        SCOPED_TRACE(localSize);
        EXPECT_EQ(workspaceOf(backwardOf({3, 5, 16, 30}), 0, localSize), 0U);
        EXPECT_EQ(workspaceOf(backwardOf({3, 5, 16, 30}), 1, localSize), 38400U);
        EXPECT_EQ(workspaceOf(backwardOf({3, 5, 16, 30}, false), 1, localSize), 38424U);
    }
}

TEST(TestHipMlopsLayernormBwdKnob, PinningAReductionBuildsItAtTheHandWrittenWorkgroup)
{
    const auto spec = backwardOf({3, 5, 16, 30});
    const auto kernels = rankedKernels(engineFor(spec),
                                       buildLayernormGraph(spec),
                                       devicePropertiesWithComputeUnits(COMPUTE_UNITS));
    for(const int64_t parallel : {0, 1})
    {
        EXPECT_EQ(pinnedTuple(kernels, KNOBS, {{"parallel", parallel}}),
                  (KnobSettings{{"parallel", parallel}, {"local_size", DEFAULT_LOCAL_SIZE}}));
    }
}

TEST(TestHipMlopsLayernormBwdKnob, PinningBothKnobsBuildsThatConfiguration)
{
    const auto spec = backwardOf({3, 5, 16, 30});
    const auto kernels = rankedKernels(engineFor(spec),
                                       buildLayernormGraph(spec),
                                       devicePropertiesWithComputeUnits(COMPUTE_UNITS));
    for(const int64_t parallel : {0, 1})
    {
        for(const int64_t localSize : LOCAL_SIZES)
        {
            const KnobSettings pins = {{"parallel", parallel}, {"local_size", localSize}};
            EXPECT_EQ(pinnedTuple(kernels, KNOBS, pins), pins);
        }
    }
}

// Pinning only the workgroup keeps the launch heuristics' reduction, whichever it is.
TEST(TestHipMlopsLayernormBwdKnob, PinningTheWorkgroupKeepsTheHeuristicsReduction)
{
    for(const auto& [dims, heuristic] : {std::make_pair(std::vector<int64_t>{3, 5, 16, 30}, 0),
                                         std::make_pair(std::vector<int64_t>{8, 5, 16, 30}, 1)})
    {
        SCOPED_TRACE(heuristic == 1 ? "parallel" : "one pass");
        const auto spec = backwardOf(dims);
        const auto kernels = rankedKernels(engineFor(spec),
                                           buildLayernormGraph(spec),
                                           devicePropertiesWithComputeUnits(COMPUTE_UNITS));
        for(const int64_t localSize : LOCAL_SIZES)
        {
            EXPECT_EQ(pinnedTuple(kernels, KNOBS, {{"local_size", localSize}}),
                      (KnobSettings{{"parallel", heuristic}, {"local_size", localSize}}));
        }
    }
}

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

/// Runs the backward of @p spec through the engine, with @p knobs pinned, against the
/// reference. With statistics in the graph, they are the reference forward's; without, the
/// kernel recomputes them with @p expectedEpsilon.
EngineRun checkBackward(const LayernormGraph& spec,
                        double expectedEpsilon,
                        const KnobSettings& knobs = {})
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
    EXPECT_NO_FATAL_FAILURE(
        executeThroughTheEngine(engineFor(spec), buildLayernormGraph(spec), buffers, &run, knobs));
    // Unpinned, the engine's maximum covers both reductions; pinned, only the one built.
    if(knobs.empty())
    {
        EXPECT_LE(run.workspaceBytes, run.maxWorkspaceBytes);
    }
    else
    {
        EXPECT_EQ(run.workspaceBytes, run.maxWorkspaceBytes);
    }

    expectNear(dx.read(), expected.dx, "dx");
    expectNear(dscale.read(), expected.dscale, "dscale");
    expectNear(dbias.read(), expected.dbias, "dbias");
    return run;
}

/// Runs the forward of @p spec through the engine, with @p knobs pinned, against the
/// reference.
void checkForward(const LayernormGraph& spec, const KnobSettings& knobs = {})
{
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
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{X_UID, x.get()}, {Y_UID, y.get()}, {SCALE_UID, scale.get()}, {BIAS_UID, bias.get()}};
    if(spec.withStatistics)
    {
        buffers.push_back({MEAN_UID, mean.get()});
        buffers.push_back({INV_VARIANCE_UID, invVariance.get()});
    }

    EngineRun run;
    ASSERT_NO_FATAL_FAILURE(
        executeThroughTheEngine(engineFor(spec), buildLayernormGraph(spec), buffers, &run, knobs));
    EXPECT_EQ(run.workspaceBytes, 0U);
    EXPECT_EQ(run.maxWorkspaceBytes, 0U);

    expectNear(y.read(), expected.y, "y");
    if(spec.withStatistics)
    {
        expectNear(mean.read(), expected.mean, "mean");
        expectNear(invVariance.read(), expected.invVariance, "invVariance");
    }
}

TEST(TestHipMlopsLayernormDevice, ForwardMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    for(const bool withStatistics : {true, false})
    {
        SCOPED_TRACE(withStatistics ? "with statistics" : "without statistics");
        LayernormGraph spec;
        spec.withStatistics = withStatistics;
        checkForward(spec);
    }
}

TEST(TestHipMlopsLayernormDevice, TheProviderAdvertisesTheForwardWorkgroupKnob)
{
    SKIP_IF_NO_DEVICES();

    const LayernormGraph spec{};
    const auto knobs = advertisedKnobs(engineFor(spec), buildLayernormGraph(spec));
    ASSERT_EQ(knobs.size(), 1U);
    EXPECT_EQ(knobs[0].name, "local_size");
    EXPECT_EQ(knobs[0].defaultValue, DEFAULT_LOCAL_SIZE);
    EXPECT_EQ(std::set<int64_t>(knobs[0].validValues.begin(), knobs[0].validValues.end()),
              LOCAL_SIZES);
}

TEST(TestHipMlopsLayernormDevice, EachForwardWorkgroupPinnedThroughTheKnobMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    // Rows of 2400, wider than every workgroup, and of 30, narrower than every workgroup.
    for(const auto& dims : {std::vector<int64_t>{3, 5, 16, 30}, std::vector<int64_t>{8, 2, 3, 5}})
    {
        for(const bool withStatistics : {true, false})
        {
            for(const int64_t localSize : LOCAL_SIZES)
            {
                SCOPED_TRACE(elementCount(dims));
                SCOPED_TRACE(withStatistics ? "with statistics" : "without statistics");
                SCOPED_TRACE(localSize);
                LayernormGraph spec;
                spec.dims = dims;
                spec.withStatistics = withStatistics;
                checkForward(spec, {{"local_size", localSize}});
            }
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

TEST(TestHipMlopsLayernormDevice, TheProviderAdvertisesTheReductionKnob)
{
    SKIP_IF_NO_DEVICES();

    // 3 rows of 2400 fit one parallel pass on a device of 2 CUs or more: the one-pass
    // reduction is the default.
    const auto spec = backwardOf({3, 5, 16, 30});
    const auto knobs = advertisedKnobs(engineFor(spec), buildLayernormGraph(spec));
    ASSERT_EQ(knobs.size(), 2U);
    EXPECT_EQ(knobs[0].name, "parallel");
    EXPECT_EQ(knobs[0].defaultValue, 0);
    EXPECT_THAT(knobs[0].validValues, ::testing::UnorderedElementsAre(0, 1));
    EXPECT_EQ(knobs[1].name, "local_size");
    EXPECT_EQ(knobs[1].defaultValue, DEFAULT_LOCAL_SIZE);
    EXPECT_EQ(std::set<int64_t>(knobs[1].validValues.begin(), knobs[1].validValues.end()),
              LOCAL_SIZES);
}

TEST(TestHipMlopsLayernormDevice, EachReductionPinnedThroughTheKnobMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    // Both reductions on a graph the heuristics give the one-pass one, so the parallel one
    // runs only because it is pinned.
    for(const bool withStatistics : {true, false})
    {
        for(const int64_t parallel : {0, 1})
        {
            SCOPED_TRACE(withStatistics ? "with statistics" : "without statistics");
            SCOPED_TRACE(parallel == 1 ? "parallel" : "one pass");
            LayernormGraph spec;
            spec.backward = true;
            spec.withStatistics = withStatistics;
            const auto run
                = checkBackward(spec, static_cast<double>(spec.epsilon), {{"parallel", parallel}});
            if(withStatistics)
            {
                // Only the parallel partial sums need a workspace.
                EXPECT_EQ(run.workspaceBytes > 0U, parallel == 1);
            }
        }
    }
}

TEST(TestHipMlopsLayernormDevice, EachBackwardConfigurationPinnedThroughTheKnobsMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    // Every reduction at every workgroup, on rows of 2400, wider than every workgroup.
    for(const bool withStatistics : {true, false})
    {
        for(const int64_t parallel : {0, 1})
        {
            for(const int64_t localSize : LOCAL_SIZES)
            {
                SCOPED_TRACE(withStatistics ? "with statistics" : "without statistics");
                SCOPED_TRACE(parallel == 1 ? "parallel" : "one pass");
                SCOPED_TRACE(localSize);
                LayernormGraph spec;
                spec.backward = true;
                spec.withStatistics = withStatistics;
                checkBackward(spec,
                              static_cast<double>(spec.epsilon),
                              {{"parallel", parallel}, {"local_size", localSize}});
            }
        }
    }
}

} // namespace

#endif // HIPDNN_ENGINE_HIP_MLOPS_INGESTOR
