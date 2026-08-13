// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseSubPack.cpp
 * @brief The second reference pack: what its matchers accept and refuse, how it ranks,
 *        and that it subtracts in the right direction.
 *
 * Deliberately smaller than the add pack's suite, which owns the exhaustive refusal
 * matrix for the shape both share. Repeating all eleven cases would assert one code
 * path twice. What is here is this pack's own: that it claims SUB and declines ADD,
 * that its bindings are ordered since subtraction does not commute, and that the
 * inherited refusals reached this matcher rather than being assumed.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Runs the sub pack's graph matcher, discarding what it binds.
bool matches(const MatchContext& context)
{
    BoundTokens bound;
    return matchesGraph(POINTWISE_SUB, context, bound);
}

/// A kernel definition naming this pack's source, since makeKernel defaults to the add
/// pack's.
hipdnn_plugin_sdk::ingestor::KernelDefinition subKernel(int64_t blockSize, const std::string& dtype)
{
    return makeKernel(blockSize, dtype, "PointwiseSub");
}

// ---------------------------------------------------------------------------
// The operation gate: this is what separates two packs over one graph shape
// ---------------------------------------------------------------------------

TEST(TestPointwiseSubMatcher, AcceptsASingleElementFloatSubtract)
{
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestPointwiseSubMatcher, RefusesTheAddGraphItsSiblingPackServes)
{
    // The two packs are distinguished by operation alone: same node count, same shape,
    // same dtypes. If this returned true both engines would claim every pointwise graph
    // and selection would be arbitrary.
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::ADD));

    EXPECT_FALSE(matches(fixture.context()));
}

TEST(TestPointwiseAddMatcher, RefusesTheSubGraphItsSiblingPackServes)
{
    // The converse, asserted here rather than in the add pack's file because it only
    // became meaningful once a second pack existed.
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    BoundTokens bound;
    EXPECT_FALSE(matchesGraph(POINTWISE_ADD, fixture.context(), bound));
}

// ---------------------------------------------------------------------------
// Refusals this pack must have inherited, not merely been assumed to
// ---------------------------------------------------------------------------

/// One graph the sub matcher must refuse, plus a readable name for a failing run.
struct SubRefusalCase
{
    std::string name;
    flatbuffers::FlatBufferBuilder (*buildGraph)();
};

class TestPointwiseSubMatcherRefusal : public ::testing::TestWithParam<SubRefusalCase>
{
};

TEST_P(TestPointwiseSubMatcherRefusal, Refuses)
{
    const GraphFixture fixture(GetParam().buildGraph());

    EXPECT_FALSE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestPointwiseSubMatcherRefusal,
    ::testing::ValuesIn(std::vector<SubRefusalCase>{
        {"MultiElementTensors",
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::SUB, data_objects::DataType::FLOAT, {1, 1, 2, 2});
         }},
        {"ATensorWithNoStrides",
         // Same refusal the add pack owes: the layout classifier dereferences strides(),
         // and applicability runs before anything has validated the graph.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::SUB,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/std::nullopt,
                                        /*inputAVirtual=*/false,
                                        /*inputAIsRuntimePassByValue=*/false,
                                        /*outputVirtual=*/false,
                                        /*omitStrides=*/true);
         }},
        {"DimsWhoseProductIsOneButAreNotAllOne",
         // {-1,-1,1,1} multiplies to 1; the kernel indexes element 0 only.
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::SUB, data_objects::DataType::FLOAT, {-1, -1, 1, 1});
         }},
        {"ARankTheDispatchPathCannotServe",
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::SUB, data_objects::DataType::FLOAT, {1});
         }},
        {"AUnaryPointwise",
         // A subtract with one operand is not a subtract.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::SUB,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/false);
         }},
        {"CrossOperandDtypeMismatch",
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::SUB,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/data_objects::DataType::HALF);
         }},
        {"ADanglingTensorUid",
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::SUB,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/DEFAULT_DANGLING_UID);
         }},
    }),
    [](const ::testing::TestParamInfo<SubRefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Kernel-scoped matching and ranking
// ---------------------------------------------------------------------------

TEST(TestPointwiseSubMatcher, RefusesAKernelBakedForAnotherDtype)
{
    // An f16 kernel handed f32 operands does not fail; it returns wrong numbers.
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    EXPECT_TRUE(matchesKernel(POINTWISE_SUB, fixture.context(), subKernel(64, "FLOAT")));
    EXPECT_FALSE(matchesKernel(POINTWISE_SUB, fixture.context(), subKernel(64, "HALF")));
}

TEST(TestPointwiseSubScore, PrefersTheLargerBlockSize)
{
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    EXPECT_GT(scoreKernel(POINTWISE_SUB, subKernel(256, "FLOAT"), fixture.context()),
              scoreKernel(POINTWISE_SUB, subKernel(64, "FLOAT"), fixture.context()));
}

// ---------------------------------------------------------------------------
// Binding order: the property subtraction has and addition does not
// ---------------------------------------------------------------------------

TEST(TestPointwiseSubBinding, BindsTheMinuendAndSubtrahendInGraphOrder)
{
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    BoundTokens bound;
    ASSERT_TRUE(matchesGraph(POINTWISE_SUB, fixture.context(), bound));

    // Swapping these computes b-a, which is a plausible answer of the wrong sign rather
    // than a failure, so it is asserted here and again numerically on device.
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_SUB.inputAToken),
              INPUT_A_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_SUB.inputBToken),
              INPUT_B_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_SUB.outputToken),
              OUTPUT_UID);
}

TEST(TestPointwiseSubBinding, BindsUnderItsOwnTokenNamesNotItsSiblings)
{
    // Both packs bind three operand uids. If they shared token names, mergeBound would
    // see two packs binding one token, agreeing here by luck and conflicting the
    // moment a graph gives them different operands.
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    BoundTokens bound;
    ASSERT_TRUE(matchesGraph(POINTWISE_SUB, fixture.context(), bound));

    EXPECT_EQ(bound.count(std::string(POINTWISE_ADD.inputAToken)), 0U);
    EXPECT_EQ(bound.count(std::string(POINTWISE_ADD.inputBToken)), 0U);
    EXPECT_EQ(bound.count(std::string(POINTWISE_ADD.outputToken)), 0U);
}

// ---------------------------------------------------------------------------
// Descriptor set
// ---------------------------------------------------------------------------

TEST(TestPointwiseSubPack, EveryKernelNamesThisPacksOwnSource)
{
    const auto set = buildPointwiseSubDescriptorSet();

    ASSERT_EQ(set.packs.size(), 1U);
    for(const auto& kernel : set.packs.front().kernels)
    {
        EXPECT_EQ(kernel.source.kind,
                  hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE);
        EXPECT_EQ(kernel.source.sourceFile, "PointwiseSub.cpp");
        EXPECT_EQ(kernel.source.entryPoint, "PointwiseSub");
    }
}

TEST(TestPointwiseSubPack, PackCrossReferencesResolveWithinTheDescriptorSet)
{
    const auto set = buildPointwiseSubDescriptorSet();
    const auto& pack = set.packs.front();

    EXPECT_EQ(pack.engineId, set.engine.id);
    EXPECT_EQ(set.engine.heuristicId, set.heuristic.id);
    EXPECT_EQ(set.engine.metadataSchemaId, set.schema.id);

    ASSERT_EQ(pack.matcherIds.size(), 2U);
    for(const auto& matcherId : pack.matcherIds)
    {
        EXPECT_TRUE(std::any_of(set.matchers.begin(), set.matchers.end(), [&](const auto& matcher) {
            return matcher.id == matcherId;
        }));
    }

    EXPECT_TRUE(std::any_of(set.dispatches.begin(),
                            set.dispatches.end(),
                            [&](const auto& dispatch) { return dispatch.id == pack.dispatchId; }));
}

TEST(TestPointwiseSubPack, SharesNoDescriptorIdWithTheAddPack)
{
    // Both sets load into one provider and descriptors reference each other by id, so a
    // collision would silently make two engines one. This is the check a loader will
    // need across installed files, exercised here while there are only two.
    const auto add = buildPointwiseAddDescriptorSet();
    const auto sub = buildPointwiseSubDescriptorSet();

    std::vector<hipdnn_plugin_sdk::ingestor::DescriptorId> addIds{
        add.engine.id, add.schema.id, add.heuristic.id, add.packs.front().id};
    for(const auto& matcher : add.matchers)
    {
        addIds.push_back(matcher.id);
    }
    for(const auto& dispatch : add.dispatches)
    {
        addIds.push_back(dispatch.id);
    }
    for(const auto& kernel : add.packs.front().kernels)
    {
        addIds.push_back(kernel.id);
    }

    std::vector<hipdnn_plugin_sdk::ingestor::DescriptorId> subIds{
        sub.engine.id, sub.schema.id, sub.heuristic.id, sub.packs.front().id};
    for(const auto& matcher : sub.matchers)
    {
        subIds.push_back(matcher.id);
    }
    for(const auto& dispatch : sub.dispatches)
    {
        subIds.push_back(dispatch.id);
    }
    for(const auto& kernel : sub.packs.front().kernels)
    {
        subIds.push_back(kernel.id);
    }

    for(const auto& subId : subIds)
    {
        EXPECT_EQ(std::find(addIds.begin(), addIds.end(), subId), addIds.end())
            << "descriptor id " << hipdnn_plugin_sdk::ingestor::toString(subId)
            << " is used by both reference packs";
    }
}

TEST(TestPointwiseSubPack, RegistersADistinctEngineName)
{
    const auto subId = registerEngineName(std::string(POINTWISE_SUB.engineName));
    const auto addId = registerEngineName(std::string(POINTWISE_ADD.engineName));

    EXPECT_NE(subId, addId);
    EXPECT_EQ(hipdnn_data_sdk::utilities::getEngineNameFromId(subId), POINTWISE_SUB.engineName);
}

// ---------------------------------------------------------------------------
// Dispatch, on device
// ---------------------------------------------------------------------------

TEST(TestPointwiseSubDispatch, SubtractsInTheRightDirection)
{
    SKIP_IF_NO_DEVICES();

    // The whole point of a second op: a+b and a-b differ, and an asymmetric operation
    // catches an operand swap that a commutative one cannot.
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB),
                               currentDeviceProperties());
    const auto& handler = dispatchHandler(POINTWISE_SUB);

    BoundTokens bound;
    ASSERT_TRUE(matchesGraph(POINTWISE_SUB, fixture.context(), bound));

    const auto prepared = handler.prepare(fixture.context(), bound, subKernel(64, "FLOAT"));
    ASSERT_NE(prepared, nullptr);

    void* deviceA = nullptr;
    void* deviceB = nullptr;
    void* deviceC = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&deviceA, sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&deviceB, sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&deviceC, sizeof(float)));

    const float a = 7.0f;
    const float b = 4.0f;
    ASSERT_EQ(hipSuccess, hipMemcpy(deviceA, &a, sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(deviceB, &b, sizeof(float), hipMemcpyHostToDevice));

    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
        hipdnnPluginDeviceBuffer_t{INPUT_A_UID, deviceA},
        hipdnnPluginDeviceBuffer_t{INPUT_B_UID, deviceB},
        hipdnnPluginDeviceBuffer_t{OUTPUT_UID, deviceC}};

    const Handle handle;
    handler.launch(handle, *prepared, buffers.data(), buffers.size(), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    float result = 0.0f;
    ASSERT_EQ(hipSuccess, hipMemcpy(&result, deviceC, sizeof(float), hipMemcpyDeviceToHost));

    // 3.0, not -3.0: an operand swap compiles, launches, and returns the wrong sign.
    EXPECT_FLOAT_EQ(result, 3.0f);

    static_cast<void>(hipFree(deviceA));
    static_cast<void>(hipFree(deviceB));
    static_cast<void>(hipFree(deviceC));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
