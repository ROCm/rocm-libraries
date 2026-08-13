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
 * Smaller than the add pack's suite, which owns the exhaustive refusal matrix for the
 * shared shape; this file covers only what is unique to SUB.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

bool matches(const MatchContext& context)
{
    BoundTokens bound;
    return matchesGraph(POINTWISE_SUB, context, bound);
}

/// Names this pack's source; makeKernel defaults to the add pack's.
hipdnn_plugin_sdk::ingestor::KernelDefinition subKernel(int64_t blockSize, const std::string& dtype)
{
    return makeKernel(blockSize, dtype, "PointwiseSub");
}

// The operation gate: this is what separates two packs over one graph shape

TEST(TestPointwiseSubMatcher, AcceptsASingleElementFloatSubtract)
{
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestPointwiseSubMatcher, RefusesTheAddGraphItsSiblingPackServes)
{
    // Same node count, shape, and dtypes; only operation distinguishes the packs.
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::ADD));

    EXPECT_FALSE(matches(fixture.context()));
}

TEST(TestPointwiseAddMatcher, RefusesTheSubGraphItsSiblingEngineServes)
{
    // The converse, and it lands on the operation matcher rather than the shared one:
    // a subtraction is a perfectly servable *shape*, so the shared matcher admits it and
    // both of this engine's packs then decline it on operation. Neither pack passing is
    // how the engine as a whole declines a graph its sibling owns.
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    BoundTokens bound;
    EXPECT_TRUE(matchesGraph(POINTWISE_ADD, fixture.context(), bound));
    EXPECT_FALSE(matchesOperation(POINTWISE_ADD, fixture.context(), bound));
    EXPECT_FALSE(matchesOperation(POINTWISE_MUL, fixture.context(), bound));
}

// Refusals this pack must have inherited, not merely been assumed to

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

// Kernel-scoped matching and ranking

TEST(TestPointwiseSubMatcher, RefusesAKernelBakedForAnotherDtype)
{
    // An f16 kernel handed f32 operands returns wrong numbers, not a failure.
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

// Binding order: the property subtraction has and addition does not

TEST(TestPointwiseSubBinding, BindsTheMinuendAndSubtrahendInGraphOrder)
{
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    BoundTokens bound;
    ASSERT_TRUE(matchesGraph(POINTWISE_SUB, fixture.context(), bound));

    // Swapping these computes b-a: a plausible wrong-sign answer, not a failure.
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_SUB.inputAToken),
              INPUT_A_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_SUB.inputBToken),
              INPUT_B_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_SUB.outputToken),
              OUTPUT_UID);
}

TEST(TestPointwiseSubBinding, BindsUnderItsOwnTokenNamesNotItsSiblings)
{
    // Shared token names would let mergeBound conflate the two packs' operands.
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::SUB));

    BoundTokens bound;
    ASSERT_TRUE(matchesGraph(POINTWISE_SUB, fixture.context(), bound));

    EXPECT_EQ(bound.count(std::string(POINTWISE_ADD.inputAToken)), 0U);
    EXPECT_EQ(bound.count(std::string(POINTWISE_ADD.inputBToken)), 0U);
    EXPECT_EQ(bound.count(std::string(POINTWISE_ADD.outputToken)), 0U);
}

// Descriptor set

TEST(TestPointwiseSubPack, EveryKernelNamesThisPacksOwnSource)
{
    const auto set = loadedSet("hipkernel:PointwiseSub");

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
    const auto set = loadedSet("hipkernel:PointwiseSub");
    const auto& pack = set.packs.front();

    EXPECT_EQ(pack.engineId, set.engine.id);
    ASSERT_TRUE(set.heuristic.has_value()) << "this pack ships a scorer";
    ASSERT_TRUE(set.engine.heuristicId.has_value());
    EXPECT_EQ(*set.engine.heuristicId, set.heuristic->id);
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
    const auto add = loadedSet("hipkernel:Pointwise");
    const auto sub = loadedSet("hipkernel:PointwiseSub");

    ASSERT_TRUE(add.heuristic.has_value());
    std::vector<hipdnn_plugin_sdk::ingestor::DescriptorId> addIds{
        add.engine.id, add.schema.id, add.heuristic->id, add.packs.front().id};
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

    ASSERT_TRUE(sub.heuristic.has_value());
    std::vector<hipdnn_plugin_sdk::ingestor::DescriptorId> subIds{
        sub.engine.id, sub.schema.id, sub.heuristic->id, sub.packs.front().id};
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

// Dispatch, on device

TEST(TestPointwiseSubDispatch, SubtractsInTheRightDirection)
{
    SKIP_IF_NO_DEVICES();

    // An asymmetric op catches an operand swap a commutative one cannot.
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

    EXPECT_FLOAT_EQ(result, 3.0f);

    static_cast<void>(hipFree(deviceA));
    static_cast<void>(hipFree(deviceB));
    static_cast<void>(hipFree(deviceC));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
