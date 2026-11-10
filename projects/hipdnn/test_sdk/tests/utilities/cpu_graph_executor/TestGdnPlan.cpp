// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceGdn.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/GdnPlan.hpp>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

namespace la = hipdnn_test_sdk::utilities::linear_attention_test_detail;

using FloatBuilder = GdnPlanBuilder<DataType::FLOAT, DataType::INT32, DataType::FLOAT>;

std::vector<int64_t> toVector(const ::flatbuffers::Vector<int64_t>* values)
{
    return values == nullptr ? std::vector<int64_t>{}
                             : std::vector<int64_t>(values->begin(), values->end());
}

std::vector<float> valuesOf(const TensorBase<float>& tensor)
{
    std::vector<float> result;
    const auto& dims = tensor.dims();
    std::vector<int64_t> index(dims.size(), 0);
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        result.push_back(tensor.getHostValue(index));
        for(size_t d = dims.size(); d-- > 0;)
        {
            if(++index[d] < dims[d])
            {
                break;
            }
            index[d] = 0;
        }
    }
    return result;
}

// Host buffers for every tensor of a linear-attention test graph (T = 8, cu_seqlens [0, 4, 8]).
// FLOAT operands get deterministic values seeded by UID; g is drawn from [gLow, gHigh].
struct GdnBuffers
{
    GdnBuffers(const GraphWrapper& graph, float gLow, float gHigh)
    {
        for(const auto& [uid, attributes] : graph.getTensorMap())
        {
            const auto dims = toVector(attributes->dims());
            const auto strides = toVector(attributes->strides());
            if(attributes->data_type() == DataType::INT32)
            {
                auto tensor = std::make_unique<Tensor<int32_t>>(dims, strides);
                auto* data = tensor->memory().hostData();
                if(uid == la::CU_SEQLENS_UID)
                {
                    data[0] = 0;
                    data[1] = 4;
                    data[2] = 8;
                }
                else
                {
                    // state_indices into a 3-row pool; row 1 is not addressed.
                    data[0] = 2;
                    data[1] = 0;
                }
                indices.emplace(uid, std::move(tensor));
                continue;
            }
            auto tensor = std::make_unique<Tensor<float>>(dims, strides);
            const auto seed = static_cast<unsigned>(uid) + 100u;
            if(uid == la::G_UID)
            {
                tensor->fillTensorWithRandomValues(gLow, gHigh, seed);
            }
            else
            {
                tensor->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
            }
            floats.emplace(uid, std::move(tensor));
        }
    }

    std::unordered_map<int64_t, void*> variantPack() const
    {
        std::unordered_map<int64_t, void*> pack;
        for(const auto& [uid, tensor] : floats)
        {
            pack[uid] = tensor->rawHostData();
        }
        for(const auto& [uid, tensor] : indices)
        {
            pack[uid] = tensor->rawHostData();
        }
        return pack;
    }

    Tensor<float>* floatTensor(int64_t uid) const
    {
        const auto it = floats.find(uid);
        return it == floats.end() ? nullptr : it->second.get();
    }

    std::unordered_map<int64_t, std::unique_ptr<Tensor<float>>> floats;
    std::unordered_map<int64_t, std::unique_ptr<Tensor<int32_t>>> indices;
};

// Runs the reference directly on `buffers`, as the plan should.
void runReference(GdnBuffers& buffers, const GdnConfig& config)
{
    GdnTensors<float> t;
    t.q = buffers.floatTensor(la::Q_UID);
    t.k = buffers.floatTensor(la::K_UID);
    t.v = buffers.floatTensor(la::V_UID);
    t.g = buffers.floatTensor(la::G_UID);
    t.beta = buffers.floatTensor(la::BETA_UID);
    t.cuSeqlens = buffers.indices.at(la::CU_SEQLENS_UID).get();
    t.o = buffers.floatTensor(la::O_UID);
    t.initialState = buffers.floatTensor(la::INITIAL_STATE_UID);
    t.aLog = buffers.floatTensor(la::A_LOG_UID);
    t.dtBias = buffers.floatTensor(la::DT_BIAS_UID);
    const auto stateIndices = buffers.indices.find(la::STATE_INDICES_UID);
    t.stateIndices = stateIndices == buffers.indices.end() ? nullptr : stateIndices->second.get();
    t.finalState = buffers.floatTensor(la::FINAL_STATE_UID);
    t.stateCheckpoints = buffers.floatTensor(la::STATE_CHECKPOINTS_UID);
    CpuFpReferenceGdn::forward(t, config);
}

void expectSame(const TensorBase<float>& actual, const TensorBase<float>& expected)
{
    const auto a = valuesOf(actual);
    const auto e = valuesOf(expected);
    ASSERT_EQ(a.size(), e.size());
    for(size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_FLOAT_EQ(a[i], e[i]) << "at flat index " << i;
    }
}

} // namespace

TEST(TestGdnPlanBuilder, IsApplicableMatchesTypes)
{
    using GraphFactory = flatbuffers::FlatBufferBuilder (*)();
    for(const GraphFactory factory :
        {GraphFactory{&createValidGdnGraph}, GraphFactory{&createValidGdnSafeGateGraph}})
    {
        auto builder = factory();
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        const auto& node = graphWrap.getNode(0);
        const auto& tensorMap = graphWrap.getTensorMap();

        EXPECT_TRUE(FloatBuilder().isApplicable(node, tensorMap));
        EXPECT_FALSE(
            (GdnPlanBuilder<DataType::HALF, DataType::INT32, DataType::FLOAT>().isApplicable(
                node, tensorMap)));
        EXPECT_FALSE(
            (GdnPlanBuilder<DataType::FLOAT, DataType::INT64, DataType::FLOAT>().isApplicable(
                node, tensorMap)));
    }

    auto kda = createValidKdaGraph();
    const GraphWrapper kdaWrap(kda.GetBufferPointer(), kda.GetSize());
    EXPECT_FALSE(FloatBuilder().isApplicable(kdaWrap.getNode(0), kdaWrap.getTensorMap()));
}

TEST(TestGdnPlanBuilder, BuildNodePlanReportsOutputs)
{
    auto builder = createValidGdnSafeGateGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const auto plan = FloatBuilder().buildNodePlan(graphWrap, graphWrap.getNode(0));
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->getOutputTensorIds(),
              (std::vector<int64_t>{la::O_UID, la::FINAL_STATE_UID, la::STATE_CHECKPOINTS_UID}));

    auto kda = createValidKdaGraph();
    const GraphWrapper kdaWrap(kda.GetBufferPointer(), kda.GetSize());
    EXPECT_THROW(FloatBuilder().buildNodePlan(kdaWrap, kdaWrap.getNode(0)), std::runtime_error);
}

// Safe gate, l2norm, sigmoid beta, initial state, checkpoints every 4 tokens: the plan matches
// the reference run directly on identical buffers.
TEST(TestGdnPlan, ExecuteMatchesReference)
{
    auto builder = createValidGdnSafeGateGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto plan = FloatBuilder().buildNodePlan(graphWrap, graphWrap.getNode(0));

    const GdnBuffers planBuffers(graphWrap, -1.0f, 1.0f);
    GdnBuffers referenceBuffers(graphWrap, -1.0f, 1.0f);
    plan->execute(planBuffers.variantPack());

    GdnConfig config;
    config.useQkL2norm = true;
    config.checkpointEveryNTokens = 4;
    config.useBetaSigmoid = true;
    config.safeGate = true;
    runReference(referenceBuffers, config);

    for(const auto uid : {la::O_UID, la::FINAL_STATE_UID, la::STATE_CHECKPOINTS_UID})
    {
        expectSame(*planBuffers.floatTensor(uid), *referenceBuffers.floatTensor(uid));
    }
}

// overwrite_initial_state: binding final_state's UID to initial_state's buffer updates it in place.
TEST(TestGdnPlan, ExecuteInPlace)
{
    auto builder = createValidGdnSafeGateGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto plan = FloatBuilder().buildNodePlan(graphWrap, graphWrap.getNode(0));

    const GdnBuffers inPlace(graphWrap, -1.0f, 1.0f);
    const GdnBuffers separate(graphWrap, -1.0f, 1.0f);
    auto pack = inPlace.variantPack();
    pack[la::FINAL_STATE_UID] = pack.at(la::INITIAL_STATE_UID);
    plan->execute(pack);
    plan->execute(separate.variantPack());

    expectSame(*inPlace.floatTensor(la::O_UID), *separate.floatTensor(la::O_UID));
    expectSame(*inPlace.floatTensor(la::INITIAL_STATE_UID),
               *separate.floatTensor(la::FINAL_STATE_UID));
}

// State pool (LINEAR gate, allow_neg_eigval): rows 2 and 0 are read and rewritten; row 1 of
// final_state keeps its contents.
TEST(TestGdnPlan, ExecuteStatePool)
{
    auto builder = createValidGdnGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto plan = FloatBuilder().buildNodePlan(graphWrap, graphWrap.getNode(0));

    const GdnBuffers planBuffers(graphWrap, 0.5f, 1.0f);
    GdnBuffers referenceBuffers(graphWrap, 0.5f, 1.0f);
    const auto untouchedBefore = valuesOf(*planBuffers.floatTensor(la::FINAL_STATE_UID));
    plan->execute(planBuffers.variantPack());

    GdnConfig config;
    config.scale = 0.35355339f;
    config.useBetaSigmoid = true;
    config.allowNegEigval = true;
    config.gateDomain = GateDomain::LINEAR;
    runReference(referenceBuffers, config);

    expectSame(*planBuffers.floatTensor(la::O_UID), *referenceBuffers.floatTensor(la::O_UID));
    expectSame(*planBuffers.floatTensor(la::FINAL_STATE_UID),
               *referenceBuffers.floatTensor(la::FINAL_STATE_UID));

    const auto after = valuesOf(*planBuffers.floatTensor(la::FINAL_STATE_UID));
    const size_t rowSize = after.size() / 3;
    for(size_t i = rowSize; i < 2 * rowSize; ++i)
    {
        EXPECT_EQ(after[i], untouchedBefore[i]);
    }
}
