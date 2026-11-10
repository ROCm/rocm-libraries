// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeRoutingPlan.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

// createValidMoeRoutingGraph(): topk_ids [3, 2] INT32 (uid 1), first_token_offset [5, 1, 1]
// INT32 (uid 2), token_index / token_ks [1, 6, 1] INT32 (uids 3, 4), num_experts 4.
struct MoeRoutingGraphTensors
{
    Tensor<int32_t> topkIds{std::vector<int64_t>{3, 2}, generateStrides({3, 2})};
    Tensor<int32_t> offsets{std::vector<int64_t>{5, 1, 1}, generateStrides({5, 1, 1})};
    Tensor<int32_t> tokenIndex{std::vector<int64_t>{1, 6, 1}, generateStrides({1, 6, 1})};
    Tensor<int32_t> tokenKs{std::vector<int64_t>{1, 6, 1}, generateStrides({1, 6, 1})};

    MoeRoutingGraphTensors()
    {
        // ids [[3, 0], [1, 3], [0, 0]]: expert 0 <- (0, 1), (2, 0), (2, 1); expert 1 <- (1, 0);
        // expert 2 <- none; expert 3 <- (0, 0), (1, 1).
        const std::vector<int32_t> ids = {3, 0, 1, 3, 0, 0};
        std::copy(ids.begin(), ids.end(), topkIds.memory().hostData());
    }

    std::unordered_map<int64_t, void*> variantPack()
    {
        return {{1, topkIds.memory().hostData()},
                {2, offsets.memory().hostData()},
                {3, tokenIndex.memory().hostData()},
                {4, tokenKs.memory().hostData()}};
    }

    void expectRouted()
    {
        const auto* offsetData = offsets.memory().hostData();
        const auto* indexData = tokenIndex.memory().hostData();
        const auto* ksData = tokenKs.memory().hostData();
        EXPECT_EQ((std::vector<int32_t>{offsetData, offsetData + 5}),
                  (std::vector<int32_t>{0, 3, 4, 4, 6}));
        EXPECT_EQ((std::vector<int32_t>{indexData, indexData + 6}),
                  (std::vector<int32_t>{0, 2, 2, 1, 0, 1}));
        EXPECT_EQ((std::vector<int32_t>{ksData, ksData + 6}),
                  (std::vector<int32_t>{1, 0, 1, 0, 0, 1}));
    }
};

void describe(TensorAttributesT& attributes,
              int64_t uid,
              DataType dataType,
              const hipdnn_data_sdk::utilities::ITensor& tensor)
{
    attributes.uid = uid;
    attributes.data_type = dataType;
    attributes.dims = tensor.dims();
    attributes.strides = tensor.strides();
}

} // namespace

TEST(TestMoeRoutingPlan, ExecutePlan)
{
    MoeRoutingGraphTensors tensors;

    MoeRoutingParams params;
    describe(params.topkIdsTensor, 1, DataType::INT32, tensors.topkIds);
    describe(params.firstTokenOffsetTensor, 2, DataType::INT32, tensors.offsets);
    describe(params.tokenIndexTensor, 3, DataType::INT32, tensors.tokenIndex);
    describe(params.tokenKsTensor, 4, DataType::INT32, tensors.tokenKs);
    params.numExperts = 4;

    MoeRoutingPlan<int32_t> patient(std::move(params));
    EXPECT_EQ(patient.getOutputTensorIds(), (std::vector<int64_t>{2, 3, 4}));

    patient.execute(tensors.variantPack());

    tensors.expectRouted();
}

TEST(TestMoeRoutingPlanBuilder, IsApplicable)
{
    auto builder = createValidMoeRoutingGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const MoeRoutingPlanBuilder<DataType::INT32> int32PlanBuilder;
    EXPECT_TRUE(int32PlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Wrong first_token_offset data type.
    const MoeRoutingPlanBuilder<DataType::INT64> int64PlanBuilder;
    EXPECT_FALSE(int64PlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing tensor in tensorMap.
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(3);
    EXPECT_FALSE(int32PlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // Incompatible node type.
    auto otherBuilder = createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_FALSE(
        int32PlanBuilder.isApplicable(otherGraphWrap.getNode(0), otherGraphWrap.getTensorMap()));
}

TEST(TestMoeRoutingPlanBuilder, BuildNodePlan)
{
    const MoeRoutingPlanBuilder<DataType::INT32> patient;

    auto builder = createValidMoeRoutingGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));
    EXPECT_NE(dynamic_cast<MoeRoutingPlan<int32_t>*>(builtPlan.get()), nullptr);

    auto otherBuilder = createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherGraphWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_THROW(patient.buildNodePlan(otherGraphWrap, otherGraphWrap.getNode(0)),
                 std::runtime_error);
}

TEST(TestMoeRoutingPlan, CpuReferenceGraphExecutorRunsMoeRoutingGraph)
{
    MoeRoutingGraphTensors tensors;
    auto builder = createValidMoeRoutingGraph();

    CpuReferenceGraphExecutor executor;
    executor.execute(builder.GetBufferPointer(), builder.GetSize(), tensors.variantPack());

    tensors.expectRouted();
}
