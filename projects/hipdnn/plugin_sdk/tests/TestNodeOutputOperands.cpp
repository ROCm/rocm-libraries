// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/node_operands_generated.h>

namespace
{
using namespace hipdnn_flatbuffers_sdk::data_objects;

using Outputs = std::vector<std::pair<std::string, int64_t>>;

// Packs a one-node graph and reports the roles and uids the node writes, in reported order.
template <typename TAttributes>
Outputs outputsOf(TAttributes attributes)
{
    GraphT graph;
    auto node = std::make_unique<NodeT>();
    node->attributes.Set(std::move(attributes));
    graph.nodes.push_back(std::move(node));

    flatbuffers::FlatBufferBuilder buffer;
    buffer.Finish(Graph::Pack(buffer, &graph));

    Outputs outputs;
    const bool known = node_operands::forEachOutputTensor(
        *GetGraph(buffer.GetBufferPointer())->nodes()->Get(0),
        [&outputs](std::string_view role, int64_t uid) { outputs.emplace_back(role, uid); });
    EXPECT_TRUE(known);
    return outputs;
}

TEST(TestNodeOutputOperands, ConvolutionForwardWritesY)
{
    ConvolutionFwdAttributesT conv;
    conv.x_tensor_uid = 1;
    conv.w_tensor_uid = 2;
    conv.y_tensor_uid = 3;

    EXPECT_EQ(outputsOf(conv), (Outputs{{"y", 3}}));
}

TEST(TestNodeOutputOperands, PointwiseWritesOut0)
{
    PointwiseAttributesT pointwise;
    pointwise.in_0_tensor_uid = 1;
    pointwise.in_1_tensor_uid = 2;
    pointwise.out_0_tensor_uid = 3;

    EXPECT_EQ(outputsOf(pointwise), (Outputs{{"out_0", 3}}));
}

TEST(TestNodeOutputOperands, BatchnormTrainingWritesItsStatistics)
{
    BatchnormAttributesT batchnorm;
    batchnorm.x_tensor_uid = 1;
    batchnorm.scale_tensor_uid = 2;
    batchnorm.bias_tensor_uid = 3;
    batchnorm.peer_stats_tensor_uid = {40, 41};
    batchnorm.y_tensor_uid = 5;
    batchnorm.mean_tensor_uid = 6;
    batchnorm.inv_variance_tensor_uid = 7;

    // peer_stats is an input, and the absent next_running_* outputs are skipped.
    EXPECT_EQ(outputsOf(batchnorm), (Outputs{{"y", 5}, {"mean", 6}, {"inv_variance", 7}}));
}

TEST(TestNodeOutputOperands, BatchnormInferenceReadsMeanAndWritesOnlyY)
{
    BatchnormInferenceAttributesT batchnorm;
    batchnorm.x_tensor_uid = 1;
    batchnorm.mean_tensor_uid = 2;
    batchnorm.inv_variance_tensor_uid = 3;
    batchnorm.scale_tensor_uid = 4;
    batchnorm.bias_tensor_uid = 5;
    batchnorm.y_tensor_uid = 6;

    EXPECT_EQ(outputsOf(batchnorm), (Outputs{{"y", 6}}));
}

TEST(TestNodeOutputOperands, SdpaForwardWritesOAndRequestedStats)
{
    SdpaAttributesT sdpa;
    sdpa.q_tensor_uid = 1;
    sdpa.k_tensor_uid = 2;
    sdpa.v_tensor_uid = 3;
    sdpa.o_tensor_uid = 4;
    sdpa.stats_tensor_uid = 5;
    sdpa.seed_tensor_uid = 6;

    EXPECT_EQ(outputsOf(sdpa), (Outputs{{"o", 4}, {"stats", 5}}));
}

TEST(TestNodeOutputOperands, SdpaBackwardReadsStatsAndWritesTheGradients)
{
    SdpaBackwardAttributesT sdpa;
    sdpa.q_tensor_uid = 1;
    sdpa.k_tensor_uid = 2;
    sdpa.v_tensor_uid = 3;
    sdpa.o_tensor_uid = 4;
    sdpa.do_tensor_uid = 5;
    sdpa.stats_tensor_uid = 6;
    sdpa.dq_tensor_uid = 7;
    sdpa.dk_tensor_uid = 8;
    sdpa.dv_tensor_uid = 9;

    EXPECT_EQ(outputsOf(sdpa), (Outputs{{"dq", 7}, {"dk", 8}, {"dv", 9}}));
}

TEST(TestNodeOutputOperands, CustomOpReportsOneIndexedRolePerOutputUid)
{
    CustomOpAttributesT custom;
    custom.input_tensor_uids = {1, 2};
    custom.output_tensor_uids = {3, 4};

    EXPECT_EQ(outputsOf(custom), (Outputs{{"output[0]", 3}, {"output[1]", 4}}));
}

TEST(TestNodeOutputOperands, MoeBackwardWritesDweight)
{
    MoeGroupedMatmulBwdAttributesT moe;
    moe.doutput_tensor_uid = 1;
    moe.token_tensor_uid = 2;
    moe.first_token_offset_tensor_uid = 3;
    moe.dweight_tensor_uid = 4;

    EXPECT_EQ(outputsOf(moe), (Outputs{{"dweight", 4}}));
}

TEST(TestNodeOutputOperands, ANodeWithoutAttributesReportsNothing)
{
    flatbuffers::FlatBufferBuilder buffer;
    GraphT graph;
    graph.nodes.push_back(std::make_unique<NodeT>());
    buffer.Finish(Graph::Pack(buffer, &graph));

    bool called = false;
    const bool known = node_operands::forEachOutputTensor(
        *GetGraph(buffer.GetBufferPointer())->nodes()->Get(0),
        [&called](std::string_view, int64_t) { called = true; });

    EXPECT_FALSE(known);
    EXPECT_FALSE(called);
}

} // namespace
