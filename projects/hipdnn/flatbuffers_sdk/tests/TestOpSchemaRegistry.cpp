// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/op_schema_registry_generated.h>

#include <flatbuffers/flatbuffers.h>

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::data_objects::op_schema;

namespace
{

template <typename Schema>
const Schema* findByName(SchemaSpan<Schema> entries, std::string_view name)
{
    for(const auto& entry : entries)
    {
        if(entry.name == name)
        {
            return &entry;
        }
    }
    return nullptr;
}

const Node& rootNode(const ::flatbuffers::DetachedBuffer& buffer)
{
    return *::flatbuffers::GetRoot<Node>(buffer.data());
}

::flatbuffers::DetachedBuffer finishNode(::flatbuffers::FlatBufferBuilder& builder,
                                         NodeAttributes attributesType,
                                         ::flatbuffers::Offset<void> attributes)
{
    builder.Finish(CreateNode(builder, 0, DataType::FLOAT, attributesType, attributes));
    return builder.Release();
}

::flatbuffers::DetachedBuffer buildCustomOpNode(const std::vector<int64_t>& inputs,
                                                const std::vector<int64_t>& outputs)
{
    ::flatbuffers::FlatBufferBuilder builder;
    const auto attributes = CreateCustomOpAttributesDirect(builder, "test.op", &inputs, &outputs);
    return finishNode(builder, NodeAttributes::CustomOpAttributes, attributes.Union());
}

::flatbuffers::DetachedBuffer buildCustomOpNodeWithoutUids()
{
    ::flatbuffers::FlatBufferBuilder builder;
    const auto attributes = CreateCustomOpAttributesDirect(builder, "test.op");
    return finishNode(builder, NodeAttributes::CustomOpAttributes, attributes.Union());
}

// ADD with in_0 = 1 and out_0 = 2. The optional in_1 is absent.
::flatbuffers::DetachedBuffer buildPointwiseAddNode()
{
    ::flatbuffers::FlatBufferBuilder builder;
    PointwiseAttributesBuilder pointwise(builder);
    pointwise.add_operation(PointwiseMode::ADD);
    pointwise.add_in_0_tensor_uid(1);
    pointwise.add_out_0_tensor_uid(2);
    const auto attributes = pointwise.Finish();
    return finishNode(builder, NodeAttributes::PointwiseAttributes, attributes.Union());
}

std::vector<int64_t> toVector(const ::flatbuffers::Vector<int64_t>* uids)
{
    return {uids->begin(), uids->end()};
}

/// Each edge's name, with a `?` suffix when the edge is optional.
std::set<std::string> edgeNames(SchemaSpan<EdgeSchema> edges)
{
    std::set<std::string> names;
    for(const auto& edge : edges)
    {
        names.insert(std::string(edge.name) + (edge.optional ? "?" : ""));
    }
    return names;
}

struct ExpectedEdges
{
    std::string_view opcode;
    std::set<std::string> operands;
    std::set<std::string> results;
};

ScalarValue enumName(std::string_view name)
{
    return ScalarValue{std::in_place_type<std::string_view>, name};
}

} // namespace

TEST(TestOpSchemaRegistry, EveryNodeAttributesMemberHasAnOp)
{
    for(const auto type : EnumValuesNodeAttributes())
    {
        if(type == NodeAttributes::NONE)
        {
            continue;
        }
        const auto* op = findOpSchema(type);
        ASSERT_NE(op, nullptr) << EnumNameNodeAttributes(type);
        EXPECT_EQ(op->attributesType, type);
    }
    EXPECT_EQ(findOpSchema(NodeAttributes::NONE), nullptr);
}

TEST(TestOpSchemaRegistry, FindOpSchemaByOpcodeReturnsThatOp)
{
    for(const auto& op : allOpSchemas())
    {
        EXPECT_EQ(findOpSchema(op.opcode), &op) << op.opcode;
    }
    EXPECT_EQ(findOpSchema("no_such_op"), nullptr);
    EXPECT_EQ(findOpSchema(""), nullptr);
}

TEST(TestOpSchemaRegistry, OpcodesAreStrictlySorted)
{
    // findOpSchema(opcode) binary-searches this order.
    const OpSchema* previous = nullptr;
    for(const auto& op : allOpSchemas())
    {
        if(previous != nullptr)
        {
            EXPECT_LT(previous->opcode, op.opcode);
        }
        previous = &op;
    }
}

TEST(TestOpSchemaRegistry, CustomOpUidVectorsAreEdgeListsNotEdges)
{
    const auto* customOp = findOpSchema("custom_op");
    ASSERT_NE(customOp, nullptr);
    EXPECT_TRUE(customOp->operands.empty());
    EXPECT_TRUE(customOp->results.empty());

    const auto* inputs = findByName(customOp->operandLists, "input_tensor_uids");
    const auto* outputs = findByName(customOp->resultLists, "output_tensor_uids");
    ASSERT_NE(inputs, nullptr);
    ASSERT_NE(outputs, nullptr);
    EXPECT_EQ(customOp->operandLists.size, 1u);
    EXPECT_EQ(customOp->resultLists.size, 1u);

    const auto customOpBuffer = buildCustomOpNode({3, 5}, {7});
    const auto& customOpNode = rootNode(customOpBuffer);
    const auto* inputUids = inputs->read(customOpNode);
    const auto* outputUids = outputs->read(customOpNode);
    ASSERT_NE(inputUids, nullptr);
    ASSERT_NE(outputUids, nullptr);
    EXPECT_EQ(toVector(inputUids), (std::vector<int64_t>{3, 5}));
    EXPECT_EQ(toVector(outputUids), (std::vector<int64_t>{7}));

    const auto withoutUidsBuffer = buildCustomOpNodeWithoutUids();
    EXPECT_EQ(inputs->read(rootNode(withoutUidsBuffer)), nullptr);
    EXPECT_EQ(outputs->read(rootNode(withoutUidsBuffer)), nullptr);

    const auto pointwiseBuffer = buildPointwiseAddNode();
    EXPECT_EQ(inputs->read(rootNode(pointwiseBuffer)), nullptr);
    EXPECT_EQ(outputs->read(rootNode(pointwiseBuffer)), nullptr);
}

TEST(TestOpSchemaRegistry, PointwiseEdgesAndAttributesReadTheNode)
{
    const auto* pointwise = findOpSchema("pointwise");
    ASSERT_NE(pointwise, nullptr);
    const auto* in0 = findByName(pointwise->operands, "in_0");
    const auto* in1 = findByName(pointwise->operands, "in_1");
    const auto* out0 = findByName(pointwise->results, "out_0");
    const auto* operation = findByName(pointwise->attributes, "operation");
    const auto* computeDataType = findByName(pointwise->attributes, "compute_data_type");
    ASSERT_NE(in0, nullptr);
    ASSERT_NE(in1, nullptr);
    ASSERT_NE(out0, nullptr);
    ASSERT_NE(operation, nullptr);
    ASSERT_NE(computeDataType, nullptr);
    EXPECT_TRUE(pointwise->operandLists.empty());
    EXPECT_TRUE(pointwise->resultLists.empty());

    const auto pointwiseBuffer = buildPointwiseAddNode();
    const auto& pointwiseNode = rootNode(pointwiseBuffer);
    EXPECT_EQ(in0->read(pointwiseNode), std::optional<int64_t>(1));
    EXPECT_FALSE(in1->read(pointwiseNode).has_value());
    EXPECT_EQ(out0->read(pointwiseNode), std::optional<int64_t>(2));
    EXPECT_EQ(operation->read(pointwiseNode), std::optional<ScalarValue>(enumName("ADD")));
    EXPECT_EQ(computeDataType->read(pointwiseNode), std::optional<ScalarValue>(enumName("FLOAT")));

    // A node carrying another op's table reads as absent, except for Node-table scalars.
    const auto customOpBuffer = buildCustomOpNode({1}, {2});
    const auto& customOpNode = rootNode(customOpBuffer);
    EXPECT_FALSE(in0->read(customOpNode).has_value());
    EXPECT_FALSE(out0->read(customOpNode).has_value());
    EXPECT_FALSE(operation->read(customOpNode).has_value());
    EXPECT_EQ(computeDataType->read(customOpNode), std::optional<ScalarValue>(enumName("FLOAT")));
}

// The registry's edge directions come from the matcher_input_tensor and
// matcher_output_tensor annotations. A flipped annotation regenerates cleanly and
// passes every structural check, then makes the matcher bind the tensor on the
// wrong side. These expectations follow each op's semantics, not the generated
// output, for the ops whose direction is easiest to get wrong: backward ops,
// ops with several results, and tensors that are an output in one op and an
// input in its counterpart.
TEST(TestOpSchemaRegistry, EdgeDirectionsFollowTheOpSemantics)
{
    const std::vector<ExpectedEdges> expectations = {
        {"convolution_bwd_data", {"dy", "w"}, {"dx"}},
        {"convolution_bwd_filter", {"x", "dy"}, {"dw"}},
        {"batchnorm",
         {"x",
          "scale",
          "bias",
          "epsilon",
          "prev_running_mean?",
          "prev_running_variance?",
          "momentum?"},
         {"y", "mean?", "inv_variance?", "next_running_mean?", "next_running_variance?"}},
        {"batchnorm_bwd",
         {"dy", "x", "mean?", "inv_variance?", "scale"},
         {"dx", "dscale", "dbias"}},
        {"layernorm", {"x", "scale", "bias", "epsilon"}, {"y", "mean?", "inv_variance?"}},
        {"rms_norm_bwd", {"dy", "x", "scale", "inv_rms"}, {"dx", "dscale", "dbias?"}},
        {"block_scale_quantize", {"x"}, {"y", "scale"}},
        {"block_scale_dequantize", {"x", "scale"}, {"y"}},
        {"resample_fwd", {"x"}, {"y", "index?"}},
        {"resample_bwd", {"dy", "index?"}, {"dx"}},
        {"sdpa_bwd",
         {"q",
          "k",
          "v",
          "o",
          "do",
          "stats",
          "scale?",
          "attn_mask?",
          "seq_len_q?",
          "seq_len_kv?",
          "seed?",
          "offset?",
          "dropout_mask?",
          "dropout_scale?",
          "dropout_scale_inv?"},
         {"dq", "dk", "dv", "dbias?"}},
    };

    for(const auto& expected : expectations)
    {
        const auto* op = findOpSchema(expected.opcode);
        ASSERT_NE(op, nullptr) << expected.opcode;
        EXPECT_EQ(edgeNames(op->operands), expected.operands) << expected.opcode;
        EXPECT_EQ(edgeNames(op->results), expected.results) << expected.opcode;
    }
}
