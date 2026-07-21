// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Unit tests for the OpSchema registry: opcode strings, operand/result role
// names + arities, EdgeReader UID extraction against real node attributes, and
// registry lookup (find / forNode / size). All graphs are built host-only with
// the header-only FlatBuffer test-SDK builders.

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/OpSchema.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <string_view>
#include <vector>

namespace {

namespace gm = hipdnn::graph_matcher;
namespace data = hipdnn_flatbuffers_sdk::data_objects;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace util = hipdnn_test_sdk::utilities;

using data::NodeAttributes;

// Reads the UIDs a single role contributes from a live node's raw attributes.
std::vector<int64_t> readRole(const gm::EdgeRole& role, const void* attrs) {
    std::vector<int64_t> out;
    role.read(attrs, out);
    return out;
}

void expectRole(const gm::EdgeRole& role, std::string_view name, gm::Arity arity) {
    EXPECT_EQ(role.name, name);
    EXPECT_EQ(role.arity, arity);
}

// The first node's raw attribute union pointer from a freshly built graph.
const void* firstNodeAttrs(const fbu::GraphWrapper& graph) {
    return graph.getNodeWrapper(0).attributes();
}

}  // namespace

// --- Registry structural invariants --------------------------------------

// Every registered schema is reachable through find(), and the count of
// reachable slots equals size(). This pins the _schemas/_byType consistency:
// a schema dropped from the table, or a stale index, reddens this.
TEST(OpSchemaRegistry, EveryRegisteredOpIsFindableAndSizeMatches) {
    const auto& reg = gm::OpSchemaRegistry::builtin();

    size_t reachable = 0;
    for (size_t i = 0; i <= static_cast<size_t>(NodeAttributes::MAX); ++i) {
        if (reg.find(static_cast<NodeAttributes>(i)) != nullptr) {
            ++reachable;
        }
    }

    EXPECT_EQ(reachable, reg.size());
    // graph.fbs registers every union member except NONE (enum values 1..MAX).
    EXPECT_EQ(reg.size(), static_cast<size_t>(NodeAttributes::MAX));
}

TEST(OpSchemaRegistry, FindNoneIsNull) {
    const auto& reg = gm::OpSchemaRegistry::builtin();
    EXPECT_EQ(reg.find(NodeAttributes::NONE), nullptr);
}

TEST(OpSchemaRegistry, FindReturnsSchemaWhoseTypeMatchesTheQuery) {
    const auto& reg = gm::OpSchemaRegistry::builtin();
    const auto* schema = reg.find(NodeAttributes::MatmulAttributes);
    ASSERT_NE(schema, nullptr);
    EXPECT_EQ(schema->type, NodeAttributes::MatmulAttributes);
}

// --- Per-op opcode + role names/arities ----------------------------------

TEST(OpSchemaRegistry, MatmulSchema) {
    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::MatmulAttributes);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->opcode, "matmul");

    ASSERT_EQ(s->operands.size(), 2u);
    expectRole(s->operands[0], "a", gm::Arity::Required);
    expectRole(s->operands[1], "b", gm::Arity::Required);

    ASSERT_EQ(s->results.size(), 1u);
    expectRole(s->results[0], "c", gm::Arity::Required);
}

TEST(OpSchemaRegistry, PointwiseSchemaOptionalInputs) {
    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::PointwiseAttributes);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->opcode, "pointwise");

    ASSERT_EQ(s->operands.size(), 3u);
    expectRole(s->operands[0], "in_0", gm::Arity::Required);
    expectRole(s->operands[1], "in_1", gm::Arity::Optional);
    expectRole(s->operands[2], "in_2", gm::Arity::Optional);

    ASSERT_EQ(s->results.size(), 1u);
    expectRole(s->results[0], "out_0", gm::Arity::Required);
}

TEST(OpSchemaRegistry, ConvFwdSchema) {
    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::ConvolutionFwdAttributes);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->opcode, "conv_fwd");

    ASSERT_EQ(s->operands.size(), 2u);
    expectRole(s->operands[0], "x", gm::Arity::Required);
    expectRole(s->operands[1], "w", gm::Arity::Required);

    ASSERT_EQ(s->results.size(), 1u);
    expectRole(s->results[0], "y", gm::Arity::Required);
}

TEST(OpSchemaRegistry, ReductionSchema) {
    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::ReductionAttributes);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->opcode, "reduction");

    ASSERT_EQ(s->operands.size(), 1u);
    expectRole(s->operands[0], "in", gm::Arity::Required);

    ASSERT_EQ(s->results.size(), 1u);
    expectRole(s->results[0], "out", gm::Arity::Required);
}

TEST(OpSchemaRegistry, SdpaFwdSchemaRequiredThenOptional) {
    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::SdpaAttributes);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->opcode, "sdpa_fwd");

    // q,k,v required, then a long tail of optional operands.
    ASSERT_EQ(s->operands.size(), 21u);
    expectRole(s->operands[0], "q", gm::Arity::Required);
    expectRole(s->operands[1], "k", gm::Arity::Required);
    expectRole(s->operands[2], "v", gm::Arity::Required);
    expectRole(s->operands[3], "attn_mask", gm::Arity::Optional);
    expectRole(s->operands[4], "scale", gm::Arity::Optional);
    for (size_t i = 3; i < s->operands.size(); ++i) {
        EXPECT_EQ(s->operands[i].arity, gm::Arity::Optional) << "operand " << i;
    }

    // o required, stats/max/etc. optional.
    ASSERT_EQ(s->results.size(), 7u);
    expectRole(s->results[0], "o", gm::Arity::Required);
    expectRole(s->results[1], "stats", gm::Arity::Optional);
    for (size_t i = 1; i < s->results.size(); ++i) {
        EXPECT_EQ(s->results[i].arity, gm::Arity::Optional) << "result " << i;
    }
}

TEST(OpSchemaRegistry, SdpaBwdSchema) {
    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::SdpaBackwardAttributes);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->opcode, "sdpa_bwd");

    ASSERT_EQ(s->operands.size(), 15u);
    expectRole(s->operands[0], "q", gm::Arity::Required);
    expectRole(s->operands[1], "k", gm::Arity::Required);
    expectRole(s->operands[2], "v", gm::Arity::Required);
    expectRole(s->operands[3], "o", gm::Arity::Required);
    expectRole(s->operands[4], "do", gm::Arity::Required);
    expectRole(s->operands[5], "stats", gm::Arity::Required);
    expectRole(s->operands[6], "scale", gm::Arity::Optional);

    ASSERT_EQ(s->results.size(), 4u);
    expectRole(s->results[0], "dq", gm::Arity::Required);
    expectRole(s->results[1], "dk", gm::Arity::Required);
    expectRole(s->results[2], "dv", gm::Arity::Required);
    expectRole(s->results[3], "dbias", gm::Arity::Optional);
}

TEST(OpSchemaRegistry, CustomOpSchemaVariadic) {
    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::CustomOpAttributes);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->opcode, "custom_op");

    ASSERT_EQ(s->operands.size(), 1u);
    expectRole(s->operands[0], "inputs", gm::Arity::Variadic);

    ASSERT_EQ(s->results.size(), 1u);
    expectRole(s->results[0], "outputs", gm::Arity::Variadic);
}

// --- EdgeReader UID extraction against real node attributes ---------------

TEST(OpSchemaEdgeReader, MatmulReadsRequiredOperandAndResultUids) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const void* attrs = firstNodeAttrs(graph);

    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::MatmulAttributes);
    ASSERT_NE(s, nullptr);

    EXPECT_EQ(readRole(s->operands[0], attrs), (std::vector<int64_t>{1}));  // a
    EXPECT_EQ(readRole(s->operands[1], attrs), (std::vector<int64_t>{2}));  // b
    EXPECT_EQ(readRole(s->results[0], attrs), (std::vector<int64_t>{3}));   // c
}

TEST(OpSchemaEdgeReader, PointwiseReadsPresentOptionalInputs) {
    // createPointwiseGraph wires in_0=1, in_1=2, in_2=3, out_0=4 (all present).
    auto builder = util::createPointwiseGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const void* attrs = firstNodeAttrs(graph);

    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::PointwiseAttributes);
    ASSERT_NE(s, nullptr);

    EXPECT_EQ(readRole(s->operands[0], attrs), (std::vector<int64_t>{1}));  // in_0
    EXPECT_EQ(readRole(s->operands[1], attrs), (std::vector<int64_t>{2}));  // in_1 (optional, set)
    EXPECT_EQ(readRole(s->operands[2], attrs), (std::vector<int64_t>{3}));  // in_2 (optional, set)
    EXPECT_EQ(readRole(s->results[0], attrs), (std::vector<int64_t>{4}));   // out_0
}

TEST(OpSchemaEdgeReader, SdpaFwdOptionalOperandAbsentReadsNothing) {
    // Default SDPA fwd graph: q=1,k=2,v=3,o=4, no attn_mask/scale/stats.
    auto builder = util::createValidSdpaFwdGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const void* attrs = firstNodeAttrs(graph);

    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::SdpaAttributes);
    ASSERT_NE(s, nullptr);

    EXPECT_EQ(readRole(s->operands[0], attrs), (std::vector<int64_t>{1}));  // q
    EXPECT_EQ(readRole(s->operands[1], attrs), (std::vector<int64_t>{2}));  // k
    EXPECT_EQ(readRole(s->operands[2], attrs), (std::vector<int64_t>{3}));  // v
    EXPECT_TRUE(readRole(s->operands[3], attrs).empty());                   // attn_mask absent
    EXPECT_TRUE(readRole(s->operands[4], attrs).empty());                   // scale absent

    EXPECT_EQ(readRole(s->results[0], attrs), (std::vector<int64_t>{4}));  // o
    EXPECT_TRUE(readRole(s->results[1], attrs).empty());                   // stats absent
}

TEST(OpSchemaEdgeReader, SdpaFwdOptionalOperandPresentIsRead) {
    // withScale=true appends a scale tensor (uid 5, since attn_mask omitted).
    auto builder = util::createValidSdpaFwdGraph(
        {2, 8, 16, 64}, {8192, 1024, 64, 1}, {2, 8, 16, 64}, {8192, 1024, 64, 1}, {2, 8, 16, 64},
        {8192, 1024, 64, 1}, {2, 8, 16, 64}, {8192, 1024, 64, 1}, data::DataType::HALF,
        /*withAttnMask=*/false,
        /*withScale=*/true);
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const void* attrs = firstNodeAttrs(graph);

    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::SdpaAttributes);
    ASSERT_NE(s, nullptr);

    EXPECT_TRUE(readRole(s->operands[3], attrs).empty());  // attn_mask still absent
    EXPECT_EQ(readRole(s->operands[4], attrs), (std::vector<int64_t>{5}));  // scale present
}

TEST(OpSchemaEdgeReader, CustomOpVariadicReadsWholeVectors) {
    // createValidCustomOpGraph: inputs=[1], outputs=[2].
    auto builder = util::createValidCustomOpGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const void* attrs = firstNodeAttrs(graph);

    const auto* s = gm::OpSchemaRegistry::builtin().find(NodeAttributes::CustomOpAttributes);
    ASSERT_NE(s, nullptr);

    EXPECT_EQ(readRole(s->operands[0], attrs), (std::vector<int64_t>{1}));  // inputs
    EXPECT_EQ(readRole(s->results[0], attrs), (std::vector<int64_t>{2}));   // outputs
}

// --- forNode resolution --------------------------------------------------

TEST(OpSchemaRegistry, ForNodeResolvesViaAttributesType) {
    auto builder = util::createValidReductionGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& reg = gm::OpSchemaRegistry::builtin();
    const auto* viaNode = reg.forNode(graph.getNodeWrapper(0));
    ASSERT_NE(viaNode, nullptr);
    EXPECT_EQ(viaNode->opcode, "reduction");
    // forNode must resolve to the same schema find() gives for that union type.
    EXPECT_EQ(viaNode, reg.find(NodeAttributes::ReductionAttributes));
}
