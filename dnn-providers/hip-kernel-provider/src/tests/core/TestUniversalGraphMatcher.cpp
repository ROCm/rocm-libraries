// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// RFC 0018 Phase 1: proves the bindings + UMD compiler + matcher end to end.
//   - Compiler: A.10 validation (accept the SDPA descriptor; reject bad schema,
//     mismatched `?`, unknown name, type errors, custom ops).
//   - Lowering: `shape` -> rank check with named dims; layout alias -> array.
//   - Matcher: structural match + criteria evaluation over a live graph, with
//     fail-closed rejection on the declarative gates.
//   - Bindings: the queryable BindingContext resolves every `$`-namespace form.
//
// The full match-equivalence battery against SdpaFwdPlanBuilder::isApplicable
// and the enumerated binding-completeness proof are the Phase 2 deliverable.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "hip_kernel_provider_common/umd/BindingContext.hpp"
#include "hip_kernel_provider_common/umd/UmdCompiler.hpp"
#include "hip_kernel_provider_common/umd/UniversalGraphMatcher.hpp"

namespace umd = hip_kernel_provider_common::umd;
namespace jlogic = hip_kernel_provider_common::jsonlogic;
namespace data = hipdnn_flatbuffers_sdk::data_objects;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

using json = nlohmann::json;

namespace
{

// The RFC §18 SDPA-forward descriptor, with the two custom-operation gates
// (strides_fit_u32, sdpa_mask_consistent) removed -- the custom-op hook is out
// of scope for this PoC. Name keys are the lowercase op-schema registry names.
json sdpaDescriptor()
{
    return json::parse(R"JSON({
      "schema": "hipdnn.umd/v1",
      "id": "9c3f5b2a-7d41-4e88-b6a0-1f2e3d4c5b6a",
      "name": "SDPA forward (d128, bf16/fp8) match",
      "nodes": [
        {"kind": "op", "id": "sdpa_fwd", "op": "sdpa_fwd",
         "operands": {"q": "$q", "k": "$k", "v": "$v",
                      "attn_mask": "$attn_mask?", "page_table_k": "$page_table_k?",
                      "page_table_v": "$page_table_v?"},
         "results":  {"o": "$o"}}
      ],
      "criteria": {"and": [
        {"==": ["$graph.node_count", 1]},
        {"in": ["$q.dtype", ["BFLOAT16", "FP8_E4M3"]]},
        {"==": ["$k.dtype", "$q.dtype"]}, {"==": ["$v.dtype", "$q.dtype"]},
        {"shape": ["$q", ["batch", "num_heads", "seqlen_q", "head_size"]]},
        {"shape": ["$k", ["batch", "kv_heads",  "seqlen_k", "head_size"]]},
        {"shape": ["$v", ["batch", "kv_heads",  "seqlen_k", "head_size"]]},
        {"==": ["$o.rank", 4]},
        {"==": ["$q.head_size", 128]},
        {"==": ["$k.head_size", "$q.head_size"]},
        {"or": [{"!": "$sdpa_fwd.dropout_probability.present"}, {"==": ["$sdpa_fwd.dropout_probability", 0.0]}]},
        {"==": ["$sdpa_fwd.alibi_mask", false]},
        {"==": ["$sdpa_fwd.padding_mask", false]},
        {"or": [{"!": "$sdpa_fwd.generate_stats.present"}, {"==": ["$sdpa_fwd.generate_stats", false]}]},
        {"!": "$attn_mask.present"},
        {"!": "$page_table_k.present"}, {"!": "$page_table_v.present"}
      ]}
    })JSON");
}

std::vector<std::int64_t> contiguousStrides(const std::vector<std::int64_t>& dims)
{
    std::vector<std::int64_t> strides(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i-- > 0;)
    {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

// Build a single-node SDPA-forward graph with the given per-tensor dims and
// dtype (contiguous strides), returning the owning builder.
flatbuffers::FlatBufferBuilder buildSdpaGraph(const std::vector<std::int64_t>& dims,
                                              data::DataType dtype,
                                              bool withAttnMask = false,
                                              bool overrideShape = false)
{
    const std::vector<std::int64_t> strides = contiguousStrides(dims);
    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dtype,
                                                               withAttnMask,
                                                               /*withScale=*/false,
                                                               /*withStats=*/false,
                                                               /*alibiMask=*/false,
                                                               /*paddingMask=*/false,
                                                               /*causalMask=*/false,
                                                               overrideShape);
}

umd::UniversalGraphMatcher makeSdpaMatcher()
{
    return umd::UniversalGraphMatcher(sdpaDescriptor());
}

constexpr umd::DeviceProperties K_DEVICE{/*ldsSize=*/65536, /*warpSize=*/64};

} // namespace

// ---- Compiler validation (A.10) ------------------------------------------

TEST(UmdCompiler, AcceptsSdpaDescriptor)
{
    EXPECT_NO_THROW(umd::UmdCompiler::compile(sdpaDescriptor()));
}

TEST(UmdCompiler, PublishesReferencedBoundSymbols)
{
    const umd::CompiledUmd c = umd::UmdCompiler::compile(sdpaDescriptor());
    // The `shape` short-hands lower to `$q.rank` etc.; named dims are read
    // directly.
    EXPECT_TRUE(c.boundSymbols.count("q.head_size") == 1);
    EXPECT_TRUE(c.boundSymbols.count("graph.node_count") == 1);
    EXPECT_TRUE(c.boundSymbols.count("sdpa_fwd.dropout_probability") == 1);
    EXPECT_TRUE(c.boundSymbols.count("attn_mask.present") == 1);
}

TEST(UmdCompiler, RejectsWrongSchema)
{
    json d = sdpaDescriptor();
    d["schema"] = "hipdnn.umd/v2";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsMalformedUuid)
{
    json d = sdpaDescriptor();
    d["id"] = "not-a-uuid";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsUnknownTopLevelKey)
{
    json d = sdpaDescriptor();
    d["priority"] = 3;
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsUnknownOperandName)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["operands"]["not_a_name"] = "$bogus";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsQuestionOnRequiredName)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["operands"]["q"] = "$q?"; // q is required
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsMissingQuestionOnOptionalName)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["operands"]["attn_mask"] = "$attn_mask"; // optional, needs ?
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsReservedRootAsNodeId)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["id"] = "device";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsTypeMismatchIntVsDtype)
{
    json d = sdpaDescriptor();
    // $q.head_size is Int; comparing it to a Dtype string is a compile error.
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$q.head_size", "BFLOAT16"]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsNonBooleanCriteria)
{
    json d = sdpaDescriptor();
    d["criteria"] = json::parse(R"({"+": ["$q.head_size", 1]})"); // Int, not Bool
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsCustomOperation)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"hipdnn.strides_fit_u32": ["$q"]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsPresentOnRequiredOperand)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"!": "$q.present"})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsWrongArity)
{
    // Binary/unary operators with the wrong argument count must refuse at
    // compile with a UmdCompileError, not escape as std::out_of_range.
    for(const char* bad : {R"({"==": ["$q.rank"]})",
                           R"({"==": ["$q.rank", 4, 4]})",
                           R"({"<": ["$q.rank"]})",
                           R"({"in": ["$q.dtype"]})",
                           R"({"and": [true]})",
                           R"({"if": [true, 1]})"})
    {
        json d = sdpaDescriptor();
        d["criteria"] = json::parse(bad);
        EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError) << bad;
    }
}

TEST(UmdCompiler, RejectsMalformedSubscript)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$q.dims[x]", 2]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsBareNodeIdAsValue)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$sdpa_fwd", 0]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsLayoutAliasRankMismatch)
{
    // $q is pinned rank 4 by the shape short-hand; a rank-5 alias can never
    // match, so it is refused at compile rather than always declining.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$q.stride_order", "ncdhw"]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, AcceptsWellFormedArithmeticAndIf)
{
    // Guard the arity checks against over-rejecting valid n-ary / binary / if
    // forms.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(
        R"({"==": [{"if": [{"<": ["$q.head_size", 64]}, {"-": ["$q.head_size", 1]}, 128]}, 128]})"));
    EXPECT_NO_THROW(umd::UmdCompiler::compile(d));
}

// ---- Matcher: accept / reject --------------------------------------------

TEST(UniversalGraphMatcher, OwnsSingleDescriptor)
{
    // A matcher has a 1:1 relationship with its UMD: it exposes that one
    // descriptor's id and compiled form.
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    EXPECT_EQ(m.umdId(), "9c3f5b2a-7d41-4e88-b6a0-1f2e3d4c5b6a");
    EXPECT_EQ(m.descriptor().id, "9c3f5b2a-7d41-4e88-b6a0-1f2e3d4c5b6a");
    EXPECT_EQ(m.descriptor().nodes.size(), 1u);
}

TEST(UniversalGraphMatcher, MatchesValidSdpaForward)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    const umd::MatchResult r = m.match(K_DEVICE, g);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.umdId, "9c3f5b2a-7d41-4e88-b6a0-1f2e3d4c5b6a");
}

TEST(UniversalGraphMatcher, DeclinesWrongHeadSize)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 64}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(UniversalGraphMatcher, DeclinesUnsupportedDtype)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::HALF);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(UniversalGraphMatcher, DeclinesWhenUnsupportedOptionalPresent)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16, /*withAttnMask=*/true);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(UniversalGraphMatcher, DeclinesOverrideShapeGraph)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128},
                                  data::DataType::BFLOAT16,
                                  /*withAttnMask=*/false,
                                  /*overrideShape=*/true);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(UniversalGraphMatcher, DeclinesGraphWithoutMatchingOpcode)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = hipdnn_test_sdk::utilities::createValidBlockScaleQuantizeGraph();
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(UniversalGraphMatcher, DeclinesNodelessGraphWithoutThrowing)
{
    // A valid but empty graph must fail closed, not throw (nodeWrappers() would
    // throw on a null nodes vector).
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = hipdnn_test_sdk::utilities::createEmptyValidGraph();
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_NO_THROW({ EXPECT_FALSE(m.match(K_DEVICE, g).matched); });
}

// ---- Bindings: queryable BindingContext (RFC 0018 §4/§15) ----------------

TEST(BindingContext, ResolvesEveryNamespaceForm)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    const umd::MatchResult r = m.match(K_DEVICE, g);
    ASSERT_TRUE(r.matched);
    const umd::BindingContext& b = r.bindings;

    // Tensor namespace.
    EXPECT_EQ(b.get("$q.uid").asInt(), 1);
    EXPECT_EQ(b.get("$q.rank").asInt(), 4);
    EXPECT_EQ(b.get("$q.dtype").asString(), "BFLOAT16");
    EXPECT_EQ(b.get("$q.dims[0]").asInt(), 2); // positional
    EXPECT_EQ(b.get("$q.head_size").asInt(), 128); // named dim
    EXPECT_EQ(b.get("$q.strides[3]").asInt(), 1);
    EXPECT_TRUE(b.get("$q.packed").asBool());
    EXPECT_FALSE(b.get("$q.virtual").asBool());

    // stride_order uses extractStrideOrder's per-dim priority encoding
    // (ApplicabilityChecks.cpp:17): contiguous rank-4 is [3,2,1,0].
    const jlogic::Value strideOrder = b.get("$q.stride_order");
    ASSERT_TRUE(strideOrder.isArray());
    EXPECT_EQ(strideOrder,
              jlogic::Value(jlogic::Value::Array{jlogic::Value(std::int64_t{3}),
                                                 jlogic::Value(std::int64_t{2}),
                                                 jlogic::Value(std::int64_t{1}),
                                                 jlogic::Value(std::int64_t{0})}));

    // Attributes namespace.
    EXPECT_FALSE(b.get("$sdpa_fwd.dropout_probability.present").asBool());
    EXPECT_FALSE(b.get("$sdpa_fwd.alibi_mask").asBool());

    // Graph and device namespaces.
    EXPECT_EQ(b.get("$graph.node_count").asInt(), 1);
    EXPECT_EQ(b.get("$device.lds_size").asInt(), 65536);
    EXPECT_EQ(b.get("$device.warp_size").asInt(), 64);
}

TEST(BindingContext, FailsClosedOnBadPaths)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    const umd::MatchResult r = m.match(K_DEVICE, g);
    ASSERT_TRUE(r.matched);
    const umd::BindingContext& b = r.bindings;

    EXPECT_TRUE(b.get("$q.dims[9]").isNull()); // out of range
    EXPECT_TRUE(b.get("$q.unknown_dim").isNull()); // unknown dim-name
    EXPECT_TRUE(b.get("$attn_mask.dtype").isNull()); // absent optional field read
    EXPECT_TRUE(b.get("$device.unknown").isNull()); // unknown device field
    EXPECT_TRUE(b.get("$nope.rank").isNull()); // unknown tensor var
}

// ---- Lowering: layout alias (A.8) ----------------------------------------

TEST(UmdCompiler, LayoutAliasMatchesContiguousLayout)
{
    const umd::UniversalGraphMatcher m(json::parse(R"JSON({
      "schema": "hipdnn.umd/v1",
      "id": "11111111-2222-3333-4444-555555555555",
      "name": "layout alias smoke",
      "nodes": [{"kind": "op", "id": "sdpa_fwd", "op": "sdpa_fwd",
                 "operands": {"q": "$q"}}],
      "criteria": {"and": [
        {"==": ["$graph.node_count", 1]},
        {"shape": ["$q", ["b", "h", "s", "d"]]},
        {"==": ["$q.stride_order", "nchw"]}
      ]}
    })JSON"));

    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, g).matched);
}

// ---- Multi-node patterns (RFC 0018 §3/A.3) -------------------------------

namespace
{

// Build a two-node SDPA graph. When `chained`, node1's output tensor (uid 4) is
// also node2's Q input, forming an edge; otherwise node2 reads an independent Q
// (uid 8), so no edge exists. `alibi2` sets node2's alibi_mask to exercise
// per-node attribute resolution.
flatbuffers::FlatBufferBuilder buildTwoNodeSdpaGraph(bool chained, bool alibi2 = false)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<std::int64_t> dims{2, 8, 16, 128};
    const std::vector<std::int64_t> strides = contiguousStrides(dims);
    const data::DataType dtype = data::DataType::BFLOAT16;

    const auto mkTensor = [&](std::int64_t uid, const char* name) {
        return data::CreateTensorAttributesDirect(builder, uid, name, dtype, &strides, &dims);
    };

    std::vector<::flatbuffers::Offset<data::TensorAttributes>> tensors;
    tensors.push_back(mkTensor(1, "q1"));
    tensors.push_back(mkTensor(2, "k1"));
    tensors.push_back(mkTensor(3, "v1"));
    tensors.push_back(mkTensor(4, "o1")); // node1 output
    tensors.push_back(mkTensor(5, "k2"));
    tensors.push_back(mkTensor(6, "v2"));
    tensors.push_back(mkTensor(7, "o2")); // node2 output
    const std::int64_t node2q = chained ? 4 : 8;
    if(!chained)
    {
        tensors.push_back(mkTensor(8, "q2"));
    }

    const auto mkSdpa
        = [&](std::int64_t q, std::int64_t k, std::int64_t v, std::int64_t o, bool alibi) {
              data::SdpaAttributesBuilder sb(builder);
              sb.add_q_tensor_uid(q);
              sb.add_k_tensor_uid(k);
              sb.add_v_tensor_uid(v);
              sb.add_o_tensor_uid(o);
              sb.add_alibi_mask(alibi);
              return sb.Finish();
          };
    const auto s1 = mkSdpa(1, 2, 3, 4, /*alibi=*/false);
    const auto s2 = mkSdpa(node2q, 5, 6, 7, alibi2);

    std::vector<::flatbuffers::Offset<data::Node>> nodes;
    nodes.push_back(data::CreateNodeDirect(
        builder, "sdpa1", dtype, data::NodeAttributes::SdpaAttributes, s1.Union()));
    nodes.push_back(data::CreateNodeDirect(
        builder, "sdpa2", dtype, data::NodeAttributes::SdpaAttributes, s2.Union()));

    auto graph = data::CreateGraphDirect(builder,
                                         "test",
                                         data::DataType::FLOAT,
                                         data::DataType::HALF,
                                         data::DataType::BFLOAT16,
                                         &tensors,
                                         &nodes,
                                         flatbuffers::nullopt,
                                         /*is_override_shape_enabled=*/false);
    builder.Finish(graph);
    return builder;
}

// A two-node fusion pattern: two SDPA nodes chained by the shared `$mid` edge,
// each bound to its own node id (`sdpa1`, `sdpa2`).
json twoNodeChainDescriptor()
{
    return json::parse(R"JSON({
      "schema": "hipdnn.umd/v1",
      "id": "22222222-3333-4444-5555-666666666666",
      "name": "two-node sdpa chain",
      "nodes": [
        {"kind": "op", "id": "sdpa1", "op": "sdpa_fwd",
         "operands": {"q": "$q", "k": "$k1", "v": "$v1"}, "results": {"o": "$mid"}},
        {"kind": "op", "id": "sdpa2", "op": "sdpa_fwd",
         "operands": {"q": "$mid", "k": "$k2", "v": "$v2"}, "results": {"o": "$out"}}
      ],
      "criteria": {"and": [
        {"==": ["$graph.node_count", 2]},
        {"shape": ["$mid", ["b", "h", "s", "d"]]},
        {"==": ["$mid.d", 128]},
        {"==": ["$q.dtype", "$out.dtype"]},
        {"==": ["$sdpa1.alibi_mask", false]},
        {"==": ["$sdpa2.alibi_mask", false]}
      ]}
    })JSON");
}

} // namespace

TEST(UmdCompiler, AcceptsMultiNodeDescriptor)
{
    const umd::CompiledUmd c = umd::UmdCompiler::compile(twoNodeChainDescriptor());
    EXPECT_EQ(c.nodes.size(), 2u);
    ASSERT_NE(c.findNode("sdpa1"), nullptr);
    ASSERT_NE(c.findNode("sdpa2"), nullptr);
    // The `$mid` edge variable is shared by both nodes (result of one, operand
    // of the other) but declared once.
    ASSERT_NE(c.findTvar("mid"), nullptr);
}

TEST(UmdCompiler, RejectsDuplicateNodeId)
{
    json d = twoNodeChainDescriptor();
    d["nodes"][1]["id"] = "sdpa1"; // collides with node 0
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, RejectsVariableProducedTwice)
{
    json d = twoNodeChainDescriptor();
    d["nodes"][1]["results"]["o"] = "$mid"; // $mid produced by both nodes
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UniversalGraphMatcher, MatchesTwoNodeChainAndBindsEachNode)
{
    const umd::UniversalGraphMatcher m(twoNodeChainDescriptor());
    auto builder = buildTwoNodeSdpaGraph(/*chained=*/true);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    const umd::MatchResult r = m.match(K_DEVICE, g);
    ASSERT_TRUE(r.matched);
    const umd::BindingContext& b = r.bindings;
    // Each pattern variable resolved to its own graph tensor; the shared edge
    // variable resolved to the tensor produced by node1 and consumed by node2.
    EXPECT_EQ(b.get("$q.uid").asInt(), 1);
    EXPECT_EQ(b.get("$mid.uid").asInt(), 4);
    EXPECT_EQ(b.get("$out.uid").asInt(), 7);
    EXPECT_EQ(b.get("$mid.d").asInt(), 128); // named dim on the edge tensor
    // Each node id resolves its own attributes namespace.
    EXPECT_FALSE(b.get("$sdpa1.alibi_mask").asBool());
    EXPECT_FALSE(b.get("$sdpa2.alibi_mask").asBool());
}

TEST(UniversalGraphMatcher, DeclinesWhenEdgeIsInconsistent)
{
    const umd::UniversalGraphMatcher m(twoNodeChainDescriptor());
    // node2 reads an independent Q, so no tensor satisfies the shared `$mid`
    // edge (node1.o != node2.q) -> no consistent assignment -> decline.
    auto builder = buildTwoNodeSdpaGraph(/*chained=*/false);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(UniversalGraphMatcher, PerNodeAttributesAreDistinct)
{
    const umd::UniversalGraphMatcher m(twoNodeChainDescriptor());
    // Chained graph, but node2's alibi_mask is set; the criterion
    // {"==": ["$sdpa2.alibi_mask", false]} must read node2's own attribute and
    // decline, proving per-node attribute resolution.
    auto builder = buildTwoNodeSdpaGraph(/*chained=*/true, /*alibi2=*/true);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

// ---- Kernel metadata ($kernel) -------------------------------------------

namespace
{

// The SDPA descriptor with one `$kernel.<field>` criterion added. Compiles only
// because the compiler resolves `$kernel.*` as a runtime-DYNAMIC scalar.
json sdpaKernelDescriptor()
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$kernel.head_dim", "$q.head_size"]})"));
    return d;
}

} // namespace

TEST(UmdCompiler, AcceptsKernelReference)
{
    // A criterion over a $kernel field compiles (DYNAMIC wildcard) and the
    // descriptor is flagged as referencing kernel metadata.
    umd::CompiledUmd c;
    ASSERT_NO_THROW(c = umd::UmdCompiler::compile(sdpaKernelDescriptor()));
    EXPECT_TRUE(c.referencesKernelMetadata);
    EXPECT_TRUE(c.boundSymbols.count("kernel.head_dim") == 1);
}

TEST(UmdCompiler, StockDescriptorDoesNotReferenceKernel)
{
    const umd::CompiledUmd c = umd::UmdCompiler::compile(sdpaDescriptor());
    EXPECT_FALSE(c.referencesKernelMetadata);
}

TEST(UmdCompiler, KernelFieldUnifiesWithAnyScalarDomain)
{
    // A $kernel field is DYNAMIC: it compiles against an int, a dtype string,
    // and a bool alike -- none is a static type error.
    for(const char* crit : {R"({"==": ["$kernel.head_dim", 128]})",
                            R"({"==": ["$kernel.target_dtype", "BFLOAT16"]})",
                            R"({"<": ["$kernel.tile_m", "$q.head_size"]})",
                            R"({"!": "$kernel.causal"})"})
    {
        json d = sdpaDescriptor();
        d["criteria"]["and"].push_back(json::parse(crit));
        EXPECT_NO_THROW(umd::UmdCompiler::compile(d)) << crit;
    }
}

TEST(UmdCompiler, KernelReferenceNeedsAField)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$kernel", 1]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, KernelFieldAsIfCondition)
{
    // A DYNAMIC $kernel field is accepted in an `if` condition, consistent with
    // its acceptance in and/or/! boolean positions.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": [{"if": ["$kernel.causal", 1, 0]}, 0]})"));
    EXPECT_NO_THROW(umd::UmdCompiler::compile(d));
}

TEST(UmdCompiler, KernelFieldDoesNotUnifyWithArray)
{
    // DYNAMIC is a scalar wildcard: comparing a $kernel field to an array
    // literal is a static type error, not a runtime-deferred check.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$kernel.tiles", [1, 2, 3]]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, KernelIsReservedAsPatternVariable)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["operands"]["q"] = "$kernel"; // bind a tensor var to the reserved root
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UmdCompiler, KernelIsReservedAsNodeId)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["id"] = "kernel";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(UniversalGraphMatcher, ReportsKernelMetadataReference)
{
    const umd::UniversalGraphMatcher mk(sdpaKernelDescriptor());
    EXPECT_TRUE(mk.referencesKernelMetadata());
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    EXPECT_FALSE(m.referencesKernelMetadata());
}

TEST(UniversalGraphMatcher, TwoArgMatchThrowsWhenDescriptorNeedsKernel)
{
    const umd::UniversalGraphMatcher m(sdpaKernelDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    // Wrong overload for a kernel-referencing UMD -> programming error.
    EXPECT_THROW(m.match(K_DEVICE, g), std::logic_error);
}

TEST(UniversalGraphMatcher, MatchesWithSuppliedKernelMetadata)
{
    const umd::UniversalGraphMatcher m(sdpaKernelDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    // head_dim matches $q.head_size (128) -> match; the value is queryable.
    const umd::MatchResult r = m.match(K_DEVICE, g, json{{"head_dim", 128}});
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.bindings.get("$kernel.head_dim").asInt(), 128);
}

TEST(UniversalGraphMatcher, DeclinesWhenKernelMetadataMismatches)
{
    const umd::UniversalGraphMatcher m(sdpaKernelDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    // head_dim != $q.head_size (128) -> criterion false -> decline.
    EXPECT_FALSE(m.match(K_DEVICE, g, json{{"head_dim", 64}}).matched);
}

TEST(UniversalGraphMatcher, EmptyKernelMetadataDeclinesComparison)
{
    // Defensive edge case: production metadata is fully resolved so every
    // referenced field is present, but an unbound/empty document must not
    // throw -- $kernel.head_dim reads null -> comparison false -> decline.
    const umd::UniversalGraphMatcher m(sdpaKernelDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_NO_THROW({ EXPECT_FALSE(m.match(K_DEVICE, g, json::object()).matched); });
}
