// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// RFC 0018 Phase 1: proves the bindings + UMD compiler + matcher end to end.
//   - Compiler: A.10 validation (accept the SDPA descriptor; reject bad schema,
//     mismatched `?`, unknown name, type errors, unknown operators).
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
#include <set>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
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

// The RFC §18 SDPA-forward descriptor. The two gates that need real C++
// (strides_fit_u32, sdpa_mask_consistent) are not criteria: the expression
// language has no custom-operation hatch, so they live in a native matcher
// beside this UMD (RFC 0018 §8). Name keys are the lowercase op-schema
// registry names.
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
// dtype (contiguous strides), returning the owning builder. `withScale` adds
// the optional `scale` operand as a pass-by-value tensor carrying a
// compile-time Float32 value of 1.0.
flatbuffers::FlatBufferBuilder buildSdpaGraph(const std::vector<std::int64_t>& dims,
                                              data::DataType dtype,
                                              bool withAttnMask = false,
                                              bool overrideShape = false,
                                              bool withScale = false)
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
                                                               withScale,
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

// The sdpaDescriptor() `id`, in the form the compiled descriptor carries it.
const hipdnn_flatbuffers_sdk::utilities::UuidBytes K_SDPA_UMD_ID
    = hipdnn_flatbuffers_sdk::utilities::parseUuid("9c3f5b2a-7d41-4e88-b6a0-1f2e3d4c5b6a");

} // namespace

// ---- Compiler validation (A.10) ------------------------------------------

TEST(TestUmdCompiler, AcceptsSdpaDescriptor)
{
    EXPECT_NO_THROW(umd::UmdCompiler::compile(sdpaDescriptor()));
}

TEST(TestUmdCompiler, PublishesReferencedBoundSymbols)
{
    const umd::CompiledUmd c = umd::UmdCompiler::compile(sdpaDescriptor());
    // The `shape` short-hands lower to `$q.rank` etc.; named dims are read
    // directly.
    EXPECT_TRUE(c.boundSymbols.count("q.head_size") == 1);
    EXPECT_TRUE(c.boundSymbols.count("graph.node_count") == 1);
    EXPECT_TRUE(c.boundSymbols.count("sdpa_fwd.dropout_probability") == 1);
    EXPECT_TRUE(c.boundSymbols.count("attn_mask.present") == 1);
}

TEST(TestUmdCompiler, RejectsWrongSchema)
{
    json d = sdpaDescriptor();
    d["schema"] = "hipdnn.umd/v2";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsMalformedUuid)
{
    json d = sdpaDescriptor();
    d["id"] = "not-a-uuid";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsWellSizedNonHexUuid)
{
    // Right shape, wrong alphabet: the id is parsed, not pattern-matched, so a
    // non-hex digit must still surface as a compile error.
    json d = sdpaDescriptor();
    d["id"] = "9c3f5b2a-7d41-4e88-b6a0-1f2e3d4c5bZZ";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsNonStringUuid)
{
    json d = sdpaDescriptor();
    d["id"] = 42;
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsUnknownTopLevelKey)
{
    json d = sdpaDescriptor();
    d["priority"] = 3;
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsUnknownOperandName)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["operands"]["not_a_name"] = "$bogus";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsQuestionOnRequiredName)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["operands"]["q"] = "$q?"; // q is required
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsMissingQuestionOnOptionalName)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["operands"]["attn_mask"] = "$attn_mask"; // optional, needs ?
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsReservedRootAsNodeId)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["id"] = "device";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsTypeMismatchIntVsDtype)
{
    json d = sdpaDescriptor();
    // $q.head_size is Int; comparing it to a Dtype string is a compile error.
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$q.head_size", "BFLOAT16"]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsNonBooleanCriteria)
{
    json d = sdpaDescriptor();
    d["criteria"] = json::parse(R"({"+": ["$q.head_size", 1]})"); // Int, not Bool
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsUnknownOperator)
{
    // A.7 is exhaustive and there is no registry a descriptor can extend it
    // through, so a namespaced key -- the form a custom operation would have
    // taken -- is refused as an unrecognized operator like any other.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"hipdnn.strides_fit_u32": ["$q"]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsPresentOnRequiredOperand)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"!": "$q.present"})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsWrongArity)
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

TEST(TestUmdCompiler, RejectsMalformedSubscript)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$q.dims[x]", 2]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsBareNodeIdAsValue)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$sdpa_fwd", 0]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsLayoutAliasRankMismatch)
{
    // $q is pinned rank 4 by the shape short-hand; a rank-5 alias can never
    // match, so it is refused at compile rather than always declining.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$q.stride_order", "ncdhw"]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, AcceptsWellFormedArithmeticAndIf)
{
    // Guard the arity checks against over-rejecting valid n-ary / binary / if
    // forms.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(
        R"({"==": [{"if": [{"<": ["$q.head_size", 64]}, {"-": ["$q.head_size", 1]}, 128]}, 128]})"));
    EXPECT_NO_THROW(umd::UmdCompiler::compile(d));
}

// ---- Matcher: accept / reject --------------------------------------------

TEST(TestUniversalGraphMatcher, OwnsSingleDescriptor)
{
    // A matcher has a 1:1 relationship with its UMD: it exposes that one
    // descriptor's id and compiled form.
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    EXPECT_EQ(m.umdId(), K_SDPA_UMD_ID);
    EXPECT_EQ(m.descriptor().id, K_SDPA_UMD_ID);
    EXPECT_EQ(m.descriptor().nodes.size(), 1u);
}

TEST(TestUniversalGraphMatcher, MatchesValidSdpaForward)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    const umd::MatchResult r = m.match(K_DEVICE, g);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.umdId, K_SDPA_UMD_ID);
}

TEST(TestUniversalGraphMatcher, DeclinesWrongHeadSize)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 64}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, DeclinesUnsupportedDtype)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::HALF);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, DeclinesWhenUnsupportedOptionalPresent)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16, /*withAttnMask=*/true);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, DeclinesOverrideShapeGraph)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128},
                                  data::DataType::BFLOAT16,
                                  /*withAttnMask=*/false,
                                  /*overrideShape=*/true);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, DeclinesGraphWithoutMatchingOpcode)
{
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = hipdnn_test_sdk::utilities::createValidBlockScaleQuantizeGraph();
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, DeclinesNodelessGraphWithoutThrowing)
{
    // A valid but empty graph must fail closed, not throw (nodeWrappers() would
    // throw on a null nodes vector).
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = hipdnn_test_sdk::utilities::createEmptyValidGraph();
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_NO_THROW({ EXPECT_FALSE(m.match(K_DEVICE, g).matched); });
}

// ---- Bindings: queryable BindingContext (RFC 0018 §4/§15) ----------------

TEST(TestBindingContext, ResolvesEveryNamespaceForm)
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

    // stride_order is published in the RFC 0017 §5 form: logical dim indices
    // outermost-first, so a contiguous rank-4 tensor is [0,1,2,3]. That is the
    // inverse of extractStrideOrder's stride-rank vector, which BindingContext
    // converts once (RFC 0018 §7/A.8).
    const jlogic::Value strideOrder = b.get("$q.stride_order");
    ASSERT_TRUE(strideOrder.isArray());
    EXPECT_EQ(strideOrder,
              jlogic::Value(jlogic::Value::Array{jlogic::Value(std::int64_t{0}),
                                                 jlogic::Value(std::int64_t{1}),
                                                 jlogic::Value(std::int64_t{2}),
                                                 jlogic::Value(std::int64_t{3})}));

    // Attributes namespace.
    EXPECT_FALSE(b.get("$sdpa_fwd.dropout_probability.present").asBool());
    EXPECT_FALSE(b.get("$sdpa_fwd.alibi_mask").asBool());

    // Graph and device namespaces.
    EXPECT_EQ(b.get("$graph.node_count").asInt(), 1);
    EXPECT_EQ(b.get("$device.lds_size").asInt(), 65536);
    EXPECT_EQ(b.get("$device.warp_size").asInt(), 64);
}

TEST(TestBindingContext, FailsClosedOnBadPaths)
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

TEST(TestUmdCompiler, LayoutAliasMatchesContiguousLayout)
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

TEST(TestBindingContext, StrideOrderIsPublishedOutermostFirst)
{
    // The conversion is observable, not cosmetic: an NHWC tensor must publish
    // [0,2,3,1] (N,H,W,C read left to right), NOT extractStrideOrder's
    // [3,0,2,1] stride-rank vector. Reading the rank vector as if it were this
    // form would name the dims H,C,W,N -- a layout that does not exist.
    const std::vector<std::int64_t> dims{2, 8, 16, 128};
    // NHWC over an (n,c,h,w) logical dim order: c varies fastest.
    const std::vector<std::int64_t> nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims, {3, 0, 2, 1});

    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(dims,
                                                                       nhwcStrides,
                                                                       dims,
                                                                       nhwcStrides,
                                                                       dims,
                                                                       nhwcStrides,
                                                                       dims,
                                                                       nhwcStrides,
                                                                       data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$q.stride_order", [0, 2, 3, 1]]})"));
    const umd::UniversalGraphMatcher m(d);

    const umd::MatchResult r = m.match(K_DEVICE, g);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.bindings.get("$q.stride_order"),
              jlogic::Value(jlogic::Value::Array{jlogic::Value(std::int64_t{0}),
                                                 jlogic::Value(std::int64_t{2}),
                                                 jlogic::Value(std::int64_t{3}),
                                                 jlogic::Value(std::int64_t{1})}));

    // The `nhwc` alias expands to the same array, so it matches the same graph.
    json aliased = sdpaDescriptor();
    aliased["criteria"]["and"].push_back(json::parse(R"({"==": ["$q.stride_order", "nhwc"]})"));
    EXPECT_TRUE(umd::UniversalGraphMatcher(aliased).match(K_DEVICE, g).matched);
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

TEST(TestUmdCompiler, AcceptsMultiNodeDescriptor)
{
    const umd::CompiledUmd c = umd::UmdCompiler::compile(twoNodeChainDescriptor());
    EXPECT_EQ(c.nodes.size(), 2u);
    ASSERT_NE(c.findNode("sdpa1"), nullptr);
    ASSERT_NE(c.findNode("sdpa2"), nullptr);
    // The `$mid` edge variable is shared by both nodes (result of one, operand
    // of the other) but declared once.
    ASSERT_NE(c.findTvar("mid"), nullptr);
}

TEST(TestUmdCompiler, RejectsDuplicateNodeId)
{
    json d = twoNodeChainDescriptor();
    d["nodes"][1]["id"] = "sdpa1"; // collides with node 0
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsVariableProducedTwice)
{
    json d = twoNodeChainDescriptor();
    d["nodes"][1]["results"]["o"] = "$mid"; // $mid produced by both nodes
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUniversalGraphMatcher, MatchesTwoNodeChainAndBindsEachNode)
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

TEST(TestUniversalGraphMatcher, DeclinesWhenEdgeIsInconsistent)
{
    const umd::UniversalGraphMatcher m(twoNodeChainDescriptor());
    // node2 reads an independent Q, so no tensor satisfies the shared `$mid`
    // edge (node1.o != node2.q) -> no consistent assignment -> decline.
    auto builder = buildTwoNodeSdpaGraph(/*chained=*/false);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, PerNodeAttributesAreDistinct)
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

TEST(TestUmdCompiler, AcceptsKernelReference)
{
    // A criterion over a $kernel field compiles (DYNAMIC wildcard) and the
    // field name it reads is recorded.
    umd::CompiledUmd c;
    ASSERT_NO_THROW(c = umd::UmdCompiler::compile(sdpaKernelDescriptor()));
    EXPECT_EQ(c.kernelFields, (std::set<std::string>{"head_dim"}));
    EXPECT_TRUE(c.boundSymbols.count("kernel.head_dim") == 1);
}

TEST(TestUmdCompiler, StockDescriptorDoesNotReferenceKernel)
{
    const umd::CompiledUmd c = umd::UmdCompiler::compile(sdpaDescriptor());
    EXPECT_TRUE(c.kernelFields.empty());
}

TEST(TestUmdCompiler, RecordsEveryDistinctKernelFieldRead)
{
    // The memoization key is the set of `$kernel.*` fields the matcher reads,
    // without the `kernel.` prefix; a field read twice contributes once, and a
    // subscripted path contributes its subscript form (RFC 0017 §5).
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$kernel.tile_m", 64]})"));
    d["criteria"]["and"].push_back(json::parse(R"({"<": ["$kernel.tile_m", "$kernel.tile_n"]})"));
    const umd::CompiledUmd c = umd::UmdCompiler::compile(d);
    EXPECT_EQ(c.kernelFields, (std::set<std::string>{"tile_m", "tile_n"}));
}

TEST(TestUmdCompiler, KernelFieldUnifiesWithAnyScalarDomain)
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

TEST(TestUmdCompiler, KernelReferenceNeedsAField)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$kernel", 1]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, KernelFieldAsIfCondition)
{
    // A DYNAMIC $kernel field is accepted in an `if` condition, consistent with
    // its acceptance in and/or/! boolean positions.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": [{"if": ["$kernel.causal", 1, 0]}, 0]})"));
    EXPECT_NO_THROW(umd::UmdCompiler::compile(d));
}

TEST(TestUmdCompiler, KernelFieldDoesNotUnifyWithArray)
{
    // DYNAMIC is a scalar wildcard: comparing a $kernel field to an array
    // literal is a static type error, not a runtime-deferred check.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$kernel.tiles", [1, 2, 3]]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, KernelIsReservedAsPatternVariable)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["operands"]["q"] = "$kernel"; // bind a tensor var to the reserved root
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, KernelIsReservedAsNodeId)
{
    json d = sdpaDescriptor();
    d["nodes"][0]["id"] = "kernel";
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUniversalGraphMatcher, ReportsKernelMetadataReference)
{
    const umd::UniversalGraphMatcher mk(sdpaKernelDescriptor());
    EXPECT_TRUE(mk.referencesKernelMetadata());
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    EXPECT_FALSE(m.referencesKernelMetadata());
}

TEST(TestUniversalGraphMatcher, PublishesKernelFieldsItReads)
{
    // The set is the loader's KMD existence check and the per-kernel
    // memoization key (RFC 0017 §5): exactly the names read, prefix stripped.
    const umd::UniversalGraphMatcher mk(sdpaKernelDescriptor());
    EXPECT_EQ(mk.kernelFields(), (std::set<std::string>{"head_dim"}));
    // A graph-only matcher reads none, which is what makes the two-argument
    // match() overload legal for it.
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    EXPECT_TRUE(m.kernelFields().empty());
}

TEST(TestUniversalGraphMatcher, PublishesBoundSymbols)
{
    // A pack's dispatch descriptor is checked against the fields its matchers
    // bind, so the set must be reachable from the matcher (RFC 0017 §6).
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    EXPECT_EQ(m.boundSymbols(), m.descriptor().boundSymbols);
    EXPECT_TRUE(m.boundSymbols().count("q.head_size") == 1);
}

TEST(TestUniversalGraphMatcher, TwoArgMatchThrowsWhenDescriptorNeedsKernel)
{
    const umd::UniversalGraphMatcher m(sdpaKernelDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    // Wrong overload for a kernel-referencing UMD -> programming error.
    EXPECT_THROW(m.match(K_DEVICE, g), std::logic_error);
}

TEST(TestUniversalGraphMatcher, MatchesWithSuppliedKernelMetadata)
{
    const umd::UniversalGraphMatcher m(sdpaKernelDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    // head_dim matches $q.head_size (128) -> match; the value is queryable.
    const umd::MatchResult r = m.match(K_DEVICE, g, json{{"head_dim", 128}});
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.bindings.get("$kernel.head_dim").asInt(), 128);
}

TEST(TestUniversalGraphMatcher, DeclinesWhenKernelMetadataMismatches)
{
    const umd::UniversalGraphMatcher m(sdpaKernelDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    // head_dim != $q.head_size (128) -> criterion false -> decline.
    EXPECT_FALSE(m.match(K_DEVICE, g, json{{"head_dim", 64}}).matched);
}

TEST(TestUniversalGraphMatcher, EmptyKernelMetadataDeclinesComparison)
{
    // Defensive edge case: production metadata is fully resolved so every
    // referenced field is present, but an unbound/empty document must not
    // throw -- $kernel.head_dim reads null -> comparison false -> decline.
    const umd::UniversalGraphMatcher m(sdpaKernelDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_NO_THROW({ EXPECT_FALSE(m.match(K_DEVICE, g, json::object()).matched); });
}

// ---- Descriptor versions (RFC 0017 §4 / A.10 §1) -------------------------

TEST(TestUmdCompiler, AcceptsAndPublishesWellFormedVersions)
{
    json d = sdpaDescriptor();
    d["version"] = "1.0";
    d["sdk_version"] = "1.2";
    umd::CompiledUmd c;
    ASSERT_NO_THROW(c = umd::UmdCompiler::compile(d));
    EXPECT_EQ(c.version, "1.0");
    EXPECT_EQ(c.sdkVersion, "1.2");
}

TEST(TestUmdCompiler, VersionKeysAreOptionalAndDefaultToOneZero)
{
    // A descriptor omitting both still compiles, and both read "1.0": the
    // version a descriptor authored before either key existed implies
    // (RFC 0018 A.1). The published value is never empty, so the per-graph SDK
    // floor always has a concrete version to compare against.
    const umd::CompiledUmd c = umd::UmdCompiler::compile(sdpaDescriptor());
    EXPECT_EQ(c.version, "1.0");
    EXPECT_EQ(c.sdkVersion, "1.0");
}

TEST(TestUmdCompiler, RejectsMalformedVersion)
{
    for(const char* bad : {"1", "1.0.0", "1.", ".0", "+1.0", "-1.0", "1.x", "one.zero", ""})
    {
        json d = sdpaDescriptor();
        d["version"] = bad;
        EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError) << bad;
        json e = sdpaDescriptor();
        e["sdk_version"] = bad;
        EXPECT_THROW(umd::UmdCompiler::compile(e), umd::UmdCompileError) << bad;
    }
}

TEST(TestUmdCompiler, RejectsNonStringVersion)
{
    json d = sdpaDescriptor();
    d["version"] = 1.0;
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUmdCompiler, RejectsVersionNewerThanTheRuntime)
{
    // RFC 0017 §4: a minor newer than the runtime's carries features this
    // runtime cannot understand, so the descriptor is refused, never silently
    // reinterpreted.
    json newerMinor = sdpaDescriptor();
    newerMinor["version"] = "1.9";
    EXPECT_THROW(umd::UmdCompiler::compile(newerMinor), umd::UmdCompileError);

    // A differing major is refused in either direction.
    for(const char* major : {"0.9", "2.0"})
    {
        json d = sdpaDescriptor();
        d["version"] = major;
        EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError) << major;
    }

    // Same for the SDK version's ceiling half.
    json newerSdk = sdpaDescriptor();
    newerSdk["sdk_version"] = "1.99";
    EXPECT_THROW(umd::UmdCompiler::compile(newerSdk), umd::UmdCompileError);
}

TEST(TestUmdCompiler, AcceptsAnOlderMinorWithinTheSameMajor)
{
    // RFC 0017 §4: "A file stamped 1.0 loads on a 1.1 runtime." An older minor
    // is never refused, so a descriptor stays loadable on the oldest runtime
    // that can serve it. This is the case the previous floor-only gate broke.
    EXPECT_TRUE(umd::UmdCompiler::versionLoadableOn("1.0", "1.1"));
    EXPECT_TRUE(umd::UmdCompiler::versionLoadableOn("1.1", "1.1"));
    EXPECT_FALSE(umd::UmdCompiler::versionLoadableOn("1.2", "1.1"));
    EXPECT_FALSE(umd::UmdCompiler::versionLoadableOn("2.0", "1.1"));
    EXPECT_FALSE(umd::UmdCompiler::versionLoadableOn("0.9", "1.1"));
    // an unparseable version loads nowhere (fail closed)
    EXPECT_FALSE(umd::UmdCompiler::versionLoadableOn("1", "1.1"));

    // The SDK version's runtime is 1.2, so a matcher authored against the
    // older 1.0 and 1.1 schemas still compiles.
    for(const char* older : {"1.0", "1.1", "1.2"})
    {
        json d = sdpaDescriptor();
        d["sdk_version"] = older;
        EXPECT_NO_THROW(umd::UmdCompiler::compile(d)) << older;
    }
}

TEST(TestUmdCompiler, VersionComparisonIsNumericNotLexicographic)
{
    // "1.10" is above "1.9" numerically but below it as text; a lexicographic
    // comparison would order these backwards.
    EXPECT_TRUE(umd::UmdCompiler::versionAtLeast("1.10", "1.9"));
    EXPECT_FALSE(umd::UmdCompiler::versionAtLeast("1.9", "1.10"));
    EXPECT_FALSE(umd::UmdCompiler::versionAtLeast("2.0", "10.0"));
    EXPECT_TRUE(umd::UmdCompiler::versionAtLeast("10.0", "2.0"));
    // equal is at the floor, not below it
    EXPECT_TRUE(umd::UmdCompiler::versionAtLeast("1.0", "1.0"));
    // an unparseable version is below every floor (fail closed)
    EXPECT_FALSE(umd::UmdCompiler::versionAtLeast("1", "1.0"));
    // the ceiling comparison is numeric too: 1.10 is newer than a 1.9 runtime
    EXPECT_FALSE(umd::UmdCompiler::versionLoadableOn("1.10", "1.9"));
    EXPECT_TRUE(umd::UmdCompiler::versionLoadableOn("1.9", "1.10"));
}

// ---- Per-graph sdk_version floor (RFC 0017 §4) ---------------------------

namespace
{

// Rebuild an SDPA graph with `min_required_engine_api_version` stamped, the
// field hipDNN computes from the optional features a graph uses (RFC 0008
// override shapes at 1.1, RFC 0016 runtime pass-by-value at 1.2). The test-SDK
// fixture leaves it unset, so the floor is set explicitly here.
flatbuffers::FlatBufferBuilder buildSdpaGraphRequiringSchema(std::uint32_t major,
                                                             std::uint32_t minor)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<std::int64_t> dims{2, 8, 16, 128};
    const std::vector<std::int64_t> strides = contiguousStrides(dims);

    std::vector<::flatbuffers::Offset<data::TensorAttributes>> tensors;
    for(const auto& [uid, name] :
        {std::pair<std::int64_t, const char*>{1, "q"}, {2, "k"}, {3, "v"}, {4, "o"}})
    {
        tensors.push_back(data::CreateTensorAttributesDirect(
            builder, uid, name, data::DataType::BFLOAT16, &strides, &dims));
    }

    data::SdpaAttributesBuilder sb(builder);
    sb.add_q_tensor_uid(1);
    sb.add_k_tensor_uid(2);
    sb.add_v_tensor_uid(3);
    sb.add_o_tensor_uid(4);
    const auto sdpa = sb.Finish();

    std::vector<::flatbuffers::Offset<data::Node>> nodes;
    nodes.push_back(data::CreateNodeDirect(builder,
                                           "sdpa_fwd",
                                           data::DataType::BFLOAT16,
                                           data::NodeAttributes::SdpaAttributes,
                                           sdpa.Union()));

    const data::EngineApiVersion required(major, minor, 0);
    auto graph = data::CreateGraphDirect(builder,
                                         "test",
                                         data::DataType::FLOAT,
                                         data::DataType::HALF,
                                         data::DataType::BFLOAT16,
                                         &tensors,
                                         &nodes,
                                         flatbuffers::nullopt,
                                         /*is_override_shape_enabled=*/false,
                                         &required);
    builder.Finish(graph);
    return builder;
}

} // namespace

TEST(TestUniversalGraphMatcher, DeclinesAMatcherBelowTheGraphsRequiredSchema)
{
    // RFC 0017 §4: a graph reports the schema version its own contents require,
    // and a matcher declaring less is skipped instead of asked -- it would
    // otherwise match on the fields it knows and silently ignore a field that
    // changes what the graph means. The descriptor is otherwise a full match.
    json d = sdpaDescriptor();
    d["sdk_version"] = "1.0";
    const umd::UniversalGraphMatcher m(d);

    auto baseline = buildSdpaGraphRequiringSchema(1, 0);
    fbu::GraphWrapper const gBaseline(baseline.GetBufferPointer(), baseline.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, gBaseline).matched);

    // The graph sets an optional feature this matcher never accounted for.
    auto newer = buildSdpaGraphRequiringSchema(1, 2);
    fbu::GraphWrapper const gNewer(newer.GetBufferPointer(), newer.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, gNewer).matched);
}

TEST(TestUniversalGraphMatcher, MatcherAtOrAboveTheGraphsRequiredSchemaRuns)
{
    // The same graph, matched by a descriptor that adopted the newer schema.
    json d = sdpaDescriptor();
    d["sdk_version"] = "1.2";
    const umd::UniversalGraphMatcher m(d);

    auto newer = buildSdpaGraphRequiringSchema(1, 2);
    fbu::GraphWrapper const gNewer(newer.GetBufferPointer(), newer.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, gNewer).matched);

    // A matcher above the floor still serves a graph that requires less.
    auto baseline = buildSdpaGraphRequiringSchema(1, 0);
    fbu::GraphWrapper const gBaseline(baseline.GetBufferPointer(), baseline.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, gBaseline).matched);
}

TEST(TestUniversalGraphMatcher, AnUnstampedGraphReadsAsTheBaselineFloor)
{
    // A hand-built fixture or a graph written before the field existed carries
    // no stamp; it reads as the 1.0 baseline rather than declining every
    // matcher, mirroring the plugin SDK's null-tolerant accessor.
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, g).matched);
}

// ---- present / not_present (RFC 0017 §5) ---------------------------------

namespace
{

// The SDPA descriptor with its `.present` field gates on the three optional
// operands replaced by the `not_present` operator over the same list.
json sdpaNotPresentDescriptor()
{
    json d = sdpaDescriptor();
    json& crit = d["criteria"]["and"];
    // Drop the three trailing `{"!": "$x.present"}` field-form gates.
    crit.erase(crit.size() - 1);
    crit.erase(crit.size() - 1);
    crit.erase(crit.size() - 1);
    crit.push_back(
        json::parse(R"({"not_present": ["$attn_mask", "$page_table_k", "$page_table_v"]})"));
    return d;
}

} // namespace

TEST(TestUmdCompiler, AcceptsPresenceOperators)
{
    EXPECT_NO_THROW(umd::UmdCompiler::compile(sdpaNotPresentDescriptor()));

    // `present`/`not_present` are the general form: unlike the `.present`
    // FIELD, they apply to a required operand too.
    for(const char* crit : {R"({"present": ["$q"]})",
                            R"({"not_present": ["$attn_mask"]})",
                            R"({"present": ["$q", "$k", "$v"]})",
                            R"({"or": [{"not_present": ["$attn_mask"]}, {"present": ["$q"]}]})"})
    {
        json d = sdpaDescriptor();
        d["criteria"]["and"].push_back(json::parse(crit));
        EXPECT_NO_THROW(umd::UmdCompiler::compile(d)) << crit;
    }
}

TEST(TestUmdCompiler, RejectsPresenceOperatorOnNonReference)
{
    for(const char* bad : {R"({"present": [4]})",
                           R"({"present": ["BFLOAT16"]})",
                           R"({"not_present": [{"+": ["$q.rank", 1]}]})",
                           R"({"present": ["$q", 4]})",
                           R"({"present": []})"})
    {
        json d = sdpaDescriptor();
        d["criteria"]["and"].push_back(json::parse(bad));
        EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError) << bad;
    }
}

TEST(TestUmdCompiler, RejectsPresenceOperatorOnUnresolvedReference)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"present": ["$not_a_tensor"]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestUniversalGraphMatcher, NotPresentOperatorMatchesWhenOptionalsAbsent)
{
    const umd::UniversalGraphMatcher m(sdpaNotPresentDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, NotPresentOperatorDeclinesWhenAnyOptionalBound)
{
    // The n-ary form is an `and`-fold: one supplied operand out of the list
    // decides the whole call.
    const umd::UniversalGraphMatcher m(sdpaNotPresentDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16, /*withAttnMask=*/true);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, PresenceOperatorsEvaluateRatherThanDecline)
{
    // The key distinction from a bare field read: a field read on an absent
    // optional resolves null and declines, but `present`/`not_present` always
    // produce a boolean, so a criterion built only from them still matches.
    json d = sdpaDescriptor();
    json& crit = d["criteria"]["and"];
    crit.erase(crit.size() - 1);
    crit.erase(crit.size() - 1);
    crit.erase(crit.size() - 1);
    // `$attn_mask` is absent; `present` on it is false, so `!` of it is true.
    crit.push_back(json::parse(R"({"!": {"present": ["$attn_mask"]}})"));
    // `$q` is a required operand and always bound.
    crit.push_back(json::parse(R"({"present": ["$q", "$k", "$v", "$o"]})"));

    const umd::UniversalGraphMatcher m(d);
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, g).matched);
}

// ---- Absent optional operands neither pass nor fail (RFC 0017 §5) --------

namespace
{

// The SDPA descriptor with its three trailing `.present` field gates dropped
// and `extra` appended, so a single criterion can be exercised against a graph
// that supplies no optional operands.
json sdpaWithCriterion(const char* extra)
{
    json d = sdpaDescriptor();
    json& crit = d["criteria"]["and"];
    crit.erase(crit.size() - 1);
    crit.erase(crit.size() - 1);
    crit.erase(crit.size() - 1);
    crit.push_back(json::parse(extra));
    return d;
}

} // namespace

TEST(TestUniversalGraphMatcher, FieldReadOnAnAbsentOptionalNeverAccepts)
{
    // RFC 0017 §5: "A dtype or layout check on an absent $bias neither passes
    // nor fails, it simply does not run." The hazard is the negative and
    // ordering forms: if an unresolved read coerced to false/0 instead of
    // propagating, each of these would evaluate TRUE and the pack would accept
    // a graph carrying an operand its kernel cannot serve.
    for(const char* crit : {R"({"!": "$attn_mask.packed"})",
                            R"({"!=": ["$attn_mask.dtype", "BFLOAT16"]})",
                            R"({"<": ["$attn_mask.rank", 5]})",
                            R"({">=": ["$attn_mask.rank", 0]})",
                            R"({"==": ["$attn_mask.dtype", "$page_table_k.dtype"]})",
                            R"({"==": [{"+": ["$attn_mask.rank", 1]}, 1]})"})
    {
        const umd::UniversalGraphMatcher m(sdpaWithCriterion(crit));
        auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
        fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(m.match(K_DEVICE, g).matched) << crit;
    }
}

TEST(TestUniversalGraphMatcher, AbsentOrPresentAndConstrainedAcceptsBothShapes)
{
    // RFC 0017 §5: the pair a pack writes when it serves an optional operand
    // only in a particular form. The `or` must see past the unresolved second
    // arm when the operand is absent, and must still enforce it when present.
    const char* crit = R"({"or": [
        {"not_present": ["$attn_mask"]},
        {"and": [{"present": ["$attn_mask"]},
                 {"==": ["$attn_mask.dtype", "BFLOAT16"]}]}
    ]})";
    const umd::UniversalGraphMatcher m(sdpaWithCriterion(crit));

    auto absent = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const gAbsent(absent.GetBufferPointer(), absent.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, gAbsent).matched);

    auto present = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16, /*withAttnMask=*/true);
    fbu::GraphWrapper const gPresent(present.GetBufferPointer(), present.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, gPresent).matched);
}

TEST(TestUniversalGraphMatcher, DefiniteFalseStillDeclinesBesideAnUnresolvedCheck)
{
    // An unresolved sibling must not rescue a criterion that definitely fails:
    // `and` short-circuits on the definite false rather than going unresolved.
    const char* crit = R"({"and": [{"==": ["$q.rank", 9]},
                                   {"==": ["$attn_mask.dtype", "BFLOAT16"]}]})";
    const umd::UniversalGraphMatcher m(sdpaWithCriterion(crit));
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, PresentOperatorSeesABoundOptional)
{
    // Same descriptor, but the graph supplies `attn_mask`: `present` now reads
    // true and the criterion declines, proving it tracks binding rather than
    // returning a constant.
    json d = sdpaDescriptor();
    json& crit = d["criteria"]["and"];
    crit.erase(crit.size() - 1);
    crit.erase(crit.size() - 1);
    crit.erase(crit.size() - 1);
    crit.push_back(json::parse(R"({"!": {"present": ["$attn_mask"]}})"));

    const umd::UniversalGraphMatcher m(d);
    auto bound = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16, /*withAttnMask=*/true);
    fbu::GraphWrapper const gb(bound.GetBufferPointer(), bound.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, gb).matched);

    auto absent = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const ga(absent.GetBufferPointer(), absent.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, ga).matched);
}

// ---- value_or_default with an expression fallback (RFC 0017 §5) ----------

TEST(TestUmdCompiler, ValueOrDefaultAcceptsFieldReferenceFallback)
{
    // "this field, else that one": both arms are Int, so the pair unifies.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(
        json::parse(R"({"==": [{"value_or_default": ["$q.head_size", "$k.head_size"]}, 128]})"));
    EXPECT_NO_THROW(umd::UmdCompiler::compile(d));
}

TEST(TestUmdCompiler, ValueOrDefaultAcceptsDynamicArm)
{
    // A `$kernel.*` arm is DYNAMIC, so the other arm supplies the result kind
    // and the comparison against an Int still type-checks.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(
        json::parse(R"({"==": [{"value_or_default": ["$kernel.tile_m", 64]}, 128]})"));
    EXPECT_NO_THROW(umd::UmdCompiler::compile(d));
    json e = sdpaDescriptor();
    e["criteria"]["and"].push_back(
        json::parse(R"({"==": [{"value_or_default": ["$q.head_size", "$kernel.tile_m"]}, 128]})"));
    EXPECT_NO_THROW(umd::UmdCompiler::compile(e));
}

TEST(TestUmdCompiler, RejectsTypeIncompatibleValueOrDefaultArms)
{
    for(const char* bad :
        {R"({"==": [{"value_or_default": ["$q.head_size", "BFLOAT16"]}, 128]})",
         R"({"==": [{"value_or_default": ["$q.dtype", 4]}, "BFLOAT16"]})",
         R"({"==": [{"value_or_default": ["$q.head_size", "$q.stride_order"]}, 128]})"})
    {
        json d = sdpaDescriptor();
        d["criteria"]["and"].push_back(json::parse(bad));
        EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError) << bad;
    }
}

TEST(TestUniversalGraphMatcher, ValueOrDefaultFallsBackToTheSecondField)
{
    // `$sdpa_fwd.dropout_probability` is an unset optional attribute, so the
    // fallback expression is evaluated and its value is what the comparison
    // sees.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(
        R"({"==": [{"value_or_default": ["$sdpa_fwd.dropout_probability", "$q.head_size"]}, 128]})"));
    const umd::UniversalGraphMatcher m(d);
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, g).matched);
}

// ---- $graph.is_override_shape_enabled (RFC 0017 §5) ----------------------

TEST(TestUmdCompiler, AcceptsGraphOverrideShapeFlag)
{
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"!": "$graph.is_override_shape_enabled"})"));
    umd::CompiledUmd c;
    ASSERT_NO_THROW(c = umd::UmdCompiler::compile(d));
    EXPECT_TRUE(c.boundSymbols.count("graph.is_override_shape_enabled") == 1);
}

TEST(TestUmdCompiler, RejectsGraphOverrideShapeFlagAgainstNonBool)
{
    // The flag is Bool; comparing it to an int is a static type error.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(
        json::parse(R"({"==": ["$graph.is_override_shape_enabled", 1]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);
}

TEST(TestBindingContext, ResolvesGraphOverrideShapeFlag)
{
    // The descriptor key `allow_override_shape` is the MATCHER's opt-in; the
    // `$graph.is_override_shape_enabled` token is the GRAPH's state. A matcher
    // that opts in still sees the graph's flag.
    json d = sdpaDescriptor();
    d["allow_override_shape"] = true;
    const umd::UniversalGraphMatcher m(d);

    auto plain = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const gp(plain.GetBufferPointer(), plain.GetSize());
    const umd::MatchResult rp = m.match(K_DEVICE, gp);
    ASSERT_TRUE(rp.matched);
    EXPECT_TRUE(rp.bindings.get("$graph.is_override_shape_enabled").isBool());
    EXPECT_FALSE(rp.bindings.get("$graph.is_override_shape_enabled").asBool());

    auto overriden = buildSdpaGraph({2, 8, 16, 128},
                                    data::DataType::BFLOAT16,
                                    /*withAttnMask=*/false,
                                    /*overrideShape=*/true);
    fbu::GraphWrapper const go(overriden.GetBufferPointer(), overriden.GetSize());
    const umd::MatchResult ro = m.match(K_DEVICE, go);
    ASSERT_TRUE(ro.matched);
    EXPECT_TRUE(ro.bindings.get("$graph.is_override_shape_enabled").asBool());
}

TEST(TestUniversalGraphMatcher, DeclinesOnGraphOverrideShapeFlagCriterion)
{
    // An opted-in matcher can still refuse an override-shape graph by reading
    // the graph's own flag.
    json d = sdpaDescriptor();
    d["allow_override_shape"] = true;
    d["criteria"]["and"].push_back(json::parse(R"({"!": "$graph.is_override_shape_enabled"})"));
    const umd::UniversalGraphMatcher m(d);

    auto overriden = buildSdpaGraph({2, 8, 16, 128},
                                    data::DataType::BFLOAT16,
                                    /*withAttnMask=*/false,
                                    /*overrideShape=*/true);
    fbu::GraphWrapper const go(overriden.GetBufferPointer(), overriden.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, go).matched);

    auto plain = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const gp(plain.GetBufferPointer(), plain.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, gp).matched);
}

// ---- $q.is_runtime_pass_by_value / $q.value_f32 (RFC 0017 §5) ------------

namespace
{

// The SDPA descriptor with the optional `scale` operand bound, so the tensor
// carrying a compile-time value is reachable as `$scale`.
json sdpaScaleDescriptor()
{
    json d = sdpaDescriptor();
    d["nodes"][0]["operands"]["scale"] = "$scale?";
    return d;
}

} // namespace

TEST(TestUmdCompiler, AcceptsTensorValueFields)
{
    json d = sdpaScaleDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"!": "$scale.is_runtime_pass_by_value"})"));
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$scale.value_f32", 1.0]})"));
    EXPECT_NO_THROW(umd::UmdCompiler::compile(d));
}

TEST(TestUmdCompiler, TensorValueFieldsAreTyped)
{
    // is_runtime_pass_by_value is Bool and value_f32 is Float; using either in
    // the wrong domain is a static type error.
    json d = sdpaScaleDescriptor();
    d["criteria"]["and"].push_back(
        json::parse(R"({"==": ["$scale.is_runtime_pass_by_value", 1]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError);

    json e = sdpaScaleDescriptor();
    e["criteria"]["and"].push_back(json::parse(R"({"==": ["$scale.value_f32", "BFLOAT16"]})"));
    EXPECT_THROW(umd::UmdCompiler::compile(e), umd::UmdCompileError);

    json f = sdpaScaleDescriptor();
    f["criteria"]["and"].push_back(json::parse(R"({"<": ["$scale.value_f32", 2.0]})"));
    EXPECT_NO_THROW(umd::UmdCompiler::compile(f));
}

TEST(TestUmdCompiler, RejectsShapeDimNameCollidingWithTensorValueFields)
{
    // A.10 §7: a `shape` short-hand may not name a dim after a reserved field.
    for(const char* name : {"is_runtime_pass_by_value", "value_f32"})
    {
        json d = sdpaDescriptor();
        d["criteria"]["and"].push_back(
            json{{"shape", json::array({"$o", json::array({"b", "h", "s", name})})}});
        EXPECT_THROW(umd::UmdCompiler::compile(d), umd::UmdCompileError) << name;
    }
}

TEST(TestBindingContext, ResolvesTensorValueFields)
{
    const umd::UniversalGraphMatcher m(sdpaScaleDescriptor());
    auto builder = buildSdpaGraph({2, 8, 16, 128},
                                  data::DataType::BFLOAT16,
                                  /*withAttnMask=*/false,
                                  /*overrideShape=*/false,
                                  /*withScale=*/true);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    const umd::MatchResult r = m.match(K_DEVICE, g);
    ASSERT_TRUE(r.matched);
    const umd::BindingContext& b = r.bindings;

    // The scale tensor carries a compile-time Float32 value of 1.0, baked into
    // the graph rather than supplied per execution.
    EXPECT_FALSE(b.get("$scale.is_runtime_pass_by_value").asBool());
    EXPECT_DOUBLE_EQ(b.get("$scale.value_f32").toNumber(), 1.0);

    // A regular data tensor carries neither.
    EXPECT_FALSE(b.get("$q.is_runtime_pass_by_value").asBool());
    EXPECT_TRUE(b.get("$q.value_f32").isNull());
}

TEST(TestBindingContext, ValueF32DeclinesWithoutACompileTimeValue)
{
    // "present only when the tensor carries a compile-time value at all": the
    // union is NONE, so the token reads null and a criterion over it declines
    // rather than seeing a zero that would satisfy a comparison.
    const umd::UniversalGraphMatcher m = makeSdpaMatcher();
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());

    const umd::MatchResult r = m.match(K_DEVICE, g);
    ASSERT_TRUE(r.matched);
    EXPECT_TRUE(r.bindings.get("$q.value_f32").isNull());
    EXPECT_FALSE(r.bindings.get("$q.value_f32").isNumber());
}

TEST(TestUniversalGraphMatcher, DeclinesOnValueF32OfAValuelessTensor)
{
    // The fail-closed consequence: a criterion reading value_f32 on a tensor
    // with no compile-time value declines the whole match.
    json d = sdpaDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$q.value_f32", 0.0]})"));
    const umd::UniversalGraphMatcher m(d);
    auto builder = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, g).matched);
}

TEST(TestUniversalGraphMatcher, MatchesOnACompileTimeScaleValue)
{
    json d = sdpaScaleDescriptor();
    d["criteria"]["and"].push_back(json::parse(R"({"==": ["$scale.value_f32", 1.0]})"));
    const umd::UniversalGraphMatcher m(d);

    auto withScale = buildSdpaGraph({2, 8, 16, 128},
                                    data::DataType::BFLOAT16,
                                    /*withAttnMask=*/false,
                                    /*overrideShape=*/false,
                                    /*withScale=*/true);
    fbu::GraphWrapper const gs(withScale.GetBufferPointer(), withScale.GetSize());
    EXPECT_TRUE(m.match(K_DEVICE, gs).matched);

    // No scale operand at all -> the field read on an absent optional declines.
    auto noScale = buildSdpaGraph({2, 8, 16, 128}, data::DataType::BFLOAT16);
    fbu::GraphWrapper const gn(noScale.GetBufferPointer(), noScale.GetSize());
    EXPECT_FALSE(m.match(K_DEVICE, gn).matched);
}

// ---- value_f32 union coercion (RFC 0017 §5) ------------------------------

namespace
{

// A single-node SDPA graph whose optional `scale` operand carries `value`, the
// tagged union arm under test, with the tensor's own data_type. The test-SDK
// fixture only ever builds the Float32 arm, so the other seven (and the
// data_type-dispatched Float8 decode) are unreachable through it.
template <typename AddValue>
flatbuffers::FlatBufferBuilder buildSdpaGraphWithScaleValue(data::DataType scaleType,
                                                            AddValue&& addValue)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<std::int64_t> dims{2, 8, 16, 128};
    const std::vector<std::int64_t> strides = contiguousStrides(dims);
    const std::vector<std::int64_t> scalarDims{1};
    const std::vector<std::int64_t> scalarStrides{1};

    std::vector<::flatbuffers::Offset<data::TensorAttributes>> tensors;
    for(const auto& [uid, name] :
        {std::pair<std::int64_t, const char*>{1, "q"}, {2, "k"}, {3, "v"}, {4, "o"}})
    {
        tensors.push_back(data::CreateTensorAttributesDirect(
            builder, uid, name, data::DataType::BFLOAT16, &strides, &dims));
    }

    // addValue writes the union arm and returns its (type, offset) pair.
    const auto [valueType, valueOffset] = addValue(builder);
    const auto scaleName = builder.CreateString("scale");
    const auto scaleDims = builder.CreateVector(scalarDims);
    const auto scaleStrides = builder.CreateVector(scalarStrides);
    data::TensorAttributesBuilder tb(builder);
    tb.add_uid(5);
    tb.add_name(scaleName);
    tb.add_data_type(scaleType);
    tb.add_dims(scaleDims);
    tb.add_strides(scaleStrides);
    tb.add_value_type(valueType);
    tb.add_value(valueOffset);
    tensors.push_back(tb.Finish());

    data::SdpaAttributesBuilder sb(builder);
    sb.add_q_tensor_uid(1);
    sb.add_k_tensor_uid(2);
    sb.add_v_tensor_uid(3);
    sb.add_o_tensor_uid(4);
    sb.add_scale_tensor_uid(5);
    const auto sdpa = sb.Finish();

    std::vector<::flatbuffers::Offset<data::Node>> nodes;
    nodes.push_back(data::CreateNodeDirect(builder,
                                           "sdpa_fwd",
                                           data::DataType::BFLOAT16,
                                           data::NodeAttributes::SdpaAttributes,
                                           sdpa.Union()));

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

// Resolve `$scale.value_f32` for a graph carrying one union arm.
jlogic::Value scaleValueF32(flatbuffers::FlatBufferBuilder& builder)
{
    const umd::UniversalGraphMatcher m(sdpaScaleDescriptor());
    fbu::GraphWrapper const g(builder.GetBufferPointer(), builder.GetSize());
    const umd::MatchResult r = m.match(K_DEVICE, g);
    EXPECT_TRUE(r.matched);
    return r.bindings.get("$scale.value_f32");
}

} // namespace

TEST(TestBindingContext, ValueF32CoercesEveryUnionArm)
{
    // RFC 0017 §5: the schema layer "coerces whichever arm is set to f32" and
    // publishes one typed token. Every arm must produce a number, not null.
    const auto f32 = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::FLOAT, [](auto& b) {
            return std::pair{
                data::TensorValue::Float32Value,
                ::flatbuffers::Offset<void>(b.CreateStruct(data::Float32Value(2.5F)).o)};
        })));
    EXPECT_DOUBLE_EQ(f32.toNumber(), 2.5);

    const auto f64 = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::DOUBLE, [](auto& b) {
            return std::pair{
                data::TensorValue::Float64Value,
                ::flatbuffers::Offset<void>(b.CreateStruct(data::Float64Value(0.25)).o)};
        })));
    EXPECT_DOUBLE_EQ(f64.toNumber(), 0.25);

    const auto i32 = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::INT32, [](auto& b) {
            return std::pair{data::TensorValue::Int32Value,
                             ::flatbuffers::Offset<void>(b.CreateStruct(data::Int32Value(-7)).o)};
        })));
    EXPECT_DOUBLE_EQ(i32.toNumber(), -7.0);

    const auto i64 = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::INT64, [](auto& b) {
            return std::pair{data::TensorValue::Int64Value,
                             ::flatbuffers::Offset<void>(b.CreateStruct(data::Int64Value(42)).o)};
        })));
    EXPECT_DOUBLE_EQ(i64.toNumber(), 42.0);

    const auto boolean = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::BOOLEAN, [](auto& b) {
            return std::pair{data::TensorValue::BoolValue,
                             ::flatbuffers::Offset<void>(b.CreateStruct(data::BoolValue(true)).o)};
        })));
    EXPECT_DOUBLE_EQ(boolean.toNumber(), 1.0);
}

TEST(TestBindingContext, ValueF32DecodesFloat8ThroughTheTensorDataType)
{
    // Float8Value stores raw bits; the tensor's data_type says which 8-bit
    // format they encode. The bits 0x38 are 1.0 in E4M3 (exponent bias 7) but
    // 0.5 in E5M2 (bias 15), so decoding through the wrong format -- or not
    // decoding at all -- is directly observable.
    const auto e4m3 = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::FP8_E4M3, [](auto& b) {
            return std::pair{
                data::TensorValue::Float8Value,
                ::flatbuffers::Offset<void>(b.CreateStruct(data::Float8Value(0x38)).o)};
        })));
    EXPECT_DOUBLE_EQ(e4m3.toNumber(), 1.0);

    const auto e5m2 = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::FP8_E5M2, [](auto& b) {
            return std::pair{
                data::TensorValue::Float8Value,
                ::flatbuffers::Offset<void>(b.CreateStruct(data::Float8Value(0x38)).o)};
        })));
    EXPECT_DOUBLE_EQ(e5m2.toNumber(), 0.5);

    // E8M0 is an 8-bit format too: bits 127 encode the 2^0 scale. It must
    // decode rather than fall through to the fail-closed null.
    const auto e8m0 = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::FP8_E8M0, [](auto& b) {
            return std::pair{data::TensorValue::Float8Value,
                             ::flatbuffers::Offset<void>(b.CreateStruct(data::Float8Value(127)).o)};
        })));
    ASSERT_FALSE(e8m0.isNull());
    EXPECT_DOUBLE_EQ(e8m0.toNumber(), 1.0);

    // INT8 raw bits are signed: 0xFF is -1, not 255.
    const auto i8 = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::INT8, [](auto& b) {
            return std::pair{
                data::TensorValue::Float8Value,
                ::flatbuffers::Offset<void>(b.CreateStruct(data::Float8Value(0xFF)).o)};
        })));
    EXPECT_DOUBLE_EQ(i8.toNumber(), -1.0);

    // UINT8 keeps the unsigned reading of the same bits.
    const auto u8 = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::UINT8, [](auto& b) {
            return std::pair{
                data::TensorValue::Float8Value,
                ::flatbuffers::Offset<void>(b.CreateStruct(data::Float8Value(0xFF)).o)};
        })));
    EXPECT_DOUBLE_EQ(u8.toNumber(), 255.0);

    // A pairing that names no 8-bit format resolves to null rather than
    // reinterpreting the bits.
    const auto mismatched = scaleValueF32(*std::make_unique<flatbuffers::FlatBufferBuilder>(
        buildSdpaGraphWithScaleValue(data::DataType::FLOAT, [](auto& b) {
            return std::pair{
                data::TensorValue::Float8Value,
                ::flatbuffers::Offset<void>(b.CreateStruct(data::Float8Value(0x38)).o)};
        })));
    EXPECT_TRUE(mismatched.isNull());
}
