// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Host-only decode/bindings test for the *universal* SDPA forward adapter. Unlike
// TestSdpaNumericParity (which drives the substrate on a gfx1151 GPU and builds
// LaunchBindings by hand, so it never touches SdpaAdapter), this exercises the
// adapter itself: it constructs real single-node SdpaAttributes flatbuffer graphs,
// wraps them as an IGraph (GraphWrapper), and calls decode()/buildBindings()/
// gridSymbols() directly. No GPU or catalog .co is required.
//
// It locks the generalization plan's four contracts:
//   (a) decode publishes the full capability vocabulary (shape + feature facts);
//   (b) for the legacy gfx1151 shape the emitted bindings still carry the exact
//       15 values the shipped kernel's args_signature names (byte-for-byte the
//       hand-built bag in TestSdpaNumericParity);
//   (c) genuinely malformed graphs (non-rank-4, dtype mismatch, non-integer GQA
//       ratio) still DECLINE in C++ (the universal safety gates);
//   (d) causal / GQA / masked / non-foldable-batch / non-contiguous-D graphs now
//       DECODE (publish facts) instead of declining -- proving the applicability
//       boundary moved from adapter C++ to per-kernel family.json data.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "catalog/CatalogTypes.hpp"
#include "ops/SdpaAdapter.hpp"

namespace
{

using namespace aot_catalog_engine;
namespace fb = flatbuffers;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

// Row-major (contiguous) strides for a BHSD tensor. `innerMult` scales the whole
// stride vector so the innermost (D) stride becomes `innerMult`; innerMult==2
// yields a non-D-contiguous layout (stride[3] != 1) with a consistent shape.
std::vector<int64_t> contiguousStrides(const std::vector<int64_t>& dims, int64_t innerMult = 1)
{
    std::vector<int64_t> strides(dims.size(), 1);
    int64_t acc = innerMult;
    for(int i = static_cast<int>(dims.size()) - 1; i >= 0; --i)
    {
        strides[static_cast<size_t>(i)] = acc;
        acc *= dims[static_cast<size_t>(i)];
    }
    return strides;
}

// Declarative description of a single-node SDPA graph. Defaults describe the
// legacy gfx1151 problem (f16, B=1, MHA H=32, D=64, contiguous, non-causal,
// unmasked) so a test tweaks only the field it cares about.
struct SdpaSpec
{
    DataType dtype = DataType::HALF;
    std::optional<DataType> kDtype; // set != dtype to force a dtype-mismatch decline
    int64_t b = 1;
    int64_t h = 32;
    int64_t hkv = 32;
    int64_t sq = 32;
    int64_t skv = 48;
    int64_t d = 64;
    int qRank = 4; // set to 3 to force a non-rank-4 decline
    int64_t innerMult = 1; // 2 -> non-contiguous D
    bool causal = false;
    bool causalBottomRight = false;
    bool alibi = false;
    bool padding = false;
    bool attnMask = false;
    bool blockMask = false;
    bool sink = false;
    bool genStats = false;
    bool paged = false;
    bool varlen = false;
    bool runtimeScale = false;
    bool fp8Descale = false; // adds descale_q/k/v/s + scale_s/o operand tensors
    std::optional<float> attnScaleValue; // baked plan-time scale (vs. 1/sqrt(D))
};

// Owns the serialized buffer and hands out an IGraph view over it (the wrapper
// keeps only a shallow pointer, so the buffer must outlive every graphWrapper()).
struct BuiltGraph
{
    std::shared_ptr<fb::DetachedBuffer> buffer;
    GraphWrapper graph() const
    {
        return GraphWrapper(buffer->data(), buffer->size());
    }
};

BuiltGraph buildSdpaGraph(const SdpaSpec& spec)
{
    fb::FlatBufferBuilder builder;
    std::vector<fb::Offset<data_objects::TensorAttributes>> tensors;

    const DataType kType = spec.kDtype.value_or(spec.dtype);

    // Q rank is configurable (rank-3 exercises the safety gate); K/V/O stay rank 4.
    std::vector<int64_t> qDims;
    if(spec.qRank == 4)
    {
        qDims = {spec.b, spec.h, spec.sq, spec.d};
    }
    else
    {
        qDims = {spec.h, spec.sq, spec.d};
    }
    const std::vector<int64_t> kDims = {spec.b, spec.hkv, spec.skv, spec.d};
    const std::vector<int64_t> vDims = {spec.b, spec.hkv, spec.skv, spec.d};
    const std::vector<int64_t> oDims = {spec.b, spec.h, spec.sq, spec.d};

    const std::vector<int64_t> qStrides = contiguousStrides(qDims, spec.innerMult);
    const std::vector<int64_t> kStrides = contiguousStrides(kDims, spec.innerMult);
    const std::vector<int64_t> vStrides = contiguousStrides(vDims, spec.innerMult);
    const std::vector<int64_t> oStrides = contiguousStrides(oDims, spec.innerMult);

    const int64_t qUid = 1;
    const int64_t kUid = 2;
    const int64_t vUid = 3;
    const int64_t oUid = 4;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, qUid, "q", spec.dtype, &qStrides, &qDims));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder, kUid, "k", kType, &kStrides, &kDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, vUid, "v", spec.dtype, &vStrides, &vDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, oUid, "o", spec.dtype, &oStrides, &oDims));

    // Optional operand tensors. Their contents are irrelevant to decode (which
    // only checks uid presence), so a rank-1 placeholder keeps them well-formed.
    const std::vector<int64_t> scalarDims = {1};
    const std::vector<int64_t> scalarStrides = {1};
    int64_t uid = 5;
    auto addPlaceholder = [&](const char* name, DataType type) {
        const int64_t id = uid++;
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, id, name, type, &scalarStrides, &scalarDims));
        return id;
    };

    fb::Optional<int64_t> attnMaskUid = fb::nullopt;
    fb::Optional<int64_t> blockMaskUid = fb::nullopt;
    fb::Optional<int64_t> sinkUid = fb::nullopt;
    fb::Optional<int64_t> pageTableKUid = fb::nullopt;
    fb::Optional<int64_t> seqLenQUid = fb::nullopt;
    fb::Optional<int64_t> scaleUid = fb::nullopt;
    fb::Optional<int64_t> descaleQUid = fb::nullopt;
    fb::Optional<int64_t> descaleKUid = fb::nullopt;
    fb::Optional<int64_t> descaleVUid = fb::nullopt;
    fb::Optional<int64_t> descaleSUid = fb::nullopt;
    fb::Optional<int64_t> scaleSUid = fb::nullopt;
    fb::Optional<int64_t> scaleOUid = fb::nullopt;
    if(spec.attnMask)
    {
        attnMaskUid = fb::Optional<int64_t>(addPlaceholder("attn_mask", spec.dtype));
    }
    if(spec.blockMask)
    {
        blockMaskUid = fb::Optional<int64_t>(addPlaceholder("block_mask", DataType::INT32));
    }
    if(spec.sink)
    {
        sinkUid = fb::Optional<int64_t>(addPlaceholder("sink", spec.dtype));
    }
    if(spec.paged)
    {
        pageTableKUid = fb::Optional<int64_t>(addPlaceholder("page_table_k", DataType::INT32));
    }
    if(spec.varlen)
    {
        seqLenQUid = fb::Optional<int64_t>(addPlaceholder("seq_len_q", DataType::INT32));
    }
    if(spec.runtimeScale)
    {
        scaleUid = fb::Optional<int64_t>(addPlaceholder("scale", DataType::FLOAT));
    }
    if(spec.fp8Descale)
    {
        descaleQUid = fb::Optional<int64_t>(addPlaceholder("descale_q", DataType::FLOAT));
        descaleKUid = fb::Optional<int64_t>(addPlaceholder("descale_k", DataType::FLOAT));
        descaleVUid = fb::Optional<int64_t>(addPlaceholder("descale_v", DataType::FLOAT));
        descaleSUid = fb::Optional<int64_t>(addPlaceholder("descale_s", DataType::FLOAT));
        scaleSUid = fb::Optional<int64_t>(addPlaceholder("scale_s", DataType::FLOAT));
        scaleOUid = fb::Optional<int64_t>(addPlaceholder("scale_o", DataType::FLOAT));
    }

    // Move the plan-time scale across optional types without an optional->value->
    // optional round trip (bugprone-optional-value-conversion): bind the value to a
    // named float first, then construct the flatbuffers Optional from it.
    fb::Optional<float> attnScale = fb::nullopt;
    if(spec.attnScaleValue.has_value())
    {
        const float scaleValue = spec.attnScaleValue.value();
        attnScale = fb::Optional<float>(scaleValue);
    }

    const auto attn
        = data_objects::CreateSdpaAttributes(builder,
                                             qUid,
                                             kUid,
                                             vUid,
                                             oUid,
                                             attnMaskUid,
                                             scaleUid,
                                             seqLenQUid, // seq_len_q_tensor_uid
                                             fb::nullopt, // seq_len_kv_tensor_uid
                                             fb::nullopt, // seed_tensor_uid
                                             fb::nullopt, // offset_tensor_uid
                                             fb::nullopt, // dropout_mask_tensor_uid
                                             fb::nullopt, // dropout_scale_tensor_uid
                                             pageTableKUid, // page_table_k_tensor_uid
                                             fb::nullopt, // page_table_v_tensor_uid
                                             blockMaskUid, // block_mask_tensor_uid
                                             sinkUid, // sink_token_tensor_uid
                                             descaleQUid, // descale_q_tensor_uid
                                             descaleKUid, // descale_k_tensor_uid
                                             descaleVUid, // descale_v_tensor_uid
                                             descaleSUid, // descale_s_tensor_uid
                                             scaleSUid, // scale_s_tensor_uid
                                             scaleOUid, // scale_o_tensor_uid
                                             fb::nullopt, // stats_tensor_uid
                                             fb::nullopt, // max_tensor_uid
                                             fb::nullopt, // sum_exp_tensor_uid
                                             fb::nullopt, // rng_dump_tensor_uid
                                             fb::nullopt, // amax_s_tensor_uid
                                             fb::nullopt, // amax_o_tensor_uid
                                             spec.genStats ? fb::Optional<bool>(true) : fb::nullopt,
                                             spec.alibi,
                                             spec.padding,
                                             spec.causal,
                                             spec.causalBottomRight,
                                             fb::nullopt, // dropout_probability
                                             attnScale,
                                             fb::nullopt, // left_bound
                                             fb::nullopt, // right_bound
                                             fb::nullopt, // max_seq_len_kv
                                             data_objects::DiagonalAlignment::TOP_LEFT,
                                             DataType::FLOAT,
                                             data_objects::AttentionImplementation::AUTO);

    std::vector<fb::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "sdpa_fwd",
                                                   spec.dtype,
                                                   data_objects::NodeAttributes::SdpaAttributes,
                                                   attn.Union()));

    const auto graphOffset = data_objects::CreateGraphDirect(
        builder, "test", DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, &tensors, &nodes);
    builder.Finish(graphOffset);

    return BuiltGraph{std::make_shared<fb::DetachedBuffer>(builder.Release())};
}

// ---- ProblemShape accessors (fail the test loudly on a missing/mistyped key) --
bool boolFact(const catalog::ProblemShape& p, const std::string& key)
{
    auto it = p.find(key);
    EXPECT_NE(it, p.end()) << "missing bool fact '" << key << "'";
    EXPECT_TRUE(std::holds_alternative<bool>(it->second)) << "fact '" << key << "' is not bool";
    return std::get<bool>(it->second);
}
int64_t intFact(const catalog::ProblemShape& p, const std::string& key)
{
    auto it = p.find(key);
    EXPECT_NE(it, p.end()) << "missing int fact '" << key << "'";
    EXPECT_TRUE(std::holds_alternative<int64_t>(it->second)) << "fact '" << key << "' is not int";
    return std::get<int64_t>(it->second);
}
std::string strFact(const catalog::ProblemShape& p, const std::string& key)
{
    auto it = p.find(key);
    EXPECT_NE(it, p.end()) << "missing string fact '" << key << "'";
    EXPECT_TRUE(std::holds_alternative<std::string>(it->second))
        << "fact '" << key << "' is not string";
    return std::get<std::string>(it->second);
}

int64_t scalarI64(const catalog::LaunchBindings& b, const std::string& key)
{
    auto it = b.scalars.find(key);
    EXPECT_NE(it, b.scalars.end()) << "missing scalar '" << key << "'";
    EXPECT_TRUE(std::holds_alternative<int64_t>(it->second)) << "scalar '" << key << "' is not i64";
    return std::get<int64_t>(it->second);
}
float scalarF32(const catalog::LaunchBindings& b, const std::string& key)
{
    auto it = b.scalars.find(key);
    EXPECT_NE(it, b.scalars.end()) << "missing scalar '" << key << "'";
    EXPECT_TRUE(std::holds_alternative<float>(it->second)) << "scalar '" << key << "' is not f32";
    return std::get<float>(it->second);
}

const ops::SdpaAdapter ADAPTER;

} // namespace

// (a) decode publishes the full capability vocabulary for the baseline problem.
TEST(TestAotCatalogSdpaDecode, PublishesFullVocabularyForGfx1151Shape)
{
    const BuiltGraph g = buildSdpaGraph(SdpaSpec{});
    const auto shape = ADAPTER.decode(g.graph());
    ASSERT_TRUE(shape.has_value()) << "baseline gfx1151-shaped graph should decode";

    EXPECT_EQ(strFact(*shape, "dtype"), "f16");
    EXPECT_EQ(intFact(*shape, "B"), 1);
    EXPECT_EQ(intFact(*shape, "H"), 32);
    EXPECT_EQ(intFact(*shape, "H_kv"), 32);
    EXPECT_EQ(intFact(*shape, "S_q"), 32);
    EXPECT_EQ(intFact(*shape, "S_kv"), 48);
    EXPECT_EQ(intFact(*shape, "D"), 64);
    EXPECT_EQ(intFact(*shape, "gqa_ratio"), 1);

    EXPECT_TRUE(boolFact(*shape, "d_contiguous"));
    EXPECT_TRUE(boolFact(*shape, "batch_foldable"));
    for(const char* key : {"causal",
                           "causal_bottom_right",
                           "has_alibi",
                           "has_padding_mask",
                           "has_attn_mask",
                           "has_block_mask",
                           "has_sink",
                           "has_dropout",
                           "paged",
                           "varlen",
                           "gen_stats",
                           "fp8",
                           "runtime_scale"})
    {
        EXPECT_FALSE(boolFact(*shape, key)) << "expected " << key << " == false on baseline";
    }
}

// (b) legacy gfx1151 bindings + grid symbols are byte-for-byte the shipped ABI.
TEST(TestAotCatalogSdpaDecode, LegacyGfx1151BindingsPreserved)
{
    const BuiltGraph g = buildSdpaGraph(SdpaSpec{});
    const auto shape = ADAPTER.decode(g.graph());
    ASSERT_TRUE(shape.has_value());

    const catalog::KernelEntry kernel; // adapter ignores it ((void)kernel)
    const catalog::LaunchBindings b = ADAPTER.buildBindings(g.graph(), *shape, kernel);

    // Pointers Q,K,V,O bound to their tensor uids.
    ASSERT_EQ(b.pointerUids.at("Q"), 1);
    ASSERT_EQ(b.pointerUids.at("K"), 2);
    ASSERT_EQ(b.pointerUids.at("V"), 3);
    ASSERT_EQ(b.pointerUids.at("O"), 4);

    // scale_log2 = (1/sqrt(64)) * log2(e).
    constexpr float EXPECTED_SCALE_LOG2 = 0.125F * 1.4426950408889634F;
    EXPECT_NEAR(scalarF32(b, "scale_log2"), EXPECTED_SCALE_LOG2, 1e-7F);

    EXPECT_EQ(scalarI64(b, "seqlen_q"), 32);
    EXPECT_EQ(scalarI64(b, "seqlen_k"), 48);

    // Contiguous BHSD: token stride = D, head stride = S*D.
    EXPECT_EQ(scalarI64(b, "stride_q_token"), 64);
    EXPECT_EQ(scalarI64(b, "stride_q_head"), 32 * 64);
    EXPECT_EQ(scalarI64(b, "stride_k_token"), 64);
    EXPECT_EQ(scalarI64(b, "stride_k_head"), 48 * 64);
    EXPECT_EQ(scalarI64(b, "stride_v_token"), 64);
    EXPECT_EQ(scalarI64(b, "stride_v_head"), 48 * 64);
    EXPECT_EQ(scalarI64(b, "stride_o_token"), 64);
    EXPECT_EQ(scalarI64(b, "stride_o_head"), 32 * 64);

    // Grid symbols the family's grid formula (ceil_div(S_q,16), H, B) references.
    const launch::SymbolTable syms = ADAPTER.gridSymbols(*shape, kernel);
    EXPECT_EQ(syms.at("S_q"), 32);
    EXPECT_EQ(syms.at("H"), 32);
    EXPECT_EQ(syms.at("B"), 1);
}

// (c) safety gates: malformed graphs still decline in C++.
TEST(TestAotCatalogSdpaDecode, DeclinesNonRank4)
{
    SdpaSpec spec;
    spec.qRank = 3;
    EXPECT_FALSE(ADAPTER.decode(buildSdpaGraph(spec).graph()).has_value());
}

TEST(TestAotCatalogSdpaDecode, DeclinesDtypeMismatch)
{
    SdpaSpec spec;
    spec.kDtype = DataType::BFLOAT16; // K disagrees with Q/V/O
    EXPECT_FALSE(ADAPTER.decode(buildSdpaGraph(spec).graph()).has_value());
}

TEST(TestAotCatalogSdpaDecode, DeclinesNonIntegerGqaRatio)
{
    SdpaSpec spec;
    spec.h = 32;
    spec.hkv = 7; // 32 % 7 != 0 -> malformed grouping
    EXPECT_FALSE(ADAPTER.decode(buildSdpaGraph(spec).graph()).has_value());
}

TEST(TestAotCatalogSdpaDecode, DeclinesUnsupportedDtype)
{
    SdpaSpec spec;
    spec.dtype = DataType::INT32; // providerDtype -> nullopt
    EXPECT_FALSE(ADAPTER.decode(buildSdpaGraph(spec).graph()).has_value());
}

// (d) the boundary moved to data: feature-rich graphs now DECODE (publish facts)
// where the old adapter hard-declined. No gfx1151 kernel accepts them, so the
// engine still declines in aggregate -- but that decision now lives in family.json.
TEST(TestAotCatalogSdpaDecode, CausalGraphDecodes)
{
    SdpaSpec spec;
    spec.causal = true;
    const auto shape = ADAPTER.decode(buildSdpaGraph(spec).graph());
    ASSERT_TRUE(shape.has_value()) << "causal graph should decode, not decline";
    EXPECT_TRUE(boolFact(*shape, "causal"));
}

TEST(TestAotCatalogSdpaDecode, GqaGraphDecodes)
{
    SdpaSpec spec;
    spec.h = 32;
    spec.hkv = 8; // gqa_ratio = 4
    const auto shape = ADAPTER.decode(buildSdpaGraph(spec).graph());
    ASSERT_TRUE(shape.has_value()) << "GQA graph should decode, not decline";
    EXPECT_EQ(intFact(*shape, "H_kv"), 8);
    EXPECT_EQ(intFact(*shape, "gqa_ratio"), 4);
}

TEST(TestAotCatalogSdpaDecode, AttnMaskGraphDecodes)
{
    SdpaSpec spec;
    spec.attnMask = true;
    const auto shape = ADAPTER.decode(buildSdpaGraph(spec).graph());
    ASSERT_TRUE(shape.has_value()) << "masked graph should decode, not decline";
    EXPECT_TRUE(boolFact(*shape, "has_attn_mask"));
}

TEST(TestAotCatalogSdpaDecode, NonFoldableBatchDecodes)
{
    SdpaSpec spec;
    spec.b = 2; // contiguous BHSD, H>1 -> batch not foldable, but still valid
    const auto shape = ADAPTER.decode(buildSdpaGraph(spec).graph());
    ASSERT_TRUE(shape.has_value()) << "B>1 graph should decode, not decline";
    EXPECT_EQ(intFact(*shape, "B"), 2);
    EXPECT_FALSE(boolFact(*shape, "batch_foldable"));
}

TEST(TestAotCatalogSdpaDecode, NonContiguousDDecodes)
{
    SdpaSpec spec;
    spec.innerMult = 2; // D axis stride 2 -> not contiguous
    const auto shape = ADAPTER.decode(buildSdpaGraph(spec).graph());
    ASSERT_TRUE(shape.has_value()) << "non-contiguous-D graph should decode, not decline";
    EXPECT_FALSE(boolFact(*shape, "d_contiguous"));
}

TEST(TestAotCatalogSdpaDecode, Bf16ShapeDecodesWithBf16Token)
{
    SdpaSpec spec;
    spec.dtype = DataType::BFLOAT16;
    const auto shape = ADAPTER.decode(buildSdpaGraph(spec).graph());
    ASSERT_TRUE(shape.has_value());
    EXPECT_EQ(strFact(*shape, "dtype"), "bf16");
    EXPECT_FALSE(boolFact(*shape, "fp8"));
}

// The baseline (no optional operands) binds ONLY Q/K/V/O -- every optional
// pointer name is absent, so a kernel that named one would fail closed.
TEST(TestAotCatalogSdpaDecode, OptionalPointersAbsentOnBaseline)
{
    const BuiltGraph g = buildSdpaGraph(SdpaSpec{});
    const auto shape = ADAPTER.decode(g.graph());
    ASSERT_TRUE(shape.has_value());
    const catalog::KernelEntry kernel;
    const catalog::LaunchBindings b = ADAPTER.buildBindings(g.graph(), *shape, kernel);

    for(const char* name : {"attn_mask",
                            "block_mask",
                            "sink",
                            "scale_tensor",
                            "seqlen_q_ptr",
                            "seqlen_kv_ptr",
                            "page_table_k",
                            "page_table_v",
                            "descale_q",
                            "descale_k",
                            "descale_v",
                            "descale_s",
                            "scale_s",
                            "scale_o",
                            "stats",
                            "lse",
                            "max",
                            "sum_exp"})
    {
        EXPECT_EQ(b.pointerUids.count(name), 0U) << "unexpected optional pointer '" << name << "'";
    }
}

// A graph carrying mask/sink/paged/varlen/runtime-scale operands binds each of
// those pointers by name (the feature surface decode() flags as facts).
TEST(TestAotCatalogSdpaDecode, OptionalFeaturePointersBoundWhenPresent)
{
    SdpaSpec spec;
    spec.attnMask = true;
    spec.blockMask = true;
    spec.sink = true;
    spec.paged = true;
    spec.varlen = true;
    spec.runtimeScale = true;
    const BuiltGraph g = buildSdpaGraph(spec);
    const auto shape = ADAPTER.decode(g.graph());
    ASSERT_TRUE(shape.has_value());
    const catalog::KernelEntry kernel;
    const catalog::LaunchBindings b = ADAPTER.buildBindings(g.graph(), *shape, kernel);

    // Every present operand is bound to a placeholder uid (>4, i.e. not Q/K/V/O).
    for(const char* name :
        {"attn_mask", "block_mask", "sink", "page_table_k", "seqlen_q_ptr", "scale_tensor"})
    {
        auto it = b.pointerUids.find(name);
        ASSERT_NE(it, b.pointerUids.end()) << "missing optional pointer '" << name << "'";
        EXPECT_GT(it->second, 4) << "pointer '" << name << "' bound to an operand uid";
    }
    // page_table_v / seqlen_kv_ptr were not supplied by this graph -> still absent.
    EXPECT_EQ(b.pointerUids.count("page_table_v"), 0U);
    EXPECT_EQ(b.pointerUids.count("seqlen_kv_ptr"), 0U);
}

// An fp8 graph publishes fp8=true and binds every (de)scale pointer a quantizing
// forward kernel names -- the gap this change closes.
TEST(TestAotCatalogSdpaDecode, Fp8DescalePointersBound)
{
    SdpaSpec spec;
    spec.dtype = DataType::FP8_E4M3;
    spec.fp8Descale = true;
    const BuiltGraph g = buildSdpaGraph(spec);
    const auto shape = ADAPTER.decode(g.graph());
    ASSERT_TRUE(shape.has_value()) << "fp8 graph should decode";
    EXPECT_EQ(strFact(*shape, "dtype"), "f8");
    EXPECT_TRUE(boolFact(*shape, "fp8"));

    const catalog::KernelEntry kernel;
    const catalog::LaunchBindings b = ADAPTER.buildBindings(g.graph(), *shape, kernel);
    for(const char* name :
        {"descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o"})
    {
        auto it = b.pointerUids.find(name);
        ASSERT_NE(it, b.pointerUids.end()) << "missing fp8 pointer '" << name << "'";
        EXPECT_GT(it->second, 4) << "fp8 pointer '" << name << "' bound to an operand uid";
    }

    // Byte strides collapse to element strides at 1 byte/element for fp8.
    EXPECT_EQ(scalarI64(b, "stride_q_token_bytes"), scalarI64(b, "stride_q_token"));
}
